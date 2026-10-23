# [ic7s9 v4: A/B/C fix: newest-policy successes + demo slice per batch, L2-to-init anchor, slow EMA, ||delta||/PPO-projection logging, early stop] [ic6 v3: + --ema] [ic6 v2: no double normalization; --steps 0 = identity pass-through] Short BC/IL retrain of diffusion UNet on demo ∪ successful rollouts (encoder frozen).
# Preloads feats/actions to CPU; normalizes actions; trains epsilon-MSE like Stage-1 BC.
import os,sys,json,glob,argparse,time
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[1]/'train/bc_stage1'))
import numpy as np,torch
import torch.nn.functional as Fnn
from wcommon_fix import *  # noqa: F403
import novib_patch
from evaluate_bc import load_policy

ap=argparse.ArgumentParser()
ap.add_argument('--policy',required=True);ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=20000);ap.add_argument('--batch',type=int,default=64);ap.add_argument('--lr',type=float,default=1e-4)
ap.add_argument('--ema',type=float,default=0.,help='[v3] EMA decay of UNet weights (0=off); saved weights = EMA');ap.add_argument('--eval-every',type=int,default=2000);ap.add_argument('--max-demos',type=int,default=0,help='0=all demos');ap.add_argument('--demo-frac',type=float,default=.125);ap.add_argument('--l2',type=float,default=1.0,help='penalty l2*||theta-theta0||^2');ap.add_argument('--max-delta',type=float,default=.05);ap.add_argument('--ppo-from',default='');ap.add_argument('--ppo-to',default='')
a=ap.parse_args();setup(2026);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text())

Fs=[]; As=[]; held=[]; isd=[]; n_demo=0; n_roll=0
for kind,path in sources:
    if kind=='demo':
        for split in ['train','validation']:
            z=np.load(Path(path)/f'demo_{split}.npz')  # load once (not per-episode)
            zf=z['feats']; za=z['actions']; zs=z['success']
            ids=[e for e in range(len(zs)) if bool(zs[e])]
            if a.max_demos and split=='train': ids=ids[:a.max_demos]
            if a.max_demos and split=='validation': ids=ids[:max(64,a.max_demos//8)]
            for e in ids:
                Fs.append(zf[e]); As.append(za[e]); held.append(split=='validation'); isd.append(True); n_demo+=1
    else:
        for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
            if '.tmp.' in f: continue
            z=np.load(f); meta=json.loads(str(z['meta']))
            if not bool(meta.get('success',False)): continue
            Fs.append(z['feats']); As.append(z['actions']); held.append(meta['seed']%10==0); isd.append(False); n_roll+=1
assert n_demo>0 and Fs, 'no IL data'
F=torch.from_numpy(np.stack(Fs).astype(np.float32))          # (N,61,1564) CPU
A=torch.from_numpy(np.stack(As).astype(np.float32))          # (N,60,8,14) CPU
held=torch.tensor(held,dtype=torch.bool); isd=torch.tensor(isd,dtype=torch.bool)
print('IL_DATA',json.dumps(dict(n=len(F),n_demo=n_demo,n_roll_success=n_roll,F=list(F.shape),A=list(A.shape))),flush=True)

p,cp=load_policy(a.policy);p.train()
for q in p.obs_encoder.parameters(): q.requires_grad_(False)
for q in p.normalizer.parameters(): q.requires_grad_(False)
# normalize actions once (BC trains in normalized space)
# [ic6] stored actions are ALREADY normalized (demo_features_v3 + rollout x); do NOT normalize again
opt=torch.optim.AdamW([q for q in p.model.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-6)
sc=p.noise_scheduler; Ttrain=sc.config.num_train_timesteps
trd=torch.where(~held&isd)[0]; trr=torch.where(~held&~isd)[0]; hod=torch.where(held&isd)[0]; hor=torch.where(held&~isd)[0]
assert len(trd) and len(trr) and len(hod) and len(hor)
nd=max(1,round(a.demo_frac*a.batch)); nr=a.batch-nd
theta0={k:v.detach().clone() for k,v in p.model.named_parameters()}
def flat(sd,keys): return torch.cat([sd[k].float().flatten() for k in keys])
pkeys=[k for k,_ in p.model.named_parameters()]
dppo=None
if a.ppo_from and a.ppo_to:
    pf,_=load_policy(a.ppo_from); pt,_=load_policy(a.ppo_to)
    sf=dict(pf.model.named_parameters()); st=dict(pt.model.named_parameters())
    dppo=(flat(st,pkeys)-flat(sf,pkeys)).detach(); del pf,pt
    print('PPO_DELTA_NORM',float(dppo.norm()),flush=True)
def delta_stats(sd):
    d=flat(sd,pkeys)-flat(theta0,pkeys); n=float(d.norm())
    r=dict(delta_norm=n)
    if dppo is not None:
        r.update(ppo_norm=float(dppo.norm()),proj_on_ppo=float(d@dppo/dppo.norm()),cos_ppo=float(d@dppo/(dppo.norm()*d.norm()+1e-12)),ratio_to_ppo=n/float(dppo.norm()))
    return r
status_path=out.with_suffix('.status.json'); t0=time.time(); best=None; best_loss=1e9
g=torch.Generator(device='cuda').manual_seed(7)

def batch_loss(eidx):
    e=eidx.to('cuda'); B=len(e)
    t=torch.randint(CHUNKS,(B,),device='cuda',generator=g)
    Fc=F[eidx].to('cuda',non_blocking=True)   # (B,61,1564)
    Ac=A[eidx].to('cuda',non_blocking=True)   # (B,60,8,14)
    cond=torch.cat([Fc[torch.arange(B),(t-1).clamp(min=0)],Fc[torch.arange(B),t]],1)
    x0=Ac[torch.arange(B),t]                  # (B,8,14) normalized
    noise=torch.randn(x0.shape,device='cuda',generator=g)
    ts=torch.randint(0,Ttrain,(B,),device='cuda',generator=g).long()
    xt=sc.add_noise(x0,noise,ts)
    pred=p.model(sample=xt,timestep=ts,local_cond=None,global_cond=cond)
    return Fnn.mse_loss(pred,noise)

rng=np.random.default_rng(7)
import copy
ema_sd={k:v.detach().clone() for k,v in p.model.state_dict().items()} if a.ema>0 else None
assert A.abs().max()<5, f'actions not in normalized range: {A.abs().max()}'
for step in range(1,a.steps+1):
    pick=np.concatenate([rng.choice(trd.numpy(),size=nd,replace=False),rng.choice(trr.numpy(),size=min(nr,len(trr)),replace=False)])
    bl=batch_loss(torch.from_numpy(pick).long())
    reg=sum(((q-theta0[k])**2).sum() for k,q in p.model.named_parameters() if q.requires_grad)
    loss=bl+a.l2*reg
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.0); opt.step()
    if ema_sd is not None:
        with torch.no_grad():
            for k,v in p.model.state_dict().items():
                if v.dtype.is_floating_point: ema_sd[k].mul_(a.ema).add_(v.detach(),alpha=1-a.ema)
                else: ema_sd[k].copy_(v)
    if step%a.eval_every==0 or step==1 or step==a.steps or step<=0:
        with torch.no_grad():
            p.model.eval()
            g.manual_seed(11)  # fixed noise for comparable held-out losses
            vr=batch_loss(hor[:256]).item(); vd=batch_loss(hod[:256]).item(); vl=vr; p.model.train()
            esd={k:ema_sd[k] for k in pkeys} if ema_sd is not None else dict(p.model.named_parameters())
            ds=delta_stats(esd); dr=delta_stats(dict(p.model.named_parameters()))
        row=dict(step=step,train_loss=float(bl.item()),l2_reg=float(reg.item()),held_roll_loss=vr,held_demo_loss=vd,saved=ds,raw=dict(delta_norm=dr['delta_norm']),grad_norm=float(gn),elapsed_s=time.time()-t0,n=int(len(F)),n_demo=n_demo,n_roll=n_roll)
        append_jsonl(out.with_suffix('.metrics.jsonl'),row); write_json(status_path,dict(status='training',updated=cst(),**row)); print(json.dumps(row),flush=True)
        if vl<best_loss: best_loss=vl
        if ds['delta_norm']>a.max_delta: print('EARLY_STOP delta',ds['delta_norm'],flush=True); break
if ema_sd is not None: p.model.load_state_dict(ema_sd)  # [v3] save EMA weights (no val-best selection)

p.eval()
save_policy(p,cp,out,dict(stage='bc_il_retrain',source_policy=a.policy,steps=a.steps,lr=a.lr,n_demo=n_demo,n_roll_success=n_roll,held_loss=best_loss,sources=sources,max_demos=a.max_demos,ema=a.ema,demo_frac=a.demo_frac,l2=a.l2,final_step=step))
fin=delta_stats(dict(p.model.named_parameters()))
write_json(status_path,dict(status='done',updated=cst(),out=str(out),held_loss=best_loss,final_step=step,final=fin,n=int(len(F)),n_demo=n_demo,n_roll=n_roll))
print('IL_DONE',out,'held_loss',best_loss,flush=True)
