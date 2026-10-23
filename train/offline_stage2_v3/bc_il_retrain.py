# [ic5] Short BC/IL retrain of diffusion UNet on demo ∪ successful rollouts (encoder frozen).
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
ap.add_argument('--eval-every',type=int,default=2000);ap.add_argument('--max-demos',type=int,default=0,help='0=all demos')
a=ap.parse_args();setup(2026);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text())

Fs=[]; As=[]; held=[]; n_demo=0; n_roll=0
for kind,path in sources:
    if kind=='demo':
        for split in ['train','validation']:
            z=np.load(Path(path)/f'demo_{split}.npz')  # load once (not per-episode)
            zf=z['feats']; za=z['actions']; zs=z['success']
            ids=[e for e in range(len(zs)) if bool(zs[e])]
            if a.max_demos and split=='train': ids=ids[:a.max_demos]
            if a.max_demos and split=='validation': ids=ids[:max(64,a.max_demos//8)]
            for e in ids:
                Fs.append(zf[e]); As.append(za[e]); held.append(split=='validation'); n_demo+=1
    else:
        for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
            if '.tmp.' in f: continue
            z=np.load(f); meta=json.loads(str(z['meta']))
            if not bool(meta.get('success',False)): continue
            Fs.append(z['feats']); As.append(z['actions']); held.append(meta['seed']%10==0); n_roll+=1
assert n_demo>0 and Fs, 'no IL data'
F=torch.from_numpy(np.stack(Fs).astype(np.float32))          # (N,61,1564) CPU
A=torch.from_numpy(np.stack(As).astype(np.float32))          # (N,60,8,14) CPU
held=torch.tensor(held,dtype=torch.bool)
print('IL_DATA',json.dumps(dict(n=len(F),n_demo=n_demo,n_roll_success=n_roll,F=list(F.shape),A=list(A.shape))),flush=True)

p,cp=load_policy(a.policy);p.train()
for q in p.obs_encoder.parameters(): q.requires_grad_(False)
for q in p.normalizer.parameters(): q.requires_grad_(False)
# normalize actions once (BC trains in normalized space)
with torch.no_grad():
    flat=A.reshape(-1,8,14)
    An=p.normalizer['action'].normalize(flat.to('cuda')).cpu().reshape_as(A)
A=An; del An,flat
opt=torch.optim.AdamW([q for q in p.model.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-6)
sc=p.noise_scheduler; Ttrain=sc.config.num_train_timesteps
tr=torch.where(~held)[0]; ho=torch.where(held)[0]
if len(ho)==0: ho=tr[:min(256,len(tr))]
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
for step in range(1,a.steps+1):
    pick=rng.choice(tr.numpy(),size=min(a.batch,len(tr)),replace=False)
    loss=batch_loss(torch.from_numpy(pick).long())
    assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.0); opt.step()
    if step%a.eval_every==0 or step==1 or step==a.steps:
        with torch.no_grad():
            p.model.eval()
            hp=rng.choice(ho.numpy(),size=min(128,len(ho)),replace=False)
            vl=batch_loss(torch.from_numpy(hp).long()).item(); p.model.train()
        row=dict(step=step,train_loss=float(loss.item()),held_loss=vl,grad_norm=float(gn),elapsed_s=time.time()-t0,n=int(len(F)),n_demo=n_demo,n_roll=n_roll)
        append_jsonl(out.with_suffix('.metrics.jsonl'),row); write_json(status_path,dict(status='training',updated=cst(),**row)); print(json.dumps(row),flush=True)
        if vl<best_loss: best_loss=vl; best={k:v.detach().cpu().clone() for k,v in p.model.state_dict().items()}
if best is not None: p.model.load_state_dict(best)
p.eval()
save_policy(p,cp,out,dict(stage='bc_il_retrain',source_policy=a.policy,steps=a.steps,lr=a.lr,n_demo=n_demo,n_roll_success=n_roll,held_loss=best_loss,sources=sources,max_demos=a.max_demos))
write_json(status_path,dict(status='done',updated=cst(),out=str(out),held_loss=best_loss,n=int(len(F)),n_demo=n_demo,n_roll=n_roll))
print('IL_DONE',out,'held_loss',best_loss,flush=True)
