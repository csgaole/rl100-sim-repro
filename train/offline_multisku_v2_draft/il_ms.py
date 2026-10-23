# [offline_multisku_v1_draft] NEW file, multi-SKU copy of offline_stage2_v3_nominal_20261017/bc_il_retrain_v4.py (STEP 7: IL retrain).
# Rule fixes vs base: sources MUST be exactly [demo] + ONE rollout dir (= this round's collect, produced by the gate-passed policy):
#   no accumulation across rounds (asserted); only successful rollouts are read; encoder + normalizer frozen and ASSERTED unchanged;
#   multi-SKU loader; per-SKU data counts logged; MS_TRAIN_ALL=1 -> held-out subset also trains (losses then in-sample);
#   --batch default 256 (base 64). Recipe unchanged: epsilon-MSE on UNet only, AdamW, L2-to-init, EMA, max-delta early stop.
import argparse,copy
from ms_common import *
import torch.nn.functional as Fnn
ap=argparse.ArgumentParser()
ap.add_argument('--policy',required=True);ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=300);ap.add_argument('--batch',type=int,default=256);ap.add_argument('--lr',type=float,default=1e-6)
ap.add_argument('--ema',type=float,default=.999);ap.add_argument('--eval-every',type=int,default=50);ap.add_argument('--demo-frac',type=float,default=.125)
ap.add_argument('--l2',type=float,default=1.0,help='penalty l2*||theta-theta0||^2');ap.add_argument('--max-delta',type=float,default=.05);ap.add_argument('--ppo-from',default='');ap.add_argument('--ppo-to',default='')
a=ap.parse_args();setup(2026);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text())
assert [k for k,_ in sources].count('demo')==1 and len(sources)==2,f'IL sources must be [demo, ONE rollout dir] (no accumulation): {sources}'
Fs=[];As=[];held=[];isd=[];skn=[];n_demo=0;n_roll=0;n_roll_fail_skipped=0
for kind,path in sources:
    if kind=='demo':
        for split in ['train','validation']:
            z=np.load(Path(path)/f'demo_{split}.npz');zf=z['feats'];za=z['actions'];zs=z['success'];zn=z['sku']
            for e in range(len(zs)):
                if not bool(zs[e]):continue
                Fs.append(zf[e]);As.append(za[e]);held.append(split=='validation');isd.append(True);skn.append(str(zn[e]));n_demo+=1
    else:
        for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
            if '.tmp.' in f:continue
            z=np.load(f);meta=json.loads(str(z['meta']))
            if not bool(meta.get('success',False)):n_roll_fail_skipped+=1;continue
            Fs.append(z['feats']);As.append(z['actions']);held.append(bool(meta.get('heldout',meta['seed']%10==0)));isd.append(False);skn.append(meta['sku']);n_roll+=1
assert n_demo>0 and n_roll>0,'no IL data'
F=torch.from_numpy(np.stack(Fs).astype(np.float32));A=torch.from_numpy(np.stack(As).astype(np.float32))
assert F.shape[-1]==FEAT,F.shape
held=torch.tensor(held,dtype=torch.bool);isd=torch.tensor(isd,dtype=torch.bool);skn=np.array(skn)
per_sku={s:dict(demo=int(((skn==s)&isd.numpy()).sum()),rollout_success=int(((skn==s)&~isd.numpy()).sum())) for s in SKUS}
info=dict(n=len(F),n_demo=n_demo,n_roll_success=n_roll,n_roll_fail_skipped=n_roll_fail_skipped,per_sku=per_sku,train_all=TRAIN_ALL,F=list(F.shape),A=list(A.shape),sources=sources)
print('IL_DATA',json.dumps(info),flush=True)
p,cp=load_policy(a.policy);p.train()
for q in p.obs_encoder.parameters():q.requires_grad_(False)
for q in p.normalizer.parameters():q.requires_grad_(False)
enc_before={k:v.detach().clone() for k,v in p.obs_encoder.state_dict().items()}
opt=torch.optim.AdamW([q for q in p.model.parameters() if q.requires_grad],lr=a.lr,weight_decay=1e-6)
sc=p.noise_scheduler;Ttrain=sc.config.num_train_timesteps
trmask=torch.ones_like(held) if TRAIN_ALL else ~held
trd=torch.where(trmask&isd)[0];trr=torch.where(trmask&~isd)[0];hod=torch.where(held&isd)[0];hor=torch.where(held&~isd)[0]
assert len(trd) and len(trr) and len(hod) and len(hor)
nd=max(1,round(a.demo_frac*a.batch));nr=a.batch-nd
theta0={k:v.detach().clone() for k,v in p.model.named_parameters()}
def flat(sd,keys):return torch.cat([sd[k].float().flatten() for k in keys])
pkeys=[k for k,_ in p.model.named_parameters()]
dppo=None
if a.ppo_from and a.ppo_to:
    pf,_=load_policy(a.ppo_from);pt,_=load_policy(a.ppo_to)
    sf=dict(pf.model.named_parameters());st=dict(pt.model.named_parameters())
    dppo=(flat(st,pkeys)-flat(sf,pkeys)).detach();del pf,pt
    print('PPO_DELTA_NORM',float(dppo.norm()),flush=True)
def delta_stats(sd):
    d=flat(sd,pkeys)-flat(theta0,pkeys);n=float(d.norm());r=dict(delta_norm=n)
    if dppo is not None:r.update(ppo_norm=float(dppo.norm()),proj_on_ppo=float(d@dppo/dppo.norm()),cos_ppo=float(d@dppo/(dppo.norm()*d.norm()+1e-12)),ratio_to_ppo=n/float(dppo.norm()))
    return r
status_path=out.with_suffix('.status.json');t0=time.time();best_loss=1e9
g=torch.Generator(device='cuda').manual_seed(7)
def batch_loss(eidx):
    B=len(eidx);t=torch.randint(CHUNKS,(B,),device='cuda',generator=g)
    Fc=F[eidx].to('cuda',non_blocking=True);Ac=A[eidx].to('cuda',non_blocking=True)
    cond=torch.cat([Fc[torch.arange(B),(t-1).clamp(min=0)],Fc[torch.arange(B),t]],1)
    x0=Ac[torch.arange(B),t];noise=torch.randn(x0.shape,device='cuda',generator=g)
    ts=torch.randint(0,Ttrain,(B,),device='cuda',generator=g).long();xt=sc.add_noise(x0,noise,ts)
    pred=p.model(sample=xt,timestep=ts,local_cond=None,global_cond=cond)
    return Fnn.mse_loss(pred,noise)
rng=np.random.default_rng(7)
ema_sd={k:v.detach().clone() for k,v in p.model.state_dict().items()} if a.ema>0 else None
assert A.abs().max()<5,f'actions not in normalized range: {A.abs().max()}'
step=0
for step in range(1,a.steps+1):
    pick=np.concatenate([rng.choice(trd.numpy(),size=min(nd,len(trd)),replace=False),rng.choice(trr.numpy(),size=min(nr,len(trr)),replace=False)])
    bl=batch_loss(torch.from_numpy(pick).long())
    reg=sum(((q-theta0[k])**2).sum() for k,q in p.model.named_parameters() if q.requires_grad)
    loss=bl+a.l2*reg;assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    gn=torch.nn.utils.clip_grad_norm_(p.model.parameters(),1.0);opt.step()
    if ema_sd is not None:
        with torch.no_grad():
            for k,v in p.model.state_dict().items():
                if v.dtype.is_floating_point:ema_sd[k].mul_(a.ema).add_(v.detach(),alpha=1-a.ema)
                else:ema_sd[k].copy_(v)
    if step%a.eval_every==0 or step==1 or step==a.steps:
        with torch.no_grad():
            p.model.eval();g.manual_seed(11)
            vr=batch_loss(hor[:256]).item();vd=batch_loss(hod[:256]).item();vl=vr;p.model.train()
            esd={k:ema_sd[k] for k in pkeys} if ema_sd is not None else dict(p.model.named_parameters())
            ds=delta_stats(esd);dr=delta_stats(dict(p.model.named_parameters()))
        row=dict(step=step,train_loss=float(bl.item()),l2_reg=float(reg.item()),held_roll_loss=vr,held_demo_loss=vd,held_in_sample=TRAIN_ALL,saved=ds,raw=dict(delta_norm=dr['delta_norm']),grad_norm=float(gn),elapsed_s=time.time()-t0,**{k:info[k] for k in ('n','n_demo','n_roll_success')})
        append_jsonl(out.with_suffix('.metrics.jsonl'),row);write_json(status_path,dict(status='training',updated=cst(),**row));print(json.dumps(row),flush=True)
        best_loss=min(best_loss,vl)
        if ds['delta_norm']>a.max_delta:print('EARLY_STOP delta',ds['delta_norm'],flush=True);break
if ema_sd is not None:p.model.load_state_dict(ema_sd)  # save EMA weights (no val-best selection)
assert all(torch.equal(v,enc_before[k]) for k,v in p.obs_encoder.state_dict().items()),'encoder changed during IL'
p.eval()
save_policy(p,cp,out,dict(stage='offline_multisku_il_draft',source_policy=a.policy,steps=a.steps,lr=a.lr,batch=a.batch,n_demo=n_demo,n_roll_success=n_roll,held_loss=best_loss,sources=sources,ema=a.ema,demo_frac=a.demo_frac,l2=a.l2,final_step=step,encoder_frozen=True))
fin=delta_stats(dict(p.model.named_parameters()))
write_json(status_path,dict(status='done',updated=cst(),out=str(out),held_loss=best_loss,final_step=step,final=fin,data=info))
print('IL_DONE',out,'held_loss',best_loss,flush=True)
