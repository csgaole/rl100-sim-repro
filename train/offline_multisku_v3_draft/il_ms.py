# [offline_multisku_v3_draft official-align 2026-10-20] IL: train Actor+Encoder (official BC / R5); Normalizer frozen.
# Data: [demo]+ONE latest-round successes only. Live-encode demo frames from MS_DEMO_CACHE RGB so encoder gets BC grads.
# Rollout rows still use stored feats (no RGB in ep npz) — documented.
import argparse,os,glob,json
from pathlib import Path
from ms_common import *
import torch.nn.functional as Fnn
ap=argparse.ArgumentParser()
ap.add_argument('--policy',required=True);ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--steps',type=int,default=1000);ap.add_argument('--batch',type=int,default=256)
ap.add_argument('--lr',type=float,default=2e-5);ap.add_argument('--encoder-lr',type=float,default=5e-6)
ap.add_argument('--ema',type=float,default=.999);ap.add_argument('--eval-every',type=int,default=50);ap.add_argument('--demo-frac',type=float,default=.125)
ap.add_argument('--l2',type=float,default=.1);ap.add_argument('--max-delta',type=float,default=.2)
ap.add_argument('--grad-clip',type=float,default=1.);ap.add_argument('--cache',default=os.environ.get('MS_DEMO_CACHE','/data02/kemove/sku5-flip-demonstrations/cache/bc_v1'))
a=ap.parse_args();setup(2026);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
sources=json.loads(Path(a.sources).read_text())
assert [k for k,_ in sources].count('demo')==1 and len(sources)==2,sources
Fs=[];As=[];held=[];isd=[];skn=[];demo_ids=[];n_demo=0;n_roll=0;n_skip=0
demo_dir=None
for kind,path in sources:
    if kind=='demo':
        demo_dir=Path(path)
        for split in ['train','validation']:
            z=np.load(demo_dir/f'demo_{split}.npz')
            for e in range(len(z['success'])):
                if not bool(z['success'][e]):continue
                Fs.append(z['feats'][e]);As.append(z['actions'][e]);held.append(split=='validation');isd.append(True)
                skn.append(str(z['sku'][e]));demo_ids.append(str(z['ids'][e]));n_demo+=1
    else:
        for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
            if '.tmp.' in f:continue
            z=np.load(f);meta=json.loads(str(z['meta']))
            if not bool(meta.get('success',False)):n_skip+=1;continue
            Fs.append(z['feats']);As.append(z['actions']);held.append(bool(meta.get('heldout',meta['seed']%10==0)));isd.append(False);skn.append(meta['sku']);n_roll+=1
assert n_demo and n_roll
F=torch.from_numpy(np.stack(Fs).astype(np.float32)).cuda()
A=torch.from_numpy(np.stack(As).astype(np.float32)).cuda()
held=torch.tensor(held);isd=torch.tensor(isd);skn=np.array(skn)
info=dict(n=len(F),n_demo=n_demo,n_roll_success=n_roll,n_roll_fail_skipped=n_skip,sources=sources)
print('IL_DATA',json.dumps(info),flush=True)
p,cp=load_policy(a.policy);p.train()
for q in p.normalizer.parameters():q.requires_grad_(False)
for q in p.obs_encoder.parameters():q.requires_grad_(True)
norm_before={k:v.detach().clone() for k,v in p.normalizer.state_dict().items()}
enc0={k:v.detach().clone() for k,v in p.obs_encoder.state_dict().items()}
act0={k:v.detach().clone() for k,v in p.model.state_dict().items()}
opt=torch.optim.AdamW([
    {'params':[q for q in p.model.parameters() if q.requires_grad],'lr':a.lr},
    {'params':[q for q in p.obs_encoder.parameters() if q.requires_grad],'lr':a.encoder_lr},
], weight_decay=1e-6)
sc=p.noise_scheduler;Ttrain=sc.config.num_train_timesteps
trd=torch.where((torch.ones_like(held) if TRAIN_ALL else ~held)&isd)[0]
trr=torch.where((torch.ones_like(held) if TRAIN_ALL else ~held)&~isd)[0]
hod=torch.where(held&isd)[0];hor=torch.where(held&~isd)[0]
nd=max(1,round(a.demo_frac*a.batch));nr=a.batch-nd
# demo RGB cache for live encode
CACHE=Path(a.cache);HAVE=CACHE.exists() and (CACHE/'train_rgb.npy').exists()
id_to_loc={};rgb={};st={};skc={}
if HAVE:
    for split,key in [('train','train'),('validation','val')]:
        info_c=json.loads((CACHE/f'{split}_complete.json').read_text())
        rgb[key]=np.load(CACHE/f'{split}_rgb.npy',mmap_mode='r')
        st[key]=np.load(CACHE/f'{split}_state.npy')
        skc[key]=np.load(CACHE/f'{split}_sku.npy')
        for j,i in enumerate(info_c['ids']):id_to_loc[str(i)]=(key,j)
    print('IL_LIVE_ENCODE',CACHE,flush=True)
else:
    print('IL_WARN no RGB cache; encoder gets L2-to-init only',flush=True)

def cond_for(eidx,t):
    """eidx,t long cuda tensors -> (B, 2*FEAT) cond; demo rows live-encoded when HAVE."""
    B=len(eidx);out_c=torch.empty(B,2*FEAT,device='cuda')
    e_cpu=eidx.detach().cpu().tolist();t_cpu=t.detach().cpu().tolist()
    live_idx=[];live_frames=[];live_sk=[]
    for j,(eid,tt) in enumerate(zip(e_cpu,t_cpu)):
        tp=max(tt-1,0)
        if eid<n_demo and HAVE and demo_ids[eid] in id_to_loc:
            key,loc=id_to_loc[demo_ids[eid]]
            frames={c:np.asarray(rgb[key][loc,[tp,tt],ki]) for ki,c in enumerate(CAMS)}
            frames['proprio']=st[key][loc,[tp,tt]]
            live_idx.append(j);live_frames.append(frames);live_sk.append(skc[key][loc])
        else:
            out_c[j]=torch.cat([F[eid,tp],F[eid,tt]],0).detach()
    if live_idx:
        # batch live frames: encode each (small nd) — loop OK for smoke; vectorize lightly
        for j,frames,sk in zip(live_idx,live_frames,live_sk):
            z=encode_frames_grad(p,frames,sk)  # (2,FEAT) with grad
            out_c[j]=z.reshape(-1)
    return out_c

def batch_loss(eidx):
    eidx=eidx.cuda().long();B=len(eidx);t=torch.randint(0,CHUNKS,(B,),device='cuda')
    cond=cond_for(eidx,t)
    x0=A[eidx,t];noise=torch.randn_like(x0);ts=torch.randint(0,Ttrain,(B,),device='cuda').long()
    xt=sc.add_noise(x0,noise,ts);pred=p.model(sample=xt,timestep=ts,local_cond=None,global_cond=cond)
    return Fnn.mse_loss(pred,noise)

ema_m={k:v.detach().clone() for k,v in p.model.state_dict().items()} if a.ema>0 else None
ema_e={k:v.detach().clone() for k,v in p.obs_encoder.state_dict().items()} if a.ema>0 else None
status_path=out.with_suffix('.status.json');t0=time.time();best=1e9;ROWS=[];last_eg=0.
for step in range(1,a.steps+1):
    pick=torch.cat([trd[torch.randint(len(trd),(nd,))],trr[torch.randint(len(trr),(nr,))]])
    pick=pick[torch.randperm(len(pick))]
    bl=batch_loss(pick)
    reg=sum(((v-act0[k])**2).sum() for k,v in p.model.named_parameters() if v.dtype.is_floating_point)
    reg=reg+sum(((v-enc0[k])**2).sum() for k,v in p.obs_encoder.state_dict().items() if v.dtype.is_floating_point)
    loss=bl+a.l2*reg;assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    egs=[q.grad.norm() for q in p.obs_encoder.parameters() if q.grad is not None]
    last_eg=float(torch.stack(egs).norm()) if egs else 0.
    gn=torch.nn.utils.clip_grad_norm_(list(p.model.parameters())+list(p.obs_encoder.parameters()),a.grad_clip);opt.step()
    if ema_m is not None:
        with torch.no_grad():
            for k,v in p.model.state_dict().items():
                if v.dtype.is_floating_point:ema_m[k].mul_(a.ema).add_(v.detach(),alpha=1-a.ema)
            for k,v in p.obs_encoder.state_dict().items():
                if v.dtype.is_floating_point:ema_e[k].mul_(a.ema).add_(v.detach(),alpha=1-a.ema)
    if step%a.eval_every==0 or step==a.steps:
        p.eval()
        with torch.no_grad():
            vr=batch_loss(hor[:min(256,len(hor))]).item() if len(hor) else float('nan')
            vd=batch_loss(hod[:min(256,len(hod))]).item() if len(hod) else float('nan')
            denc=float(torch.stack([(p.obs_encoder.state_dict()[k]-enc0[k]).float().norm() for k in enc0 if enc0[k].dtype.is_floating_point]).norm())
            dact=float(torch.stack([(p.model.state_dict()[k]-act0[k]).float().norm() for k in act0 if act0[k].dtype.is_floating_point]).norm())
        p.train()
        row=dict(step=step,train_loss=float(bl.item()),l2_reg=float(reg.item()),held_roll_loss=vr,held_demo_loss=vd,
                 enc_grad_norm=last_eg,enc_delta_norm=denc,actor_delta_norm=dact,grad_norm=float(gn),elapsed_s=time.time()-t0)
        append_jsonl(out.with_suffix('.metrics.jsonl'),row);ROWS.append(row);print(json.dumps(row),flush=True)
        best=min(best,vr if vr==vr else best)
        if dact>a.max_delta:print('EARLY_STOP',dact,flush=True);break
if ema_m is not None:
    p.model.load_state_dict(ema_m);p.obs_encoder.load_state_dict(ema_e)
assert all(torch.equal(v,norm_before[k]) for k,v in p.normalizer.state_dict().items()),'normalizer changed'
enc_changed=any(((p.obs_encoder.state_dict()[k]-enc0[k]).abs().sum()>0).item() for k in enc0 if enc0[k].dtype.is_floating_point and enc0[k].numel()>0)
save_policy(p,cp,out,dict(stage='offline_multisku_il_v3_official',source_policy=a.policy,lr=a.lr,encoder_lr=a.encoder_lr,
    steps=a.steps,encoder_trained=True,encoder_changed=enc_changed,normalizer_frozen=True,live_demo_encode=HAVE,last_enc_grad=last_eg))
write_json(status_path,dict(status='done',updated=cst(),out=str(out),held_loss=best,final_step=step,final=ROWS[-1] if ROWS else {},
    data=info,encoder_changed=enc_changed,last_enc_grad_norm=last_eg,live_demo_encode=HAVE))
print('IL_DONE',out,'enc_grad',last_eg,'encoder_changed',enc_changed,'live',HAVE,flush=True)
