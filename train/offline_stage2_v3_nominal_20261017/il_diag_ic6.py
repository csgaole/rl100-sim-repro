# [ic6 diag] e88 epsilon-loss on demo-train / demo-val / real rollouts; cached vs fresh re-encode of demo-train episodes.
import sys,os,glob,json
from pathlib import Path
sys.path.insert(0,'.');sys.path.insert(0,'../bc_stage1')
import numpy as np,torch
from wcommon_fix import *;import novib_patch
from evaluate_bc import load_policy
p,cp=load_policy(os.environ['V3_START_POLICY']);p.eval();sc=p.noise_scheduler
def L(F,A,n=3000,seed=0):
    g=torch.Generator(device='cuda').manual_seed(seed);F=torch.as_tensor(np.asarray(F),device='cuda');A=torch.as_tensor(np.asarray(A),device='cuda');E=len(F)
    e=torch.randint(E,(n,),device='cuda',generator=g);t=torch.randint(60,(n,),device='cuda',generator=g)
    cond=torch.cat([F[e,(t-1).clamp(min=0)],F[e,t]],1);x0=A[e,t];res={}
    for tf in [5,50,95]:
        noise=torch.randn(x0.shape,device='cuda',generator=g);ts=torch.full((n,),tf,device='cuda').long()
        with torch.no_grad():pr=p.model(sample=sc.add_noise(x0,noise,ts),timestep=ts,global_cond=cond)
        res[f't{tf}']=round(((pr-noise)**2).mean().item(),5)
    return res
D=Path(os.environ['DEMO_DATA']);R={}
for s in ['train','validation']:
    z=np.load(D/f'demo_{s}.npz');ids=list(z['ids'])
    for src in ['expanded_v1','wide_v1']:
        ix=[i for i,x in enumerate(ids) if str(x).startswith(src)][:200]
        R[f'{s}_{src}']=dict(n=len(ix),loss=L(z['feats'][ix],z['actions'][ix]),act_absmax=float(np.abs(z['actions'][ix]).max()))
fs=sorted(glob.glob(os.environ['ROLL_GLOB'],recursive=True))
R['rollouts']=dict(n=len(fs),loss=L(np.stack([np.load(f)['feats'] for f in fs]),np.stack([np.load(f)['actions'] for f in fs])))
# fresh re-encode of first 3 cached demo-train episodes per source (read-only caches)
SKU=Path("/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/data")
z=np.load(D/'demo_train.npz');ids=list(z['ids'])
for name,C in [("expanded_v1",SKU/"expanded_v1/cache"),("wide_v1",SKU/"p3_cache_wide_v1")]:
    info=json.loads((C/"train_complete.json").read_text());rgb=np.load(C/"train_rgb.npy",mmap_mode="r");state=np.load(C/"train_state.npy",mmap_mode="r")
    out=[]
    for e in range(3):
        j=ids.index(f"{name}:{info['ids'][e]}")
        Fn=encode_frames(p,{**{c:np.asarray(rgb[e,:,k]) for k,c in enumerate(CAMS)},"proprio":np.asarray(state[e])}).cpu().numpy()
        Fc=z['feats'][j];out.append(dict(maxabs=float(np.abs(Fn-Fc).max()),rel=float(np.abs(Fn-Fc).mean()/(np.abs(Fc).mean()+1e-9))))
    R[f'reencode_{name}']=out
print('DIAG',json.dumps(R),flush=True)
