# Sample K chunks from e88 at fixed dummy cond? Better: replay first-frame feats from a prior collect if present,
# else synthesize random cond and report relative norms. Prefer real feats from smoke collect.
import os,sys,json,glob
os.environ.setdefault('MUJOCO_GL','egl')
sys.path[:0]=['.','../bc_stage1']
import numpy as np,torch
import novib_patch
from evaluate_bc import load_policy
import s2common_v3 as S0
import s2_stdfix as S1
from pathlib import Path

pol=os.environ['V3_START_POLICY'];p,cp=load_policy(pol);p.eval()
# Use demo-like random cond in encoder feature space: load a few eps from prior smoke if any
cands=sorted(glob.glob('/data02/kemove/rl100-sim-repro/runs/offline_stage2_v3e88_smoke_20261014_1122/round1/collect/eps/ep_*.npz'))[:8]
assert cands, 'need prior eps'
feats=[]
for f in cands:
    z=np.load(f);feats.append(z['feats'][0])  # first chunk state, shape (1564,)
F=torch.tensor(np.stack(feats),device='cuda',dtype=torch.float32)  # N,1564
# policy_cond needs 2-frame; approximate with duplicate
cond=torch.cat([F,F],-1)  # N,3128 — wait FEAT=1564 is already per-frame; check policy_cond
# Actually in s2common, policy_cond uses D['F'] with shape (n,61,1564) and takes frames t,t+1.
# Here F is 1564. Encoder output for batch_worker cond is f.reshape(N,-1) with f shape (N,2,FEAT/2?) 
# batch_worker: f=encode -> (N,2,d); cond=f.reshape(N,-1) = (N, 2*d). d=1564/ something.
# From FEAT=1564 = 2-frame? Looking at critic_state / encode: feats append f.reshape(N,2,-1)[:,1] so each frame is 782?
# encode returns p.obs_encoder(...) which for 2 hist frames. f.reshape(N,2,-1) => FEAT=1564 means per-frame 782, cond=2*782=1564.
# batch_worker cond = f.reshape(N,-1) with f from encode(hist) of shape matching 2 frames -> (N,1564).
cond=F  # (N,1564) from stored per-frame feat at chunk0 — WRONG for cond which needs 2-frame.
# Rebuild 2-frame cond from ep feats[0] and feats[0] duplicate stored is single frame.
# Stored feats are per-chunk single-frame ([:,1] of 2-frame encode). For sampling we need policy_cond = concat(frame_t, frame_t+1).
# Use feats[0] and feats[1] from each ep.
conds=[]
for f in cands:
    z=np.load(f);ff=z['feats'];conds.append(np.concatenate([ff[0],ff[1]],-1))
cond=torch.tensor(np.stack(conds),device='cuda',dtype=torch.float32)
print('cond',cond.shape)
K=int(os.environ.get('K','32'))
scale=p.normalizer['action'].params_dict['scale'].cuda().reshape(1,1,14)
offset=p.normalizer['action'].params_dict['offset'].cuda().reshape(1,1,14)
def unnorm(x):return x/scale+offset  # DiffusionPolicy LinearNormalizer

def run(tag,mod,cap,arm=False,clipmean=False,fixfinal=True):
    os.environ['S2_STD_MAX']=str(cap)
    mod.STD_MAX=float(cap);S0.STD_MAX=float(cap)
    if mod is S1:
        mod.FIX_FINAL=fixfinal;mod.CLIP_IN_MEAN=clipmean;mod.ARM_ONLY=arm
    g=torch.Generator(device='cuda').manual_seed(0)
    # det
    with torch.no_grad():
        det=mod.sample_chain(p.model,p,cond,gen=torch.Generator(device='cuda').manual_seed(0),stochastic=False)
        xs=[]
        for k in range(K):
            gk=torch.Generator(device='cuda').manual_seed(1000+k)
            xs.append(mod.sample_chain(p.model,p,cond,gen=gk,stochastic=True))
        sto=torch.stack(xs)  # K,N,8,14
    d=sto-det.unsqueeze(0)
    dn=d.abs();dr=unnorm(sto)-unnorm(det).unsqueeze(0)
    # per-step norms: re-run one sample with trace
    _,xt,yt,lps=mod.sample_chain(p.model,p,cond[:1],gen=torch.Generator(device='cuda').manual_seed(7),stochastic=True,trace=True)
    step_norms=[float(x.norm().item()) for x in xt]
    arm_idx=list(range(6))+list(range(7,13)); grip=[6,13]
    rec=dict(tag=tag,cap=cap,arm_only=arm,clip_in_mean=clipmean,fix_final=fixfinal,
      mae_norm=float(dn.mean()),mae_norm_arm=float(dn[...,arm_idx].mean()),mae_norm_grip=float(dn[...,grip].mean()),
      mae_real=float(dr.abs().mean()),mae_real_arm=float(dr[...,arm_idx].abs().mean()),mae_real_grip=float(dr[...,grip].abs().mean()),
      rmse_norm=float(dn.pow(2).mean().sqrt()),p95_norm=float(dn.reshape(-1).quantile(0.95)),
      per_dim_mae_norm=dn.mean((0,1,2)).tolist(),per_dim_mae_real=dr.abs().mean((0,1,2)).tolist(),
      step_xt_norms=step_norms,det_abs_mean=float(det.abs().mean()),sto_abs_mean=float(sto.abs().mean()))
    print(json.dumps({k:rec[k] for k in rec if k not in ('per_dim_mae_norm','per_dim_mae_real','step_xt_norms')}))
    return rec

out=[]
for cap in (0.01,0.03,0.1):
    out.append(run(f'baseline_cap{cap}',S0,cap,fixfinal=False))
out.append(run('fixfinal_only',S1,0.03,arm=False,clipmean=False,fixfinal=True))
out.append(run('stdfix_arm',S1,0.03,arm=True,clipmean=False,fixfinal=True))
out.append(run('stdfix_arm_clipmean',S1,0.03,arm=True,clipmean=True,fixfinal=True))
out.append(run('stdfix_arm_0.1',S1,0.1,arm=True,clipmean=False,fixfinal=True))
out.append(run('stdfix_arm_clipmean_0.1',S1,0.1,arm=True,clipmean=True,fixfinal=True))
out.append(run('clipmean_only_0.1',S1,0.1,arm=False,clipmean=True,fixfinal=True))
Path(os.environ['OUT']).mkdir(parents=True,exist_ok=True)
json.dump(out,open(os.environ['OUT']+'/action_dev.json','w'),indent=1)
print('WROTE',os.environ['OUT']+'/action_dev.json')
