import os,sys,json,time,numpy as np
sys.argv=[sys.argv[0]];sys.path.insert(0,'/data02/kemove/rl100-sim-repro/train/offline_stage2_wide')
from wcommon import *
from pathlib import Path
d=Path('/data02/kemove/tmp/enc_smoke/roll');pol='/data02/kemove/rl100-sim-repro/runs/online_rl_pilot/heldout/it010/it010.pt'
p,cp=load_policy(pol);import copy;start=copy.deepcopy(p.model).eval();model=configure_online_policy(p,False);opt,enc_opt=make_policy_optimizers(p,1e-7,False,1.0)
z=np.load(sorted((d/'data').glob('*.npz'))[0]);L=z['ep_len'];print('ep_len',L,'files',z.files)
fsz=sum(f.stat().st_size for f in (d/'data').glob('*.npz'));n=int(L.sum());print(f'npz {fsz/1e6:.1f} MB for {n} samples -> {fsz/n/1e3:.1f} KB/sample')
R=pack_rollout_obs([{k:z[k] for k in z.files if k.startswith('rgb_') or k=='prop'}],L)
C=torch.tensor(z['cond'],device='cuda').float();X=torch.tensor(z['xs'],device='cuda');Y=torch.tensor(z['ys'],device='cuda');LP=torch.tensor(z['lp'],device='cuda')
j=torch.arange(min(n,64),device='cuda')
c=live_cond(p,R,j,32);print('live vs stored cond max|diff|',(c-C[j]).abs().max().item(),'stored absmax',C[j].abs().max().item())
p.noise_scheduler.set_timesteps(10);TS=[t.to('cuda') for t in p.noise_scheduler.timesteps]
torch.cuda.reset_peak_memory_stats();t0=time.time();zero_policy_opts(opt,enc_opt);adv=torch.randn(len(j),device='cuda');lr0=[]
for s in range(10):
    mean,std=kernel(p,model(X[j,s],TS[s],global_cond=c),TS[s],X[j,s]);logr=logprob(mean,std,Y[j,s])-LP[j,s];lr0.append(logr.abs().max().item())
    ratio=logr.clamp(-20,20).exp();l=-torch.minimum(ratio*adv,ratio.clamp(.85,1.15)*adv).mean()/10;assert torch.isfinite(l);l.backward()
egn=encoder_backward(p,R,j,c,32);ugn=float(torch.stack([q.grad.norm() for q in model.parameters() if q.grad is not None]).norm())
torch.cuda.synchronize();print(f'loss {l.item():.4g} finite; init |logratio| max {max(lr0):.3g}; enc grad norm {egn:.4g}; unet grad norm {ugn:.4g}; normalizer grads',[q.grad for q in p.normalizer.parameters() if q.grad is not None][:1])
P=[q for q in p.obs_encoder.parameters() if q.numel()>0];w0=[q.detach().clone() for q in P];step_policy_opts(opt,enc_opt);print('encoder params changed after step:',sum(int((a-b.detach()).abs().max().item()>0) for a,b in zip(w0,P)),'/',len(P))
with torch.no_grad():
    for nm,cc in [('stored',C[j]),('live',c.detach())]:
        print(nm,'cond init |logratio| max',max((logprob(*kernel(p,start(X[j,s],TS[s],global_cond=cc),TS[s],X[j,s]),Y[j,s])-LP[j,s]).abs().max().item() for s in range(10)))
print(f'64-sample minibatch (enc_mb 32) live-encoder update {time.time()-t0:.2f}s, peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
