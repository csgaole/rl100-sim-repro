"""Numerical contracts: (1) kernel mean == upstream step_mean (deployment sampler) on a real obs;
(2) per-frame feature encoding == 2-frame stacked encoding used by evaluate_bc; (3) encoder deterministic."""
from s2common import *
setup(0);p,cp=load_policy(STAGE1);p.eval()
env,meta=make_env(700001,.005,2.);obs=env.observe();env.close()
z=encode_obs(p,obs);z2=encode_obs(p,obs);assert torch.equal(z,z2),'encoder nondeterministic'
o={k:torch.as_tensor(np.stack([obs['rgb'][k]]*2).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255 for k in CAMS}
o['agent_pos']=torch.as_tensor(np.stack([obs['proprio']]*2),device='cuda',dtype=torch.float32)
with torch.no_grad():ref_cond=p.obs_encoder(p.normalizer.normalize(o)).reshape(1,-1)
cond=torch.cat([z,z])[None];e1=(ref_cond-cond).abs().max().item();print('cond max diff',e1);assert e1<1e-4
noise=torch.randn((1,8,14),device='cuda',generator=torch.Generator(device='cuda').manual_seed(5))
with torch.no_grad():
    x=noise.clone();p.noise_scheduler.set_timesteps(10)
    for t in p.noise_scheduler.timesteps:
        eps=p.model(x,t.to('cuda'),global_cond=ref_cond);x=p.noise_scheduler.step_mean(eps,t,x,eta=1.0,generator=torch.Generator(device='cuda').manual_seed(1)).prev_sample
    mine=sample_chain(p.model,p,ref_cond,stochastic=False,noise=noise)
e2=(x-mine).abs().max().item();print('kernel vs step_mean max diff',e2);assert e2<1e-5
with torch.no_grad():
    xx,xs,ys,lps=sample_chain(p.model,p,ref_cond.repeat(4,1),torch.Generator(device='cuda').manual_seed(3),stochastic=True,trace=True)
print('stochastic final spread',(xx-mine).abs().mean().item(),'logprobs',[round(l.mean().item(),1) for l in lps])
print('SMOKE_KERNEL_PASSED')
