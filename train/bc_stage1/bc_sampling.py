"""BC-only DDIM sampling with the standard terminal alpha convention.

The vendored RL likelihood scheduler is left unchanged. This implementation is
an explicit diagnostic/BC correction, not an upstream-exact RL sampler.
"""
import numpy as np
import torch
from diffusers import DDIMScheduler

@torch.inference_mode()
def sample_standard_ddim(p, history, generator):
    obs={k:torch.as_tensor(np.stack([h['rgb'][k] for h in history]).transpose(0,3,1,2),device=p.device,dtype=torch.float32)/255 for k in p.rgb_obs_keys}
    obs['agent_pos']=torch.as_tensor(np.stack([h['proprio'] for h in history]),device=p.device,dtype=torch.float32)
    cond=p.obs_encoder(p.normalizer.normalize(obs)).reshape(1,-1)
    scheduler=getattr(p,'_bc_standard_scheduler',None)
    if scheduler is None:
        scheduler=DDIMScheduler.from_config(p.noise_scheduler.config)
        p._bc_standard_scheduler=scheduler
    scheduler.set_timesteps(10)
    x=torch.randn((1,p.n_action_steps,14),device=p.device,generator=generator)
    for t in scheduler.timesteps:
        eps=p.model(x,t.to(p.device),global_cond=cond)
        x=scheduler.step(eps,int(t),x,eta=0,generator=generator).prev_sample
    actions=p.normalizer['action'].unnormalize(x)[0].cpu().numpy()
    actions[:,[6,13]]=.012
    assert actions.shape==(8,14) and np.isfinite(actions).all()
    return actions
