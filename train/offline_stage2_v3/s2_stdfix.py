# [proposed fix] DDIM stochastic sampler with correct terminal step (new file; leaves s2common_v3.py untouched).
# Root cause: s2common_v3.kernel used prev=max(t-step,0) so the last step (t=1->prev_raw=-9) took
# alphas_cumprod[0] and injected sigma~0.020 into x0. Early-step noise is mostly denoised away;
# terminal noise is not. That is why S2_STD_MAX=0.01 seemed OK (caps 0.020->0.01) and >=0.03 collapsed.
# Fix: when prev_raw<0 use noise_scheduler.final_alpha_cumprod (1.0 with set_alpha_to_one) => sigma=0.
# Env: S2_STD_MAX (required for clamp on non-final steps), S2_FIX_FINAL=1 (default on),
#       S2_FLOOR_FINAL=0|1 (paper-style sigma_min on terminal; hurts rollouts on this task),
#       S2_CLIP_IN_MEAN=0|1, S2_ARM_ONLY=0|1, S2_GRIP_ONLY=0|1.
# Wire-in: batch_worker_stdfix.py imports this kernel for sto mode; det path unchanged (step_mean).
# PPO note: sampling with terminal sigma=0 makes y=mean; use std.clamp(min=1e-8) inside logprob.
# A) S2_FIX_FINAL=1: prev_raw<0 -> final_alpha_cumprod (sigma=0 at terminal DDIM step)
# B) S2_CLIP_IN_MEAN=1: clipped sigma enters BOTH mean direction and noise (paper Eq.5+23)
# C) S2_ARM_ONLY=1: zero noise on gripper cols 6,13
import os, torch
from s2common_v3 import STD_MIN, logprob, JOINTS
import s2common_v3 as _base
STD_MAX=float(os.environ.get("S2_STD_MAX", _base.STD_MAX))
FIX_FINAL=os.environ.get("S2_FIX_FINAL","1")!="0"
CLIP_IN_MEAN=os.environ.get("S2_CLIP_IN_MEAN","0")=="1"
ARM_ONLY=os.environ.get("S2_ARM_ONLY","0")=="1"
FLOOR_FINAL=os.environ.get("S2_FLOOR_FINAL","0")=="1"
GRIP_ONLY=os.environ.get("S2_GRIP_ONLY","0")=="1"
GRIP=(6,13)

def kernel(p,eps,t,x):
    sc=p.noise_scheduler; idx=int(t)
    step=sc.config.num_train_timesteps//sc.num_inference_steps
    prev_raw=idx-step
    a=sc.alphas_cumprod[idx].to(x.device,x.dtype)
    if FIX_FINAL and prev_raw<0:
        ap=sc.final_alpha_cumprod.to(x.device,x.dtype)
    else:
        ap=sc.alphas_cumprod[max(prev_raw,0)].to(x.device,x.dtype)
    clean=((x-(1-a)**.5*eps)/a**.5).clamp(-1,1)
    var=((1-ap)/(1-a)*(1-a/ap)).clamp(min=0.)
    sig_raw=var.sqrt()
    if FIX_FINAL and prev_raw<0:
        std=sig_raw  # exact 0 with final_alpha_cumprod=1
        if FLOOR_FINAL: std=std.clamp(STD_MIN, STD_MAX)  # paper-style sigma_min floor
    else:
        std=sig_raw.clamp(STD_MIN, STD_MAX)
    if CLIP_IN_MEAN:
        mean=ap**.5*clean+(1-ap-std**2).clamp(min=0)**.5*eps
    else:
        mean=ap**.5*clean+(1-ap-var).clamp(min=0)**.5*eps
    return mean,std

def mask_std(std, ref):
    if ARM_ONLY:
        m=torch.ones_like(ref); m[...,GRIP]=0; return std*m
    if GRIP_ONLY:
        m=torch.zeros_like(ref); m[...,GRIP]=1; return std*m
    return std

def sample_chain(model,p,cond,gen=None,stochastic=True,trace=False,noise=None):
    p.noise_scheduler.set_timesteps(10)
    x=noise.clone() if noise is not None else torch.randn((cond.shape[0],8,14),device=cond.device,generator=gen)
    xs=[]; ys=[]; lps=[]
    for t in p.noise_scheduler.timesteps:
        eps=model(x,t.to(cond.device),global_cond=cond); mean,std=kernel(p,eps,t,x)
        std_m=mask_std(std,x)
        y=mean+std_m*torch.randn(x.shape,device=x.device,generator=gen) if stochastic else mean
        if trace:
            xs.append(x); ys.append(y)
            std_lp=std.clamp(min=1e-8)
            lps.append(logprob(mean,std_lp,y))
        x=y
    return (x,xs,ys,lps) if trace else x