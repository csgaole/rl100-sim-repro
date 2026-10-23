# [diag] Stage-2 sampler fixes + diagnostics helpers. Does not modify s2common_v3.py.
# Fixes vs s2common_v3.kernel:
#  A) final step: when prev_raw<0 use final_alpha_cumprod (=1 with set_alpha_to_one) so sigma=0 (DDIM terminal)
#  B) optional paper-faithful clip: put clipped sigma into BOTH mean direction and noise (Eq.5 + Eq.23)
#  C) optional arm-only noise: zero std on gripper cols 6,13 (overwritten to 0.012 at exec anyway)
# Env: S2_STD_MAX, S2_FIX_FINAL=1, S2_CLIP_IN_MEAN=1, S2_ARM_ONLY=1
import os,math
import torch
from s2common_v3 import *  # noqa: F401,F403 — re-export; we override kernel/sample_chain below
from s2common_v3 import STD_MIN,STD_MAX,JOINTS,logprob

FIX_FINAL=os.environ.get('S2_FIX_FINAL','1')!='0'
CLIP_IN_MEAN=os.environ.get('S2_CLIP_IN_MEAN','0')=='1'
ARM_ONLY=os.environ.get('S2_ARM_ONLY','0')=='1'
GRIP= (6,13)

def kernel(p,eps,t,x):
    sc=p.noise_scheduler;idx=int(t)
    step=sc.config.num_train_timesteps//sc.num_inference_steps
    prev_raw=idx-step
    if FIX_FINAL and prev_raw<0:
        a=sc.alphas_cumprod[idx].to(x.device,x.dtype);ap=sc.final_alpha_cumprod.to(x.device,x.dtype)
    else:
        prev=max(prev_raw,0)
        a=sc.alphas_cumprod[idx].to(x.device,x.dtype);ap=sc.alphas_cumprod[prev].to(x.device,x.dtype)
    clean=((x-(1-a)**.5*eps)/a**.5).clamp(-1,1)
    var=(1-ap)/(1-a)*(1-a/ap)
    # numerical guard when ap==1 (final step)
    var=torch.clamp(var,min=0.)
    sig_raw=var.sqrt()
    std=sig_raw.clamp(STD_MIN,STD_MAX)
    if FIX_FINAL and prev_raw<0:
        std=sig_raw  # keep exact 0 at terminal; do not floor
    if CLIP_IN_MEAN:
        mean=ap**.5*clean+(1-ap-std**2).clamp(min=0)**.5*eps
    else:
        mean=ap**.5*clean+(1-ap-var).clamp(min=0)**.5*eps
    return mean,std

def _mask_std(std,ref):
    if not ARM_ONLY:return std
    m=torch.ones_like(ref);m[...,GRIP]=0
    return std*m

def sample_chain(model,p,cond,gen=None,stochastic=True,trace=False,noise=None):
    p.noise_scheduler.set_timesteps(10)
    x=noise.clone() if noise is not None else torch.randn((cond.shape[0],8,14),device=cond.device,generator=gen)
    xs=[];ys=[];lps=[]
    for t in p.noise_scheduler.timesteps:
        eps=model(x,t.to(cond.device),global_cond=cond);mean,std=kernel(p,eps,t,x)
        std_m=_mask_std(std,x)
        y=mean+std_m*torch.randn(x.shape,device=x.device,generator=gen) if stochastic else mean
        if trace:xs.append(x);ys.append(y);lps.append(logprob(mean,std.clamp(STD_MIN,STD_MAX).clamp(min=1e-8),y))
        x=y
    return (x,xs,ys,lps) if trace else x
