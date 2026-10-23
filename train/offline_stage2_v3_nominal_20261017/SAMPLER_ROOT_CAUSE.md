# Stochastic collapse root cause (e88)

## Formula
eta=1 DDIM posterior: sigma = sqrt( (1-a_prev)/(1-a) * (1-a/a_prev) ), then clamp to [STD_MIN, S2_STD_MAX].
Noise added in **normalized** action space on x_t->x_{t-1} (eps-prediction). Unnormalize after full 10-step chain; gripper cols overwritten to 0.012 at exec.

## Bug
Last timestep t=1 has prev_raw=-9. Scheduler intends final_alpha_cumprod=1 (set_alpha_to_one) => sigma=0.
Our s2common_v3.kernel used prev=max(prev_raw,0)=0 => alphas_cumprod[0] => **sigma≈0.020** injected into the **final** sample.
Early-step raw sigma is 0.26–0.87 but is largely corrected by later denoising; **terminal noise is not**.

## Why max=0.01 "worked" (~59%) and max>=0.03 collapsed (~0%)
With max=0.01, final sigma clamped 0.020→0.01 (tolerable).
With max>=0.03, final sigma passes through at 0.020 → near-total failure.
Arm-only noise with buggy final still collapses; fix-final restores success at 0.03 and 0.1.

## Fix
s2_stdfix.py + batch_worker_stdfix.py: use final_alpha_cumprod when prev_raw<0.
Optional S2_FLOOR_FINAL=1 for paper-style sigma_min on terminal (PPO log-prob support).
