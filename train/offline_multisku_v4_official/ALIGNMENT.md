# offline_multisku_v4_official — alignment map

Date: 2026-10-20 Asia/Shanghai  
Plan: `/workspace/offline_official_align_plan.md` §9 defaults (Bill approved).  
Official: `RL-100` `rl100_2d_epsilon.yaml` + `train_ddp.py` / `unio4` / `uni_ppo.py`.

## Order
| Step | v4 | Official |
|------|----|----------|
| 1 | IL (accumulate) | IL |
| 2 | Feature reencode (full) | obs2feature |
| 3 | IQL (single-GPU) | IQL / critic |
| 4 | Dynamics ensemble 7/5 | dynamics |
| 5 | BPPO DDP | unio4 BPPO + train_ddp |
| 6 | OPE (dyn) + sim gate (option C) | OPE + eval |
| 7 | Collect | rollout |

## Hyperparams (§9)
- BPPO: 8000 / lr 1e-5 / global batch 512 / clip 0.25 + linear decay
- No offline KL anchor; no per-dim adaptive KL (approx_kl log only)
- Critic: Q512 no LN, lr 1e-4, τ 0.005, V512, expectile 0.7
- γ 0.99; std max keep 0.1; DDIM η=1 σ_final=0; IC ±5mm/±2°; sparse reward; keep_epsilon
- Promotion **C**: sim pooled gate AND dynamics-OPE both pass
- PPO DDP world_size ≤ 6 on GPUs 2–7 while BC holds 0–1; batch_per_gpu = 512/N
- IQL remains single-GPU

## Files
| File | Role |
|------|------|
| `pipeline_official_ms.py` | Official-order state machine |
| `ppo_official_ms.py` | BPPO + optional torchrun DDP |
| `dynamics_ms.py` | Ensemble TM → dynamics.pt |
| `ope_dynamics_ms.py` | Decision OPE (not reference-only) |
| `iql_ms.py` / `ms_common.py` | Official critic width/τ/lr/γ |
| `il_ms.py` | Accumulate historical successes |
| `pipeline_ms_v3_archived.py` | Read-only v3 order archive |

## Non-goals
- Do not resume `offline_multisku_v3_20261020_1851`
- Do not overwrite `offline_multisku_v3_draft`
- Do not touch BC s43 on GPUs 0–1
