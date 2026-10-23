# BPPO fix flags (2026-10-21 Asia/Shanghai)

Patched in `ppo_official_ms.py` / `ms_common.logprob` / `pipeline_official_ms.py`.
Old unio4 behavior is the default when `--short` is omitted.

| Flag | Default (legacy) | `--short` default | Meaning |
|---|---|---|---|
| `--lr` | 1e-5 | 1e-6 | Adam lr |
| `--lp-reduce` | sum (sum-96) | mean | mean log-prob over 8×12 joint dims |
| `--early-stop-kl` | 0 (off) | 0.05 | stop when approx_kl exceeds |
| `--clip-decay` / `--no-clip-decay` | decay ON | fixed clip 0.25 | no linear clip decay |
| `--ope-old-refresh` / `--no-ope-old-refresh` | ON | OFF | no OPE-gated pi_old refresh |
| `--anchor` | 0 | (unchanged; pass `--anchor 6`) | optional KL anchor to round-start |
| `--steps` / `--milestones` | 8000 / 1000,2000,4000,8000 | 500 / 100,250,500 | short validation |

Pipeline passthrough: `--bppo-short`, `--bppo-lp-reduce`, `--bppo-early-stop-kl`,
`--bppo-no-clip-decay`, `--bppo-no-ope-old-refresh`, `--bppo-anchor`.
