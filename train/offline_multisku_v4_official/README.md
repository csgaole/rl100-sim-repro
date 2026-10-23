# offline_multisku_v4_official

Official-alignment offline RL for multi-SKU flip (plan §9). **New directory — does not resume v3.**

## Quick start (after Bill smoke OK)
```bash
CONFIRM_LAUNCH=yes \
MS_START_POLICY=/data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt \
S2_OUT=/data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_YYYYMMDD_HHMM \
EXCLUDE_GPUS=0,1 \
bash launch_v4_official.sh
```

Smoke:
```bash
CONFIRM_LAUNCH=yes MS_START_POLICY=... S2_OUT=.../offline_multisku_v4_smoke_... \
EXCLUDE_GPUS=0,1 bash launch_v4_official.sh --smoke --rounds 1 --collect-r0 10
```

See `ALIGNMENT.md` for diff vs official / v3.
