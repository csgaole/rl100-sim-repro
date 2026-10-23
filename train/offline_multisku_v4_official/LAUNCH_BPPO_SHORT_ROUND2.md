# Round2 short BPPO validation (DO NOT LAUNCH until Bill confirms)
# Asia/Shanghai note written 2026-10-21
# Free GPUs at check time: only GPU 4 fully free (0 MiB).
# Busy (DO NOT TOUCH): liu 0-3, yu 5, so101 2/6/7.
# batch 512 divisible by world_size=1 → CUDA_VISIBLE_DEVICES=4, nproc_per_node=1

export S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 MUJOCO_GL=egl OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=4
cd /data02/kemove/rl100-sim-repro/train/offline_multisku_v4_official
mkdir -p /data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_20261020_2153/round2/bppo_short_fix

/data02/kemove/sku09-bc-migration/env/bin/python -m torch.distributed.run --standalone --nproc_per_node=1 \
  /data02/kemove/rl100-sim-repro/train/offline_multisku_v4_official/ppo_official_ms.py \
  --policy /data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt \
  --critic /data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_20261020_2153/round2/iql/critic.pt \
  --sources /data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_20261020_2153/round2/sources_for_feat.json \
  --out /data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_20261020_2153/round2/bppo_short_fix \
  --short \
  --batch 512 \
  --clip 0.25 \
  --anchor 6 \
  --adv-samples 4 --omega 0.7 \
  --ope-every 50 --ope-states 2048 --grad-clip 0.5

# --short implies: lr=1e-6, lp-reduce=mean, early-stop-kl=0.05,
#   no-clip-decay, no-ope-old-refresh, steps=500, milestones=100,250,500
# Optional omit --anchor 6 for no KL anchor.
