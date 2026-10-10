#!/usr/bin/env bash
# Diagnostic: raw round-1 v2 PPO trajectory with forced snapshots, gated as they appear (GPU7).
set -euo pipefail
cd "$(dirname "$0")";REPO=$(cd ../.. && pwd);V2=$REPO/runs/offline_stage2_wide_v2;O=$V2/round1_forced
PY=/data02/kemove/sku09-bc-migration/env/bin/python
export S2_STD_MAX=0.01 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4 S2_OUT=$V2 CUDA_VISIBLE_DEVICES=${GPU:-7} MUJOCO_EGL_DEVICE_ID=${GPU:-7}
mkdir -p $O/ppo
setsid nohup $PY -u ppo_wide_v2.py --policy $REPO/runs/offline_stage2_wide/round2/ppo/cand_step0150.pt --anchor-ref $REPO/runs/bc_stage1/selected_development.pt \
  --critic $V2/round1/iql/critic.pt --sources $V2/round1/sources.json --out $O/ppo --steps ${STEPS:-600} --lr 2e-06 --anchor 3.0 --nom-batch 32 \
  --force-save-steps 75,150,300,450,600 --no-revert --no-patience > $O/ppo.log 2>&1 < /dev/null &
echo $! > $O/ppo.pid
unset CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID
setsid nohup $PY -u forced_eval.py --ppo-dir $O/ppo --out $O --steps 75,150,300,450,600 --gpus ${GPU:-7} --per-gpu ${PER_GPU:-5} > $O/forced_eval.stdout 2>&1 < /dev/null &
echo $! > $O/eval.pid;echo "PPO $(cat $O/ppo.pid) EVAL $(cat $O/eval.pid) OUT $O"
