#!/usr/bin/env bash
# Usage: GPUS=5,6,7 TRAIN_GPU=5 bash train/offline_stage2/start.sh [--smoke]
set -euo pipefail
cd "$(dirname "$0")";REPO=$(cd ../.. && pwd)
PYTHON=${S2_PYTHON:-/data02/kemove/sku09-bc-migration/env/bin/python}
export S2_STD_MAX=${S2_STD_MAX:-0.01} MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4;mkdir -p $TMPDIR
if [ "${1:-}" = "--smoke" ]; then export S2_OUT=$REPO/runs/offline_stage2_smoke; else export S2_OUT=$REPO/runs/offline_stage2; fi
mkdir -p $S2_OUT
setsid nohup "$PYTHON" -u pipeline.py --gpus ${GPUS:-5,6,7} --train-gpu ${TRAIN_GPU:-5} --rounds ${ROUNDS:-3} --collect-per-gpu ${PER_GPU:-4} --ppo-steps ${PPO_STEPS:-2000} "$@" >> $S2_OUT/pipeline.stdout 2>&1 < /dev/null &
echo $! > $S2_OUT/pipeline.pid;echo "PID $(cat $S2_OUT/pipeline.pid) OUT $S2_OUT"
