#!/usr/bin/env bash
# Usage: GPU=0 bash train/bc_stage1/start.sh   (fresh)   |   GPU=0 RESUME=1 bash train/bc_stage1/start.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
PYTHON=${BC_PYTHON:-/data02/kemove/sku09-bc-migration/env/bin/python}
GPU=${GPU:?set GPU index}
export CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU" MUJOCO_GL=egl OMP_NUM_THREADS=8
unset CUDA_LAUNCH_BLOCKING
OUT=$REPO/runs/bc_stage1; mkdir -p "$OUT"
EXTRA=""; [ "${RESUME:-0}" = 1 ] && EXTRA="--resume"
nohup "$PYTHON" -u train_bc_stage1.py --microbatch ${MICROBATCH:-64} --precision fp32 $EXTRA >> "$OUT/training.log" 2>&1 &
echo $! > "$OUT/train.pid"; echo "PID $(cat $OUT/train.pid) GPU $GPU LOG $OUT/training.log"
