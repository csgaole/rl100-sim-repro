#!/usr/bin/env bash
# [NOMINAL 2026-10-17] Usage: S2_OUT=<new dir> V3_START_POLICY=<nominal BC selected_development.pt> GPUS=0,1,2,3 bash start_nominal.sh [pipeline args]
set -euo pipefail
cd "$(dirname "$0")"
PY=/data02/kemove/sku09-bc-migration/env/bin/python
export V3_START_POLICY=${V3_START_POLICY:?set nominal BC policy} S2_STD_MAX=${S2_STD_MAX:-0.1} S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4
mkdir -p $S2_OUT; G1=${GPUS%%,*}
[ -f $S2_OUT/data/demo_train.npz ] || CUDA_VISIBLE_DEVICES=$G1 S2_OUT=$S2_OUT $PY -u demo_features_v3.py --policy $V3_START_POLICY > $S2_OUT/demo_features.log 2>&1
setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$G1 $PY -u wide_pipeline_ic7s9.py --gpus $GPUS --per-gpu ${PER_GPU:-3} --rounds ${ROUNDS:-3} --gate-n 100 --screen-n 60 --seed-rollouts \"\" $* >> $S2_OUT/pipeline.stdout 2>&1" < /dev/null > $S2_OUT/launcher.log 2>&1 &
echo $! > $S2_OUT/launcher.pid; echo "LAUNCHER PID $!"
