#!/usr/bin/env bash
# Usage: S2_OUT=<new dir> GPUS=5,6,7 S2_STD_MAX=0.1 bash start_v3_ic5.sh [pipeline args]
set -euo pipefail
cd "$(dirname "$0")"
PY=/data02/kemove/sku09-bc-migration/env/bin/python
export V3_START_POLICY=/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/runs/p3_bc_v3_novib_seed42/development_v2/candidate_0088.pt
export S2_STD_MAX=${S2_STD_MAX:?set S2_STD_MAX} S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4
G1=${GPUS%%,*}
setsid nohup bash -c "CUDA_VISIBLE_DEVICES=$G1 $PY -u demo_features_v3.py > $S2_OUT/demo_features.log 2>&1 && $PY -u wide_pipeline_ic6.py --gpus $GPUS --per-gpu ${PER_GPU:-3} --rounds 3 $* >> $S2_OUT/pipeline.stdout 2>&1" < /dev/null > $S2_OUT/launcher.log 2>&1 &
echo $! > $S2_OUT/launcher.pid; echo "LAUNCHER PID $!"
