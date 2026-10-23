#!/usr/bin/env bash
# [offline_multisku_v1_draft] DRAFT launcher - NOT executed by the drafting agent. Copy of the base start_nominal.sh pattern.
# Usage (only after Bill approves):
#   CONFIRM_LAUNCH=yes S2_OUT=/data02/kemove/rl100-sim-repro/runs/offline_multisku_v1_<date> bash launch_ms_draft.sh [pipeline args]
# Optional env: GPUS=auto|<allow-list>  EXCLUDE_GPUS (default 0,1,2,3,5,6 = only GPUs 4+7; Bill 2026-10-19)  PER_GPU=8  ROUNDS=3  MS_START_POLICY=<ckpt>  MS_TRAIN_ALL=1
set -euo pipefail
cd "$(dirname "$0")"
[ "${CONFIRM_LAUNCH:-}" = "yes" ] || { echo "refuse: set CONFIRM_LAUNCH=yes (draft; launch needs Bill's approval)"; exit 1; }
PY=/data02/kemove/sku09-bc-migration/env/bin/python
export S2_OUT=${S2_OUT:?set S2_OUT to a NEW run dir}
export MS_START_POLICY=${MS_START_POLICY:-/data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt}
export S2_STD_MAX=0.1 S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4 MS_TRAIN_ALL=${MS_TRAIN_ALL:-1}
export EXCLUDE_GPUS=${EXCLUDE_GPUS:-0,1,2,3,5,6}
if [ -e "$S2_OUT/pipeline_state_ms.json" ] && [ "${RESUME:-0}" != "1" ]; then echo "refuse: $S2_OUT has a pipeline state (set RESUME=1 to resume)"; exit 1; fi
[ -f "$MS_START_POLICY" ] || { echo "missing start policy $MS_START_POLICY"; exit 1; }
# ---- free GPUs only: no compute process on the card and >= 16 GB free; honour EXCLUDE_GPUS (default 0,1,2,3)
mapfile -t FREE < <(
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | while IFS=', ' read -r i u f; do
    echo "$busy" | grep -q "$u" && continue; [ "$f" -ge 16000 ] || continue
    case ",${EXCLUDE_GPUS}," in *",$i,"*) continue;; esac; echo "$i"; done)
[ ${#FREE[@]} -gt 0 ] || { echo "refuse: no free GPU right now"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; exit 1; }
G1=${FREE[0]}; echo "free GPUs now: ${FREE[*]} (demo encoding on GPU $G1); EXCLUDE_GPUS=$EXCLUDE_GPUS"
mkdir -p "$S2_OUT"
[ -f "$S2_OUT/data/demo_train.npz" ] || CUDA_VISIBLE_DEVICES=$G1 MUJOCO_EGL_DEVICE_ID=$G1 $PY -u demo_features_ms.py --policy "$MS_START_POLICY" > "$S2_OUT/demo_features.log" 2>&1
# pipeline process itself needs no GPU (children get CUDA_VISIBLE_DEVICES per free GPU)
setsid nohup bash -c "CUDA_VISIBLE_DEVICES= $PY -u pipeline_ms.py --gpus ${GPUS:-auto} --exclude-gpus '$EXCLUDE_GPUS' --per-gpu ${PER_GPU:-8} --rounds ${ROUNDS:-3} $* >> $S2_OUT/pipeline.stdout 2>&1" < /dev/null > "$S2_OUT/launcher.log" 2>&1 &
echo $! > "$S2_OUT/launcher.pid"; echo "LAUNCHER PID $!"
