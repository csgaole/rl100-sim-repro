#!/usr/bin/env bash
# [offline_multisku_v4_official] Launcher. NEW run dir only; never touches v3.
# Usage:
#   CONFIRM_LAUNCH=yes MS_START_POLICY=<ckpt> S2_OUT=/data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_<tag> \
#     bash launch_v4_official.sh [--smoke] [pipeline args]
# Default EXCLUDE_GPUS=0,1 (BC s43). PPO DDP uses remaining free GPUs (≤6).
set -euo pipefail
cd "$(dirname "$0")"
[ "${CONFIRM_LAUNCH:-}" = "yes" ] || { echo "refuse: set CONFIRM_LAUNCH=yes"; exit 1; }
PY=/data02/kemove/sku09-bc-migration/env/bin/python
export S2_OUT=${S2_OUT:?set S2_OUT to a NEW offline_multisku_v4_* run dir}
export MS_START_POLICY=${MS_START_POLICY:?set MS_START_POLICY}
case "$S2_OUT" in
  *offline_multisku_v1_*|*offline_multisku_v2_*|*offline_multisku_v3_*) echo "refuse: S2_OUT must be a v4 run dir"; exit 1;;
esac
case "$S2_OUT" in
  *offline_multisku_v4*) ;;
  *) echo "refuse: S2_OUT should contain offline_multisku_v4"; exit 1;;
esac
export S2_STD_MAX=0.1 S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4 MS_TRAIN_ALL=${MS_TRAIN_ALL:-1}
export EXCLUDE_GPUS=${EXCLUDE_GPUS:-0,1}
if [ -e "$S2_OUT/pipeline_state_ms.json" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "refuse: $S2_OUT has pipeline state (set RESUME=1 to resume)"; exit 1
fi
[ -f "$MS_START_POLICY" ] || { echo "missing start policy $MS_START_POLICY"; exit 1; }
mapfile -t FREE < <(
  busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits | while IFS=', ' read -r i u f; do
    echo "$busy" | grep -q "$u" && continue; [ "$f" -ge 16000 ] || continue
    case ",${EXCLUDE_GPUS}," in *",$i,"*) continue;; esac; echo "$i"; done)
[ ${#FREE[@]} -gt 0 ] || { echo "refuse: no free GPU"; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv; exit 1; }
G1=${FREE[0]}; echo "free GPUs: ${FREE[*]} (demo encode on $G1); EXCLUDE_GPUS=$EXCLUDE_GPUS"
mkdir -p "$S2_OUT"
# write exclude override so pipeline free_gpus() matches launcher
echo "$EXCLUDE_GPUS" | tr ',' '\n' | sed '/^$/d' > "$S2_OUT/exclude_gpus_override.txt" || true
# Actually free_gpus expects comma or space list of ints on one line replace — write comma form
echo -n "$EXCLUDE_GPUS" > "$S2_OUT/exclude_gpus_override.txt"
[ -f "$S2_OUT/data/demo_train.npz" ] || CUDA_VISIBLE_DEVICES=$G1 MUJOCO_EGL_DEVICE_ID=$G1 $PY -u demo_features_ms.py --policy "$MS_START_POLICY" > "$S2_OUT/demo_features.log" 2>&1
setsid nohup bash -c "CUDA_VISIBLE_DEVICES= $PY -u pipeline_official_ms.py --gpus ${GPUS:-auto} --exclude-gpus '$EXCLUDE_GPUS' --per-gpu ${PER_GPU:-8} --rounds ${ROUNDS:-3} $* >> $S2_OUT/pipeline.stdout 2>&1" < /dev/null > "$S2_OUT/launcher.log" 2>&1 &
echo $! > "$S2_OUT/launcher.pid"; echo "LAUNCHER PID $!"
