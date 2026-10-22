#!/usr/bin/env bash
# Minimal online GAE-PPO launcher for piper multisku (official-aligned).
# Usage:
#   CONFIRM_LAUNCH=yes bash run.sh --smoke
#   CONFIRM_LAUNCH=yes bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
[ "${CONFIRM_LAUNCH:-}" = "yes" ] || { echo "refuse: set CONFIRM_LAUNCH=yes"; exit 1; }

PY=/data02/kemove/sku09-bc-migration/env/bin/python
REPO=/data02/kemove/rl100-sim-repro
# Best offline short-BPPO (KL=1 mean, cand_step0250) = 488/500 pooled
START=${MS_START_POLICY:-$REPO/runs/offline_multisku_v4_20261020_2153/round2/bppo_short_billB_kl1/cand_step0250.pt}
OUT=${S2_OUT:-$REPO/runs/online_billB_official_20261022}
GATE_BASELINE=${GATE_BASELINE:-488}
BC_BASELINE=${BC_BASELINE:-485}

export S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 S2_STD_MAX=0.1 MUJOCO_GL=egl
export TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4

[ -f "$START" ] || { echo "missing start $START"; exit 1; }

SMOKE=0
for x in "$@"; do
  case "$x" in --smoke) SMOKE=1;; esac
done

if [ "$SMOKE" = "1" ]; then
  OUT_SMOKE=${S2_OUT_SMOKE:-$REPO/runs/online_billB_official_20261022_smoke}
  export S2_OUT="$OUT_SMOKE"
  mkdir -p "$OUT_SMOKE"
  echo "SMOKE launch -> $OUT_SMOKE start=$START"
  exec "$PY" -u online_gae_ppo_ms.py \
    --start "$START" --out "$OUT_SMOKE" \
    --gate-baseline "$GATE_BASELINE" --bc-baseline "$BC_BASELINE" \
    --gpus "${GPUS:-7,6,5,4,3}" --exclude-gpus "${EXCLUDE_GPUS:-0,1,2}" \
    --smoke
fi

export S2_OUT="$OUT"
mkdir -p "$OUT"
if [ -e "$OUT/status.json" ] && [ "${RESUME:-0}" != "1" ]; then
  echo "refuse: $OUT has status (set RESUME=1 to resume with --continue-run)"
  exit 1
fi

CONT_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then CONT_FLAG="--continue-run"; fi

echo "REAL launch -> $OUT start=$START baseline=$GATE_BASELINE"
setsid nohup bash -c "\"$PY\" -u \"$(pwd)/online_gae_ppo_ms.py\" \
  --start \"$START\" --out \"$OUT\" \
  --gate-baseline \"$GATE_BASELINE\" --bc-baseline \"$BC_BASELINE\" \
  --gpus \"${GPUS:-7,6,5,4,3}\" --exclude-gpus \"${EXCLUDE_GPUS:-0,1,2}\" \
  --eps \"${EPS:-60}\" --rounds \"${ROUNDS:-20}\" \
  $CONT_FLAG >> \"$OUT/launcher.stdout\" 2>&1" < /dev/null > "$OUT/launcher.log" 2>&1 &
echo $! > "$OUT/launcher.pid"
echo "LAUNCHER PID $(cat "$OUT/launcher.pid")"
