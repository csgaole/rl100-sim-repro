#!/usr/bin/env bash
# One-command short BPPO validation (Bill B) for round2 from candidate_0300.
# Does NOT touch pipeline / IQL / Dyn / foreign jobs. Does NOT delete artifacts.
# Usage:
#   CONFIRM_LAUNCH=yes bash launch_bppo_short_r2.sh
#   CONFIRM_LAUNCH=yes BPPO_GPUS=4 bash launch_bppo_short_r2.sh
#   CONFIRM_LAUNCH=yes KL_ANCHOR_COEF=6 bash launch_bppo_short_r2.sh
#   CONFIRM_LAUNCH=yes DRY_RUN=1 bash launch_bppo_short_r2.sh   # print cmd only
set -euo pipefail
cd "$(dirname "$0")"
[ "${CONFIRM_LAUNCH:-}" = "yes" ] || { echo "refuse: set CONFIRM_LAUNCH=yes"; exit 1; }
PY=/data02/kemove/sku09-bc-migration/env/bin/python
RUN=${RUN:-/data02/kemove/rl100-sim-repro/runs/offline_multisku_v4_20261020_2153}
POLICY=${POLICY:-/data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt}
CRITIC=${CRITIC:-$RUN/round2/iql/critic.pt}
SOURCES=${SOURCES:-$RUN/round2/sources_for_feat.json}
OUT=${OUT:-$RUN/round2/bppo_short_billB}
BATCH=${BATCH:-512}
STEPS=${STEPS:-500}
KL_ANCHOR_COEF=${KL_ANCHOR_COEF:-0}
PREFER=${PREFER_GPUS:-4,5,6,7}
EXCLUDE_ALWAYS=${EXCLUDE_GPUS:-0,1,2,3}

# GPU is "foreign-busy" if any compute process is on it
mapfile -t FREE < <(python3 - <<'PY'
import subprocess, os
prefer=os.environ.get("PREFER","4,5,6,7")
excl=os.environ.get("EXCLUDE_ALWAYS","0,1,2,3")
prefer_set={int(x) for x in prefer.split(",") if x.strip()!=""}
excl_set={int(x) for x in excl.split(",") if x.strip()!=""}
uu=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid,memory.free","--format=csv,noheader,nounits"], text=True)
busy_uuids=set()
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid","--format=csv,noheader"], text=True).strip()
for line in apps.splitlines() if apps else []:
    busy_uuids.add(line.strip())
cands=[]
for line in uu.strip().splitlines():
    idx_s, uid, free_s = [x.strip() for x in line.split(",")]
    idx=int(idx_s); free=float(free_s)
    if idx in excl_set: continue
    if idx not in prefer_set: continue
    if uid in busy_uuids: continue
    if free < 20000: continue
    cands.append(idx)
print("\n".join(str(i) for i in cands))
PY
)
# export for python
export PREFER="$PREFER" EXCLUDE_ALWAYS="$EXCLUDE_ALWAYS"

if [ -n "${BPPO_GPUS:-}" ]; then
  IFS=',' read -r -a FREE <<< "$BPPO_GPUS"
fi
[ ${#FREE[@]} -gt 0 ] || { echo "refuse: no free GPU among prefer=$PREFER (excluding busy+EXCLUDE=$EXCLUDE_ALWAYS)"; nvidia-smi; exit 1; }

N=1
for cand in 4 2 1; do
  if [ ${#FREE[@]} -ge "$cand" ] && [ $((BATCH % cand)) -eq 0 ]; then N=$cand; break; fi
done
USE=("${FREE[@]:0:$N}")
GPU_LIST=$(IFS=,; echo "${USE[*]}")
echo "SHORT_BPPO gpus=$GPU_LIST n=$N batch=$BATCH steps=$STEPS out=$OUT"

mkdir -p "$OUT"
EXTRA=()
if [ "$KL_ANCHOR_COEF" != "0" ]; then
  EXTRA+=(--anchor "$KL_ANCHOR_COEF")
fi
CMD=(env CUDA_VISIBLE_DEVICES="$GPU_LIST" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$N"
  "$(pwd)/ppo_official_ms.py"
  --policy "$POLICY" --critic "$CRITIC" --sources "$SOURCES" --out "$OUT"
  --short --steps "$STEPS" --batch "$BATCH" --clip 0.25 --no-clip-decay
  --lp-reduce mean --early-stop-kl 0.05 --no-ope-old-refresh
  --milestones 100,250,500 --lr 1e-6
  --adv-samples 4 --omega 0.7 --ope-every 50 --ope-states 2048 --grad-clip 0.5
  "${EXTRA[@]}")
{
  printf 'EXACT_CMD '
  printf '%q ' "${CMD[@]}"
  echo
} | tee "$OUT/launch_cmd.txt"

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1 — not launching"
  exit 0
fi
nohup "${CMD[@]}" >> "$OUT/ppo_short.stdout" 2>&1 &
echo $! > "$OUT/launcher.pid"
echo "LAUNCHED pid=$! log=$OUT/ppo_short.stdout"
