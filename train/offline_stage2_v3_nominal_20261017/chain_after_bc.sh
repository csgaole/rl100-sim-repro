#!/usr/bin/env bash
# [NOMINAL 2026-10-17] Wait for all nominal BC runs to finish, pick best selected_development.pt (nominal 880000-880099 successes, tie->lower val loss), start offline R1.
set -uo pipefail
B=/data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017; RUNS="s42_lr2e-4 s43_lr2e-4 s44_lr2e-4 s42_lr1e-4"
D=$(cd "$(dirname "$0")" && pwd)
while :; do n=0; for r in $RUNS; do st=$(python3 -c "import json;print(json.load(open(\"$B/$r/status.json\")).get(\"status\"))" 2>/dev/null); pid=$(cat $B/$r.pid); { [ "$st" = complete ] || ! kill -0 $pid 2>/dev/null; } && n=$((n+1)); done; [ $n -eq 4 ] && break; sleep 300; done
BEST=$(python3 - <<PY
import json,os
best=None
for r in "$RUNS".split():
    p="$B/"+r+"/development_nominal/decision.json"
    if not os.path.exists(p):continue
    s=json.load(open(p))["selected"];k=(s["successes"],-s["validation_loss"])
    if best is None or k>best[0]:best=(k,r,s)
print(best[1]);json.dump(dict(run=best[1],selected=best[2]),open("$B/BC_SELECTION.json","w"),indent=2)
PY
)
[ -z "$BEST" ] && { echo "no selection" ; exit 1; }
OUT=/data02/kemove/rl100-sim-repro/runs/offline_stage2_v3nominal_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M)
mkdir -p $OUT; cp $B/$BEST/selected_development.pt $OUT/start_policy_nominal_bc.pt
echo "BC=$BEST -> $OUT" ; S2_OUT=$OUT V3_START_POLICY=$OUT/start_policy_nominal_bc.pt GPUS=0,1,2,3 PER_GPU=${PER_GPU:-3} ROUNDS=${ROUNDS:-3} bash $D/start_nominal.sh
