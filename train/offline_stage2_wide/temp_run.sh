#!/usr/bin/env bash
# Held-out noise-temperature test (x_T <- T*x_T), 2 lanes on one GPU. Usage: GPU=1 bash temp_run.sh
cd "$(dirname "$0")";R=/data02/kemove/rl100-sim-repro/runs;O=$R/temp_heldout;mkdir -p $O
PY=/data02/kemove/sku09-bc-migration/env/bin/python;export S2_STD_MAX=0.01 MUJOCO_GL=egl TMPDIR=/data02/kemove/tmp OMP_NUM_THREADS=4
IT10=$R/online_rl_pilot/heldout/it010/it010.pt;R2=$R/offline_stage2_wide/round2/ppo/cand_step0150.pt;S1=$R/bc_stage1/selected_development.pt
run(){ # name policy T
  [ -f $O/$1_T$3/result.json ] && return 0
  $PY -u heldout_temp.py --policy $2 --temperature $3 --out $O/$1_T$3 --gpu ${GPU:-1} --workers ${WORKERS:-8} --ref it010_T1=$R/online_rl_pilot/heldout/it010/result.json >> $O/$1_T$3.stdout 2>&1; }
( run it010 $IT10 0.5; run r2cand $R2 0.5; run stage1 $S1 0.5; run it010 $IT10 0.0; run stage1 $S1 0.0 ) &
( run it010 $IT10 0.3; run r2cand $R2 0.3; run stage1 $S1 0.3; run r2cand $R2 0.0 ) &
wait;echo TEMP_ALL_DONE $(date)
