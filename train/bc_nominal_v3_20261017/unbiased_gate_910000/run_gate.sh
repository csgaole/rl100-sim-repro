#!/bin/bash
# Unbiased re-gate: identical to nominal_gate.run_stage (5 procs x 20 eps, eval_novib.py, --sampler upstream, defaults p5_y2) but seed base 910000.
H=/data02/kemove/rl100-sim-repro/train/bc_nominal_v3_20261017
O=$H/unbiased_gate_910000
C=/data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017/s43_lr2e-4/development_v2/candidate_0200.pt
G=2
export CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G MUJOCO_GL=egl OMP_NUM_THREADS=4
date +%s > $O/start_epoch.txt; sha256sum $C > $O/checkpoint.sha256
for k in 0 1 2 3 4; do s=$((910000+k*20))
 /data02/kemove/sku09-bc-migration/env/bin/python $H/eval_novib.py --checkpoint $C --output $O/eval_0200_p$k.json --seed-start $s --episodes 20 --sampler upstream > $O/eval_0200_p$k.log 2>&1 &
 echo $! >> $O/pids.txt
done
wait; date +%s > $O/end_epoch.txt; echo DONE > $O/DONE
