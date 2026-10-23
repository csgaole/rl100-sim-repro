#!/bin/bash
# GPU NAME SEED LR
cd /data02/kemove/rl100-sim-repro/train/bc_nominal_v3_20261017; G=$1; N=$2
CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G MUJOCO_GL=egl P3_TRAIN_GPU=$G nohup /data02/kemove/sku09-bc-migration/env/bin/python -u train_bc_nominal.py --run-dir /data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017/$N --mirror-dir /data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017/${N}_mirror --seed $3 --lr $4 --eval-gpus $G --eval-procs-per-gpu 5 --max-epochs 400 --first-gate-epoch 100 --gate-every 50 --min-stop-epoch 200 --microbatch 64 --precision fp32 > /data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017/$N.log 2>&1 &
echo $! > /data02/kemove/rl100-sim-repro/runs/bc_nominal_v3_20261017/$N.pid; echo $N pid $! gpu $G
