#!/bin/bash
# GPU NAME SEED LR MICROBATCH   (multi-SKU BC; effective batch 512 = SKU09 recipe, lr 2e-4 unchanged)
cd /data02/kemove/rl100-sim-repro/train/bc_multisku_v1_20261018; G=$1; N=$2; R=/data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018
[ -e $R/$N ] && { echo "refuse: $R/$N exists"; exit 1; }
used=$(nvidia-smi --id=$G --query-gpu=memory.used --format=csv,noheader,nounits); [ "$G" = "7" ] && [ "$used" -lt 1000 ] || { echo "refuse: GPU $G not 7 or busy ($used MiB)"; exit 1; }
CUDA_VISIBLE_DEVICES=$G MUJOCO_EGL_DEVICE_ID=$G MUJOCO_GL=egl P3_TRAIN_GPU=$G nohup /data02/kemove/sku09-bc-migration/env/bin/python -u train_bc_multisku.py --run-dir $R/$N --mirror-dir $R/${N}_mirror --seed $3 --lr $4 --eval-gpus $G --eval-procs-per-gpu 10 --max-epochs 400 --first-gate-epoch 100 --gate-every 50 --min-stop-epoch 200 --microbatch $5 --effective-batch 512 --precision fp32 > $R/$N.log 2>&1 &
echo $! > $R/$N.pid; echo $N pid $! gpu $G
