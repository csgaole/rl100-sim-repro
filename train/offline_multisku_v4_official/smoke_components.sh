#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/data02/kemove/sku09-bc-migration/env/bin/python}
export S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 S2_STD_MAX=0.1 MUJOCO_GL=egl MS_SMOKE=1
echo "py_compile..."
$PY -m py_compile ms_common.py iql_ms.py il_ms.py tm_ms.py dynamics_ms.py ope_dynamics_ms.py \
  ppo_official_ms.py pipeline_official_ms.py amq_ms.py batch_worker_ms.py reencode_features_ms.py demo_features_ms.py
echo "Critic/GAMMA check (ms_common only)..."
$PY - <<'PY'
import os
os.environ.setdefault('S2_FIX_FINAL','1'); os.environ.setdefault('S2_FLOOR_FINAL','0'); os.environ.setdefault('S2_STD_MAX','0.1')
import ms_common as m
c=m.Critic(m.SD,m.ACT_DIM)
# first Linear out_features
q_lin=c.q1[0]; assert q_lin.out_features==512, q_lin.out_features
# no LayerNorm in q1
assert not any(isinstance(x,__import__('torch').nn.LayerNorm) for x in c.q1), c.q1
assert abs(m.GAMMA-0.99)<1e-9
print('CRITIC_OK Q512 noLN gamma', m.GAMMA, 'SD', m.SD)
PY
$PY pipeline_official_ms.py --help | head -25
echo "SMOKE_COMPONENTS_DONE"
