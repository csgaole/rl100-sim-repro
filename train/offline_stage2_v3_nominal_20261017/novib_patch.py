"""Make train/bc_stage1/evaluate_bc.load_policy build the NO-VIB BCPolicy (P3 v3 checkpoints have no vib heads).
Import BEFORE anything calls load_policy. Strict state_dict loading is kept (any mismatch still raises)."""
import sys
from pathlib import Path
_H=Path(__file__).resolve().parent;_R=_H.parents[1]
for p in (str(_R/"train/bc_stage1"),str(_H)):
    if p in sys.path:sys.path.remove(p)
sys.path.insert(0,str(_R/"train/bc_stage1"));sys.path.insert(0,str(_H))
import evaluate_bc as _EB, bc_model_novib as _NV
_EB.BCPolicy=_NV.BCPolicy
assert _EB.BCPolicy is _NV.BCPolicy
