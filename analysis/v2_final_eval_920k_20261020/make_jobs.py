import sys,json;sys.path.insert(0,"/data02/kemove/rl100-sim-repro/train/offline_multisku_v2_draft")
from ms_common import init_pose,SKUS
from pathlib import Path
R="/data02/kemove/rl100-sim-repro/";O=Path(R+"analysis/v2_final_eval_920k_20261020")
P=dict(v2=R+"runs/offline_multisku_v2_20261019_1743/round3/il_policy.pt",start=R+"runs/offline_multisku_v1_20261019_2100/promoted_round3.pt",bc=R+"runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt")
E=[]
for sku in SKUS:
  for s in range(920000,920020):
    dx,dy,yw=init_pose(s,.005,2.);E.append(dict(seed=s,sku=sku,dx=dx,dy=dy,yaw_deg=yw,cell=sku))
print(SKUS,len(E))
for n,p in P.items():
  d=O/n;d.mkdir(exist_ok=True)
  (d/"jobs.json").write_text(json.dumps(dict(policy=p,out=str(d),jobs=[dict(id=f"final_{i//5:04d}",kind="eval",mode="det",episodes=E[i:i+5]) for i in range(0,100,5)]),indent=1))
