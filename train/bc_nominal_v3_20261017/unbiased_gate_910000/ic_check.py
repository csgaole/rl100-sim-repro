import numpy as np,json,glob
def ic(seed,ph=.005,yh=2.):
    r=np.random.default_rng(seed);dx,dy=r.uniform(-ph,ph,2);return float(dx),float(dy),float(r.uniform(-yh,yh))
def check(pat):
    rows=[json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f)]
    bad=[r["seed"] for r in rows if max(abs(a-b) for a,b in zip(ic(r["seed"]),(r["dx"],r["dy"],r["yaw_deg"])))>0]
    return len(rows),bad,rows
n,bad,_=check("runs/bc_nominal_v3_20261017/s43_lr2e-4/development_nominal/eval_0200_p*.episodes.jsonl")
print("880000 gate rows",n,"mismatch",bad)
n,bad,rows=check("train/bc_nominal_v3_20261017/unbiased_gate_910000/eval_0200_p*.episodes.jsonl")
print("910000 gate rows so far",n,"mismatch",bad)
for s in range(910000,910010):
    dx,dy,y=ic(s);print(s,"dx_mm=%.4f dy_mm=%.4f yaw_deg=%.4f"%(dx*1e3,dy*1e3,y))
