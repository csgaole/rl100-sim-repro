import json,glob,math,numpy as np
from collections import Counter
O="/data02/kemove/rl100-sim-repro/train/bc_nominal_v3_20261017/unbiased_gate_910000"
A="/data02/kemove/rl100-pi05/logs/eval_sku09/eval_20261018_093835_gate910k_A_step11000.episodes.jsonl"
def ic(s):
    r=np.random.default_rng(s);dx,dy=r.uniform(-.005,.005,2);return float(dx),float(dy),float(r.uniform(-2,2))
R=[json.load(open(f)) for f in sorted(glob.glob(O+"/eval_0200_p?.json"))]
rows=sorted([json.loads(l) for f in sorted(glob.glob(O+"/eval_0200_p?.episodes.jsonl")) for l in open(f)],key=lambda r:r["seed"])
assert [r["seed"] for r in rows]==list(range(910000,910100))
sha=R[0]["checkpoint_sha256"];assert all(r["checkpoint_sha256"]==sha and r["sampler"]=="upstream" and r["position_half_range_m"]==.005 and r["yaw_half_range_deg"]==2. and r["success_criteria_version"]=="flip_pose_no_drop_v2" for r in R)
icbad=[r["seed"] for r in rows if (r["dx"],r["dy"],r["yaw_deg"])!=ic(r["seed"])]
S=dict(seeds=[910000,910099],checkpoint_sha256=sha,successes=sum(r["successes"] for r in R),strict_successes=sum(r["strict_successes"] for r in R),unsettled_successes=sum(r["unsettled_successes"] for r in R),episodes=sum(r["episodes"] for r in R),
 failure_counts={k:sum(r["failure_counts"][k] for r in R) for k in R[0]["failure_counts"]},reasons=dict(Counter(r["reason"] for r in rows)),strict_reasons=dict(Counter(r["strict_reason"] for r in rows)),stability=dict(Counter(r["stability_quality"] for r in rows)),
 ic_formula_mismatch=icbad,first10_ic=[(r["seed"],r["dx"],r["dy"],r["yaw_deg"]) for r in rows[:10]])
open(O+"/per_episode_success.jsonl","w").writelines(json.dumps(dict(seed=r["seed"],success=r["success"],strict_success=r["strict_success"],reason=r["reason"],strict_reason=r["strict_reason"],stability_quality=r["stability_quality"],dx=r["dx"],dy=r["dy"],yaw_deg=r["yaw_deg"]))+"\n" for r in rows)
a={j["seed"]:j for j in map(json.loads,open(A))}
def mcn(key):
    b=sum(1 for r in rows if r[key] and not a[r["seed"]][key]);c=sum(1 for r in rows if not r[key] and a[r["seed"]][key]);n=b+c
    p=min(1.,2*sum(math.comb(n,i) for i in range(min(b,c)+1))/2**n) if n else 1.
    return dict(rl100=sum(r[key] for r in rows),pi05_A=sum(a[r["seed"]][key] for r in rows),rl100_only=b,pi05_only=c,exact_two_sided_p=p,rl100_only_seeds=[r["seed"] for r in rows if r[key] and not a[r["seed"]][key]],pi05_only_seeds=[r["seed"] for r in rows if not r[key] and a[r["seed"]][key]])
S["pi05_ic_mismatch"]=[r["seed"] for r in rows if (r["dx"],r["dy"],r["yaw_deg"])!=(a[r["seed"]]["dx"],a[r["seed"]]["dy"],a[r["seed"]]["yaw_deg"])]
S["mcnemar_vs_pi05_A_step11000"]=dict(success=mcn("success"),strict_success=mcn("strict_success"))
S["wall_s"]=int(open(O+"/end_epoch.txt").read())-int(open(O+"/start_epoch.txt").read())
json.dump(S,open(O+"/summary.json","w"),indent=2);print(json.dumps(S,indent=1))
