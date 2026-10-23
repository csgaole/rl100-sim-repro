import json,math
from pathlib import Path
O=Path(__file__).resolve().parent;N=["v2","start","bc"];R={}
for n in N:
  rows=[json.loads(l) for f in sorted((O/n/"rows").glob("*.jsonl")) for l in open(f)];assert len(rows)==100,(n,len(rows))
  R[n]={(r["sku"],r["seed"]):r for r in rows}
SK=sorted({k[0] for k in R["v2"]});out={}
L=["# v2 final unbiased eval, seeds 920000-920019 x 5 SKUs (det DDIM10 eta=1, flip_up_no_floor_v3)",""]
def wil(k,n):
  z=1.96;ph=k/n;c=(ph+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);return [round(c-h,3),round(c+h,3)]
for n in N:
  rs=list(R[n].values());k=sum(r["success"] for r in rs);ks=sum(bool(r["strict_success"]) for r in rs)
  per={s:sum(r["success"] for r in rs if r["sku"]==s) for s in SK};pers={s:sum(bool(r["strict_success"]) for r in rs if r["sku"]==s) for s in SK}
  fr={};[fr.__setitem__(r["reason"],fr.get(r["reason"],0)+1) for r in rs if not r["success"]]
  sr={};[sr.__setitem__(r["strict_reason"],sr.get(r["strict_reason"],0)+1) for r in rs if not r["strict_success"]]
  fps={};[fps.setdefault(r["sku"],{}).__setitem__(r["reason"],fps.get(r["sku"],{}).get(r["reason"],0)+1) for r in rs if not r["success"]]
  out[n]=dict(policy=rs[0]["policy"],success=k,wilson95=wil(k,100),strict=ks,per_sku=per,per_sku_strict=pers,fail_reasons=fr,fail_by_sku=fps,strict_fail_reasons=sr)
  L.append(f"- **{n}** {k}/100 {wil(k,100)} strict {ks}/100 per-SKU {per} strict-per-SKU {pers} fails {fps} strict-fails {sr}")
def sign(w,l):
  n=w+l
  return 1.0 if n==0 else min(1,2*sum(math.comb(n,i) for i in range(min(w,l)+1))/2**n)
L.append("");out["paired"]={}
for a,b in [("v2","start"),("v2","bc"),("start","bc")]:
  for crit in ["success","strict_success"]:
    w=sum(bool(R[a][k][crit]) and not R[b][k][crit] for k in R[a]);l=sum(bool(R[b][k][crit]) and not R[a][k][crit] for k in R[a])
    out["paired"][f"{a}_vs_{b}_{crit}"]=dict(wins=w,losses=l,p=round(sign(w,l),4))
    L.append(f"- {a} vs {b} [{crit}]: wins {w} losses {l} two-sided sign p={sign(w,l):.4f}")
(O/"summary.json").write_text(json.dumps(out,indent=1));(O/"REPORT.md").write_text("\n".join(L)+"\n");print("\n".join(L))
