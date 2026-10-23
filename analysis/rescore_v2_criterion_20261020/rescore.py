# [2026-10-20] NEW, CPU-only, read-only. Rescore saved rollout rows with v2 criterion: final label_up>.98 AND never dropped (bottom<.74); no velocity terms.
import json,glob,collections
R=".."+"/../runs/"
R="/data02/kemove/rl100-sim-repro/runs/"
V1=R+"offline_multisku_v1_20261019_2100/gate/"
B=R+"bc_multisku_v1_20261018/s43_lr2e-4_mb256/"
def rows(pat):return [json.loads(l) for f in sorted(glob.glob(pat)) for l in open(f) if l.strip()]
def new_ok(r):
    lr=r.get("loose_reason",r["reason"]);return r["final_label_up"]>.98 and lr not in ("drop","incomplete")
def old_ok(r):return bool(r["strict_success"])
def loose_ok(r):return bool(r.get("loose_success",r["success"]))
SK=["SKU5A","SKU5B","SKU5C","SKU5D","SKU5E"]
def rep(name,rs):
    out=dict(name=name,n=len(rs),strict=sum(map(old_ok,rs)),old_loose=sum(map(loose_ok,rs)),new=sum(map(new_ok,rs)),per_sku={})
    for s in SK:
        g=[r for r in rs if r["sku"]==s]
        if g:out["per_sku"][s]=dict(n=len(g),strict=sum(map(old_ok,g)),new=sum(map(new_ok,g)),new_wrong_face=sum((not new_ok(r)) and r["final_label_up"]<=.98 for r in g),new_drop=sum((not new_ok(r)) and r["final_label_up"]>.98 for r in g))
    out["new_not_loose"]=[(r["sku"],r["seed"]) for r in rs if new_ok(r)!=loose_ok(r)]
    out["new_fail"]=[(r["sku"],r["seed"],round(r["final_label_up"],3),r.get("loose_reason",r["reason"])) for r in rs if not new_ok(r)]
    return out
sets={
 "candidate_0300 gate pooled200 (start_select 880000 + start2_confirm 880020)":rows(V1+"start_select/rows/*.jsonl")+rows(V1+"start2_confirm/rows/*.jsonl"),
 "promoted_round3.pt (r3_m100) pooled200 (r3_m100_select + r3_m100_confirm)":rows(V1+"r3_m100_select/rows/*.jsonl")+rows(V1+"r3_m100_confirm/rows/*.jsonl"),
 "r1 IL (round1/il_policy.pt) pooled200 (r1_il_select + r3_input_confirm)":rows(V1+"r1_il_select/rows/*.jsonl")+rows(V1+"r3_input_confirm/rows/*.jsonl"),
 "BC cand0300 880000 dev set (development_nominal, 100)":rows(B+"development_nominal/eval_0300_*.episodes.jsonl"),
 "BC cand0300 910000 unbiased (100)":rows(B+"unbiased_910000/episodes_all.jsonl"),
}
res=[]
for k,v in sets.items():
    for r in v:
        if "sku" not in r:r["sku"]="?"
    o=rep(k,v);res.append(o);print(json.dumps(o))
json.dump(res,open("rescore_results.json","w"),indent=1)
