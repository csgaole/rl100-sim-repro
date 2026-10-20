# [offline_multisku_v2_draft 2026-10-20] NEW helper (read-only). Prints the best ACCEPTED policy of the v1 run to use as v2 MS_START_POLICY.
# Accepted = v1 start policy, any gate-promoted candidate, any adopted IL policy (v1's own accept chain). Ranked by POOLED strict successes
# (select 880000-set + confirm 880020-set) where both cached gate results exist; policies lacking a confirm result are listed with select only
# and flagged (run a confirm gate for them before choosing, or accept v1's final current_policy). Never touches 920000+.
import json,sys,hashlib
from pathlib import Path
RUN=Path(sys.argv[1] if len(sys.argv)>1 else '/data02/kemove/rl100-sim-repro/runs/offline_multisku_v1_20261019_2100')
S=json.loads((RUN/'pipeline_state_ms.json').read_text())
status=json.loads((RUN/'status_ms.json').read_text()).get('status')
acc=[('start',S['start_policy'])]+[(f'promoted{i}',p) for i,p in enumerate(S.get('promoted',[]))]
for r,R in sorted(S['rounds'].items()):
    if R.get('il_decision',{}).get('next_input')=='il':acc.append((f'r{r}_il',R['il_decision']['path']))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<23),b''):h.update(b)
    return h.hexdigest()
idx=S.get('gate_index',{});rows=[]
for name,p in acc:
    s=sha(p);res={}
    for which,seed0 in (('select',S['seed_table']['select'][0]),('confirm',S['seed_table']['confirm'][0])):
        k=f"{s}:{which}:{seed0}:{S['seed_table']['episodes_per_sku']}"
        if k in idx and Path(idx[k]).exists():res[which]=json.loads(Path(idx[k]).read_text())['successes']
    rows.append(dict(name=name,path=p,sha=s[:12],select=res.get('select'),confirm=res.get('confirm'),pooled=(res['select']+res['confirm']) if len(res)==2 else None))
print(json.dumps(dict(v1_status=status,v1_final_current_policy=S['current_policy'],accepted=rows),indent=1))
full=[r for r in rows if r['pooled'] is not None]
best_sel=max(rows,key=lambda r:(r['select'] or -1))
print('SUGGEST (pooled, complete only):',max(full,key=lambda r:r['pooled'])['path'] if full else None)
print('SUGGEST (select-only, incl. policies without confirm):',best_sel['path'],'| v1 final current_policy:',S['current_policy'])
if status!='complete':print('WARNING: v1 not complete yet - re-run after it finishes')
