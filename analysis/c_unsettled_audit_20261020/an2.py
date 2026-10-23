import json,glob,numpy as np,collections
R=[json.loads(l) for f in sorted(glob.glob('pos_*.jsonl')) for l in open(f)]
print('fidelity',sum(r['rec_strict']==r['rep_strict'] and r['rec_last48']==r['rep_last48'] for r in R),len(R))
def cls(r):
    if r['rec_strict']=='success':return 'success'
    return 'belt' if r['outfeed_contact_frac48']>0.5 and r['lin_max48']>0.1 else 'rest_jitter' if r['lin_max48']<0.05 else 'other'
for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']:
    G=[r for r in R if r['sku']==s]
    for c in ['success','belt','rest_jitter','other']:
        H=[r for r in G if cls(r)==c]
        if not H:continue
        y=[r['pos_end'][1] for r in H];vy=[r['vy_end'] for r in H]
        cn=collections.Counter(tuple(r['contacts_end']) for r in H).most_common(3)
        print(s,c,'n',len(H),'y_end med %.3f [%.3f,%.3f]'%(np.median(y),min(y),max(y)),'vy_end med %.3f'%np.median(vy),'outfeed_contact_end',sum(r['outfeed_contact_end'] for r in H),
              'lin48max med %.3g ang48max med %.3g'%(np.median([r['lin_max48'] for r in H]),np.median([r['ang_max48'] for r in H])),
              'pass@+2s',sum(r['ok_end_plus']['480'] for r in H),'contacts',cn)
J=[r for r in R if r['sku']=='SKU5C' and cls(r)=='rest_jitter']
print('\nC rest_jitter detail n',len(J))
for k in ['lin_max48','ang_max48','rot_deg_last48','disp_mm_last48','rot_deg_ext','disp_mm_ext','rec_last48','fail_ang48','fail_lin48']:
    a=np.array([r[k] for r in J]);print('  ',k,'med %.4g p90 %.4g max %.4g'%(np.median(a),np.percentile(a,90),a.max()))
print('  first_permanent_stable_step',collections.Counter('never' if r['first_permanent_stable_step'] is None else (r['first_permanent_stable_step']-480)//48*48 for r in J))
for lt,at,w in [(.03,.3,48),(.05,.3,48),(.03,.5,48),(.03,.15,24)]:
    pass
# window variants for rest_jitter: fraction of last48 steps stable, & median-of-window criterion
print('  stable steps last48 dist',collections.Counter(min(r['rec_last48']//8*8,48) for r in J))
O=[r for r in R if r['sku']=='SKU5C' and cls(r)=='other'];print('C other',[(r['seed'],round(r['lin_max48'],3),round(r['ang_max48'],3),r['outfeed_contact_frac48'],r['contacts_end']) for r in O][:10])
