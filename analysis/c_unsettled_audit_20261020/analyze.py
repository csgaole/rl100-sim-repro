import json,glob,numpy as np,collections
R=[json.loads(l) for f in sorted(glob.glob('out_*.jsonl')) for l in open(f)]
fid=sum(r['rec_strict']==r['rep_strict'] and r['rec_last48']==r['rep_last48'] and abs(r['rec_peak']-r['rep_peak'])<1e-6 for r in R)
print('replay fidelity exact',fid,'/',len(R))
U=[r for r in R if r['rec_strict']=='unsettled'];C=[r for r in R if r['rec_strict']=='success']
def q(a):a=np.array(a);return 'med %.4g p10 %.4g p90 %.4g max %.4g'%(np.median(a),np.percentile(a,10),np.percentile(a,90),a.max())
for name,G in [('CONTROL strict-success',C)]+[(s+' unsettled',[r for r in U if r['sku']==s]) for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']]:
    if not G:continue
    print('\n==',name,'n',len(G))
    for k in ['lin_end','ang_end','lin_max48','ang_max48','rot_deg_last48','disp_mm_last48','rot_deg_ext','disp_mm_ext','up_end','up_min48','grip_force_end','grip_force_max48','rec_last48']:
        print('  ',k,q([r[k] for r in G]))
    print('   failing-cond steps (of 48) mean: lin %.1f ang %.1f up %.1f bot %.1f'%tuple(np.mean([r[k] for r in G]) for k in ['fail_lin48','fail_ang48','fail_up48','fail_bot48']))
    print('   any-step fail type: lin',sum(r['fail_lin48']>0 for r in G),'ang',sum(r['fail_ang48']>0 for r in G),'up',sum(r['fail_up48']>0 for r in G),'bot',sum(r['fail_bot48']>0 for r in G))
    print('   label face correct at end (up>.98):',sum(r['up_end']>.98 for r in G),' up_end_ext>.98:',sum(r['up_end_ext']>.98 for r in G))
    for k in ['0','12','24','48','96','120','240','480']:print('   pass if horizon +%s steps (hold): %d'%(k,sum(r['ok_end_plus'][k] for r in G)),end=';')
    print()
    fp=[r['first_permanent_stable_step'] for r in G];print('   permanently-stable-from step (480=ep end, None=never in +2s):',collections.Counter('never' if x is None else '<=480' if x<=480 else '<=528' if x<=528 else '<=600' if x<=600 else '<=720' if x<=720 else '<=960' for x in fp))
    # threshold sweeps on the original window
    for lt,at in [(.03,.15),(.05,.15),(.03,.3),(.05,.3),(.1,.5),(.15,1.0)]:
        n=sum(r['fail_up48']==0 and r['fail_bot48']==0 and r['lin_max48']<lt and r['ang_max48']<at for r in G)
        print('   thr lin<%.2f ang<%.2f -> pass %d'%(lt,at,n),end=';')
    print()
    # rocking diagnosis: ang trend in last 96 steps
    sl=[np.polyfit(range(96),r['ang_trace_last96'],1)[0]*240 for r in G]
    print('   ang-vel slope last 0.4s (rad/s per s):',q(sl))
    sign_changes=[]
    print('   ext ang every 0.05s (median over eps) first 10:',np.round(np.median([r['ang_ext_every12'][:10] for r in G],0),3).tolist())
    print('   ext lin every 0.05s (median) first 10:',np.round(np.median([r['lin_ext_every12'][:10] for r in G],0),4).tolist())
# examples C
print('\nC examples:')
for r in sorted([r for r in U if r['sku']=='SKU5C'],key=lambda r:r['ang_max48'])[::40]:
    print({k:(round(r[k],4) if isinstance(r[k],float) else r[k]) for k in ['round','seed','rec_last48','lin_end','ang_end','lin_max48','ang_max48','rot_deg_last48','disp_mm_last48','rot_deg_ext','up_end','grip_force_end','first_permanent_stable_step']},'ang_last96_every8',r['ang_trace_last96'][::8])
