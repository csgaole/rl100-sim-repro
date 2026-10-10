"""Aggregate generalization sweep -> summary.csv/json, heatmap PNG, failure characterization, REPORT data."""
import os,sys,json,math,glob,collections
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=Path(os.environ.get('SWEEP_OUT',str(ROOT/'runs/generalization_sweep')))
sys.path.insert(0,str(HERE));from sweep_eval import cells,POS,YAW,NEP
def wilson(k,n,z=1.95996398454):
    if n==0:return [None,None]
    p=k/n;c=(p+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);return [round(c-h,4),round(c+h,4)]
rows={};table=[]
for c in cells():
    f=OUT/'cells'/f"{c['name']}.jsonl"
    R=[json.loads(l) for l in open(f)] if f.exists() else []
    R={r['seed']:r for r in R};R=[R[k] for k in sorted(R)];rows[c['name']]=R
    k=sum(bool(r['success']) for r in R);n=len(R);reasons=collections.Counter(r['reason'] for r in R if not r['success'])
    table.append(dict(cell=c['name'],pos_half_mm=c['pos_mm'],yaw_half_deg=c['yaw_deg'],factor=c['factor'],value=c['value'],episodes=n,complete=n==NEP,
        successes=k,success_rate=round(k/n,4) if n else None,wilson95_lo=wilson(k,n)[0],wilson95_hi=wilson(k,n)[1],
        wrong_face=reasons.get('wrong_face',0),drop=reasons.get('drop',0),below_height=reasons.get('below_height',0),incomplete=reasons.get('incomplete',0),
        other_fail={k2:v for k2,v in reasons.items() if k2 not in('wrong_face','drop','below_height','incomplete')}))
import csv
with open(OUT/'summary.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0].keys()));w.writeheader();[w.writerow({**t,'other_fail':json.dumps(t['other_fail'])}) for t in table]
(OUT/'summary.json').write_text(json.dumps(table,indent=1))
# heatmap
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    H=np.full((len(YAW),len(POS)),np.nan);N=np.zeros_like(H)
    for t in table:
        if t['factor']=='none' and t['episodes']:H[YAW.index(t['yaw_half_deg']),POS.index(t['pos_half_mm'])]=t['success_rate']*100;N[YAW.index(t['yaw_half_deg']),POS.index(t['pos_half_mm'])]=t['episodes']
    fig,ax=plt.subplots(figsize=(7.5,6));im=ax.imshow(H,origin='lower',cmap='RdYlGn',vmin=0,vmax=100)
    ax.set_xticks(range(len(POS)));ax.set_xticklabels([f'±{p}' for p in POS]);ax.set_yticks(range(len(YAW)));ax.set_yticklabels([f'±{y}' for y in YAW])
    ax.set_xlabel('position half-range per axis (mm)');ax.set_ylabel('yaw half-range (deg)')
    for i in range(len(YAW)):
        for j in range(len(POS)):
            if not np.isnan(H[i,j]):ax.text(j,i,f'{H[i,j]:.0f}%\n(n={int(N[i,j])})',ha='center',va='center',fontsize=8)
    ax.add_patch(plt.Rectangle((-.5,-.5),1,1,fill=False,ec='blue',lw=2))
    ax.set_title('Stage-1 BC (epoch250) success, upstream DDIM10 eta=1\npaired seeds 600000+, blue = demo range (±5mm, ±2°)');fig.colorbar(im,label='success %')
    fig.tight_layout();fig.savefig(OUT/'heatmap_success.png',dpi=130);plt.close(fig)
except Exception as e:print('plot failed',e)
# failure characterization (grid cells only)
A=[r for c in cells() if c['factor']=='none' for r in rows[c['name']]]
def stats(R):
    out={}
    if not R:return out
    s=np.array([bool(r['success']) for r in R])
    out['n']=len(R);out['success']=int(s.sum())
    # where failures happen: chunk where label_up first exceeds 0 (passes vertical) and max label_up
    def first(arr,th):
        i=[k for k,v in enumerate(arr) if v>th];return i[0] if i else None
    ups=[r['diag']['up'] for r in R];mx=np.array([max(u) for u in ups])
    fails=[r for r in R if not r['success']]
    out['fail_max_label_up_hist']=np.histogram([max(r['diag']['up']) for r in fails],bins=[-1.01,-.5,0,.5,.9,.98,1.01])[0].tolist() if fails else []
    out['fail_reached_vertical(up>0)']=sum(max(r['diag']['up'])>0 for r in fails)
    out['fail_reached_upright(up>0.98)_but_lost']=sum(max(r['diag']['up'])>.98 for r in fails)
    out['median_chunk_pass_vertical_success']=float(np.median([first(r['diag']['up'],0) for r in R if r['success'] and first(r['diag']['up'],0) is not None])) if s.any() else None
    fv=[first(r['diag']['up'],0) for r in fails if first(r['diag']['up'],0) is not None]
    out['median_chunk_pass_vertical_fail']=float(np.median(fv)) if fv else None
    def fc(r,arm):
        f=r['diag'][f'f{arm}'];return first(f,1.)
    for arm in range(2):
        out[f'median_first_contact_chunk_arm{arm}_success']=float(np.median([x for x in (fc(r,arm) for r in R if r['success']) if x is not None] or [np.nan]))
        out[f'median_first_contact_chunk_arm{arm}_fail']=float(np.median([x for x in (fc(r,arm) for r in fails) if x is not None] or [np.nan]))
        out[f'fail_no_contact_arm{arm}']=sum(fc(r,arm) is None for r in fails)
    return out
char={}
# by init-pose bins over all grid episodes
def rate(R):return [sum(bool(r['success']) for r in R),len(R)]
dx=np.array([r['dx'] for r in A])*1000;dy=np.array([r['dy'] for r in A])*1000;yw=np.array([r['yaw_deg'] for r in A]);sc=np.array([bool(r['success']) for r in A])
def binned(v,edges,name):
    res=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(v>=lo)&(v<hi);res.append(dict(bin=f'[{lo},{hi})',n=int(m.sum()),success_rate=round(float(sc[m].mean()),3) if m.sum() else None))
    char[name]=res
if len(A):
    binned(dx,[-80,-40,-20,-10,-5,0,5,10,20,40,80.01],'by_dx_mm');binned(dy,[-80,-40,-20,-10,-5,0,5,10,20,40,80.01],'by_dy_mm')
    binned(yw,[-45,-30,-20,-10,-5,-2,0,2,5,10,20,30,45.01],'by_yaw_deg')
    r2=np.hypot(dx,dy);binned(r2,[0,5,10,20,40,60,80,120],'by_radial_offset_mm')
    char['grid_all']=stats(A)
    # simple logistic-free attribution: success vs |dx|,|dy|,|yaw| within cells that have both outcomes
    char['corr_success_vs']={k:round(float(np.corrcoef(sc.astype(float),v)[0,1]),3) for k,v in dict(abs_dx=abs(dx),abs_dy=abs(dy),abs_yaw=abs(yw),dx=dx,dy=dy,yaw=yw).items()} if 0<sc.mean()<1 else None
    # yaw sign x dy sign quadrant for wider cells (yaw half>=10 or pos>=20)
    W=[(r,d1,d2,y1) for r,d1,d2,y1 in zip(A,dx,dy,yw) if r['yaw_half_deg']>=10 or r['pos_mm']>=20]
    q=collections.defaultdict(lambda:[0,0])
    for r,d1,d2,y1 in W:key=f"dy{'+' if d2>=0 else '-'}_yaw{'+' if y1>=0 else '-'}";q[key][0]+=bool(r['success']);q[key][1]+=1
    char['wide_cells_quadrant_dy_sign_x_yaw_sign']={k:dict(success=v[0],n=v[1],rate=round(v[0]/v[1],3)) for k,v in sorted(q.items())}
    q=collections.defaultdict(lambda:[0,0])
    for r,d1,d2,y1 in W:key=f"dx{'+' if d1>=0 else '-'}";q[key][0]+=bool(r['success']);q[key][1]+=1
    char['wide_cells_dx_sign']={k:dict(success=v[0],n=v[1],rate=round(v[0]/v[1],3)) for k,v in sorted(q.items())}
(OUT/'failure_characterization.json').write_text(json.dumps(char,indent=1))
done=sum(t['episodes'] for t in table);print('episodes',done,'/',len(table)*NEP)
for t in table:print(f"{t['cell']:28s} {t['successes']:3d}/{t['episodes']:3d} wf={t['wrong_face']} drop={t['drop']} bh={t['below_height']} inc={t['incomplete']} {t['other_fail'] or ''} CI={t['wilson95_lo']}-{t['wilson95_hi']}")
