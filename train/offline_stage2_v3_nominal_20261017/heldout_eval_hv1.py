# [hv1] Held-out eval (seeds 700000+, reserved held-out block): 840-ep demo-cell mix (same strata/cell proportions as the 420 gate,
# mixcells.mix_episodes) + nominal-100 (p5_y2, init_pose as gate), same worker + 'det' (deployment DDIM10 eta=1) sampler as the gate.
import os,sys,json,time,subprocess,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import numpy as np, mixcells
from wcommon_fix import init_pose
OUT=Path(sys.argv[1]);GPUS=[int(x) for x in sys.argv[2].split(',')];PER=int(sys.argv[3]);SH=8;PY=sys.executable
R='/data02/kemove/rl100-sim-repro/runs'
POL={'r2_il4':f'{R}/offline_stage2_v3e88_ic7s6_20261015_0031/round2_s9/il_v4_policy.pt',
     'r1_m150':f'{R}/offline_stage2_v3e88_ic7s6_20261015_0031/promoted_round1.pt',
     'e88':'/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/runs/p3_bc_v3_novib_seed42/development_v2/candidate_0088.pt',
     'fix2_best':f'{R}/offline_stage2_v3e88_ic_fix2_20261014_1424/promoted_round1.pt'}
MIX0,NMIX,NOM0,NNOM=700000,840,701000,100
def log(m):
    s=time.strftime('[%F %T CST] ')+m;print(s,flush=True);open(OUT/'heldout.log','a').write(s+'\n')
wide=mixcells.mix_episodes(MIX0,NMIX)
nom=[]
for s in range(NOM0,NOM0+NNOM):
    dx,dy,yw=init_pose(s,.005,2.);nom.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal',stratum='nominal100',pos_half_mm=5,yaw_half_deg=2.))
jobs=[dict(id=f'wide_{i:04d}',kind='eval',mode='det',episodes=wide[i:i+SH]) for i in range(0,len(wide),SH)]+[dict(id=f'nominal_{i:04d}',kind='eval',mode='det',episodes=nom[i:i+SH]) for i in range(0,len(nom),SH)]
def free():
    o=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    return {int(x):int(y) for x,y in (l.split(',') for l in o.strip().splitlines())}
env0=dict(os.environ,MUJOCO_GL='egl',OMP_NUM_THREADS='4')
for attempt in range(3):
    procs=[]
    for n,p in POL.items():
        d=OUT/n;d.mkdir(exist_ok=True);jf=d/'jobs.json'
        spec=dict(policy=p,out=str(d),jobs=jobs)
        if not jf.exists():jf.write_text(json.dumps(spec))
        for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
            if not (d/'done'/c.name).exists():c.unlink()
        todo=[j for j in jobs if not (d/'done'/j['id']).exists()]
        if not todo:continue
        fr=free();ok=[g for g in GPUS if fr.get(g,0)>=12000]
        if not ok:log('no GPU with >=12GB free; wait');time.sleep(120);continue
        (d/'logs').mkdir(exist_ok=True)
        for i,g in enumerate([g for g in ok for _ in range(PER)]):
            procs.append(subprocess.Popen([PY,'-u',str(HERE/'batch_worker_fix.py'),'--jobs',str(jf),'--wid',str(attempt*1000+i)],cwd=str(HERE),
                env=dict(env0,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g)),stdout=open(d/'logs'/f'w_a{attempt}_{i}.log','a'),stderr=subprocess.STDOUT))
        log(f'attempt {attempt+1} {n}: {len(todo)} shards, workers on gpus {ok}x{PER}')
    if not procs:break
    (OUT/'worker_pids.json').write_text(json.dumps([q.pid for q in procs]))
    while any(q.poll() is None for q in procs):
        time.sleep(30);log('progress '+' '.join(f"{n}:{len(list((OUT/n/'done').glob('*')))}/{len(jobs)}" for n in POL))
# ---- analysis ----
def wil(k,n,z=1.96):
    if n==0:return [0,0]
    p=k/n;dn=1+z*z/n;c=(p+z*z/(2*n))/dn;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/dn;return [round(c-h,4),round(c+h,4)]
def binom_p(b,c):  # two-sided exact sign test on discordant pairs
    n=b+c;k=min(b,c)
    if n==0:return 1.
    return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n)
res={}
for n in POL:
    d=OUT/n;assert len(list((d/'done').glob('*')))==len(jobs),f'{n} incomplete'
    rows=[json.loads(l) for j in jobs for l in open(d/'rows'/f"{j['id']}.jsonl")]
    out={}
    for key,sel in [('mix840',lambda r:r.get('stratum')!='nominal100'),('nominal',lambda r:r.get('stratum')=='nominal'),('expanded',lambda r:r.get('stratum')=='expanded'),('wide',lambda r:r.get('stratum')=='wide'),('nominal100',lambda r:r.get('stratum')=='nominal100')]:
        rr=[r for r in rows if sel(r)];k=sum(bool(r['success']) for r in rr);out[key]=dict(k=k,n=len(rr),rate=round(k/max(1,len(rr)),4),wilson95=wil(k,len(rr)))
    out['outcomes']={f"{r['stratum']}:{r['seed']}":bool(r['success']) for r in rows}
    out['fails']={};[out['fails'].__setitem__(r['reason'],out['fails'].get(r['reason'],0)+1) for r in rows if not r['success']]
    res[n]=out
cmp={}
for ref in ['e88','fix2_best']:
    for n in POL:
        if n==ref:continue
        for key in ['mix840','nominal100']:
            A=res[n]['outcomes'];B=res[ref]['outcomes'];ks=[k for k in A if (k.startswith('nominal100') if key=='nominal100' else not k.startswith('nominal100'))]
            b=sum(A[k] and not B[k] for k in ks);c=sum(B[k] and not A[k] for k in ks);dl=(b-c)/len(ks)
            se=math.sqrt(max(0,(b+c)/len(ks)-dl*dl)/len(ks))
            cmp[f'{n}_vs_{ref}_{key}']=dict(delta=b-c,delta_rate=round(dl,4),ci95=[round(dl-1.96*se,4),round(dl+1.96*se,4)],wins=b,losses=c,sign_test_p_two_sided=round(binom_p(b,c),4))
(OUT/'results.json').write_text(json.dumps(dict(seeds=dict(mix=[MIX0,MIX0+NMIX-1],nominal100=[NOM0,NOM0+NNOM-1]),policies=POL,results={n:{k:v for k,v in r.items() if k!='outcomes'} for n,r in res.items()},paired=cmp),indent=1))
L=['| policy | mix840 | nominal | expanded | wide | nominal-100 |','|---|---|---|---|---|---|']
for n,r in res.items():L.append(f'| {n} | '+' | '.join(f"{r[k]['k']}/{r[k]['n']} ({100*r[k]['rate']:.1f}%, {100*r[k]['wilson95'][0]:.1f}-{100*r[k]['wilson95'][1]:.1f})" for k in ['mix840','nominal','expanded','wide','nominal100'])+' |')
L+=['','| comparison | delta | delta rate (95% CI) | wins/losses | sign p (2-sided) |','|---|---|---|---|---|']
for k,v in cmp.items():L.append(f"| {k} | {v['delta']:+d} | {100*v['delta_rate']:+.1f}% ({100*v['ci95'][0]:+.1f}, {100*v['ci95'][1]:+.1f}) | {v['wins']}/{v['losses']} | {v['sign_test_p_two_sided']} |")
(OUT/'RESULTS.md').write_text('\n'.join(L)+'\n');log('DONE\n'+'\n'.join(L))
