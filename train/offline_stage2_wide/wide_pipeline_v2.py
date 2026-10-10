"""Stage-2-wide v2: continue wide-range offline RL FROM the round-2 candidate with nominal-range protection.
Round r (policy = current incumbent):
  collect N eps (default 1500): (1-nominal_frac) wide-range (same wide sampler as v1) + nominal_frac nominal-range
    (init_pose(seed, 5mm, 2deg) = Stage-1 eval distribution), 2/3 deployment sampler + 1/3 small-sigma, fresh seeds
    650000+(r-1)*5000 ...
  -> IQL on ALL data (demos, old stage-2 round-1, v1 wide rounds 1-2, all v2 rounds)
  -> PPO v2 (wide + nominal advantage states; KL anchor to STAGE-1 on demo + nominal states)
  -> real closed-loop gate per candidate on the SAME paired seeds as before:
       6 wide cells x 50 (620000-620049) + nominal 100 (880000-880099).
PROMOTE iff  wide_delta_vs_incumbent >= MIN_DELTA (5)  AND  exact one-sided McNemar/sign p(better) < ALPHA (.05)
        AND  nominal vs Stage-1 paired one-sided p(worse) >= ALPHA (no significant nominal drop).
Best passing = max wide (tie: max nominal). Starting incumbent = round-2 candidate (wide 229/300, reused v1 gate result).
STOP when a round promotes nothing, or the promoted wide gain < MIN_DELTA, or after --rounds (5).
Final: best policy (last promoted; else the starting incumbent, flagged) vs Stage-1 (and vs start) on held-out paired
seeds 600000-600099 for the 6 gate cells + nominal (5mm/2deg); optional 36-cell grid sweep. 910000-910099 untouched."""
from wcommon import *
HERE=Path(__file__).resolve().parent
import argparse,subprocess,shutil,traceback,csv
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=5)
ap.add_argument('--gpus',default='7');ap.add_argument('--per-gpu',type=int,default=2);ap.add_argument('--train-gpu',type=int,default=-1)
ap.add_argument('--collect-n',type=int,default=1500);ap.add_argument('--nominal-frac',type=float,default=.25);ap.add_argument('--sto-frac',type=float,default=1/3)
ap.add_argument('--seed-base',type=int,default=650000)
ap.add_argument('--ppo-steps',type=int,default=600);ap.add_argument('--iql-steps',type=int,default=100000);ap.add_argument('--lr',type=float,default=2e-6)
ap.add_argument('--anchor',type=float,default=3.);ap.add_argument('--nom-batch',type=int,default=32)
ap.add_argument('--min-delta',type=int,default=5);ap.add_argument('--alpha',type=float,default=.05)
ap.add_argument('--start-policy',default=str(ROOT/'runs/offline_stage2_wide/round2/ppo/cand_step0150.pt'))
ap.add_argument('--v1-dir',default=str(ROOT/'runs/offline_stage2_wide'))
ap.add_argument('--final-n',type=int,default=100);ap.add_argument('--final-sweep',type=int,default=1)
a=ap.parse_args();SM=a.smoke
GPUS=[int(x) for x in a.gpus.split(',')];PY=sys.executable;HERE_=str(HERE);V1=Path(a.v1_dir);START=Path(a.start_policy)
NCOL=a.collect_n if not SM else 12;SEED_BASE=a.seed_base if not SM else 698000
SH=25 if not SM else 4;GN=GATE_N if not SM else 2;NN=NOMINAL['n'] if not SM else 2;FN=a.final_n if not SM else 2
assert OUT.resolve()!=V1.resolve(),'refusing to write into the v1 run dir'
OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'pipeline.log';state_path=OUT/'pipeline_state.json'
S=json.loads(state_path.read_text()) if state_path.exists() else dict(started=cst(),rounds={},current_policy=str(START),start_policy=str(START),promoted=[],durations={})
S.update(pid=os.getpid(),gpus=GPUS,per_gpu=a.per_gpu,smoke=SM,planned_rounds=a.rounds,args=vars(a),std_max=STD_MAX)
def save_state():write_json(state_path,S)
def log(msg):
    line=f'[{cst()}] {msg}';print(line,flush=True)
    with open(LOG,'a') as f:f.write(line+'\n')
def summ_gate(g):
    return None if not g else dict(wide=f"{g['wide_successes']}/{g['wide_episodes']}",cells=g['cells'],nominal=f"{g['nominal_successes']}/{g['nominal_episodes']}")
def status(phase,**kw):
    write_json(OUT/'status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=cst(),smoke=SM,gpus=GPUS,per_gpu=a.per_gpu,current_policy=S['current_policy'],
        promoted=S['promoted'],stage1_gate=summ_gate(S.get('stage1_gate')),incumbent_gate=summ_gate(S.get('ref_gate')),
        rounds_done=[k for k,v in S['rounds'].items() if v.get('done')],planned_rounds=a.rounds,**kw))
def wilson(k,n):
    if n==0:return None
    z=1.95996398454;ph=k/n;c=(ph+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);return [round(c-h,4),round(c+h,4)]
def env_for(g):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4',S2_OUT=str(OUT))
def train_gpu():
    if a.train_gpu>=0:return a.train_gpu
    out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    used={int(x):int(y) for x,y in (l.split(',') for l in out.strip().splitlines())};return min(GPUS,key=lambda g:used.get(g,1e9))
def run(cmd,g,logf,phase,poll=None):
    log(f'RUN gpu{g}: {" ".join(map(str,cmd))}')
    pr=subprocess.Popen([PY,'-u',*map(str,cmd)],cwd=HERE_,env=env_for(g),stdout=open(logf,'a'),stderr=subprocess.STDOUT)
    while pr.poll() is None:
        status(phase,child_pid=pr.pid,child_log=str(logf),child_status=(json.loads(Path(poll).read_text()) if poll and Path(poll).exists() else None));time.sleep(30 if not SM else 3)
    if pr.returncode!=0:raise RuntimeError(f'{phase} failed rc={pr.returncode}, see {logf}')
def timed(key,fn):
    t=time.time();r=fn();S['durations'][key]=round(time.time()-t,1);save_state();return r
def run_jobs(policy,d,jobs,phase):
    d.mkdir(parents=True,exist_ok=True);jf=d/'jobs.json';spec=dict(policy=str(policy),out=str(d),jobs=jobs)
    if jf.exists():assert json.loads(jf.read_text())==spec,f'job spec changed in {d}'
    else:write_json(jf,spec)
    for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
        if not (d/'done'/c.name).exists():c.unlink()
    todo=[j for j in jobs if not (d/'done'/j['id']).exists()]
    if todo:
        (d/'logs').mkdir(exist_ok=True);slots=[g for g in GPUS for _ in range(a.per_gpu)][:len(todo)];procs=[];t0=time.time();n0=len(jobs)-len(todo)
        log(f'{phase}: {len(todo)} jobs on {len(slots)} workers (gpus {GPUS})')
        for i,g in enumerate(slots):
            procs.append(subprocess.Popen([PY,'-u',HERE/'batch_worker.py','--jobs',jf,'--wid',str(i)],cwd=HERE_,env=env_for(g),stdout=open(d/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT))
        S.setdefault('worker_pids',{})[phase]=[q.pid for q in procs];save_state()
        while any(q.poll() is None for q in procs):
            nd=len(list((d/'done').glob('*')));el=time.time()-t0;rate=(nd-n0)/el if el>0 else 0
            status(phase,jobs_done=nd,jobs_total=len(jobs),workers=sum(q.poll() is None for q in procs),worker_pids=[q.pid for q in procs],
                   eta_min=round((len(jobs)-nd)/rate/60,1) if rate>0 else None,dir=str(d));time.sleep(20 if not SM else 3)
        bad=[q.returncode for q in procs if q.returncode!=0]
        if bad:raise RuntimeError(f'{phase}: worker failed rc={bad}, see {d}/logs')
    nd=len(list((d/'done').glob('*')));assert nd==len(jobs),f'{phase}: {nd}/{len(jobs)} jobs done'
    rows=[]
    for j in jobs:rows+=[json.loads(l) for l in open(d/'rows'/f"{j['id']}.jsonl")]
    return rows
def shards(eps,kind,mode,prefix):
    return [dict(id=f'{prefix}_{i:04d}',kind=kind,mode=mode,episodes=eps[i:i+SH]) for i in range(0,len(eps),SH)]
def gate_episodes(seed0,n,cells):
    E=[]
    for pos,yaw in cells:
        for s in range(seed0,seed0+n):
            dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell=f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
    return E
def summarize(rows,cells_order):
    res=dict(cells={},fails={},outcomes={})
    for r in rows:
        c=r['cell'];x=res['cells'].setdefault(c,[0,0]);x[0]+=bool(r['success']);x[1]+=1;res['outcomes'][f"{c}:{r['seed']}"]=bool(r['success'])
        if not r['success']:f=res['fails'].setdefault(c,{});f[r['reason']]=f.get(r['reason'],0)+1
    res['cells']={c:res['cells'][c] for c in cells_order if c in res['cells']};return res
def gate(policy,name):
    d=OUT/'gate'/name;rp=d/'result.json';sha=sha256(policy)
    if rp.exists() and json.loads(rp.read_text())['sha256']==sha:return json.loads(rp.read_text())
    wide=gate_episodes(GATE_SEED0,GN,GATE_CELLS);nom=gate_episodes(NOMINAL['seed0'],NN,[(5,2.)])
    for e in nom:e['cell']='nominal'
    rows=run_jobs(policy,d,shards(wide,'eval','det','wide')+shards(nom,'eval','det','nominal'),f'gate:{name}')
    sm=summarize(rows,[f'p{p}_y{y}' for p,y in GATE_CELLS]+['nominal'])
    ws=sum(v[0] for c,v in sm['cells'].items() if c!='nominal');wn=sum(v[1] for c,v in sm['cells'].items() if c!='nominal')
    res=dict(name=name,policy=str(policy),sha256=sha,finished=cst(),wide_successes=ws,wide_episodes=wn,wide_wilson95=wilson(ws,wn),
        nominal_successes=sm['cells']['nominal'][0],nominal_episodes=sm['cells']['nominal'][1],cells=sm['cells'],fails=sm['fails'],outcomes=sm['outcomes'])
    write_json(rp,res);log(f"GATE {name}: wide {ws}/{wn} {sm['cells']} nominal {res['nominal_successes']}/{res['nominal_episodes']} fails {sm['fails']}");return res
def binom_tail(k,n):return sum(math.comb(n,i) for i in range(k,n+1))/2**n if n else 1.
def paired(c,r,nominal):
    """Exact paired (McNemar / sign) test on discordant pairs over identical seed keys."""
    keys=[k for k in r['outcomes'] if k.startswith('nominal')==nominal];assert all(k in c['outcomes'] for k in keys),'unpaired keys'
    g=sum(c['outcomes'][k] and not r['outcomes'][k] for k in keys);l=sum(r['outcomes'][k] and not c['outcomes'][k] for k in keys)
    sc=sum(c['outcomes'][k] for k in keys);sr=sum(r['outcomes'][k] for k in keys)
    return dict(cand=sc,ref=sr,n=len(keys),delta=sc-sr,gained=g,lost=l,p_better_one_sided=binom_tail(g,g+l),p_worse_one_sided=binom_tail(l,g+l))
def percell(c,r):
    return {cell:[c['cells'][cell][0],r['cells'][cell][0],c['cells'][cell][0]-r['cells'][cell][0]] for cell in r['cells'] if cell in c['cells']}
def collect(r,policy,n):
    base=SEED_BASE+(r-1)*5000;check_seeds(base,base+n-1);nn_=int(round(n*a.nominal_frac));nw=n-nn_;E={'det':[],'sto':[]}
    for s in range(base,base+n):
        rng=np.random.default_rng(s+5000000)
        if s<base+nw:   # wide sampler (identical to v1)
            pos=int(rng.choice([20,40,60,80],p=[.1,.3,.3,.3]));yaw=float(rng.choice([10.,20.,30.,45.]))
            dx,dy=rng.uniform(-pos,pos,2)/1000;yw=float(rng.uniform(-yaw,yaw))
            if rng.random()<.35:dy=abs(dy)
            ep=dict(seed=s,dx=float(dx),dy=float(dy),yaw_deg=yw,cell=f'p{pos}_y{int(yaw)}',pos_half_mm=pos,yaw_half_deg=yaw,wide=True,nominal=False,collect_round=r)
        else:           # nominal sampler = Stage-1 eval distribution (evaluate_bc / gate nominal: +-5mm, +-2deg)
            dx,dy,yw=init_pose(s,NOMINAL['pos_mm']/1000,NOMINAL['yaw'])
            ep=dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal',pos_half_mm=NOMINAL['pos_mm'],yaw_half_deg=NOMINAL['yaw'],wide=False,nominal=True,collect_round=r)
        mode='sto' if rng.random()<a.sto_frac else 'det';E[mode].append(ep)
    d=OUT/f'round{r}'/'collect'
    rows=run_jobs(policy,d,shards(E['det'],'collect','det','det')+shards(E['sto'],'collect','sto','sto'),f'round{r}:collect')
    out={}
    for key,f in [('by_range',lambda x:'nominal' if x.get('nominal') else 'wide'),('by_sampler',lambda x:x['sampler']),('by_pos_mm',lambda x:x['pos_half_mm']),('by_yaw',lambda x:x['yaw_half_deg'])]:
        t={}
        for x in rows:v=t.setdefault(str(f(x)),[0,0,{}]);v[0]+=bool(x['success']);v[1]+=1;v[2][x['reason']]=v[2].get(x['reason'],0)+(not x['success'])
        out[key]={k:dict(successes=v[0],episodes=v[1],fails={kk:vv for kk,vv in v[2].items() if vv}) for k,v in sorted(t.items())}
    out['total']=dict(successes=sum(bool(x['success']) for x in rows),episodes=len(rows),seeds=[base,base+n-1],nominal_seeds=[base+nw,base+n-1]);return out
def sources(r):
    src=[['demo',str(DEMO_DIR)],['old_stage2_round1',str(OLD/'round1'/'rollouts')],
         ['wide_round1',str(V1/'round1'/'collect'/'eps')],['wide_round2',str(V1/'round2'/'collect'/'eps')]]
    return src+[[f'v2_round{k}',str(OUT/f'round{k}'/'collect'/'eps')] for k in range(1,r+1)]

def fmt_p(p):return f'{p:.3g}'
def report():
    L=['# Stage-2-wide v2: continue from round-2 candidate with nominal protection\n',f'_Auto-generated {cst()}; state `{state_path}`, log `{LOG}`._\n',
       '## Setup',f'- Start policy (starting incumbent): `{S["start_policy"]}`; Stage-1: `{STAGE1}`',
       f'- Per round: collect {NCOL} eps ({a.nominal_frac:.0%} nominal +-5mm/+-2deg, rest wide; {a.sto_frac:.2f} small-sigma sampler), seeds {SEED_BASE}+5000*(r-1); IQL {a.iql_steps} steps on all data; PPO lr {a.lr}, <= {a.ppo_steps} steps, advantage batch 96 wide + {a.nom_batch} nominal states (per-group normalised), KL anchor to Stage-1 on 32 demo + 32 nominal-rollout states, weight {a.anchor} (v1: 1.0, demo only, ref = stage input).',
       f'- Gate (paired, same seeds as v1): 6 wide cells x 50 (620000-620049) + nominal 100 (880000-880099). PROMOTE iff wide delta vs incumbent >= +{a.min_delta} AND exact one-sided McNemar p(better) < {a.alpha} AND nominal vs Stage-1 one-sided p(worse) >= {a.alpha}. '
       f'(With Stage-1 nominal 100/100, the nominal condition means at most 4 nominal losses vs Stage-1.)',
       f'- Stop: no promotion in a round, or promoted gain < +{a.min_delta}, or {a.rounds} rounds.',
       f'- Current policy: `{S["current_policy"]}`; promoted: {S["promoted"]}; stop: {S.get("stop_reason")}\n']
    if S.get('stage1_gate'):
        L.append('## Reference gate results');L.append('| policy | wide | '+' | '.join(c for c in S['stage1_gate']['cells'])+' |');L.append('|---|---|'+'---|'*len(S['stage1_gate']['cells']))
        for nm,g in [('Stage-1',S['stage1_gate']),('start (v1 r2_m150)',S.get('start_gate'))]:
            if g:L.append(f'| {nm} | {g["wide_successes"]}/{g["wide_episodes"]} | '+' | '.join(f'{v[0]}/{v[1]}' for v in g['cells'].values())+' |')
    for r,v in sorted(S['rounds'].items(),key=lambda x:int(x[0])):
        L.append(f'\n## Round {r}\n- policy_in: `{v.get("policy_in")}`; incumbent wide at start: {v.get("incumbent_wide")}; started {v.get("started")}; finished {v.get("finished")}')
        if 'collect' in v:L.append(f'- collect: total {v["collect"]["total"]}; by_range {json.dumps({k:(x["successes"],x["episodes"],x["fails"]) for k,x in v["collect"]["by_range"].items()})}')
        if 'iql' in v:L.append('- IQL: '+json.dumps({k:v['iql'].get('final',{}).get(k) for k in ['v_auc_all_wide','v_auc_t0_wide','v_auc_t20_wide','v_auc_all_rollout','heldout_td_mse']}))
        if 'ppo' in v:L.append(f'- PPO: steps {v["ppo"].get("steps_run")}, accepted OPE evals {v["ppo"].get("accepted")}, stop {v["ppo"].get("stop_reason")}, candidates {[c["milestone"] for c in v["ppo"].get("candidates",[])]}, input dev {v["ppo"].get("input_deviation")}')
        if v.get('gate'):
            L.append('\n| cand | wide | nominal | vs incumbent (d, +/-, p) | vs Stage-1 wide (d, p) | nominal vs Stage-1 (lost/gained, p_worse) | pass |');L.append('|---|---|---|---|---|---|---|')
            for nm,gg in v['gate'].items():
                ci,cs,cn=gg['vs_incumbent'],gg['vs_stage1'],gg['nominal_vs_stage1']
                L.append(f'| {nm} | {gg["wide"]}/{gg["wide_episodes"]} | {gg["nominal"]}/100 | {ci["delta"]:+d}, +{ci["gained"]}/-{ci["lost"]}, {fmt_p(ci["p_better_one_sided"])} | {cs["delta"]:+d}, {fmt_p(cs["p_better_one_sided"])} | {cn["lost"]}/{cn["gained"]}, {fmt_p(cn["p_worse_one_sided"])} | {gg["passes"]} ({"; ".join(gg["why"]) or "ok"}) |')
        L.append(f'- promoted: {v.get("promoted")}')
    if S.get('final'):L.append('\n## Final evaluation\n```json\n'+json.dumps(S['final'],indent=1,ensure_ascii=False,default=str)+'\n```')
    (OUT/'REPORT.md').write_text('\n'.join(L)+'\n')

def final_sweep(best):
    d=OUT/'final_sweep';grid=[f'p{p}_y{y}' for p in [5,10,20,40,60,80] for y in [2,5,10,20,30,45]]
    env=dict(os.environ,SWEEP_CKPT=str(best),SWEEP_OUT=str(d),SWEEP_CELLS=','.join(grid))
    if SM:env.update(SWEEP_NEP='2',SWEEP_SHARD='2',SWEEP_CELLS='p5_y2,p80_y45')
    sw=ROOT/'eval/generalization_sweep'
    njobs=len(json.loads(subprocess.run([PY,'-c','import sweep_eval,json;print(json.dumps(sweep_eval.jobs()))'],cwd=sw,env=env,capture_output=True,text=True,check=True).stdout))
    if not (d/'done').exists() or len(list((d/'done').glob('*')))<njobs:
        subprocess.run([PY,'sweep_eval.py','launch','--gpus',','.join(map(str,GPUS)),'--per-gpu',str(a.per_gpu)],cwd=sw,env=env,check=True)
        pids=json.loads((d/'workers.json').read_text())['pids'];log(f'final sweep launched pids {pids}');t0=time.time()
        while any(Path(f'/proc/{q}').exists() for q in pids):
            nd=len(list((d/'done').glob('*')));status('final:sweep',jobs_done=nd,jobs_total=njobs,worker_pids=pids,eta_min=round((njobs-nd)*(time.time()-t0)/max(nd,1)/60,1) if nd else None);time.sleep(30 if not SM else 3)
    nd=len(list((d/'done').glob('*')));assert nd==njobs,f'final sweep {nd}/{njobs}'
    base=ROOT/'runs/generalization_sweep/cells';tot=dict(stage1=0,best=0,gained=0,lost=0,episodes=0);cells={}
    for c in (grid if not SM else ['p5_y2','p80_y45']):
        B={json.loads(l)['seed']:bool(json.loads(l)['success']) for l in open(d/'cells'/f'{c}.jsonl')}
        A={json.loads(l)['seed']:bool(json.loads(l)['success']) for l in open(base/f'{c}.jsonl')} if (base/f'{c}.jsonl').exists() else {}
        ks=[k for k in B if k in A];g=sum(B[k] and not A[k] for k in ks);l=sum(A[k] and not B[k] for k in ks)
        cells[c]=dict(stage1=sum(A[k] for k in ks),best=sum(B[k] for k in ks),n=len(ks),gained=g,lost=l)
        for k in ['stage1','best','gained','lost']:tot[k]+=cells[c][k]
        tot['episodes']+=len(ks)
    tot['p_better_one_sided']=binom_tail(tot['gained'],tot['gained']+tot['lost']);tot['p_worse_one_sided']=binom_tail(tot['lost'],tot['gained']+tot['lost'])
    return dict(dir=str(d),seeds='600000-600099',total=tot,cells=cells)

def heldout_eval(policy,name):
    """6 gate cells + nominal (5mm/2deg) on held-out paired seeds 600000..600000+FN-1 (never used for selection)."""
    cells=GATE_CELLS+[(5,2.)];eps=gate_episodes(600000,FN,cells)
    for e in eps:
        if e['pos_half_mm']==5:e['cell']='nominal'
    rows=run_jobs(policy,OUT/'final_heldout'/name,shards(eps,'eval','det','h'),f'final:heldout:{name}')
    sm=summarize(rows,[f'p{p}_y{y}' for p,y in GATE_CELLS]+['nominal']);ws=sum(v[0] for c,v in sm['cells'].items() if c!='nominal');wn=sum(v[1] for c,v in sm['cells'].items() if c!='nominal')
    return dict(policy=str(policy),wide_successes=ws,wide_episodes=wn,nominal_successes=sm['cells']['nominal'][0],nominal_episodes=sm['cells']['nominal'][1],cells=sm['cells'],fails=sm['fails'],outcomes=sm['outcomes'])

def preseed(name,src):
    d=OUT/'gate'/name
    if not (d/'result.json').exists() and Path(src).exists():
        d.mkdir(parents=True,exist_ok=True);shutil.copy2(src,d/'result.json');log(f'gate {name}: reusing v1 result {src} (same paired seeds; sha checked)')

def main():
    log(f'pipeline v2 start pid={os.getpid()} smoke={SM} gpus={GPUS} per_gpu={a.per_gpu} std_max={STD_MAX} start={START}');save_state()
    if not SM:preseed('stage1',V1/'gate/stage1/result.json');preseed('start',V1/'gate/r2_m150/result.json')
    if 'stage1_gate' not in S:S['stage1_gate']=timed('stage1_gate',lambda:gate(STAGE1,'stage1'));save_state()
    if 'start_gate' not in S:
        S['start_gate']=timed('start_gate',lambda:gate(START,'start'));S['ref_gate']=dict(S['start_gate'],policy=str(START));save_state()
    report()
    for r in range(1,a.rounds+1):
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):
            if not R.get('promoted'):break
            continue
        if S.get('stop_reason'):break
        pol=S['current_policy'];ref=S['ref_gate'];R['policy_in']=pol;R['incumbent_wide']=ref['wide_successes'];R.setdefault('started',cst());save_state()
        if 'collect' not in R:
            R['collect']=timed(f'r{r}_collect',lambda:collect(r,pol,NCOL));save_state()
            log(f'round{r} collect {json.dumps(R["collect"]["total"])} by_range {json.dumps({k:(v["successes"],v["episodes"]) for k,v in R["collect"]["by_range"].items()})}');report()
        sp=OUT/f'round{r}'/'sources.json';write_json(sp,sources(r));idir=OUT/f'round{r}'/'iql'
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql_wide.py','--sources',sp,'--out',idir,'--steps',a.iql_steps if not SM else 300,'--eval-every',5000 if not SM else 100],
                train_gpu(),OUT/f'round{r}'/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state();report()
        pdir=OUT/f'round{r}'/'ppo'
        if not (pdir/'report.json').exists():
            extra=['--steps',a.ppo_steps,'--lr',a.lr,'--anchor',a.anchor,'--nom-batch',a.nom_batch] if not SM else ['--steps',20,'--ope-every',5,'--batch',16,'--nom-batch',8,'--demo-batch',8,'--nom-anchor-batch',8,'--ope-states',64,'--milestones','10,20','--accept-any']
            timed(f'r{r}_ppo',lambda:run([HERE/'ppo_wide_v2.py','--policy',pol,'--anchor-ref',STAGE1,'--critic',idir/'critic.pt','--sources',sp,'--out',pdir,*extra],
                train_gpu(),OUT/f'round{r}'/'ppo.log',f'round{r}:ppo',poll=pdir/'status.json'))
        R['ppo']={k:v for k,v in json.loads((pdir/'report.json').read_text()).items() if k!='ope_history_tail'};save_state();report()
        R.setdefault('gate',{})
        for c in R['ppo']['candidates']:
            nm=f"r{r}_m{c['milestone']}"
            if nm in R['gate']:continue
            res=timed(f'gate_{nm}',lambda:gate(c['path'],nm))
            ci=paired(res,ref,False);cs=paired(res,S['stage1_gate'],False);cn=paired(res,S['stage1_gate'],True);cni=paired(res,ref,True)
            why=[]
            if ci['delta']<a.min_delta:why.append(f'wide delta {ci["delta"]:+d} < +{a.min_delta}')
            if ci['p_better_one_sided']>=a.alpha:why.append(f'wide p {ci["p_better_one_sided"]:.3g} >= {a.alpha}')
            if cn['p_worse_one_sided']<a.alpha:why.append(f'nominal significantly worse than Stage-1 (lost {cn["lost"]}, gained {cn["gained"]}, p {cn["p_worse_one_sided"]:.3g})')
            R['gate'][nm]=dict(path=c['path'],wide=res['wide_successes'],wide_episodes=res['wide_episodes'],cells=res['cells'],fails=res['fails'],nominal=res['nominal_successes'],
                vs_incumbent=ci,vs_stage1=cs,nominal_vs_stage1=cn,nominal_vs_incumbent=cni,cells_vs_stage1=percell(res,S['stage1_gate']),cells_vs_incumbent=percell(res,ref),
                passes=bool(not why or SM),why=why,ppo_candidate={k:v for k,v in c.items() if k not in ('path','ope')})
            log(f'round{r} gate {nm}: wide {res["wide_successes"]} (inc {ref["wide_successes"]}, d {ci["delta"]:+d}, p {ci["p_better_one_sided"]:.3g}) nominal {res["nominal_successes"]} (vs S1 lost {cn["lost"]} p_worse {cn["p_worse_one_sided"]:.3g}) -> {"PASS" if not why else "FAIL: "+"; ".join(why)}')
            save_state();report()
        passing=[(v['wide'],v['nominal'],k) for k,v in R['gate'].items() if v['passes']]
        if passing:
            _,_,best=max(passing);src=R['gate'][best]['path'];dst=OUT/f'promoted_round{r}.pt';shutil.copy2(src,dst);gain=R['gate'][best]['vs_incumbent']['delta']
            S['current_policy']=str(dst);S['promoted'].append(str(dst));S['ref_gate']=dict(json.loads((OUT/'gate'/best/'result.json').read_text()),policy=str(dst))
            R['promoted']=dict(candidate=best,path=str(dst),wide_gain_vs_incumbent=gain);log(f'round{r} PROMOTED {best} -> {dst} (wide +{gain})')
            if gain<a.min_delta and not SM:S['stop_reason']=f'round {r}: wide gain {gain:+d} < +{a.min_delta} (not meaningful)'
        else:
            R['promoted']=None;S['stop_reason']=f'round {r}: no candidate passed the gate (incumbent wide {ref["wide_successes"]}/{ref["wide_episodes"]})';log(f'round{r}: no promotion -> stop')
        R['done']=True;R['finished']=cst();save_state();report()
        if S.get('stop_reason'):break
    if not S.get('stop_reason'):S['stop_reason']=f'max rounds ({a.rounds})'
    log(f'STOP: {S["stop_reason"]}');save_state()
    best=S['current_policy'];F=S.setdefault('final',{});F['policy']=best
    F['note']='best = last promoted policy' if S['promoted'] else 'NO v2 promotion: best = starting incumbent (v1 round-2 candidate), which itself fails the nominal-vs-Stage-1 condition'
    F['gate_seeds_vs_stage1']=dict(note='selection-biased (gate seeds used for promotion)',wide=paired(S['ref_gate'],S['stage1_gate'],False),nominal=paired(S['ref_gate'],S['stage1_gate'],True),cells=percell(S['ref_gate'],S['stage1_gate']))
    save_state();report()
    H=F.setdefault('heldout',{})
    for nm,pp in [('best',best),('stage1',STAGE1)]+([('start',START)] if best!=str(START) else []):
        if nm not in H:H[nm]=timed(f'final_heldout_{nm}',lambda:heldout_eval(pp,nm));save_state()
    F['heldout_comparison']=dict(seeds=f'600000-{600000+FN-1}',best_vs_stage1=dict(wide=paired(H['best'],H['stage1'],False),nominal=paired(H['best'],H['stage1'],True),cells=percell(H['best'],H['stage1'])),
        **({'best_vs_start':dict(wide=paired(H['best'],H['start'],False),nominal=paired(H['best'],H['start'],True),cells=percell(H['best'],H['start']))} if 'start' in H else {}))
    log(f'final heldout: {json.dumps({k:(v["wide_successes"],v["nominal_successes"]) for k,v in H.items()})}');save_state();report()
    if a.final_sweep and 'sweep' not in F:
        F['sweep']=timed('final_sweep',lambda:final_sweep(best));save_state();report()
    S['finished']=cst();save_state();report()
    write_json(OUT/'status.json',dict(status='complete',updated=cst(),stop_reason=S['stop_reason'],final_policy=best,promoted=S['promoted'],
        heldout={k:dict(wide=f'{v["wide_successes"]}/{v["wide_episodes"]}',nominal=f'{v["nominal_successes"]}/{v["nominal_episodes"]}') for k,v in H.items()},
        rounds={k:dict(promoted=v.get('promoted'),gate={n:dict(wide=g['wide'],nominal=g['nominal'],passes=g['passes'],why=g['why']) for n,g in v.get('gate',{}).items()}) for k,v in S['rounds'].items()}))
    log('pipeline v2 complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
