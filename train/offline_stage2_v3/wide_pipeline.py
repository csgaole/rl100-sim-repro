# [offline_stage2_v3 copy of train/offline_stage2_wide/wide_pipeline.py: start=v3 e88 (no-VIB), demos re-encoded, no Stage-1-encoder rollouts, strict success, optional nominal non-inferiority gate]
"""Stage-2 offline RL, WIDE perturbation range (rl100-sim-repro, SKU09 flip, flip_pose_no_drop_v2).
Round r: collect rollouts (deployment sampler + small-sigma stochastic) with the current policy over the wide
region -> IQL on demos + first-Stage-2 round-1 rollouts + all wide rollouts -> offline PPO (wide-state
advantage, nominal KL anchor, OPE = sanity filter) -> REAL closed-loop gate on every candidate snapshot:
6 wide cells x 50 eps (seeds 620000-620049, paired) + nominal dev 880000-880099 -> promote best passing.
End: best policy -> full 50-setting sweep on 600000-600099 (direct comparison with Stage-1 sweep) + final
nominal 910000-910099. Resumable (state file + done markers)."""
from wcommon import *
HERE=Path(__file__).resolve().parent  # this dir (s2common.HERE points at train/offline_stage2)
import argparse,subprocess,shutil,traceback,csv
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=3)
ap.add_argument('--gpus',default='5,6,7');ap.add_argument('--per-gpu',type=int,default=3);ap.add_argument('--train-gpu',type=int,default=-1)
ap.add_argument('--collect',default='3000,1500,1500');ap.add_argument('--sto-frac',type=float,default=1/3)
ap.add_argument('--ppo-steps',type=int,default=600);ap.add_argument('--iql-steps',type=int,default=100000);ap.add_argument('--lr',type=float,default=3e-6)
ap.add_argument('--anchor',type=float,default=1.);ap.add_argument('--skip-final',action='store_true',help='[v3] skip VIB-only generalization sweep; held-out 700 is run separately with the no-VIB eval');ap.add_argument('--min-delta-frac',type=float,default=.03);ap.add_argument('--nominal-min',type=int,default=98);ap.add_argument('--nominal-vs-ref',action='store_true',help='[v3] nominal gate = candidate nominal >= reference nominal (non-inferiority) instead of absolute --nominal-min')
a=ap.parse_args();SM=a.smoke
GPUS=[int(x) for x in a.gpus.split(',')];PY=sys.executable;HERE_=str(HERE)
NCOL=[int(x) for x in a.collect.split(',')] if not SM else [8,6]
SH=25 if not SM else 4;GN=GATE_N if not SM else 2;NN=NOMINAL['n'] if not SM else 2
OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'pipeline.log';state_path=OUT/'pipeline_state.json'
S=json.loads(state_path.read_text()) if state_path.exists() else dict(started=cst(),rounds={},current_policy=str(STAGE1),promoted=[],durations={})
S.update(pid=os.getpid(),gpus=GPUS,per_gpu=a.per_gpu,smoke=SM,planned_rounds=a.rounds,args=vars(a),std_max=STD_MAX)
def save_state():write_json(state_path,S)
def log(msg):
    line=f'[{cst()}] {msg}';print(line,flush=True)
    with open(LOG,'a') as f:f.write(line+'\n')
def status(phase,**kw):
    write_json(OUT/'status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=cst(),smoke=SM,gpus=GPUS,current_policy=S['current_policy'],
        promoted=S['promoted'],stage1_gate=summ_gate(S.get('stage1_gate')),current_gate=summ_gate(S.get('ref_gate')),
        rounds_done=[k for k,v in S['rounds'].items() if v.get('done')],planned_rounds=a.rounds,**kw))
def summ_gate(g):
    return None if not g else dict(wide=f"{g['wide_successes']}/{g['wide_episodes']}",cells=g['cells'],nominal=f"{g['nominal_successes']}/{g['nominal_episodes']}")
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
def compare(c,r):
    keys=[k for k in r['outcomes'] if not k.startswith('nominal')];b=sum(c['outcomes'][k] and not r['outcomes'][k] for k in keys);cc=sum(r['outcomes'][k] and not c['outcomes'][k] for k in keys)
    n=b+cc;p=sum(math.comb(n,k) for k in range(b,n+1))/2**n if n else 1.
    return dict(delta=c['wide_successes']-r['wide_successes'],gained=b,lost=cc,sign_test_p_one_sided=p)
def collect(r,policy,n):
    base=630000+(r-1)*5000;check_seeds(base,base+n-1);E={'det':[],'sto':[]}
    for s in range(base,base+n):
        rng=np.random.default_rng(s+5000000);pos=int(rng.choice([20,40,60,80],p=[.1,.3,.3,.3]));yaw=float(rng.choice([10.,20.,30.,45.]))
        dx,dy=rng.uniform(-pos,pos,2)/1000;yw=float(rng.uniform(-yaw,yaw))
        if rng.random()<.35:dy=abs(dy)
        mode='sto' if rng.random()<a.sto_frac else 'det'
        E[mode].append(dict(seed=s,dx=float(dx),dy=float(dy),yaw_deg=yw,cell=f'p{pos}_y{int(yaw)}',pos_half_mm=pos,yaw_half_deg=yaw,wide=True,collect_round=r))
    d=OUT/f'round{r}'/'collect'
    rows=run_jobs(policy,d,shards(E['det'],'collect','det','det')+shards(E['sto'],'collect','sto','sto'),f'round{r}:collect')
    out={}
    for key,f in [('by_sampler',lambda x:x['sampler']),('by_pos_mm',lambda x:x['pos_half_mm']),('by_yaw',lambda x:x['yaw_half_deg']),('by_dy_sign',lambda x:'dy+' if x['dy']>=0 else 'dy-')]:
        t={}
        for x in rows:v=t.setdefault(str(f(x)),[0,0,{}]);v[0]+=bool(x['success']);v[1]+=1;v[2][x['reason']]=v[2].get(x['reason'],0)+(not x['success'])
        out[key]={k:dict(successes=v[0],episodes=v[1],fails={kk:vv for kk,vv in v[2].items() if vv}) for k,v in sorted(t.items())}
    out['total']=dict(successes=sum(bool(x['success']) for x in rows),episodes=len(rows),seeds=[base,base+n-1]);return out
def sources(r):
    src=[['demo',str(DEMO_DIR)]]  # [v3] old Stage-1-encoder rollouts are feature-incompatible (no images stored) -> dropped
    return src+[[f'wide_round{k}',str(OUT/f'round{k}'/'collect'/'eps')] for k in range(1,r+1)]

HEAD=(HERE/'REPORT_HEAD.md').read_text() if (HERE/'REPORT_HEAD.md').exists() else '# Stage-2 wide\n'
def report():
    L=[HEAD,f'\n## 自动进度（更新于 {cst()}）\n',f'- 状态：`{OUT}/status.json`；日志：`{LOG}`；当前策略：`{S["current_policy"]}`；已晋升：{S["promoted"]}']
    if S.get('stage1_gate'):L.append(f'- Stage-1 基线（门控网格）：{json.dumps(summ_gate(S["stage1_gate"]),ensure_ascii=False)}；失败：{json.dumps(S["stage1_gate"]["fails"],ensure_ascii=False)}')
    for r,v in sorted(S['rounds'].items()):
        vv={k:x for k,x in v.items() if k not in ('iql',)};L.append(f'\n### Round {r}\n');L.append('```json\n'+json.dumps(vv,indent=1,ensure_ascii=False,default=str)+'\n```')
        if 'iql' in v:L.append('IQL：'+json.dumps({k:v['iql'].get('final',{}).get(k) for k in ['v_auc_all_wide','v_auc_t0_wide','v_auc_t20_wide','v_auc_t45_wide','v_auc_all_rollout','heldout_td_mse']}))
    if S.get('final'):L.append('\n## 最终评估\n```json\n'+json.dumps(S['final'],indent=1,ensure_ascii=False,default=str)+'\n```')
    (OUT/'REPORT.md').write_text('\n'.join(L)+'\n')

def final_sweep(best):
    d=OUT/'final_sweep';env=dict(os.environ,SWEEP_CKPT=str(best),SWEEP_OUT=str(d),**({'SWEEP_NEP':'2','SWEEP_SHARD':'2','SWEEP_CELLS':'p5_y2,p80_y45'} if SM else {}))
    sw=ROOT/'eval/generalization_sweep'
    njobs=len(json.loads(subprocess.run([PY,'-c','import sweep_eval,json;print(json.dumps(sweep_eval.jobs()))'],cwd=sw,env=env,capture_output=True,text=True,check=True).stdout))
    if not (d/'done').exists() or len(list((d/'done').glob('*')))<njobs:
        subprocess.run([PY,'sweep_eval.py','launch','--gpus',','.join(map(str,GPUS)),'--per-gpu',str(a.per_gpu)],cwd=sw,env=env,check=True)
        pids=json.loads((d/'workers.json').read_text())['pids'];log(f'final sweep launched pids {pids}');t0=time.time()
        while any(Path(f'/proc/{q}').exists() for q in pids):
            nd=len(list((d/'done').glob('*')));status('final:sweep',jobs_done=nd,jobs_total=njobs,worker_pids=pids,eta_min=round((njobs-nd)*(time.time()-t0)/max(nd,1)/60,1) if nd else None);time.sleep(30 if not SM else 3)
    nd=len(list((d/'done').glob('*')));assert nd==njobs,f'final sweep {nd}/{njobs}'
    subprocess.run([PY,'analyze.py'],cwd=sw,env=env,check=True,stdout=open(d/'analyze.log','w'),stderr=subprocess.STDOUT)
    base={r['cell']:r for r in csv.DictReader(open(ROOT/'runs/generalization_sweep/summary.csv'))};new={r['cell']:r for r in csv.DictReader(open(d/'summary.csv'))}
    comp=[]
    for c,r in new.items():
        if int(r['episodes'] or 0)==0:continue
        b=base.get(c,{});comp.append(dict(cell=c,stage1=int(b.get('successes',0) or 0),stage1_episodes=int(b.get('episodes',0) or 0),stage2_wide=int(r['successes']),episodes=int(r['episodes']),
            delta=int(r['successes'])-int(b.get('successes',0) or 0),stage2_wrong_face=int(r['wrong_face']),stage2_drop=int(r['drop'])))
    with open(OUT/'final_comparison.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(comp[0]));w.writeheader();w.writerows(comp)
    grid=[x for x in comp if not x['cell'].startswith('probe')]
    return dict(dir=str(d),grid_stage1=sum(x['stage1'] for x in grid),grid_stage2=sum(x['stage2_wide'] for x in grid),grid_episodes=sum(x['episodes'] for x in grid),
        comparison_csv=str(OUT/'final_comparison.csv'),heatmap=str(d/'heatmap_success.png'))

def main():
    log(f'pipeline start pid={os.getpid()} smoke={SM} gpus={GPUS} per_gpu={a.per_gpu} std_max={STD_MAX}');save_state()
    if 'stage1_gate' not in S:
        S['stage1_gate']=timed('stage1_gate',lambda:gate(STAGE1,'stage1'));S['ref_gate']=S['stage1_gate'];save_state()
    report()
    for r in range(1,a.rounds+1):
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):continue
        pol=S['current_policy'];R['policy_in']=pol;R.setdefault('started',cst());save_state()
        if 'collect' not in R:R['collect']=timed(f'r{r}_collect',lambda:collect(r,pol,NCOL[min(r,len(NCOL))-1]));save_state();log(f'round{r} collect {json.dumps(R["collect"]["total"])} by_pos {json.dumps({k:(v["successes"],v["episodes"]) for k,v in R["collect"]["by_pos_mm"].items()})}');report()
        sp=OUT/f'round{r}'/'sources.json';write_json(sp,sources(r));idir=OUT/f'round{r}'/'iql';g=train_gpu()
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql_wide.py','--sources',sp,'--out',idir,'--steps',a.iql_steps if not SM else 300,'--eval-every',5000 if not SM else 100],
                g,OUT/f'round{r}'/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state();report()
        pdir=OUT/f'round{r}'/'ppo'
        if not (pdir/'report.json').exists():
            extra=['--steps',a.ppo_steps,'--lr',a.lr,'--anchor',a.anchor] if not SM else ['--steps',20,'--ope-every',5,'--batch',16,'--demo-batch',8,'--ope-states',64,'--milestones','10,20','--accept-any']
            timed(f'r{r}_ppo',lambda:run([HERE/'ppo_wide.py','--policy',pol,'--critic',idir/'critic.pt','--sources',sp,'--out',pdir,*extra],
                train_gpu(),OUT/f'round{r}'/'ppo.log',f'round{r}:ppo',poll=pdir/'status.json'))
        R['ppo']={k:v for k,v in json.loads((pdir/'report.json').read_text()).items() if k!='ope_history_tail'};save_state();report()
        ref=S['ref_gate'];need=math.ceil(a.min_delta_frac*ref['wide_episodes']);R.setdefault('gate',{})
        for c in R['ppo']['candidates']:
            nm=f"r{r}_m{c['milestone']}"
            if nm in R['gate']:continue
            res=timed(f'gate_{nm}',lambda:gate(c['path'],nm));cr=compare(res,ref);c1=compare(res,S['stage1_gate'])
            ok=res['nominal_successes']>=((ref['nominal_successes'] if a.nominal_vs_ref else a.nominal_min) if not SM else 0) and cr['delta']>=need and cr['sign_test_p_one_sided']<.05
            R['gate'][nm]=dict(path=c['path'],wide=res['wide_successes'],wide_episodes=res['wide_episodes'],cells=res['cells'],fails=res['fails'],nominal=res['nominal_successes'],
                vs_ref=cr,vs_stage1=c1,required_delta=need,passes=bool(ok or SM),action_dev_demo_states=c['action_dev_demo_states'],action_dev_wide_states=c['action_dev_wide_states'])
            save_state();report()
        passing=[(v['wide'],-v['action_dev_demo_states'],k) for k,v in R['gate'].items() if v['passes']]
        if passing:
            _,_,best=max(passing);src=R['gate'][best]['path'];dst=OUT/f'promoted_round{r}.pt';shutil.copy2(src,dst)
            S['current_policy']=str(dst);S['promoted'].append(str(dst));S['ref_gate']=dict(json.loads((OUT/'gate'/best/'result.json').read_text()),policy=str(dst));R['promoted']=dict(candidate=best,path=str(dst));log(f'round{r} PROMOTED {best} -> {dst}')
        else:R['promoted']=None;log(f'round{r}: no candidate passed the real gate (ref wide {ref["wide_successes"]}/{ref["wide_episodes"]}, need +{need}, p<.05, nominal>={a.nominal_min})')
        R['done']=True;R['finished']=cst();save_state();report()
    best=S['current_policy']
    if S['promoted'] and not a.skip_final:
        F=S.setdefault('final',{});F['policy']=best
        if 'sweep' not in F:F['sweep']=timed('final_sweep',lambda:final_sweep(best));save_state();report()
        if 'nominal_910000' not in F:
            eps=gate_episodes(910000,100 if not SM else 2,[(5,2.)]);rows=run_jobs(best,OUT/'final_910000',shards(eps,'eval','det','final'),'final:910000')
            k=sum(bool(x['success']) for x in rows);F['nominal_910000']=dict(successes=k,episodes=len(rows),wilson95=wilson(k,len(rows)),
                fails={rr:sum(1 for x in rows if not x['success'] and x['reason']==rr) for rr in {x['reason'] for x in rows if not x['success']}},stage1_reference='99/100')
    elif not S['promoted']:S['final']=dict(policy=str(STAGE1),note='no candidate passed the real gate; start policy (v3 e88) remains selected')
    else:S['final']=dict(policy=best,note='final sweep skipped (--skip-final); run heldout700 with the no-VIB eval')
    S['finished']=cst();save_state();report()
    write_json(OUT/'status.json',dict(status='complete',updated=cst(),final=S['final'],promoted=S['promoted'],stage1_gate=summ_gate(S['stage1_gate']),
        rounds={k:dict(promoted=v.get('promoted'),gate={n:dict(wide=g['wide'],nominal=g['nominal'],vs_ref=g['vs_ref'],passes=g['passes']) for n,g in v.get('gate',{}).items()}) for k,v in S['rounds'].items()}))
    log('pipeline complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
