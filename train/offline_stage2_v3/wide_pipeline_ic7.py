# [v3-ic7] RL-100 Algorithm 1 order per round: IQL -> offline PPO -> closed-loop 420-mix gate vs input (replaces OPE) -> collect with promoted/input -> IL on demos+successes -> next input.
#   Round 1 starts from e88 (no collect/IL first); IQL data = demos + existing e88 rollouts (fix2 round1) + own collects of earlier rounds.
# [v3-ic6b] IL lr 1e-6 + EMA .99, 2000 steps (lr1e-4/no-EMA IL -> 0/18; lr1e-6+EMA -> 16/18 = e88)
# [v3-ic6] ic5 + IL v2 (no double action normalization) + vs_input report fix + IL-only gate (diagnostic)
# [v3-ic5] After each collect: short BC/IL on demo∪successful rollouts; then IQL+PPO.
# Round>=2: lr 1e-7, anchor 3, milestones 50/100/150. Promote by closed-loop 420-mix vs INPUT policy only
# (no nominal gate for selection). Best mix wins if > input mix and sign-test p<.05 vs input (or smoke).
# Sampler: s2common_v3fix terminal σ=0, S2_STD_MAX from env. New file.
# [v3-ic3] ic2 + fixed terminal-step sampler for collection and PPO (s2common_v3fix)
# [v3-ic2: gate = demo cell mix (420 det eps, seeds 640000-640419 reserved) + nominal 880000-880099 non-inferiority; S2_STD_MAX from env, recorded]
# [offline_stage2_v3 copy of train/offline_stage2_wide/wide_pipeline.py: start=v3 e88 (no-VIB), demos re-encoded, no Stage-1-encoder rollouts, strict success, optional nominal non-inferiority gate]
"""Stage-2 offline RL, WIDE perturbation range (rl100-sim-repro, SKU09 flip, flip_pose_no_drop_v2).
Round r: collect rollouts (deployment sampler + small-sigma stochastic) with the current policy over the wide
region -> IQL on demos + first-Stage-2 round-1 rollouts + all wide rollouts -> offline PPO (wide-state
advantage, nominal KL anchor, OPE = sanity filter) -> REAL closed-loop gate on every candidate snapshot:
6 wide cells x 50 eps (seeds 620000-620049, paired) + nominal dev 880000-880099 -> promote best passing.
End: best policy -> full 50-setting sweep on 600000-600099 (direct comparison with Stage-1 sweep) + final
nominal 910000-910099. Resumable (state file + done markers)."""
from wcommon_fix import *
HERE=Path(__file__).resolve().parent  # this dir (s2common.HERE points at train/offline_stage2)
import argparse,subprocess,shutil,traceback,csv
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=3)
ap.add_argument('--gpus',default='5,6,7');ap.add_argument('--per-gpu',type=int,default=3);ap.add_argument('--train-gpu',type=int,default=-1)
ap.add_argument('--collect',default='3000,1500,1500');ap.add_argument('--sto-frac',type=float,default=1.0);ap.add_argument('--ic-seed0',type=int,default=660000)
ap.add_argument('--ppo-steps',type=int,default=600);ap.add_argument('--iql-steps',type=int,default=100000);ap.add_argument('--lr',type=float,default=3e-7)
ap.add_argument('--anchor',type=float,default=1.);ap.add_argument('--skip-final',action='store_true',help='[v3] skip VIB-only generalization sweep; held-out 700 is run separately with the no-VIB eval');ap.add_argument('--min-delta-frac',type=float,default=.03);ap.add_argument('--nominal-min',type=int,default=98);ap.add_argument('--nominal-vs-ref',action='store_true');ap.add_argument('--il-steps',type=int,default=2000);ap.add_argument('--il-lr',type=float,default=1e-6);ap.add_argument('--il-ema',type=float,default=.99);ap.add_argument('--r2-lr',type=float,default=1e-7);ap.add_argument('--r2-anchor',type=float,default=3.);ap.add_argument('--r2-milestones',default='50,100,150')
ap.add_argument('--gate-n',type=int,default=420);ap.add_argument('--seed-rollouts',default='/data02/kemove/rl100-sim-repro/runs/offline_stage2_v3e88_ic_fix2_20261014_1424/round1/collect/eps')
a=ap.parse_args();SM=a.smoke;IC_SEED0=a.ic_seed0
import mixcells
FORBIDDEN_W.append((640000,649999))  # [ic2] reserved: mix gate 640000-640999, std smoke 645000-649999
GATE2_SEED0=640000;assert a.gate_n<=1000
assert 'S2_STD_MAX' in os.environ and abs(STD_MAX-float(os.environ['S2_STD_MAX']))<1e-12,'[ic2] set S2_STD_MAX explicitly'
GPUS=[int(x) for x in a.gpus.split(',')];PY=sys.executable;HERE_=str(HERE)
NCOL=[int(x) for x in a.collect.split(',')] if not SM else [8,6]
SH=25 if not SM else 4;GN=a.gate_n if not SM else 18;NN=NOMINAL['n'] if not SM else 2
OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'pipeline.log';state_path=OUT/'pipeline_state.json'
S=json.loads(state_path.read_text()) if state_path.exists() else dict(started=cst(),rounds={},current_policy=str(STAGE1),promoted=[],durations={})
S.update(pid=os.getpid(),gpus=GPUS,per_gpu=a.per_gpu,smoke=SM,planned_rounds=a.rounds,args=vars(a),std_max=STD_MAX,std_min=STD_MIN,sampler_fix='terminal sigma=0 (s2common_v3fix)',gate='demo-cell-mix det strict, seeds 640000+gate_n; nominal 880000-880099 vs ref')
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
            procs.append(subprocess.Popen([PY,'-u',HERE/'batch_worker_fix.py','--jobs',jf,'--wid',str(i)],cwd=HERE_,env=env_for(g),stdout=open(d/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT))
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
    wide=mixcells.mix_episodes(GATE2_SEED0,GN);nom=gate_episodes(NOMINAL['seed0'],NN,[(5,2.)])
    for e in nom:e['cell']='nominal'
    rows=run_jobs(policy,d,shards(wide,'eval','det','wide')+shards(nom,'eval','det','nominal'),f'gate:{name}')
    sm=summarize(rows,mixcells.CELL_ORDER+['nominal'])
    strata={k:[sum(bool(r['success']) for r in rows if r.get('stratum')==k),sum(r.get('stratum')==k for r in rows)] for k in ('nominal','expanded','wide')}
    ws=sum(v[0] for c,v in sm['cells'].items() if c!='nominal');wn=sum(v[1] for c,v in sm['cells'].items() if c!='nominal')
    res=dict(name=name,policy=str(policy),sha256=sha,finished=cst(),wide_successes=ws,wide_episodes=wn,wide_wilson95=wilson(ws,wn),
        nominal_successes=sm['cells']['nominal'][0],nominal_episodes=sm['cells']['nominal'][1],cells=sm['cells'],strata=strata,fails=sm['fails'],outcomes=sm['outcomes'])
    write_json(rp,res);log(f"GATE {name}: mix {ws}/{wn} strata {strata} {sm['cells']} nominal {res['nominal_successes']}/{res['nominal_episodes']} fails {sm['fails']}");return res
def compare(c,r):
    keys=[k for k in r['outcomes'] if not k.startswith('nominal')];b=sum(c['outcomes'][k] and not r['outcomes'][k] for k in keys);cc=sum(r['outcomes'][k] and not c['outcomes'][k] for k in keys)
    n=b+cc;p=sum(math.comb(n,k) for k in range(b,n+1))/2**n if n else 1.
    return dict(delta=c['wide_successes']-r['wide_successes'],gained=b,lost=cc,sign_test_p_one_sided=p)
# [v3-ic] initial conditions drawn from the SAME distribution as the v3 BC training demos:
# cell mix = demo train counts (nominal 240, expanded 9x80, wide 9x~80), stratum mix inside each cell = demo
# collectors (corner 8%, boundary 24%, uniform 68%; collect_expanded/collect_wide_v1 pose_for), same env placement
# wide=True on ALL rollouts so PPO/IQL/OPE use states from the whole aligned distribution (ppo_wide samples wide-flagged states only).
# (reset(seed=0), rng(0) +-5mm jitter, +0.05 y offset, gain 1200) as demos/eval. All episodes stochastic (sto).
DEMO_CELLS=[(5,2.,240)]+[(p,y,80) for p in (10,20,50) for y in (5.,10.,15.)]+[(40,10.,80),(40,30.,80),(60,20.,80),(60,30.,80),(60,45.,79),(80,10.,80),(80,20.,80),(80,30.,78),(80,45.,78)]
def demo_pose(rng,p,y):
    dx,dy=rng.uniform(-p,p,2);yaw=rng.uniform(-y,y);u=rng.random()
    if p<=.005:return float(dx),float(dy),float(yaw)  # original nominal demos: plain uniform +-5mm/+-2deg
    if u<.08:dx=p*rng.choice([-1,1]);dy=p*rng.choice([-1,1]);yaw=y*rng.choice([-1,1])
    elif u<.32:
        dim=int(rng.integers(3));sg=rng.choice([-1,1])
        if dim==0:dx=sg*p
        elif dim==1:dy=sg*p
        else:yaw=sg*y
    return float(dx),float(dy),float(yaw)
def collect(r,policy,n):
    base=IC_SEED0+(r-1)*5000;check_seeds(base,base+n-1);E={'det':[],'sto':[]}
    W=np.array([c[2] for c in DEMO_CELLS],float);W/=W.sum()
    for s in range(base,base+n):
        rng=np.random.default_rng(s+5000000);ci=int(rng.choice(len(DEMO_CELLS),p=W));pm,yaw,_=DEMO_CELLS[ci]
        dx,dy,yw=demo_pose(rng,pm/1000,yaw);pos=pm
        mode='sto' if rng.random()<a.sto_frac else 'det'
        E[mode].append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell=f'p{pos}_y{int(yaw)}',pos_half_mm=pos,yaw_half_deg=yaw,wide=True,ic_cell_wide=bool(pos>=40 or yaw>=10),nominal=bool(pos<=5),collect_round=r))
    assert a.sto_frac<1 or not E['det'],'sto-frac=1 must yield no det episodes'
    d=OUT/f'round{r}'/'collect'
    rows=run_jobs(policy,d,shards(E['det'],'collect','det','det')+shards(E['sto'],'collect','sto','sto'),f'round{r}:collect')
    out={}
    for key,f in [('by_sampler',lambda x:x['sampler']),('by_pos_mm',lambda x:x['pos_half_mm']),('by_yaw',lambda x:x['yaw_half_deg']),('by_dy_sign',lambda x:'dy+' if x['dy']>=0 else 'dy-')]:
        t={}
        for x in rows:v=t.setdefault(str(f(x)),[0,0,{}]);v[0]+=bool(x['success']);v[1]+=1;v[2][x['reason']]=v[2].get(x['reason'],0)+(not x['success'])
        out[key]={k:dict(successes=v[0],episodes=v[1],fails={kk:vv for kk,vv in v[2].items() if vv}) for k,v in sorted(t.items())}
    out['total']=dict(successes=sum(bool(x['success']) for x in rows),episodes=len(rows),seeds=[base,base+n-1]);return out
def sources7(k):
    """[ic7] demos + seed rollouts (existing e88 data) + this run's collects of rounds 1..k"""
    src=[['demo',str(DEMO_DIR)]]+([['seed_rollouts',a.seed_rollouts]] if a.seed_rollouts else [])
    return src+[[f'wide_round{j}',str(OUT/f'round{j}'/'collect'/'eps')] for j in range(1,k+1)]
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
    log(f'pipeline start pid={os.getpid()} smoke={SM} gpus={GPUS} per_gpu={a.per_gpu} std=[{STD_MIN},{STD_MAX}] gate_n={GN}');save_state()
    if 'stage1_gate' not in S:
        S['stage1_gate']=timed('stage1_gate',lambda:gate(STAGE1,'stage1'));S['ref_gate']=S['stage1_gate'];save_state()
    report()
    for r in range(1,a.rounds+1):
        # [ic7] RL-100 Alg.1 order: IQL -> PPO -> closed-loop gate vs input -> collect (pol_after) -> IL on demos+successes -> next input
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):continue
        pol=S['current_policy'];R['policy_in']=pol;R.setdefault('started',cst());save_state()
        sp=OUT/f'round{r}'/'sources_iql.json';write_json(sp,sources7(r-1))
        idir=OUT/f'round{r}'/'iql'
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql_wide_fix.py','--sources',sp,'--out',idir,'--steps',a.iql_steps if not SM else 300,'--eval-every',5000 if not SM else 100],
                train_gpu(),OUT/f'round{r}'/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state();report()
        pdir=OUT/f'round{r}'/'ppo'
        if not (pdir/'report.json').exists():
            if SM: extra=['--steps',30,'--ope-every',10,'--batch',16,'--demo-batch',8,'--ope-states',64,'--milestones','10,30','--lr',a.lr if r==1 else a.r2_lr,'--anchor',a.anchor if r==1 else a.r2_anchor,'--accept-any']
            elif r==1: extra=['--steps',a.ppo_steps,'--lr',a.lr,'--anchor',a.anchor,'--milestones','150,300,450,600']
            else: extra=['--steps',min(a.ppo_steps,300),'--lr',a.r2_lr,'--anchor',a.r2_anchor,'--milestones',a.r2_milestones]
            timed(f'r{r}_ppo',lambda:run([HERE/'ppo_wide_fix3.py','--policy',pol,'--critic',idir/'critic.pt','--sources',sp,'--out',pdir,*extra],
                train_gpu(),OUT/f'round{r}'/'ppo.log',f'round{r}:ppo',poll=pdir/'status.json'))
        R['ppo']={k:v for k,v in json.loads((pdir/'report.json').read_text()).items() if k!='ope_history_tail'};save_state();report()
        if 'input_gate' not in R:
            R['input_gate']=timed(f'gate_r{r}_input',lambda:gate(pol,f'r{r}_input'));save_state();report()
        inp=R['input_gate'];need=math.ceil(a.min_delta_frac*inp['wide_episodes']);R.setdefault('gate',{})
        for c in R['ppo']['candidates']:
            nm=f"r{r}_m{c['milestone']}"
            if nm in R['gate']:continue
            res=timed(f'gate_{nm}',lambda:gate(c['path'],nm));cr=compare(res,inp);c1=compare(res,S['stage1_gate'])
            ok=cr['delta']>=need and cr['sign_test_p_one_sided']<.05
            if SM: ok=res['wide_successes']>=inp['wide_successes']
            R['gate'][nm]=dict(path=c['path'],wide=res['wide_successes'],wide_episodes=res['wide_episodes'],cells=res['cells'],strata=res.get('strata'),fails=res['fails'],nominal=res['nominal_successes'],
                vs_input=cr,vs_stage1=c1,required_delta=need,passes=bool(ok),action_dev_demo_states=c['action_dev_demo_states'],action_dev_wide_states=c['action_dev_wide_states'])
            save_state();report()
        if 'pol_after' not in R:
            ranked=sorted(R['gate'].items(),key=lambda kv:(kv[1]['wide'],-kv[1]['action_dev_demo_states']),reverse=True)
            passing=[(k,v) for k,v in ranked if v['passes']]
            if passing:
                best,bv=passing[0];dst=OUT/f'promoted_round{r}.pt';shutil.copy2(bv['path'],dst);S['promoted'].append(str(dst))
                R['promoted']=dict(candidate=best,path=str(dst),mix=bv['wide'],input_mix=inp['wide_successes']);R['pol_after']=str(dst);R['pol_after_mix']=bv['wide']
                log(f'round{r} PROMOTED {best} mix {bv["wide"]}/{bv["wide_episodes"]} vs input {inp["wide_successes"]}/{inp["wide_episodes"]}')
            else:
                R['promoted']=None;R['pol_after']=pol;R['pol_after_mix']=inp['wide_successes'];log(f'round{r}: no PPO candidate beat input (input {inp["wide_successes"]}/{inp["wide_episodes"]}); keep input')
            save_state();report()
        pa=R['pol_after']
        if 'collect' not in R:R['collect']=timed(f'r{r}_collect',lambda:collect(r,pa,NCOL[min(r,len(NCOL))-1]));save_state();log(f'round{r} collect(pol_after) {json.dumps(R["collect"]["total"])}');report()
        spi=OUT/f'round{r}'/'sources_il.json';write_json(spi,sources7(r))
        il_path=OUT/f'round{r}'/'il_policy.pt'
        if 'il' not in R:
            timed(f'r{r}_il',lambda:run([HERE/'bc_il_retrain_v3.py','--policy',pa,'--sources',spi,'--out',il_path,'--steps',a.il_steps if not SM else 200,'--lr',a.il_lr,'--ema',a.il_ema],
                train_gpu(),OUT/f'round{r}'/'il.log',f'round{r}:il',poll=str(il_path.with_suffix('.status.json'))))
            R['il']=json.loads(il_path.with_suffix('.status.json').read_text());save_state();log(f'round{r} IL {json.dumps(R["il"])}')
        if 'il_gate' not in R:
            R['il_gate']=timed(f'gate_r{r}_il',lambda:gate(il_path,f'r{r}_il'));log(f"round{r} IL-only gate mix {R['il_gate']['wide_successes']}/{R['il_gate']['wide_episodes']} (pol_after {R['pol_after_mix']})")
        S['current_policy']=str(il_path)  # [ic7] IL result is next round's input (Alg.1)
        R['done']=True;R['finished']=cst();save_state();report()
    # [ic7] best offline ckpt = highest overall 420-mix among all gated policies
    allg=[(S['stage1_gate']['wide_successes'],str(STAGE1),'stage1')]
    for k,R in S['rounds'].items():
        if 'il_gate' in R: allg.append((R['il_gate']['wide_successes'],R['il_gate']['policy'],f'r{k}_il'))
        for n,g in R.get('gate',{}).items(): allg.append((g['wide'],g['path'],n))
    S['best_by_mix']=dict(zip(['mix','policy','name'],max(allg,key=lambda x:x[0])));save_state();log(f"BEST by 420-mix: {S['best_by_mix']}")
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
        rounds={k:dict(promoted=v.get('promoted'),gate={n:dict(wide=g['wide'],nominal=g['nominal'],vs_input=g.get('vs_input'),passes=g['passes']) for n,g in v.get('gate',{}).items()}) for k,v in S['rounds'].items()}))
    log('pipeline complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
