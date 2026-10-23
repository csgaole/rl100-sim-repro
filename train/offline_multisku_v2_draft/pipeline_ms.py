# CRITERION v2 (Bill 2026-10-20): success = final label_up>.98 AND no floor drop (bottom never <.74); NO velocity/settle terms, all SKUs, belt carry = success (implemented in batch_worker_ms.py; old strict kept as diagnostic strict_success). Wherever this file says strict below, read: new criterion.
# [offline_multisku_v2_draft 2026-10-20] NEW file = copy of offline_multisku_v1_draft/pipeline_ms.py (v1 untouched). v2 changes (Bill 2026-10-20):
#   * PPO lr 1e-6 in ALL rounds (v1: 3e-7 r1 / 1e-7 r2+), KL anchor coef 2 (v1: 6), milestones 50,100,150,300 every round,
#     approx_kl early-stop guard passed to ppo_ms.py (--kl-stop 0.02 rolling mean over 10 steps); stop events logged.
#   * FORMAL GATE = POOLED 200 eps per policy: select 880000-880019 + confirm 880020-880039 (20/SKU each, 5 SKUs, strict). Candidate ranked
#     by pooled successes; PROMOTE iff pooled(cand) - pooled(input) >= --pool-min-delta (2) AND exact one-sided paired sign test
#     (= exact McNemar on the 200 paired outcomes) p < --pool-p (0.2). IL gate = pooled not-worse (IL pooled >= pre-IL pooled).
#   * Start policy must be given explicitly (MS_START_POLICY; chosen after v1 finishes, see pick_v1_best.py). Collect seeds 2200000+ (v1 used 2100000+).
#   * BC repro check compares STRICT vs strict (v1 compared strict 90 against loose 97).
# [offline_multisku_v1_draft 2026-10-19] NEW file. Multi-SKU, rule-compliant copy of
#   offline_stage2_v3_nominal_20261017/wide_pipeline_ic7s9.py (base file untouched). DRAFT - not launched.
"""Offline RL-100 rounds for the multi-SKU (SKU5A..E) diffusion flip policy. Strict per-round order (never reordered/merged):
  STEP1 critic (IQL)  ->  STEP2 transition model (reference only)  ->  STEP3 offline PPO (pi_old = round start, KL 6)
  -> STEP4 collect  ->  STEP5 evaluate / gate  ->  STEP6 merge  ->  STEP7 IL retrain (+ IL not-worse gate, else fall back)
Gate = OVERALL mix only (sum over the 5 SKUs), 20 eps/SKU per seed set: SELECT on 880000.. (+19) per SKU, PAIRED CONFIRM on 880020..
(+19) per SKU; FINAL unbiased reserved 920000.. (+19) per SKU (never gated/collected here). 910k is NOT used. Deployment sampler (DDIM10 eta=1 step mean, final sigma 0), STRICT flip_pose_no_drop_v2. Per-SKU numbers are logged only.
--gate-placement pre_collect (default): the formal gate of this round's PPO candidates runs at the end of STEP3 (3b) so that STEP4
  collects with the gate-PASSED policy (or, if nothing is promoted, the round INPUT policy - logged as such); STEP5 = evaluation of the
  collected rollouts (per-SKU success / failure reasons) + the round's gate record.
--gate-placement post_collect: STEP4 collects with the round INPUT policy (= last policy that passed a formal gate) and STEP5 gates the
  PPO candidates (literal reading of the 7-step order). See README 'decisions needed'.
Collect & gate ICs: uniform +-5mm / +-2deg (init_pose), SKUs evenly interleaved. Collect sampler = stochastic DDIM10 eta=1, std clamp
[.0067,.1], terminal sigma 0 (identical kernel to PPO's likelihood). Sparse terminal reward, no human correction, failures kept.
Resumable (state file + done markers). GPUs: only GPUs with NO foreign compute process (nvidia-smi) and enough free memory are used."""
from ms_common import *
import argparse,subprocess,shutil,traceback,threading
ap=argparse.ArgumentParser()
ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=3)
ap.add_argument('--gpus',default='auto',help="'auto' = any of 0-7 that is free; or an allow-list like 6,7 (still filtered for free)")
ap.add_argument('--exclude-gpus',default='0,1,2,3,5,6',help='never use these GPUs (default 0,1,2,3,5,6: only 4+7; Bill 2026-10-19 rebalance)')
ap.add_argument('--per-gpu',type=int,default=8,help='rollout workers per free GPU');ap.add_argument('--min-free-mib',type=int,default=16000)
ap.add_argument('--shard-size',type=int,default=5,help='episodes per lock-step worker batch');ap.add_argument('--gate-placement',choices=['pre_collect','post_collect'],default='pre_collect')
ap.add_argument('--collect-r0',type=int,default=1500,help='round-0 bootstrap collect with the start policy (critic needs failures)')
ap.add_argument('--collect',default='1500,1500,1500',help='episodes per round (multiple of 5 -> even per SKU)');ap.add_argument('--collect-seed0',type=int,default=2200000)
ap.add_argument('--ic-pos-mm',type=float,default=5.);ap.add_argument('--ic-yaw-deg',type=float,default=2.)
ap.add_argument('--iql-steps',type=int,default=100000);ap.add_argument('--iql-batch',type=int,default=2048)
ap.add_argument('--tm-steps',type=int,default=20000);ap.add_argument('--tm-batch',type=int,default=4096);ap.add_argument('--amq-states',type=int,default=1024)
ap.add_argument('--ppo-steps',type=int,default=300);ap.add_argument('--ppo-batch',type=int,default=256);ap.add_argument('--lr',type=float,default=1e-6);ap.add_argument('--r2-lr',type=float,default=1e-6)
ap.add_argument('--anchor',type=float,default=2.);ap.add_argument('--milestones',default='50,100,150,300');ap.add_argument('--r2-milestones',default='50,100,150,300')
ap.add_argument('--gate-per-sku',type=int,default=20);ap.add_argument('--select-seed0',type=int,default=880000);ap.add_argument('--confirm-seed0',type=int,default=880020);ap.add_argument('--final-seed0',type=int,default=920000,help='FINAL unbiased reserved seed0 (20/SKU); never used by this pipeline')
ap.add_argument('--pool-min-delta',type=int,default=2,help='pooled (select+confirm, 200 eps) candidate - input successes must be >= this')
ap.add_argument('--pool-p',type=float,default=.2,help='AND exact one-sided paired sign test (McNemar) p on the 200 pooled pairs must be < this')
ap.add_argument('--kl-stop',type=float,default=.02,help='PPO early stop: rolling-mean approx_kl > this (<=0 disables)');ap.add_argument('--kl-stop-window',type=int,default=10)
ap.add_argument('--il-steps',type=int,default=300);ap.add_argument('--il-batch',type=int,default=256);ap.add_argument('--il-lr',type=float,default=1e-6);ap.add_argument('--il-ema',type=float,default=.999)
ap.add_argument('--il-l2',type=float,default=1.0);ap.add_argument('--il-demo-frac',type=float,default=.125);ap.add_argument('--il-max-delta',type=float,default=.05)
a=ap.parse_args();SM=a.smoke
assert OUT is not None and abs(STD_MAX-0.1)<1e-12,'set S2_OUT and S2_STD_MAX=0.1 (rule 7)'
NCOL=[int(x) for x in a.collect.split(',')] if not SM else [10,10]
assert all(n%len(SKUS)==0 for n in NCOL+[a.collect_r0]),'collect sizes must be multiples of 5 (even per SKU)'
NPS=a.gate_per_sku if not SM else 2;SH=a.shard_size if not SM else 5;PY=sys.executable;HERE_=str(HERE)
for s0 in (a.select_seed0,a.confirm_seed0,a.final_seed0):assert any(lo<=s0 and s0+NPS-1<=hi for lo,hi in FORBIDDEN_W),'gate/final seeds must sit in a reserved (never-collected) block'
assert a.select_seed0!=a.confirm_seed0 and a.select_seed0!=a.final_seed0 and a.confirm_seed0!=a.final_seed0,'select/confirm/final seed0 must be distinct'
assert a.select_seed0!=910000 and a.confirm_seed0!=910000 and a.final_seed0!=910000,'910k is historical BC unbiased only; never used as gate or final here'
assert a.final_seed0==920000 or (920000<=a.final_seed0<=920099),'final unbiased must be in the 920000+ reserved block'
assert a.select_seed0+NPS-1<a.confirm_seed0 or a.confirm_seed0+NPS-1<a.select_seed0,'select and confirm seed ranges must not overlap'
POS=a.ic_pos_mm/1000;YAW=a.ic_yaw_deg
OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'pipeline.log';state_path=OUT/'pipeline_state_ms.json'
RD=lambda r:OUT/f'round{r}'
S=json.loads(state_path.read_text()) if state_path.exists() else dict(started=cst(),rounds={},current_policy=str(START_POLICY),promoted=[],durations={})
S.update(pid=os.getpid(),smoke=SM,planned_rounds=a.rounds,args=vars(a),std=[STD_MIN,STD_MAX],train_all=TRAIN_ALL,start_policy=str(START_POLICY),seed_table=dict(select=[a.select_seed0,a.select_seed0+NPS-1],confirm=[a.confirm_seed0,a.confirm_seed0+NPS-1],final_unbiased_reserved=[a.final_seed0,a.final_seed0+NPS-1],episodes_per_sku=NPS,skus=SKUS,note='910k unused; final never gated/collected by this pipeline'))
_LK=threading.RLock()
def save_state():
    with _LK:write_json(state_path,json.loads(json.dumps(S,default=str)))
def log(msg):
    line=f'[{cst()}] {msg}';print(line,flush=True)
    with _LK,open(LOG,'a') as f:f.write(line+'\n')
def status(phase,**kw):
    with _LK:write_json(OUT/'status_ms.json',dict(status='running',phase=phase,pid=os.getpid(),updated=cst(),smoke=SM,current_policy=S['current_policy'],promoted=S['promoted'],
        rounds_done=[k for k,v in S['rounds'].items() if v.get('done')],planned_rounds=a.rounds,**kw))
def timed(key,fn):
    t=time.time();r=fn();S['durations'][key]=round(time.time()-t,1);save_state();return r
# ---------------- GPU policy: free GPUs only ----------------
ALLOW=list(range(8)) if a.gpus=='auto' else [int(x) for x in a.gpus.split(',')]
EXCL={int(x) for x in a.exclude_gpus.split(',') if x}
def own_pids():
    par={}
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:par[int(d)]=int(open(f'/proc/{d}/stat').read().rsplit(')',1)[1].split()[1])
            except Exception:pass
    mine={os.getpid()};grow=True
    while grow:
        grow=False
        for c,pp in par.items():
            if pp in mine and c not in mine:mine.add(c);grow=True
    return mine
def free_gpus(min_free=None):
    """GPUs in the allow-list, not excluded, with NO foreign compute, and >= min_free MiB free. Re-reads $S2_OUT/exclude_gpus_override.txt each call."""
    min_free=a.min_free_mib if min_free is None else min_free
    excl=set(EXCL)
    ov=OUT/'exclude_gpus_override.txt'
    if ov.exists():
        excl|={int(x) for x in ov.read_text().replace(',',' ').split() if x.strip().lstrip('-').isdigit()}
    q=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],capture_output=True,text=True).stdout
    mine=own_pids();busy=set()
    for l in apps.strip().splitlines():
        pid,uuid=[x.strip() for x in l.split(',')[:2]]
        if pid.isdigit() and int(pid) not in mine:busy.add(uuid)
    res=[]
    for l in q.strip().splitlines():
        i,uuid,fr=[x.strip() for x in l.split(',')];i=int(i)
        if i in ALLOW and i not in excl and uuid not in busy and int(fr)>=min_free:res.append(i)
    return res
def wait_free(phase):
    while True:
        g=free_gpus()
        if g:return g
        log(f'{phase}: no free GPU (foreign processes on all allowed GPUs); waiting 120s');status(phase,waiting_for_gpu=True);time.sleep(120)
def env_for(g):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4',S2_OUT=str(OUT),MS_SMOKE='1' if SM else '0')
def run(cmd,logf,phase,poll=None):
    g=wait_free(phase)[0];log(f'RUN gpu{g}: {" ".join(map(str,cmd))}')
    pr=subprocess.Popen([PY,'-u',*map(str,cmd)],cwd=HERE_,env=env_for(g),stdout=open(logf,'a'),stderr=subprocess.STDOUT)
    while pr.poll() is None:
        status(phase,gpu=g,child_pid=pr.pid,child_log=str(logf),child_status=(json.loads(Path(poll).read_text()) if poll and Path(poll).exists() else None));time.sleep(30 if not SM else 3)
    if pr.returncode!=0:raise RuntimeError(f'{phase} failed rc={pr.returncode}, see {logf}')
def run_jobs(policy,d,jobs,phase,share=1):
    """Rollout jobs on all currently-free GPUs x per_gpu workers (divided by `share` when several evaluations run concurrently)."""
    d.mkdir(parents=True,exist_ok=True);jf=d/'jobs.json';spec=dict(policy=str(policy),out=str(d),jobs=jobs)
    if jf.exists():assert json.loads(jf.read_text())==spec,f'job spec changed in {d}'
    else:write_json(jf,spec)
    for attempt in range(3):
        for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
            if not (d/'done'/c.name).exists():c.unlink()
        todo=[j for j in jobs if not (d/'done'/j['id']).exists()]
        if not todo:break
        ok=wait_free(phase);pg=max(1,a.per_gpu//share)
        (d/'logs').mkdir(exist_ok=True);slots=[g for g in ok for _ in range(pg)][:len(todo)];procs=[];t0=time.time();n0=len(jobs)-len(todo)
        log(f'{phase}: attempt {attempt+1}: {len(todo)} shards on {len(slots)} workers (free gpus {ok})')
        for i,g in enumerate(slots):
            procs.append(subprocess.Popen([PY,'-u',HERE/'batch_worker_ms.py','--jobs',jf,'--wid',str(attempt*1000+i)],cwd=HERE_,env=env_for(g),stdout=open(d/'logs'/f'worker_a{attempt}_{i}.log','a'),stderr=subprocess.STDOUT))
        while any(q.poll() is None for q in procs):
            nd=len(list((d/'done').glob('*')));el=time.time()-t0;rate=(nd-n0)/el if el>0 else 0
            status(phase,jobs_done=nd,jobs_total=len(jobs),workers=sum(q.poll() is None for q in procs),eta_min=round((len(jobs)-nd)/rate/60,1) if rate>0 else None,dir=str(d));time.sleep(20 if not SM else 3)
        bad=[q.returncode for q in procs if q.returncode!=0]
        if bad:log(f'{phase}: attempt {attempt+1}: {len(bad)} workers rc!=0 (see {d}/logs)')
    nd=len(list((d/'done').glob('*')));assert nd==len(jobs),f'{phase}: {nd}/{len(jobs)} shards done'
    rows=[]
    for j in jobs:rows+=[json.loads(l) for l in open(d/'rows'/f"{j['id']}.jsonl")]
    return rows
def shards(eps,kind,mode,prefix):return [dict(id=f'{prefix}_{i:04d}',kind=kind,mode=mode,episodes=eps[i:i+SH]) for i in range(0,len(eps),SH)]
def wilson(k,n):
    if n==0:return None
    z=1.95996398454;ph=k/n;c=(ph+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n);return [round(c-h,4),round(c+h,4)]
# ---------------- gate: overall mix over 5 SKUs x NPS seeds ----------------
def gate_episodes(seed0):
    E=[]
    for sku in SKUS:
        for s in range(seed0,seed0+NPS):
            dx,dy,yw=init_pose(s,.005,2.);E.append(dict(seed=s,sku=sku,dx=dx,dy=dy,yaw_deg=yw,cell=sku))
    return E
SETS={'select':a.select_seed0,'confirm':a.confirm_seed0}
def gate(policy,name,which,share=1):
    seed0=SETS[which];d=OUT/'gate'/f'{name}_{which}';rp=d/'result.json';sha=sha256(policy);key=f'{sha}:{which}:{seed0}:{NPS}'
    with _LK:hit=S.setdefault('gate_index',{}).get(key)
    if hit and Path(hit).exists():log(f'GATE {name}:{which} = cached {hit} (same policy sha, same seeds)');return json.loads(Path(hit).read_text())
    if rp.exists() and json.loads(rp.read_text())['sha256']==sha:return json.loads(rp.read_text())
    rows=run_jobs(policy,d,shards(gate_episodes(seed0),'eval','det',which),f'gate:{name}:{which}',share)
    per={s:[sum(bool(r['success']) for r in rows if r['sku']==s),sum(r['sku']==s for r in rows)] for s in SKUS}
    fails={}
    for r in rows:
        if not r['success']:f=fails.setdefault(r['sku'],{});f[r['reason']]=f.get(r['reason'],0)+1
    k=sum(v[0] for v in per.values());n=sum(v[1] for v in per.values())
    res=dict(name=name,set=which,seeds=[seed0,seed0+NPS-1],policy=str(policy),sha256=sha,finished=cst(),successes=k,episodes=n,wilson95=wilson(k,n),
             per_sku_logged_only=per,fails=fails,outcomes={f"{r['sku']}:{r['seed']}":bool(r['success']) for r in rows})
    write_json(rp,res)
    with _LK:S['gate_index'][key]=str(rp);save_state()
    log(f"GATE {name}:{which} overall {k}/{n} per-SKU(log only) { {s:v[0] for s,v in per.items()} } fails {fails}");return res
def concurrent(fn,items):
    """independent evaluations of the SAME step run concurrently; with fewer free GPUs than items they share GPUs (workers divided)."""
    from concurrent.futures import ThreadPoolExecutor
    seen=set();items=[x for x in items if not (x[0] in seen or seen.add(x[0]))]
    if not items:return {}
    with ThreadPoolExecutor(len(items)) as ex:
        fut={nm:ex.submit(fn,pth,nm,which,len(items)) for nm,pth,which in items}
        return {nm:f.result() for nm,f in fut.items()}
def paired(c,r):
    keys=sorted(r['outcomes']);assert keys==sorted(c['outcomes'])
    b=sum(c['outcomes'][k] and not r['outcomes'][k] for k in keys);l=sum(r['outcomes'][k] and not c['outcomes'][k] for k in keys);n=b+l
    p=sum(math.comb(n,k) for k in range(b,n+1))/2**n if n else 1.
    return dict(delta=c['successes']-r['successes'],gained=b,lost=l,sign_test_p_one_sided=p)
BC_REF={'7b0bb92ade1eb43845ac55d3017b8752b2141cf6bef619b99f0ec79b52fa9114':dict(select=97,select_loose=97,select_old_strict=90)}  # candidate_0300 880000-set: BC eval STRICT 90 (loose 97). Gate here is strict -> compare strict.
def concurrent2(items):
    """like concurrent(gate,...) but keyed by (name,which) so one policy can be gated on select AND confirm in the same batch."""
    from concurrent.futures import ThreadPoolExecutor
    seen=set();items=[x for x in items if not ((x[0],x[2]) in seen or seen.add((x[0],x[2])))]
    if not items:return {}
    with ThreadPoolExecutor(len(items)) as ex:
        fut={(nm,which):ex.submit(gate,pth,nm,which,len(items)) for nm,pth,which in items}
        return {k:f.result() for k,f in fut.items()}
def pooled(rs,rc):
    """pool select+confirm results of one policy into a 200-episode record (outcome keys prefixed by set)."""
    o={f'select:{k}':v for k,v in rs['outcomes'].items()};o.update({f'confirm:{k}':v for k,v in rc['outcomes'].items()})
    return dict(successes=rs['successes']+rc['successes'],episodes=rs['episodes']+rc['episodes'],select=rs['successes'],confirm=rc['successes'],outcomes=o,
                per_sku={s:[rs['per_sku_logged_only'][s][0]+rc['per_sku_logged_only'][s][0],rs['per_sku_logged_only'][s][1]+rc['per_sku_logged_only'][s][1]] for s in SKUS})
def gate_pooled(policy,name):
    g=concurrent2([(name,policy,'select'),(name,policy,'confirm')]);return pooled(g[(name,'select')],g[(name,'confirm')])
# ---------------- collect ----------------
def collect(r,policy,n):
    base=a.collect_seed0+r*10000;check_seeds(base,base+n-1);E=[]
    for i in range(n):
        s=base+i;dx,dy,yw=init_pose(s,POS,YAW)
        E.append(dict(seed=s,sku=SKUS[i%len(SKUS)],dx=dx,dy=dy,yaw_deg=yw,cell=SKUS[i%len(SKUS)],heldout=bool((i//len(SKUS))%10==0),collect_round=r,pos_half_mm=a.ic_pos_mm,yaw_half_deg=a.ic_yaw_deg))
    rows=run_jobs(policy,RD(r)/'collect',shards(E,'collect','sto','sto'),f'round{r}:collect')
    per={}
    for x in rows:
        v=per.setdefault(x['sku'],dict(successes=0,episodes=0,fails={}));v['episodes']+=1;v['successes']+=bool(x['success'])
        if not x['success']:v['fails'][x['reason']]=v['fails'].get(x['reason'],0)+1
    return dict(policy=str(policy),total=dict(successes=sum(bool(x['success']) for x in rows),episodes=len(rows),seeds=[base,base+n-1]),per_sku=per)
def sources_all(k):
    """demos + round-0 bootstrap + collects of rounds 1..k (ALL success and failure rollouts)."""
    src=[['demo',str(DEMO_DIR)]]+([['round0',str(RD(0)/'collect'/'eps')]] if a.collect_r0>0 else [])
    return src+[[f'round{j}',str(RD(j)/'collect'/'eps')] for j in range(1,k+1)]
def formal_gate(r,R,pol):
    """POOLED gate (v2): every candidate and the input on select (880000-set) + confirm (880020-set) = 200 eps each, strict, overall mix.
    best = max pooled successes (tie -> smaller action deviation from round start). PROMOTE iff pooled(best)-pooled(input) >= pool_min_delta
    AND exact one-sided paired sign test (McNemar) p < pool_p. Final unbiased 920000+ is NOT used here."""
    if 'gate_decision' in R:return R['gate_decision']
    inp=f'r{r}_input';cands={f"r{r}_m{c['milestone']}":c['path'] for c in R['ppo']['candidates']}
    g=concurrent2([(inp,pol,w) for w in ('select','confirm')]+[(n,pth,w) for n,pth in cands.items() for w in ('select','confirm')])
    P={n:pooled(g[(n,'select')],g[(n,'confirm')]) for n in [inp,*cands]};pi=P[inp]
    dev={f"r{r}_m{c['milestone']}":c['action_dev_vs_round_start'] for c in R['ppo']['candidates']}
    ranked=sorted(cands,key=lambda n:(P[n]['successes'],-dev[n]),reverse=True);best=ranked[0] if ranked else None
    G=dict(rule=f'pooled200: cand-input >= {a.pool_min_delta} AND one-sided paired sign p < {a.pool_p}',input_pooled=pi['successes'],input_select=pi['select'],input_confirm=pi['confirm'],
           pooled={n:P[n]['successes'] for n in cands},select={n:P[n]['select'] for n in cands},confirm={n:P[n]['confirm'] for n in cands},
           pooled_per_sku={n:P[n]['per_sku'] for n in [inp,*cands]},pooled_paired={n:paired(P[n],pi) for n in cands},best=best,
           kl_early_stop=R['ppo'].get('kl_early_stop'))
    G['pool_pass']=bool(best and P[best]['successes']-pi['successes']>=a.pool_min_delta and G['pooled_paired'][best]['sign_test_p_one_sided']<a.pool_p)
    if G['pool_pass']:
        dst=OUT/f'promoted_round{r}.pt';shutil.copy2(cands[best],dst);S['promoted'].append(str(dst))
        G.update(promoted=best,promoted_path=str(dst),pol_after=str(dst),pol_after_pooled=P[best]['successes'],pol_after_select=P[best]['select'])
        log(f"round{r} GATE PROMOTED {best}: pooled {P[best]['successes']}/200 vs input {pi['successes']}/200 paired {G['pooled_paired'][best]}")
    else:
        G.update(promoted=None,pol_after=pol,pol_after_pooled=pi['successes'],pol_after_select=pi['select'])
        log(f"round{r} GATE no promotion (pooled {G['pooled']} vs input {pi['successes']}; best={best} paired={G['pooled_paired'].get(best)}) -> keep INPUT policy")
    R['gate_decision']=G;save_state();return G
def report():
    L=[f'# offline_multisku_v2 (draft pipeline, pooled gate) - auto report {cst()}',f'- state `{state_path}`, log `{LOG}`, current `{S["current_policy"]}`, promoted {S["promoted"]}',
       f"- start gates: {json.dumps({k:S.get(k,{}).get('successes') for k in ('start_select','start_confirm')})}"]
    for r,v in sorted(S['rounds'].items()):
        L.append(f'\n## Round {r}\n```json\n'+json.dumps({k:x for k,x in v.items() if k not in ('iql','tm','ppo')},indent=1,default=str)+'\n```')
    (OUT/'REPORT_ms.md').write_text('\n'.join(L)+'\n')

def main():
    log(f'pipeline start pid={os.getpid()} smoke={SM} placement={a.gate_placement} allow={ALLOW} exclude={sorted(EXCL)} free_now={free_gpus()} std=[{STD_MIN},{STD_MAX}] train_all={TRAIN_ALL}');save_state()
    assert (DEMO_DIR/'demo_train.npz').exists(),'run demo_features_ms.py first (launch script does)'
    # start policy on both gate sets (round-1 input reference; reproducibility check vs the BC evaluation)
    if 'start_select' not in S:
        g=concurrent(gate,[('start',str(START_POLICY),'select'),('start2',str(START_POLICY),'confirm')]);S['start_select']=g['start'];S['start_confirm']=g['start2']
        ref=BC_REF.get(S['start_select']['sha256'])
        if ref:S['start_repro']=dict(select_strict=[S['start_select']['successes'],ref['select']],select_match=S['start_select']['successes']==ref['select'],bc_loose=ref['select_loose'],confirm_note='880020 confirm has no BC reference')
        S['start_pooled']=S['start_select']['successes']+S['start_confirm']['successes']
        log(f"START policy gates: select {S['start_select']['successes']}/{S['start_select']['episodes']} (seed0={a.select_seed0}) confirm {S['start_confirm']['successes']}/{S['start_confirm']['episodes']} (seed0={a.confirm_seed0}); final reserved {a.final_seed0}+; repro_vs_BC_select={S.get('start_repro')}");save_state()
    if a.collect_r0>0 and 'round0_collect' not in S:
        log('round0 BOOTSTRAP_COLLECT with the start BC policy (gives the round-1 critic success+failure rollouts)')
        S['round0_collect']=timed('r0_collect',lambda:collect(0,str(START_POLICY),a.collect_r0 if not SM else 10));save_state();log(f"round0 DONE {json.dumps(S['round0_collect']['total'])}")
    report()
    for r in range(1,a.rounds+1):
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):continue
        pol=S['current_policy'];R['policy_in']=pol;R.setdefault('started',cst());save_state()
        sp=RD(r)/'sources_all.json';write_json(sp,sources_all(r-1))
        # STEP 1 critic
        idir=RD(r)/'iql';log(f'round{r} STEP1_CRITIC (IQL) on {sources_all(r-1)}')
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql_ms.py','--sources',sp,'--out',idir,'--steps',a.iql_steps if not SM else 300,'--batch',a.iql_batch,'--eval-every',5000 if not SM else 100],RD(r)/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state();log(f'round{r} STEP1_DONE');report()
        # STEP 2 transition model (reference only)
        tdir=RD(r)/'transition';log(f'round{r} STEP2_TM (reference only)')
        if not (tdir/'transition.pt').exists():
            timed(f'r{r}_tm',lambda:run([HERE/'tm_ms.py','--sources',sp,'--critic',idir/'critic.pt','--out',tdir,'--steps',a.tm_steps if not SM else 300,'--batch',a.tm_batch,'--eval-every',2000 if not SM else 100],RD(r)/'tm.log',f'round{r}:tm',poll=tdir/'status.json'))
        R['tm']=json.loads((tdir/'status.json').read_text());save_state();log(f'round{r} STEP2_DONE held_mse={R["tm"].get("held_mse")}')
        # STEP 3 offline PPO (pi_old = pol fixed, KL anchor 6 to pol, all accumulated data)
        pdir=RD(r)/'ppo';lr=a.lr if r==1 else a.r2_lr;ms=a.milestones if r==1 else a.r2_milestones;log(f'round{r} STEP3_PPO pi_old=round-start {pol} lr={lr} anchor={a.anchor} kl_stop={a.kl_stop}/w{a.kl_stop_window}')
        if not (pdir/'report.json').exists():
            extra=['--steps',30,'--batch',16,'--ope-states',64,'--milestones','10,30'] if SM else ['--steps',a.ppo_steps,'--batch',a.ppo_batch,'--milestones',ms]
            timed(f'r{r}_ppo',lambda:run([HERE/'ppo_ms.py','--policy',pol,'--critic',idir/'critic.pt','--sources',sp,'--out',pdir,'--lr',lr,'--anchor',a.anchor,'--kl-stop',a.kl_stop,'--kl-stop-window',a.kl_stop_window,*extra],RD(r)/'ppo.log',f'round{r}:ppo',poll=pdir/'status.json'))
        R['ppo']={k:v for k,v in json.loads((pdir/'report.json').read_text()).items() if k!='ope_history_tail'};save_state()
        if R['ppo'].get('kl_early_stop'):log(f"round{r} PPO KL_EARLY_STOP {json.dumps(R['ppo']['kl_early_stop'])}")
        log(f"round{r} PPO kl_summary {json.dumps(R['ppo'].get('kl_summary'))}")
        if R['ppo'].get('clip_never_active'):log(f'round{r} WARNING: PPO clip never active (ratio stayed within 1+-eps for all steps) - see {pdir}/metrics.jsonl')
        aq=RD(r)/'amq.json'
        if not aq.exists():
            pl=[['input',pol]]+[[f"r{r}_m{c['milestone']}",c['path']] for c in R['ppo']['candidates']]
            timed(f'r{r}_amq',lambda:run([HERE/'amq_ms.py','--policies',json.dumps(pl),'--critic',idir/'critic.pt','--transition',tdir/'transition.pt','--sources',sp,'--out',aq,'--states',a.amq_states if not SM else 64],RD(r)/'amq.log',f'round{r}:amq'))
        R['amq_reference_only']={k:round(v['paired_gain_vs_input'],4) for k,v in json.loads(aq.read_text())['results'].items()};save_state()
        log(f"round{r} STEP3_DONE clip_active_steps={R['ppo'].get('clip_active_steps')} max_clip_frac={R['ppo'].get('max_clip_frac')} amq(ref only)={R['amq_reference_only']}");report()
        if a.gate_placement=='pre_collect':
            log(f'round{r} STEP3b_FORMAL_GATE pooled200 (select {a.select_seed0}+ + confirm {a.confirm_seed0}+, 20/SKU each; promote iff +{a.pool_min_delta} and p<{a.pool_p})')
            G=timed(f'r{r}_gate',lambda:formal_gate(r,R,pol))
            R['collect_policy']=G['pol_after'];R['collect_policy_name']=G['promoted'] or 'input (no candidate promoted)'
        else:
            R['collect_policy']=pol;R['collect_policy_name']='input (= last formally gated policy; post_collect placement)'
        # STEP 4 collect
        log(f"round{r} STEP4_COLLECT policy={R['collect_policy_name']} path={R['collect_policy']}")
        if 'collect' not in R:R['collect']=timed(f'r{r}_collect',lambda:collect(r,R['collect_policy'],NCOL[min(r,len(NCOL))-1]));save_state()
        log(f"round{r} STEP4_DONE {json.dumps(R['collect']['total'])} per-SKU {json.dumps({s:[v['successes'],v['episodes']] for s,v in R['collect']['per_sku'].items()})}");report()
        # STEP 5 evaluate / gate
        R['data_eval']=dict(collect_policy=R['collect_policy_name'],total=R['collect']['total'],per_sku=R['collect']['per_sku'])
        if a.gate_placement=='post_collect':
            log(f'round{r} STEP5_FORMAL_GATE');G=timed(f'r{r}_gate',lambda:formal_gate(r,R,pol))
        G=R['gate_decision'];R['pol_after']=G['pol_after'];R['pol_after_select']=G['pol_after_select'];R['pol_after_pooled']=G['pol_after_pooled'];save_state()
        log(f"round{r} STEP5_DONE eval={json.dumps(R['data_eval']['total'])} gate promoted={G['promoted']} pol_after={R['pol_after']}");report()
        # STEP 6 merge (all demos + all rollouts so far -> next round's critic/TM/PPO data)
        spm=RD(r)/'sources_merged.json';write_json(spm,sources_all(r))
        R['merge']=dict(sources=str(spm),n_sources=len(sources_all(r)),own_collect_episodes=sum(S['rounds'][str(k)]['collect']['total']['episodes'] for k in range(1,r+1)));save_state();log(f"round{r} STEP6_MERGE {json.dumps(R['merge'])}")
        # STEP 7 IL retrain: demos + THIS round's collected SUCCESSES only (no accumulation), encoder frozen; not-worse gate else fall back
        pa=R['pol_after'];il_path=RD(r)/'il_policy.pt';spil=RD(r)/'sources_il.json';write_json(spil,[['demo',str(DEMO_DIR)],[f'round{r}',str(RD(r)/'collect'/'eps')]])
        log(f"round{r} STEP7_IL base={pa} data=demos + round{r} collect successes (collected by {R['collect_policy_name']})")
        if 'il' not in R:
            ppo_args=['--ppo-from',R['policy_in'],'--ppo-to',pa] if pa!=R['policy_in'] else []
            timed(f'r{r}_il',lambda:run([HERE/'il_ms.py','--policy',pa,'--sources',spil,'--out',il_path,'--steps',a.il_steps if not SM else 20,'--batch',a.il_batch,'--lr',a.il_lr,'--ema',a.il_ema,'--l2',a.il_l2,'--demo-frac',a.il_demo_frac,'--max-delta',a.il_max_delta,'--eval-every',50 if not SM else 5,*ppo_args],
                RD(r)/'il.log',f'round{r}:il',poll=str(il_path.with_suffix('.status.json'))))
            R['il']=json.loads(il_path.with_suffix('.status.json').read_text());save_state()
        if 'il_gate' not in R:
            gi=timed(f'r{r}_il_gate',lambda:gate_pooled(il_path,f'r{r}_il'));R['il_gate']=gi['successes'];R['il_gate_detail']=dict(select=gi['select'],confirm=gi['confirm'],per_sku=gi['per_sku']);save_state()
        ok=R['il_gate']>=R['pol_after_pooled']   # IL gate: POOLED (200 eps) not-worse vs the pre-IL policy (its pooled result is cached from the formal gate)
        R['il_decision']=dict(rule='pooled200 not-worse',next_input='il' if ok else 'pre_il_fallback',il_pooled=R['il_gate'],pre_il_pooled=R['pol_after_pooled'],path=str(il_path) if ok else pa)
        log(f"round{r} STEP7_DONE IL pooled {R['il_gate']}/200 vs pre-IL {R['pol_after_pooled']}/200 -> next input {R['il_decision']['next_input']} ({R['il_decision']['path']})")
        S['current_policy']=R['il_decision']['path'];R['done']=True;R['finished']=cst();save_state();report()
    S['final']=dict(policy=S['current_policy'],promoted=S['promoted'],final_seed0=a.final_seed0,note=f'final unbiased evaluation is NOT part of this pipeline; reserved seeds {a.final_seed0}..+{NPS-1} x 5 SKUs (never gated/collected)')
    S['finished']=cst();save_state();report()
    write_json(OUT/'status_ms.json',dict(status='complete',updated=cst(),final=S['final']));log('pipeline complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status_ms.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
