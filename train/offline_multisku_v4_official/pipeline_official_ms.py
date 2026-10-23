# [offline_multisku_v4_official 2026-10-20] Official-align pipeline (plan §9 defaults).
# Per-round order: IL → Feature reencode → IQL → Dynamics → BPPO → (gate/collect).
# BPPO: 8000 / lr1e-5 / global batch 512 / clip0.25+decay; NO KL anchor; NO per-dim adaptive.
# Critic: Q512 no LN, lr1e-4, τ0.005. Dynamics ensemble into OPE. IL data accumulate.
# Promotion option C: sim pooled gate AND dynamics-OPE both pass. Keep sparse/DDIM/IC/epsilon.
# PPO DDP: world_size≤6 on GPUs excluding BC 0–1; batch_per_gpu = 512/N. IQL single-GPU.
# Do NOT resume v3. New runs under runs/offline_multisku_v4_* only.
"""Offline RL-100 multi-SKU official-order pipeline. Resumable. Free GPUs only (exclude BC)."""
from ms_common import *

import argparse,subprocess,shutil,traceback,threading
ap=argparse.ArgumentParser()
ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=3)
ap.add_argument('--gpus',default='auto')
ap.add_argument('--exclude-gpus',default='0,1',help='BC holds 0-1; offline uses 2-7')
ap.add_argument('--per-gpu',type=int,default=8);ap.add_argument('--min-free-mib',type=int,default=16000)
ap.add_argument('--shard-size',type=int,default=5)
ap.add_argument('--gate-placement',choices=['pre_collect','post_collect'],default='pre_collect')
ap.add_argument('--collect-r0',type=int,default=1500)
ap.add_argument('--collect',default='1500,1500,1500');ap.add_argument('--collect-seed0',type=int,default=2300000)
ap.add_argument('--ic-pos-mm',type=float,default=5.);ap.add_argument('--ic-yaw-deg',type=float,default=2.)
ap.add_argument('--iql-steps',type=int,default=100000);ap.add_argument('--iql-batch',type=int,default=256)
ap.add_argument('--iql-lr',type=float,default=1e-4);ap.add_argument('--iql-tau',type=float,default=.005)
ap.add_argument('--dyn-steps',type=int,default=20000);ap.add_argument('--dyn-batch',type=int,default=4096)
ap.add_argument('--n-ensemble',type=int,default=7);ap.add_argument('--n-elites',type=int,default=5)
ap.add_argument('--amq-states',type=int,default=1024);ap.add_argument('--ope-min-gain',type=float,default=0.0)
ap.add_argument('--ppo-steps',type=int,default=8000);ap.add_argument('--ppo-batch',type=int,default=512)
ap.add_argument('--lr',type=float,default=1e-5);ap.add_argument('--clip',type=float,default=.25)
ap.add_argument('--milestones',default='1000,2000,4000,8000')
ap.add_argument('--bppo-short',action='store_true',help='BPPO fix short validation defaults via ppo_official_ms --short')
ap.add_argument('--bppo-lp-reduce',choices=['sum','mean'],default=None)
ap.add_argument('--bppo-early-stop-kl',type=float,default=None)
ap.add_argument('--bppo-no-clip-decay',action='store_true')
ap.add_argument('--bppo-no-ope-old-refresh',action='store_true')
ap.add_argument('--bppo-anchor',type=float,default=None,help='optional KL anchor coef (e.g. 6); omit=0')
ap.add_argument('--ddp-world',type=int,default=0,help='0=auto min(6, free GPUs); PPO DDP world_size')
ap.add_argument('--gate-per-sku',type=int,default=50)
ap.add_argument('--select-seed0',type=int,default=880000);ap.add_argument('--confirm-seed0',type=int,default=880050)
ap.add_argument('--final-seed0',type=int,default=920000)
ap.add_argument('--pool-min-delta',type=int,default=2);ap.add_argument('--pool-p',type=float,default=.2)
ap.add_argument('--adv-samples',type=int,default=4);ap.add_argument('--omega',type=float,default=.7)
ap.add_argument('--hard-gate-per-sku',type=int,default=0);ap.add_argument('--hard-seed0',type=int,default=880500)
ap.add_argument('--hard-ic-pos-mm',type=float,default=10.);ap.add_argument('--hard-ic-yaw-deg',type=float,default=5.)
ap.add_argument('--il-steps',type=int,default=1000);ap.add_argument('--il-batch',type=int,default=256)
ap.add_argument('--il-lr',type=float,default=2e-5);ap.add_argument('--il-encoder-lr',type=float,default=5e-6)
ap.add_argument('--il-ema',type=float,default=.999);ap.add_argument('--il-l2',type=float,default=.1)
ap.add_argument('--il-demo-frac',type=float,default=.125);ap.add_argument('--il-max-delta',type=float,default=.2)
a=ap.parse_args();SM=a.smoke

assert OUT is not None and abs(STD_MAX-0.1)<1e-12,'set S2_OUT and S2_STD_MAX=0.1 (rule 7)'
NCOL=[int(x) for x in a.collect.split(',')] if not SM else [5,5]
assert all(n%len(SKUS)==0 for n in NCOL+[a.collect_r0]),'collect sizes must be multiples of 5 (even per SKU)'
NPS=a.gate_per_sku if not SM else 1;SH=a.shard_size if not SM else 5;PY=sys.executable;HERE_=str(HERE)
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
    """GPUs in the allow-list, not excluded, with NO foreign compute, and >= min_free MiB free. Re-reads $S2_OUT/exclude_gpus_override.txt each call; if present, REPLACES CLI --exclude-gpus (not union)."""
    min_free=a.min_free_mib if min_free is None else min_free
    excl=set(EXCL)
    ov=OUT/'exclude_gpus_override.txt'
    if ov.exists():
        excl={int(x) for x in ov.read_text().replace(',',' ').split() if x.strip().lstrip('-').isdigit()}  # REPLACE CLI exclude when override present
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
def gate_episodes(seed0,n=None,pos=.005,yaw=2.):
    E=[];n=n or NPS
    for sku in SKUS:
        for s in range(seed0,seed0+n):
            dx,dy,yw=init_pose(s,pos,yaw);E.append(dict(seed=s,sku=sku,dx=dx,dy=dy,yaw_deg=yw,cell=sku))
    return E
SETS={'select':a.select_seed0,'confirm':a.confirm_seed0,'hard':a.hard_seed0}
NSET={'select':NPS,'confirm':NPS,'hard':(a.hard_gate_per_sku if not SM else 1)}
GIC={'select':(.005,2.),'confirm':(.005,2.),'hard':(a.hard_ic_pos_mm/1000,a.hard_ic_yaw_deg)}
def gate(policy,name,which,share=1):
    seed0=SETS[which];d=OUT/'gate'/f'{name}_{which}';rp=d/'result.json';sha=sha256(policy);key=f'{sha}:{which}:{seed0}:{NSET[which]}'
    with _LK:hit=S.setdefault('gate_index',{}).get(key)
    if hit and Path(hit).exists():log(f'GATE {name}:{which} = cached {hit} (same policy sha, same seeds)');return json.loads(Path(hit).read_text())
    if rp.exists() and json.loads(rp.read_text())['sha256']==sha:return json.loads(rp.read_text())
    rows=run_jobs(policy,d,shards(gate_episodes(seed0,NSET[which],*GIC[which]),'eval','det',which),f'gate:{name}:{which}',share)
    per={s:[sum(bool(r['success']) for r in rows if r['sku']==s),sum(r['sku']==s for r in rows)] for s in SKUS}
    fails={}
    for r in rows:
        if not r['success']:f=fails.setdefault(r['sku'],{});f[r['reason']]=f.get(r['reason'],0)+1
    k=sum(v[0] for v in per.values());n=sum(v[1] for v in per.values())
    res=dict(name=name,set=which,seeds=[seed0,seed0+NSET[which]-1],policy=str(policy),sha256=sha,finished=cst(),successes=k,episodes=n,wilson95=wilson(k,n),
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

def sources_il_accum(k):
    """v4: demos + ALL successful-rollout dirs from round0..k (accumulate; failures skipped inside il_ms)."""
    src=[['demo',str(DEMO_DIR)]]
    if a.collect_r0>0: src.append(['round0',str(RD(0)/'collect'/'eps')])
    src += [[f'round{j}',str(RD(j)/'collect'/'eps')] for j in range(1,k+1)]
    return src
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
    if a.hard_gate_per_sku>0:  # LOG ONLY, never a decision
        h=concurrent2([(inp,pol,'hard')]+([(best,cands[best],'hard')] if best else []));G['hard_ic_log_only']={k[0]:v['successes'] for k,v in h.items()};log(f"round{r} HARD-IC (log only) {G['hard_ic_log_only']}")
    R['gate_decision']=G;save_state();return G
def report():
    L=[f'# offline_multisku_v2 (draft pipeline, pooled gate) - auto report {cst()}',f'- state `{state_path}`, log `{LOG}`, current `{S["current_policy"]}`, promoted {S["promoted"]}',
       f"- start gates: {json.dumps({k:S.get(k,{}).get('successes') for k in ('start_select','start_confirm')})}"]
    for r,v in sorted(S['rounds'].items()):
        L.append(f'\n## Round {r}\n```json\n'+json.dumps({k:x for k,x in v.items() if k not in ('iql','tm','ppo')},indent=1,default=str)+'\n```')
    (OUT/'REPORT_ms.md').write_text('\n'.join(L)+'\n')


def promote_option_C(r, R, pol):
    """Option C: BOTH sim pooled gate AND dynamics-OPE must pass for the same candidate."""
    G = formal_gate(r, R, pol)  # sim gate (may already promote — we re-decide)
    # Undo premature promote from formal_gate if OPE fails; re-apply jointly
    ope = R.get('ope_dyn', {})
    ope_best = ope.get('best')
    ope_pass_names = {n for n, v in (ope.get('results') or {}).items()
                      if n != ope.get('input') and v.get('ope_pass')}
    sim_promoted = G.get('promoted')
    # Map sim milestone name r{r}_mX <-> ope names
    joint = None
    if sim_promoted and sim_promoted in ope_pass_names:
        joint = sim_promoted
    elif sim_promoted and ope_best and sim_promoted == ope_best:
        joint = sim_promoted
    # If formal_gate already copied a promote file but OPE fails, revert
    Cdec = dict(rule='option_C: sim_pooled_gate AND dynamics_OPE both pass (same candidate preferred)',
                sim=dict(promoted=sim_promoted, pool_pass=G.get('pool_pass'), pooled=G.get('pooled'),
                         input_pooled=G.get('input_pooled')),
                ope=dict(best=ope_best, any_pass=ope.get('any_pass'), gains={
                    n: v.get('paired_gain_vs_input') for n, v in (ope.get('results') or {}).items()}),
                joint_promoted=None)
    # Require BOTH: sim pool_pass candidate must also have ope_pass
    if G.get('pool_pass') and sim_promoted and sim_promoted in ope_pass_names:
        dst = OUT/f'promoted_round{r}.pt'
        src_path = {f"r{r}_m{c['milestone']}": c['path'] for c in R['ppo']['candidates']}[sim_promoted]
        shutil.copy2(src_path, dst)
        if str(dst) not in S['promoted']:
            S['promoted'].append(str(dst))
        Cdec['joint_promoted'] = sim_promoted
        G.update(promoted=sim_promoted, promoted_path=str(dst), pol_after=str(dst),
                 pol_after_pooled=G['pooled'][sim_promoted], option_C=Cdec)
        log(f"round{r} OPTION_C PROMOTED {sim_promoted} (sim+OPE)")
    else:
        # clear any file formal_gate may have written
        dst = OUT/f'promoted_round{r}.pt'
        # keep pol as input
        G.update(promoted=None, promoted_path=None, pol_after=pol,
                 pol_after_pooled=G['input_pooled'], pol_after_select=G.get('input_select'),
                 option_C=Cdec)
        # remove from promoted list if we added under failed C
        S['promoted'] = [p for p in S['promoted'] if f'promoted_round{r}.pt' not in p]
        if dst.exists():
            try: dst.unlink()
            except Exception: pass
        log(f"round{r} OPTION_C no promotion (sim_pass={G.get('pool_pass')} sim={sim_promoted} ope_pass={sorted(ope_pass_names)}) -> keep INPUT")
    R['gate_decision'] = G; R['option_C'] = Cdec; save_state(); return G

def report():
    L=[f'# offline_multisku_v4_official - auto report {cst()}',
       f'- state `{state_path}`, log `{LOG}`, current `{S["current_policy"]}`, promoted {S["promoted"]}',
       f"- start gates: {json.dumps({k:S.get(k,{}).get('successes') for k in ('start_select','start_confirm')})}"]
    for r,v in sorted(S['rounds'].items()):
        L.append(f'\n## Round {r}\n```json\n'+json.dumps({k:x for k,x in v.items() if k not in ('iql','dyn','ppo')},indent=1,default=str)+'\n```')
    (OUT/'REPORT_ms.md').write_text('\n'.join(L)+'\n')

def main():
    log(f'v4_official pipeline start pid={os.getpid()} smoke={SM} placement={a.gate_placement} allow={ALLOW} exclude={sorted(EXCL)} free_now={free_gpus()} std=[{STD_MIN},{STD_MAX}] gamma={GAMMA}');save_state()
    assert (DEMO_DIR/'demo_train.npz').exists(),'run demo_features_ms.py first'
    if 'start_select' not in S:
        g=concurrent(gate,[('start',str(START_POLICY),'select'),('start2',str(START_POLICY),'confirm')]);S['start_select']=g['start'];S['start_confirm']=g['start2']
        ref=BC_REF.get(S['start_select']['sha256'])
        if ref:S['start_repro']=dict(select_strict=[S['start_select']['successes'],ref['select']],select_match=S['start_select']['successes']==ref['select'],bc_loose=ref['select_loose'])
        S['start_pooled']=S['start_select']['successes']+S['start_confirm']['successes']
        log(f"START gates select {S['start_select']['successes']} confirm {S['start_confirm']['successes']}");save_state()
    if a.collect_r0>0 and 'round0_collect' not in S:
        log('round0 BOOTSTRAP_COLLECT')
        S['round0_collect']=timed('r0_collect',lambda:collect(0,str(START_POLICY),a.collect_r0 if not SM else 5));save_state()
        log(f"round0 DONE {json.dumps(S['round0_collect']['total'])}")
    report()
    for r in range(1,a.rounds+1):
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):continue
        pol=S['current_policy'];R['policy_in']=pol;R.setdefault('started',cst());save_state()
        sp_prev=RD(r)/'sources_all_prev.json';write_json(sp_prev,sources_all(r-1))
        # ---- STEP A: IL (accumulate) on current policy ----
        il_path=RD(r)/'il_policy.pt';spil=RD(r)/'sources_il.json'
        write_json(spil,sources_il_accum(r-1))
        log(f'round{r} STEP_IL accumulate sources={sources_il_accum(r-1)} base={pol}')
        if 'il' not in R:
            wait_free(f'round{r}:il')
            g0=free_gpus()[0]
            timed(f'r{r}_il',lambda:run([HERE/'il_ms.py','--policy',pol,'--sources',spil,'--out',il_path,
                '--steps',a.il_steps if not SM else 10,'--batch',a.il_batch,'--lr',a.il_lr,
                '--encoder-lr',a.il_encoder_lr,'--ema',a.il_ema,'--l2',a.il_l2,
                '--demo-frac',a.il_demo_frac,'--max-delta',a.il_max_delta,'--eval-every',50 if not SM else 5],
                RD(r)/'il.log',f'round{r}:il',poll=str(il_path.with_suffix('.status.json'))))
            # run() uses env_for which needs a GPU — patch by setting CUDA in run_jobs style
            R['il']=json.loads(il_path.with_suffix('.status.json').read_text());save_state()
        # IL not-worse gate (pooled)
        if 'il_gate' not in R:
            if SM:
                # smoke: skip redundant pre_il sim gate; baseline = start_pooled
                gi=timed(f'r{r}_il_gate',lambda:gate_pooled(il_path,f'r{r}_il'))
                R['input_pooled_before_il']=S.get('start_pooled', gi['successes'])
                R['il_gate']=gi['successes'];R['il_gate_detail']=dict(select=gi['select'],confirm=gi['confirm'],smoke_skip_pre_il=True);save_state()
            else:
                gi=timed(f'r{r}_il_gate',lambda:gate_pooled(il_path,f'r{r}_il'))
                if 'input_pooled_before_il' not in R:
                    gi0=timed(f'r{r}_pre_il_gate',lambda:gate_pooled(pol,f'r{r}_pre_il'))
                    R['input_pooled_before_il']=gi0['successes']
                R['il_gate']=gi['successes'];R['il_gate_detail']=dict(select=gi['select'],confirm=gi['confirm']);save_state()
        ok=R['il_gate']>=R['input_pooled_before_il']
        if SM and ok:
            # smoke: keep start-policy feat stamps; still exercised IL train+gate, but do not adopt encoder change
            log(f'round{r} IL smoke: would ADOPT but forcing FALLBACK to avoid full reencode')
            ok=False
        R['il_decision']=dict(rule='pooled not-worse',adopt=ok,il_pooled=R['il_gate'],
                              pre_il_pooled=R['input_pooled_before_il'],path=str(il_path) if ok else pol,
                              smoke_force_fallback=bool(SM))
        pol_il=R['il_decision']['path'];log(f"round{r} IL {'ADOPT' if ok else 'FALLBACK'} -> {pol_il}")
        # ---- STEP B: Feature reencode (full) if IL adopted ----
        sp_feat=RD(r)/'sources_for_feat.json';write_json(sp_feat,sources_all(r-1))
        if ok and 'reencode' not in R:
            log(f'round{r} STEP_FEAT reencode all with {pol_il}')
            if SM:
                R['reencode']=dict(skipped_smoke=True, reason='avoid 450-demo wall; IL forced fallback', updated=cst());save_state()
                log(f'round{r} STEP_FEAT skipped_smoke')
            else:
                re_cmd=[HERE/'reencode_features_ms.py','--policy',pol_il,'--sources',sp_feat,'--force']
                timed(f'r{r}_reencode',lambda:run(re_cmd, RD(r)/'reencode.log',f'round{r}:reencode'))
                R['reencode']=dict(policy=str(pol_il),updated=cst());save_state()
        # ---- STEP C: IQL (single GPU) ----
        idir=RD(r)/'iql';log(f'round{r} STEP_IQL')
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql_ms.py','--sources',sp_feat,'--out',idir,
                '--steps',a.iql_steps if not SM else 100,'--batch',a.iql_batch,
                '--lr',a.iql_lr,'--tau',a.iql_tau,'--eval-every',5000 if not SM else 100],
                RD(r)/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state()
        # ---- STEP D: Dynamics ensemble ----
        ddir=RD(r)/'dynamics';log(f'round{r} STEP_DYNAMICS ensemble {a.n_ensemble}/{a.n_elites}')
        if not (ddir/'dynamics.pt').exists():
            timed(f'r{r}_dyn',lambda:run([HERE/'dynamics_ms.py','--sources',sp_feat,'--critic',idir/'critic.pt','--out',ddir,
                '--steps',a.dyn_steps if not SM else 50,'--batch',a.dyn_batch if not SM else 256,
                '--n-ensemble',a.n_ensemble if not SM else 2,'--n-elites',a.n_elites if not SM else 1,
                '--eval-every',2000 if not SM else 50],
                RD(r)/'dyn.log',f'round{r}:dyn',poll=ddir/'status.json'))
        R['dyn']=json.loads((ddir/'status.json').read_text());save_state()
        # ---- STEP E: BPPO DDP ----
        pdir=RD(r)/'bppo';log(f'round{r} STEP_BPPO official 8000/1e-5/512/clip0.25 policy={pol_il}')
        if not (pdir/'report.json').exists():
            gpus=free_gpus();
            if not gpus: wait_free(f'round{r}:bppo'); gpus=free_gpus()
            batch=a.ppo_batch if not SM else None
            # Official: keep global batch 512; pick largest world_size|batch among free GPUs (<=6).
            # Avoids assert when len(free) does not divide 512 (e.g. 5 free while GPU5 has yu).
            if a.ddp_world>0:
                n=max(1,min(a.ddp_world,len(gpus),6))
            else:
                cand=min(6,len(gpus))
                if SM:
                    n=max(1,cand)
                else:
                    n=max((k for k in range(cand,0,-1) if batch%k==0), default=1)
            use=sorted(gpus)[-n:]; visible=','.join(map(str,use))
            steps=a.ppo_steps if not SM else 30
            # smoke: global batch divisible by n, small
            if SM: batch=max(n, n*2); steps=8; ms='4,8'
            else: ms=a.milestones
            if batch%n!=0:
                batch=(batch//n)*n; log(f'BPPO_WARN shrunk global batch to {batch} for world={n}')
            assert batch%n==0 and n>=1
            log(f'BPPO_WORLD pick n={n} gpus={use} free={gpus} global_batch={batch} local={batch//n}')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=visible,MUJOCO_GL='egl',OMP_NUM_THREADS='4',S2_OUT=str(OUT),MS_SMOKE='1' if SM else '0')
            cmd=[PY,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={n}',
                 str(HERE/'ppo_official_ms.py'),'--policy',pol_il,'--critic',str(idir/'critic.pt'),
                 '--sources',str(sp_feat),'--out',str(pdir),'--steps',str(steps),'--batch',str(batch),
                 '--lr',str(a.lr),'--clip',str(a.clip),'--milestones',ms,
                 '--adv-samples',str(a.adv_samples if not SM else 2),'--omega',str(a.omega),
                 '--ope-every',str(50 if not SM else 5),'--ope-states',str(64 if SM else 2048),'--grad-clip','0.5']
            if getattr(a,'bppo_short',False): cmd.append('--short')
            if getattr(a,'bppo_lp_reduce',None): cmd += ['--lp-reduce', a.bppo_lp_reduce]
            if getattr(a,'bppo_early_stop_kl',None) is not None: cmd += ['--early-stop-kl', str(a.bppo_early_stop_kl)]
            if getattr(a,'bppo_no_clip_decay',False): cmd.append('--no-clip-decay')
            if getattr(a,'bppo_no_ope_old_refresh',False): cmd.append('--no-ope-old-refresh')
            if getattr(a,'bppo_anchor',None) is not None: cmd += ['--anchor', str(a.bppo_anchor)]
            log(f'BPPO_DDP world={n} gpus={use} batch_global={batch} local={batch//n}')
            def _run_bppo():
                pdir.mkdir(parents=True,exist_ok=True)
                lf=RD(r)/'ppo.log'
                with open(lf,'w') as f:
                    proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=env,cwd=str(HERE))
                while proc.poll() is None:
                    status(f'round{r}:bppo',child_pid=proc.pid,dir=str(pdir));time.sleep(20 if not SM else 3)
                if proc.returncode!=0: raise RuntimeError(f'BPPO failed rc={proc.returncode} see {lf}')
            timed(f'r{r}_bppo',_run_bppo)
        R['ppo']={k:v for k,v in json.loads((pdir/'report.json').read_text()).items() if k!='ope_history_tail'};save_state()
        log(f"round{r} BPPO done clip_active={R['ppo'].get('clip_active_steps')} world={R['ppo'].get('world_size')}")
        # ---- OPE dynamics (decision) ----
        oq=RD(r)/'ope_dyn.json'
        if not oq.exists():
            pl=[['input',pol_il]]+[[f"r{r}_m{c['milestone']}",c['path']] for c in R['ppo']['candidates']]
            timed(f'r{r}_ope',lambda:run([HERE/'ope_dynamics_ms.py','--policies',json.dumps(pl),
                '--critic',idir/'critic.pt','--dynamics',ddir/'dynamics.pt','--sources',sp_feat,
                '--out',oq,'--states',a.amq_states if not SM else 64,'--min-gain',a.ope_min_gain],
                RD(r)/'ope_dyn.log',f'round{r}:ope'))
        R['ope_dyn']=json.loads(oq.read_text());save_state()
        # also keep amq log-only for continuity
        aq=RD(r)/'amq.json'
        if not aq.exists() and (ddir/'transition.pt').exists():
            pl=[['input',pol_il]]+[[f"r{r}_m{c['milestone']}",c['path']] for c in R['ppo']['candidates']]
            timed(f'r{r}_amq',lambda:run([HERE/'amq_ms.py','--policies',json.dumps(pl),'--critic',idir/'critic.pt',
                '--transition',ddir/'transition.pt','--sources',sp_feat,'--out',aq,'--states',a.amq_states if not SM else 64],
                RD(r)/'amq.log',f'round{r}:amq'))
            if aq.exists():
                R['amq_log']={k:round(v['paired_gain_vs_input'],4) for k,v in json.loads(aq.read_text())['results'].items()}
        # ---- Gate option C + collect ----
        if a.gate_placement=='pre_collect':
            log(f'round{r} STEP_GATE option_C')
            G=timed(f'r{r}_gate',lambda:promote_option_C(r,R,pol_il))
            R['collect_policy']=G['pol_after'];R['collect_policy_name']=G['promoted'] or 'input (no C promotion)'
        else:
            R['collect_policy']=pol_il;R['collect_policy_name']='il_policy (post_collect)'
        log(f"round{r} STEP_COLLECT {R['collect_policy_name']}")
        if 'collect' not in R:
            R['collect']=timed(f'r{r}_collect',lambda:collect(r,R['collect_policy'],NCOL[min(r,len(NCOL))-1]));save_state()
        if a.gate_placement=='post_collect':
            G=timed(f'r{r}_gate',lambda:promote_option_C(r,R,pol_il))
        G=R['gate_decision'];R['pol_after']=G['pol_after'];R['pol_after_pooled']=G.get('pol_after_pooled');save_state()
        # next round input = promoted or IL policy (if C failed keep pol_il which may still be better than raw)
        S['current_policy']=R['pol_after']
        spm=RD(r)/'sources_merged.json';write_json(spm,sources_all(r))
        R['merge']=dict(sources=str(spm),n_sources=len(sources_all(r)));R['done']=True;R['finished']=cst();save_state();report()
        log(f"round{r} DONE pol_after={R['pol_after']}")
    S['final']=dict(policy=S['current_policy'],promoted=S['promoted'],note='v4_official complete; final unbiased 920k reserved')
    S['finished']=cst();save_state();report()
    write_json(OUT/'status_ms.json',dict(status='complete',updated=cst(),final=S['final']));log('pipeline complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status_ms.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
