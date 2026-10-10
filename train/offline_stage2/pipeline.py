"""Stage-2 offline RL orchestrator (rl100-sim-repro, SKU09 flip, flip_pose_no_drop_v2).
Round r: collect rollouts with the current policy -> IQL on demos+all rollouts -> offline PPO with OPE
gate -> closed-loop dev gate (880000-880099 narrow + 720000-720099 expanded diagnostic) -> promote.
End: promoted policy (if any) final eval on 910000-910099. Resumable via per-stage done markers."""
from s2common import *
import argparse,subprocess,os,shutil,traceback
ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--rounds',type=int,default=3)
ap.add_argument('--gpus',default='5,6,7');ap.add_argument('--train-gpu',type=int,default=5);ap.add_argument('--collect-per-gpu',type=int,default=4)
ap.add_argument('--ppo-steps',type=int,default=2000);ap.add_argument('--iql-steps',type=int,default=100000)
a=ap.parse_args();SM=a.smoke
GPUS=[int(x) for x in a.gpus.split(',')];PY=sys.executable;HERE_=str(HERE)
PLAN=[(.005,2.,150),(.01,5.,150),(.02,10.,150),(.04,20.,150)] if not SM else [(.005,2.,1),(.04,20.,1)]
SHARD=25 if not SM else 1
EXP=dict(seed=790000,pos=.02,yaw=10.)  # expanded-range development diagnostic (non-reserved)
NEP=100 if not SM else 2
OUT.mkdir(parents=True,exist_ok=True);LOG=OUT/'pipeline.log'
state_path=OUT/'pipeline_state.json'
S=json.loads(state_path.read_text()) if state_path.exists() else dict(started=cst(),rounds={},current_policy=str(STAGE1),promoted=[],durations={})
S['pid']=os.getpid();S['std_max']=STD_MAX;S['gpus']=GPUS;S['train_gpu']=a.train_gpu;S['smoke']=SM;S['planned_rounds']=a.rounds
def save_state():write_json(state_path,S)
def log(msg):
    line=f'[{cst()}] {msg}';print(line,flush=True)
    with open(LOG,'a') as f:f.write(line+'\n')
def status(phase,**kw):
    write_json(OUT/'status.json',dict(status='running',phase=phase,pid=os.getpid(),updated=cst(),smoke=SM,gpus=GPUS,train_gpu=a.train_gpu,
        current_policy=S['current_policy'],promoted=S['promoted'],rounds_done=[k for k,v in S['rounds'].items() if v.get('done')],
        planned_rounds=a.rounds,baseline=S.get('baseline'),**kw))
def env_for(g):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4',S2_OUT=str(OUT))
def run(cmd,g,logf,phase,poll=None):
    log(f'RUN gpu{g}: {" ".join(map(str,cmd))}')
    pr=subprocess.Popen([PY,'-u',*map(str,cmd)],cwd=HERE_,env=env_for(g),stdout=open(logf,'a'),stderr=subprocess.STDOUT)
    while pr.poll() is None:
        status(phase,child_pid=pr.pid,child_log=str(logf),child_status=(json.loads(Path(poll).read_text()) if poll and Path(poll).exists() else None));time.sleep(30 if not SM else 5)
    if pr.returncode!=0:raise RuntimeError(f'{phase} failed rc={pr.returncode}, see {logf}')
def timed(key,fn):
    t=time.time();r=fn();S['durations'][key]=time.time()-t;save_state();return r
def evaluate(ckpt,name,seed,pos,yaw):
    dest=OUT/'evals'/f'{name}.json'
    if dest.exists() and json.loads(dest.read_text())['checkpoint_sha256']==sha256(ckpt):return json.loads(dest.read_text())
    dest.parent.mkdir(parents=True,exist_ok=True)
    run([HERE/'eval_parallel.py','--checkpoint',ckpt,'--out',dest,'--seed-start',seed,'--episodes',NEP,'--pos',pos,'--yaw',yaw,'--gpus',','.join(map(str,GPUS)),'--per-gpu',2 if not SM else 1],
        GPUS[0],OUT/'evals'/f'{name}.log',f'eval:{name}')
    r=json.loads(dest.read_text());log(f'EVAL {name}: {r["successes"]}/{r["episodes"]} fails={r["failure_reasons"]}');return r
def collect(r,policy):
    d=OUT/f'round{r}'/'rollouts';jobs=[]
    for ri,(pos,yaw,n) in enumerate(PLAN):
        base=700000+r*20000+ri*5000;check_collection_seeds(base,n)
        for s in range(base,base+n,SHARD):jobs.append((d/f'range{ri}',s,min(SHARD,base+n-s),pos,yaw))
    total=sum(j[2] for j in jobs);slots=[g for g in GPUS for _ in range(a.collect_per_gpu)];running={}
    logs=OUT/f'round{r}'/'collect_logs';logs.mkdir(parents=True,exist_ok=True);t0=time.time()
    def done_count():return sum(len(list(Path(x).glob('ep_*.npz'))) for x in {j[0] for j in jobs})
    pending=[j for j in jobs if not all((j[0]/f'ep_{s}.npz').exists() for s in range(j[1],j[1]+j[2]))]
    while pending or running:
        for slot in range(len(slots)):
            if slot not in running and pending:
                out,s,n,pos,yaw=pending.pop(0);out.mkdir(parents=True,exist_ok=True)
                running[slot]=(subprocess.Popen([str(x) for x in [PY,'-u',HERE/'collect.py','--policy',policy,'--out',out,'--seed-start',s,'--count',n,'--pos',pos,'--yaw',yaw]],
                    cwd=HERE_,env=env_for(slots[slot]),stdout=open(logs/f'{s}.log','a'),stderr=subprocess.STDOUT),(out,s,n,pos,yaw))
        for slot,(pr,job) in list(running.items()):
            if pr.poll() is not None:
                del running[slot]
                if pr.returncode!=0:raise RuntimeError(f'collect shard {job[1]} failed rc={pr.returncode}')
        dc=done_count();el=time.time()-t0
        status(f'round{r}:collect',collect_done=dc,collect_total=total,collect_workers=len(running),collect_eta_min=round(el/max(dc,1)*(total-dc)/60,1) if dc else None);time.sleep(15)
    eps=[json.loads(l) for x in {j[0] for j in jobs} for l in open(x/'episodes.jsonl')]
    seen={};[seen.__setitem__(e['seed'],e) for e in eps];eps=list(seen.values())
    summ={}
    for ri,(pos,yaw,n) in enumerate(PLAN):
        sub=[e for e in eps if e['position_half_range_m']==pos and e['yaw_half_range_deg']==yaw]
        summ[f'{int(pos*1000)}mm_{int(yaw)}deg']=dict(episodes=len(sub),successes=sum(bool(e['success']) for e in sub),
            fails={k:sum(1 for e in sub if not e['success'] and e['reason']==k) for k in {e['reason'] for e in sub if not e['success']}})
    return summ

REPORT_HEAD=open(HERE/'REPORT_TEMPLATE.md').read() if (HERE/'REPORT_TEMPLATE.md').exists() else '# Stage-2\n'
def report():
    L=[REPORT_HEAD,f'\n## 自动进度（更新于 {cst()}）\n',f'- 状态文件：`{OUT}/status.json`；日志：`{LOG}`',
       f'- 基线（Stage-1 selected_development.pt）：{json.dumps(S.get("baseline"),ensure_ascii=False)}',f'- 当前策略：`{S["current_policy"]}`；已晋升：{S["promoted"]}']
    for r,v in sorted(S['rounds'].items()):
        L.append(f'\n### Round {r}\n');L.append('```json\n'+json.dumps(v,indent=1,ensure_ascii=False,default=str)+'\n```')
    if S.get('final'):L.append('\n## 最终评估（910000–910099）\n```json\n'+json.dumps(S['final'],indent=1,ensure_ascii=False)+'\n```')
    (OUT/'REPORT.md').write_text('\n'.join(L)+'\n')

def main():
    log(f'pipeline start pid={os.getpid()} smoke={SM} gpus={GPUS} train_gpu={a.train_gpu}');save_state()
    # ---- prep: demo features + baselines ----
    if not (OUT/'data'/'demo_validation.npz').exists():
        timed('demo_features',lambda:run([HERE/'demo_features.py']+(['--limit','4'] if SM else []),a.train_gpu,OUT/'demo_features.log','prep:demo_features'))
    if 'baseline' not in S:
        s1=sha256(STAGE1);dev=ROOT/'runs/bc_stage1/development/eval_0250.json';narrow=None
        if not SM and dev.exists():
            j=json.loads(dev.read_text())
            if j['checkpoint_sha256']==s1 and j['episodes']==100 and j['seed_start']==880000 and j['sampler']=='upstream':narrow=dict(successes=j['successes'],episodes=100,source=str(dev))
        if narrow is None:narrow=evaluate(STAGE1,'stage1_dev_narrow',880000,.005,2.)
        expd=evaluate(STAGE1,'stage1_dev_expanded',EXP['seed'],EXP['pos'],EXP['yaw'])
        S['baseline']=dict(narrow=narrow['successes'],expanded=expd['successes'],episodes=NEP if SM else 100,final_910000=99,sha256=s1)
        S['ref']=dict(S['baseline']);save_state()
    report()
    for r in range(1,a.rounds+1):
        R=S['rounds'].setdefault(str(r),{})
        if R.get('done'):continue
        pol=S['current_policy'];R['policy_in']=pol;R.setdefault('started',cst());save_state()
        if 'collect' not in R:R['collect']=timed(f'r{r}_collect',lambda:collect(r,pol));save_state();log(f'round{r} collect {R["collect"]}');report()
        rounds=','.join(str(i) for i in range(1,r+1));idir=OUT/f'round{r}'/'iql'
        if not (idir/'critic.pt').exists():
            timed(f'r{r}_iql',lambda:run([HERE/'iql.py','--rounds',rounds,'--out',idir,'--steps',a.iql_steps if not SM else 300,'--eval-every',5000 if not SM else 100],
                a.train_gpu,OUT/f'round{r}'/'iql.log',f'round{r}:iql',poll=idir/'status.json'))
        R['iql']=json.loads((idir/'status.json').read_text());save_state();report()
        pdir=OUT/f'round{r}'/'ppo'
        if not (pdir/'report.json').exists():
            extra=['--steps',a.ppo_steps] if not SM else ['--steps',20,'--ope-every',10,'--batch',32,'--ope-states',64,'--accept-any']
            timed(f'r{r}_ppo',lambda:run([HERE/'ppo.py','--policy',pol,'--critic',idir/'critic.pt','--rounds',rounds,'--out',pdir,*extra],
                a.train_gpu,OUT/f'round{r}'/'ppo.log',f'round{r}:ppo',poll=pdir/'status.json'))
        R['ppo']=json.loads((pdir/'report.json').read_text());save_state();report()
        if R['ppo']['accepted']:
            cand=pdir/'accepted_latest.pt'
            n=evaluate(cand,f'round{r}_dev_narrow',880000,.005,2.);x=evaluate(cand,f'round{r}_dev_expanded',EXP['seed'],EXP['pos'],EXP['yaw'])
            ref=S['ref'];promote=n['successes']>=ref['narrow']-1 and n['successes']+x['successes']>ref['narrow']+ref['expanded']
            R['dev_eval']=dict(narrow=n['successes'],expanded=x['successes'],narrow_fails=n['failure_reasons'],expanded_fails=x['failure_reasons'],ref=dict(ref),promoted=promote)
            if promote or SM:
                dst=OUT/f'promoted_round{r}.pt';shutil.copy2(cand,dst);S['current_policy']=str(dst);S['promoted'].append(str(dst))
                S['ref']=dict(narrow=n['successes'],expanded=x['successes']);log(f'round{r} PROMOTED {dst}')
            else:log(f'round{r} candidate NOT promoted (dev gate)')
        else:R['dev_eval']=dict(skipped='no OPE-accepted candidate');log(f'round{r}: no OPE-accepted candidate')
        R['done']=True;R['finished']=cst();save_state();report()
    if S['promoted']:
        best=S['promoted'][-1];f=evaluate(best,'final_910000',910000 if not SM else 880000,.005,2.)
        S['final']=dict(policy=best,successes=f['successes'],episodes=f['episodes'],wilson95=f['wilson95'],failure_reasons=f['failure_reasons'],stage1_reference='99/100 (runs/bc_stage1/final_eval_fresh910000.json)')
    else:S['final']=dict(policy=str(STAGE1),note='no Stage-2 policy passed the dev gate; Stage-1 remains selected (99/100 on 910000-910099)')
    S['finished']=cst();save_state();report()
    write_json(OUT/'status.json',dict(status='complete',updated=cst(),final=S['final'],promoted=S['promoted'],baseline=S.get('baseline'),rounds={k:v.get('dev_eval') for k,v in S['rounds'].items()}))
    log('pipeline complete')
if __name__=='__main__':
    try:main()
    except BaseException as e:
        write_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),updated=cst(),error=repr(e),traceback=traceback.format_exc()));log(f'FAILED {e!r}');raise
