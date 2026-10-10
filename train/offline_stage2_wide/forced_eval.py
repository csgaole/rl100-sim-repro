"""Gate forced PPO checkpoints (diagnostic). Waits for <ppo_dir>/forced_stepNNNN.pt, runs the pipeline gate
(6 wide cells x 50 @620000 + nominal 100 @880000, batch_worker deployment sampler) and paired exact one-sided
McNemar vs the round-2 candidate and Stage-1 gate results; held-out (600000-600099, 6 cells x 100 + nominal 100)
if wide >= r2cand+HELDOUT_DELTA, paired with the pipeline's final_heldout results."""
from wcommon import *
import argparse,subprocess
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--ppo-dir',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',default='75,150,300,450,600')
ap.add_argument('--v2',default=str(ROOT/'runs/offline_stage2_wide_v2'));ap.add_argument('--gpus',default='7');ap.add_argument('--per-gpu',type=int,default=5)
ap.add_argument('--heldout-delta',type=int,default=5);a=ap.parse_args()
PD=Path(a.ppo_dir);O=Path(a.out);O.mkdir(parents=True,exist_ok=True);V2=Path(a.v2);GPUS=[int(x) for x in a.gpus.split(',')];PY=sys.executable;SH=25
def log(m):
    l=f'[{cst()}] {m}';print(l,flush=True)
    with open(O/'forced_eval.log','a') as f:f.write(l+'\n')
def env_for(g):return dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
def run_jobs(policy,d,jobs):
    d.mkdir(parents=True,exist_ok=True);jf=d/'jobs.json';spec=dict(policy=str(policy),out=str(d),jobs=jobs)
    if jf.exists():assert json.loads(jf.read_text())==spec
    else:write_json(jf,spec)
    for c in (d/'claims').glob('*') if (d/'claims').exists() else []:
        if not (d/'done'/c.name).exists():c.unlink()
    todo=[j for j in jobs if not (d/'done'/j['id']).exists()]
    if todo:
        (d/'logs').mkdir(exist_ok=True);slots=[g for g in GPUS for _ in range(a.per_gpu)][:len(todo)]
        ps=[subprocess.Popen([PY,'-u',HERE/'batch_worker.py','--jobs',jf,'--wid',str(i)],cwd=str(HERE),env=env_for(g),stdout=open(d/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT) for i,g in enumerate(slots)]
        for q in ps:q.wait()
        assert all(q.returncode==0 for q in ps),f'worker failed in {d}'
    rows=[]
    for j in jobs:rows+=[json.loads(l) for l in open(d/'rows'/f"{j['id']}.jsonl")]
    return rows
def eps_for(seed0,n,cells):
    E=[]
    for pos,yaw in cells:
        for s in range(seed0,seed0+n):
            dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal' if pos==5 else f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
    return E
def summarize(rows,policy):
    cells={};fails={};out={}
    for r in rows:
        c=r['cell'];x=cells.setdefault(c,[0,0]);x[0]+=bool(r['success']);x[1]+=1;out[f"{c}:{r['seed']}"]=bool(r['success'])
        if not r['success']:f=fails.setdefault(c,{});f[r['reason']]=f.get(r['reason'],0)+1
    order=[f'p{p}_y{y}' for p,y in GATE_CELLS]+['nominal'];cells={c:cells[c] for c in order if c in cells}
    return dict(policy=str(policy),sha256=sha256(policy),wide_successes=sum(v[0] for c,v in cells.items() if c!='nominal'),wide_episodes=sum(v[1] for c,v in cells.items() if c!='nominal'),
        nominal_successes=cells['nominal'][0],nominal_episodes=cells['nominal'][1],cells=cells,fails=fails,outcomes=out)
def tail(k,n):return sum(math.comb(n,i) for i in range(k,n+1))/2**n if n else 1.
def paired(c,r,nominal):
    ks=[k for k in r['outcomes'] if k.startswith('nominal')==nominal];g=sum(c['outcomes'][k] and not r['outcomes'][k] for k in ks);l=sum(r['outcomes'][k] and not c['outcomes'][k] for k in ks)
    return dict(delta=g-l,gained=g,lost=l,p_better=tail(g,g+l),p_worse=tail(l,g+l))
def evaluate(policy,d,seed0,n,nn_):
    rp=d/'result.json'
    if rp.exists():return json.loads(rp.read_text())
    E=eps_for(seed0,n,GATE_CELLS)+eps_for(NOMINAL['seed0'] if seed0==GATE_SEED0 else seed0,nn_,[(5,2.)])
    res=summarize(run_jobs(policy,d,[dict(id=f'j_{i:04d}',kind='eval',mode='det',episodes=E[i:i+SH]) for i in range(0,len(E),SH)]),policy);write_json(rp,res);return res
G_R2=json.loads((V2/'gate/start/result.json').read_text());G_S1=json.loads((V2/'gate/stage1/result.json').read_text())
H_R2=json.loads((V2/'pipeline_state.json').read_text())['final']['heldout'];H_BEST,H_S1=H_R2['best'],H_R2['stage1']
summary=[]
for st in [int(x) for x in a.steps.split(',')]:
    ck=PD/f'forced_step{st:04d}.pt'
    while not ck.exists():
        rep=PD/'report.json'
        if rep.exists() and not ck.exists():log(f'PPO finished without step {st}');break
        time.sleep(20)
    if not ck.exists():continue
    time.sleep(5);log(f'gate step {st}')
    g=evaluate(ck,O/'gate'/f'step{st:04d}',GATE_SEED0,GATE_N,NOMINAL['n'])
    rec=dict(step=st,path=str(ck),wide=g['wide_successes'],cells={c:v[0] for c,v in g['cells'].items()},nominal=g['nominal_successes'],fails=g['fails'],
        vs_r2_wide=paired(g,G_R2,False),vs_s1_wide=paired(g,G_S1,False),vs_r2_nominal=paired(g,G_R2,True),vs_s1_nominal=paired(g,G_S1,True))
    fr=[json.loads(l) for l in open(PD/'forced.jsonl')];rec['ope']=next((x for x in fr if x['step']==st),None)
    if rec['vs_r2_wide']['delta']>=a.heldout_delta:
        log(f'step {st}: wide {rec["wide"]} >= r2cand+{a.heldout_delta} -> held-out eval')
        h=evaluate(ck,O/'heldout'/f'step{st:04d}',600000,100,100)
        rec['heldout']=dict(wide=h['wide_successes'],nominal=h['nominal_successes'],cells={c:v[0] for c,v in h['cells'].items()},
            vs_best_wide=paired(h,H_BEST,False),vs_s1_wide=paired(h,H_S1,False),vs_best_nominal=paired(h,H_BEST,True),vs_s1_nominal=paired(h,H_S1,True))
    summary.append(rec);write_json(O/'forced_summary.json',summary)
    log(f'step {st}: wide {rec["wide"]}/300 cells {rec["cells"]} | vs r2 {rec["vs_r2_wide"]} | vs S1 {rec["vs_s1_wide"]} | nominal vs S1 {rec["vs_s1_nominal"]} vs r2 {rec["vs_r2_nominal"]} | OPE {None if not rec["ope"] else (rec["ope"]["ope_gain_vs_input"],rec["ope"]["ope_gain_vs_input_se"])}'+(f' | heldout {rec["heldout"]["wide"]}/600 nom {rec["heldout"]["nominal"]}' if 'heldout' in rec else ''))
log('FORCED_EVAL_DONE')
