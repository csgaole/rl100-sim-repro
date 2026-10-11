"""[heldout_temp.py = heldout_eval.py + --temperature (batch_worker_temp.py) + extra paired refs (--ref NAME=result.json)]
Held-out eval of a checkpoint, same protocol as forced_eval.py's held-out branch:
seeds 600000-600099 x 6 gate cells + nominal 600000-600099 (pos 5mm / yaw 2deg), batch_worker.py deterministic deployment
sampler, paired exact one-sided McNemar vs the v2 pipeline final_heldout results (best = round-2 candidate, stage1).
Usage: python heldout_eval.py --policy CKPT --out DIR --gpu G --workers N"""
from wcommon import *
import argparse,subprocess
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--policy',required=True);ap.add_argument('--out',required=True);ap.add_argument('--gpu',type=int,default=1);ap.add_argument('--workers',type=int,default=8)
ap.add_argument('--temperature',type=float,required=True);ap.add_argument('--ref',action='append',default=[]);ap.add_argument('--v2',default=str(ROOT/'runs/offline_stage2_wide_v2'));a=ap.parse_args()
O=Path(a.out);O.mkdir(parents=True,exist_ok=True);PY=sys.executable;SH=25
def log(m):
    l=f'[{cst()}] {m}';print(l,flush=True)
    with open(O/'heldout.log','a') as f:f.write(l+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),MUJOCO_EGL_DEVICE_ID=str(a.gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
E=[]
for pos,yaw in GATE_CELLS+[(5,2.)]:
    for s in range(600000,600100):
        dx,dy,yw=init_pose(s,pos/1000,yaw);E.append(dict(seed=s,dx=dx,dy=dy,yaw_deg=yw,cell='nominal' if pos==5 else f'p{pos}_y{yaw}',pos_half_mm=pos,yaw_half_deg=yaw))
jobs=[dict(id=f'j_{i:04d}',kind='eval',mode='det',episodes=E[i:i+SH]) for i in range(0,len(E),SH)]
jf=O/'jobs.json';spec=dict(policy=str(a.policy),out=str(O),temperature=a.temperature,jobs=jobs)
if jf.exists():assert json.loads(jf.read_text())==spec
else:write_json(jf,spec)
for attempt in range(4):
    for c in (O/'claims').glob('*') if (O/'claims').exists() else []:
        if not (O/'done'/c.name).exists():c.unlink()
    todo=[j for j in jobs if not (O/'done'/j['id']).exists()]
    if not todo:break
    (O/'logs').mkdir(exist_ok=True);t0=time.time()
    ps=[subprocess.Popen([PY,'-u',HERE/'batch_worker_temp.py','--jobs',jf,'--wid',str(i)],cwd=str(HERE),env=env,stdout=open(O/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT) for i in range(min(a.workers,len(todo)))]
    for q in ps:q.wait()
    rcs=[q.returncode for q in ps];log(f'attempt {attempt}: return codes {rcs}, {time.time()-t0:.0f}s')
    if any(r!=0 for r in rcs):time.sleep(60)
assert len(list((O/'done').glob('*')))==len(jobs),'workers failed repeatedly'
rows=[]
for j in jobs:rows+=[json.loads(l) for l in open(O/'rows'/f"{j['id']}.jsonl")]
cells={};fails={};out={}
for r in rows:
    c=r['cell'];x=cells.setdefault(c,[0,0]);x[0]+=bool(r['success']);x[1]+=1;out[f"{c}:{r['seed']}"]=bool(r['success'])
    if not r['success']:f=fails.setdefault(c,{});f[r['reason']]=f.get(r['reason'],0)+1
order=[f'p{p}_y{y}' for p,y in GATE_CELLS]+['nominal'];cells={c:cells[c] for c in order}
def tail(k,n):return sum(math.comb(n,i) for i in range(k,n+1))/2**n if n else 1.
def paired(ref,keys):
    g=sum(out[k] and not ref['outcomes'][k] for k in keys);l=sum(ref['outcomes'][k] and not out[k] for k in keys)
    return dict(cand=sum(out[k] for k in keys),ref=sum(ref['outcomes'][k] for k in keys),delta=g-l,gained=g,lost=l,p_better=tail(g,g+l),p_worse=tail(l,g+l))
H=json.loads((Path(a.v2)/'pipeline_state.json').read_text())['final']['heldout'];REF=dict(r2cand=H['best'],stage1=H['stage1'])
for x in a.ref:n,f=x.split('=',1);REF[n]=json.loads(Path(f).read_text())
wk=[k for k in out if not k.startswith('nominal')];nk=[k for k in out if k.startswith('nominal')]
res=dict(temperature=a.temperature,policy=str(a.policy),sha256=sha256(a.policy),wide=sum(out[k] for k in wk),wide_n=len(wk),nominal=sum(out[k] for k in nk),nominal_n=len(nk),cells=cells,fails=fails,
    refs={n:dict(policy=R.get('policy'),wide=paired(R,wk),nominal=paired(R,nk),cells={c:paired(R,[k for k in wk if k.startswith(c+':')]) for c in order[:-1]}) for n,R in REF.items()},outcomes=out)
write_json(O/'result.json',res)
log(f"HELDOUT T={a.temperature} {a.policy}: wide {res['wide']}/600 nominal {res['nominal']}/100 cells {[cells[c][0] for c in order]} | vs r2cand {res['refs']['r2cand']['wide']} nom {res['refs']['r2cand']['nominal']} | vs S1 {res['refs']['stage1']['wide']} nom {res['refs']['stage1']['nominal']}"+''.join(f" | vs {n} {res['refs'][n]['wide']} nom {res['refs'][n]['nominal']}" for n in REF if n not in ('r2cand','stage1')))
log('HELDOUT_DONE')
