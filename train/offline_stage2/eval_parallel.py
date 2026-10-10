"""Closed-loop evaluation = Stage-1 protocol (train/bc_stage1/evaluate_bc.py, --sampler upstream,
DDIM10 eta=1 mean), split into seed shards across GPUs/processes and merged. Per-seed results are
deterministic, so sharding does not change outcomes."""
from s2common import write_json,cst,ROOT,BC_DIR,sha256
import argparse,subprocess,os,sys,json,math
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
ap.add_argument('--seed-start',type=int,required=True);ap.add_argument('--episodes',type=int,default=100)
ap.add_argument('--pos',type=float,default=.005);ap.add_argument('--yaw',type=float,default=2.)
ap.add_argument('--gpus',default='5,6,7');ap.add_argument('--per-gpu',type=int,default=2)
a=ap.parse_args();s0=a.seed_start;assert s0+a.episodes-1<900000 or s0>900999,'reserved seeds'
out=Path(a.out);sd=out.with_suffix('');sd.mkdir(parents=True,exist_ok=True)
gpus=[int(x) for x in a.gpus.split(',')];slots=[g for g in gpus for _ in range(a.per_gpu)];k=len(slots)
per=math.ceil(a.episodes/k);procs=[]
for j,g in enumerate(slots):
    st=s0+j*per;n=min(per,s0+a.episodes-st)
    if n<=0:continue
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
    cmd=[sys.executable,'-u','evaluate_bc.py','--checkpoint',a.checkpoint,'--output',str(sd/f'shard_{j}.json'),'--seed-start',str(st),'--episodes',str(n),
         '--sampler','upstream','--position-half-range',str(a.pos),'--yaw-half-range',str(a.yaw)]
    procs.append(subprocess.Popen(cmd,cwd=BC_DIR,env=env,stdout=open(sd/f'shard_{j}.log','w'),stderr=subprocess.STDOUT))
codes=[q.wait() for q in procs];assert all(c==0 for c in codes),f'eval shard failed {codes}'
rows=[]
for j in range(len(procs)):rows+= [json.loads(l) for l in open(sd/f'shard_{j}.episodes.jsonl')]
rows.sort(key=lambda r:r['seed']);assert [r['seed'] for r in rows]==list(range(s0,s0+a.episodes))
kk=sum(bool(r['success']) for r in rows);n=len(rows);z=1.95996398454;ph=kk/n;c=(ph+z*z/(2*n))/(1+z*z/n);h=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
fails={}
for r in rows:
    if not r['success']:fails[r['reason']]=fails.get(r['reason'],0)+1
res=dict(checkpoint=a.checkpoint,checkpoint_sha256=sha256(a.checkpoint),seed_start=s0,episodes=n,successes=kk,success_rate=ph,wilson95=[c-h,c+h],
    failure_reasons=fails,failed_seeds=[r['seed'] for r in rows if not r['success']],sampler='upstream',position_half_range_m=a.pos,yaw_half_range_deg=a.yaw,finished=cst())
with open(out.with_suffix('.episodes.jsonl'),'w') as f:
    for r in rows:f.write(json.dumps(r)+'\n')
write_json(out,res);print('EVAL',json.dumps(res),flush=True)
