# [v3-ic2] std-cap smoke: e88 det vs sto(S2_STD_MAX) on identical demo-mix seeds 645000+ (reserved). Usage: OUT=dir GPU=g N=120 python smoke_std.py
import os,sys,json,subprocess,time;from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import mixcells
OUT=Path(os.environ['OUT']);N=int(os.environ.get('N',120));G=os.environ['GPU'];W=int(os.environ.get('W',4))
pol=os.environ['V3_START_POLICY'];E=mixcells.mix_episodes(645000,N);assert E[-1]['seed']<650000
jobs=[dict(id=f'{m}_{i:04d}',kind='eval',mode=m,episodes=E[i:i+15]) for m in ('det','sto') for i in range(0,N,15)]
OUT.mkdir(parents=True);(OUT/'jobs.json').write_text(json.dumps(dict(policy=pol,out=str(OUT),jobs=jobs)));(OUT/'logs').mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES=G,MUJOCO_EGL_DEVICE_ID=G,MUJOCO_GL='egl',OMP_NUM_THREADS='4')
P=[subprocess.Popen([sys.executable,'-u','batch_worker.py','--jobs',str(OUT/'jobs.json'),'--wid',str(i)],env=env,stdout=open(OUT/'logs'/f'w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(W)]
print('worker pids',[p.pid for p in P],flush=True);[p.wait() for p in P];assert all(p.returncode==0 for p in P)
R=[json.loads(l) for f in sorted((OUT/'rows').glob('*.jsonl')) for l in open(f)];S={}
for r in R:
    for k in ('all',r['stratum']):x=S.setdefault((r['sampler'],k),[0,0]);x[0]+=bool(r['success']);x[1]+=1
d={m:{r['seed']:bool(r['success']) for r in R if r['sampler']==m} for m in ('det','sto')}
res=dict(std_max=float(os.environ['S2_STD_MAX']),n=N,by={f'{m}:{k}':v for (m,k),v in sorted(S.items())},
  det_only=sum(d['det'][s] and not d['sto'][s] for s in d['det']),sto_only=sum(d['sto'][s] and not d['det'][s] for s in d['det']))
res['drop_pts']=100*(S[('det','all')][0]/S[('det','all')][1]-S[('sto','all')][0]/S[('sto','all')][1])
(OUT/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res))
