import os,sys,json,subprocess;from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import mixcells
OUT=Path(os.environ['OUT']);N=int(os.environ.get('N','50'));G=os.environ['GPU'];W=int(os.environ.get('W','4'))
pol=os.environ['V3_START_POLICY'];E=mixcells.mix_episodes(646000,N);assert E[-1]['seed']<650000
jobs=[dict(id=f'{m}_{i:04d}',kind='eval',mode=m,episodes=E[i:i+10]) for m in ('det','sto') for i in range(0,N,10)]
OUT.mkdir(parents=True,exist_ok=True);(OUT/'jobs.json').write_text(json.dumps(dict(policy=pol,out=str(OUT),jobs=jobs)));(OUT/'logs').mkdir(exist_ok=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=G,MUJOCO_EGL_DEVICE_ID=G,MUJOCO_GL='egl',OMP_NUM_THREADS='4')
worker=os.environ.get('WORKER','batch_worker_stdfix.py')
P=[subprocess.Popen([sys.executable,'-u',worker,'--jobs',str(OUT/'jobs.json'),'--wid',str(i)],env=env,stdout=open(OUT/'logs'/f'w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(W)]
print('pids',[p.pid for p in P],'worker',worker,'std',os.environ.get('S2_STD_MAX'),'fix',os.environ.get('S2_FIX_FINAL'),'arm',os.environ.get('S2_ARM_ONLY'),'clipmean',os.environ.get('S2_CLIP_IN_MEAN'),flush=True)
[p.wait() for p in P];assert all(p.returncode==0 for p in P)
R=[json.loads(l) for f in sorted((OUT/'rows').glob('*.jsonl')) for l in open(f)]
S={}
for r in R:
  for k in ('all',r.get('stratum','?')):x=S.setdefault((r['sampler'],k),[0,0]);x[0]+=bool(r['success']);x[1]+=1
d={m:{r['seed']:bool(r['success']) for r in R if r['sampler']==m} for m in ('det','sto')}
res=dict(std_max=float(os.environ['S2_STD_MAX']),n=N,fix_final=os.environ.get('S2_FIX_FINAL'),arm=os.environ.get('S2_ARM_ONLY'),clipmean=os.environ.get('S2_CLIP_IN_MEAN'),
  by={f'{m}:{k}':v for (m,k),v in sorted(S.items())},
  det_only=sum(d['det'][s] and not d['sto'][s] for s in d['det']),sto_only=sum(d['sto'][s] and not d['det'][s] for s in d['det']))
res['drop_pts']=100*(S[('det','all')][0]/S[('det','all')][1]-S[('sto','all')][0]/S[('sto','all')][1])
(OUT/'summary.json').write_text(json.dumps(res,indent=1));print(json.dumps(res),flush=True)
