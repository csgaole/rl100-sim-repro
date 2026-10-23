# det gate of several policies on mix_episodes(640000,N) (prefix of the reserved gate block). Usage: OUT= GPU= N= POLICIES=a,b W=
import os,sys,json,subprocess;from pathlib import Path
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)));import mixcells
OUT=Path(os.environ['OUT']);N=int(os.environ.get('N','60'));G=os.environ['GPU'];W=int(os.environ.get('W','4'))
E=mixcells.mix_episodes(640000,N);env=dict(os.environ,CUDA_VISIBLE_DEVICES=G,MUJOCO_EGL_DEVICE_ID=G,MUJOCO_GL='egl',OMP_NUM_THREADS='4');res={}
for pol in os.environ['POLICIES'].split(','):
    d=OUT/Path(pol).stem;d.mkdir(parents=True);jobs=[dict(id=f'det_{i:04d}',kind='eval',mode='det',episodes=E[i:i+10]) for i in range(0,N,10)]
    (d/'jobs.json').write_text(json.dumps(dict(policy=pol,out=str(d),jobs=jobs)))
    P=[subprocess.Popen([sys.executable,'-u','batch_worker_fix.py','--jobs',str(d/'jobs.json'),'--wid',str(i)],env=env,stdout=open(d/f'w{i}.log','w'),stderr=subprocess.STDOUT) for i in range(W)]
    [q.wait() for q in P];assert all(q.returncode==0 for q in P)
    R=[json.loads(l) for f in (d/'rows').glob('*.jsonl') for l in open(f)];st={}
    for r in R:x=st.setdefault(r['stratum'],[0,0]);x[0]+=bool(r['success']);x[1]+=1
    res[pol]=dict(all=[sum(bool(r['success']) for r in R),len(R)],strata=st);print(pol,json.dumps(res[pol]),flush=True)
(OUT/'summary.json').write_text(json.dumps(res,indent=1))
