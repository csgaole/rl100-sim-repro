"""Nominal-only development gate (p5_y2: |dx|,|dy|<=5mm, |yaw|<=2deg) on seeds 880000-880099, upstream DDIM10 eta=1.
Evaluates through train/bc_stage1/evaluate_bc.py + offline_stage2_v3/novib_patch (the exact V3_START_POLICY loader, strict).
Selection: max successes, tie -> lower val loss. Patience 3 (success +2 or val loss -1% resets) after min_stop_epoch."""
import json,os,subprocess,sys,shutil,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROTOCOL=dict(version='nominal_p5y2_v1',seeds=[880000,880099],episodes=100,sampler='upstream_eta1',loader='bc_stage1/evaluate_bc.py+novib_patch strict',patience=3)
def run_stage(out,epoch,step,cand,gpu,procs,min_stop_epoch,smoke=False,mirror=None):
    out=Path(out);folder=out/'development_nominal';folder.mkdir(exist_ok=True);t0=time.time()
    n=4 if smoke else 100;per=-(-n//procs);sha=hashlib.sha256(Path(cand).read_bytes()).hexdigest();ps=[];parts=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
    print(f'NOMINAL_GATE_START epoch={epoch} cand={cand}',flush=True)
    for k in range(procs):
        s=880000+k*per;m=min(per,880000+n-s)
        if m<=0:break
        res=folder/f'eval_{epoch:04d}_p{k}.json';parts.append(res)
        if res.exists():continue
        ps.append(subprocess.Popen([sys.executable,str(HERE/'eval_novib.py'),'--checkpoint',str(cand),'--output',str(res),'--seed-start',str(s),'--episodes',str(m),'--sampler','upstream'],stdout=open(folder/f'eval_{epoch:04d}_p{k}.log','a'),stderr=subprocess.STDOUT,env=env))
    for p in ps:
        if p.wait()!=0:raise RuntimeError(f'nominal gate eval failed epoch {epoch}')
    R=[json.loads(r.read_text()) for r in parts]
    assert all(r['checkpoint_sha256']==sha and r['sampler']=='upstream' and r['position_half_range_m']==.005 and r['yaw_half_range_deg']==2. for r in R)
    succ=sum(r['successes'] for r in R);eps=sum(r['episodes'] for r in R);vl=R[0]['validation_loss'] or 0.
    cur=dict(epoch=epoch,step=step,successes=succ,episodes=eps,validation_loss=vl,candidate=str(cand),sha256=sha,failure_counts={k:sum(r['failure_counts'][k] for r in R) for k in R[0]['failure_counts']})
    path=folder/'decision.json';old=json.loads(path.read_text()) if path.exists() else {}
    prog=old.get('progress')
    if not prog:prog=dict(success_reference=succ,loss_reference=vl,stale=0);si=li=True
    else:
        si=succ>=prog['success_reference']+2;li=vl<=prog['loss_reference']*.99
        if si:prog['success_reference']=succ
        if li:prog['loss_reference']=vl
        prog['stale']=0 if si or li else prog['stale']+1
    sel=old.get('selected')
    if sel is None or (succ,-vl)>(sel['successes'],-sel['validation_loss']):
        sel=cur;shutil.copy2(cand,out/'selected_development.tmp');(out/'selected_development.tmp').replace(out/'selected_development.pt')
    stop=prog['stale']>=3 and epoch>=min_stop_epoch
    d=dict(epoch=epoch,current=cur,progress=prog,success_improved=si,loss_improved=li,selected=sel,stop=stop,stop_reason='patience3' if stop else None,protocol=PROTOCOL,seconds=round(time.time()-t0))
    for p_ in (path,folder/f'decision_{epoch:04d}.json'):
        tmp=p_.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p_)
    if mirror:Path(mirror).mkdir(parents=True,exist_ok=True);shutil.copy2(path,Path(mirror)/f'decision_{epoch:04d}.json')
    print('NOMINAL_GATE',json.dumps(dict(epoch=epoch,successes=succ,episodes=eps,val=vl,selected_epoch=sel['epoch'],selected_succ=sel['successes'],stale=prog['stale'],stop=stop)),flush=True)
    return d
