"""Per-SKU nominal development gate (copy of bc_nominal_v3 nominal_gate.py): each SKU (SKU5A..E on scene_sku10..14) is evaluated
on p5_y2 nominal IC (|dx|,|dy|<=5mm, |yaw|<=2deg), seeds 880000-880019 (20 eps/SKU, 100 total), upstream DDIM10 eta=1 (step_mean:
no injected noise, final sigma 0), success flip_pose_no_drop_v2. Selection: max total successes, tie -> lower val loss.
Patience 3 (total success +2 or val loss -1% resets) after min_stop_epoch."""
import json,os,subprocess,sys,shutil,hashlib,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
SKUS=['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E'];SEED0=880000;PER_SKU=20
PROTOCOL=dict(version='nominal_p5y2_multisku_v1',skus=SKUS,seeds=[SEED0,SEED0+PER_SKU-1],episodes_per_sku=PER_SKU,episodes=PER_SKU*len(SKUS),sampler='upstream_eta1_final_sigma0',success='flip_pose_no_drop_v2',loader='evaluate_bc_multisku.py strict',patience=3)
def run_stage(out,epoch,step,cand,gpu,procs,min_stop_epoch,smoke=False,mirror=None):
    out=Path(out);folder=out/'development_nominal';folder.mkdir(exist_ok=True);t0=time.time()
    n=1 if smoke else PER_SKU;pps=max(1,procs//len(SKUS));per=-(-n//pps);sha=hashlib.sha256(Path(cand).read_bytes()).hexdigest();ps=[];parts=[]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
    print(f'NOMINAL_GATE_START epoch={epoch} cand={cand}',flush=True)
    for sku in SKUS:
        for k in range(pps):
            s=SEED0+k*per;m=min(per,SEED0+n-s)
            if m<=0:break
            res=folder/f'eval_{epoch:04d}_{sku}_p{k}.json';parts.append((sku,res))
            if res.exists():continue
            ps.append(subprocess.Popen([sys.executable,str(HERE/'evaluate_bc_multisku.py'),'--checkpoint',str(cand),'--output',str(res),'--seed-start',str(s),'--episodes',str(m),'--sampler','upstream','--sku',sku],stdout=open(folder/f'eval_{epoch:04d}_{sku}_p{k}.log','a'),stderr=subprocess.STDOUT,env=env))
    for p in ps:
        if p.wait()!=0:raise RuntimeError(f'nominal gate eval failed epoch {epoch}')
    R=[(sku,json.loads(r.read_text())) for sku,r in parts]
    assert all(r['checkpoint_sha256']==sha and r['sampler']=='upstream' and r['position_half_range_m']==.005 and r['yaw_half_range_deg']==2. and r['sku']==sku and r['success_criteria_version']=='flip_pose_no_drop_v2' for sku,r in R)
    per_sku={s:dict(successes=sum(r['successes'] for k,r in R if k==s),episodes=sum(r['episodes'] for k,r in R if k==s),strict=sum(r['strict_successes'] for k,r in R if k==s),
                    failure_counts={f:sum(r['failure_counts'][f] for k,r in R if k==s) for f in R[0][1]['failure_counts']}) for s in SKUS}
    succ=sum(v['successes'] for v in per_sku.values());eps=sum(v['episodes'] for v in per_sku.values());vl=R[0][1]['validation_loss'] or 0.
    cur=dict(epoch=epoch,step=step,successes=succ,episodes=eps,validation_loss=vl,candidate=str(cand),sha256=sha,per_sku=per_sku,failure_counts={f:sum(v['failure_counts'][f] for v in per_sku.values()) for f in R[0][1]['failure_counts']})
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
    print('NOMINAL_GATE',json.dumps(dict(epoch=epoch,successes=succ,episodes=eps,per_sku={s:v['successes'] for s,v in per_sku.items()},val=vl,selected_epoch=sel['epoch'],selected_succ=sel['successes'],stale=prog['stale'],stop=stop)),flush=True)
    return d
