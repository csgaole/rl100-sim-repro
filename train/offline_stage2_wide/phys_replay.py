"""Physical-state probe, step 1 (read-only on all data): exact physics-only REPLAY of stored wide-range rollouts
(same mechanism as wide_events.py, verified exact) logging privileged sim state at EVERY chunk boundary
c=0..60 (state before executing chunk c; aligned with stored BC feature feats[c]).
Logged per boundary (names in 'names'): parcel pos/rotmat/label_up/bottom/vel, arm joint q/qvel (incl. gripper),
TCP pos/rotmat, TCP-parcel offsets, contact forces/counts by partner (gripper/other link per arm, table/infeed/
outfeed/floor), outfeed flags, running aggregates (max label_up, min bottom, peak gripper forces, contact steps).
Usage: python phys_replay.py --sources SOURCES.json --kinds wide_round1,wide_round2,v2_round1 --out runs/phys_probe/phys.npz"""
import os,sys,json,time,argparse,glob
os.environ.setdefault('MUJOCO_GL','egl');os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'train/bc_stage1'));sys.path.insert(0,str(HERE))
import numpy as np,torch,mujoco
from evaluate_bc import load_policy,task_state,step_policy_action,assess_task
from wide_events import init_env
STAGE1=str(ROOT/'runs/bc_stage1/selected_development.pt')
PARTNERS=['arm0_grip','arm0_other','arm1_grip','arm1_other','transfer_table','infeed','outfeed','floor']
def names():
    n=[f'parcel_pos_{a}' for a in 'xyz']+[f'parcel_R{i}{j}' for i in range(3) for j in range(3)]+['label_up','bottom']
    n+=[f'parcel_angvel_{a}' for a in 'xyz']+[f'parcel_linvel_{a}' for a in 'xyz']
    n+=[f'q_{i}' for i in range(16)]+[f'qvel_{i}' for i in range(16)]
    for k in range(2):n+=[f'tcp{k}_pos_{a}' for a in 'xyz']+[f'tcp{k}_R{i}{j}' for i in range(3) for j in range(3)]+[f'tcp{k}_minus_parcel_{a}' for a in 'xyz']
    n+=[f'cforce_{p}' for p in PARTNERS]+[f'ccount_{p}' for p in PARTNERS]+['outfeed_contact','outfeed_inside']
    n+=['run_max_label_up','run_min_bottom','run_peak_force_arm0','run_peak_force_arm1','run_contact_steps_arm0','run_contact_steps_arm1','run_drop']
    return n
NAMES=names()
def snapshot(e,agg):
    m,d=e.model,e.data;b=m.body('parcel').id;s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,b,v,0)
    x=list(d.xpos[b])+list(d.xmat[b])+[s['label_up'],s['bottom']]+list(v)+list(d.qpos[:16])+list(d.qvel[:16])
    for k in range(2):
        sid=m.site(f'arm{k}_tcp').id;x+=list(d.site_xpos[sid])+list(d.site_xmat[sid])+list(d.site_xpos[sid]-d.xpos[b])
    cf,cc=contacts(m,d);x+=cf+cc+[float(s['outfeed_contact']),float(s['outfeed_inside'])]+[agg[k] for k in ['maxup','minbot','pk0','pk1','cs0','cs1','drop']]
    return np.array(x,np.float32)
def contacts(m,d):
    cf=dict.fromkeys(PARTNERS,0.);cc=dict.fromkeys(PARTNERS,0.)
    for ci in range(d.ncon):
        c=d.contact[ci];nm=[m.geom(int(g)).name or '' for g in c.geom]
        if 'parcel_box' not in nm:continue
        o=[x for x in nm if x!='parcel_box'];o=o[0] if o else ''
        if o.startswith('arm'):p=o[:4]+('_grip' if 'gripper_link' in o else '_other')
        elif o in PARTNERS:p=o
        else:continue
        f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f);cf[p]+=max(0.,float(f[0]));cc[p]+=1
    return [cf[p] for p in PARTNERS],[cc[p] for p in PARTNERS]
def replay(args):
    meta,acts,sem=args
    env=init_env(meta);e=env.physics;m,d=e.model,e.data;body=m.body('parcel').id
    s=task_state(e);agg=dict(maxup=s['label_up'],minbot=s['bottom'],pk0=0.,pk1=0.,cs0=0.,cs1=0.,drop=0.)
    P=[snapshot(e,agg)];stab=[];drop=False;up=bot=None
    try:
        for c in range(60):
            for aa in acts[c]:
                step_policy_action(e,aa,sem);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                stab.append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                drop|=s['bottom']<.74;up,bot=s['label_up'],s['bottom']
                agg['maxup']=max(agg['maxup'],up);agg['minbot']=min(agg['minbot'],bot);agg['pk0']=max(agg['pk0'],s['forces'][0]);agg['pk1']=max(agg['pk1'],s['forces'][1])
                agg['cs0']+=s['forces'][0]>.5;agg['cs1']+=s['forces'][1]>.5;agg['drop']=float(drop)
            env.observe();P.append(snapshot(e,agg))
        out=assess_task(up,bot,drop,True,stab)
        return np.stack(P),bool(out['success']),str(out.get('reason','')),float(up),float(bot)
    finally:env.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--kinds',default='wide_round1,wide_round2,v2_round1')
    ap.add_argument('--out',required=True);ap.add_argument('--procs',type=int,default=48);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    t0=time.time();src=dict((k,p) for k,p in json.loads(Path(a.sources).read_text()));pol={}
    def getp(path):
        if path not in pol:p,_=load_policy(path);pol[path]=p
        return pol[path]
    jobs=[];rows=[]
    for kind in a.kinds.split(','):
        files=[f for f in sorted(glob.glob(str(Path(src[kind])/'**'/'ep_*.npz'),recursive=True)) if '.tmp.' not in f]
        if a.limit:files=files[:a.limit]
        for f in files:
            z=np.load(f);meta=json.loads(str(z['meta']));p=getp(meta.get('policy',STAGE1))
            with torch.no_grad():acts=p.normalizer['action'].unnormalize(torch.as_tensor(z['actions'],device='cuda')).cpu().numpy()
            acts[...,[6,13]]=.012;jobs.append((meta,acts,p.action_semantics));rows.append(dict(file=f,source=kind,**{k:meta.get(k) for k in ['seed','dx','dy','yaw_deg','cell','pos_half_mm','yaw_half_deg','sampler','success','reason','policy','final_label_up','final_bottom_m']},wide=bool(meta.get('wide',True)),nominal=bool(meta.get('nominal',False))))
    print('loaded',len(jobs),f'{time.time()-t0:.0f}s',flush=True)
    import multiprocessing as mp
    with mp.get_context('fork').Pool(a.procs) as pool:res=pool.map(replay,jobs,chunksize=4)
    PH=np.stack([r[0] for r in res]);sr=np.array([r[1] for r in res]);sm=np.array([bool(r['success']) for r in rows])
    dup=np.array([abs(r[3]-m['final_label_up']) for r,m in zip(res,rows)]);dbo=np.array([abs(r[4]-m['final_bottom_m']) for r,m in zip(res,rows)])
    match=(sr==sm)&(dup<1e-6)&(dbo<1e-6)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez(a.out,phys=PH,names=np.array(NAMES),rows=np.array(json.dumps(rows)),success_replay=sr,match=match)
    print('REPLAY_DONE',len(rows),'phys',PH.shape,'success_match',float((sr==sm).mean()),'exact_terminal_match',float(match.mean()),f'{time.time()-t0:.0f}s',flush=True)
