"""Critic-v2 ablation helper (read-only on all data): recover per-env-step task state for every episode used
by the wide IQL critic, so the outcome-decision step can be located.
 * rollouts (ep_*.npz store only frozen features + normalized action chunks, no per-step info): deterministic
   physics-only REPLAY of the stored actions from the stored init pose (identical init to batch_worker.init_env /
   s2common.make_env; actions unnormalized on GPU with the episode's own policy normalizer exactly as at
   collection; gripper cols = .012). Replay fidelity is verified against the stored terminal metadata
   (final_label_up, final_bottom_m, success, peak_force_N).
 * demos: per-step diagnostics/label_up, diagnostics/bottom, actions/effective read from the source H5 (read-only).
Output: one npz per source episode list with label_up/bottom/stable (E,480) and executed actions (E,480,14).
Usage: python wide_events.py --sources SOURCES.json --out DIR [--procs 32]"""
import os,sys,json,time,argparse,glob
os.environ.setdefault('MUJOCO_GL','egl');os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'train/bc_stage1'))
import numpy as np,torch,mujoco
from evaluate_bc import load_policy,LogisticsSortingEnv,task_state,step_policy_action,assess_task
STAGE1=str(ROOT/'runs/bc_stage1/selected_development.pt')
_SEM=None
def init_env(ep):
    env=LogisticsSortingEnv(render=False);env.reset(seed=0);e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]+=[ep['dx'],.05+ep['dy']];yw=np.pi/2+np.deg2rad(ep['yaw_deg'])
    d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
    return env
def replay(args):
    meta,acts,sem=args
    env=init_env(meta);e=env.physics;m,d=e.model,e.data;body=m.body('parcel').id
    up=np.zeros(480);bot=np.zeros(480);stab=np.zeros(480,bool);drop=False;peak=0.;k=0
    try:
        for c in range(60):
            for aa in acts[c]:
                step_policy_action(e,aa,sem);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                stab[k]=s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15
                up[k]=s['label_up'];bot[k]=s['bottom'];drop|=s['bottom']<.74;peak=max(peak,max(s['forces']));k+=1
            env.observe()
        out=assess_task(up[-1],bot[-1],drop,True,list(stab))
        return up,bot,stab,dict(success=bool(out['success']),final_label_up=float(up[-1]),final_bottom_m=float(bot[-1]),peak_force_N=float(peak))
    finally:env.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--out',required=True);ap.add_argument('--procs',type=int,default=32);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);sources=json.loads(Path(a.sources).read_text())
    pol={}
    def getp(path):
        if path not in pol:p,_=load_policy(path);pol[path]=p
        return pol[path]
    import multiprocessing as mp
    for kind,path in sources:
        dst=out/f'{kind}.npz'
        if dst.exists():print('skip',kind,flush=True);continue
        t0=time.time()
        if kind=='demo':
            import h5py
            DATA=ROOT/'data/demos/sku09-original300';man={}
            for line in open(DATA/'manifest.jsonl'):r=json.loads(line);man[r['index']]=r
            UP=[];BOT=[];ST=[];ACT=[];keys=[];succ=[]
            for split in ['train','validation']:
                z=np.load(Path(path)/f'demo_{split}.npz')
                for i in z['ids']:
                    r=man[int(i)]
                    with h5py.File(DATA/'episodes'/Path(r['path']).name,'r') as f:
                        UP.append(f['diagnostics/label_up'][:]);BOT.append(f['diagnostics/bottom'][:]);ST.append(f['diagnostics/stable'][:]);ACT.append(f['actions/effective'][:].astype(np.float32))
                    keys.append(f'demo_{split}:{int(i)}');succ.append(bool(r['result']['success']))
                assert [bool(x) for x in z['success']]==succ[-len(z['ids']):]
            np.savez(dst,keys=np.array(keys),label_up=np.stack(UP),bottom=np.stack(BOT),stable=np.stack(ST),actions=np.stack(ACT),
                     success_meta=np.array(succ),success_replay=np.array(succ),match=np.ones(len(keys),bool))
            print('DEMO',len(keys),f'{time.time()-t0:.0f}s',flush=True);continue
        files=[f for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)) if '.tmp.' not in f]
        jobs=[];metas=[]
        for f in files:
            z=np.load(f);meta=json.loads(str(z['meta']));p=getp(meta.get('policy',STAGE1))
            with torch.no_grad():acts=p.normalizer['action'].unnormalize(torch.as_tensor(z['actions'],device='cuda')).cpu().numpy()
            acts[...,[6,13]]=.012;jobs.append((meta,acts,p.action_semantics));metas.append(meta)
        print(kind,'loaded',len(jobs),f'{time.time()-t0:.0f}s',flush=True)
        with mp.get_context('fork').Pool(a.procs) as pool:res=pool.map(replay,jobs,chunksize=4)
        UP=np.stack([r[0] for r in res]).astype(np.float32);BOT=np.stack([r[1] for r in res]).astype(np.float32);ST=np.stack([r[2] for r in res])
        sm=np.array([bool(m['success']) for m in metas]);sr=np.array([r[3]['success'] for r in res])
        dup=np.array([abs(r[3]['final_label_up']-m['final_label_up']) for r,m in zip(res,metas)]);dbo=np.array([abs(r[3]['final_bottom_m']-m['final_bottom_m']) for r,m in zip(res,metas)])
        dpk=np.array([abs(r[3]['peak_force_N']-m['peak_force_N']) for r,m in zip(res,metas)])
        match=(sm==sr)&(dup<1e-6)&(dbo<1e-6)
        ACT=np.stack([j[1].reshape(480,14) for j in jobs]).astype(np.float32)
        np.savez(dst,keys=np.array(files),label_up=UP,bottom=BOT,stable=ST,actions=ACT,success_meta=sm,success_replay=sr,match=match,d_up=dup,d_bottom=dbo,d_peak=dpk)
        print('ROLL',kind,len(files),'success_match',float((sm==sr).mean()),'exact_terminal_match',float(match.mean()),'max_dup',dup.max(),'max_dbot',dbo.max(),'max_dpeak',dpk.max(),f'{time.time()-t0:.0f}s',flush=True)
    print('EVENTS_DONE',flush=True)
