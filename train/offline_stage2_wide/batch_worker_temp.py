"""[batch_worker_temp.py = batch_worker.py + noise temperature for the det sampler, nothing else changed]
Temperature (as /data02/kemove/rl100-temp-sweep/temp_sweep.py): x_T <- T * x_T, same generator draw, so the
generator stream (and pairing with T=1.0 results) is unchanged; T=1.0 is bit-identical to batch_worker.py; T=0 -> x_T=0.
JOBS.json field "temperature" (default 1.0).
Batched lock-step rollout worker (evaluation AND data collection) for offline_stage2_wide.
Episode logic mirrors train/bc_stage1/evaluate_bc.episode exactly (same init given (dx,dy,yaw), gains 1200,
assess_task / flip_pose_no_drop_v2). The 'det' sampler is the deployment sampler (upstream DDIM10 eta=1
step_mean, generator seed+10000000, x_T draw + 10 discarded variance draws per chunk) and is the code
verified seed-for-seed against evaluate_bc in eval/generalization_sweep. The 'sto' sampler is the Stage-2
behaviour/likelihood policy of s2common (same mean, + N(0, clamp(sigma,.0067,S2_STD_MAX)), generator seed+20000000).
Collection jobs also save per-frame frozen-encoder features (61x1564) and normalized action chunks (60x8x14).
Usage: python batch_worker.py --jobs JOBS.json --wid i   (claims jobs via O_EXCL files; resumable)"""
import os,sys,json,time,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'train/bc_stage1'));sys.path.insert(0,str(ROOT/'train/offline_stage2'))
import numpy as np,torch,mujoco
from evaluate_bc import load_policy,LogisticsSortingEnv,task_state,step_policy_action,assess_task
from s2common import kernel
ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--wid',type=int,default=0);a=ap.parse_args()
torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
JF=json.loads(Path(a.jobs).read_text());TEMP=float(JF.get('temperature',1.0));OUT=Path(JF['out']);p,cp=load_policy(JF['policy']);p.eval();CAMS=p.rgb_obs_keys
def init_env(ep):
    env=LogisticsSortingEnv(render=True,image_size=(320,240));env.reset(seed=0);e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]+=[ep['dx'],.05+ep['dy']];yw=np.pi/2+np.deg2rad(ep['yaw_deg'])
    d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
    return env
def encode(hist,N):
    o={k:torch.as_tensor(np.stack([np.stack([h['rgb'][k] for h in H]) for H in hist]).reshape(N*2,240,320,3).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255 for k in CAMS}
    o['agent_pos']=torch.as_tensor(np.stack([np.stack([h['proprio'] for h in H]) for H in hist]).reshape(N*2,28),device='cuda',dtype=torch.float32)
    return p.obs_encoder(p.normalizer.normalize(o))
@torch.inference_mode()
def run_batch(eps,mode,collect):
    E=[init_env(ep) for ep in eps];N=len(E)
    try:
        obs=[env.observe() for env in E];hist=[[o,o] for o in obs]
        gens=[torch.Generator(device='cuda').manual_seed(ep['seed']+(10000000 if mode=='det' else 20000000)) for ep in eps]
        st=[dict(stable=[],drop=False,peak=0.,maxup=-9.,s=None) for _ in E];feats=[];acts_n=[]
        for chunk in range(60):
            f=encode(hist,N);cond=f.reshape(N,-1);feats.append(f.reshape(N,2,-1)[:,1].float().cpu().numpy())
            x=torch.cat([torch.randn((1,8,14),device='cuda',generator=g) for g in gens])
            if mode=='det' and TEMP!=1.0:x=x*TEMP
            p.noise_scheduler.set_timesteps(10)
            for t in p.noise_scheduler.timesteps:
                eps_=p.model(x,t.to('cuda'),global_cond=cond)
                if mode=='det':
                    x=p.noise_scheduler.step_mean(eps_,t,x,eta=1.0,variance_noise=torch.zeros_like(x)).prev_sample
                    for g in gens:torch.randn((1,8,14),device='cuda',generator=g)
                else:
                    mean,std=kernel(p,eps_,t,x);x=mean+std*torch.cat([torch.randn((1,8,14),device='cuda',generator=g) for g in gens])
            if collect:acts_n.append(x.float().cpu().numpy())
            acts=p.normalizer['action'].unnormalize(x).cpu().numpy();acts[:,:,[6,13]]=.012
            assert np.isfinite(acts).all()
            for i,env in enumerate(E):
                e=env.physics;m,d=e.model,e.data;body=m.body('parcel').id;S=st[i]
                for aa in acts[i]:
                    step_policy_action(e,aa,p.action_semantics);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                    S['stable'].append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                    S['drop']|=s['bottom']<.74;S['peak']=max(S['peak'],max(s['forces']));S['maxup']=max(S['maxup'],s['label_up'])
                S['s']=s;hist[i]=[hist[i][-1],env.observe()]
        if collect:feats.append(encode(hist,N).reshape(N,2,-1)[:,1].float().cpu().numpy())
        rows=[]
        for i,(ep,S) in enumerate(zip(eps,st)):
            s=S['s'];out=assess_task(s['label_up'],s['bottom'],S['drop'],len(S['stable'])==480,S['stable'])
            out={k:(v.item() if hasattr(v,'item') else v) for k,v in out.items()}
            rr=dict(ep,**out);rr.update(sampler=mode,peak_force_N=float(S['peak']),final_up=float(s['label_up']),max_label_up=float(S['maxup']));rows.append(rr)
        if collect:
            F=np.stack(feats,1);A=np.stack(acts_n,1)
            return rows,F,A
        return rows,None,None
    finally:
        for env in E:env.close()
for d in ['claims','done','rows','eps']:(OUT/d).mkdir(parents=True,exist_ok=True)
for j in JF['jobs']:
    k=j['id']
    if (OUT/'done'/k).exists():continue
    try:fd=os.open(OUT/'claims'/k,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(a.wid).encode());os.close(fd)
    except FileExistsError:continue
    t=time.time();rows,F,A=run_batch(j['episodes'],j['mode'],j['kind']=='collect')
    for r in rows:r.update(policy=JF['policy'],batch_wall_s=round(time.time()-t,1))
    if F is not None:
        for i,r in enumerate(rows):
            f=OUT/'eps'/f"ep_{r['seed']}.npz";tmp=OUT/'eps'/f"ep_{r['seed']}.tmp.npz"
            np.savez(tmp,meta=np.array(json.dumps(r)),feats=F[i],actions=A[i]);tmp.replace(f)
    with open(OUT/'rows'/f'{k}.jsonl','w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    (OUT/'done'/k).write_text(str(a.wid));print(time.strftime('%H:%M:%S'),'job',k,j['mode'],sum(bool(r['success']) for r in rows),'/',len(rows),f'{time.time()-t:.0f}s',flush=True)
print('WORKER_DONE',flush=True)
