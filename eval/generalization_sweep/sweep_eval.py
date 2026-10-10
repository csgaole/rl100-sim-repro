"""Stage-1 generalization sweep. Episode logic is a line-for-line mirror of
train/bc_stage1/evaluate_bc.episode (same init, same evaluate_bc.sample(eta=1.0) upstream DDIM10
deployment sampler, same generator seed+10000000, same assess_task) plus optional single-factor
perturbations applied after the standard init, and per-chunk privileged diagnostics (evaluation only).
Paired design: every cell uses seeds 600000..600099 (initial pose = same uniform draw scaled by range).
Usage: python sweep_eval.py launch   |   python sweep_eval.py worker --wid i
Batched lock-step workers (25 envs/process); verified against evaluate_bc on Stage-1 dev seeds."""
import os,sys,json,time,argparse,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=Path(os.environ.get('SWEEP_OUT',str(ROOT/'runs/generalization_sweep')))
CKPT=Path(os.environ.get('SWEEP_CKPT',str(ROOT/'runs/bc_stage1/selected_development.pt')))
SEED0=int(os.environ.get('SWEEP_SEED0','600000'));NEP=int(os.environ.get('SWEEP_NEP','100'));SHARD=int(os.environ.get('SWEEP_SHARD','25'))
POS=[5,10,20,40,60,80];YAW=[2,5,10,20,30,45]
PROBES=[('mass',.5),('mass',1.5),('mass',2.),('friction',.5),('friction',2.),('size',.9),('size',1.1),
        ('arm_init',.015),('arm_init',.05),('light',.4),('light',1.6),('cam_shift',.02),('cam_shift',.05),('gain',1600.)]
def cells():
    C=[dict(name=f'p{p}_y{y}',pos_mm=p,yaw_deg=y,factor='none',value=None) for p in POS for y in YAW]
    C+=[dict(name=f'probe_{f}_{v}',pos_mm=5,yaw_deg=2,factor=f,value=v) for f,v in PROBES]
    only=os.environ.get('SWEEP_CELLS')
    return [c for c in C if c['name'] in only.split(',')] if only else C
def jobs():
    J=[]
    for c in cells():
        for s in range(SEED0,SEED0+NEP,SHARD):J.append(dict(cell=c,seed_start=s,count=min(SHARD,SEED0+NEP-s)))
    # nominal + axes first so partial results are informative
    pri=lambda j:(0 if j['cell']['factor']=='none' and (j['cell']['pos_mm']==5 or j['cell']['yaw_deg']==2 or POS.index(j['cell']['pos_mm'])==YAW.index(j['cell']['yaw_deg'])) else 1 if j['cell']['factor']=='none' else 2)
    return sorted(J,key=pri)

def apply_pert(e,factor,value,seed):
    import numpy as np,mujoco
    m,d=e.model,e.data;b=m.body('parcel').id;g=m.geom('parcel_box').id
    if factor=='mass':m.body_mass[b]*=value;m.body_inertia[b]*=value
    elif factor=='friction':m.geom_friction[g,0]*=value
    elif factor=='size':
        adr=m.jnt_qposadr[m.joint('parcel_free').id];dz=(value-1)*m.geom_size[g,2];m.geom_size[g]*=value;d.qpos[adr+2]+=dz
        m.geom_rbound[g]*=value
    elif factor=='arm_init':
        q=d.qpos[e.qidx]+np.random.default_rng(seed+1).uniform(-value,value,e.qidx.shape);d.qpos[e.qidx]=q;d.ctrl[e.aidx]=q;e.command=q.copy()
    elif factor=='light':m.light_diffuse[:]*=value
    elif factor=='cam_shift':m.cam_pos[m.camera('global').id]+=np.array([value,0.,0.])
    elif factor=='gain':m.actuator_gainprm[e.aidx.ravel(),0]=value;m.actuator_biasprm[e.aidx.ravel(),1]=-value
    elif factor!='none':raise ValueError(factor)
    mujoco.mj_forward(m,d)

def worker(wid):
    """Batched lock-step evaluation: B envs per process, one batched encoder/UNet call per chunk.
    Per-env generator streams are consumed exactly as evaluate_bc.sample does (x_T draw + 10 discarded
    step_mean variance draws per chunk), so each env sees the same initial noises as evaluate_bc."""
    os.environ.setdefault('MUJOCO_GL','egl')
    sys.path.insert(0,str(ROOT/'train/bc_stage1'))
    import numpy as np,torch,mujoco
    from evaluate_bc import load_policy,LogisticsSortingEnv,task_state,step_policy_action,assess_task
    torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p,cp=load_policy(CKPT);J=json.loads((OUT/'jobs.json').read_text());CAMS=p.rgb_obs_keys
    def init_env(seed,pos,yaw,factor,value):
        rng=np.random.default_rng(seed);dx,dy=rng.uniform(-pos,pos,2);yaw_deg=float(rng.uniform(-yaw,yaw))
        env=LogisticsSortingEnv(render=True,image_size=(320,240))
        env.reset(seed=0);e=env.physics;m,d=e.model,e.data
        adr=m.jnt_qposadr[m.joint('parcel_free').id]
        d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
        d.qpos[adr:adr+2]+=[dx,.05+dy];yw=np.pi/2+np.deg2rad(yaw_deg)
        d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0]
        m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
        apply_pert(e,factor,value,seed)
        return env,dict(seed=seed,dx=float(dx),dy=float(dy),yaw_deg=yaw_deg)
    @torch.inference_mode()
    def run_batch(seeds,pos,yaw,factor,value):
        E=[init_env(s,pos,yaw,factor,value) for s in seeds];N=len(E)
        try:
            obs=[env.observe() for env,_ in E];hist=[[o,o] for o in obs]
            gens=[torch.Generator(device='cuda').manual_seed(s+10000000) for s in seeds]
            st=[dict(stable=[],drop=False,peak=0.,diag=dict(up=[],bottom=[],x=[],y=[],z=[],yawp=[],f0=[],f1=[],tcp0=[],tcp1=[]),s=None) for _ in E]
            for chunk in range(60):
                o={k:torch.as_tensor(np.stack([np.stack([h['rgb'][k] for h in H]) for H in hist]).reshape(N*2,240,320,3).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255 for k in CAMS}
                o['agent_pos']=torch.as_tensor(np.stack([np.stack([h['proprio'] for h in H]) for H in hist]).reshape(N*2,28),device='cuda',dtype=torch.float32)
                cond=p.obs_encoder(p.normalizer.normalize(o)).reshape(N,-1)
                x=torch.cat([torch.randn((1,8,14),device='cuda',generator=g) for g in gens])
                p.noise_scheduler.set_timesteps(10)
                for t in p.noise_scheduler.timesteps:
                    eps=p.model(x,t.to('cuda'),global_cond=cond)
                    x=p.noise_scheduler.step_mean(eps,t,x,eta=1.0,variance_noise=torch.zeros_like(x)).prev_sample
                    for g in gens:torch.randn((1,8,14),device='cuda',generator=g)
                acts=p.normalizer['action'].unnormalize(x).cpu().numpy();acts[:,:,[6,13]]=.012
                assert np.isfinite(acts).all()
                for i,(env,meta) in enumerate(E):
                    e=env.physics;m,d=e.model,e.data;body=m.body('parcel').id;S=st[i]
                    for a in acts[i]:
                        step_policy_action(e,a,p.action_semantics);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                        S['stable'].append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                        S['drop']|=s['bottom']<.74;S['peak']=max(S['peak'],max(s['forces']))
                    S['s']=s;R=d.xmat[body].reshape(3,3);D=S['diag']
                    D['up'].append(round(s['label_up'],4));D['bottom'].append(round(s['bottom'],4))
                    for k,val in zip('xyz',s['position']):D[k].append(round(val,4))
                    D['yawp'].append(round(float(np.degrees(np.arctan2(R[1,0],R[0,0]))),2));D['f0'].append(round(s['forces'][0],1));D['f1'].append(round(s['forces'][1],1))
                    for arm in range(2):D[f'tcp{arm}'].append([round(float(z),4) for z in d.site_xpos[m.site(f'arm{arm}_tcp').id]])
                    ob=env.observe();hist[i]=[hist[i][-1],ob]
            rows=[]
            for (env,meta),S in zip(E,st):
                s=S['s'];out=assess_task(s['label_up'],s['bottom'],S['drop'],len(S['stable'])==480,S['stable'])
                out={k:(v.item() if hasattr(v,'item') else v) for k,v in out.items()}
                rows.append(dict(meta,**out,peak_force_N=float(S['peak']),final_up=float(s['label_up']),diag=S['diag']))
            return rows
        finally:
            for env,_ in E:env.close()
    (OUT/'claims').mkdir(exist_ok=True);(OUT/'done').mkdir(exist_ok=True);(OUT/'cells').mkdir(exist_ok=True)
    for k,j in enumerate(J):
        if (OUT/'done'/f'{k}').exists():continue
        try:fd=os.open(OUT/'claims'/f'{k}',os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(wid).encode());os.close(fd)
        except FileExistsError:continue
        c=j['cell'];t=time.time()
        rows=run_batch(list(range(j['seed_start'],j['seed_start']+j['count'])),c['pos_mm']/1000,float(c['yaw_deg']),c['factor'],c['value'])
        for r in rows:r.update(cell=c['name'],pos_mm=c['pos_mm'],yaw_half_deg=c['yaw_deg'],factor=c['factor'],value=c['value'],batch_wall_s=round(time.time()-t,1))
        with open(OUT/'cells'/f"{c['name']}.jsonl",'a') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        (OUT/'done'/f'{k}').write_text(str(wid));print(time.strftime('%H:%M:%S'),'job',k,c['name'],sum(r['success'] for r in rows),'/',len(rows),f'{time.time()-t:.0f}s',flush=True)
    print('WORKER_DONE',flush=True)

def launch(gpus,per_gpu):
    import subprocess
    OUT.mkdir(parents=True,exist_ok=True);jp=OUT/'jobs.json'
    if jp.exists():assert json.loads(jp.read_text())==jobs(),'job list changed'
    else:jp.write_text(json.dumps(jobs()))
    # release stale claims (claimed but not done) from a previous interrupted launch
    if (OUT/'claims').exists():
        for f in (OUT/'claims').iterdir():
            if not (OUT/'done'/f.name).exists():f.unlink()
    (OUT/'logs').mkdir(exist_ok=True);pids=[]
    for i,g in enumerate([g for g in gpus for _ in range(per_gpu)]):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),MUJOCO_EGL_DEVICE_ID=str(g),MUJOCO_GL='egl',OMP_NUM_THREADS='4')
        pr=subprocess.Popen([sys.executable,'-u',__file__,'worker','--wid',str(i)],env=env,stdout=open(OUT/'logs'/f'worker_{i}.log','a'),stderr=subprocess.STDOUT,start_new_session=True)
        pids.append(pr.pid)
    (OUT/'workers.json').write_text(json.dumps(dict(pids=pids,gpus=gpus,per_gpu=per_gpu,launched=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),jobs=len(jobs()))))
    print('launched',pids)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['launch','worker']);ap.add_argument('--wid',type=int,default=0)
    ap.add_argument('--gpus',default='5,6,7');ap.add_argument('--per-gpu',type=int,default=2);a=ap.parse_args()
    if a.mode=='launch':launch([int(x) for x in a.gpus.split(',')],a.per_gpu)
    else:worker(a.wid)
