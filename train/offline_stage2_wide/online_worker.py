"""Online-RL rollout worker (sim, on-policy). Lock-step batches of envs (init identical to batch_worker /
evaluate_bc), stochastic diffusion policy = s2common kernel (upstream step_mean mean + N(0, clamp(sigma,.0067,
S2_STD_MAX))) with K DDIM steps; records the full denoising trace (x_in, y_out, logp per step) for PPO.
Event-based termination: drop (bottom<.74, exact: any drop => failure) immediately; early wrong_face at chunk
boundary c>=20 if label_up<0, parcel at rest (|v|<.02, |w|<.2) and no arm-parcel contact (validated on 5625 replayed
wide episodes: 0/3854 successes would be cut, 82% of wrong_face cut at mean chunk 39); otherwise 60 chunks and
assess_task. Success terminal chunk T = k*//8, k* = first step after which (label_up>.98 & bottom>.79) holds to the end.
Usage: python online_worker.py --jobs JOBS.json --wid i  (same claim protocol as batch_worker.py; JOBS: policy, K, out, jobs[{id,episodes}])
Env init / lock-step batching / encoding mirror batch_worker.py (the verified collection+gate worker), which is reused unchanged for all evaluation."""
import os,sys,json,time,argparse
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'train/bc_stage1'));sys.path.insert(0,str(ROOT/'train/offline_stage2'))
import numpy as np,torch,mujoco
from evaluate_bc import load_policy,LogisticsSortingEnv,task_state,step_policy_action,assess_task
from s2common import kernel,logprob
ap=argparse.ArgumentParser();ap.add_argument('--jobs',required=True);ap.add_argument('--wid',type=int,default=0);a=ap.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
J=json.loads(Path(a.jobs).read_text());p,cp=load_policy(J['policy']);p.eval();CAMS=p.rgb_obs_keys;K=int(J['K'])
WF_MIN_CHUNK=int(J.get('wf_min_chunk',20))
def init_env(ep):
    env=LogisticsSortingEnv(render=True,image_size=(320,240));env.reset(seed=0);e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]+=[ep['dx'],.05+ep['dy']];yw=np.pi/2+np.deg2rad(ep['yaw_deg'])
    d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
    return env
def encode(hist):
    N=len(hist)
    o={k:torch.as_tensor(np.stack([np.stack([h['rgb'][k] for h in H]) for H in hist]).reshape(N*2,240,320,3).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255 for k in CAMS}
    o['agent_pos']=torch.as_tensor(np.stack([np.stack([h['proprio'] for h in H]) for H in hist]).reshape(N*2,28),device='cuda',dtype=torch.float32)
    return p.obs_encoder(p.normalizer.normalize(o)).reshape(N,2,-1)
def arm_contact(m,d):
    for ci in range(d.ncon):
        c=d.contact[ci];nm=[m.geom(int(g)).name or '' for g in c.geom]
        if 'parcel_box' in nm and any(x.startswith('arm') for x in nm):return True
    return False
@torch.inference_mode()
def run_batch(eps):
    E=[init_env(ep) for ep in eps];N=len(E)
    try:
        hist=[[o,o] for o in (env.observe() for env in E)]
        gens=[torch.Generator(device='cuda').manual_seed(int(ep['seed'])+30000000) for ep in eps]
        S=[dict(up=[],bot=[],stable=[],drop=False,peak=0.,alive=True,reason=None,chunks=0,cond=[],xs=[],ys=[],lp=[]) for _ in E]
        p.noise_scheduler.set_timesteps(K);TS=p.noise_scheduler.timesteps
        for chunk in range(60):
            act=[i for i in range(N) if S[i]['alive']]
            if not act:break
            f=encode([hist[i] for i in act]);cond=f.reshape(len(act),-1)
            x=torch.cat([torch.randn((1,8,14),device='cuda',generator=gens[i]) for i in act]);xs=[];ys=[];lps=[]
            for t in TS:
                eps_=p.model(x,t.to('cuda'),global_cond=cond);mean,std=kernel(p,eps_,t,x)
                y=mean+std*torch.cat([torch.randn((1,8,14),device='cuda',generator=gens[i]) for i in act])
                xs.append(x);ys.append(y);lps.append(logprob(mean,std,y));x=y
            XS=torch.stack(xs,1).cpu().numpy();YS=torch.stack(ys,1).cpu().numpy();LP=torch.stack(lps,1).cpu().numpy();C=cond.float().cpu().numpy()  # fp32: kernel std ~.007, fp16 cond shifts the mean enough to bias PPO ratios
            acts=p.normalizer['action'].unnormalize(x).cpu().numpy();acts[:,:,[6,13]]=.012;assert np.isfinite(acts).all()
            for j,i in enumerate(act):
                env=E[i];e=env.physics;m,d=e.model,e.data;body=m.body('parcel').id;s_=S[i]
                s_['cond'].append(C[j]);s_['xs'].append(XS[j]);s_['ys'].append(YS[j]);s_['lp'].append(LP[j])
                for aa in acts[j]:
                    step_policy_action(e,aa,p.action_semantics);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                    s_['stable'].append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                    s_['drop']|=s['bottom']<.74;s_['peak']=max(s_['peak'],max(s['forces']));s_['up'].append(s['label_up']);s_['bot'].append(s['bottom'])
                s_['chunks']=chunk+1;s_['last']=(s,v.copy());hist[i]=[hist[i][-1],env.observe()]
                if s_['drop']:s_['alive']=False;s_['reason']='drop'
                elif chunk+1>=WF_MIN_CHUNK and s['label_up']<0 and np.linalg.norm(v[3:])<.02 and np.linalg.norm(v[:3])<.2 and not arm_contact(m,d):
                    s_['alive']=False;s_['reason']='wrong_face_early'
        out=[]
        for ep,s_ in zip(eps,S):
            if s_['reason'] is None:
                s,_=s_['last'];r=assess_task(s['label_up'],s['bottom'],s_['drop'],len(s_['stable'])==480,s_['stable'])
                succ=bool(r['success']);reason=r['reason']
            else:succ=False;reason=s_['reason']
            up=np.array(s_['up']);bot=np.array(s_['bot']);T=s_['chunks']-1;ks=None
            if succ:
                cnd=(up>.98)&(bot>.79);bad=np.nonzero(~cnd)[0];ks=int(bad[-1])+1 if len(bad) else 0;T=min(ks//8,s_['chunks']-1)
            out.append(dict(meta=dict(ep,success=succ,reason=reason,chunks_run=s_['chunks'],T=T,k_star=ks,peak_force_N=float(s_['peak'])),
                cond=np.stack(s_['cond'][:T+1]),xs=np.stack(s_['xs'][:T+1]),ys=np.stack(s_['ys'][:T+1]),lp=np.stack(s_['lp'][:T+1])))
        return out
    finally:
        for env in E:env.close()
def save(path,res):
    L=[len(r['cond']) for r in res];tmp=str(path)+'.tmp.npz'
    np.savez(tmp,meta=np.array(json.dumps([r['meta'] for r in res])),ep_len=np.array(L),cond=np.concatenate([r['cond'] for r in res]),
         xs=np.concatenate([r['xs'] for r in res]).astype(np.float32),ys=np.concatenate([r['ys'] for r in res]).astype(np.float32),lp=np.concatenate([r['lp'] for r in res]).astype(np.float32))
    os.replace(tmp,path)
# same job protocol as batch_worker.py (jobs.json, O_EXCL claim files, done markers; resumable)
OUT=Path(J['out'])
for d in ['claims','done','data']:(OUT/d).mkdir(parents=True,exist_ok=True)
t0=time.time()
for j in J['jobs']:
    k=j['id']
    if (OUT/'done'/k).exists():continue
    try:fd=os.open(OUT/'claims'/k,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(a.wid).encode());os.close(fd)
    except FileExistsError:continue
    tb=time.time();res=run_batch(j['episodes']);save(OUT/'data'/f'{k}.npz',res)
    (OUT/'done'/k).write_text(str(a.wid));print(time.strftime('%H:%M:%S'),'job',k,sum(r['meta']['success'] for r in res),'/',len(res),f'{time.time()-tb:.0f}s',flush=True)
print('WORKER_DONE',f'{time.time()-t0:.0f}s',flush=True)
