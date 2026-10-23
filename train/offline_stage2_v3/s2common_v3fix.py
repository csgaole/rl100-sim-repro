# [v3fix] copy of s2common_v3.py: terminal DDIM step uses final_alpha_cumprod (sigma=0, no floor); logprob std clamp min 1e-8. PPO must skip terminal step (is_terminal).
import os as _o;assert _o.environ.get('S2_FIX_FINAL')=='1' and _o.environ.get('S2_FLOOR_FINAL','0')=='0','[v3fix] need S2_FIX_FINAL=1 S2_FLOOR_FINAL=0'
"""rl100-sim-repro Stage-2 (offline RL) shared utilities.

Policy = Stage-1 diffusion BC (runs/bc_stage1/selected_development.pt). The obs encoder is FROZEN
during Stage-2 (RL-100 freezes the IL encoder within an offline round); only the UNet is updated.

Denoising kernel: `kernel()` reproduces upstream DDIMSchedulerExtended.step_mean (eta=1, prev
timestep clamped at 0, i.e. the exact 'upstream' sampler used by every Stage-1 evaluation) as the
MEAN; the stochastic policy adds N(0, clamp(sigma_eta1, .0067, S2_STD_MAX=.01)). The stochastic policy is the
behaviour policy for data collection and the likelihood model for PPO; deployment/eval is mean-only
(evaluate_bc.py --sampler upstream), identical to Stage-1.
"""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json,time,hashlib,math,glob
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np,torch,torch.nn as nn,mujoco
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]  # [v3 copy of train/offline_stage2/s2common.py]
BC_DIR=ROOT/'train/bc_stage1'
sys.path.insert(0,str(BC_DIR))
sys.path.insert(0,str(HERE))
import novib_patch  # [v3] no-VIB policy class
from evaluate_bc import load_policy  # noqa: also puts env/ + task_success on sys.path
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
from action_codec import step_policy_action
from task_success import assess_task,SUCCESS_VERSION
CAMS=('global','left_wrist','right_wrist');JOINTS=[0,1,2,3,4,5,7,8,9,10,11,12]
STAGE1=Path(os.environ['V3_START_POLICY'])  # [v3] start policy = P3 v3 e88 (name kept for code reuse)
OUT=Path(os.environ.get('S2_OUT',str(ROOT/'runs/offline_stage2')))
FEAT=1564;CHUNKS=60;STD_MIN=.0067;STD_MAX=float(os.environ.get('S2_STD_MAX','.01'));GAMMA=.99
# Never collect on reserved / dev-eval / final-eval seeds.
FORBIDDEN=[(900000,900999),(880000,880099),(910000,910099),(790000,790099)]  # last = expanded dev diagnostic

def check_collection_seeds(start,n):
    for lo,hi in FORBIDDEN:assert start+n-1<lo or start>hi,f'collection seeds {start}+{n} overlap {lo}-{hi}'
def cst(ts=None):
    return datetime.fromtimestamp(ts or time.time(),timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S CST')
def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def append_jsonl(path,obj):
    with open(path,'a') as f:f.write(json.dumps(obj,default=str)+'\n')
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<23),b''):h.update(b)
    return h.hexdigest()
def setup(seed,threads=4):
    torch.set_num_threads(threads);torch.manual_seed(seed);np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False

# ---------------- diffusion kernel ----------------
def kernel(p,eps,t,x):
    sc=p.noise_scheduler;idx=int(t);prev_raw=idx-sc.config.num_train_timesteps//sc.num_inference_steps;term=prev_raw<0
    a=sc.alphas_cumprod[idx].to(x.device,x.dtype);ap=(sc.final_alpha_cumprod if term else sc.alphas_cumprod[prev_raw]).to(x.device,x.dtype)
    clean=((x-(1-a)**.5*eps)/a**.5).clamp(-1,1)
    var=((1-ap)/(1-a)*(1-a/ap)).clamp(min=0.)
    mean=ap**.5*clean+(1-ap-var).clamp(min=0.)**.5*eps
    if term:return mean,torch.zeros_like(var)  # [fix] DDIM terminal step: sigma=0 exactly, no STD_MIN floor
    return mean,var.sqrt().clamp(STD_MIN,STD_MAX)
def is_terminal(p,t):
    sc=p.noise_scheduler;return int(t)-sc.config.num_train_timesteps//sc.num_inference_steps<0

def logprob(mean,std,y):
    # Gripper columns (6,13) are overwritten with a constant width before execution: excluded.
    return torch.distributions.Normal(mean,std.clamp(min=1e-8)).log_prob(y)[...,JOINTS].sum((1,2))

def sample_chain(model,p,cond,gen=None,stochastic=True,trace=False,noise=None):
    p.noise_scheduler.set_timesteps(10)
    x=noise.clone() if noise is not None else torch.randn((cond.shape[0],8,14),device=cond.device,generator=gen)
    xs=[];ys=[];lps=[]
    for t in p.noise_scheduler.timesteps:
        eps=model(x,t.to(cond.device),global_cond=cond);mean,std=kernel(p,eps,t,x)
        y=mean+std*torch.randn(x.shape,device=x.device,generator=gen) if stochastic else mean
        if trace:xs.append(x);ys.append(y);lps.append(logprob(mean,std,y))
        x=y
    return (x,xs,ys,lps) if trace else x

def joint_flat(x):return x[...,JOINTS].flatten(-2)

# ---------------- observation encoding ----------------
@torch.no_grad()
def encode_frames(p,frames):
    """frames: cam->(B,3,240,320) uint8, 'proprio'->(B,28). Returns (B,1564) per-frame features
    (policy global_cond = concat[prev_frame, cur_frame])."""
    obs={c:torch.as_tensor(np.ascontiguousarray(frames[c]),device='cuda').float()/255 for c in CAMS}
    obs['agent_pos']=torch.as_tensor(np.asarray(frames['proprio']),device='cuda',dtype=torch.float32)
    return p.obs_encoder(p.normalizer.normalize(obs))

def encode_obs(p,obs):
    return encode_frames(p,{**{c:obs['rgb'][c].transpose(2,0,1)[None] for c in CAMS},'proprio':obs['proprio'][None]})[0]

def unnormalize(p,x):
    a=p.normalizer['action'].unnormalize(x).cpu().numpy();a[...,[6,13]]=.012;return a

# ---------------- environment (identical init to evaluate_bc.episode) ----------------
def make_env(seed,pos,yaw):
    rng=np.random.default_rng(seed);dx,dy=rng.uniform(-pos,pos,2);yaw_deg=float(rng.uniform(-yaw,yaw))
    env=LogisticsSortingEnv(render=True,image_size=(320,240));env.reset(seed=0);e=env.physics;m,d=e.model,e.data
    adr=m.jnt_qposadr[m.joint('parcel_free').id]
    d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
    d.qpos[adr:adr+2]+=[dx,.05+dy];ang=np.pi/2+np.deg2rad(yaw_deg)
    d.qpos[adr+3:adr+7]=[0,np.cos(ang/2),np.sin(ang/2),0]
    m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
    return env,dict(seed=seed,dx=float(dx),dy=float(dy),yaw_deg=yaw_deg,position_half_range_m=pos,yaw_half_range_deg=yaw)

@torch.no_grad()
def rollout(p,model,seed,pos,yaw,stochastic=True):
    env,meta=make_env(seed,pos,yaw);e=env.physics;m,d=e.model,e.data
    gen=torch.Generator(device='cuda').manual_seed(seed+20000000)
    try:
        obs=env.observe();z=encode_obs(p,obs);prev=z;feats=[z];xs=[];stable=[];drop=False;peak=0.;bmin=[]
        for chunk in range(CHUNKS):
            cond=torch.cat([prev,z])[None];prev=z
            x=sample_chain(model,p,cond,gen,stochastic);xs.append(x[0].cpu().numpy());cb=9.
            for a in unnormalize(p,x)[0]:
                step_policy_action(e,a,p.action_semantics);s=task_state(e);v=np.zeros(6)
                mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('parcel').id,v,0)
                stable.append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                drop|=s['bottom']<.74;peak=max(peak,max(s['forces']));cb=min(cb,s['bottom'])
            bmin.append(cb);obs=env.observe();z=encode_obs(p,obs);feats.append(z)
        out=assess_task(s['label_up'],s['bottom'],drop,len(stable)==480,stable)
        meta.update({k:(v.item() if hasattr(v,'item') else v) for k,v in out.items()})
        meta.update(peak_force_N=float(peak),final_label_up=float(s['label_up']),stochastic=stochastic,success_criteria_version=SUCCESS_VERSION)
        return meta,dict(feats=torch.stack(feats).float().cpu().numpy(),actions=np.array(xs,np.float32),chunk_bottom_min=np.array(bmin,np.float32))
    finally:env.close()

# ---------------- checkpoints ----------------
def save_policy(p,cp,path,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    d=dict(ema={k:v.detach().cpu() for k,v in p.state_dict().items()},config=dict(cp['config'],stage2=meta),
           epoch=cp.get('epoch'),step=cp.get('step'),validation=cp.get('validation',{}),stage2=meta)
    tmp=path.with_name(path.name+'.tmp');torch.save(d,tmp);tmp.replace(path)

# ---------------- dataset ----------------
def load_dataset(out,rounds,device='cuda'):
    F=[];A=[];succ=[];held=[];src=[];seeds=[]
    for split in ['train','validation']:
        z=np.load(out/'data'/f'demo_{split}.npz')
        for e in range(len(z['success'])):
            F.append(z['feats'][e]);A.append(z['actions'][e]);succ.append(bool(z['success'][e]));held.append(split=='validation');src.append('demo');seeds.append(-1)
    for r in rounds:
        for f in sorted(glob.glob(str(out/f'round{r}'/'rollouts'/'*'/'ep_*.npz'))):
            z=np.load(f);meta=json.loads(str(z['meta']))
            F.append(z['feats']);A.append(z['actions']);succ.append(bool(meta['success']));held.append(meta['seed']%10==0);src.append(f'round{r}');seeds.append(meta['seed'])
    F=torch.tensor(np.stack(F),device=device,dtype=torch.float32);A=torch.tensor(np.stack(A),device=device,dtype=torch.float32)
    succ=torch.tensor(succ,device=device,dtype=torch.float32);held=torch.tensor(held,device=device)
    R=torch.zeros(len(succ),CHUNKS,device=device);R[:,-1]=succ
    return dict(F=F,A=A,AJ=joint_flat(A),R=R,success=succ,heldout=held,source=src,seeds=seeds,n=len(succ))

def feat_stats(D):
    f=D['F'][~D['heldout']].reshape(-1,FEAT)
    return dict(mu=f.mean(0),sd=f.std(0).clamp_min(1e-3))

def critic_state(D,stats,e,t):
    """Critic input: normalized [z_{t-1}, z_t] plus normalized chunk clock (critic-only input)."""
    prev=D['F'][e,(t-1).clamp(min=0)];cur=D['F'][e,t]
    clock=(t.float()/CHUNKS*2-1)[:,None]
    return torch.cat([(prev-stats['mu'])/stats['sd'],(cur-stats['mu'])/stats['sd'],clock],1)

def policy_cond(D,e,t):
    return torch.cat([D['F'][e,(t-1).clamp(min=0)],D['F'][e,t]],1)

# ---------------- critic ----------------
def mlp(i,o,h=512,depth=3):
    layers=[];d=i
    for _ in range(depth):layers+=[nn.Linear(d,h),nn.ReLU()];d=h
    layers.append(nn.Linear(d,o));return nn.Sequential(*layers)

class Critic(nn.Module):
    def __init__(self,sd,ad,h=512,depth=3):
        super().__init__();self.q1=mlp(sd+ad,1,h,depth);self.q2=mlp(sd+ad,1,h,depth);self.v=mlp(sd,1,h,depth)
    def q(self,s,a):
        x=torch.cat([s,a],1);return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)
    def qmin(self,s,a):
        q1,q2=self.q(s,a);return torch.minimum(q1,q2)
    def value(self,s):return self.v(s).squeeze(-1)

def auc(scores,labels):
    scores=np.asarray(scores,float);labels=np.asarray(labels,bool);pos=scores[labels];neg=scores[~labels]
    if len(pos)==0 or len(neg)==0:return None
    return float((pos[:,None]>neg[None,:]).mean()+.5*(pos[:,None]==neg[None,:]).mean())

def gpu_free_list(cands,max_used_mib=20000):
    import subprocess
    out=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout
    used={int(a):int(b) for a,b in (l.split(',') for l in out.strip().splitlines())}
    return [g for g in cands if used.get(g,1e9)<max_used_mib]
