# [offline_multisku_v1_draft 2026-10-19] Multi-SKU shared utilities. NEW file; merges (copies of) the base pipeline's
#   offline_stage2_v3_nominal_20261017/{s2common_v3.py, s2common_v3fix.py, wcommon_fix.py} (no edits to those files).
# Changes vs base:
#  * policy loader = train/bc_multisku_v1_20261018/evaluate_bc_multisku.load_policy (strict; BCPolicy with low_dim key sku_cond,
#    p.sku_table from the checkpoint config). Encoder features are 1568-D = rgb 3x512 (sorted cams) + agent_pos 28 + sku_cond 4
#    (MultiImageObsEncoder concatenates sorted low_dim keys: agent_pos, sku_cond) -> the POLICY consumes SKU through its global cond.
#  * every episode (demo or rollout) carries its SKU name + raw sku_cond (SK, 4-D); critic_state() appends SK EXPLICITLY, so the
#    CRITIC consumes SKU conditioning as well (in addition to the normalized copy inside the features).
#  * held-out subset is an explicit per-episode flag (rollouts: meta['heldout'], set by the pipeline SKU-balanced; demos: validation split).
#    MS_TRAIN_ALL=1 (default) -> trainers use ALL episodes (rules 4/5); the held-out subset is then only an (in-sample) diagnostic.
#  * DDIM kernel identical to s2common_v3fix: eta=1 posterior sigma clamped to [STD_MIN=.0067, S2_STD_MAX=.1], terminal step sigma=0.
#  * make_env/rollout/load_dataset (unused single-SKU helpers) dropped; the rollout worker (batch_worker_ms.py) owns the env.
import os,sys,json,time,hashlib,math,glob
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
from datetime import datetime,timezone,timedelta
import numpy as np,torch,torch.nn as nn
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]          # /data02/kemove/rl100-sim-repro
MS_BC_DIR=ROOT/'train/bc_multisku_v1_20261018'
for _p in (str(MS_BC_DIR),str(HERE)):
    if _p not in sys.path:sys.path.insert(0,_p)
assert os.environ.get('S2_FIX_FINAL')=='1' and os.environ.get('S2_FLOOR_FINAL','0')=='0','need S2_FIX_FINAL=1 S2_FLOOR_FINAL=0 (terminal DDIM sigma=0)'
SKUS=['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']
CAMS=('global','left_wrist','right_wrist');JOINTS=[0,1,2,3,4,5,7,8,9,10,11,12]
START_POLICY=Path(os.environ.get('MS_START_POLICY','/data02/kemove/rl100-sim-repro/runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt'))
OUT=Path(os.environ['S2_OUT']) if 'S2_OUT' in os.environ else None
DEMO_DIR=(OUT/'data') if OUT else None
FEAT=1568;SKU_DIM=4;ACT_DIM=96;CHUNKS=60;GAMMA=.99
STD_MIN=.0067;STD_MAX=float(os.environ.get('S2_STD_MAX','0.1'))
SD=2*FEAT+1+SKU_DIM   # critic state dim: [z_{t-1}, z_t, clock, sku_cond]
TRAIN_ALL=os.environ.get('MS_TRAIN_ALL','1')=='1'
# seed blocks reserved for gates / finals / other studies: never used for collection
FORBIDDEN_W=[(600000,600099),(620000,620099),(640000,649999),(700000,799999),(790000,790099),(880000,880999),(900000,900999),(910000,910999),(920000,920099),(1441000,1442999)]  # 880xxx=select+confirm gates; 910xxx=historical BC unbiased (never used here); 920xxx=FINAL unbiased (never gated/collected)
def check_seeds(lo,hi):
    for a,b in FORBIDDEN_W:assert hi<a or lo>b,f'seeds {lo}-{hi} overlap reserved {a}-{b}'
def init_pose(seed,pos_m,yaw_deg):
    """Identical to evaluate_bc_multisku.episode: rng(seed).uniform(+-pos,2) then uniform(+-yaw) (uniform +-5mm/+-2deg)."""
    rng=np.random.default_rng(seed);dx,dy=rng.uniform(-pos_m,pos_m,2);yw=float(rng.uniform(-yaw_deg,yaw_deg))
    return float(dx),float(dy),yw
def load_policy(path):
    from evaluate_bc_multisku import load_policy as _lp   # lazy: CPU-only import tests never touch CUDA
    return _lp(path)

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

# ---------------- diffusion kernel (verbatim logic of s2common_v3fix.kernel / sample_chain) ----------------
def kernel(p,eps,t,x):
    sc=p.noise_scheduler;idx=int(t);prev_raw=idx-sc.config.num_train_timesteps//sc.num_inference_steps;term=prev_raw<0
    a=sc.alphas_cumprod[idx].to(x.device,x.dtype);ap=(sc.final_alpha_cumprod if term else sc.alphas_cumprod[prev_raw]).to(x.device,x.dtype)
    clean=((x-(1-a)**.5*eps)/a**.5).clamp(-1,1)
    var=((1-ap)/(1-a)*(1-a/ap)).clamp(min=0.)
    mean=ap**.5*clean+(1-ap-var).clamp(min=0.)**.5*eps
    if term:return mean,torch.zeros_like(var)  # DDIM terminal step: sigma=0 exactly (no floor)
    return mean,var.sqrt().clamp(STD_MIN,STD_MAX)
def is_terminal(p,t):
    sc=p.noise_scheduler;return int(t)-sc.config.num_train_timesteps//sc.num_inference_steps<0
def logprob(mean,std,y):
    # gripper columns (6,13) are overwritten with a constant before execution: excluded
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

# ---------------- observation encoding (frozen encoder; adds sku_cond) ----------------
@torch.no_grad()
def encode_frames(p,frames,sku_cond):
    """frames: cam->(B,3,240,320) uint8, 'proprio'->(B,28); sku_cond: raw (4,) or (B,4). Returns (B,FEAT)."""
    obs={c:torch.as_tensor(np.ascontiguousarray(frames[c]),device='cuda').float()/255 for c in CAMS}
    B=obs[CAMS[0]].shape[0]
    obs['agent_pos']=torch.as_tensor(np.asarray(frames['proprio']),device='cuda',dtype=torch.float32)
    sk=np.asarray(sku_cond,np.float32);sk=np.broadcast_to(sk,(B,SKU_DIM)) if sk.ndim==1 else sk
    obs['sku_cond']=torch.as_tensor(np.ascontiguousarray(sk),device='cuda',dtype=torch.float32)
    return p.obs_encoder(p.normalizer.normalize(obs))

# ---------------- dataset ----------------
def load_dataset_w(sources,device='cuda'):
    """sources: list of [kind,path]; 'demo' = dir with demo_{train,validation}.npz (demo_features_ms.py); any other kind =
    rollout dir searched recursively for ep_<seed>.npz (batch_worker_ms.py). Success AND failure rollouts are all kept."""
    F=[];A=[];succ=[];held=[];src=[];seeds=[];SK=[];SKN=[]
    for kind,path in sources:
        if kind=='demo':
            for split in ['train','validation']:
                z=np.load(Path(path)/f'demo_{split}.npz');zf=z['feats'];za=z['actions'];zs=z['success'];zk=z['sku_cond'];zn=z['sku']
                for e in range(len(zs)):
                    F.append(zf[e].copy());A.append(za[e].copy());succ.append(bool(zs[e]));held.append(split=='validation');src.append('demo');seeds.append(-1);SK.append(zk[e]);SKN.append(str(zn[e]))
        else:
            for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
                if '.tmp.' in f:continue
                z=np.load(f);meta=json.loads(str(z['meta']))
                F.append(z['feats']);A.append(z['actions']);succ.append(bool(meta['success']));held.append(bool(meta.get('heldout',meta['seed']%10==0)));src.append(kind);seeds.append(meta['seed'])
                SK.append(np.asarray(meta['sku_cond'],np.float32));SKN.append(meta['sku'])
    F=torch.tensor(np.stack(F),device=device,dtype=torch.float32);A=torch.tensor(np.stack(A),device=device,dtype=torch.float32)
    assert F.shape[-1]==FEAT and F.shape[1]==CHUNKS+1 and A.shape[1:]==(CHUNKS,8,14),(F.shape,A.shape)
    succ=torch.tensor(succ,device=device,dtype=torch.float32);held=torch.tensor(held,device=device)
    R=torch.zeros(len(succ),CHUNKS,device=device);R[:,-1]=succ   # sparse terminal success reward
    SKt=torch.tensor(np.stack(SK),device=device,dtype=torch.float32)
    return dict(F=F,A=A,AJ=joint_flat(A),R=R,success=succ,heldout=held,source=src,seeds=seeds,SK=SKt,sku=SKN,n=len(succ))
def train_mask(D):
    """MS_TRAIN_ALL=1: every episode trains (demos + all rollouts); else base behaviour (exclude held-out subset)."""
    return torch.ones_like(D['heldout']) if TRAIN_ALL else ~D['heldout']
def feat_stats(D):
    f=D['F'][train_mask(D)].reshape(-1,FEAT)
    return dict(mu=f.mean(0),sd=f.std(0).clamp_min(1e-3))
def critic_state(D,stats,e,t):
    """Critic input: normalized [z_{t-1}, z_t], chunk clock, and the episode's raw sku_cond (explicit SKU conditioning)."""
    prev=D['F'][e,(t-1).clamp(min=0)];cur=D['F'][e,t]
    clock=(t.float()/CHUNKS*2-1)[:,None]
    return torch.cat([(prev-stats['mu'])/stats['sd'],(cur-stats['mu'])/stats['sd'],clock,D['SK'][e]],1)
def policy_cond(D,e,t):
    return torch.cat([D['F'][e,(t-1).clamp(min=0)],D['F'][e,t]],1)   # features already contain normalized sku_cond

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

# ---------------- checkpoints ----------------
def save_policy(p,cp,path,meta):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    d=dict(ema={k:v.detach().cpu() for k,v in p.state_dict().items()},config=dict(cp['config'],stage2=meta),
           epoch=cp.get('epoch'),step=cp.get('step'),validation=cp.get('validation',{}),stage2=meta)
    tmp=path.with_name(path.name+'.tmp');torch.save(d,tmp);tmp.replace(path)
