"""offline_stage2_wide shared utilities: reuses train/offline_stage2/s2common (diffusion kernel, critic,
feature conventions) with a separate output dir and a wide-range dataset loader."""
import os,sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
os.environ.setdefault('S2_OUT',str(_ROOT/'runs/offline_stage2_wide'))
sys.path.insert(0,str(_ROOT/'train/offline_stage2'))
from s2common import *  # noqa
OLD=ROOT/'runs/offline_stage2'            # read-only: demo features + round-1 rollouts of the first Stage-2
DEMO_DIR=OLD/'data'
# seed blocks never used for collection here (dev / reserved / final / old diag / sweep / gate / all old stage-2)
FORBIDDEN_W=[(600000,600099),(620000,620099),(700000,799999),(880000,880099),(900000,900999),(910000,910099)]
GATE_CELLS=[(40,10),(40,30),(60,20),(60,45),(80,10),(80,45)];GATE_SEED0=620000;GATE_N=50
NOMINAL=dict(seed0=880000,n=100,pos_mm=5,yaw=2.)
def check_seeds(lo,hi):
    for a,b in FORBIDDEN_W:assert hi<a or lo>b,f'seeds {lo}-{hi} overlap forbidden {a}-{b}'
def init_pose(seed,pos_m,yaw_deg):
    """Identical to evaluate_bc.episode / sweep: rng(seed).uniform(+-pos,2) then uniform(+-yaw)."""
    rng=np.random.default_rng(seed);dx,dy=rng.uniform(-pos_m,pos_m,2);yw=float(rng.uniform(-yaw_deg,yaw_deg))
    return float(dx),float(dy),yw

def episode_files(sources):
    out=[]
    for kind,path in sources:
        if kind=='demo':out.append((kind,path))
        else:out+= [(kind,f) for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True))]
    return out

def load_dataset_w(sources,device='cuda'):
    """sources: list of [kind,path]; kind 'demo' (dir with demo_{train,validation}.npz) or a rollout tag
    (dir searched recursively for ep_<seed>.npz). wide = rollout with |init| range >= 40 mm or yaw >= 10 deg
    (round-1 of the first Stage-2 was <= 40mm/20deg; all new wide collection is wide)."""
    F=[];A=[];succ=[];held=[];src=[];seeds=[];wide=[];nom=[]
    for kind,path in sources:
        if kind=='demo':
            for split in ['train','validation']:
                z=np.load(Path(path)/f'demo_{split}.npz')
                for e in range(len(z['success'])):
                    F.append(z['feats'][e]);A.append(z['actions'][e]);succ.append(bool(z['success'][e]));held.append(split=='validation');src.append('demo');seeds.append(-1);wide.append(False);nom.append(False)
        else:
            for f in sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'),recursive=True)):
                if '.tmp.' in f:continue
                z=np.load(f);meta=json.loads(str(z['meta']))
                F.append(z['feats']);A.append(z['actions']);succ.append(bool(meta['success']));held.append(meta['seed']%10==0);src.append(kind);seeds.append(meta['seed'])
                wide.append(bool(meta.get('wide',meta.get('position_half_range_m',0)>=.04 or meta.get('yaw_half_range_deg',0)>=10)))
                nom.append(bool(meta.get('nominal',False)))  # v2: nominal-range (+-5mm/+-2deg) rollouts
    F=torch.tensor(np.stack(F),device=device,dtype=torch.float32);A=torch.tensor(np.stack(A),device=device,dtype=torch.float32)
    succ=torch.tensor(succ,device=device,dtype=torch.float32);held=torch.tensor(held,device=device);wide=torch.tensor(wide,device=device);nom=torch.tensor(nom,device=device)
    R=torch.zeros(len(succ),CHUNKS,device=device);R[:,-1]=succ
    return dict(F=F,A=A,AJ=joint_flat(A),R=R,success=succ,heldout=held,wide=wide,nominal=nom,source=src,seeds=seeds,n=len(succ))

# ---------------- online RL: encoder freeze / lr (official uni_ppo style) ----------------
def configure_online_policy(p, fix_encoder=False):
    """Keep policy in eval() (BN frozen, matches prior online RL). Always freeze normalizer.
    If fix_encoder: freeze obs_encoder (old behavior). Else: unfreeze obs_encoder for training."""
    p.eval()
    p.normalizer.requires_grad_(False)
    if fix_encoder:
        p.obs_encoder.requires_grad_(False)
    else:
        p.obs_encoder.requires_grad_(True)
    return p.model

def make_policy_optimizers(p, lr, fix_encoder=False, encoder_lr_scale=1.0):
    """Official-style Adam setup (uni_ppo.transfer2online).
    fix_encoder -> Adam(model) only.
    else if encoder_lr_scale==1.0 -> single Adam(policy.parameters()) at lr (normalizer frozen so no grads).
    else -> Adam(model) at lr + Adam(obs_encoder) at lr*scale.
    Returns (opt, enc_opt) where enc_opt is None unless two-Adam path."""
    if fix_encoder:
        return torch.optim.Adam(p.model.parameters(), lr=lr), None
    if float(encoder_lr_scale) != 1.0:
        opt = torch.optim.Adam(p.model.parameters(), lr=lr)
        enc_opt = torch.optim.Adam(p.obs_encoder.parameters(), lr=lr * float(encoder_lr_scale))
        return opt, enc_opt
    return torch.optim.Adam(p.parameters(), lr=lr), None

def policy_grad_params(p, fix_encoder=False):
    """Params to clip for the policy update (model, plus encoder when unfrozen)."""
    if fix_encoder:
        return list(p.model.parameters())
    return list(p.obs_encoder.parameters()) + list(p.model.parameters())

def set_policy_lrs(opt, enc_opt, lr, encoder_lr_scale=1.0):
    for g in opt.param_groups:
        g['lr'] = lr
    if enc_opt is not None:
        for g in enc_opt.param_groups:
            g['lr'] = lr * float(encoder_lr_scale)

def zero_policy_opts(opt, enc_opt, set_to_none=True):
    opt.zero_grad(set_to_none=set_to_none)
    if enc_opt is not None:
        enc_opt.zero_grad(set_to_none=set_to_none)

def step_policy_opts(opt, enc_opt):
    opt.step()
    if enc_opt is not None:
        enc_opt.step()


# ---------------- online RL: live re-encoding of stored rollout obs (official obs2feature style) ----------------
def pack_rollout_obs(OB,L):
    """OB: per-file dicts {rgb_<cam>:(n,H,W,3) uint8, prop:(n,28)} or None. Frame i = obs at chunk start of sample i;
    policy cond of sample i = enc([frame prev(i), frame i]) with prev(i)=i-1 (= i at episode start, as hist=[o,o])."""
    if not OB or any(o is None for o in OB):return None
    rgb={c:np.concatenate([o['rgb_'+c] for o in OB]) for c in CAMS};prop=np.concatenate([o['prop'] for o in OB]).astype(np.float32)
    Lc=np.asarray(L);st=np.concatenate([[0],np.cumsum(Lc)[:-1]]).astype(np.int64);prev=np.arange(len(prop))-1;prev[st]=st
    assert len(prop)==int(Lc.sum())
    return dict(rgb=rgb,prop=prop,prev=prev)
def encode_rollout_obs(p,R,idx):
    idx=np.asarray(idx);both=np.stack([R['prev'][idx],idx],1).reshape(-1)
    o={c:torch.from_numpy(R['rgb'][c][both]).cuda().permute(0,3,1,2).float()/255 for c in CAMS}
    o['agent_pos']=torch.from_numpy(R['prop'][both]).cuda().float()
    return p.obs_encoder(p.normalizer.normalize(o)).reshape(len(idx),-1)
def live_cond(p,R,j,enc_mb=64):
    """Current-encoder cond (no graph) as a grad-requiring leaf; PPO backward accumulates d loss/d cond into .grad,
    encoder_backward then pushes it through the encoder in enc_mb chunks (bounded activation memory)."""
    idx=j.cpu().numpy()
    with torch.no_grad():c=torch.cat([encode_rollout_obs(p,R,idx[s:s+enc_mb]) for s in range(0,len(idx),enc_mb)])
    return c.requires_grad_(True)
def encoder_backward(p,R,j,cond,enc_mb=64):
    if cond.grad is None:return 0.
    idx=j.cpu().numpy();g=cond.grad
    for s in range(0,len(idx),enc_mb):encode_rollout_obs(p,R,idx[s:s+enc_mb]).backward(g[s:s+enc_mb])
    gs=[q.grad.norm() for q in p.obs_encoder.parameters() if q.grad is not None]
    return float(torch.stack(gs).norm()) if gs else 0.
