# [v3fix] wcommon importing s2common_v3fix
# [offline_stage2_v3 copy of train/offline_stage2_wide/wcommon.py]
"""offline_stage2_wide shared utilities: reuses train/offline_stage2/s2common (diffusion kernel, critic,
feature conventions) with a separate output dir and a wide-range dataset loader."""
import os,sys
from pathlib import Path
_ROOT=Path(__file__).resolve().parents[2]
assert 'S2_OUT' in os.environ,'[v3] set S2_OUT to the new run dir'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from s2common_v3fix import *  # noqa
OLD=ROOT/'runs/offline_stage2'            # read-only: demo features + round-1 rollouts of the first Stage-2
DEMO_DIR=OUT/'data'  # [v3] demo features re-encoded with the v3 encoder (demo_features_v3.py)
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
                z=np.load(Path(path)/f'demo_{split}.npz');zf=z['feats'];za=z['actions'];zs=z['success']  # [v3] load arrays once (per-episode z[...] re-read whole array, O(N^2) + huge RSS)
                for e in range(len(zs)):
                    F.append(zf[e].copy());A.append(za[e].copy());succ.append(bool(zs[e]));held.append(split=='validation');src.append('demo');seeds.append(-1);wide.append(False);nom.append(False)
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
