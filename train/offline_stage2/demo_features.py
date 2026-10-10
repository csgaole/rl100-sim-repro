"""Encode the Stage-1 demo cache (train 240 / validation 30; test never read) with the frozen Stage-1
encoder; actions = normalized actions/effective (the BC label). Demos are read from the Stage-1 cache
(built read-only from the hard-linked H5s)."""
from s2common import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--policy',default=str(STAGE1));ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
setup(0);p,cp=load_policy(a.policy);p.eval()
CACHE=ROOT/'runs/bc_stage1/cache';DATA=ROOT/'data/demos/sku09-original300'
man={}
for line in open(DATA/'manifest.jsonl'):
    r=json.loads(line);man[r['index']]=bool(r['result']['success'])
for split in ['train','validation']:
    ids=json.loads((DATA/'splits.json').read_text())[split]
    assert json.loads((CACHE/f'{split}_complete.json').read_text())['ids']==ids
    if a.limit:ids=ids[:a.limit]
    rgb=np.load(CACHE/f'{split}_rgb.npy',mmap_mode='r');state=np.load(CACHE/f'{split}_state.npy');action=np.load(CACHE/f'{split}_action.npy')
    F=np.zeros((len(ids),61,FEAT),np.float32)
    for e in range(len(ids)):
        F[e]=encode_frames(p,{**{c:rgb[e,:,k] for k,c in enumerate(CAMS)},'proprio':state[e]}).cpu().numpy()
        if e%30==0:print('DEMO',split,e,len(ids),flush=True)
    with torch.no_grad():A=p.normalizer['action'].normalize(torch.from_numpy(action[:len(ids)]).cuda()).cpu().numpy()
    succ=np.array([man[i] for i in ids])
    (OUT/'data').mkdir(parents=True,exist_ok=True);tmp=OUT/'data'/f'demo_{split}.tmp.npz'
    np.savez(tmp,feats=F,actions=A.astype(np.float32),success=succ,ids=np.array(ids));tmp.replace(OUT/'data'/f'demo_{split}.npz')
    print('DEMO_DONE',split,len(ids),'success',int(succ.sum()),flush=True)
