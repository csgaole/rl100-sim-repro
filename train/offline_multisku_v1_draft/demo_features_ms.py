# [offline_multisku_v1_draft] NEW file, multi-SKU copy of offline_stage2_v3_nominal_20261017/demo_features_v3.py.
"""Encode the multi-SKU BC demos (sku5-flip-demonstrations cache bc_v1: train 400 = 80/SKU, validation 50 = 10/SKU; spare/test never read)
with the FROZEN start-policy encoder (incl. sku_cond) -> $S2_OUT/data/demo_{train,validation}.npz
fields: feats (E,61,1568) float32, actions (E,60,8,14) normalized with the policy normalizer, success (all True),
ids, sku (E,) names, sku_cond (E,4) raw [L,W,H,mass]/[.30,.25,.15,.60] (checked against the checkpoint sku_cond_table)."""
from ms_common import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--policy',default=str(START_POLICY));ap.add_argument('--limit',type=int,default=0)
ap.add_argument('--cache',default='/data02/kemove/sku5-flip-demonstrations/cache/bc_v1');a=ap.parse_args()
setup(0);p,cp=load_policy(a.policy);p.eval();C=Path(a.cache);(OUT/'data').mkdir(parents=True,exist_ok=True)
assert (OUT/'data'/'demo_train.npz').exists()==False,'refuse overwrite demo features'
EXPECT={'train':400,'validation':50}
for split in ['train','validation']:
    info=json.loads((C/f'{split}_complete.json').read_text());ids=info['ids'];assert len(ids)==EXPECT[split],(split,len(ids))
    sku_of={str(k):v for k,v in info['sku_of_id'].items()};names=[sku_of[str(i)] for i in ids]
    from collections import Counter;assert Counter(names)==Counter({s:EXPECT[split]//5 for s in SKUS}),Counter(names)
    rgb=np.load(C/f'{split}_rgb.npy',mmap_mode='r');state=np.load(C/f'{split}_state.npy');action=np.load(C/f'{split}_action.npy');sk=np.load(C/f'{split}_sku.npy')
    for e,n in enumerate(names):assert np.allclose(sk[e],p.sku_table[n]),(e,n)
    n=a.limit or len(ids);F=np.zeros((n,61,FEAT),np.float32)
    for e in range(n):
        F[e]=encode_frames(p,{**{c:np.asarray(rgb[e,:,k]) for k,c in enumerate(CAMS)},'proprio':state[e]},sk[e]).cpu().numpy()
        if e%50==0:print('DEMO',split,e,n,flush=True)
    with torch.no_grad():A=p.normalizer['action'].normalize(torch.from_numpy(action[:n]).cuda()).cpu().numpy().astype(np.float32)
    tmp=OUT/'data'/f'demo_{split}.tmp.npz'
    np.savez(tmp,feats=F,actions=A,success=np.ones(n,bool),ids=np.array(ids[:n]),sku=np.array(names[:n]),sku_cond=sk[:n].astype(np.float32))
    tmp.replace(OUT/'data'/f'demo_{split}.npz');print('DEMO_DONE',split,n,F.shape,dict(Counter(names[:n])),flush=True)
write_json(OUT/'data'/'demo_features.json',dict(policy=a.policy,policy_sha256=sha256(a.policy),cache=a.cache,feat_dim=FEAT,updated=cst()))
