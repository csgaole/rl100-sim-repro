"""Build the Stage-1 BC cache from data/demos/sku09-original300 (H5 opened READ-ONLY; demos are hard links).
Format identical to the historical sku09 cache (train_bc.py): rgb uint8 (E,61,3cam,3,240,320),
state float32 (E,61,28) = observations/proprio, action float32 (E,60,8,14) = actions/effective.
Optionally cross-checks against the historical cache (state/action exact, RGB sampled)."""
import json,hashlib,sys,argparse
from pathlib import Path
import numpy as np,h5py
ROOT=Path(__file__).resolve().parents[2];DATA=ROOT/'data/demos/sku09-original300';CACHE=ROOT/'runs/bc_stage1/cache'
OLD=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/runs/bc_stage1_seed42/cache')
def build(split):
    ids=json.loads((DATA/'splits.json').read_text())[split];marker=CACHE/f'{split}_complete.json'
    if marker.exists():print('exists',split);return
    rgb=np.lib.format.open_memmap(CACHE/f'{split}_rgb.npy',mode='w+',dtype=np.uint8,shape=(len(ids),61,3,3,240,320))
    state=np.zeros((len(ids),61,28),np.float32);action=np.zeros((len(ids),60,8,14),np.float32)
    for j,idx in enumerate(ids):
        with h5py.File(DATA/'episodes'/f'episode_{idx:04d}.h5','r') as f:
            for k,c in enumerate(['global','left_wrist','right_wrist']):rgb[j,:,k]=f['observations/rgb/'+c][:].transpose(0,3,1,2)
            state[j]=f['observations/proprio'][:];action[j]=f['actions/effective'][:].reshape(60,8,14)
        if j%30==0:print('CACHE',split,j,len(ids),flush=True)
    rgb.flush();del rgb;np.save(CACHE/f'{split}_state.npy',state);np.save(CACHE/f'{split}_action.npy',action)
    tmp=marker.with_suffix('.tmp');tmp.write_text(json.dumps(dict(ids=ids,manifest_sha256=hashlib.sha256((DATA/'manifest.jsonl').read_bytes()).hexdigest(),action_field='actions/effective',source=str(DATA)),indent=2));tmp.replace(marker)
def crosscheck(split):
    if not (OLD/f'{split}_complete.json').exists():return dict(split=split,checked=False)
    new_s=np.load(CACHE/f'{split}_state.npy');new_a=np.load(CACHE/f'{split}_action.npy')
    old_s=np.load(OLD/f'{split}_state.npy');old_a=np.load(OLD/f'{split}_action.npy')
    nr=np.load(CACHE/f'{split}_rgb.npy',mmap_mode='r');orr=np.load(OLD/f'{split}_rgb.npy',mmap_mode='r')
    rng=np.random.default_rng(0);eps=rng.choice(len(nr),min(8,len(nr)),replace=False)
    rgb_eq=all(np.array_equal(nr[e],orr[e]) for e in eps)
    return dict(split=split,checked=True,state_equal=bool(np.array_equal(new_s,old_s)),action_equal=bool(np.array_equal(new_a,old_a)),rgb_sampled_equal=bool(rgb_eq),rgb_episodes_checked=[int(e) for e in eps])
if __name__=='__main__':
    CACHE.mkdir(parents=True,exist_ok=True)
    for s in ['train','validation']:build(s)  # test split never read
    res=[crosscheck(s) for s in ['train','validation']];(CACHE/'crosscheck_vs_historical.json').write_text(json.dumps(res,indent=2));print(json.dumps(res))
    assert all(r.get('checked') is False or (r['state_equal'] and r['action_equal'] and r['rgb_sampled_equal']) for r in res),'cache mismatch vs historical'
