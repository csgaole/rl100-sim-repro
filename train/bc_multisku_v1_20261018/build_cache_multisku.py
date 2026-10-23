"""Build the multi-SKU BC cache (same format as sku09 src/p3_build_cache.py) into a NEW dir.
rgb uint8 (E,61,3,3,240,320), state (E,61,28), action (E,60,8,14) from actions/effective, plus sku (E,4) raw cond.
Also writes the dataset index (splits_bc_v1.json, combined_manifest_bc_v1.json) next to the data (new files only).
Stratified per SKU: 80 train / 10 validation / 10 spare (spare is never cached or read by training)."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np,h5py
from concurrent.futures import ProcessPoolExecutor
DATA=Path('/data02/kemove/sku5-flip-demonstrations/data/v1')
SKUS=[('A','SKU5A',[.25,.20,.15,.5893]),('B','SKU5B',[.27,.22,.12,.5600]),('C','SKU5C',[.30,.20,.10,.4714]),('D','SKU5D',[.30,.25,.07,.4125]),('E','SKU5E',[.25,.16,.14,.4400])]
SCALE=np.array([.30,.25,.15,.60])   # sku_cond = [L,W,H,mass]/SCALE ; the policy normalizer then maps limits to [-1,1]
def sku_cond(raw):return (np.asarray(raw,np.float64)/SCALE).astype(np.float32)
def one(args):
    out,name,j,path,sha=args
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==sha,path
    rgb=np.load(out/f'{name}_rgb.npy',mmap_mode='r+')
    with h5py.File(path,'r') as f:
        for k,c in enumerate(['global','left_wrist','right_wrist']):rgb[j,:,k]=f['observations/rgb/'+c][:].transpose(0,3,1,2)
        st=f['observations/proprio'][:].astype(np.float32);ac=f['actions/effective'][:].reshape(60,8,14).astype(np.float32)
        assert st.shape==(61,28) and f['actions/effective'].shape==(480,14)
    rgb.flush();del rgb;return j,st,ac
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=32);a=ap.parse_args()
    if a.out.exists():raise SystemExit(f'Refuse: {a.out} exists')
    sp=DATA/'splits_bc_v1.json';mp=DATA/'combined_manifest_bc_v1.json'
    if sp.exists() or mp.exists():raise SystemExit('Refuse: dataset index already exists')
    man=[];split=dict(train=[],validation=[],spare=[])
    for k,(sid,key,raw) in enumerate(SKUS):
        rows={r['index']:r for r in map(json.loads,open(DATA/f'sku{sid}/manifest.jsonl'))}
        assert sorted(rows)==list(range(100)),sid
        for i in range(100):
            r=rows[i];man.append(dict(index=k*100+i,sku_id=sid,sku=key,episode=i,path=r['path'],sha256=r['sha256'],sku_raw=raw,sku_cond=sku_cond(raw).tolist(),seed=r['config']['seed'],peak_force_N=r['result']['peak_gripper_force_N']))
        perm=np.random.default_rng(20261018+k).permutation(100)
        split['train']+=sorted(int(k*100+i) for i in perm[:80]);split['validation']+=sorted(int(k*100+i) for i in perm[80:90]);split['spare']+=sorted(int(k*100+i) for i in perm[90:])
    mp.write_text(json.dumps(man,indent=1));split['note']='stratified per SKU 80/10/10 with rng(20261018+k); spare never used in training/gates';sp.write_text(json.dumps(split,indent=1))
    msha=hashlib.sha256(mp.read_bytes()).hexdigest();by={r['index']:r for r in man}
    a.out.mkdir(parents=True)
    for name in ['train','validation']:
        ids=split[name]
        np.lib.format.open_memmap(a.out/f'{name}_rgb.npy',mode='w+',dtype=np.uint8,shape=(len(ids),61,3,3,240,320)).flush()
        state=np.zeros((len(ids),61,28),np.float32);act=np.zeros((len(ids),60,8,14),np.float32)
        with ProcessPoolExecutor(a.workers) as ex:
            for j,st,ac in ex.map(one,[(a.out,name,j,by[i]['path'],by[i]['sha256']) for j,i in enumerate(ids)]):state[j]=st;act[j]=ac
        np.save(a.out/f'{name}_state.npy',state);np.save(a.out/f'{name}_action.npy',act)
        np.save(a.out/f'{name}_sku.npy',np.stack([by[i]['sku_cond'] for i in ids]).astype(np.float32))
        (a.out/f'{name}_complete.json').write_text(json.dumps(dict(ids=ids,manifest_sha256=msha,builder='build_cache_multisku.py',sku_scale=SCALE.tolist(),sku_of_id={str(i):by[i]['sku'] for i in ids})))
        print('CACHE',name,len(ids),flush=True)
    print('CACHE_DONE',a.out,flush=True)
