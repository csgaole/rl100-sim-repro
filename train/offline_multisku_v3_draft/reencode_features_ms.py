# [offline_multisku_v3_draft P0] Full feature re-encode: demos + ALL rollouts under one IL/policy encoder.
# Requires RGB in ep.npz (batch_worker store_obs). Writes feat_ckpt/feat_spec stamp into every file.
# Usage: python reencode_features_ms.py --policy IL.pt --sources sources_merged.json [--force]
from ms_common import *
import argparse
from collections import Counter
ap=argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
ap.add_argument('--sources', required=True, help='json list of [kind,path] (same as IQL)')
ap.add_argument('--cache', default='/data02/kemove/sku5-flip-demonstrations/cache/bc_v1')
ap.add_argument('--force', action='store_true', help='overwrite existing feats')
ap.add_argument('--limit-eps', type=int, default=0, help='smoke: only first N rollouts total')
a=ap.parse_args(); setup(0)
p,cp=load_policy(a.policy); p.eval()
meta=feat_meta_for_policy(a.policy)
sources=json.loads(Path(a.sources).read_text())
print('REENCODE_START', json.dumps(dict(policy=a.policy, meta=meta, sources=sources)), flush=True)
n_demo=0; n_roll=0; n_skip_rgb=0; errors=[]

# ---- demos from BC RGB cache ----
C=Path(a.cache)
EXPECT={'train':400,'validation':50}
for kind,path in sources:
    if kind!='demo': continue
    out_dir=Path(path); out_dir.mkdir(parents=True, exist_ok=True)
    for split in ['train','validation']:
        info=json.loads((C/f'{split}_complete.json').read_text()); ids=info['ids']
        sku_of={str(k):v for k,v in info['sku_of_id'].items()}; names=[sku_of[str(i)] for i in ids]
        rgb=np.load(C/f'{split}_rgb.npy', mmap_mode='r')
        state=np.load(C/f'{split}_state.npy'); action=np.load(C/f'{split}_action.npy'); sk=np.load(C/f'{split}_sku.npy')
        n=len(ids); F=np.zeros((n,61,FEAT), np.float32)
        for e in range(n):
            frames={**{c:np.asarray(rgb[e,:,k]) for k,c in enumerate(CAMS)}, 'proprio':state[e]}
            F[e]=encode_frames(p, frames, sk[e]).cpu().numpy()
            if e%50==0: print('DEMO', split, e, n, flush=True)
        with torch.no_grad():
            A=p.normalizer['action'].normalize(torch.from_numpy(action[:n]).cuda()).cpu().numpy().astype(np.float32)
        tmp=out_dir/f'demo_{split}.tmp.npz'
        np.savez(tmp, feats=F, actions=A, success=np.ones(n,bool), ids=np.array(ids[:n]),
                 sku=np.array(names[:n]), sku_cond=sk[:n].astype(np.float32),
                 feat_ckpt=np.array(meta['feat_ckpt']), feat_ckpt_sha256=np.array(meta['feat_ckpt_sha256']),
                 feat_spec=np.array(meta['feat_spec']), feat_dim=np.array(meta['feat_dim']))
        tmp.replace(out_dir/f'demo_{split}.npz'); n_demo+=n
        print('DEMO_DONE', split, n, flush=True)
    write_json(out_dir/'demo_features.json', dict(**meta, policy=a.policy, cache=a.cache, updated=cst(), n_demo=n_demo))

# ---- rollouts: require RGB ----
n_lim=a.limit_eps
for kind,path in sources:
    if kind=='demo': continue
    files=sorted(glob.glob(str(Path(path)/'**'/'ep_*.npz'), recursive=True))
    files=[f for f in files if '.tmp.' not in f]
    for fi,f in enumerate(files):
        if n_lim and n_roll>=n_lim: break
        z=np.load(f); m=json.loads(str(z['meta']))
        need=[f'rgb_{c}' for c in CAMS]+['prop']
        if any(k not in z.files for k in need):
            n_skip_rgb+=1; errors.append(f); continue
        frames={c:np.asarray(z[f'rgb_{c}']) for c in CAMS}
        # accept HWC (61,H,W,3) or CHW (61,3,H,W)
        for c in CAMS:
            arr=frames[c]
            if arr.ndim==4 and arr.shape[-1]==3:
                frames[c]=arr.transpose(0,3,1,2)  # -> (61,3,H,W)
            elif not (arr.ndim==4 and arr.shape[1]==3):
                raise SystemExit(f'bad rgb shape {f} {c} {arr.shape}')
        frames['proprio']=np.asarray(z['prop'], np.float32)
        assert frames['proprio'].shape[0]==61 and frames[CAMS[0]].shape[0]==61, (f, {k:v.shape for k,v in frames.items()})
        sk=np.asarray(m['sku_cond'], np.float32)
        feats=encode_frames(p, frames, sk).cpu().numpy().astype(np.float32)
        actions=z['actions']  # already normalized at collect time; keep
        m2=dict(m, **meta, feat_reencoded_at=cst(), feat_reencoded_from=a.policy)
        tmp=Path(str(f)+'.re.tmp.npz')
        kw=dict(meta=np.array(json.dumps(m2)), feats=feats, actions=actions)
        for c in CAMS: kw[f'rgb_{c}']=z[f'rgb_{c}']
        kw['prop']=z['prop']
        np.savez_compressed(tmp, **kw); tmp.replace(f); n_roll+=1
        if n_roll%20==0: print('EP', n_roll, f, flush=True)

if n_skip_rgb:
    raise SystemExit(
        f'REENCODE_FAIL {n_skip_rgb} rollouts missing RGB (rgb_* + prop). '
        f'Collect must store_obs. examples={errors[:5]}'
    )

out_manifest=dict(**meta, policy=a.policy, sources=sources, n_demo=n_demo, n_roll=n_roll, updated=cst())
# write next to first demo dir or S2_OUT
man_path=Path(sources[0][1])/'feat_reencode_manifest.json' if sources else Path('feat_reencode_manifest.json')
if sources[0][0]!='demo':
    man_path=Path(sources[0][1])/'feat_reencode_manifest.json'
write_json(man_path, out_manifest)
print('REENCODE_DONE', json.dumps(out_manifest), flush=True)
# verify
assert_uniform_feat_space(sources, context='REENCODE_VERIFY')
