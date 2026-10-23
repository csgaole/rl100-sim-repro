"""Generate bc_model_multisku.py, evaluate_bc_multisku.py, train_bc_multisku.py from the originals with asserted edits."""
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
SRC=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src')
def patch(src,reps):
    for a,b in reps:
        assert src.count(a)==1,(a[:80],src.count(a));src=src.replace(a,b)
    return src
def write(name,text):
    p=H/name
    if p.exists():raise SystemExit(f'refuse overwrite {p}')
    p.write_text(text);print('wrote',p)
# ---- model: identical to bc_model_novib + low_dim obs key sku_cond (4) -> concatenated into global cond (per obs step)
m=patch((SRC/'bc_model_novib.py').read_text(),[
 ('"""[COPY of bc_model.py with use_vib=False (P3 v3)]','"""[MULTI-SKU copy of bc_model_novib.py: adds low_dim obs key sku_cond=[L,W,H,mass]/[.30,.25,.15,.60] (4D),\nconcatenated by MultiImageObsEncoder into the global cond of BOTH obs steps (UNet global_cond_dim += 8). Nothing else changed.]\n[COPY of bc_model.py with use_vib=False (P3 v3)]'),
 ("ROOT=Path(__file__).resolve().parents[1]","ROOT=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations')"),
 ("        shape['obs']['agent_pos']=dict(shape=[28],type='low_dim')","        shape['obs']['agent_pos']=dict(shape=[28],type='low_dim')\n        shape['obs']['sku_cond']=dict(shape=[4],type='low_dim')"),
])
write('bc_model_multisku.py',m)
# ---- evaluator
E=Path('/data02/kemove/rl100-sim-repro/train/bc_stage1/evaluate_bc.py').read_text()
e=patch(E,[
 ('"""[rl100-sim-repro copy','"""[MULTI-SKU copy of rl100-sim-repro train/bc_stage1/evaluate_bc.py: --sku selects scene (SKU5A..E = scene_sku10..14); sku_cond fed from the checkpoint config table; everything else (p5_y2 IC, samplers, flip_pose_no_drop_v2) unchanged]\n[rl100-sim-repro copy'),
 ("from bc_model import ROOT,BCPolicy","sys.path.insert(0,'/data02/kemove/rl100-sim-repro/train/bc_stage1');sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))\nfrom bc_model_multisku import BCPolicy\nROOT=Path('/data02/kemove/rl100-sim-repro')"),
 ("    p=BCPolicy(dict(agent_pos=np.zeros((2,28),np.float32),action=np.zeros((2,14),np.float32)),pretrained=False)",
  "    p=BCPolicy(dict(agent_pos=np.zeros((2,28),np.float32),action=np.zeros((2,14),np.float32),sku_cond=np.zeros((2,4),np.float32)),pretrained=False)"),
 ("    p.load_state_dict(cp['ema'],strict=True);p.cuda().eval();p.use_aug=False",
  "    p.load_state_dict(cp['ema'],strict=True);p.cuda().eval();p.use_aug=False\n    p.sku_table=cp['config']['sku_cond_table']"),
 ("    obs['agent_pos']=torch.as_tensor(np.stack([h['proprio'] for h in history]),device='cuda',dtype=torch.float32)\n    obs=p.normalizer.normalize(obs)",
  "    obs['agent_pos']=torch.as_tensor(np.stack([h['proprio'] for h in history]),device='cuda',dtype=torch.float32)\n    obs['sku_cond']=torch.as_tensor(np.stack([p.sku_cond]*len(history)),device='cuda',dtype=torch.float32)\n    obs=p.normalizer.normalize(obs)"),
 ("def episode(p,seed,sample_fn=None,position_half_range=.005,yaw_half_range=2.):","def episode(p,seed,sample_fn=None,position_half_range=.005,yaw_half_range=2.,sku='SKU09'):"),
 ("    with LogisticsSortingEnv(render=True,image_size=(320,240)) as env:","    with LogisticsSortingEnv(sku=sku,render=True,image_size=(320,240)) as env:"),
 ("ap.add_argument('--yaw-half-range',type=float,default=2.);args=ap.parse_args()","ap.add_argument('--yaw-half-range',type=float,default=2.);ap.add_argument('--sku',required=True);args=ap.parse_args()"),
 ("sha=hashlib.sha256(path.read_bytes()).hexdigest();p,cp=load_policy(path);rows=[]","sha=hashlib.sha256(path.read_bytes()).hexdigest();p,cp=load_policy(path);rows=[]\n    p.sku_cond=np.asarray(p.sku_table[args.sku],np.float32)"),
 ("row=episode(p,seed,sample_fn=sample_fn,position_half_range=args.position_half_range,yaw_half_range=args.yaw_half_range)","row=episode(p,seed,sample_fn=sample_fn,position_half_range=args.position_half_range,yaw_half_range=args.yaw_half_range,sku=args.sku)"),
 ("    report=dict(success_criteria_version=SUCCESS_VERSION,","    report=dict(sku=args.sku,sku_cond=p.sku_cond.tolist(),success_criteria_version=SUCCESS_VERSION,"),
])
write('evaluate_bc_multisku.py',e)
# ---- trainer
T=Path('/data02/kemove/rl100-sim-repro/train/bc_nominal_v3_20261017/train_bc_nominal.py').read_text()
t=patch(T,[
 ('"""[NOMINAL 2026-10-17 copy','"""[MULTI-SKU 2026-10-18 copy of rl100-sim-repro train/bc_nominal_v3_20261017/train_bc_nominal.py] data = sku5-flip-demonstrations/data/v1 (5 SKUs x 100; stratified 80 train/10 val/10 spare per SKU), cache built by build_cache_multisku.py; original300 asserts removed; SKU conditioning (sku_cond 4D -> global cond) via bc_model_multisku; per-SKU nominal p5_y2 gate (nominal_gate_multisku, 20 eps x 5 SKUs, seeds 880000-880019, upstream DDIM10 eta=1, final sigma 0). Loss/model core/optimizer/EMA/lr schedule unchanged; --effective-batch added (default 512 = recipe).]\n[NOMINAL 2026-10-17 copy'),
 ("from bc_model_novib import BCPolicy,ROOT","from bc_model_multisku import BCPolicy,ROOT\nimport a100_runtime as _rt\n_host_batch_orig=_rt.host_batch\ndef _host_batch_ms(data,ids,pin=True):\n    r=_host_batch_orig(data,ids,pin);s=np.repeat(data['sku'][ids//60][:,None],2,1)\n    r['sku_cond']=torch.from_numpy(np.ascontiguousarray(s)).pin_memory() if pin else torch.from_numpy(s);return r\n_rt.host_batch=_host_batch_ms"),
 ("import nominal_gate\n","import nominal_gate_multisku as nominal_gate\n"),
 ("DATA=ROOT/'data/expanded_v1'","DATA=Path('/data02/kemove/sku5-flip-demonstrations/data/v1')\nCACHE=Path('/data02/kemove/sku5-flip-demonstrations/cache/bc_v1')\nSPLITS='splits_bc_v1.json';MANIFEST='combined_manifest_bc_v1.json'"),
 ("RUN_FAMILY='p3_v3_novib_absolute'  # kept for loader compatibility; config.data_subset marks nominal","RUN_FAMILY='multisku_v1_novib_absolute'"),
 ("    ids=json.loads((DATA/'splits.json').read_text())[split]\n    marker","    ids=json.loads((DATA/SPLITS).read_text())[split]\n    marker"),
 ("    assert info['manifest_sha256']==hashlib.sha256((DATA/'combined_manifest.json').read_bytes()).hexdigest(),'cache built from a different combined_manifest.json'\n    return dict(rgb=np.load(cache/f'{split}_rgb.npy',mmap_mode='r'),state=np.load(cache/f'{split}_state.npy'),action=np.load(cache/f'{split}_action.npy'),n=len(ids)*60,ids=ids)",
  "    assert info['manifest_sha256']==hashlib.sha256((DATA/MANIFEST).read_bytes()).hexdigest(),'cache built from a different combined manifest'\n    return dict(rgb=np.load(cache/f'{split}_rgb.npy',mmap_mode='r'),state=np.load(cache/f'{split}_state.npy'),action=np.load(cache/f'{split}_action.npy'),sku=np.load(cache/f'{split}_sku.npy'),n=len(ids)*60,ids=ids,sku_of_id=info['sku_of_id'])"),
 ("    obs['agent_pos']=torch.as_tensor(np.stack([data['state'][ep,prev],data['state'][ep,t]],1),device='cuda')\n",
  "    obs['agent_pos']=torch.as_tensor(np.stack([data['state'][ep,prev],data['state'][ep,t]],1),device='cuda')\n    obs['sku_cond']=torch.as_tensor(np.repeat(data['sku'][ep][:,None],2,1),device='cuda')\n"),
 ("    rows=json.loads((DATA/'combined_manifest.json').read_text());return {r['index']:int(r.get('cell',-1)) for r in rows}","    rows=json.loads((DATA/MANIFEST).read_text());return {r['index']:int(r.get('cell',-1)) for r in rows}"),
 ("    rng=np.random.default_rng(7);g={'val_old300':list(range(len(val['ids']))),'train_old300_10ep':sorted(rng.choice(len(train['ids']),10,replace=False).tolist())}",
  "    rng=np.random.default_rng(7);g={}\n    for e,i in enumerate(val['ids']):g.setdefault('val_'+val['sku_of_id'][str(i)],[]).append(e)\n    for s in sorted(set(train['sku_of_id'].values())):\n        es=[e for e,i in enumerate(train['ids']) if train['sku_of_id'][str(i)]==s];g['train_'+s+'_4ep']=sorted(rng.choice(es,4,replace=False).tolist())"),
 ("    ap.add_argument('--cache-dir',type=Path,default=DATA/'cache')","    ap.add_argument('--cache-dir',type=Path,default=CACHE)"),
 ("choices=[8,16,32,64,128],default=64)","choices=[8,16,32,64,128,256,512],default=64);ap.add_argument('--effective-batch',type=int,default=512)"),
 ("""    def o300(d,n):
        assert all(i<300 for i in d['ids'][:n]) and all(i>=300 for i in d['ids'][n:])
        return dict(rgb=d['rgb'][:n],state=d['state'][:n],action=d['action'][:n],n=n*60,ids=list(d['ids'][:n]))
    train=o300(load_data('train',args.cache_dir),240);val=o300(load_data('validation',args.cache_dir),30)
    _s=json.loads(Path('/data02/kemove/rl100-sim-repro/data/demos/sku09-original300/splits.json').read_text());assert sorted(train['ids'])==sorted(_s['train']) and sorted(val['ids'])==sorted(_s['validation'])
    assert len(train['ids'])==240 and len(val['ids'])==30""",
"""    train=load_data('train',args.cache_dir);val=load_data('validation',args.cache_dir)
    from collections import Counter
    assert len(train['ids'])==400 and len(val['ids'])==50 and not set(train['ids'])&set(val['ids'])
    assert Counter(train['sku_of_id'].values())==Counter({s:80 for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']}) and Counter(val['sku_of_id'].values())==Counter({s:10 for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']})
    _man={r['index']:r for r in json.loads((DATA/MANIFEST).read_text())};sku_cond_table={}
    for r in _man.values():sku_cond_table.setdefault(r['sku'],r['sku_cond'])
    for d in (train,val):assert all(np.allclose(d['sku'][e],sku_cond_table[d['sku_of_id'][str(i)]]) for e,i in enumerate(d['ids']))"""),
 ("    normalizer_data=dict(agent_pos=train['state'].reshape(-1,28),action=train['action'].reshape(-1,14))",
  "    normalizer_data=dict(agent_pos=train['state'].reshape(-1,28),action=train['action'].reshape(-1,14),sku_cond=train['sku'])"),
 ("epochs=args.max_epochs;effective_batch=512;","epochs=args.max_epochs;effective_batch=args.effective_batch;assert effective_batch%args.microbatch==0;"),
 ("seed=args.seed,data_subset='original300_only_nominal_p5y2(train240/val30;no wide;no expanded cells)',epochs=epochs",
  "seed=args.seed,data_subset='sku5_v1 multi-SKU (A..E x 100; train 80/val 10/spare 10 per SKU); nominal p5_y2',sku_cond_table=sku_cond_table,sku_cond_definition='[L,W,H,mass]/[.30,.25,.15,.60] then LinearNormalizer limits; low_dim obs key -> global cond (both obs steps)',epochs=epochs"),
 ("test_episodes=30,","test_episodes=50,spare_note='50 spare episodes (10/SKU) never read',"),
 ("nominal_trainer_sha256=sha256(Path(__file__).resolve()),","nominal_trainer_sha256=sha256(Path(__file__).resolve()),multisku_sources_sha256={n:sha256(Path(__file__).resolve().parent/n) for n in ['bc_model_multisku.py','evaluate_bc_multisku.py','nominal_gate_multisku.py','build_cache_multisku.py']},"),
 ("        atomic_json(OUT/'status.json',dict(status='development_evaluation',pid=os.getpid(),epoch=completed_epoch,step=step,epochs=epochs))",
  "        atomic_json(OUT/'status.json',dict(status='development_evaluation',pid=os.getpid(),epoch=completed_epoch,step=step,epochs=epochs));torch.cuda.empty_cache()"),
])
write('train_bc_multisku.py',t)
