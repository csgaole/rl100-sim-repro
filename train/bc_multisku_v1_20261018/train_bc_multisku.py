"""[MULTI-SKU 2026-10-18 copy of rl100-sim-repro train/bc_nominal_v3_20261017/train_bc_nominal.py] data = sku5-flip-demonstrations/data/v1 (5 SKUs x 100; stratified 80 train/10 val/10 spare per SKU), cache built by build_cache_multisku.py; original300 asserts removed; SKU conditioning (sku_cond 4D -> global cond) via bc_model_multisku; per-SKU nominal p5_y2 gate (nominal_gate_multisku, 20 eps x 5 SKUs, seeds 880000-880019, upstream DDIM10 eta=1, final sigma 0). Loss/model core/optimizer/EMA/lr schedule unchanged; --effective-batch added (default 512 = recipe).]
[NOMINAL 2026-10-17 copy of sku09 src/train_bc_v3_novib.py] data restricted to original300 (expanded_v1 ids<300 = first 240 train / 30 val; identical to rl100-sim-repro data/demos/sku09-original300 splits), NO wide, NO expanded cells; nominal p5_y2 dev gate on 880000-880099 (nominal_gate.py). Loss/model/optimizer/EMA unchanged; --seed/--lr added."""
"""[P3 v3: copy of train_bc_expanded_v2.py] no-VIB encoder (bc_model_novib), data = expanded_v1 + wide_v1 (separate caches, no copy),
action-error validation metric, gates via p3_dev_gate_v3 (novib deterministic eval, all gates eligible). Defaults: 88 epochs, gates 13,26,..,78,88.
P3-revised Stage A: BC retrain on 300 original + 900 expanded_v1 demos (absolute 14D joint targets).
Derived from src/train_bc_expanded.py (exported copy sha 0b5e1ae5...). Loss math / model / optimizer / EMA / batching
are unchanged. Changes (all deliberate, see P3_修订执行方案.md):
  * action space hard-wired to 'absolute' (the exported script defaulted to delta_target); no delta branches.
  * --run-dir is REQUIRED and must not exist (fresh) -- nothing under runs/ or data/ is ever overwritten.
  * update budget default 154 epochs x 113 = 17402 updates (~= Stage-1 17400).
  * development gates: first at epoch 26 (~2938 updates ~= Stage-1 epoch 100), then every 13 epochs (~1469 ~= Stage-1 50 epochs),
    plus a final gate at the last epoch; evaluated on SEPARATE idle GPU(s) via p3_dev_gate; loud logging to stdout and to a
    mirror directory outside the run dir; failures raise (never silent).
  * the gate candidate defaults to the CURRENT EMA at the gate epoch (val-best froze under the v1 plateau).
  * per-subset validation (old300 / new900 / each expanded cell) + EMA no-aug loss on fixed train samples.
  * --smoke: a few updates + tiny validation + a 4-episode gate end-to-end, then exits.
Reads data/expanded_v1 READ-ONLY."""
import os,sys
sys.path.insert(0,'/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src')
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,copy,json,math,random,time,traceback,hashlib,sys,shutil
from pathlib import Path
import numpy as np,torch
from bc_model_multisku import BCPolicy,ROOT
import a100_runtime as _rt
_host_batch_orig=_rt.host_batch
def _host_batch_ms(data,ids,pin=True):
    r=_host_batch_orig(data,ids,pin);s=np.repeat(data['sku'][ids//60][:,None],2,1)
    r['sku_cond']=torch.from_numpy(np.ascontiguousarray(s)).pin_memory() if pin else torch.from_numpy(s);return r
_rt.host_batch=_host_batch_ms
from rl_100.model.diffusion.ema_model import EMAModel
from diffusers.optimization import get_scheduler
from a100_runtime import install_deferred_metrics,prefetch_batches,gradient_health
import nominal_gate_multisku as nominal_gate
DATA=Path('/data02/kemove/sku5-flip-demonstrations/data/v1')
CACHE=Path('/data02/kemove/sku5-flip-demonstrations/cache/bc_v1')
SPLITS='splits_bc_v1.json';MANIFEST='combined_manifest_bc_v1.json'
RUN_FAMILY='multisku_v1_novib_absolute'
WIDE=ROOT/'data/wide_v1'
OUT=None

def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<24),b''):h.update(b)
    return h.hexdigest()

def load_data(split,cache):
    ids=json.loads((DATA/SPLITS).read_text())[split]
    marker=cache/f'{split}_complete.json'
    if not marker.exists():raise RuntimeError(f'Verified cache marker missing: {marker}; build it first (do not write into data/expanded_v1)')
    info=json.loads(marker.read_text())
    assert info['ids']==ids,'cache ids != splits.json'
    assert info['manifest_sha256']==hashlib.sha256((DATA/MANIFEST).read_bytes()).hexdigest(),'cache built from a different combined manifest'
    return dict(rgb=np.load(cache/f'{split}_rgb.npy',mmap_mode='r'),state=np.load(cache/f'{split}_state.npy'),action=np.load(cache/f'{split}_action.npy'),sku=np.load(cache/f'{split}_sku.npy'),n=len(ids)*60,ids=ids,sku_of_id=info['sku_of_id'])

class CatRGB:
    '''Read-only concatenation of per-source rgb memmaps along the episode axis (no copy).'''
    def __init__(s,parts):s.parts=parts;s.off=np.cumsum([0]+[p.shape[0] for p in parts])
    def __getitem__(s,key):
        ep,t=key;ep=np.asarray(ep);t=np.asarray(t);out=np.empty((len(ep),)+s.parts[0].shape[2:],np.uint8)
        for k,p in enumerate(s.parts):
            m=(ep>=s.off[k])&(ep<s.off[k+1])
            if m.any():out[m]=p[ep[m]-s.off[k],t[m]]
        return out

def load_wide(split,cache):
    ids=json.loads((WIDE/'splits.json').read_text())[split]
    info=json.loads((cache/f'{split}_complete.json').read_text())
    assert info['ids']==ids,'wide cache ids != wide_v1/splits.json'
    assert info['manifest_sha256']==hashlib.sha256((WIDE/'capture_manifest.json').read_bytes()).hexdigest(),'wide cache from different manifest'
    return dict(rgb=np.load(cache/f'{split}_rgb.npy',mmap_mode='r'),state=np.load(cache/f'{split}_state.npy'),action=np.load(cache/f'{split}_action.npy'),n=len(ids)*60,ids=ids)

def merge(a,b):
    return dict(rgb=CatRGB([a['rgb'],b['rgb']]),state=np.concatenate([a['state'],b['state']]),action=np.concatenate([a['action'],b['action']]),n=a['n']+b['n'],ids=list(a['ids'])+list(b['ids']))

J=[0,1,2,3,4,5,7,8,9,10,11,12]
@torch.no_grad()
def action_error(p,data,indices,seed=4242,bs=32):
    '''Eval mode, no aug: DDIM10 eta=0 from fixed noise; mean |pred-gt| over the 12 arm joints (rad), 8-step chunk.'''
    errs=[];g=torch.Generator(device='cuda').manual_seed(seed)
    for j in range(0,len(indices),bs):
        b=batch(data,indices[j:j+bs]);B=len(b['action'])
        n=p.normalizer.normalize(b['obs']);n={k:v.reshape(-1,*v.shape[2:]) for k,v in n.items()}
        cond=p.obs_encoder(n).reshape(B,-1);x=torch.randn((B,8,14),device='cuda',generator=g)
        p.noise_scheduler.set_timesteps(10)
        for t in p.noise_scheduler.timesteps:
            x=p.noise_scheduler.step_mean(p.model(x,t.to('cuda'),global_cond=cond),t,x,eta=0.0,generator=g).prev_sample
        a=p.normalizer['action'].unnormalize(x);gt=b['action'][:,1:]
        errs.append((a[...,J]-gt[...,J]).abs().mean(dim=(1,2)))
    e=torch.cat(errs);return float(e.mean()),bool(torch.isfinite(e).all())

def batch(data,indices):
    ep=indices//60;t=indices%60;prev=np.maximum(t-1,0)
    images=np.stack([data['rgb'][ep,prev],data['rgb'][ep,t]],1)
    obs={c:torch.as_tensor(np.ascontiguousarray(images[:,:,k]),device='cuda',dtype=torch.float32)/255 for k,c in enumerate(['global','left_wrist','right_wrist'])}
    obs['agent_pos']=torch.as_tensor(np.stack([data['state'][ep,prev],data['state'][ep,t]],1),device='cuda')
    obs['sku_cond']=torch.as_tensor(np.repeat(data['sku'][ep][:,None],2,1),device='cuda')
    a=data['action'][ep,t]
    a=np.concatenate([a[:,:1],a],1)  # structural prefix, unchanged
    return dict(obs=obs,action=torch.as_tensor(a,device='cuda'))

def cell_of():
    rows=json.loads((DATA/MANIFEST).read_text());return {r['index']:int(r.get('cell',-1)) for r in rows}

@torch.no_grad()
def eval_loss(ema,data,indices,seed,bs=8):
    """EMA, no augmentation, fixed noise seed. Same procedure as the per-epoch validation pass."""
    out={};n=len(indices)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        for j in range(0,n,bs):
            ids=indices[j:j+bs];_,parts=ema.compute_loss(batch(data,ids))
            for k,v in parts.items():out[k]=out.get(k,0)+float(v)*len(ids)/n
    return out

def subset_groups(val,train,cells,smoke):
    rng=np.random.default_rng(7);g={}
    for e,i in enumerate(val['ids']):g.setdefault('val_'+val['sku_of_id'][str(i)],[]).append(e)
    for s in sorted(set(train['sku_of_id'].values())):
        es=[e for e,i in enumerate(train['ids']) if train['sku_of_id'][str(i)]==s];g['train_'+s+'_4ep']=sorted(rng.choice(es,4,replace=False).tolist())
    idx={k:np.concatenate([np.arange(e*60,e*60+60) for e in v]) for k,v in g.items()}
    if smoke:idx={k:v[:8] for k,v in idx.items()}
    return idx
def _old_subset_groups(val,train,cells,smoke):
    g={}
    for e,i in enumerate(val['ids']):
        key='val_old300' if i<300 else (f'val_cell{cells[i]}' if i<1200 else 'val_wide')
        g.setdefault(key,[]).append(e)
    rng=np.random.default_rng(7)
    old=[e for e,i in enumerate(train['ids']) if i<300];new=[e for e,i in enumerate(train['ids']) if i>=300]
    g['train_old300_10ep']=sorted(rng.choice(old,10,replace=False).tolist());g['train_new900_10ep']=sorted(rng.choice(new,10,replace=False).tolist())
    idx={k:np.concatenate([np.arange(e*60,e*60+60) for e in v]) for k,v in g.items()}
    if smoke:idx={k:v[:8] for k,v in list(idx.items())[:3]}
    return idx

def main():
    global OUT
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--cache-dir',type=Path,default=CACHE)
    ap.add_argument('--wide-cache-dir',type=Path,default=ROOT/'data/p3_cache_wide_v1')
    ap.add_argument('--microbatch',type=int,choices=[8,16,32,64,128,256,512],default=64);ap.add_argument('--effective-batch',type=int,default=512)
    ap.add_argument('--precision',choices=['fp32','tf32'],default='fp32')
    ap.add_argument('--max-epochs',type=int,default=88)
    ap.add_argument('--first-gate-epoch',type=int,default=13);ap.add_argument('--gate-every',type=int,default=13)
    ap.add_argument('--min-stop-epoch',type=int,default=65)
    ap.add_argument('--gate-candidate',choices=['current_ema','val_best'],default='current_ema')
    ap.add_argument('--eval-gpus',default='',help='physical GPU for nominal gate (own GPU)');ap.add_argument('--seed',type=int,default=42);ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--eval-procs-per-gpu',type=int,default=4)
    ap.add_argument('--mirror-dir',type=Path,required=True,help='copy of gate decisions/history OUTSIDE the run dir')
    ap.add_argument('--subset-every',type=int,default=5)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--smoke-updates',type=int,default=3)
    args=ap.parse_args()
    OUT=args.run_dir.resolve()
    if args.resume:
        if not (OUT/'last.pt').exists():raise RuntimeError('No last.pt in this run dir; resume only of own run')
    elif OUT.exists():raise RuntimeError(f'Refuse: run dir already exists: {OUT}')
    if 'A100' not in torch.cuda.get_device_name():raise RuntimeError('This entrypoint is reserved for A100')
    OUT.mkdir(parents=True,exist_ok=True);args.mirror_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=args.precision=='tf32';torch.backends.cudnn.allow_tf32=args.precision=='tf32';torch.backends.cudnn.benchmark=False
    atomic_json(OUT/'status.json',dict(status='preparing',pid=os.getpid(),device=torch.cuda.get_device_name()))
    train=load_data('train',args.cache_dir);val=load_data('validation',args.cache_dir)
    from collections import Counter
    assert len(train['ids'])==400 and len(val['ids'])==50 and not set(train['ids'])&set(val['ids'])
    assert Counter(train['sku_of_id'].values())==Counter({s:80 for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']}) and Counter(val['sku_of_id'].values())==Counter({s:10 for s in ['SKU5A','SKU5B','SKU5C','SKU5D','SKU5E']})
    _man={r['index']:r for r in json.loads((DATA/MANIFEST).read_text())};sku_cond_table={}
    for r in _man.values():sku_cond_table.setdefault(r['sku'],r['sku_cond'])
    for d in (train,val):assert all(np.allclose(d['sku'][e],sku_cond_table[d['sku_of_id'][str(i)]]) for e,i in enumerate(d['ids']))
    aerr_idx=np.array([e*60+t for e in range(0,len(val['ids']),4) for t in (0,15,30,45)])
    cells=cell_of();groups=subset_groups(val,train,cells,args.smoke)
    normalizer_data=dict(agent_pos=train['state'].reshape(-1,28),action=train['action'].reshape(-1,14),sku_cond=train['sku'])
    model=BCPolicy(normalizer_data).cuda();install_deferred_metrics(model);ema=copy.deepcopy(model);em=EMAModel(ema,power=.75,inv_gamma=1.,max_value=.9999)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    epochs=args.max_epochs;effective_batch=args.effective_batch;assert effective_batch%args.microbatch==0;steps_epoch=math.ceil(train['n']/effective_batch);total=epochs*steps_epoch
    scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=total)
    gate_epochs=sorted(set(list(range(args.first_gate_epoch,epochs+1,args.gate_every))+[epochs]))
    src=Path('/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/src')
    deps=['train_bc_v3_novib.py','p3_dev_gate_v3.py','p3_eval_suite_v3.py','bc_model_novib.py','evaluate_bc_novib.py','p3_build_cache_wide.py','a100_runtime.py','evaluate_bc.py']
    config=dict(stage='BC_only',run_family=RUN_FAMILY,action_semantics='absolute_joint_target_v1',action_field='actions/effective',action_space='absolute',
        data_root=str(DATA),cache_dir=str(args.cache_dir),output_dir=str(OUT),initialization='fresh_seed42_R3M_encoder_random_UNet_and_aux_heads_no_old_BC_weights',use_vib=False,wide_cache_dir=str(args.wide_cache_dir),
        data_sources=dict(expanded_v1='splits.json train/validation (old300+expanded900)',wide_v1='splits.json train 715 / validation 89; test 89 untouched'),action_error_metric='mean abs 12-joint err rad, DDIM10 eta0 fixed noise seed4242, every 4th val episode x t{0,15,30,45}',
        upstream_commit='6b6afceb7eb12525398207b5d6e66908863dc385',recipe='official 2D diffusion chunk launcher + base YAML (unchanged loss math)',
        device=torch.cuda.get_device_name(),seed=args.seed,data_subset='sku5_v1 multi-SKU (A..E x 100; train 80/val 10/spare 10 per SKU); nominal p5_y2',sku_cond_table=sku_cond_table,sku_cond_definition='[L,W,H,mass]/[.30,.25,.15,.60] then LinearNormalizer limits; low_dim obs key -> global cond (both obs steps)',epochs=epochs,effective_batch=effective_batch,microbatch=args.microbatch,
        optimizer_updates=total,steps_per_epoch=steps_epoch,lr=args.lr,betas=[.95,.999],weight_decay=1e-6,lr_warmup_steps=500,
        obs_steps=2,action_steps=8,horizon=9,action_dim=14,proprio_dim=28,clock_input=False,precision=args.precision,
        network='official ConditionalUnet1D [256,512,1024], independent R3M ResNet18 per camera, GroupNorm',
        parameters=sum(p.numel() for p in model.parameters()),recon_weight=.05,kl_beta=.0005,kl_annealing=False,
        augmentation='official random shifts pad4; encoder resize224; no crop',ema='official power=.75 max=.9999',
        train_episodes=len(train['ids']),val_episodes=len(val['ids']),test_episodes=50,spare_note='50 spare episodes (10/SKU) never read',test_accessed=False,
        gate_epochs=gate_epochs,gate_candidate=args.gate_candidate,min_stop_epoch=args.min_stop_epoch,eval_gpus=args.eval_gpus,
        dev_protocol=nominal_gate.PROTOCOL,smoke=args.smoke,
        source_sha256={d:sha256(src/d) for d in deps if (src/d).exists()},nominal_trainer_sha256=sha256(Path(__file__).resolve()),multisku_sources_sha256={n:sha256(Path(__file__).resolve().parent/n) for n in ['bc_model_multisku.py','evaluate_bc_multisku.py','nominal_gate_multisku.py','build_cache_multisku.py']},
        r3m_sha256=sha256(ROOT/'assets/r3m_resnet18.pt'))
    os.environ['P3_EVAL_BC_SHA']=config['source_sha256']['evaluate_bc_novib.py']
    if not args.resume:
        prov=OUT/'provenance';prov.mkdir(exist_ok=True)
        for d in deps:
            if (src/d).exists():shutil.copy2(src/d,prov/d)
    atomic_json(OUT/'config.json',config);print('CONFIG',json.dumps(config),flush=True)
    print('GATE_SCHEDULE',json.dumps(dict(gate_epochs=gate_epochs,updates=[e*steps_epoch for e in gate_epochs])),flush=True)
    step=0;start_epoch=0;start_offset=0;best=float('inf')
    if args.resume:
        cp=torch.load(OUT/'last.pt',map_location='cuda',weights_only=False)
        assert cp['config'].get('run_family')==RUN_FAMILY,'Refuse foreign checkpoint'
        assert cp['config']['precision']==args.precision and cp['config']['microbatch']==args.microbatch and cp['config']['epochs']==epochs,'Resume must preserve precision/microbatch/epochs'
        model.load_state_dict(cp['model']);ema.load_state_dict(cp['ema']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        step=cp['step'];start_epoch=cp['next_epoch'];start_offset=cp['next_offset'];best=cp['best'];em.optimization_step=cp['ema_step']
        torch.set_rng_state(cp['torch_rng'].cpu());torch.cuda.set_rng_state(cp['cuda_rng'].cpu());np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng'])
    def save(name,next_epoch,next_offset):
        payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            step=step,next_epoch=next_epoch,next_offset=next_offset,best=best,ema_step=em.optimization_step,config=config,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        p=OUT/name;tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
    def save_ema(name,epoch,vr):
        tmp=OUT/(name+'.tmp');torch.save(dict(ema=ema.state_dict(),config=config,epoch=epoch,step=step,validation=vr),tmp);tmp.replace(OUT/name)
    def gate(completed_epoch,vr):
        cand_dir=OUT/'development_v2';cand_dir.mkdir(exist_ok=True)
        cand=cand_dir/f'candidate_{completed_epoch:04d}.pt'
        if not cand.exists():
            if args.gate_candidate=='current_ema' and vr is not None:save_ema(f'development_v2/candidate_{completed_epoch:04d}.pt',completed_epoch,vr)
            else:shutil.copy2(OUT/'best.pt',cand)
        atomic_json(OUT/'status.json',dict(status='development_evaluation',pid=os.getpid(),epoch=completed_epoch,step=step,epochs=epochs));torch.cuda.empty_cache()
        decision=nominal_gate.run_stage(OUT,completed_epoch,step,cand,gpu=int(args.eval_gpus),procs=args.eval_procs_per_gpu,min_stop_epoch=args.min_stop_epoch,smoke=args.smoke,mirror=args.mirror_dir)
        if decision['stop'] and not args.smoke:
            atomic_json(OUT/'status.json',dict(status='complete',completion_reason=decision['stop_reason'],epochs_completed=completed_epoch,max_epochs=epochs,step=step,pid=os.getpid(),selected_checkpoint='selected_development.pt',decision=decision))
            return True
        return False
    def validate(epoch,full_subsets):
        ema.eval();ema.use_aug=False
        nval=val['n'] if not args.smoke else 16
        vals=eval_loss(ema,val,np.arange(nval),919)
        vals['action_err_rad'],vals['action_err_finite']=action_error(ema,val,aerr_idx[:8] if args.smoke else aerr_idx)
        vr=dict(epoch=epoch,step=step,**vals)
        with open(OUT/'validation.jsonl','a') as f:f.write(json.dumps(vr)+'\n')
        print('VALIDATION',json.dumps(vr),flush=True)
        if full_subsets:
            t0=time.time();sub={}
            for gi,(k,idx) in enumerate(groups.items()):
                data=train if k.startswith('train_') else val
                sub[k]=eval_loss(ema,data,idx,919+gi)['bc_loss']
            vo=[k for k in sub if k.startswith('val_cell')]
            n_new=sum(len(groups[k]) for k in vo)
            if n_new:sub['val_new900']=sum(sub[k]*len(groups[k]) for k in vo)/n_new
            row=dict(epoch=epoch,step=step,bc_loss=sub,seconds=time.time()-t0)
            with open(OUT/'validation_subsets.jsonl','a') as f:f.write(json.dumps(row)+'\n')
            print('VAL_SUBSETS',json.dumps(row),flush=True)
        return vr
    started=time.time();starting_step=step
    if start_offset==0 and start_epoch in gate_epochs and not (OUT/'development_v2'/f'decision_{start_epoch:04d}.json').exists():
        print(f'RESUME_GATE epoch {start_epoch}',flush=True)
        if gate(start_epoch,None):return
    for epoch in range(start_epoch,epochs):
        model.train();model.use_aug=True;order=np.random.default_rng(420000+1000*(args.seed-42)+epoch).permutation(train['n'])
        for offset in range(start_offset if epoch==start_epoch else 0,train['n'],effective_batch):
            indices=order[offset:offset+effective_batch];optimizer.zero_grad(set_to_none=True);losses={}
            for b,count in prefetch_batches(train,indices,args.microbatch):
                loss,parts=model.compute_loss(b)
                (loss*(count/len(indices))).backward()
                for k,v in parts.items():losses[k]=losses.get(k,0)+v*count/len(indices)
            encoder_grad=gradient_health(model)
            keys=list(losses);numbers=torch.stack([torch.as_tensor(losses[k],device='cuda') for k in keys]).cpu().tolist();losses=dict(zip(keys,numbers))
            if not all(math.isfinite(v) for v in numbers):raise RuntimeError('Non-finite loss')
            optimizer.step();scheduler.step();em.step(model);step+=1
            row=dict(status='training',pid=os.getpid(),epoch=epoch+1,epochs=epochs,step=step,total_steps=total,
                **losses,lr=optimizer.param_groups[0]['lr'],encoder_grad_norm=encoder_grad,
                elapsed_s=time.time()-started,seconds_per_update=(time.time()-started)/max(1,step-starting_step),gpu_peak_GB=torch.cuda.max_memory_allocated()/1e9)
            with open(OUT/'metrics.jsonl','a') as f:f.write(json.dumps(row)+'\n')
            atomic_json(OUT/'status.json',row);print(json.dumps(row),flush=True)
            if step==1:save('last.pt',epoch,offset+len(indices))
            if args.smoke and step>=args.smoke_updates:
                vr=validate(epoch+1,True)
                tmp=OUT/'best.tmp';torch.save(dict(ema=ema.state_dict(),config=config,epoch=epoch+1,step=step,validation=vr),tmp);tmp.replace(OUT/'best.pt')
                save('last.pt',epoch,offset+len(indices))
                gate(epoch+1,vr)
                atomic_json(OUT/'status.json',dict(status='smoke_complete',step=step,full_training_started=False))
                print('SMOKE_PASSED',flush=True);return
        e1=epoch+1
        vr=validate(e1,e1%args.subset_every==0 or e1 in gate_epochs or e1==1)
        if vr['bc_loss']<best:
            best=vr['bc_loss'];tmp=OUT/'best.tmp';torch.save(dict(ema=ema.state_dict(),config=config,epoch=e1,step=step,validation=vr),tmp);tmp.replace(OUT/'best.pt')
        save('last.pt',e1,0)
        if e1 in gate_epochs:
            if gate(e1,vr):return
    atomic_json(OUT/'status.json',dict(status='complete',completion_reason='max_epochs',epochs_completed=epochs,step=step,best_validation_loss=best,pid=os.getpid(),selected_checkpoint='selected_development.pt'))
    print('TRAINING_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT is not None and (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text()).get('pid')==os.getpid():
            atomic_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),error=repr(e),traceback=traceback.format_exc()))
        print('TRAINING_FAILED',repr(e),flush=True)
        raise
