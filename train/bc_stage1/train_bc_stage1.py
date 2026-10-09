"""rl100-sim-repro Stage-1 BC. Copy of the proven sku09 A100 trainer (bc_epoch252 recipe);
only data/output paths, run_family and the development-gate sampler differ. Loss math unchanged.
Single-GPU stage-1 BC launcher. No critic, dynamics, PPO, or distillation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,copy,json,math,random,time,traceback,hashlib,sys
from pathlib import Path
import numpy as np,h5py,torch
from bc_model import BCPolicy,ROOT,R3M_PATH
from rl_100.model.diffusion.ema_model import EMAModel
from diffusers.optimization import get_scheduler
from stage_gate import run_stage
from a100_runtime import install_deferred_metrics,prefetch_batches,gradient_health
DATA=ROOT/'data/demos/sku09-original300'  # hard links: READ ONLY
OUT=ROOT/'runs/bc_stage1'
CACHE=OUT/'cache'  # built by prepare_cache.py from DATA (actions/effective)
RUN_FAMILY='rl100_sim_repro_bc_stage1_v1'


def atomic_json(path,obj):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def load_data(split):
    ids=json.loads((DATA/'splits.json').read_text())[split]
    marker=CACHE/f'{split}_complete.json'
    if not marker.exists():
        raise RuntimeError('Verified demonstration cache is required; prepare it before this fresh A100 run')
    assert json.loads(marker.read_text())['ids']==ids
    assert json.loads(marker.read_text())['manifest_sha256']==hashlib.sha256((DATA/'manifest.jsonl').read_bytes()).hexdigest()
    return dict(rgb=np.load(CACHE/f'{split}_rgb.npy',mmap_mode='r'),state=np.load(CACHE/f'{split}_state.npy'),action=np.load(CACHE/f'{split}_action.npy'),n=len(ids)*60)

def batch(data,indices):
    ep=indices//60;t=indices%60;prev=np.maximum(t-1,0)
    images=np.stack([data['rgb'][ep,prev],data['rgb'][ep,t]],1)
    obs={c:torch.as_tensor(np.ascontiguousarray(images[:,:,k]),device='cuda',dtype=torch.float32)/255 for k,c in enumerate(['global','left_wrist','right_wrist'])}
    obs['agent_pos']=torch.as_tensor(np.stack([data['state'][ep,prev],data['state'][ep,t]],1),device='cuda')
    a=data['action'][ep,t]
    # Official no_pre_action=True removes the first n_obs_steps-1 actions.
    # Prefix is structural only: loss always sees exactly the future 8 actions.
    a=np.concatenate([a[:,:1],a],1)
    return dict(obs=obs,action=torch.as_tensor(a,device='cuda'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--microbatch',type=int,choices=[8,16,32,64,128],default=32);ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true');ap.add_argument('--precision',choices=['fp32','tf32'],default='fp32');args=ap.parse_args()
    if not args.resume and any((OUT/n).exists() for n in ['last.pt','best.pt','metrics.jsonl','validation.jsonl','development']):raise RuntimeError('Fresh run directory already has training state; use own-run --resume or archive it explicitly')
    if args.resume and not (OUT/'last.pt').exists():raise RuntimeError('No native A100 checkpoint; old-run resume is forbidden')
    if 'A100' not in torch.cuda.get_device_name():raise RuntimeError('This entrypoint is reserved for A100')
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=args.precision=='tf32';torch.backends.cudnn.allow_tf32=args.precision=='tf32';torch.backends.cudnn.benchmark=False
    atomic_json(OUT/'status.json',dict(status='preparing',pid=os.getpid(),device=torch.cuda.get_device_name()))
    train=load_data('train');val=load_data('validation')
    normalizer_data=dict(agent_pos=train['state'].reshape(-1,28),action=train['action'].reshape(-1,14))
    model=BCPolicy(normalizer_data).cuda();install_deferred_metrics(model);ema=copy.deepcopy(model);em=EMAModel(ema,power=.75,inv_gamma=1.,max_value=.9999)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    epochs=600;effective_batch=512;steps_epoch=math.ceil(train['n']/effective_batch);total=epochs*steps_epoch
    scheduler=get_scheduler('cosine',optimizer,num_warmup_steps=500,num_training_steps=total)
    config=dict(stage='BC_only',run_family=RUN_FAMILY,action_semantics='absolute_joint_target_v1',action_field='actions/effective',data_root=str(DATA),output_dir=str(OUT),repo='rl100-sim-repro',trainer_source='copied from sku09-original300-absolute-bc/provenance/train_bc_a100.py (paths/gate sampler only)',initialization='fresh_seed42_R3M_encoder_random_UNet_and_aux_heads_no_old_BC_weights',upstream_commit='6b6afceb7eb12525398207b5d6e66908863dc385',recipe='official 2D diffusion chunk launcher + base YAML',
        device=torch.cuda.get_device_name(),seed=42,epochs=epochs,effective_batch=effective_batch,microbatch=args.microbatch,
        optimizer_updates=total,steps_per_epoch=steps_epoch,lr=2e-4,betas=[.95,.999],weight_decay=1e-6,lr_warmup_steps=500,
        obs_steps=2,action_steps=8,horizon=9,action_dim=14,proprio_dim=28,clock_input=False,precision=args.precision,
        network='official ConditionalUnet1D [256,512,1024], independent R3M ResNet18 per camera, GroupNorm',
        parameters=sum(p.numel() for p in model.parameters()),recon_weight=.05,kl_beta=.0005,kl_annealing=False,
        augmentation='official random shifts pad4; encoder resize224; no crop',ema='official power=.75 max=.9999',
        train_episodes=240,val_episodes=30,test_episodes=30,test_accessed=False,
        checkpoint_selection='validation-best EMA through each stage; development successes then validation-loss tie-break; final test untouched',
        stopping_protocol='development_v2_upstream_sampler: first100 then50 epochs; 100 fixed development seeds880000..880099 with upstream DDIM10 eta=1 sampler; patience3; success_delta2/100 or validation relative decrease1%; max600',
        adaptation='A100 native: pinned prefetch; copy stream; tensor-only metric returns (.item to .detach AST adaptation); batched gradient checks; unchanged loss math; no official full train entrypoint',
        r3m_sha256=hashlib.sha256(R3M_PATH.read_bytes()).hexdigest())
    atomic_json(OUT/'config.json',config);print('CONFIG',json.dumps(config),flush=True)
    step=0;start_epoch=0;start_offset=0;best=float('inf')
    if args.resume:
        cp=torch.load(OUT/'last.pt',map_location='cuda',weights_only=False)
        assert cp['config'].get('run_family')==RUN_FAMILY, 'Refuse foreign checkpoint'
        assert cp['config']['precision']==args.precision and cp['config']['microbatch']==args.microbatch, 'Resume must preserve precision and microbatch'
        model.load_state_dict(cp['model']);ema.load_state_dict(cp['ema']);optimizer.load_state_dict(cp['optimizer']);scheduler.load_state_dict(cp['scheduler'])
        step=cp['step'];start_epoch=cp['next_epoch'];start_offset=cp['next_offset'];best=cp['best'];em.optimization_step=cp['ema_step']
        torch.set_rng_state(cp['torch_rng'].cpu());torch.cuda.set_rng_state(cp['cuda_rng'].cpu());np.random.set_state(cp['numpy_rng']);random.setstate(cp['python_rng'])
    def save(name,next_epoch,next_offset):
        payload=dict(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            step=step,next_epoch=next_epoch,next_offset=next_offset,best=best,ema_step=em.optimization_step,config=config,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        p=OUT/name;tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
    started=time.time();starting_step=step
    def gate(completed_epoch):
        atomic_json(OUT/'status.json',dict(status='development_evaluation',pid=os.getpid(),epoch=completed_epoch,step=step,epochs=epochs))
        decision=run_stage(OUT,completed_epoch)
        if decision['stop']:
            atomic_json(OUT/'status.json',dict(status='complete',completion_reason='development_plateau',epochs_completed=completed_epoch,max_epochs=epochs,step=step,pid=os.getpid(),selected_checkpoint='selected_development.pt',decision=decision))
            return True
        return False
    # A crash during evaluation resumes that gate before another training epoch.
    if start_offset==0 and start_epoch>=100 and start_epoch%50==0:
        if gate(start_epoch):return
    for epoch in range(start_epoch,epochs):
        model.train();model.use_aug=True;order=np.random.default_rng(420000+epoch).permutation(train['n'])
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
            if args.smoke:print('SMOKE_PASSED',flush=True);return
        ema.eval();ema.use_aug=False;vals={}
        with torch.no_grad(),torch.random.fork_rng(devices=[0]):
            torch.manual_seed(919);torch.cuda.manual_seed_all(919)
            for j in range(0,val['n'],8):
                ids=np.arange(j,min(val['n'],j+8));loss,parts=ema.compute_loss(batch(val,ids))
                for k,v in parts.items():vals[k]=vals.get(k,0)+v*len(ids)/val['n']
        vals={k:float(v) for k,v in vals.items()}
        vr=dict(epoch=epoch+1,step=step,**vals)
        with open(OUT/'validation.jsonl','a') as f:f.write(json.dumps(vr)+'\n')
        print('VALIDATION',json.dumps(vr),flush=True)
        if vals['bc_loss']<best:
            best=vals['bc_loss'];tmp=OUT/'best.tmp';torch.save(dict(ema=ema.state_dict(),config=config,epoch=epoch+1,step=step,validation=vr),tmp);tmp.replace(OUT/'best.pt')
        save('last.pt',epoch+1,0)
        if (epoch+1)%50==0:
            import shutil
            shutil.copy2(OUT/'best.pt',OUT/f'best_through_epoch_{epoch+1:04d}.pt')
        if epoch+1>=100 and (epoch+1)%50==0:
            if gate(epoch+1):return
    atomic_json(OUT/'status.json',dict(status='complete',completion_reason='max_epochs',epochs_completed=epochs,step=step,best_validation_loss=best,pid=os.getpid(),selected_checkpoint='selected_development.pt'))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if (OUT/'status.json').exists() and json.loads((OUT/'status.json').read_text()).get('pid')==os.getpid():
            atomic_json(OUT/'status.json',dict(status='failed',pid=os.getpid(),error=repr(e),traceback=traceback.format_exc()))
        raise
