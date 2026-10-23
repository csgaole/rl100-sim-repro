"""[MULTI-SKU copy of rl100-sim-repro train/bc_stage1/evaluate_bc.py: --sku selects scene (SKU5A..E = scene_sku10..14); sku_cond fed from the checkpoint config table; everything else (p5_y2 IC, samplers, flip_pose_no_drop_v2) unchanged]
[rl100-sim-repro copy of sku09-original300-absolute-bc/src/evaluate_bc.py; only env/task_success paths changed]
Fixed independent development seeds. No expert control in inference."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,time,hashlib,sys,math
from pathlib import Path
import numpy as np,torch,mujoco
sys.path.insert(0,'/data02/kemove/rl100-sim-repro/train/bc_stage1');sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from bc_model_multisku import BCPolicy
ROOT=Path('/data02/kemove/rl100-sim-repro')
sys.path.insert(0,str(ROOT/'env/logistics-sorting-sim/src'))
sys.path.insert(0,str(ROOT/'env/sku09-task-success'))
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
from action_codec import checkpoint_semantics,step_policy_action
from task_success import assess_task,SUCCESS_VERSION

def load_policy(path):
    cp=torch.load(path,map_location='cpu',weights_only=False)
    # Shapes only; every normalization value is replaced from the checkpoint.
    p=BCPolicy(dict(agent_pos=np.zeros((2,28),np.float32),action=np.zeros((2,14),np.float32),sku_cond=np.zeros((2,4),np.float32)),pretrained=False)
    p.action_semantics=checkpoint_semantics(cp.get('config',{}))
    p.load_state_dict(cp['ema'],strict=True);p.cuda().eval();p.use_aug=False
    p.sku_table=cp['config']['sku_cond_table']
    return p,cp

@torch.inference_mode()
def sample(p,history,generator,eta=0.0):
    obs={k:torch.as_tensor(np.stack([h['rgb'][k] for h in history]).transpose(0,3,1,2),device='cuda',dtype=torch.float32)/255 for k in p.rgb_obs_keys}
    obs['agent_pos']=torch.as_tensor(np.stack([h['proprio'] for h in history]),device='cuda',dtype=torch.float32)
    obs['sku_cond']=torch.as_tensor(np.stack([p.sku_cond]*len(history)),device='cuda',dtype=torch.float32)
    obs=p.normalizer.normalize(obs);cond=p.obs_encoder(obs).reshape(1,-1)
    x=torch.randn((1,8,14),device='cuda',generator=generator)
    p.noise_scheduler.set_timesteps(10)
    for t in p.noise_scheduler.timesteps:
        eps=p.model(x,t.to('cuda'),global_cond=cond)
        x=p.noise_scheduler.step_mean(eps,t,x,eta=eta,generator=generator).prev_sample
    actions=p.normalizer['action'].unnormalize(x)[0].cpu().numpy()
    actions[:,[6,13]]=.012
    assert actions.shape==(8,14) and np.isfinite(actions).all()
    return actions

def episode(p,seed,sample_fn=None,position_half_range=.005,yaw_half_range=2.,sku='SKU09'):
    rng=np.random.default_rng(seed);dx,dy=rng.uniform(-position_half_range,position_half_range,2);yaw_deg=float(rng.uniform(-yaw_half_range,yaw_half_range))
    with LogisticsSortingEnv(sku=sku,render=True,image_size=(320,240)) as env:
        env.reset(seed=0);e=env.physics;m,d=e.model,e.data
        adr=m.jnt_qposadr[m.joint('parcel_free').id]
        d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
        d.qpos[adr:adr+2]+=[dx,.05+dy];yaw=np.pi/2+np.deg2rad(yaw_deg)
        d.qpos[adr+3:adr+7]=[0,np.cos(yaw/2),np.sin(yaw/2),0]
        m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
        obs=env.observe();history=[obs,obs];gen=torch.Generator(device='cuda').manual_seed(seed+10000000)
        stable=[];drop=False;peak=0.;latency=[]
        for chunk in range(60):
            torch.cuda.synchronize();start=time.perf_counter();actions=(sample_fn or sample)(p,history,gen);torch.cuda.synchronize();latency.append(time.perf_counter()-start)
            for a in actions:
                step_policy_action(e,a,p.action_semantics);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,m.body('parcel').id,v,0)
                stable.append(s['label_up']>.98 and s['bottom']>.79 and np.linalg.norm(v[3:])<.03 and np.linalg.norm(v[:3])<.15)
                drop|=s['bottom']<.74;peak=max(peak,max(s['forces']))
            obs=env.observe();history=[history[-1],obs]
        outcome=assess_task(s['label_up'],s['bottom'],drop,len(stable)==480,stable)
        times=[(j+48)/240 for j in range(433) if all(stable[j:j+48])]
        return dict(action_semantics=p.action_semantics,seed=seed,dx=float(dx),dy=float(dy),yaw_deg=yaw_deg,**outcome,
            first_stable_s=times[0] if times else None,peak_force_N=peak,final_up=s['label_up'],inference_latency_s=latency)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seed-start',type=int,default=880000);ap.add_argument('--episodes',type=int,default=100);ap.add_argument('--sampler',choices=['legacy','standard','upstream'],default='legacy');ap.add_argument('--position-half-range',type=float,default=.005);ap.add_argument('--yaw-half-range',type=float,default=2.);ap.add_argument('--sku',required=True);args=ap.parse_args()
    assert args.seed_start+args.episodes<=900000 or args.seed_start>900999,'Development must not use final test seeds'
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from bc_sampling import sample_standard_ddim
    sample_fn=sample_standard_ddim if args.sampler=='standard' else (lambda p,h,g:sample(p,h,g,eta=1.0)) if args.sampler=='upstream' else sample
    path=Path(args.checkpoint);dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    sha=hashlib.sha256(path.read_bytes()).hexdigest();p,cp=load_policy(path);rows=[]
    p.sku_cond=np.asarray(p.sku_table[args.sku],np.float32)
    raw=dest.with_suffix('.episodes.jsonl')
    # No duplicated partial results on retry; this file is an audit log, not a resume source.
    with open(raw,'w') as f:
        for seed in range(args.seed_start,args.seed_start+args.episodes):
            row=episode(p,seed,sample_fn=sample_fn,position_half_range=args.position_half_range,yaw_half_range=args.yaw_half_range,sku=args.sku);rows.append(row);f.write(json.dumps(row)+'\n');f.flush();print(seed,row['reason'],flush=True)
    k=sum(r['success'] for r in rows);n=len(rows);z=1.95996398454;ph=k/n;center=(ph+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n)
    report=dict(sku=args.sku,sku_cond=p.sku_cond.tolist(),success_criteria_version=SUCCESS_VERSION,strict_successes=sum(r['strict_success'] for r in rows),unsettled_successes=sum(r['stability_quality']=='unsettled_success' for r in rows),action_semantics=p.action_semantics,position_half_range_m=args.position_half_range,yaw_half_range_deg=args.yaw_half_range,sampler=args.sampler,successes=k,episodes=n,success_rate=ph,wilson95=[center-half,center+half],checkpoint_sha256=sha,
        checkpoint_epoch=cp.get('epoch',cp.get('next_epoch')),checkpoint_step=cp['step'],validation_loss=cp.get('validation',{}).get('bc_loss'),seed_start=args.seed_start,
        failure_counts={key:sum(r['reason']==key for r in rows) for key in ['drop','wrong_face','below_height','incomplete']},device=torch.cuda.get_device_name())
    tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(dest)
    print(json.dumps(report),flush=True)
if __name__=='__main__':main()
