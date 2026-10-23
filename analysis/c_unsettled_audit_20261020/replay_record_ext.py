"""[c_unsettled_audit 2026-10-20] NEW file. Copy of failure_review_e300/replay_record.py episode logic (identical IC, generator seed,
upstream eta=1 sampler -> bit-exact to evaluate_bc_multisku gate episodes), plus: (1) overlay of parcel |lin v| / |ang v| vs the strict
settle thresholds (lin<0.03 m/s, ang<0.15 rad/s, label_up>.98, bottom>.79, all of last 48 steps), (2) EXTENSION past the 2.0 s episode end:
the policy keeps running for --ext-chunks more chunks (default 45 = 1.5 s) so residual motion is visible. Read-only wrt all run outputs."""
import os,sys,json,argparse,math
os.environ.setdefault('MUJOCO_GL','egl')
T='/data02/kemove/rl100-sim-repro/train/bc_multisku_v1_20261018';sys.path.insert(0,T)
import numpy as np,torch,mujoco,imageio
from PIL import Image,ImageDraw
from evaluate_bc_multisku import LogisticsSortingEnv,task_state,step_policy_action,assess_task,load_policy,sample
LIN_T,ANG_T=.03,.15
def flip_deg(lu):return math.degrees(math.acos(max(-1,min(1,-lu))))
def run(p,seed,sku,out,ext):
    rng=np.random.default_rng(seed);dx,dy=rng.uniform(-.005,.005,2);yaw_deg=float(rng.uniform(-2.,2.))
    frames=[];tr=dict(t=[],label_up=[],bottom=[],pos=[],forces=[],lin=[],ang=[],linv=[],stable=[],outfeed_contact=[],vy=[])
    with LogisticsSortingEnv(sku=sku,render=True,image_size=(320,240)) as env:
        env.reset(seed=0);e=env.physics;m,d=e.model,e.data
        adr=m.jnt_qposadr[m.joint('parcel_free').id]
        d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
        d.qpos[adr:adr+2]+=[dx,.05+dy];yaw=np.pi/2+np.deg2rad(yaw_deg)
        d.qpos[adr+3:adr+7]=[0,np.cos(yaw/2),np.sin(yaw/2),0]
        m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
        obs=env.observe();history=[obs,obs];gen=torch.Generator(device='cuda').manual_seed(seed+10000000)
        stable=[];drop=False;bid=m.body('parcel').id;cur=dict(lin=0.,ang=0.)
        def frame(o,t,s,phase):
            im=np.concatenate([o['rgb'][k] for k in ['global','left_wrist','right_wrist']],1);img=Image.fromarray(im);dr=ImageDraw.Draw(img)
            st=stable[-48:];n48=sum(st)
            dr.rectangle([0,0,959,30],fill=(0,0,0))
            dr.text((3,2),f'{sku} seed {seed} t={t:.3f}s label_up={s["label_up"]:+.3f} flip={flip_deg(s["label_up"]):.0f}deg F=[{s["forces"][0]:.0f},{s["forces"][1]:.0f}]N z_bot={s["bottom"]:.3f}  {phase}',fill=(255,255,0))
            lc=(0,255,0) if cur['lin']<LIN_T else (255,80,80);ac=(0,255,0) if cur['ang']<ANG_T else (255,80,80)
            dr.text((3,16),f'|lin v|={cur["lin"]:.3f} m/s (<{LIN_T})',fill=lc);dr.text((230,16),f'|ang v|={cur["ang"]:.3f} rad/s (<{ANG_T})',fill=ac)
            dr.text((480,16),f'stable steps in last 48: {n48}/48  vy={s["vy"]:+.3f}',fill=(255,255,255))
            if phase.startswith('EXT'):dr.rectangle([0,0,959,239],outline=(255,0,255),width=3)
            frames.append(np.asarray(img))
        frame(obs,0.,task_state(e),'EPISODE')
        outcome=None
        for chunk in range(60+ext):
            actions=sample(p,history,gen,eta=1.0)
            for a in actions:
                step_policy_action(e,a,p.action_semantics);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,v,0)
                lin=float(np.linalg.norm(v[3:]));ang=float(np.linalg.norm(v[:3]));cur.update(lin=lin,ang=ang)
                ok=s['label_up']>.98 and s['bottom']>.79 and lin<LIN_T and ang<ANG_T;stable.append(ok)
                if chunk<60:drop|=s['bottom']<.74
                tr['t'].append(float(d.time));tr['label_up'].append(s['label_up']);tr['bottom'].append(s['bottom']);tr['pos'].append(s['position']);tr['forces'].append(s['forces'])
                tr['lin'].append(lin);tr['ang'].append(ang);tr['linv'].append(v[3:].tolist());tr['stable'].append(ok);tr['outfeed_contact'].append(s['outfeed_contact']);tr['vy'].append(s['vy'])
            if chunk==59:outcome=assess_task(s['label_up'],s['bottom'],drop,len(stable)==480,stable[:480])
            obs=env.observe();history=[history[-1],obs]
            frame(obs,float(d.time),s,'EPISODE' if chunk<59 else ('EPISODE END (strict judged here)' if chunk==59 else f'EXTENSION +{(chunk-59)*8/240:.2f}s (policy keeps running)'))
    os.makedirs(out,exist_ok=True);tag=f'{sku}_s{seed}_unsettled_ext'
    imageio.mimwrite(f'{out}/{tag}.mp4',frames,fps=15,macro_block_size=1)  # half speed (camera 30 Hz)
    idx=[0,15,30,45,54,60,68,75,90,len(frames)-1]
    Image.fromarray(np.concatenate([frames[i] for i in idx],0)).resize((960*2//3,len(idx)*240*2//3)).save(f'{out}/{tag}_keyframes.png')
    np.savez_compressed(f'{out}/{tag}_trace.npz',**{k:np.array(v) for k,v in tr.items()})
    S=np.array(tr['stable']);L=np.array(tr['lin']);A=np.array(tr['ang']);P=np.array(tr['pos'])
    tail=np.flip(np.cumprod(np.flip(S))).astype(bool);perm=int(np.argmax(tail)) if tail.any() else None
    first48=next((i for i in range(48,len(S)+1) if S[i-48:i].all()),None)
    return dict(seed=seed,sku=sku,**{k:(bool(v) if isinstance(v,(bool,np.bool_)) else v) for k,v in outcome.items()},
        lin_end=float(L[479]),ang_end=float(A[479]),lin_max_last48=float(L[432:480].max()),ang_max_last48=float(A[432:480].max()),
        lin_end_ext=float(L[-1]),ang_end_ext=float(A[-1]),pos_end=P[479].tolist(),pos_end_ext=P[-1].tolist(),vy_end=float(tr['vy'][479]),outfeed_contact_end=bool(tr['outfeed_contact'][479]),
        first_full48_window_end_s=None if first48 is None else first48/240,permanently_stable_from_s=None if perm is None else perm/240,ext_s=ext*8/240,n_frames=len(frames))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint');ap.add_argument('--out');ap.add_argument('--jobs');ap.add_argument('--ext-chunks',type=int,default=45);a=ap.parse_args()
    torch.set_num_threads(8);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    p,cp=load_policy(a.checkpoint)
    for j in a.jobs.split(','):
        sku,seed=j.split(':');p.sku_cond=np.asarray(p.sku_table[sku],np.float32);r=run(p,int(seed),sku,a.out,a.ext_chunks);r['checkpoint']=a.checkpoint
        print(json.dumps(r),flush=True)
        with open(f'{a.out}/replay_ext_summary.jsonl','a') as fh:fh.write(json.dumps(r)+'\n')
