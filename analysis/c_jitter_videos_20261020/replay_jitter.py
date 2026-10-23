# [2026-10-20] NEW, CPU-only (no CUDA, MUJOCO_GL=osmesa). Open-loop replay of SAVED v1 collect actions (same init/step logic as c_unsettled_audit/pos_check.py)
# rendered in the style of replay_record_ext.py: 1.5s extension (policy last target held), magenta border in extension, |lin v|/|ang v| overlay, half speed (fps15),
# PLUS a strip plotting |ang v| vs the 0.15 rad/s threshold.
import os,sys,json
os.environ["CUDA_VISIBLE_DEVICES"]="";os.environ.setdefault("MUJOCO_GL","egl")
ROOT="/data02/kemove/rl100-sim-repro"
sys.path.insert(0,ROOT+"/train/bc_stage1");sys.path.insert(0,ROOT+"/env/logistics-sorting-sim/src");sys.path.insert(0,ROOT+"/env/sku09-task-success")
import numpy as np,mujoco,imageio,torch
from PIL import Image,ImageDraw
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
from action_codec import step_policy_action
from task_success import assess_task
LIN_T,ANG_T=.03,.15
out=sys.argv[1];EXT=45
for seed in sys.argv[2:]:
    f=f"{ROOT}/runs/offline_multisku_v1_20261019_2100/round0/collect/eps/ep_{seed}.npz";z=np.load(f);meta=json.loads(str(z["meta"]))
    cp=torch.load(meta["policy"],map_location="cpu",weights_only=False);e_=cp["ema"]
    sc=e_["normalizer.params_dict.action.scale"].float().numpy();off=e_["normalizer.params_dict.action.offset"].float().numpy();sem=cp["config"].get("action_semantics","absolute_joint_target_v1")
    acts=((z["actions"].astype(np.float32)-off)/sc).astype(np.float32);acts[:,:,[6,13]]=.012
    acts=np.concatenate([acts,np.repeat(acts[-1:,-1:],EXT,0).repeat(8,1)],0)
    sku=meta["sku"];frames=[];LIN=[];ANG=[];ST=[];keyidx=[]
    with LogisticsSortingEnv(sku=sku,render=True,image_size=(320,240)) as env:
        env.reset(seed=0);e=env.physics;m,d=e.model,e.data;adr=m.jnt_qposadr[m.joint("parcel_free").id]
        d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2);d.qpos[adr:adr+2]+=[meta["dx"],.05+meta["dy"]];yw=np.pi/2+np.deg2rad(meta["yaw_deg"])
        d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0];m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
        body=m.body("parcel").id;pos0=None
        def frame(o,t,s,phase,lin,ang):
            im=np.concatenate([o["rgb"][k] for k in ["global","left_wrist","right_wrist"]],1);img=Image.new("RGB",(960,240+90));img.paste(Image.fromarray(im),(0,0));dr=ImageDraw.Draw(img)
            dr.rectangle([0,0,959,30],fill=(0,0,0))
            dr.text((3,2),f"{sku} seed {seed} (collect cand0300, replay) t={t:.3f}s label_up={s[chr(108)+chr(97)+chr(98)+chr(101)+chr(108)+chr(95)+chr(117)+chr(112)]:+.3f} z_bot={s[chr(98)+chr(111)+chr(116)+chr(116)+chr(111)+chr(109)]:.3f}  {phase}",fill=(255,255,0))
            dr.text((3,16),f"|lin v|={lin:.3f} m/s (<{LIN_T})",fill=(0,255,0) if lin<LIN_T else (255,80,80));dr.text((230,16),f"|ang v|={ang:.3f} rad/s (<{ANG_T})",fill=(0,255,0) if ang<ANG_T else (255,80,80))
            n48=sum(ST[-48:]);dr.text((480,16),f"stable steps in last 48: {n48}/48",fill=(255,255,255))
            if phase.startswith("EXT"):dr.rectangle([0,0,959,239],outline=(255,0,255),width=3)
            # ang-vel strip: x = step (0..480+360), y = |ang v| 0..0.4
            Y0=240;H=90;W=960;N=480+EXT*8;ymax=.4
            dr.rectangle([0,Y0,W,Y0+H],fill=(20,20,20))
            X=lambda i:int(i/N*(W-1));Yf=lambda v:Y0+H-3-int(min(v,ymax)/ymax*(H-8))
            dr.line([0,Yf(ANG_T),W,Yf(ANG_T)],fill=(255,60,60),width=1);dr.text((3,Yf(ANG_T)-11),"ang v threshold 0.15 rad/s",fill=(255,60,60))
            dr.line([X(480),Y0,X(480),Y0+H],fill=(255,0,255));dr.text((X(480)+3,Y0+2),"episode end (2.0s)",fill=(255,0,255));dr.text((3,Y0+2),f"|ang v| (y 0..{ymax})",fill=(180,180,180))
            dr.line([X(432),Y0,X(432),Y0+H],fill=(90,90,90));dr.text((X(432)-95,Y0+2),"strict window ->",fill=(150,150,150))
            pts=[(X(i),Yf(a)) for i,a in enumerate(ANG)]
            if len(pts)>1:dr.line(pts,fill=(80,200,255),width=1)
            for i,a in enumerate(ANG):
                if a>=ANG_T:dr.ellipse([X(i)-2,Yf(a)-2,X(i)+2,Yf(a)+2],fill=(255,80,80))
            frames.append(np.asarray(img))
        obs=env.observe();s0=task_state(e);frame(obs,0.,s0,"EPISODE",0.,0.)
        pos0=np.array(d.xpos[body]);dmax=0.
        for c in range(60+EXT):
            for aa in acts[c]:
                step_policy_action(e,aa,sem);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
                lin=float(np.linalg.norm(v[3:]));ang=float(np.linalg.norm(v[:3]));LIN.append(lin);ANG.append(ang);ST.append(s["label_up"]>.98 and s["bottom"]>.79 and lin<LIN_T and ang<ANG_T)
            obs=env.observe();ph="EPISODE" if c<59 else ("EPISODE END (strict judged on last 48 steps of this chunk)" if c==59 else f"EXTENSION +{(c-59)*8/240:.2f}s (last target held)")
            frame(obs,float(d.time),s,ph,lin,ang)
        pos1=np.array(d.xpos[body])
    ST=np.array(ST);ANGa=np.array(ANG)
    tag=f"SKU5C_s{seed}_jitter_ext";os.makedirs(out,exist_ok=True)
    imageio.mimwrite(f"{out}/{tag}.mp4",frames,fps=15,macro_block_size=1)
    idx=[0,15,30,45,54,58,60,62,68,75,90,len(frames)-1];idx=[i for i in idx if i<len(frames)]
    Image.fromarray(np.concatenate([frames[i] for i in idx],0)).resize((960*2//3,len(idx)*330*2//3)).save(f"{out}/{tag}_keyframes.png")
    np.savez_compressed(f"{out}/{tag}_trace.npz",lin=np.array(LIN),ang=ANGa,stable=ST)
    w=slice(432,480);r=dict(seed=int(seed),sku=sku,policy=meta["policy"],rec_strict=meta["strict_reason"],rec_stable_last48=meta["stable_steps_last_48"],rep_stable_last48=int(ST[w].sum()),ang_max_last48=float(ANGa[w].max()),n_ang_over_0p15_last48=int((ANGa[w]>=ANG_T).sum()),lin_max_last48=float(np.array(LIN)[w].max()),ang_over_steps_last48=[int(i) for i in np.where(ANGa[w]>=ANG_T)[0]],ang_end_ext=float(ANGa[-1]),disp_mm_total=float(np.linalg.norm(pos1-pos0)*1000))
    print(json.dumps(r),flush=True);open(f"{out}/jitter_summary.jsonl","a").write(json.dumps(r)+"\n")
