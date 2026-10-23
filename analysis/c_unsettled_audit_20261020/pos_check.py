# [c_unsettled_audit 2026-10-20] NEW read-only analysis script (pos_check variant: adds final position / outfeed contact / vy / parcel contacts). CPU-only open-loop replay of SAVED collect actions
# (offline_multisku_v1 round*/collect/eps/ep_*.npz) with the exact init/step/stable logic of batch_worker_ms.py,
# logging per-step parcel velocity, then a hold-last-target extension to see whether the parcel settles.
import os,sys,json,glob,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ.setdefault('MUJOCO_GL','egl')
ROOT='/data02/kemove/rl100-sim-repro'
sys.path.insert(0,ROOT+'/train/bc_stage1');sys.path.insert(0,ROOT+'/env/logistics-sorting-sim/src');sys.path.insert(0,ROOT+'/env/sku09-task-success')
import numpy as np,mujoco
from logistics_sorting_sim import LogisticsSortingEnv
from logistics_sorting_sim.task import task_state
from action_codec import step_policy_action
from task_success import assess_task
ap=argparse.ArgumentParser();ap.add_argument('--files',required=True);ap.add_argument('--out',required=True);ap.add_argument('--ext',type=int,default=480);a=ap.parse_args()
NORM={}
def norm_for(pol):
    if pol not in NORM:
        import torch
        cp=torch.load(pol,map_location='cpu',weights_only=False);e=cp['ema']
        NORM[pol]=(e['normalizer.params_dict.action.scale'].float().numpy(),e['normalizer.params_dict.action.offset'].float().numpy(),cp['config'].get('action_semantics','absolute_joint_target_v1'))
    return NORM[pol]
def stab(s,v):
    lin=float(np.linalg.norm(v[3:]));ang=float(np.linalg.norm(v[:3]))
    return (s['label_up']>.98 and s['bottom']>.79 and lin<.03 and ang<.15),lin,ang
files=[l.strip() for l in open(a.files) if l.strip()]
with open(a.out,'w') as fo:
    for f in files:
        z=np.load(f);meta=json.loads(str(z['meta']));sc,off,sem=norm_for(meta['policy'])
        acts=((z['actions'].astype(np.float32)-off)/sc).astype(np.float32);acts[:,:,[6,13]]=.012
        env=LogisticsSortingEnv(sku=meta['sku'],render=False,image_size=(320,240));env.reset(seed=0);e=env.physics;m,d=e.model,e.data
        adr=m.jnt_qposadr[m.joint('parcel_free').id]
        d.qpos[adr:adr+2]-=np.random.default_rng(0).uniform(-.005,.005,2)
        d.qpos[adr:adr+2]+=[meta['dx'],.05+meta['dy']];yw=np.pi/2+np.deg2rad(meta['yaw_deg'])
        d.qpos[adr+3:adr+7]=[0,np.cos(yw/2),np.sin(yw/2),0]
        m.actuator_gainprm[e.aidx.ravel(),0]=1200;m.actuator_biasprm[e.aidx.ravel(),1]=-1200;mujoco.mj_forward(m,d)
        body=m.body('parcel').id;ST=[];LIN=[];ANG=[];UP=[];BOT=[];FRC=[];POS=[];QUAT=[];OC=[];OI=[];VY=[];CN=[]
        drop=False;peak=0.
        def rec(aa):
            global drop,peak
            step_policy_action(e,aa,sem);s=task_state(e);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,body,v,0)
            ok,lin,ang=stab(s,v);ST.append(ok);OC.append(bool(s['outfeed_contact']));OI.append(bool(s['outfeed_inside']));VY.append(float(s['vy']));CN.append(sorted({n for ci in range(d.ncon) for n in [m.geom(int(g)).name or str(int(g)) for g in d.contact[ci].geom] if 'parcel_box' in [m.geom(int(g)).name for g in d.contact[ci].geom] and n!='parcel_box'}));LIN.append(lin);ANG.append(ang);UP.append(s['label_up']);BOT.append(s['bottom']);FRC.append(max(s['forces']));POS.append(list(d.xpos[body]));QUAT.append(list(d.xquat[body]))
            return s
        drop=False;peak=0.
        for c in range(60):
            for aa in acts[c]:
                s=rec(aa);drop|=s['bottom']<.74;peak=max(peak,max(s['forces']))
        out=assess_task(s['label_up'],s['bottom'],drop,len(ST)==480,ST)
        last=acts[-1,-1]
        for k in range(a.ext):rec(last)   # extension: hold the final absolute joint target
        ST=np.array(ST);LIN=np.array(LIN);ANG=np.array(ANG);UP=np.array(UP);FRC=np.array(FRC);POS=np.array(POS);Q=np.array(QUAT)
        def win_ok(end):return bool(ST[end-48:end].all())
        def qang(i,j):
            dot=abs(float(np.dot(Q[i],Q[j])));return float(2*np.degrees(np.arccos(min(1.,dot))))
        w=slice(432,480)
        # earliest settle: first index t>=48 such that ST[t-48:t] all true AND stays stable through end of extension
        tail_ok=np.flip(np.cumprod(np.flip(ST))).astype(bool)
        first_perm=int(np.argmax(tail_ok)) if tail_ok.any() else None
        r=dict(file=f,round=meta.get('collect_round'),seed=meta['seed'],sku=meta['sku'],rec_strict=meta['strict_reason'],rec_loose=meta['loose_reason'],
            rep_strict=out['strict_reason'],rep_loose=out['reason'],rec_last48=meta['stable_steps_last_48'],rep_last48=int(ST[w].sum()),
            rec_final_up=meta['final_up'],rep_final_up=float(UP[479]),rec_peak=meta['peak_force_N'],rep_peak=float(peak),
            lin_end=float(LIN[479]),ang_end=float(ANG[479]),lin_max48=float(LIN[w].max()),ang_max48=float(ANG[w].max()),
            lin_mean48=float(LIN[w].mean()),ang_mean48=float(ANG[w].mean()),up_min48=float(UP[w].min()),up_end=float(UP[479]),bot_min48=float(np.array(BOT)[w].min()),
            fail_lin48=int((LIN[w]>=.03).sum()),fail_ang48=int((ANG[w]>=.15).sum()),fail_up48=int((UP[w]<=.98).sum()),fail_bot48=int((np.array(BOT)[w]<=.79).sum()),
            grip_force_end=float(FRC[479]),grip_force_max48=float(FRC[w].max()),
            rot_deg_last48=qang(432,479),disp_mm_last48=float(np.linalg.norm(POS[479]-POS[432])*1000),
            rot_deg_ext=qang(479,479+a.ext),disp_mm_ext=float(np.linalg.norm(POS[479+a.ext]-POS[479])*1000),up_end_ext=float(UP[479+a.ext]),
            pos_end=POS[479].tolist(),pos_end_ext=POS[-1].tolist(),outfeed_contact_end=OC[479],outfeed_inside_end=OI[479],vy_end=VY[479],contacts_end=CN[479],outfeed_contact_frac48=float(np.mean(OC[432:480])),contacts_ext_end=CN[-1],ok_end_plus={str(k):win_ok(480+k) for k in (0,12,24,48,96,120,240,480) if 480+k<=len(ST)},
            first_permanent_stable_step=first_perm,
            ang_trace_last96=[round(x,4) for x in ANG[384:480].tolist()],lin_trace_last96=[round(x,4) for x in LIN[384:480].tolist()],
            ang_ext_every12=[round(x,4) for x in ANG[480::12].tolist()],lin_ext_every12=[round(x,4) for x in LIN[480::12].tolist()])
        fo.write(json.dumps(r)+'\n');fo.flush();env.close()
