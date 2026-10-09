"""MuJoCo RGB/proprio environment. Parcel state is evaluation-only."""
import json
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parent
class SortingEnv:
 def __init__(self,sku=1,seed=0,render=False):
  self.sku=sku;self.rng=np.random.default_rng(seed);self.model=mujoco.MjModel.from_xml_path(str(ROOT/f'scene_sku{sku:02d}.xml'));self.data=mujoco.MjData(self.model);self.dt=self.model.opt.timestep
  self.joint_ids=[[self.model.joint(f'arm{i}_joint{j}').id for j in range(1,7)] for i in range(2)]
  self.qidx=np.array([[self.model.jnt_qposadr[j] for j in arm] for arm in self.joint_ids]);self.vidx=np.array([[self.model.jnt_dofadr[j] for j in arm] for arm in self.joint_ids])
  self.aidx=np.array([[self.model.actuator(f'arm{i}_joint{j}_drive').id for j in range(1,7)] for i in range(2)])
  self.grip_q=np.array([[self.model.jnt_qposadr[self.model.joint(f'arm{i}_gripper_joint{j}').id] for j in [1,2]] for i in range(2)])
  self.grip_v=np.array([[self.model.jnt_dofadr[self.model.joint(f'arm{i}_gripper_joint{j}').id] for j in [1,2]] for i in range(2)])
  self.renderer=mujoco.Renderer(self.model,240,320) if render else None
  self.history=[];self.reset()
 def reset(self,label='down',location='table',randomize=False):
  mujoco.mj_resetData(self.model,self.data)
  q=np.array(json.loads((ROOT/'configs/park.json').read_text()))
  if randomize:q+=self.rng.uniform(-.015,.015,q.shape)
  self.data.qpos[self.qidx]=q;self.data.ctrl[self.aidx]=q
  for i in range(2):
   self.data.qpos[self.grip_q[i]]=[.006,-.006]
   for j,w in zip([1,2],[.006,-.006]):self.data.ctrl[self.model.actuator(f'arm{i}_gripper_joint{j}_drive').id]=w
  adr=self.model.jnt_qposadr[self.model.joint('parcel_free').id]
  if label=='up':self.data.qpos[adr+3:adr+7]=[1,0,0,0]
  if location=='infeed':self.data.qpos[adr:adr+2]=[-.8,-.2]
  if location=='outfeed':self.data.qpos[adr:adr+2]=[0.,-.8]
  mujoco.mj_forward(self.model,self.data);self.command=q.copy();self.command_v=np.zeros((2,6));self.peak_actual_a=0.;self.peak_command_a=0.;self.peak_tcp_a=np.zeros(2);self.previous_tcp_v=np.zeros((2,3));self.history=[]
 def proprio(self):
  q=self.data.qpos[self.qidx];v=self.data.qvel[self.vidx];g=self.data.qpos[self.grip_q];gv=self.data.qvel[self.grip_v]
  state=np.concatenate([np.r_[q[i],g[i,0]-g[i,1]] for i in range(2)]);vel=np.concatenate([np.r_[v[i],gv[i,0]-gv[i,1]] for i in range(2)])
  return np.r_[state,vel]
 def observe(self):
  images={}
  if self.renderer:
   for name in ['global','left_wrist','right_wrist']:
    self.renderer.update_scene(self.data,camera=name);images[name]=self.renderer.render().copy()
  return dict(rgb=images,proprio=self.proprio(),timestamp=float(self.data.time),rgb_timestamps={name:float(self.data.time) for name in images})
 def step(self,action):
  action=np.asarray(action);target=np.array([action[:6],action[7:13]])
  lo=self.model.jnt_range[np.array(self.joint_ids),0]+.005;hi=self.model.jnt_range[np.array(self.joint_ids),1]-.005;target=np.clip(target,lo,hi)
  desired=(target-self.command)/self.dt
  new_v=np.clip(desired,-3.,3.) # Acceleration is telemetry only; retain joint speed bound.
  command=self.command+new_v*self.dt;self.peak_command_a=max(self.peak_command_a,float(abs(new_v-self.command_v).max()/self.dt))
  self.data.ctrl[self.aidx]=command;oldv=self.data.qvel[self.vidx].copy();before=self.proprio();effective=np.concatenate([np.r_[command[i],.012] for i in range(2)])
  self.history.append((float(self.data.time),before.copy(),effective.copy()))
  mujoco.mj_step(self.model,self.data)
  mujoco.mj_forward(self.model,self.data) # Refresh post-integration poses before camera/state snapshots.
  self.peak_actual_a=max(self.peak_actual_a,float(abs(self.data.qvel[self.vidx]-oldv).max()/self.dt));self.command=command;self.command_v=new_v
  for i in range(2):
   jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(self.model,self.data,jp,jr,self.model.site(f'arm{i}_tcp').id)
   tcp_v=jp@self.data.qvel;self.peak_tcp_a[i]=max(self.peak_tcp_a[i],float(np.linalg.norm(tcp_v-self.previous_tcp_v[i])/self.dt));self.previous_tcp_v[i]=tcp_v
  if not np.isfinite(self.data.qpos).all():raise RuntimeError('Nonfinite physics state')
  return self.metrics()
 def metrics(self):
  b=self.model.body('parcel').id;pos=self.data.xpos[b];rot=self.data.xmat[b].reshape(3,3);size=self.model.geom_size[self.model.geom('parcel_box').id];ext=abs(rot)@size
  on_table=pos[0]-ext[0]>=-.2 and pos[0]+ext[0]<=.6 and pos[1]-ext[1]>=-.33 and pos[1]+ext[1]<=.03
  return dict(timestamp=float(self.data.time),position=pos.copy().tolist(),label_up=float(rot[2,2]),on_table=bool(on_table),push_cleared=bool(pos[1]+ext[1]<-.34),fell=bool(pos[2]-ext[2]<.74),actual_peak_acceleration=self.peak_actual_a,command_peak_acceleration=self.peak_command_a,tcp_peak_acceleration_m_s2=self.peak_tcp_a.tolist(),tcp_acceleration_limit_m_s2=None,tcp_acceleration_gated=False,joint_acceleration_limit_rad_s2=None,joint_acceleration_gated=False,constraint_policy="mujoco_no_acceleration_cap_v1")
 def close(self):
  if self.renderer:self.renderer.close()
