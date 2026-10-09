"""Opt-in central camera. Call after env creation/reset and before observing. No policy weights changed."""
import numpy as np
import mujoco
POSITION=np.array([0.,.25,1.5])
TARGET=np.array([0.,-.4,.87])
FOVY=50.
def configure(env):
    cid=env.model.camera('global').id
    back=POSITION-TARGET;back/=np.linalg.norm(back)
    right=np.cross([0.,0.,1.],back);right/=np.linalg.norm(right)
    up=np.cross(back,right)
    env.model.cam_pos[cid]=POSITION
    mujoco.mju_mat2Quat(env.model.cam_quat[cid],np.stack([right,up,back],axis=1).ravel())
    env.model.cam_fovy[cid]=FOVY
    env.model.vis.quality.offsamples=0
    mujoco.mj_forward(env.model,env.data)
    return env
