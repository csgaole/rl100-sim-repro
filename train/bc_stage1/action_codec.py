"""Versioned policy action semantics. All environment actuators remain absolute.

Deltas apply at each 240 Hz physical step (never relative to the chunk start).
Gripper columns are absolute fixed widths in metres, not joint increments.
"""
import numpy as np
ABSOLUTE='absolute_joint_target_v1'
DELTA_TARGET='joint_delta_previous_effective_target_v1'
DELTA_MEASURED='joint_delta_measured_position_v1'
MODES={'absolute':ABSOLUTE,'delta_target':DELTA_TARGET,'delta_measured':DELTA_MEASURED}
JOINTS=np.array([0,1,2,3,4,5,7,8,9,10,11,12]);GRIPPERS=np.array([6,13])
WIDTH=.012

def validate(semantics):
 if semantics not in MODES.values():raise ValueError(f'Unsupported action semantics: {semantics}')
 return semantics

def checkpoint_semantics(config):
 semantics=config.get('action_semantics')
 if semantics is None:
  if 'delta' in config.get('run_family','').lower():raise ValueError('Delta checkpoint missing explicit action semantics')
  return ABSOLUTE # Historical checkpoints are explicitly legacy absolute only.
 return validate(semantics)

def controller_command(physics):
 return np.concatenate([np.r_[physics.command[i],WIDTH] for i in range(2)])

def encode(absolute,reference,semantics):
 validate(semantics);out=np.array(absolute,dtype=np.float64,copy=True)
 if semantics!=ABSOLUTE:out[...,JOINTS]-=np.asarray(reference)[...,JOINTS]
 out[...,GRIPPERS]=WIDTH
 return out

def decode(action,reference,semantics):
 validate(semantics);out=np.array(action,dtype=np.float64,copy=True)
 if out.shape!=(14,) or not np.isfinite(out).all():raise ValueError('Action must be 14 finite values')
 if semantics!=ABSOLUTE:out[JOINTS]+=np.asarray(reference)[JOINTS]
 out[GRIPPERS]=WIDTH
 return out

def absolute_target(physics,action,semantics):
 validate(semantics)
 reference=physics.proprio()[:14] if semantics==DELTA_MEASURED else controller_command(physics)
 return decode(action,reference,semantics)

def step_policy_action(physics,action,semantics):
 # Read the reference afresh on EVERY physical step. physics.step applies limits
 # and updates physics.command to the target actually sent to the actuators.
 return physics.step(absolute_target(physics,action,semantics))

def project_numpy(action,previous,lo,hi,dt,semantics,measured=None):
 validate(semantics)
 if semantics==DELTA_MEASURED and measured is None:raise ValueError('Measured-reference delta requires per-step measured joints')
 reference=measured if semantics==DELTA_MEASURED else previous
 out=decode(action,reference,semantics)
 target=np.clip(out[JOINTS],np.asarray(lo).reshape(-1),np.asarray(hi).reshape(-1))
 out[JOINTS]=np.asarray(previous)[JOINTS]+np.clip(target-np.asarray(previous)[JOINTS],-3*dt,3*dt)
 return out
