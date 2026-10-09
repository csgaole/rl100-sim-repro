"""Validated wrist mounting, fixed to respective link6; keep center main camera."""
import numpy as np,mujoco,json
from .main_camera_center_v2 import configure as configure_main
PROFILE=json.loads('{\n  "version": "center_wrist_v2_preview",\n  "coordinates": "local to fixed armN_link6, not sliding fingers",\n  "cameras": [\n    {\n      "arm": 0,\n      "pos": [\n        0.0,\n        -0.12,\n        0.0\n      ],\n      "target": [\n        0,\n        0,\n        0.24\n      ],\n      "fovy": 95,\n      "rotation": [\n        [\n          -1.0,\n          0.0,\n          0.0\n        ],\n        [\n          0.0,\n          0.8944271909999157,\n          -0.44721359549995787\n        ],\n        [\n          -0.0,\n          -0.44721359549995787,\n          -0.8944271909999157\n        ]\n      ],\n      "camera_name": "left_wrist",\n      "parent_body": "arm0_link6"\n    },\n    {\n      "arm": 1,\n      "pos": [\n        0.0,\n        0.12,\n        -0.03\n      ],\n      "target": [\n        0,\n        0,\n        0.24\n      ],\n      "fovy": 110,\n      "rotation": [\n        [\n          -1.0,\n          0.0,\n          0.0\n        ],\n        [\n          0.0,\n          0.9138115486202572,\n          0.40613846605344756\n        ],\n        [\n          0.0,\n          0.40613846605344756,\n          -0.9138115486202572\n        ]\n      ],\n      "camera_name": "right_wrist",\n      "parent_body": "arm1_link6"\n    }\n  ],\n  "not_used_in_existing_checkpoints": false\n}')
def configure(env):
 configure_main(env)
 for c in PROFILE['cameras']:
  cid=env.model.camera(c['camera_name']).id
  assert env.model.body(env.model.cam_bodyid[cid]).name==c['parent_body']
  env.model.cam_pos[cid]=c['pos']
  mujoco.mju_mat2Quat(env.model.cam_quat[cid],np.array(c['rotation']).ravel())
  env.model.cam_fovy[cid]=c['fovy']
 mujoco.mj_forward(env.model,env.data)
 return env
