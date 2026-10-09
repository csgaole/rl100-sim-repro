"""Policy-independent task wrapper preserving the SKU09 experiment contract."""
import mujoco
import numpy as np
from PIL import Image
from ._physics import SortingEnv
from .camera_center_wrist_v2 import configure

SKU_INDEX = {"SKU02": 1, "SKU09": 4}


def task_state(env):
    """Privileged evaluator data; never part of the policy observation."""
    model, data = env.model, env.data
    body = model.body("parcel").id
    position = data.xpos[body]
    rotation = data.xmat[body].reshape(3, 3)
    extent = abs(rotation) @ model.geom_size[model.geom("parcel_box").id]
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    forces = np.zeros(2)
    outfeed_contact = False
    for ci in range(data.ncon):
        contact = data.contact[ci]
        names = [model.geom(int(g)).name or "" for g in contact.geom]
        if "parcel_box" not in names:
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, ci, force)
        if "outfeed" in names and force[0] > .01:
            outfeed_contact = True
        for arm in range(2):
            if any(n.startswith(f"arm{arm}_") and "gripper_link" in n for n in names):
                forces[arm] += max(0, float(force[0]))
    return dict(
        outfeed_contact=outfeed_contact,
        outfeed_inside=bool(position[0]-extent[0] >= -.325 and position[0]+extent[0] <= .325
                            and position[1]-extent[1] >= -1.83 and position[1]+extent[1] <= -.33),
        trailing_y=float(position[1]+extent[1]), vy=float(velocity[4]),
        forces=forces.tolist(), label_up=float(rotation[2, 2]),
        position=position.copy().tolist(), bottom=float(position[2]-extent[2]),
    )


class OutcomeTracker:
    def __init__(self):
        self.stable = 0
        self.start_y = None

    def update(self, state, time):
        if state["bottom"] < .74:
            return "drop"
        qualified = (state["label_up"] > .98 and state["outfeed_contact"]
                     and state["outfeed_inside"] and max(state["forces"]) < .05
                     and -.6 < state["vy"] < -.1)
        if qualified:
            if self.start_y is None:
                self.start_y = state["position"][1]
            self.stable += 1
        else:
            self.stable, self.start_y = 0, None
        if (self.stable >= 48 and self.start_y-state["position"][1] >= .1
                and state["trailing_y"] <= -.65):
            return "success"
        if time >= 10-1e-8:
            return "watchdog"
        return None


class LogisticsSortingEnv:
    """reset -> (obs, info); step -> (obs, reward, terminated, truncated, info).

    This is a small NumPy API, not a registered gymnasium.Env. step accepts one
    to eight 14D absolute joint targets, each executed for one 1/240 s tick.
    Gripper columns 6 and 13 are reserved: the legacy controller fixes width .012m.
    """
    def __init__(self, sku="SKU09", render=True, image_size=(320, 240)):
        if sku not in SKU_INDEX:
            raise ValueError(f"sku must be one of {list(SKU_INDEX)}")
        self.sku = sku
        self.physics = SortingEnv(SKU_INDEX[sku], render=False)
        configure(self.physics)
        e = self.physics
        e.model.actuator_gainprm[e.aidx.ravel(), 0] = 1600
        e.model.actuator_biasprm[e.aidx.ravel(), 1] = -1600
        self.render_enabled = render
        if render:
            e.renderer = mujoco.Renderer(e.model, height=image_size[1], width=image_size[0])
        self.reset()

    def reset(self, seed=0):
        e = self.physics
        e.reset()
        adr = e.model.jnt_qposadr[e.model.joint("parcel_free").id]
        e.data.qpos[adr:adr+2] += np.random.default_rng(seed).uniform(-.005, .005, 2)
        mujoco.mj_forward(e.model, e.data)
        self.tracker, self.reason = OutcomeTracker(), None
        self.history = []
        return self.observe(), {"seed": seed, "sku": self.sku}

    def observe(self):
        """RGB/proprio only. No parcel pose, force, or success signal."""
        obs = self.physics.observe()
        return obs

    def policy_observation(self):
        """Optional BC adapter: 2x3x96x96 uint8 cameras plus 2x28 proprio.

        Call once after reset and after each step; this maintains camera history.
        Normalization and batch/device conversion belong to the policy adapter.
        """
        obs = self.observe()
        if not obs["rgb"]:
            raise RuntimeError("policy_observation requires render=True")
        self.history = (self.history+[obs])[-2:]
        frames = self.history if len(self.history) == 2 else self.history*2
        result = {"agent_pos": np.stack([f["proprio"] for f in frames])}
        for camera, key in [("global", "image_overview"), ("left_wrist", "image_left"), ("right_wrist", "image_right")]:
            result[key] = np.stack([np.asarray(Image.fromarray(f["rgb"][camera]).resize((96, 96)))
                                    for f in frames]).transpose(0, 3, 1, 2)
        return result

    def step(self, actions):
        if self.reason is not None:
            raise RuntimeError("Episode finished; call reset before step")
        actions = np.asarray(actions, dtype=np.float64)
        if actions.shape == (14,):
            actions = actions[None]
        if actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= 8:
            raise ValueError("actions must have shape (14,) or (N,14), 1<=N<=8")
        if not np.isfinite(actions).all():
            raise ValueError("actions must be finite")
        effective = []
        for action in actions:
            self.physics.step(action)
            effective.append(self.physics.history[-1][2].copy())
            state = task_state(self.physics)
            self.reason = self.tracker.update(state, float(self.physics.data.time))
            if self.reason is not None:
                break
        return (self.observe(), float(self.reason == "success"),
                self.reason in ("success", "drop"), self.reason == "watchdog",
                {"reason": self.reason, "executed_ticks": len(effective),
                 "effective_actions": np.stack(effective), "evaluation": state})

    def close(self):
        self.physics.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
