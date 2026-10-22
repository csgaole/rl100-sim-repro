#!/usr/bin/env python3
# [online_multisku_official 2026-10-22] Minimal official-aligned online GAE-PPO for piper multisku.
# Outer loop per round:
#   (1) collect fresh rollouts (IC ±5mm/±2°, s2_stdfix sampler, ALL outcomes incl. failures; NO retry-until-success)
#   (2) official-style compute_gae_per_env ONCE with CURRENT V (advantages frozen for K_epochs)
#   (3) PPO clip update K_epochs (NO KL-anchor loss, NO omega, NO IQL advantage; adv_norm + reward_scaling on)
#   (4) buffer.reset()
#   (5) linear lr decay; save lr.txt each round (restore on --continue-run)
#   (6) demo-mix closed-loop gate vs start-ckpt measured score (488) + log vs BC 485; rollback if worse than start
#   (7) log V loss, explained variance, approx_kl, clip_frac, lr_actor/lr_critic, rollout success
# V initialized fresh unless --load-value given (preserve critic across relaunch). Reuses offline env/rollout/collect (batch_worker_ms) + ms_common sampler.
"""Minimal online GAE-PPO launcher (official-aligned)."""
from __future__ import annotations
import argparse, json, math, os, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OFFLINE = HERE.parent / "offline_multisku_v4_official"
ROOT = HERE.parents[1]
sys.path.insert(0, str(OFFLINE))

os.environ.setdefault("S2_FIX_FINAL", "1")
os.environ.setdefault("S2_FLOOR_FINAL", "0")
os.environ.setdefault("S2_STD_MAX", "0.1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TMPDIR", "/data02/kemove/tmp")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ms_common import (
    SKUS, CHUNKS, FEAT, SD, GAMMA,
    init_pose, load_policy, setup, write_json, append_jsonl, sha256, cst,
    kernel, is_terminal, logprob, mlp, check_seeds,
)

PY = "/data02/kemove/sku09-bc-migration/env/bin/python"
WORKER = OFFLINE / "batch_worker_ms.py"
_LK = threading.RLock()

DEFAULT_LR_A = 3.0e-6
DEFAULT_LR_C = 3.0e-4
DEFAULT_EPS_CLIP = 0.2
DEFAULT_K_EPOCHS = 10
DEFAULT_GAMMA = 0.99
DEFAULT_LAMDA = 0.95
DEFAULT_MAX_STEPS = 1_000_000
DEFAULT_MINI = 128
BC_POOLED_485 = 485

ap = argparse.ArgumentParser()
ap.add_argument("--start", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rounds", type=int, default=20)
ap.add_argument("--eps", type=int, default=60)
ap.add_argument("--collect-seed0", type=int, default=2400000)
ap.add_argument("--ic-pos-mm", type=float, default=5.0)
ap.add_argument("--ic-yaw-deg", type=float, default=2.0)
ap.add_argument("--lr-a", type=float, default=DEFAULT_LR_A)
ap.add_argument("--lr-c", type=float, default=DEFAULT_LR_C)
ap.add_argument("--clip", type=float, default=DEFAULT_EPS_CLIP)
ap.add_argument("--k-epochs", type=int, default=DEFAULT_K_EPOCHS)
ap.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
ap.add_argument("--lamda", type=float, default=DEFAULT_LAMDA)
ap.add_argument("--max-train-steps", type=int, default=DEFAULT_MAX_STEPS)
ap.add_argument("--mini-batch", type=int, default=DEFAULT_MINI)
ap.add_argument("--lp-reduce", choices=["sum", "mean"], default="sum")
ap.add_argument("--grad-clip", type=float, default=0.5)
ap.add_argument("--gpus", default="7,6,5,4,3")
ap.add_argument("--exclude-gpus", default="0,1,2")
ap.add_argument("--per-gpu", type=int, default=4)
ap.add_argument("--shard-size", type=int, default=5)
ap.add_argument("--min-free-mib", type=int, default=16000)
ap.add_argument("--select-seed0", type=int, default=880000)
ap.add_argument("--confirm-seed0", type=int, default=880050)
ap.add_argument("--gate-per-sku", type=int, default=50)
ap.add_argument("--gate-baseline", type=int, default=488)
ap.add_argument("--bc-baseline", type=int, default=BC_POOLED_485)
ap.add_argument("--no-gate", action="store_true")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--continue-run", action="store_true")
ap.add_argument("--load-value", default="",
                help="warm-start critic from value.pt (preserves trained V across relaunch; actor still from --start)")
ap.add_argument("--load-reward-scaling", default="",
                help="optional reward_scaling.json to pair with --load-value")
ap.add_argument("--fix-encoder", action="store_true", default=True,
                help="freeze encoder (default True for feature-buffer minimal path; feats from collect)")
a = ap.parse_args()

OUT = Path(a.out)
os.environ["S2_OUT"] = str(OUT)
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "online.log"
SM = a.smoke
if SM:
    a.eps = min(a.eps, 5)
    a.k_epochs = min(a.k_epochs, 2)
    a.rounds = 1
    a.no_gate = True
    a.shard_size = 5
    a.per_gpu = 2

POS = a.ic_pos_mm / 1000.0
YAW = a.ic_yaw_deg
SH = a.shard_size
NPS = a.gate_per_sku
PREFER = [int(x) for x in a.gpus.split(",") if x.strip() != ""]
EXCL = {int(x) for x in a.exclude_gpus.split(",") if x.strip() != ""}

def log(msg: str) -> None:
    line = f"[{cst()}] {msg}"
    print(line, flush=True)
    with _LK, open(LOG, "a") as f:
        f.write(line + "\n")


def free_gpus(min_free=None):
    min_free = a.min_free_mib if min_free is None else min_free
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout
    busy = set()
    for l in apps.strip().splitlines():
        parts = [x.strip() for x in l.split(",")]
        if len(parts) >= 2:
            busy.add(parts[1])
    parent_bound = set()
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd and "," not in cvd and cvd.lstrip("-").isdigit():
        parent_bound.add(int(cvd))
    by_idx = {}
    for l in q.strip().splitlines():
        i, uuid, fr = [x.strip() for x in l.split(",")]
        i = int(i)
        if i in EXCL or uuid in busy:
            continue
        if int(float(fr)) < min_free:
            continue
        by_idx[i] = int(float(fr))
    ordered = [g for g in PREFER if g in by_idx] + [g for g in sorted(by_idx) if g not in PREFER]
    non_parent = [g for g in ordered if g not in parent_bound]
    return non_parent if non_parent else ordered


def env_for(g: int) -> dict:
    e = dict(os.environ)
    e.update(
        CUDA_VISIBLE_DEVICES=str(g), MUJOCO_EGL_DEVICE_ID=str(g), MUJOCO_GL="egl",
        OMP_NUM_THREADS="4", S2_OUT=str(OUT),
        S2_FIX_FINAL=os.environ.get("S2_FIX_FINAL", "1"),
        S2_FLOOR_FINAL=os.environ.get("S2_FLOOR_FINAL", "0"),
        S2_STD_MAX=os.environ.get("S2_STD_MAX", "0.1"), MS_SMOKE="1" if SM else "0", TMPDIR="/data02/kemove/tmp",
    )
    return e


def shards(eps, kind, mode, prefix):
    return [dict(id=f"{prefix}_{i:04d}", kind=kind, mode=mode, episodes=eps[i:i + SH]) for i in range(0, len(eps), SH)]


def run_jobs(policy, d: Path, jobs, phase: str):
    d.mkdir(parents=True, exist_ok=True)
    jf = d / "jobs.json"
    spec = dict(policy=str(policy), out=str(d), jobs=jobs)
    if jf.exists():
        if json.loads(jf.read_text()) != spec:
            raise RuntimeError(f"job spec changed in {d}")
    else:
        write_json(jf, spec)
    for attempt in range(3):
        claims, done = d / "claims", d / "done"
        if claims.exists():
            for c in claims.glob("*"):
                if not (done / c.name).exists():
                    c.unlink()
        todo = [j for j in jobs if not (done / j["id"]).exists()]
        if not todo:
            break
        ok = free_gpus()
        while not ok:
            log(f"{phase}: waiting for free GPU (prefer {PREFER}, excl {sorted(EXCL)})")
            time.sleep(60)
            ok = free_gpus()
        (d / "logs").mkdir(exist_ok=True)
        slots = [g for g in ok for _ in range(a.per_gpu)][:len(todo)]
        log(f"{phase}: attempt {attempt+1}: {len(todo)} shards on {len(slots)} workers gpus={ok}")
        procs = []
        for i, g in enumerate(slots):
            lf = open(d / "logs" / f"w_a{attempt}_{i}.log", "a")
            procs.append(subprocess.Popen(
                [PY, "-u", str(WORKER), "--jobs", str(jf), "--wid", str(attempt * 1000 + i)],
                cwd=str(OFFLINE), env=env_for(g), stdout=lf, stderr=subprocess.STDOUT,
            ))
        for pr in procs:
            pr.wait()
        bad = [pr.returncode for pr in procs if pr.returncode != 0]
        if bad:
            log(f"{phase}: attempt {attempt+1}: {len(bad)} workers rc!=0")
    nd = len(list((d / "done").glob("*"))) if (d / "done").exists() else 0
    assert nd == len(jobs), f"{phase}: {nd}/{len(jobs)} shards done"
    rows = []
    for j in jobs:
        rows += [json.loads(l) for l in open(d / "rows" / f"{j['id']}.jsonl")]
    return rows


def collect_episodes(round_i: int, policy: Path, n: int):
    base = a.collect_seed0 + round_i * 10000
    check_seeds(base, base + n - 1)
    E = []
    for i in range(n):
        s = base + i
        dx, dy, yw = init_pose(s, POS, YAW)
        E.append(dict(
            seed=s, sku=SKUS[i % len(SKUS)], dx=dx, dy=dy, yaw_deg=yw,
            cell=SKUS[i % len(SKUS)], heldout=False, collect_round=round_i,
            pos_half_mm=a.ic_pos_mm, yaw_half_deg=a.ic_yaw_deg,
        ))
    d = OUT / f"round{round_i}" / "collect"
    rows = run_jobs(policy, d, shards(E, "collect", "sto", "sto"), f"r{round_i}:collect")
    n_ok = sum(bool(r["success"]) for r in rows)
    assert len(rows) == n, f"collect returned {len(rows)} != {n}"
    log(f"COLLECT r{round_i}: {n_ok}/{n} success (failures kept={n - n_ok}) IC=±{a.ic_pos_mm}mm/±{a.ic_yaw_deg}°")
    return rows, d / "eps"


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = 1e-4

    def update(self, x):
        x = np.asarray(x, np.float64)
        batch_mean = np.atleast_1d(x).mean(axis=0)
        batch_var = np.atleast_1d(x).var(axis=0)
        batch_count = np.atleast_1d(x).shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot
        self.mean, self.var, self.count = new_mean, M2 / tot, tot

    @property
    def std(self):
        return np.sqrt(self.var)


class RewardScaling:
    def __init__(self, gamma):
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=())
        self.R = 0.0

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(np.array([self.R]))
        return float(x / (self.running_ms.std + 1e-8))

    def reset_episode(self):
        self.R = 0.0

    def state_dict(self):
        return dict(mean=float(self.running_ms.mean), var=float(self.running_ms.var), count=float(self.running_ms.count))

    def load_state_dict(self, d):
        self.running_ms.mean = np.array(d["mean"], np.float64)
        self.running_ms.var = np.array(d["var"], np.float64)
        self.running_ms.count = float(d["count"])


def compute_gae_per_env(rewards, dones, dws, vs, vs_, gamma, lamda, n_action_steps=1):
    """Official uni_ppo.compute_gae_per_env semantics."""
    steps, env_num = rewards.shape[0], rewards.shape[1]
    gamma_eff = gamma ** n_action_steps
    deltas = rewards + gamma_eff * (1.0 - dws) * vs_ - vs
    adv = torch.zeros_like(rewards)
    gae = torch.zeros(env_num, 1, device=rewards.device)
    for t in reversed(range(steps)):
        gae = deltas[t] + gamma_eff * lamda * gae * (1.0 - dones[t])
        adv[t] = gae
    v_target = adv + vs
    return adv.reshape(-1, 1), v_target.reshape(-1, 1)

@torch.no_grad()
def rebuild_trace_one(p, model, feats, seed, lp_reduce="sum"):
    """Replay batch_worker sto diffusion for one episode (seed+20M generator)."""
    device = feats.device
    gen = torch.Generator(device=device).manual_seed(int(seed) + 20_000_000)
    p.noise_scheduler.set_timesteps(10)
    traces = []
    for t_chunk in range(CHUNKS):
        z_prev = feats[max(t_chunk - 1, 0)]
        z_cur = feats[t_chunk]
        cond = torch.cat([z_prev, z_cur], 0).unsqueeze(0)
        x = torch.randn((1, 8, 14), device=device, generator=gen)
        xs, ys, lps = [], [], []
        for t in p.noise_scheduler.timesteps:
            eps = model(x, t.to(device), global_cond=cond)
            mean, std = kernel(p, eps, t, x)
            noise = torch.randn(x.shape, device=device, generator=gen)
            y = mean + std * noise
            xs.append(x); ys.append(y)
            lps.append(logprob(mean, std, y, reduce=lp_reduce))
            x = y
        traces.append(dict(xs=xs, ys=ys, lps=lps, action=x, cond=cond))
    return traces


def load_round_buffer(eps_dir: Path, p, model, device, reward_scaling: RewardScaling, lp_reduce="sum"):
    files = sorted(eps_dir.glob("ep_*.npz"))
    assert files, f"no episodes in {eps_dir}"
    all_succ, all_seeds, all_sku, all_sk, all_F = [], [], [], [], []
    all_xs, all_ys, all_lps = [], [], []
    n_fail = 0
    for f in files:
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        feats = torch.tensor(z["feats"], device=device, dtype=torch.float32)
        acts = torch.tensor(z["actions"], device=device, dtype=torch.float32)
        succ = bool(meta["success"])
        if not succ:
            n_fail += 1
        traces = rebuild_trace_one(p, model, feats, meta["seed"], lp_reduce=lp_reduce)
        recon = torch.stack([tr["action"].squeeze(0) for tr in traces], 0)
        max_err = (recon - acts).abs().max().item()
        assert max_err < 1e-3, f"trace rebuild mismatch seed={meta['seed']} max_err={max_err}"
        xs = torch.stack([torch.stack(tr["xs"], 0).squeeze(1) for tr in traces], 0)
        ys = torch.stack([torch.stack(tr["ys"], 0).squeeze(1) for tr in traces], 0)
        lps = torch.stack([torch.stack([lp.view(-1)[0] for lp in tr["lps"]], 0) for tr in traces], 0)
        all_xs.append(xs); all_ys.append(ys); all_lps.append(lps)
        all_F.append(feats); all_succ.append(succ); all_seeds.append(meta["seed"])
        all_sku.append(meta["sku"]); all_sk.append(np.asarray(meta["sku_cond"], np.float32))

    N = len(files)
    rewards = torch.zeros(CHUNKS, N, 1, device=device)
    dones = torch.zeros(CHUNKS, N, 1, device=device)
    dws = torch.zeros(CHUNKS, N, 1, device=device)
    for i, succ in enumerate(all_succ):
        reward_scaling.reset_episode()
        for t in range(CHUNKS):
            raw = 1.0 if (t == CHUNKS - 1 and succ) else 0.0
            rewards[t, i, 0] = reward_scaling(raw)
        dones[-1, i, 0] = 1.0
        dws[-1, i, 0] = 1.0

    F = torch.stack(all_F, 0)
    SK = torch.tensor(np.stack(all_sk), device=device, dtype=torch.float32)
    xs = torch.stack(all_xs, 0).permute(1, 0, 2, 3, 4).contiguous()
    ys = torch.stack(all_ys, 0).permute(1, 0, 2, 3, 4).contiguous()
    lps = torch.stack(all_lps, 0).permute(1, 0, 2).contiguous()
    buf = dict(
        F=F, SK=SK, succ=all_succ, seeds=all_seeds, sku=all_sku,
        xs=xs, ys=ys, lps=lps, rewards=rewards, dones=dones, dws=dws,
        n_env=N, n_fail=n_fail, n_success=sum(all_succ),
    )
    log(f"BUFFER built: N={N} success={buf['n_success']} fail={n_fail} (failures in buffer=yes)")
    return buf


class OnlineBuffer:
    def __init__(self):
        self.data = None
        self.alive = False

    def set(self, data):
        self.data = data
        self.alive = True

    def reset(self):
        self.data = None
        self.alive = False


def feat_stats_from_F(F):
    f = F.reshape(-1, FEAT)
    return dict(mu=f.mean(0), sd=f.std(0).clamp_min(1e-3))


def critic_state_from(F, SK, stats, e_idx, t):
    prev = F[e_idx, (t - 1).clamp(min=0)]
    cur = F[e_idx, t]
    clock = (t.float() / CHUNKS * 2 - 1)[:, None]
    return torch.cat([(prev - stats["mu"]) / stats["sd"], (cur - stats["mu"]) / stats["sd"], clock, SK[e_idx]], 1)


def policy_cond_from(F, e_idx, t):
    return torch.cat([F[e_idx, (t - 1).clamp(min=0)], F[e_idx, t]], 1)


def explained_var(y_pred, y_true):
    y_pred = y_pred.detach().flatten()
    y_true = y_true.detach().flatten()
    var_y = torch.var(y_true)
    if float(var_y) < 1e-8:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / (var_y + 1e-8))


def save_lr_txt(path: Path, lr_a, lr_c, total_steps, round_i):
    path.write_text(json.dumps(dict(lr_a=lr_a, lr_c=lr_c, total_steps=total_steps, round=round_i, updated=cst()), indent=2))


def load_lr_txt(path: Path):
    return json.loads(path.read_text())


def gate_episodes(seed0, n, pos, yaw):
    E = []
    for sku in SKUS:
        for s in range(seed0, seed0 + n):
            dx, dy, yw = init_pose(s, pos, yaw)
            E.append(dict(seed=s, sku=sku, dx=dx, dy=dy, yaw_deg=yw, cell=sku))
    return E


def wilson(k, n):
    if n == 0:
        return None
    z = 1.95996398454
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def run_gate(policy: Path, name: str, which: str):
    seed0 = a.select_seed0 if which == "select" else a.confirm_seed0
    d = OUT / "gate" / f"{name}_{which}"
    rp = d / "result.json"
    sha = sha256(policy)
    if rp.exists() and json.loads(rp.read_text()).get("sha256") == sha:
        return json.loads(rp.read_text())
    rows = run_jobs(policy, d, shards(gate_episodes(seed0, NPS, POS, YAW), "eval", "det", which), f"gate:{name}:{which}")
    k = sum(bool(r["success"]) for r in rows)
    n = len(rows)
    res = dict(name=name, set=which, seeds=[seed0, seed0 + NPS - 1], policy=str(policy),
               sha256=sha, finished=cst(), successes=k, episodes=n, wilson95=wilson(k, n),
               outcomes={f"{r['sku']}:{r['seed']}": bool(r["success"]) for r in rows})
    write_json(rp, res)
    log(f"GATE {name}:{which} {k}/{n}")
    return res


def gate_pooled(policy: Path, name: str):
    rs = run_gate(policy, name, "select")
    rc = run_gate(policy, name, "confirm")
    pooled = rs["successes"] + rc["successes"]
    return dict(
        successes=pooled, episodes=rs["episodes"] + rc["episodes"],
        select=rs["successes"], confirm=rc["successes"],
        select_wilson=rs.get("wilson95"), confirm_wilson=rc.get("wilson95"),
        delta_vs_start=pooled - a.gate_baseline, delta_vs_bc485=pooled - a.bc_baseline,
    )

def ppo_update(p, model, V, opt_a, opt_c, buf, stats, adv, v_target, vs_flat, args_clip, k_epochs, mini, device):
    """K_epochs PPO clip only. Advantages frozen. NO KL-anchor, NO omega."""
    N = buf["n_env"]
    B = N * CHUNKS
    SI = [(i, ts) for i, ts in enumerate(p.noise_scheduler.timesteps) if not is_terminal(p, ts)]
    actor_losses, critic_losses, approx_kls, clip_fracs = [], [], [], []
    F, SK = buf["F"], buf["SK"]

    for epoch in range(k_epochs):
        for index in BatchSampler(SubsetRandomSampler(range(B)), mini, False):
            idx = torch.tensor(index, device=device, dtype=torch.long)
            t_idx = idx // N
            e_idx = idx % N
            cond = policy_cond_from(F, e_idx, t_idx)
            adv_b = adv[idx]
            vt_b = v_target[idx]
            vs_b = vs_flat[idx]
            mb = len(index)

            loss_a = 0.0
            for i_step, ts in SI:
                x_in = buf["xs"][t_idx, e_idx, i_step]
                y_out = buf["ys"][t_idx, e_idx, i_step]
                lp_old = buf["lps"][t_idx, e_idx, i_step]
                eps = model(x_in, ts.to(device).expand(mb), global_cond=cond)
                mean, std = kernel(p, eps, ts, x_in)
                lp_now = logprob(mean, std, y_out, reduce=a.lp_reduce)
                logr = (lp_now - lp_old).clamp(-20, 20)
                ratio = logr.exp()
                surr1 = ratio * adv_b.squeeze(-1)
                surr2 = ratio.clamp(1 - args_clip, 1 + args_clip) * adv_b.squeeze(-1)
                loss_a = loss_a + (-torch.min(surr1, surr2).mean())
                with torch.no_grad():
                    approx_kls.append(float(((ratio - 1) - logr).mean().item()))
                    clip_fracs.append(float(((ratio - 1).abs() > args_clip).float().mean().item()))
            loss_a = loss_a / max(1, len(SI))
            opt_a.zero_grad(set_to_none=True)
            loss_a.backward()
            nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
            opt_a.step()
            actor_losses.append(float(loss_a.item()))

            s = critic_state_from(F, SK, stats, e_idx, t_idx)
            v_s = V(s)
            old_v_clipped = vs_b + (v_s - vs_b).clamp(-args_clip, args_clip)
            value_loss = (v_s - vt_b.detach()).pow(2)
            value_loss_clipped = (old_v_clipped - vt_b.detach()).pow(2)
            critic_loss = torch.max(value_loss, value_loss_clipped).mean()
            opt_c.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(V.parameters(), a.grad_clip)
            opt_c.step()
            critic_losses.append(float(critic_loss.item()))

    return dict(
        actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0,
        critic_loss=float(np.mean(critic_losses)) if critic_losses else 0.0,
        approx_kl=float(np.mean(approx_kls)) if approx_kls else 0.0,
        clip_frac=float(np.mean(clip_fracs)) if clip_fracs else 0.0,
        k_epochs=k_epochs,
        n_minibatches=len(actor_losses),
    )


def lr_decay(opt_a, opt_c, lr_a0, lr_c0, total_steps, max_steps):
    lr_a_now = lr_a0 * (1 - total_steps / max_steps)
    lr_c_now = lr_c0 * (1 - total_steps / max_steps)
    for pg in opt_a.param_groups:
        pg["lr"] = lr_a_now
    for pg in opt_c.param_groups:
        pg["lr"] = lr_c_now
    return lr_a_now, lr_c_now


def save_policy_ckpt(p, cp, path, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(
        ema={k: v.detach().cpu() for k, v in p.state_dict().items()},
        config=dict(cp["config"], stage2=meta),
        epoch=cp.get("epoch"), step=cp.get("step"),
        validation=cp.get("validation", {}), stage2=meta,
    )
    tmp = path.with_name(path.name + ".tmp")
    torch.save(d, tmp)
    tmp.replace(path)


def main():
    setup(20261022)
    assert Path(a.start).exists(), a.start
    start_sha = sha256(a.start)
    # Bind main process to first free preferred GPU (never liu/yu/foreign).
    gpus = free_gpus()
    assert gpus, f"no free GPU among prefer={PREFER} excl={sorted(EXCL)}"
    main_gpu = gpus[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(main_gpu)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(main_gpu)
    device = torch.device("cuda")
    log(f"MAIN_GPU physical={main_gpu} (CUDA_VISIBLE_DEVICES={main_gpu})")

    protocol = dict(
        recipe="online GAE-PPO official-aligned minimal",
        start_policy=str(a.start), start_sha256=start_sha,
        gate_baseline_start_pooled=a.gate_baseline, bc_baseline_pooled=a.bc_baseline,
        ic_pos_mm=a.ic_pos_mm, ic_yaw_deg=a.ic_yaw_deg,
        sampler=("s2_stdfix DDIM10 eta=1 std[0.0067,0.1] "
                 + ("terminal_sigma=0" if os.environ.get("S2_FIX_FINAL","1")=="1"
                    else "terminal_sigma=official_nonzero (prev clamped>=0)")),
        S2_FIX_FINAL=int(os.environ.get("S2_FIX_FINAL", "1")),
        S2_FLOOR_FINAL=int(os.environ.get("S2_FLOOR_FINAL", "0")),
        terminal_sigma_mode=("bill_zero" if os.environ.get("S2_FIX_FINAL","1")=="1" else "official_nonzero"),
        no_kl_anchor=True, no_omega=True, no_iql_advantage=True,
        no_retry_until_success=True, failures_enter_buffer=True,
        is_load_value=bool(a.load_value), load_value_path=a.load_value or None,
        load_reward_scaling_path=a.load_reward_scaling or None,
        use_adv_norm=True, use_reward_scaling=True, use_lr_decay=True,
        fix_encoder=bool(a.fix_encoder), K_epochs=a.k_epochs,
        lr_a=a.lr_a, lr_c=a.lr_c, clip=a.clip, gamma=a.gamma, lamda=a.lamda,
        max_train_steps=a.max_train_steps, eps_per_round=a.eps, smoke=SM,
        prefer_gpus=PREFER, exclude_gpus=sorted(EXCL),
        note_fix_encoder="feats frozen from collect; encoder not updated (minimal path). Official yaml has fix_encoder=False with raw-obs buffer.",
    )
    write_json(OUT / "protocol.json", protocol)
    log(f"START online_gae_ppo_ms out={OUT} start={a.start} sha={start_sha[:16]}… "
        f"baseline_gate={a.gate_baseline} bc485={a.bc_baseline} smoke={SM}")

    p, cp = load_policy(a.start)
    p = p.to(device); p.eval()
    if a.fix_encoder:
        p.obs_encoder.requires_grad_(False)
        p.normalizer.requires_grad_(False)
    model = p.model
    V = mlp(SD, 1, 512, 3, layernorm=False).to(device)
    if a.load_value:
        vp = Path(a.load_value)
        assert vp.exists(), f"--load-value missing: {vp}"
        V.load_state_dict(torch.load(vp, map_location=device, weights_only=False))
        log(f"LOADED critic from {vp}")
    opt_a = torch.optim.Adam(model.parameters(), lr=a.lr_a, eps=1e-5)
    opt_c = torch.optim.Adam(V.parameters(), lr=a.lr_c, eps=1e-5)
    lr_a, lr_c = a.lr_a, a.lr_c
    total_steps = 0
    reward_scaling = RewardScaling(a.gamma)
    if a.load_reward_scaling:
        rsp = Path(a.load_reward_scaling)
        assert rsp.exists(), f"--load-reward-scaling missing: {rsp}"
        reward_scaling.load_state_dict(json.loads(rsp.read_text()))
        log(f"LOADED reward_scaling from {rsp}")
    best_policy = OUT / "best_policy.pt"
    best_score = a.gate_baseline
    current_policy_path = OUT / "current_policy.pt"
    save_policy_ckpt(p, cp, current_policy_path, dict(stage="online_init", source=str(a.start)))
    save_policy_ckpt(p, cp, best_policy, dict(stage="online_best_init", source=str(a.start), gate=a.gate_baseline))

    start_round = 0
    if a.continue_run and (OUT / "lr.txt").exists():
        lr_state = load_lr_txt(OUT / "lr.txt")
        lr_a, lr_c = lr_state["lr_a"], lr_state["lr_c"]
        total_steps = lr_state["total_steps"]
        start_round = int(lr_state.get("round", 0)) + 1
        for pg in opt_a.param_groups: pg["lr"] = lr_a
        for pg in opt_c.param_groups: pg["lr"] = lr_c
        if (OUT / "value.pt").exists():
            V.load_state_dict(torch.load(OUT / "value.pt", map_location=device, weights_only=False))
        if (OUT / "reward_scaling.json").exists():
            reward_scaling.load_state_dict(json.loads((OUT / "reward_scaling.json").read_text()))
        if current_policy_path.exists():
            p2, cp2 = load_policy(current_policy_path)
            p.load_state_dict(p2.state_dict()); cp = cp2; model = p.model
            opt_a = torch.optim.Adam(model.parameters(), lr=lr_a, eps=1e-5)
            log(f"CONTINUE from round {start_round} lr_a={lr_a} lr_c={lr_c} total_steps={total_steps}")

    smoke_checks = {}
    buffer = OnlineBuffer()

    for round_i in range(start_round, a.rounds):
        t_round = time.time()
        rd = OUT / f"round{round_i}"
        rd.mkdir(parents=True, exist_ok=True)
        log(f"==== ROUND {round_i} begin ====")

        rows, eps_dir = collect_episodes(round_i, current_policy_path, a.eps)
        rollout_succ = sum(bool(r["success"]) for r in rows) / max(1, len(rows))

        model.eval()
        buf = load_round_buffer(eps_dir, p, model, device, reward_scaling, lp_reduce=a.lp_reduce)
        buffer.set(buf)
        stats = feat_stats_from_F(buf["F"])

        # (2) GAE once — advantages FROZEN for K_epochs
        V.eval()
        with torch.no_grad():
            vs_list, vs__list = [], []
            for t in range(CHUNKS):
                e_idx = torch.arange(buf["n_env"], device=device)
                t_idx = torch.full((buf["n_env"],), t, device=device, dtype=torch.long)
                s = critic_state_from(buf["F"], buf["SK"], stats, e_idx, t_idx)
                t2 = (t_idx + 1).clamp(max=CHUNKS - 1)
                s2 = critic_state_from(buf["F"], buf["SK"], stats, e_idx, t2)
                vs_list.append(V(s).view(-1, 1))
                vs__list.append(V(s2).view(-1, 1))
            vs = torch.stack(vs_list, 0)
            vs_ = torch.stack(vs__list, 0)
            adv, v_target = compute_gae_per_env(
                buf["rewards"], buf["dones"], buf["dws"], vs, vs_, a.gamma, a.lamda, n_action_steps=1
            )
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            vs_flat = vs.reshape(-1, 1)
            ev = explained_var(vs_flat, v_target)
            adv_frozen_id = id(adv)
            adv_checksum = float(adv.sum().item())
        smoke_checks["gae_once_adv_frozen"] = True
        log(f"GAE done once: adv_mean={float(adv.mean()):.4f} adv_std={float(adv.std()):.4f} EV={ev:.4f}")

        # (3) PPO K_epochs
        model.train(); V.train()
        assert protocol["no_kl_anchor"] and protocol["no_omega"]
        metrics = ppo_update(
            p, model, V, opt_a, opt_c, buf, stats, adv, v_target, vs_flat,
            a.clip, a.k_epochs, a.mini_batch, device,
        )
        assert id(adv) == adv_frozen_id and abs(float(adv.sum().item()) - adv_checksum) < 1e-6
        smoke_checks["adv_unchanged_across_K_epochs"] = True
        smoke_checks["no_kl_anchor"] = True
        smoke_checks["no_omega"] = True
        smoke_checks["ic_pm5_pm2"] = abs(a.ic_pos_mm - 5.0) < 1e-9 and abs(a.ic_yaw_deg - 2.0) < 1e-9
        smoke_checks["failures_enter_buffer_design"] = True
        smoke_checks["failures_in_buffer_count"] = int(buf["n_fail"])

        # (4) buffer.reset
        buffer.reset()
        assert buffer.data is None and not buffer.alive
        smoke_checks["buffer_reset_after_update"] = True

        # (5) lr decay + lr.txt
        total_steps += buf["n_env"] * CHUNKS
        lr_a, lr_c = lr_decay(opt_a, opt_c, a.lr_a, a.lr_c, total_steps, a.max_train_steps)
        save_lr_txt(OUT / "lr.txt", lr_a, lr_c, total_steps, round_i)
        smoke_checks["lr_txt_written"] = (OUT / "lr.txt").exists()
        # would-restore check: parse back
        restored = load_lr_txt(OUT / "lr.txt")
        smoke_checks["lr_txt_would_restore"] = (
            abs(restored["lr_a"] - lr_a) < 1e-15 and restored["total_steps"] == total_steps
        )
        write_json(OUT / "reward_scaling.json", reward_scaling.state_dict())
        torch.save(V.state_dict(), OUT / "value.pt")

        save_policy_ckpt(p, cp, current_policy_path,
            dict(stage="online_gae_ppo", round=round_i, lr_a=lr_a, lr_c=lr_c, total_steps=total_steps, **metrics))
        cand_path = rd / f"policy_round{round_i:03d}.pt"
        save_policy_ckpt(p, cp, cand_path, dict(stage="online_round", round=round_i, **metrics))

        round_time = time.time() - t_round
        row = dict(
            round=round_i, rollout_success=rollout_succ, rollout_n=buf["n_env"],
            rollout_n_success=buf["n_success"], rollout_n_fail=buf["n_fail"],
            approx_kl=metrics["approx_kl"], clip_frac=metrics["clip_frac"],
            v_loss=metrics["critic_loss"], actor_loss=metrics["actor_loss"],
            explained_variance=ev, lr_actor=lr_a, lr_critic=lr_c,
            total_steps=total_steps, round_time_s=round(round_time, 1), finished=cst(),
        )
        append_jsonl(OUT / "metrics.jsonl", row)
        write_json(rd / "metrics.json", row)
        log(f"UPDATE r{round_i}: succ={rollout_succ:.3f} approx_kl={metrics['approx_kl']:.6f} "
            f"clip_frac={metrics['clip_frac']:.4f} Vloss={metrics['critic_loss']:.4f} EV={ev:.4f} "
            f"lr_a={lr_a:.3e} lr_c={lr_c:.3e} time={round_time:.0f}s")

        if not a.no_gate:
            t_gate = time.time()
            g = gate_pooled(cand_path, f"r{round_i}")
            g["gate_time_s"] = round(time.time() - t_gate, 1)
            write_json(rd / "gate.json", g)
            log(f"GATE r{round_i}: pooled={g['successes']}/{g['episodes']} "
                f"delta_vs_start({a.gate_baseline})={g['delta_vs_start']} delta_vs_bc485={g['delta_vs_bc485']}")
            if g["successes"] < best_score:
                log(f"ROLLBACK r{round_i}: {g['successes']} < best {best_score}; restoring best_policy")
                p_best, cp_best = load_policy(best_policy)
                p.load_state_dict(p_best.state_dict()); cp = cp_best; model = p.model
                opt_a = torch.optim.Adam(model.parameters(), lr=lr_a, eps=1e-5)
                save_policy_ckpt(p, cp, current_policy_path,
                    dict(stage="rollback", from_round=round_i, best_score=best_score))
                row["rollback"] = True; row["gate"] = g
            else:
                best_score = g["successes"]
                save_policy_ckpt(p, cp, best_policy, dict(stage="online_best", round=round_i, gate=g))
                row["rollback"] = False; row["gate"] = g
            append_jsonl(OUT / "metrics.jsonl", dict(event="gate", **{k: v for k, v in row.items() if k != "gate"}, gate=g))
        else:
            log("GATE skipped (--no-gate or --smoke)")

        write_json(OUT / "status.json", dict(
            status="running", round=round_i, updated=cst(), last=row,
            smoke_checks=smoke_checks if SM else None,
        ))
        if SM:
            write_json(OUT / "smoke_checks.json", smoke_checks)
            log(f"SMOKE_CHECKS {json.dumps(smoke_checks)}")
            break

    write_json(OUT / "status.json", dict(
        status="done" if not SM else "smoke_done", updated=cst(),
        smoke_checks=smoke_checks if SM else None,
    ))
    log("DONE")


if __name__ == "__main__":
    main()
