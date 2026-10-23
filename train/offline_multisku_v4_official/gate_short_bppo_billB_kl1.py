#!/usr/bin/env python3
"""Closed-loop sim gates for short BPPO candidates (select+confirm, v4 protocol)."""
from __future__ import annotations
import os, sys, json, time, math, hashlib, threading, subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

OUT = Path(os.environ["S2_OUT"])
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ms_common import SKUS, init_pose, write_json, sha256, cst

PYBIN = sys.executable
NPS = 50
SH = 5
SELECT0, CONFIRM0 = 880000, 880050
POS, YAW = 0.005, 2.0
PER_GPU = 8
GPUS = [int(x) for x in os.environ.get("GATE_GPUS", "7").split(",") if x.strip() != ""]
_LK = threading.RLock()
LOG = OUT / "gate" / "short_bppo_billB_kl1_gate.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

def log(msg: str) -> None:
    line = f"[{cst()}] {msg}"
    print(line, flush=True)
    with _LK, open(LOG, "a") as f:
        f.write(line + "\n")

def env_for(g: int) -> dict:
    e = dict(os.environ)
    e.update(CUDA_VISIBLE_DEVICES=str(g), MUJOCO_EGL_DEVICE_ID=str(g), MUJOCO_GL="egl", OMP_NUM_THREADS="4", S2_OUT=str(OUT))
    return e

def gate_episodes(seed0: int) -> list:
    E = []
    for sku in SKUS:
        for s in range(seed0, seed0 + NPS):
            dx, dy, yw = init_pose(s, POS, YAW)
            E.append(dict(seed=s, sku=sku, dx=dx, dy=dy, yaw_deg=yw, cell=sku))
    return E

def shards(eps, kind, mode, prefix):
    return [dict(id=f"{prefix}_{i:04d}", kind=kind, mode=mode, episodes=eps[i : i + SH]) for i in range(0, len(eps), SH)]

def wilson(k, n):
    if n == 0:
        return None
    z = 1.95996398454
    ph = k / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]

def run_jobs(policy: Path, d: Path, jobs: list, phase: str, share: int = 1) -> list:
    d.mkdir(parents=True, exist_ok=True)
    jf = d / "jobs.json"
    spec = dict(policy=str(policy), out=str(d), jobs=jobs)
    if jf.exists():
        old = json.loads(jf.read_text())
        if old != spec:
            raise RuntimeError(f"job spec changed in {d}")
    else:
        write_json(jf, spec)
    for attempt in range(3):
        claims = d / "claims"
        done = d / "done"
        if claims.exists():
            for c in claims.glob("*"):
                if not (done / c.name).exists():
                    c.unlink()
        todo = [j for j in jobs if not (done / j["id"]).exists()]
        if not todo:
            break
        (d / "logs").mkdir(exist_ok=True)
        pg = max(1, PER_GPU // max(1, share))
        slots = [g for g in GPUS for _ in range(pg)][: len(todo)]
        log(f"{phase}: attempt {attempt+1}: {len(todo)} shards on {len(slots)} workers gpus={GPUS} pg={pg}")
        procs = []
        for i, g in enumerate(slots):
            procs.append(
                subprocess.Popen(
                    [PYBIN, "-u", str(HERE / "batch_worker_ms.py"), "--jobs", str(jf), "--wid", str(attempt * 1000 + i)],
                    cwd=str(HERE),
                    env=env_for(g),
                    stdout=open(d / "logs" / f"worker_a{attempt}_{i}.log", "a"),
                    stderr=subprocess.STDOUT,
                )
            )
        while any(q.poll() is None for q in procs):
            nd = len(list((d / "done").glob("*"))) if (d / "done").exists() else 0
            log(f"{phase}: progress {nd}/{len(jobs)}")
            time.sleep(30)
        bad = [q.returncode for q in procs if q.returncode not in (0, None)]
        if bad:
            log(f"{phase}: attempt {attempt+1}: bad rcs={bad}")
    nd = len(list((d / "done").glob("*")))
    if nd != len(jobs):
        raise RuntimeError(f"{phase}: {nd}/{len(jobs)} shards done")
    rows = []
    for j in jobs:
        rows += [json.loads(l) for l in open(d / "rows" / f"{j['id']}.jsonl")]
    return rows

def gate(policy: Path, name: str, which: str, share: int = 1) -> dict:
    seed0 = SELECT0 if which == "select" else CONFIRM0
    d = OUT / "gate" / f"{name}_{which}"
    rp = d / "result.json"
    sha = sha256(policy)
    if rp.exists() and json.loads(rp.read_text()).get("sha256") == sha:
        log(f"GATE {name}:{which} cached {rp}")
        return json.loads(rp.read_text())
    rows = run_jobs(policy, d, shards(gate_episodes(seed0), "eval", "det", which), f"gate:{name}:{which}", share)
    per = {s: [sum(bool(r["success"]) for r in rows if r["sku"] == s), sum(r["sku"] == s for r in rows)] for s in SKUS}
    fails = {}
    for r in rows:
        if not r["success"]:
            f = fails.setdefault(r["sku"], {})
            f[r["reason"]] = f.get(r["reason"], 0) + 1
    k = sum(v[0] for v in per.values())
    n = sum(v[1] for v in per.values())
    res = dict(
        name=name,
        set=which,
        seeds=[seed0, seed0 + NPS - 1],
        policy=str(policy),
        sha256=sha,
        finished=cst(),
        successes=k,
        episodes=n,
        wilson95=wilson(k, n),
        per_sku_logged_only=per,
        fails=fails,
        outcomes={f"{r['sku']}:{r['seed']}": bool(r["success"]) for r in rows},
    )
    write_json(rp, res)
    log(f"GATE {name}:{which} overall {k}/{n} fails {fails}")
    return res

def pooled(rs, rc):
    return dict(
        successes=rs["successes"] + rc["successes"],
        episodes=rs["episodes"] + rc["episodes"],
        select=rs["successes"],
        confirm=rc["successes"],
        select_wilson=rs.get("wilson95"),
        confirm_wilson=rc.get("wilson95"),
    )

def main():
    short = Path(os.environ.get("SHORT_BPPO_DIR", str(OUT / "round2" / "bppo_short_billB_kl1")))
    cands = {
        "kl1_s100": short / "cand_step0100.pt",
        "kl1_s250": short / "cand_step0250.pt",
        "kl1_s500": short / "cand_step0500.pt",
    }
    for n, p in cands.items():
        assert p.exists(), p
    items = [(n, p, w) for n, p in cands.items() for w in ("select", "confirm")]
    log(f"START short BPPO KL1 gates n={len(items)} gpus={GPUS} skus={SKUS}")
    summary = {"started": cst(), "gpus": GPUS, "candidates": {}, "baseline_start_pooled": 485, "baseline_note": "start_select 243 + start_confirm 242"}
    write_json(OUT / "gate" / "short_bppo_billB_kl1_status.json", dict(status="running", updated=cst(), items=[x[0]+":"+x[2] for x in items]))
    with ThreadPoolExecutor(len(items)) as ex:
        futs = {ex.submit(gate, p, n, w, len(items)): (n, w) for n, p, w in items}
        results = {}
        for fut in futs:
            n, w = futs[fut]
            results[(n, w)] = fut.result()
    for n in cands:
        pl = pooled(results[(n, "select")], results[(n, "confirm")])
        summary["candidates"][n] = dict(
            path=str(cands[n]),
            select=pl["select"],
            confirm=pl["confirm"],
            pooled=pl["successes"],
            episodes=pl["episodes"],
            delta_vs_start_pooled=pl["successes"] - 485,
            select_wilson=pl["select_wilson"],
            confirm_wilson=pl["confirm_wilson"],
        )
        log(f"POOLED {n}: {pl['successes']}/500 (select {pl['select']}/250 confirm {pl['confirm']}/250) delta_vs_start={pl['successes']-485}")
    summary["finished"] = cst()
    summary["status"] = "done"
    write_json(OUT / "gate" / "short_bppo_billB_kl1_summary.json", summary)
    write_json(OUT / "gate" / "short_bppo_billB_kl1_status.json", dict(status="done", updated=cst(), summary=summary["candidates"]))
    log(f"DONE summary -> {OUT/'gate'/'short_bppo_billB_kl1_summary.json'}")

if __name__ == "__main__":
    main()
