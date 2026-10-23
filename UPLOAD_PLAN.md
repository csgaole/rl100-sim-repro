# UPLOAD_PLAN — public GitHub `rl100-sim-repro` (follow-trainable)

Prepared: 2026-10-23 (Asia/Shanghai). Prep only — no push, no `gh` create, no auth remote.

Bill confirmed (2026-10-23): include sim env&task, demo data, weights, checkpoints via Git LFS.
Do **not** upload all of `runs/` (282G / ~407 `.pt` files).

## Include (proposed)

| Path | Role | Approx size |
|---|---|---|
| `env/` | Sim env + SKU09 task | ~11M |
| `train/` | BC / offline / online training entrypoints | ~5.0M |
| `third_party/` | Vendored RL-100 bits | ~5.9M |
| `eval/` | Eval scripts | ~48K |
| `analysis/**/*.py` | Analysis scripts only (artifacts gitignored) | ~tens of KB |
| `README.md`, `data/README.md`, `UPLOAD_PLAN.md` | Docs | ~few KB |
| `.gitignore`, `.gitattributes` | Ignore rules + Git LFS patterns | tiny |
| `data/demos/sku09-original300/` | 300× H5 episodes + `manifest.jsonl` + `splits.json` + `SCHEMA.md` | ~1.6G |
| `models/pretrained/r3m_resnet18.pt` | R3M ResNet18 weights | ~315M |
| `checkpoints/online_best_495.pt` | Curated online best (hardlink ← `runs/online_billB_official_20261022_lra1e6_sigma0_v2/best_policy.pt`) | ~555M |
| `checkpoints/offline_kl1_s250_488.pt` | Curated offline KL1 step250 (hardlink ← `runs/offline_multisku_v4_20261020_2153/round2/bppo_short_billB_kl1/cand_step0250.pt`) | ~555M |

### Approximate upload total

- Code (env+train+third_party+eval+analysis scripts+docs): **~22M**
- Demos (sku09 H5): **~1.6G**
- R3M: **~315M**
- 2 curated checkpoints: **~1.11G**
- **Grand total ≈ 3.17G** (measured) (almost all via Git LFS)

## Exclude

- Entire `runs/` tree (282G, ~407 `.pt`) — keep local only
- Smoke / pilot junk under analysis non-`.py` artifacts (videos, mp4, tar, jsonl dumps, reports already ignored)
- `*.bak*`
- Adroit / Meta-World zarr **not present on disk** (see Missing below) — do not invent or copy without Bill OK

## Curated checkpoints (hardlinks on /data02)

Same filesystem (device id shared with `runs/`), created with `ln` (not `cp`):

```
checkpoints/online_best_495.pt     ← runs/online_billB_official_20261022_lra1e6_sigma0_v2/best_policy.pt
checkpoints/offline_kl1_s250_488.pt ← runs/offline_multisku_v4_20261020_2153/round2/bppo_short_billB_kl1/cand_step0250.pt
```

Originals under `runs/` are **not moved**. Public consumers only need `checkpoints/`.

## Git LFS patterns (`.gitattributes`)

- `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.zarr`, `*.zarr/**`
- `data/demos/**/episodes/**`
- `checkpoints/**`, `models/pretrained/**`

`.gitignore` still ignores `runs/` and blanket `*.pt`/`*.h5`/`*.zarr`, but **un-ignores** `data/demos/**`, `models/pretrained/**`, `checkpoints/**` so LFS can track them.

## Missing vs README (do not block sku09 follow-train)

`data/README.md` / root `README.md` still mention:

- `data/demos/adroit_door_medium.zarr` (~415M) — **absent** under this repo
- `data/demos/metaworld_peg_insert/demonstrations.zarr` (~440M) — **absent**

SKU09 follow-train path does **not** require them. Decide later whether to hardlink from `/data02/kemove/rl100_rgb_experiment/...` or drop from README.

## Staging policy (this prep)

Done now:

- `checkpoints/` hardlinks
- `.gitignore` rewrite (LFS paths allowed)
- `.gitattributes` + `git lfs install --local`
- `UPLOAD_PLAN.md` + README note

**Not** done (await Bill go-ahead on exact set):

- `git add` of ~3G demos/models/checkpoints
- `git commit` of LFS payload
- `gh repo create` / `git remote add` / `git push` (needs Bill auth via #1 1:1)

## Next-step blockers

1. Bill must authenticate `gh` / GitHub on Mac (#1 1:1) before create/push.
2. Confirm exact include set (especially: skip missing Adroit/Meta-World zarr?).
3. After OK: `git add` LFS paths + code → commit → create public `rl100-sim-repro` → push via Mac bridge (A100→GitHub HTTPS TLS historically broken).
