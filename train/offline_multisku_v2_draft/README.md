# offline_multisku_v2_draft (DRAFT, NOT LAUNCHED) - changes vs v1 (offline_multisku_v1_draft, untouched)
Drafted 2026-10-20 CST. Only py_compile + bash -n were run; nothing launched, no GPU used.
| item | v1 | v2 |
|---|---|---|
| PPO lr | 3e-7 (r1), 1e-7 (r2+) | **1e-6 all rounds** |
| KL anchor coef | 6 | **2** (pi_old fixed per round; anchor + approx_kl logged as before) |
| PPO milestones | 150,300 (r1) / 50,100,150,300 | 50,100,150,300 every round |
| PPO KL guard | none | **early stop** when rolling-mean (10 steps) approx_kl(theta||pi_old) > 0.02; KL_EARLY_STOP logged (pipeline.log, ppo kl_early_stop.jsonl, report.json); candidate saved at the stop step; report.json has kl_summary (max/mean/last of approx_kl, anchor_kl, clip_frac) |
| formal gate | select 880000 (>= +1) then confirm 880020 (not worse) | **pooled 200 eps** (select+confirm, 20/SKU each, strict). promote iff pooled(cand)-pooled(input) >= 2 AND exact one-sided paired sign test (= McNemar) p < 0.2 on the 200 pairs. Best = max pooled (tie: smaller action deviation) |
| IL gate | select not-worse | **pooled 200 not-worse** (IL pooled >= pre-IL pooled) |
| start policy | candidate_0300 default | **MS_START_POLICY required** (choose after v1 ends: `python pick_v1_best.py`) |
| collect seeds | 2100000+ | 2200000+ (round r: +r*10000) |
| BC repro check | strict 90 vs loose 97 (apples/oranges) | strict vs strict (90), loose 97 noted |
| unchanged | final unbiased 920000+ untouched/asserted reserved; 910k unused; GPUs default exclude 0,1,2,3,5,6 (use 4,7) + exclude_gpus_override.txt; CONFIRM_LAUNCH=yes guard; launcher refuses S2_OUT inside a v1 run dir |
Files changed: ppo_ms.py, pipeline_ms.py, launch_ms_draft.sh; new: pick_v1_best.py. All others byte-identical copies of v1.
Note: pooled-gate cost = 2 sets x (input + 4 candidates) = 10 x 100-episode evals per round (input cached across rounds when same sha).
Launch (after Bill approves): CONFIRM_LAUNCH=yes MS_START_POLICY=<ckpt> S2_OUT=/data02/kemove/rl100-sim-repro/runs/offline_multisku_v2_<date> bash launch_ms_draft.sh
---
# offline_multisku_v1_draft — multi-SKU offline RL-100 config (DRAFT, NOT LAUNCHED)

Dir: `/data02/kemove/rl100-sim-repro/train/offline_multisku_v1_draft/` (drafted 2026-10-19 CST; **5 decisions CONFIRMED by Bill 2026-10-19**, applied same day). Every file here is a NEW copy-and-edit;
no existing file was modified, nothing was launched, no GPU was used (only `python -m py_compile` and a CPU-only test with `CUDA_VISIBLE_DEVICES=""`).

* Base pipeline: `train/offline_stage2_v3_nominal_20261017/` (wide_pipeline_ic7s9.py, iql_wide_fix.py, transition_model_ic7.py, ppo_wide_fix3.py,
  bc_il_retrain_v4.py, batch_worker_fix.py, amq_ope_ic7.py, demo_features_v3.py, s2common_v3fix.py/wcommon_fix.py, start_nominal.sh).
* Start policy: `runs/bc_multisku_v1_20261018/s43_lr2e-4_mb256/development_v2/candidate_0300.pt`
  (sha256 7b0bb92a…fa9114, the same file as selected_development.pt). BC evaluation of this checkpoint: 880000-set **97/100**
  (A20 B20 C19 D20 E18). Historical BC also reported 910000-set 99/100 — **910k is unused by this pipeline**; final unbiased is 920000+.
* Multi-SKU conditioning: reused `train/bc_multisku_v1_20261018/{evaluate_bc_multisku.py (strict loader + sku_cond_table), bc_model_multisku.py}`
  (imported, not copied or changed). Scenes SKU5A..E = scene_sku10..14 via `LogisticsSortingEnv(sku=...)`.

## Files
| new file | copied from (base) | role | diff vs base (lines +/−) |
|---|---|---|---|
| `ms_common.py` | s2common_v3.py + s2common_v3fix.py + wcommon_fix.py | shared: loader, kernel, dataset, critic | +80 / −114 (vs s2common_v3fix) |
| `demo_features_ms.py` | demo_features_v3.py | encode the 450 multi-SKU demos with the frozen start encoder | +26 / −23 |
| `batch_worker_ms.py` | batch_worker_fix.py | lock-step rollout worker (gate det / collect sto) | +18 / −25 |
| `iql_ms.py` | iql_wide_fix.py | STEP1 critic | +20 / −20 |
| `tm_ms.py` | transition_model_ic7.py | STEP2 transition model (reference only) | +15 / −15 |
| `amq_ms.py` | amq_ope_ic7.py | AM-Q number (reference only, logged) | +13 / −14 |
| `ppo_ms.py` | ppo_wide_fix3.py | STEP3 offline PPO | +58 / −81 |
| `il_ms.py` | bc_il_retrain_v4.py | STEP7 IL retrain | +65 / −81 |
| `pipeline_ms.py` | wide_pipeline_ic7s9.py | orchestrator (7 steps, gates, GPU policy, resume) | +243 / −316 |
| `launch_ms_draft.sh` | start_nominal.sh | launcher (refuses unless `CONFIRM_LAUNCH=yes`) | +23 / −6 |
| `test_cpu_import.py` | — | CPU-only check (no GPU, no env rollout) | new |

## Per-round order (strict, never reordered/merged; each step finishes before the next one starts)
1. **Critic** (IQL) on demos + ALL rollouts so far (round-0 bootstrap + rounds 1..r−1, success AND failure).
2. **Transition model** on the same data. Reference only: it feeds the logged AM-Q number and never makes a decision.
3. **Offline PPO** from the round-start policy. pi_old = round-start policy, fixed for the whole round. KL anchor ×6 to the round-start policy. Data = all accumulated data.
   With `--gate-placement pre_collect` (default), **3b formal gate** runs on the PPO candidates (see Gate).
4. **Collect** 1500 episodes (300/SKU, evenly interleaved), stochastic sampler, using the **gate-passed policy**. If no candidate is promoted, it collects with the **round INPUT policy**, and the log line `STEP4_COLLECT policy=input (no candidate promoted)` says so (state key `collect_policy_name`).
5. **Evaluate / gate**: per-SKU success and failure reasons of the collected rollouts, plus the round's gate record
   (with `--gate-placement post_collect`, the formal gate of the PPO candidates runs here instead; see Decisions).
6. **Merge**: demos + round 0 + all collects 1..r → next round's critic/TM/PPO data (`sources_merged.json`).
7. **IL retrain** of the round's post-gate policy (promoted, otherwise input) on **demos + successful rollouts of THIS round's collect only**.
   No accumulation (asserted), encoder frozen (asserted unchanged). IL gate on the select set: adopted if IL ≥ pre-IL (not worse), otherwise **fall back** to the pre-IL policy.
   The result is the next round's input.

Round 0 (once): demo encoding; start-policy gates on both seed sets (a reproducibility check against the BC evaluation numbers above);
a 1500-episode bootstrap collect with the start policy (300/SKU) so that the round-1 critic has failures.

## Gate (overall mix only)
* Episodes: for each of the 5 SKUs, seeds `880000..880019` (**select** set) and `880020..880039` (**paired confirm** set), 20/SKU = 100 per set. **Final unbiased** reserved `920000..920019` (20/SKU) — never gated or collected by this pipeline. **910k is unused** (historical BC unbiased only).
  ICs come from `init_pose(seed, 5 mm, 2°)`, identical to evaluate_bc_multisku.episode. Deployment sampler: DDIM10 eta=1 step mean (no injected noise, final σ 0), the same generator draws as the BC eval.
  Success = STRICT `flip_pose_no_drop_v2`.
* Decision uses only the **overall** success count. Per-SKU counts are logged and not used.
* Select: the best candidate by overall successes on the select set (tie → smaller action deviation from the round start) must reach input + `select_min_delta`.
* Confirm: candidate vs input on the confirm set (paired, same seeds) must reach Δ ≥ `confirm_min_delta`. Gained/lost counts and the one-sided sign-test p are logged for select, confirm and pooled.
* Promote only if select AND confirm pass. Gate results are cached by (policy sha256, seed set), so the same policy is never re-evaluated.

## Parameter table
| parameter | draft value | base value (ic7s9 / start_nominal) | note |
|---|---|---|---|
| rounds | 3 | 3 | |
| start policy | bc_multisku candidate_0300 | nominal single-SKU BC | |
| SKUs | SKU5A..E, even interleave (i mod 5) | SKU09 only | |
| demos | 400 train + 50 val (80+10 per SKU), sku5 cache bc_v1 | 240+30 original300 | spare/test never read |
| feature dim | 1568 = 3×512 + 28 + 4 sku_cond | 1564 | verified on CPU |
| critic input | [z_{t−1}, z_t, clock, sku_cond] (3141) | [z_{t−1}, z_t, clock] (3129) | explicit SKU |
| IC (collect & gate) | uniform ±5 mm / ±2° | same (nominal) | |
| sampler (collect / PPO likelihood) | DDIM K=10, eta=1, σ clamp [0.0067, 0.1], terminal σ=0 | same (S2_STD_MAX=0.1) | asserted |
| sampler (gate) | DDIM10 eta=1 step mean, final σ 0 | same | |
| reward | sparse terminal success (strict) | same | no human correction; failures kept |
| round-0 bootstrap collect | 1500 (300/SKU), seeds 2100000+ | 1500 | |
| collect per round | 1500,1500,1500 (300/SKU), seeds 2100000+r·10000 | 3000,1500,1500 | **CONFIRMED** Bill 2026-10-19 |
| sto fraction | 1.0 (all stochastic) | 1.0 | |
| held-out flag | every 10th episode per SKU (i//5 mod 10 == 0); demo validation | seed mod 10 | |
| MS_TRAIN_ALL | 1 → critic/TM/PPO/IL train on ALL episodes (held-out = in-sample diagnostics) | 0 (10% held out) | **decide** |
| IQL steps / batch / lr / expectile | 100000 / **2048** / 1e-4 / 0.7, γ .99, Polyak .005 every 2 steps | 100000 / 512 | larger batch |
| TM steps / batch / lr | 20000 / **4096** / 3e-4 | 20000 / 1024 | reference only |
| AM-Q | H=5, 1024 states, logged only | screen/logged | no decisions |
| PPO steps | 300 every round | 300 (min(600,300)) | **decide** |
| PPO batch | **256** (anchor computed on the same batch) | 96 (+32 demo anchor batch) | larger batch |
| PPO lr | r1 3e-7, r≥2 1e-7 | same | **CONFIRMED** Bill 2026-10-19; if clip_frac stays 0 raise to 1e-6–3e-6 |
| PPO pi_old | round-start policy, fixed for the round | refreshed every step (ratio ≡ 1) | rule 3 |
| KL anchor coef | **6 every round**, to the round-start policy | 1 (r1), 3 (r≥2) | rule 2 |
| PPO clip | 0.2 → 0.1 linear, grad clip 1, Adam | same | clip activity logged |
| PPO data | all accumulated: demos + every rollout | wide-flagged rollouts only (+ demo anchor) | rule 4 |
| PPO OPE | log only | accept / revert / patience stop | |
| PPO candidates | r1: m150, m300; r≥2: m50, m100, m150, m300 | same | |
| gate sets | select 880000..19; confirm 880020..39; **final reserved 920000..19** (unused here); 910k unused | 1441000 mix + 880000 nominal; 60-ep screen | rule 8; Bill 2026-10-19 |
| select_min_delta / confirm_min_delta | 1 / 0 (not worse) | delta ≥ 3% & sign p<.05 | **CONFIRMED** Bill 2026-10-19 (accept rare promotions) |
| gate placement | pre_collect | screen before collect | **CONFIRMED** Bill 2026-10-19 |
| IL data | demos + THIS round's collect successes | demos + newest round's successes | rule 6, asserted |
| IL steps / batch / lr / EMA / L2 / demo frac / max-delta | 300 / **256** / 1e-6 / .999 / 1.0 / .125 / .05 | 300 / 64 / same | larger batch |
| IL gate | select set, IL ≥ pre-IL else fall back (`--il-confirm` adds the confirm set) | 420-mix ≥ pre-IL | |
| GPUs | `auto` + **EXCLUDE_GPUS=0,1,2,3 default**; only free GPUs (≥16 GB, no foreign compute) | fixed list 0-3 | rule 9; Bill 2026-10-19 |
| rollout workers | 8 per free GPU, 5 episodes per lock-step shard | 3 per GPU, shard 8 | |

## Changes vs base (rules 1-9)
1. **Order.** Strict 1→7 as above. The 60-episode screen pick is gone. Collect uses the formally gated policy, or the input policy (logged explicitly) when nothing is promoted. AM-Q/TM never decide.
2. **KL anchor** = 6 every round (`--anchor 6`). The reference is the round-start policy.
3. **pi_old.** Fixed round-start copy. Base did `old.load_state_dict(model.state_dict())` every step with one optimizer step per batch, so the pre-update ratio was always 1 and the clip never acted. Now the ratio drifts with θ. `clip_frac`, `ratio_mean`, `clip_active_steps` and `max_clip_frac` are logged; the pipeline warns if the clip never acted. The step-1 log-ratio is asserted to be 0.
4. **PPO data** = demos + all rollouts (all rounds, success and fail). The base's wide-only filter is removed.
5. **Critic and TM** train on demos + all rollouts (success and fail). There is an assert that failures exist.
6. **IL** = demos + successes of this round's collect only (asserted 2 sources). The encoder is frozen and checked by equality. IL gate not worse, else fall back.
7. **ICs.** Uniform ±5 mm/±2° for collect and gate. Collect and PPO use one kernel: DDIM K=10, eta=1, clamp [0.0067, 0.1], terminal σ=0. `S2_STD_MAX=0.1`, `S2_FIX_FINAL=1` and `S2_FLOOR_FINAL=0` are asserted. Sparse reward, no human correction, failures kept.
8. **Gate.** Overall mix only: 20/SKU per set, select 880000+, paired confirm 880020+, strict flip_pose_no_drop_v2. Final unbiased reserved 920000+ (never used here). 910k unused.
9. **GPUs and batch.** Larger batches for IQL/TM/PPO/IL. Parallel rollout workers on every free GPU, and concurrent candidate gates in the same step (workers divided across them). Only free GPUs are used, re-checked each attempt; the pipeline waits rather than sharing a busy card.
* **Multi-SKU.** The policy consumes SKU through the encoder's `sku_cond` key. The CPU test showed the last 4 feature dims are exactly the normalized sku_cond, and changing only the SKU changes only those dims. The critic and TM also take raw sku_cond explicitly. Collect and gate are SKU-balanced, and held-out flags are SKU-balanced (the base's `seed%10` would have put every held-out rollout on one SKU).
* Removed from base: wide/expanded cells, mixcells, the 60-episode screen, the OPE accept/revert, the final generalization sweep and the old single-SKU loaders.

## Estimated runtime / GPU need (estimates, not measured)
Base measurements (ic7s9 round 1, 4 GPUs × 3 workers): IQL 100k@512 811 s, TM 120 s, PPO 300 steps 541 s, collect 3000 episodes 901 s (≈3.3 episodes/s), 600 gate episodes 301 s, IL 60 s.
At 19:20 CST on 2026-10-19 only **GPU 7** was free (GPU 0-6 were held by liu and yu jobs).

| item | 1 free GPU (8 workers) | per extra free GPU |
|---|---|---|
| demo encoding (450 episodes) + start gates (200 episodes) + round-0 collect (1500) | ~20-30 min | rollouts scale ~linearly |
| STEP1 IQL 100k @2048 | ~25-40 min | — (single GPU) |
| STEP2 TM 20k @4096 + AM-Q | ~6-10 min | — |
| STEP3 PPO 300 @256 | ~15-25 min | — |
| 3b gate: 200-400 select + ≤200 confirm episodes | ~4-8 min | |
| STEP4 collect 1500 | ~12-17 min | |
| STEP7 IL 300 @256 + IL gate 100 episodes | ~3-5 min | |
| **per round** | **~65-105 min** | |
| **3 rounds + round 0** | **~4-6 h** | |

GPU memory (estimated): rollout worker ~2-3 GB each (8 → ~20-25 GB). IQL/TM/PPO keep the whole dataset on GPU: about 2.5 GB at round 3 (≈6450 episodes × 61 × 1568 fp32), so <15 GB per training step. Everything fits on one 80 GB A100.
Training steps run on one GPU, one at a time (strict order). Only rollouts and gates parallelize across GPUs.

## How to launch (after approval only)
```
cd /data02/kemove/rl100-sim-repro/train/offline_multisku_v1_draft
CONFIRM_LAUNCH=yes S2_OUT=/data02/kemove/rl100-sim-repro/runs/offline_multisku_v1_<YYYYMMDD_HHMM> \
  PER_GPU=8 ROUNDS=3 bash launch_ms_draft.sh
# EXCLUDE_GPUS defaults to 0,1,2,3 (override only if Bill says otherwise)
# optional: --smoke  (tiny end-to-end; still needs a GPU)
```
The script encodes demos (if missing) on the first free GPU, then starts `pipeline_ms.py` with setsid/nohup. Logs: `$S2_OUT/pipeline.log`, `status_ms.json`, `REPORT_ms.md`.
Resume with `RESUME=1` (state file + done markers). `CONFIRM_LAUNCH=yes` guard remains.

## Checks done (2026-10-19)
* `python -m py_compile` on all 10 .py files with `/data02/kemove/sku09-bc-migration/env/bin/python` (3.10.19): OK. `bash -n launch_ms_draft.sh`: OK.
* `CUDA_VISIBLE_DEVICES= python test_cpu_import.py` → `CPU_TEST_PASS`. It checks:
  * the encoder low-dim keys are [agent_pos, sku_cond] and FEAT=1568;
  * the SKU dims are the last 4;
  * terminal σ=0, the σ clamp holds and the step-1 log-ratio is 0 (9 stochastic steps);
  * synthetic demo+rollout loading works, with critic SD 3141 and the TM shapes.
  
  Temp files and `__pycache__` were removed afterwards.

## Seed table (Bill 2026-10-19)
| role | seed0 | range per SKU | episodes | used by pipeline? |
|---|---|---|---|---|
| **select** (promote step 1) | 880000 | 880000–880019 × 5 SKUs | 100 | yes — overall mix gate |
| **confirm** (promote step 2) | 880020 | 880020–880039 × 5 SKUs | 100 | yes — paired not-worse vs input |
| **final unbiased** (reserved) | 920000 | 920000–920019 × 5 SKUs | 100 | **no** — reserved only; never gated/collected here |
| 910k (historical BC) | 910000 | — | — | **unused** — kept in FORBIDDEN_W so collect cannot touch it |

Collect seeds: round r at `2100000 + r·10000` (round 0 bootstrap at 2100000). All gate/final blocks sit in `FORBIDDEN_W`.

## Decisions — CONFIRMED by Bill 2026-10-19
1. **Gate placement = `pre_collect`.** Formal gate of PPO candidates runs at end of STEP3; STEP4 collects with the gated policy. If nothing is promoted, collect uses the round INPUT policy and logs `collect_policy_name = input (no candidate promoted)`.
2. **Gate rule kept:** select ≥ input+1, confirm ≥ input (not worse). Rare promotions accepted (start policy is already ~97/100 on select).
3. **Final unbiased = 920000+.** Confirm moved to **880020–880039**. Select stays **880000–880019**. **910k is not used** for any gate or final. 920000–920019 is reserved for a separate final eval (not run by this pipeline).
4. **PPO lr stays 3e-7 (R1) / 1e-7 (R2+).** If `clip_frac` / `clip_active_steps` stay 0 across a round (pipeline warns), raise lr to **1e-6–3e-6** and re-draft/re-run that round.
5. **Collect 1500/round (300/SKU).** `EXCLUDE_GPUS=0,1,2,3` is the default in `launch_ms_draft.sh` and in `pipeline_ms.py --exclude-gpus`.

### Still open (not decided)
- **MS_TRAIN_ALL=1** (default): critic AUC / IL held-out loss are in-sample. Set 0 for the base 10% held-out diagnostics.
- **Larger batches** (IQL 2048, TM 4096, PPO 256, IL 256) at fixed lr/steps: keep, or scale lr/steps?
- **IL gate:** select set only (default), or also `--il-confirm`.

## SUCCESS CRITERION CHANGE (Bill, 2026-10-20) - flip_up_no_floor_v3
success = episode complete (480 steps) AND final label_up > 0.98 AND never fell to the floor (parcel bottom < 0.74 m at any step = drop). NO velocity / settle terms, all SKUs
(belt carry-away counts as success; no belt-relative velocity logic). Old strict (last-48 stable) is still logged as strict_success/strict_reason/stable_steps_last_48 (diagnostic only).
Files: batch_worker_ms.py (rows success/task_success/reason now use it; success_criteria_version=flip_up_no_floor_v3; backup batch_worker_ms.py.bak_pre_criterion),
pipeline_ms.py (header note + BC_REF select=97 instead of strict 90; backup pipeline_ms.py.bak_pre_criterion). Gate/IL/pooled-200/sign-test/collect-success/IL-data all consume row[success] so they inherit it.
NOT launched; only py_compile run. NOTE: pick_v1_best.py ranks v1 gates by v1 strict counts (unchanged). Rescore: analysis/rescore_v2_criterion_20261020/.
