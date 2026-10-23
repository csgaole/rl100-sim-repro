# rl100-sim-repro

> Public follow-trainable snapshot (prep 2026-10-23 Asia/Shanghai).
> Requires **Git LFS** (`git lfs install` before clone/pull).
> Full training `runs/` (hundreds of GB) are **not** in the public repo — only curated checkpoints under `checkpoints/`.

## Follow-trainable contents

- **Env & task**: `env/` (logistics-sorting-sim + SKU09 `task_success` / `flip_pose_no_drop_v2`)
- **Training code**: `train/`, `third_party/`, `eval/`, analysis `*.py` scripts
- **Demos**: `data/demos/sku09-original300/` (300 H5 episodes + manifest + splits + SCHEMA)
- **Weights**: `models/pretrained/r3m_resnet18.pt` (R3M ResNet18)
- **Curated checkpoints** (Git LFS):
  - `checkpoints/online_best_495.pt`
  - `checkpoints/offline_kl1_s250_488.pt`

See `UPLOAD_PLAN.md` for include/exclude and size estimates (~3.17G (measured) total).

---

> 复制日期：2026-10-09（CST，UTC+8）

## 本仓库当前内容

本仓库为 **follow-trainable** 公开快照：含仿真环境/任务、SKU09 示教、R3M 预训练权重、以及 2 个精选 checkpoint（Git LFS）。
**完整 `runs/` 训练树不上传**（本地可保留）。Adroit / Meta-World zarr 若未附带，见 `data/README.md` / `UPLOAD_PLAN.md`。

```
env/logistics-sorting-sim/        SKU09 物流分拣仿真环境（独立拷贝，可安全修改）
env/sku09-task-success/           SKU09 成功判据实现 task_success.py（参考拷贝）
third_party/rl100/                RL-100 官方 env 相关代码（vendor 拷贝，6MB）
data/demos/sku09-original300/     SKU09 300 条 H5 专家示教 + splits.json + manifest.jsonl + SCHEMA.md
# (optional / may be absent) data/demos/adroit_door_medium.zarr
# (optional / may be absent) data/demos/metaworld_peg_insert/demonstrations.zarr
models/pretrained/r3m_resnet18.pt   R3M weights (Git LFS)
checkpoints/*.pt                    curated online/offline ckpts (Git LFS)
data/README.md                    数据清单（来源、数量、成功判据）
```

## SKU09 成功判据（重要）

- SKU09 使用的成功判据是 **`flip_pose_no_drop_v2`**：**只判翻转（flip）**，不包含出料（outfeed）。
  - 完整 2 s episode；最终 label-up > 0.98；最终包裹底面 > 0.79 m；全程底面不低于 0.74 m。
  - 线速度/角速度/最后 200 ms 稳定窗口只作为质量诊断（`strict_success`），不计入 success/reward。
- **不要与 `piper_bc_restart` 的 outfeed（出料）判据混用。** 两者任务定义不同，指标不可比。

## 来源路径

| 目标 | 来源 |
|---|---|
| env/logistics-sorting-sim | /data02/kemove/sku09-original300-absolute-bc/vendor/logistics-sorting-sim |
| env/sku09-task-success/task_success.py | /data02/kemove/sku09-original300-absolute-bc/src/task_success.py |
| third_party/rl100 | /data02/kemove/sku09-original300-absolute-bc/vendor/rl100 |
| data/demos/sku09-original300 | /data02/kemove/sku09-original300-absolute-bc/data/{episodes,splits.json,manifest.jsonl} + SCHEMA.md（项目根目录） |
| data/demos/adroit_door_medium.zarr | /data02/kemove/rl100_rgb_experiment/data/adroit_door_medium.zarr |
| data/demos/metaworld_peg_insert/demonstrations.zarr | /data02/kemove/rl100_rgb_experiment/v2_release_assets_20260922/stage/rl100-v2-data/demonstrations.zarr |

注意：`data/demos/` 下的文件是对源文件的**硬链接**（rsync --link-dest，同在 /data02，节省空间）。
请把示教数据视为**只读**；原地修改会同时改动源文件。如需修改，先 `cp` 出独立副本。
（env/ 与 third_party/ 是独立拷贝，不是硬链接。）示教 H5、R3M 与 curated checkpoints 经 Git LFS 纳入；完整 `runs/` 仍被 `.gitignore` 排除。

根磁盘 `/` 已满（100%），所有内容必须放在 /data02 下。

## 建议下一步

以 `/data02/kemove/sku09-original300-absolute-bc` 为参考，接入 **BC → offline → online** 三阶段：
1. BC：用 data/demos/sku09-original300 的 train/validation/test（240/30/30）split 训练绝对动作 BC；
2. offline RL：在 BC 策略基础上用 demo（及后续收集的 rollout）做离线改进；
3. online RL：在 env/logistics-sorting-sim 中在线微调，评估统一使用 `flip_pose_no_drop_v2`。

## Stage-1 BC 入口（2026-10-09 新增）

`train/bc_stage1/`：从 sku09-original300-absolute-bc 拷贝的 bc_epoch252 同配方训练器（仅改路径与开发评估采样器）。
启动 `GPU=<id> bash train/bc_stage1/start.sh`，输出 `runs/bc_stage1/`，报告 `runs/bc_stage1/REPORT.md`。
R3M 预训练权重需放在 `models/pretrained/r3m_resnet18.pt`（来自旧仓库 assets，SHA-256 b4dd6f40…36d2）。
