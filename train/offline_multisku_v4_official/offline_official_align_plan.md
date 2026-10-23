# 离线 RL 按官方 repo 完整改造 — Diff / PR 方案（确认前不改代码）

日期：2026-10-20 Asia/Shanghai（修订：多卡并行 PPO 升为一期必做）  
对照：官方 `RL-100` @ 已 vendored 路径，主 yaml **`rl100_2d_epsilon.yaml`** + `train_ddp.py` / `unio4` / `uni_ppo.py`  
当前：`offline_multisku_v3`（commit 一带 `4713a04` + P0 `966566f`），run `offline_multisku_v3_20261020_1851`（已暂停）  
原则：**新目录实施，不覆盖暂停 run**；BC s43 不动。

---

## 0. 目标与非目标

**目标**：离线策略/管线按官方 ε 离线环对齐（顺序、超参、Dyn/OPE、无离线 KL loss、IL 数据与 Critic/BPPO 默认）。

**非目标（除非 Bill 另批）**  
- 不中途 resume 旧 v3 PPO  
- 不自动重接 GPU2/3 跑旧配方  
- 不把 R5 截图配方当作官方  

**建议默认保留（与官方 ε 兼容 / Bill 固定）** — 方案里标为「保留项」，确认卡可选去掉：  
- DDIM η=1、终步 σ=0、std clip 下界 0.0067（epsilon 同族；上界 0.1 是否改 null 见下）  
- IC ±5mm/±2°  
- 无人工纠偏  
- 闭环仿真门控（可与 OPE **并行保留**，晋升规则需定）  
- 稀疏/终局成功 reward（官方无 duration 塑形；与 `final_reward` 旗标含义不同）

---

## 1. 管线顺序（P0 结构）

| | 当前 v3 | 官方 | 改造 |
|---|---|---|---|
| 每轮主序 | IQL → TM → PPO → collect → eval → merge → IL | **IL → Feature → IQL → Dynamics → BPPO**（再 rollout/飞轮） | **改为 IL-first**：每轮先 IL（或轮初 IL）→ 全量特征重抽 → IQL → Dynamics → BPPO → collect → gate |
| 严格 7 步 Bill 旧序 | 已固化 | 颠覆 | **必须改**；文档与 `pipeline_ms` 重写 |

**PR 切片 A**：新 `pipeline_official_ms.py`（或 `offline_multisku_v4_official/`），状态机按官方序；旧 `pipeline_ms.py` 只读归档。

---

## 2. BPPO / 离线 PPO（P0 超参）

| 项 | 当前 | 官方 epsilon `unio4` | 改造 |
|---|---|---|---|
| steps | 3000 | **8000** | → 8000 |
| lr | 1e-6（r2+ 3e-7） | **1e-5** | → 1e-5；取消 r2 额外降 lr 或改为官方无此档 |
| batch | 256 | **512** | → 512 |
| clip | 0.1 固定 | **0.25**，`is_clip_decay: true` | → 0.25 + decay |
| grad clip | 0.5 | 代码硬编码 0.5 | 保持 |
| entropy | 0 | 0 | 保持 |
| KL anchor=6 | 有 | **无** | **删除** |
| 每维自适应 KL | 有 | 离线无（online 才有 adaptive_kl） | **删除离线 KL 调度**；可选仅日志 approx_kl |
| π_old | 轮起点固定 +（v3 有 refresh 意图） | `is_update_old_policy: True` | **按官方刷新 π_old** |
| 并行 | 单卡 | `train_ddp` | **一期必做多卡并行 PPO（DDP）**（Bill 已定）；接官方 `train_ddp`；按 **world_size** 换算 **per-GPU batch = global512/N**；lr 跟官方同卡配置（默认不另做 linear scale，除非官方脚本已有） |

**PR 切片 B**：`ppo_official_ms.py` 从官方 `uni_ppo` / `train_ddp` 离线路移植/薄封装，去掉 anchor；对齐 clip/lr/steps/batch；**多卡 DDP 与官方同路径**。IQL 是否一期 DDP 另开（默认仍可单卡）。

---

## 3. Critic / IQL（P0）

| 项 | 当前（贴 R5） | 官方 epsilon | 改造 |
|---|---|---|---|
| Q | 1024+LN | **512、无 LN** | → 官方默认 |
| V | 512 | 512 | 保持 |
| q/v lr | 1e-5 | **1e-4** | → 1e-4 |
| τ | 0.0025 | **0.005** | → 0.005 |
| expectile/ω | 0.7 | 0.7 | 保持 |
| fix_encoder | true | true | 保持 |
| steps | 100k | `num_critic_epochs: 600`（量级不同） | 用官方 epoch/步数语义换算，或明确「等价更新次数」表 |

**PR 切片 C**：`iql_official_ms.py` / `ms_common` Critic 结构改回官方宽；fail-fast 特征同空间保留（P0 已做，继续）。

---

## 4. Dynamics（P0 功能）

| 项 | 当前 | 官方 | 改造 |
|---|---|---|---|
| 角色 | TM 跑、AM-Q **仅日志** | LDDM/dynamics **进 OPE/备份** | **进决策**：OPE 可拒绝/回退更新 |
| ensemble | defer 7/5 | 正式 ensemble | 按官方 `dynamics` 段启用 |

**PR 切片 D**：接官方 dynamics 训练 + OPE；晋升策略见 §6。

---

## 5. IL / 特征（P0）

| 项 | 当前 | 官方 | 改造 |
|---|---|---|---|
| 时机 | PPO/collect **之后** | **IQL 之前** | 随 §1 |
| 数据 | demos + **仅本轮成功** | 累积 finetune | → **累积**（demos + 历史成功；失败是否进 IL 跟官方 loader） |
| encoder | IL 解冻双 lr | BC/IL 动 encoder；离线再 freeze | 保持「IL 动 / IQL·BPPO 冻」 |
| 特征 | P0 全量 reencode + RGB | IL 后 obs2feature | 保持全量同空间；IL 后强制 reencode |

**PR 切片 E**：IL 数据源改累积；pipeline 调用点移到轮前。

---

## 6. 门控 / 晋升（需 Bill 拍板）

| 项 | 当前 | 官方 | 选项 |
|---|---|---|---|
| 主晋升 | 仿真 mix gate | OPE / eval | **A**：OPE 主、仿真辅；**B**：仿真主、OPE 辅；**C**：两者都过才晋升 |
| IC | ±5/±2 | 任务相关 | **保留 ±5/±2**（建议） |

**默认提案**：**C**（OPE 不差 + 仿真 gate 不差）最稳，墙钟更长。

---

## 7. 奖励 / 采样 / γ（P1）

| 项 | 当前 | 官方 | 提案 |
|---|---|---|---|
| 奖励 | 稀疏 success | 环境 reward + `final_reward`（优势时机，**非** duration 塑形） | **保留稀疏**；实现上对接 final_reward 优势路径 |
| 采样 | DDIM η=1 σ_终=0，clip [0.0067,0.1] | epsilon：下界 0.0067，**max null** | 保留 η/σ0；**是否放开 std max** 单开开关 |
| 策略族 | **epsilon / DDIM** | 另有 `rl100_2d_flow.yaml` | **本改造仍跟 epsilon**（见 §9.13）；Flow 另开实验轨 |
| γ | 0.999/chunk | **0.99** | → 0.99（或保留 chunk 语义并文档化） |

---

## 8. 工程落地

1. 新目录：`train/offline_multisku_v4_official/`（复制再改，不碰 `…_1851`）  
2. 文档：本文件 + 逐文件对照表（官方符号 ↔ 我们脚本）  
3. Smoke：短 IQL / 短 BPPO / 一次 OPE / 小门控  
4. 全量：停 BC 空卡或 exclude 0,1 后启动  
5. **禁止**：resume 旧 v3 配方；中途改正在停的 run

---

## 9. 确认清单（Bill 回复即可开工）

**开工授权**（Bill 2026-10-20「按以上方案，开始离线RL」）：未逐条回复的项按下方默认实施。

1. IL-first → **Y**  
2. 删 KL anchor + 离线每维自适应 → **Y**  
3. BPPO 8000 / 1e-5 / 512 / clip0.25+decay → **Y**  
4. Critic Q512 无 LN、lr1e-4、τ0.005 → **Y**  
5. Dynamics 进 OPE/晋升 → **Y**  
6. IL 数据累积 → **Y**  
7. 晋升 → **C**（OPE + 仿真都过）  
8. 保留稀疏、DDIM/σ0、IC±5/±2、仿真 gate → **Y**  
9. std clip max → **暂保持 0.1**（可后开 null 开关）  
10. γ → **0.99**  
11. 一期多卡 PPO DDP → **已定 Y**  
12. IQL/Critic 一期 DDP → **N**  
13. Flow → **已定 N（keep_epsilon）**

**GPU 满载规则**：BC 占 0–1 时，离线只用 **2–7**（PPO DDP + collect/gate）；全 8 卡需 BC 结束或另批。禁止 resume 旧 v3。

---

## 10. 建议工期（确认后）

- Day0：目录骨架 + 超参 yaml + **PPO DDP 骨架**（切片 B）  
- Day1：pipeline 顺序 + IL 累积（A/E）+ Critic 对齐（C）  
- Day2：Dynamics+OPE（D）+ 门控策略  
- Day3：多卡 smoke（PPO）→ 修 → 再谈全量  

（实际以 #1 机器与官方源码挂载为准。）
