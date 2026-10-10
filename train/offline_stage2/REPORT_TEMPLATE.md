# Stage-2 离线 RL（RL-100 复现，SKU09 翻转，flip_pose_no_drop_v2）— REPORT

起点策略：`runs/bc_stage1/selected_development.pt`（epoch 250；dev 880000–880099 100/100，final 910000–910099 **99/100**）。
代码：`train/offline_stage2/`；输出：`runs/offline_stage2/`（status.json / pipeline.log / pipeline_state.json / round*/ / evals/）。

## 已有工作复查（sku09-original300-absolute-bc/rl_stage2）

- 旧仓库做过多轮离线 RL（IQL + 动力学 AM-Q + BPPO），**均未得到改进**：旧 BC 181/200 → 被拒绝候选 142/200（新增失败全为 wrong_face）；V10 保守 BPPO 也未证明改进；V11 诊断因 Q 动作排序检查失败而停止。旧 run 数据（rollout H5、critic、feature cache）已不在磁盘上（runs/ 仅剩 32K 的失败状态文件），无法直接复用。
- 复用的是**经验与代码思路**：rl_core.py 的 DDIM 核/链式采样与逐步对数似然、rollout 结构、BPPO 事务式更新；以及 rl100-offline-failure-analysis 的结论（只有成功示教的 critic 缺乏动作排序能力；PPO 偏移大时会破坏翻转动作；Q/AM-Q 的小幅上升不等于闭环提升）。因此本次：先采集含失败的 rollout，再训练 IQL；离线门控之后**必须**过闭环 dev 门控才晋升。

## 实现的流程（每轮 r = 1..3）

1. **采集**：用当前策略的随机版（上游 step_mean 均值 + N(0, clamp(σ_eta1, .0067, .01))），每轮 600 条完整 2 s episode：±5mm/±2°、±10mm/±5°、±20mm/±10°、±40mm/±20° 各 150 条；种子 700000+r·20000+range·5000（不碰 880000–880099、900000–900999、910000–910099）。成功/失败全部保留。稀疏奖励：最后一个 chunk 成功=1，否则 0。
2. **IQL**（上游 rl100_2d_epsilon.yaml critic 配置）：double Q + V，expectile ω=0.7，γ=0.99/chunk，MLP 512×3，lr 1e-4，τ=0.005 每 2 步，batch 512，100k 步。输入 = 冻结 Stage-1 编码器特征 [z_{t-1}, z_t]（各 1564 维，按训练集标准化）+ chunk 时钟；动作 = 8×12 关节维（归一化，夹爪列恒定被排除）。数据 = 示教 270 条（train+val）+ 所有已采 rollout；按 episode 留出 10%（示教 validation 30 条 + seed%10==0 的 rollout）做 held-out 诊断（TD、MC 回报 RMSE、V 对成功/失败的 AUC）。
3. **离线策略改进**（上游 unio4/BPPO 风格）：K=10 去噪步视为 MDP；在数据集状态上从 old 随机策略采样整条去噪链，优势 A=minQ(s,a0)−V(s)（batch 标准化），每个去噪步用 PPO clip 目标各做一次 Adam 更新（lr 1e-6，clip 0.25→0.1 线性衰减，grad clip 1.0，batch 128），每轮最多 400 外循环步；编码器冻结，只更新 UNet。old（采样/似然参考）策略每个外循环步刷新为当前策略（标准 PPO 快照）。
4. **OPE 门控**：每 25 步，在固定 2048 个 held-out 状态上，用部署采样器（均值链、共享初始噪声）对当前策略与“上一个被接受策略”做配对 minQ 打分；增益 > max(2·SE, 1e-3) 才接受并保存候选；连续 4 次不接受则回退到上一个被接受策略并重置优化器；连续 12 次不接受则提前停止。
5. **闭环 dev 门控**（Stage-1 同协议 evaluate_bc.py --sampler upstream）：dev 880000–880099（±5mm/±2°，100 ep）+ 扩展诊断 790000–790099（±20mm/±10°，100 ep）。晋升条件：narrow ≥ 参考−1 且 narrow+expanded > 参考之和。晋升后下一轮用新策略采集与改进。
6. **最终**：最后一个晋升策略在 910000–910099 上评估 100 ep，与 Stage-1 的 99/100 对比；若没有任何晋升，Stage-1 保持选中。

## 与 RL-100 的偏差（如实记录）

- **无学习的动力学模型 / AM-Q**：OPE 用 IQL 的 minQ 在 held-out 状态上做配对比较（相当于 horizon-0 的 AM-Q），并加真实仿真闭环 dev 门控作为最终把关（仿真里可行且更可靠；旧仓库的 AM-Q 曾给出与闭环相反的结论）。
- **critic 额外输入 chunk 时钟**（固定 2 s 任务、只在结尾判定成功，价值依赖剩余时间）；actor 输入不变。
- **似然只算 12 个关节维**（夹爪列执行前被覆盖为常数）；随机核 σ 上限 0.01（σ 下限 .0067 同上游 clip_std_min）。原因（启动前实测，±5mm/±2°）：Stage-1 均值策略 3/3 成功，但 σ 上限 0.1 时随机策略仅 4/8、0.03 时 3/8、0.01 时 7/8——该策略对去噪噪声极其敏感，σ 太大时行为策略与部署策略相差过远，且 PPO 信赖域过宽。部署采样仍是 Stage-1 的上游 step_mean（eta=1 均值），确保与 99/100 可比。
- **old 策略每步刷新 + lr 1e-6 + batch 128 + 每轮最多 400 步**（上游：old 仅在 OPE 提升时更新、lr 1e-5、batch 512、8000 步）。原因（启动前实测）：σ≈0.007–0.01 时 96 维联合似然比对参数极敏感，按上游设置 20 步后 max|log ratio|≈1.4e4、clip 比例 100%、梯度为 0，而 Adam 动量仍继续推动参数（动作偏移 0.029），上游锚定方式在本任务直接失效；GPU 与其它任务共享，batch 256 实测 8.8 s/步。
- **未做** consistency distillation、每轮后的 IL 重训/编码器更新、online RL（Stage-3）。
- 示教来自 Stage-1 cache（由硬链接 H5 只读构建），test 30 条从未读取。
