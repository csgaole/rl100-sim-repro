# HDF5 schema v1.0

回合固定 2 秒。索引 k 对应动作施加前 t=k/240，动作作用于 [k/240,(k+1)/240]。图像索引 j 对应 t=j/30；图像 j 对齐从动作 8j 开始的后续动作块，不使用未来图像预测过去动作。

- `observations/rgb/{global,left_wrist,right_wrist}`：uint8，(61,240,320,3)，无覆盖文字，RGB。
- `observations/timestamp`、`observations/rgb_timestamp/{camera}`：(61,)，三相机同一物理状态、无中间 step；理想仿真时钟网格秒值。
- `observations/proprio`：(61,28)，前 14 维为左 q[6]/宽度、右 q[6]/宽度，后 14 维按相同次序为速度。单位 rad、m、rad/s、m/s。
- `proprio/state`：(480,28)，动作前 240 Hz 状态；`proprio/timestamp`：(480,)。
- `actions/requested`、`actions/effective`：(480,14)，float64；第 6、13 列固定 0.012 m。前者为请求，后者为实际生效的执行器关节位置目标。
- `actions/actuator_ctrl`：(480,nu)，本 tick 真正写入 MuJoCo 的完整 ctrl，含夹爪执行器。
- `actions/timestamp`：(480,)，从 0 到 479/240。
- `outcome/reward`：(480,)，仅最后一个 tick 的 2 秒任务成功奖励为 1。
- `outcome/terminated`：(480,)，仅最后一个为 true。成功轨迹均完整保留到 2 秒；不会在首次稳定时提前裁短。

以下内容不得作为默认策略输入：

- `diagnostics/qpos`、`qvel`：动作前完整仿真状态（含物体），对齐 `diagnostics/pre_step_timestamp`。
- `diagnostics/object_pose`：(480,7)，动作后物体世界位置 xyz、四元数 wxyz。
- `diagnostics/object_velocity`：(480,6)，动作后世界系角速度 xyz、线速度 xyz。
- `diagnostics/contact_force`：(480,2)，动作后各臂夹爪与物体接触的正法向力之和，N。
- `diagnostics/bottom`、`label_up`、`stable`：动作后评估量；对齐 `diagnostics/post_step_timestamp`，从 1/240 到 2 秒。
- `diagnostics/phase`：动作前参考轨迹索引；`recovery_active`：本 tick 前是否触发恢复。它们是专家内部状态，不是传感器。
- `diagnostics/final_qpos`、`final_qvel`：2 秒最终仿真状态。

HDF5 属性：

- `config_json`：初始完整 qpos/qvel/ctrl、位置/朝向扰动、质量/惯量、逐 geom 摩擦和名称、控制增益、频率、图像规格、控制器全部参数、搜索种子、动作语义。
- `result_json`：最终成功/终止原因、首次完成连续 200 ms 稳定的时间、峰值夹爪力、最终速度/姿态、恢复触发 tick。
- `no_recovery_counterfactual_json`：仅恢复类，关闭纠正的同条件失败摘要。
- `provenance`、`category`、`episode_id`、`schema_version`：来源与索引。

依赖版本、仿真 Git revision 和本地快照内容哈希在 `reports/provenance.json`；逐回合文件 SHA-256 在清单与审计文件中。配置 JSON 及 diagnostics 与 observations 分开；训练读取器通过白名单字段读取，禁止将 HDF5 所有字段自动拼接为输入。
