# data/ 清单（2026-10-09 CST 复制）

所有文件为源文件的硬链接（rsync -a --link-dest，同一文件系统 /data02），请视为只读。

## demos/sku09-original300/
- 来源：/data02/kemove/sku09-original300-absolute-bc/data/ （SCHEMA.md 来自项目根目录）
- episodes/：300 个 H5 专家示教（`ls episodes | wc -l` = 300），约 1.6G
- splits.json：train 240 / validation 30 / test 30
- manifest.jsonl：300 行，每条含 sha256、bytes、category、result（success/reason 等）
- SCHEMA.md：H5 字段说明
- 未复制：runs/、models/ 及任何权重/rollout
- 成功判据：`flip_pose_no_drop_v2`（仅翻转，不含出料）。不要与 piper_bc_restart 的 outfeed 判据混用。

## sku09-flip-demonstrations（已跳过）
- 来源：/data02/kemove/sku09-bc-migration/sku09-flip-demonstrations/data/episodes
- 跳过原因：300 个文件、文件名一致，`diff -rq` 与 sku09-original300/episodes 内容逐字节相同（inode 不同，是独立拷贝）。无需重复。
- 该目录下的 expanded_v1 (48G)、delta_target_v1、candidates、rejected 等均未复制。

## demos/adroit_door_medium.zarr
- 来源：/data02/kemove/rl100_rgb_experiment/data/adroit_door_medium.zarr
- 约 415M；meta/episode_ends shape=[200]（200 条 episode）；2819 个文件
- data/: action, state, img, depth, point_cloud, reward, done, timeout, return, next_*

## demos/metaworld_peg_insert/demonstrations.zarr
- 来源：/data02/kemove/rl100_rgb_experiment/v2_release_assets_20260922/stage/rl100-v2-data/demonstrations.zarr
- 约 440M；meta/episode_ends shape=[100]（100 条 episode）；含 train/val_episode_indices、episode_seeds；3027 个文件
- data/: action, state, full_state, image_behindGripper, image_corner2, reward, success, done, timeout, return, next_*
- 未复制：同目录 expanded.zarr（1.3G，按要求优先使用 100 ep 版本）
