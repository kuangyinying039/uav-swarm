# 三层局部地图与自适应权重 MAPPO 使用说明

## 已实现结构

每架 UAV 维护互不共享内存的三层局部搜索地图：目标存在概率层 `P`、环境不确定度层 `E` 和信息陈旧度/回访层 `S`。`S=0` 表示刚被本机或有效通信消息刷新，随时间向 `1` 老化。地图信息只在当前时刻有效的 UAV 通信边上传递；完全断联时不会融合，恢复链路后才融合较新信息。

异构图包含 UAV、目标/搜索前沿、动态障碍三类节点。目标类节点同时携带 `P/E/S`、是否可见、是否跟踪、前沿标记、回访标记和信念年龄。真实但不可见的目标不会泄露给 actor。

MAPPO 默认不再直接输出 9 个离散航向，而是输出 Dirichlet 分布的 5 维正权重：

1. 目标存在概率收益；
2. 未知区域探索收益；
3. 陈旧区域回访收益；
4. 通信连通保持收益；
5. 局部安全间隔收益。

环境在名义动作集合上计算五项收益，并用策略输出的权重选择综合收益最大的动作；DWA–ORCA 安全层仍只负责最终可执行性修正。

## 快速验证

在 `repro` 目录运行：

```bash
python verify_three_layer_revisit.py
```

该脚本验证观测和异构节点维度、断联不融合/全通信融合、陈旧区域回访增益，以及关闭回访层后第三项收益严格为零。

## 训练命令

单 seed 冒烟训练：

```bash
python train_cooperative_marl.py --algo mappo --search-weights --episodes 100 --seed 11 --out ../outputs/mappo_weight_seed11.json
```

正式多 seed 训练：

```bash
python train_multi_seed_cooperative_marl.py --algo mappo --search-weights --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_three_layer_weighted
```

独立测试：

```bash
python evaluate_mappo_checkpoint.py --checkpoint ../outputs/mappo_weight_seed11.pt --seeds 101,113,127,139,151
```

## 必要消融

直接航向 MAPPO（旧策略形式）：

```bash
python train_multi_seed_cooperative_marl.py --algo mappo --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_direct_actions
```

移除回访层：

```bash
python train_multi_seed_cooperative_marl.py --algo mappo --disable-revisit-map --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_no_revisit
```

UAV-only GAT：

```bash
python train_multi_seed_cooperative_marl.py --algo mappo --no-hetero-entities --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_uav_only
```

除消融开关外，必须保持环境配置、训练 episode、训练 seeds 和独立测试 seeds 一致。比较时报告覆盖率、发现率、完成率、碰撞暴露率、`revisit_gain`、停滞率和安全层干预率，并报告均值与标准差。

## 兼容性

本次改动将 UAV 观测由 37 维扩展为 41 维、目标节点由 10 维扩展为 13 维。当前默认 actor 直接输出 9 个离散动作 logits；五维 Dirichlet 搜索权重仅保留为消融，使用 `--search-weights` 启用。两种动作头的 checkpoint 结构不同，切换后需要重新训练。

## GPU、进度与断点恢复

训练器默认使用 `--device auto`：CUDA 可用时选择 GPU，否则使用 CPU。启动日志会明确打印 `device=cuda` 或 `device=cpu`。PPO 更新按时间步 minibatch 批量执行异构 GAT，不再逐时间步重复前向传播。

```bash
# 单 seed；每 10 轮打印，每 100 轮保存 *_latest.pt
python -u train_cooperative_marl.py --algo mappo --device cuda --episodes 5000 \
  --seed 11 --log-interval 10 --checkpoint-interval 100 \
  --out ../outputs/mappo_weight_seed11.json

# 从中间 checkpoint 恢复，--episodes 表示最终总轮数
python -u train_cooperative_marl.py --algo mappo --device cuda --episodes 5000 \
  --seed 11 --resume ../outputs/mappo_weight_seed11_latest.pt \
  --out ../outputs/mappo_weight_seed11.json

# 多 seed 自动恢复各 seed 已存在的 *_latest.pt
python -u train_multi_seed_cooperative_marl.py --algo mappo --device cuda \
  --episodes 5000 --seeds 11,23,41,59,83 --resume-existing \
  --out-dir ../outputs/mappo_three_layer_weighted
```

训练历史新增 `approx_kl`、`clip_fraction`、`explained_variance` 和 `ppo_early_stop`。若 KL 经常触发早停或裁剪率长期过高，应降低学习率；若 critic 的解释方差长期小于零，应检查奖励尺度或降低 value loss 难度。
