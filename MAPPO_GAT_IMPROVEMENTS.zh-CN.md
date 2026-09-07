# MAPPO+GAT 改进说明与实验协议

## 本次实现改进

当前 MAPPO+GAT 采用集中训练、分散执行框架，并完成以下增强：

1. 使用共享团队奖励对应的单一集中式 critic，减少多个重复 critic 的估计漂移。
2. 使用 GAE(λ) 估计优势函数，并对优势做标准化。
3. PPO 更新采用随机时间步 minibatch 和多轮更新，而不是逐步累积整条轨迹损失。
4. 学习率由初始值线性衰减到 `lr_end_factor` 对应比例；熵系数由初始值线性衰减到 `entropy_coef_end`。
5. 每个训练 episode 使用不同随机种子生成目标、障碍、干扰和故障过程，降低固定场景记忆。
6. GAT 节点输入新增信念信息年龄和局部干扰强度，使注意力编码能够区分新鲜/陈旧信息以及不同通信质量。
7. 训练历史新增 policy loss、value loss、entropy、learning rate、entropy coefficient 和安全层干预率。
8. 奖励新增覆盖率增量、新访问网格、逐 UAV 可避免停滞、真实碰撞和永久失效分量。
9. UAV 观测新增分区 frontier 相对方向、分区访问覆盖、frontier 价值和连续停滞状态。
10. MAPPO 训练会保存同名 `.pt` checkpoint，可在未参与训练的 seeds 上执行确定性评估。

## 论文中建议采用的准确表述

可将方法描述为“弱通信感知的局部时变异构图 MAPPO”。图中显式包含 UAV、目标 belief 和动态障碍三类节点。目标 belief 节点由当前可见目标与 TPM frontier 假设共同组成；未被传感器观测的真实目标不会进入 actor。障碍节点只包含局部观测半径内风险最高的动态障碍。UAV-UAV 消息受实时通信邻接矩阵约束，UAV-target 与 UAV-obstacle 使用关系特定的 masked multi-head cross-attention。集中式 critic 使用全局状态训练，actor 在执行阶段仅使用局部观测和当前可用通信图。

默认容量为每架 UAV 6 个目标 belief 节点和 4 个障碍节点，实体不足时通过 mask 屏蔽，因此支持动态节点数量并保持批训练张量形状稳定。

## 异构图消融

三组核心消融命令如下：

```bash
# MLP：完全无图
python train_multi_seed_cooperative_marl.py --algo mappo --no-gat --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_mlp

# UAV-only GAT：保留通信图，关闭目标/障碍实体节点
python train_multi_seed_cooperative_marl.py --algo mappo --no-hetero-entities --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_uav_gat

# 完整异构图 GAT（默认）
python train_multi_seed_cooperative_marl.py --algo mappo --episodes 5000 --seeds 11,23,41,59,83 --out-dir ../outputs/mappo_hetero_gat
```

实体容量可通过 `--hetero-target-nodes`、`--hetero-obstacle-nodes` 和 `--entity-observation-radius` 调整。正式对比中除消融开关外，其余环境和算法参数必须保持一致。

## 独立评估

训练输出 JSON 的同时会生成同名 `.pt`。例如：

```bash
python train_cooperative_marl.py --algo mappo --episodes 5000 --seed 11 --out ../outputs/mappo_gat_seed11.json
```

随后使用与训练 seeds 不重叠的测试场景，确定性选择最大概率动作，并在相同场景运行启发式：

```bash
python evaluate_mappo_checkpoint.py --checkpoint ../outputs/mappo_gat_seed11.pt --seeds 101,113,127,139,151
```

评估会生成 JSON 和 CSV，分别报告 MAPPO 与启发式的均值、标准差和逐 seed 指标。旧模型的观测维度与新环境不一致，不能继续加载，完成本次改动后必须重新训练。

## 推荐训练命令

先运行单 seed 功能验证：

```powershell
python .\repro\train_cooperative_marl.py --algo mappo --episodes 100 --search-steps 180 --gat-heads 4 --gat-layers 2 --hidden-dim 128 --lr 0.0003 --lr-end-factor 0.1 --clip-eps 0.2 --entropy-coef 0.01 --entropy-coef-end 0.002 --batch-size 32 --update-epochs 5
```

正式实验建议至少五个训练 seed：

```powershell
python .\repro\train_multi_seed_cooperative_marl.py --algo mappo --episodes 2000 --seeds 11,23,41,59,83 --search-steps 180 --gat-heads 4 --gat-layers 2 --hidden-dim 128 --dropout 0.05 --lr 0.0003 --lr-end-factor 0.1 --clip-eps 0.2 --entropy-coef 0.01 --entropy-coef-end 0.002 --batch-size 32 --update-epochs 5
```

GAT 消融至少包括：无 GAT、2-head GAT、4-head GAT。所有方法应使用相同训练 episode 数、环境随机序列和安全层。

## 应报告的结果

除总奖励外，至少报告覆盖率、发现率、跟踪率、完成率、碰撞率、通信中断率、失效率、安全层干预率以及五个 seed 的均值和标准差。若 MAPPO+GAT 的安全层干预率更低，可支持“策略本身形成了更安全的动作偏好”；若只有碰撞率下降但干预率未下降，则优势主要来自共享安全层。
