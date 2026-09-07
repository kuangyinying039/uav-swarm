# 围捕学习检查与场景配置

> 奖励已更新：以下奖励审计是历史记录，当前组成以 [PURSUIT_REWARDS.md](PURSUIT_REWARDS.md) 为准。

本次仅修改围捕路径，没有修改搜索代码，没有正式训练。截图的 MAPPO 0/20 和 MPC 11/20 不能证明修改后性能；必须重新训练和独立评估。

## 已修复的实现问题

1. `train_pursuit_with_demos.py`：围捕 PPO 禁用 dropout。旧策略采样和更新时的随机 dropout 掩码不同，会使未更新参数时的动作概率比也偏离 1。动作探索仍由可学习的高斯分布提供。共享搜索训练配置不变。
2. `quadrotor_pursuit_env.py`：到达任务截止时间属于终止失败，停止价值自举。旧实现一面给予超时失败惩罚，一面把结束当作可以继续估值的时间截断。训练器提前截取 rollout 的截断语义不变。
3. 同一文件：补齐平面观测中的目标相对高度、估计三维速度、剩余时间。输入来自本机跟踪估计，不读目标真值。原图网络已有目标三维节点；因此缺失高度不是默认 GAT 零捕获的充分解释，但影响无图网络，并且旧观测没有明确截止时间。

观测协议版本为 `policy_observation_version=2`。旧示范与模型不应继续混用，即使张量维数相同；命令入口会拒绝旧协议。请重新采集，使用新文件和输出目录。

## 奖励组成与局限

默认奖励：成功捕获 +100；超时失败 -50；每步 -0.05；距离接近奖励为 `2 * clip(上步最近捕获距离差 - 本步距离, -1, 1)`；另有碰撞、控制拒绝、控制平滑、通信和估计项。

300 步未成功，仅时间和超时就累计 -65。因此负奖励本身正常。接近奖励只反映最近一架无人机，且长时间接近后又远离会抵消早期收益，对其余队员的协作指导偏弱。它也不是含折扣因子的严格势函数奖励，不应声称保证最优策略不变。

本次先修实现，保留奖励权重以避免一次混入多个无法归因的实验因素。训练已有 `reward_diagnostics.jsonl`，记录各项累计奖励、最近捕获距离差和失效机数；`reward_capture.svg` 在一张图上展示奖励与捕获率。先看是否持续接近目标，或碰撞/拒绝控制占主导，再决定增加团队接近项或重调惩罚。BC loss 下降只代表拟合教师动作，不等于闭环捕获成功。

## 修改场景的文件

**建议直接编辑 `repro/configs/pursuit_v2/learning_start.json`，通过 `--env-config` 传入所有算法。** JSON 中未写的参数继承默认配置。

| 内容 | 文件/字段 |
|---|---|
| 场地水平边长、障碍数量 | JSON 的 `grid_size`、`building_count` |
| 障碍宽度范围 | JSON 增加 `building_min_size`、`building_max_size` |
| 高度范围和建筑高度 | `pursuit_evasion_3d_env.py` 的 `Pursuit3DConfig`；也可用同名字段覆盖 |
| 追逃速度、加速度、响应时间、噪声 | `quadrotor_pursuit_env.py` 的 `QuadrotorPursuitConfig` |
| 初始距离 | JSON 的 `initial_distance_min/max` |
| 随机初始方位与预测逃逸算法 | `pursuit_scenarios.py` |
| 距离捕获判据及额外接近奖励 | `quadrotor_pursuit_env.py` |
| 基础捕获、超时、碰撞等奖励 | `pursuit_evasion_env.py` |
| MAPPO、行为克隆及 DAgger 纠偏 | `train_pursuit_with_demos.py` |

默认场景：25×25 m，高度 1–12 m，5 个建筑障碍，初始距离 8–16 m，追捕水平速度 1.4 m/s，目标 1.82 m/s。新起步配置保留场地，改为 2 个障碍、随机初距 6–10 m、目标水平速度 1.54 m/s（速度比 1.1），仍保留预测逃逸、持续带噪观测和随机方位。这些是未做硬件测量的仿真假设，不是论文原值，也不保证易学。

捕获条件已按用户要求更新：任一有效无人机到目标的三维中心距离 ≤1 m，当前步立即成功，与姿态无关。`target_diameter=0.5` 为仿真等效直径，`capture_radius=2*target_diameter` 自动推导；请修改 target_diameter，不要单独覆盖派生的 capture_radius。旧网具示范和检查点会被拒绝，需要重新采集和训练。这是团队协助任一成员完成距离捕获，不要求多机同时包围。

## 论文提供的参考

- [DualCL](https://arxiv.org/html/2312.12255v2)，V-A：圆形场地半径 0.9、高 1.2，0–3 个圆柱障碍（半径 0.3），追捕速度 1.0、最终目标速度 2.4，距离捕获阈值 0.12，每回合 800 步。采用任务难度与初始状态双课程；高速度比是课程终点，不是要求随机初始策略直接学会。
- [OPEN](https://arxiv.org/html/2409.15866v3)，V-A：场地半径 0.9、高 1.2，4–5 个圆柱障碍（半径 0.1），3 追 1，速度 1.0 对 1.3，距离捕获阈值 0.3，800 步。包含遮挡下预测与自适应场景学习。

本项目现也采用三维距离捕获，但阈值遵从用户指定的 1 m。不能只照搬障碍数量或捕获半径；还需考虑障碍占地、场地/捕获尺度比、速度比与回合物理时长。项目 300 步×0.2 s=60 s。起步配置借鉴渐进难度，不声称复现论文。所有对比必须使用相同配置、测试种子、观测来源和捕获条件，起步配置成绩不能与旧默认场景成绩混比。

## Linux / VSCode 终端运行顺序

以下每条都是单行。先同步修改后的代码到服务器，在项目根目录运行。

重新采集 10 条成功示范：

```bash
python repro/train_pursuit_with_demos.py collect --env-config repro/configs/pursuit_v2/learning_start.json --teacher mpc --demo-successes 10 --demo-attempts 100 --demos outputs/pursuit_distance1m_start_demos.pt
```

先行为克隆，并使用已有 DAgger 功能在学生实际访问的状态上让教师纠偏：

```bash
python repro/train_pursuit_with_demos.py pretrain --env-config repro/configs/pursuit_v2/learning_start.json --demos outputs/pursuit_distance1m_start_demos.pt --bc-updates 1000 --correction-rounds 3 --correction-episodes 5 --correction-updates 500 --torch-threads 1 --seed 11 --out outputs/pursuit_distance1m_init
```

在验证种子上比较纠偏策略与 MPC。若仍零捕获，先检查初始化策略和奖励诊断，不要直接扩大到长训练：

```bash
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/pursuit_distance1m_init/corrected.pt --methods mappo mpc --eval-seed 4000000 --eval-episodes 20 --out outputs/pursuit_distance1m_init_check
```

进行 MAPPO 微调：

```bash
python repro/train_pursuit_with_demos.py train --checkpoint outputs/pursuit_distance1m_init/corrected.pt --demos outputs/pursuit_distance1m_start_demos.pt --demo-weight 0.1 --episodes 1000 --validation-interval 50 --validation-episodes 20 --torch-threads 1 --seed 11 --out outputs/pursuit_distance1m_train
```

最终独立测试（与上述验证种子不同），检查点自带环境配置：

```bash
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/pursuit_distance1m_train/best_capture.pt --methods mappo apf frpn mpc --eval-seed 3000000 --eval-episodes 100 --out outputs/pursuit_distance1m_test
```

这组流程验证起步场景；之后再单独训练默认困难场景。当前 CLI 没有自动跨环境课程或允许任意配置覆盖旧检查点的功能，不应把同场景断点续训称作课程迁移。
