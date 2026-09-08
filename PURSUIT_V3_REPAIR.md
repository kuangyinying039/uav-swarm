# 围捕执行、奖励与 PPO 修复（2026-09-08）

## 适用范围

修复 `train_pursuit_with_demos.py` 的连续三维围捕路径。保持原始场景初始化、目标速度、观测噪声、300 步期限、任意一架无人机进入捕获半径即成功的规则。没有植入教师动作、自动开启模仿学习、降低目标速度或放宽捕获条件。离散搜索任务的 PPO 更新不受影响。

配置增加 `execution_reward_version=3`；策略算法标识为 `mappo_pursuit_safe_v3`。旧 checkpoint 和旧示范数据不能直接续训，代码会明确拒绝。原 `pursuit_v2/speed_ratio_1_1.json` 只覆盖场景速度，仍然可用；目录名字中的 v2 表示场景版本，不是新的执行/奖励版本。

## 已确认的问题与对应修复

### 安全执行

下载的旧轨迹中，策略持续输出正 x/y 速度，抵达 24.8 m 边界仍向外飞；第 3000 回合 900 个无人机动作中 870 个被整步拒绝。

现在先根据边界、建筑、球形障碍物和共享队友状态构造速度约束，预算中考虑速度响应延迟和制动距离。顺序投影修正危险的法向速度，保留可行的切向分量；速度与加速度限制仍由同一飞控模型执行。每个积分子步做路径穿越检查，队友之间检查同步相对运动路径。投影不能得到可执行轨迹时尝试制动，最后才用显式记录的紧急停车。

这是一种工程化局部安全过滤器，不是经过证明的 CBF/QP 最优投影，也不是实机安全认证。非可恢复动量、初始重叠和移动障碍物仍需要紧急处理；停车兜底属于仿真约定。安全层不读取目标真值、不负责追击目标。所有基准与 MAPPO 共用此层。

另外修复了重复执行：父环境的二维离散 DWA/ORCA 原本可能把传入的悬停占位动作变成二维移动，改写已经完成飞控积分的位置。连续围捕现在只允许父环境接收未改写的悬停占位动作；位移和安全控制只执行一次。高度上安全分离的无人机不会再被二维规则额外移动。

### PPO

- 参考官方 MAPPO 的连续动作头，将输出层改为 `gain=0.01` 的正交初始化、零偏置，避免初始化即存在大的共同方向偏置。
- 初始高斯标准差由 0.2 改为 0.4，最低标准差为 0.05；熵正则使用 tanh 后动作分布的蒙特卡洛熵估计，能对均值饱和提供反馈。
- 默认学习率 `1e-4`、每回合 3 个 epoch、时间步 batch 256、clip 0.1、KL 目标 0.01。`--lr`、`--update-epochs`、`--target-kl` 可在新训练时配置。
- 更新前，在整段轨迹上重新计算动作概率并核对采样记录；误差超过 `2e-3` 明确报错。
- 每次候选 actor 更新后，用所有轨迹状态计算旧/新高斯的解析 KL。tanh 是双射，因此这是安全过滤前策略的 KL。超过 `1.5 * target_kl` 则同时恢复参数和 Adam 状态，减半步长重试，最多 5 次；全部失败则保留上一次有效策略。KL 达到目标即停止本轮 actor 更新，critic 仍完成更新。
- critic 输入使用 LayerNorm、损失使用 Huber；价值目标保留原奖励单位，不做每回合独立的 return 标准化。停用无人机不参与 actor loss。
- PPO 始终记录和评分原始采样动作，不能用安全层修正后的动作替换 PPO 的行为动作。

这是对本任务的稳定化实现，不声称等同于官方 MAPPO 的所有实现细节或已经证明收敛。

### 奖励

距离势函数使用 `p_i = exp(-max(d_i - capture_radius, 0) / 12)`，停用无人机取零。团队势函数由 `2 * mean(p_i)`、`4 * max(p_i)` 和 `2 * enclosure` 组成。最近机项对应当前任意一架捕获即可成功的目标，平均接近项和高度敏感的包围项提供团队协作反馈。

每个势函数项的奖励为 `gamma * Phi(next) - Phi(current)`，`gamma=0.995` 必须与 PPO 一致。成功、超时及其他真正终止状态的下一势函数均为零。这样避免终点漏处理或来回移动造成势函数收益累积。去掉原本每步发放的 proximity 正奖励。对于完整终止轨迹，折扣势函数总和恒等于负初始势函数；测试覆盖成功和超时两种终点。此性质仅适用于势函数项，新增的安全代价本身仍定义了任务偏好。

保留捕获 +100、超时 -50、每步 -0.05；建筑近距离惩罚权重由 0.2 改为 0.03，并增加边界、球形障碍物和队友的连续有界距离代价。安全修正幅度惩罚权重为 0.05，按活动无人机平均且单机幅度截断到 1；只有紧急处理额外扣 0.2，按活动无人机比例计算。避免固定的大额拒绝代价长期淹没追击信号。

## 新指标的解释

- `controller_feasible_rate`：活动无人机未触发紧急处理的比例，包含已经过安全修正的可执行动作。与旧版“原始命令未被拒绝比例”不再同义。
- `safety_correction_rate`：安全层改变速度参考或紧急处理的比例。可行率高不代表策略已学会避障，必须同时看此项。
- `safety_correction_magnitude`、`safety_reason_counts`、`emergency_stop_rate`：修正大小、原因及紧急处理比例。
- `desired_velocities` 和 `executed_velocity_references`：策略提出的参考与安全层实际交给飞控的速度参考；后者也不是带惯性的实际速度。
- `exact_policy_kl`、`pre_update_logprob_error`、`ppo_backtracks`、`ppo_accepted_steps`、`policy_std`、`action_mean_saturation`：PPO 更新是否正常。
- `entropy` 现在是 tanh 后分布的估计熵，不能直接与旧版未变换高斯熵比较。

这些指标写入训练历史、奖励诊断及轨迹；评估报告也记录修正和紧急处理率。不可仅凭可行率接近 1 或回报比旧奖励下更高宣称训练成功。

## 远程运行

覆盖补丁包中的 `repro/` 文件，使用新输出目录，从头训练。先运行回归测试（需要 numpy、torch 和项目原有依赖）：

```bash
python -m unittest discover -s repro -p 'test_pursuit*.py'
python -m unittest discover -s repro -p 'test_quadrotor_pursuit.py'
python -m unittest discover -s repro -p 'test_velocity_lidar.py'
```

先跑 300 回合检查学习趋势，再决定长程训练预算：

```bash
python repro/train_pursuit_with_demos.py train --bc-updates 0 --demo-weight 0 --episodes 300 --env-config repro/configs/pursuit_v2/speed_ratio_1_1.json --out outputs/pursuit_v3_ratio11_pure_mappo_seed11 --seed 11 --validation-interval 50 --validation-episodes 10 --device auto
```

同条件独立评估最终策略与基准：

```bash
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/pursuit_v3_ratio11_pure_mappo_seed11/final.pt --out outputs/pursuit_v3_ratio11_eval_seed11 --eval-seed 3000000 --eval-episodes 50 --methods mappo apf frpn mpc --device auto
```

不能继续将旧版 50 回合基准统计作为新版的公平对照。新执行和奖励改变了 MDP，需要重新评估同种子基准。v3 checkpoint 可正常续训；续训沿用 checkpoint 的配置。

## 参考依据

1. Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022：[论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html)。[官方仓库](https://github.com/marlbenchmark/on-policy)、[动作分布实现](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/distributions.py)、[MLP 归一化](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/utils/mlp.py)、[MAPPO 更新](https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py)。本次核对的是公开主分支在 2026-09-08 提供的内容，不是一个固定 commit 的完整复现。
2. Dalal et al., *Safe Exploration in Continuous Action Spaces*：[作者论文](https://arxiv.org/abs/1801.08757)。参考“尽量少地修正动作”的安全层思想；本项目使用几何和飞控模型，没有复现其学习约束模型或宣称相同理论性质。
3. Ng, Harada, Russell, *Policy Invariance Under Reward Transformations*, ICML 1999：[作者托管论文](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)。参考折扣势函数差分形式，并正确处理有限期限终点。
4. Engstrom et al., *Implementation Matters in Deep Policy Gradients*, ICLR 2020：[作者论文](https://arxiv.org/abs/2005.12729)。支持将实现细节作为 PPO 排查重点；本项目的全轨迹解析 KL 回退属于工程设计，不冒称该论文的原算法。

本次验证结果见同目录 `PURSUIT_V3_VALIDATION.json`。短程试验用于检查实现、更新稳定性和执行异常，不代替完整多种子训练与独立成功率评估。
