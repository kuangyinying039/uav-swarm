# 动态弱通信无人机协同搜索与跟踪项目说明书

本文档按当前代码实现编写，重点说明每个文件承担的功能、主环境运行机制、训练入口、评估入口和关键指标定义。当前项目已经从“论文复现 demo”调整为“动态弱通信条件下的多无人机协同控制研究环境”。

## 1. 当前项目目标

本项目研究的问题是：

> 在存在通信丢失、干扰机、动态障碍和动态目标的条件下，多架无人机如何通过协同控制尽可能降低弱通信带来的影响，并完成搜索与跟踪任务。

因此当前主线不再是复现某一篇论文的完整实验协议，而是构造一个可训练、可消融、可分析的多智能体环境。

当前重点包括：

- 多 UAV 协同搜索。
- 动态弱通信和随机断联。
- 干扰机对通信质量的影响。
- 动态目标发现、跟踪和完成。
- 动态障碍和 UAV 间安全约束。
- DWA/ORCA-like 安全层。
- MAPPO/QMIX 多智能体强化学习训练。
- GAT 图注意力编码器用于弱通信邻居信息建模。

## 2. 代码文件总览

核心代码位于 `repro/` 目录。

| 文件 | 当前定位 |
|---|---|
| `tpm_search_base_env.py` | TPM 目标概率图和基础搜索环境 |
| `cooperative_search_env.py` | 当前主环境：弱通信协同搜索与跟踪 |
| `marl_trainers.py` | MAPPO/QMIX/GAT/MLP 训练器实现 |
| `train_cooperative_marl.py` | 单随机种子训练入口 |
| `train_multi_seed_cooperative_marl.py` | 多随机种子训练入口 |
| `evaluate_ablation_cases.py` | 环境机制消融实验 |
| `evaluate_environment_sensitivity.py` | 环境参数敏感性实验 |
| `evaluate_marl_sensitivity.py` | MARL 超参数敏感性实验 |
| `benchmark_avoidance_layer.py` | DWA/ORCA-like 避障层基准测试 |
| `run_reference_heuristic_demos.py` | 参考机制启发式 demo |
| `graph_tracking_reference_env.py` | 图跟踪参考环境 |
| `run_legacy_reproduction_wrapper.py` | 兼容旧入口的包装脚本 |
| `PROJECT_RUN_MANUAL.zh-CN.md` | 当前说明书 |

## 3. 每个文件的详细作用

### 3.1 `tpm_search_base_env.py`

这个文件提供项目的基础搜索底座。

主要内容：

- `WangConfig`：基础环境配置。
- `WangSearchInterferenceEnv`：基础 TPM 搜索环境。
- `prob_to_log_odds()` / `log_odds_to_prob()`：概率和 log-odds 表示之间的转换。
- `uncertainty_search_policy()`：基于 TPM 不确定性的启发式搜索策略。
- `greedy_spectrum_policy()`：保留的基础频谱启发式策略，当前主环境不再依赖它做 UAV 信道选择。
- `run_wang_heuristic_demo()`：基础 TPM 搜索机制 demo。

它实现的核心功能：

- 在栅格地图中采样目标。
- 为每个格子维护目标概率图 TPM。
- 用传感器检测结果更新局部 TPM。
- 计算搜索覆盖率 `coverage`。
- 提供基础动作 mask，包括边界和转向约束。
- 保留扫频干扰机相关状态，用于给主环境提供干扰强度。

当前 TPM 初始化默认是未知先验：

```python
tpm_prior_base = 0.5
tpm_target_prior_enabled = False
tpm_target_prior_strength = 0.72
tpm_target_prior_sigma = 0.75
```

也就是说，默认全图初始概率为 0.5，不会把真实目标位置附近设为高概率。这样初始 `coverage` 为 0，不会因为 `theta0=0.05` 造成覆盖率虚高。

如果需要做“带先验信息”的实验，可以在代码中开启：

```python
cfg = WeakCommConfig(tpm_target_prior_enabled=True)
```

也可以在命令行中开启：

```bash
python ./repro/evaluate_ablation_cases.py --enable-target-prior
```

开启后，环境会在初始目标位置附近生成二维高斯 bump。高斯强度由 `tpm_target_prior_strength` 控制，宽度由 `tpm_target_prior_sigma` 控制。默认强度为 0.72，sigma 为 0.75。

### 3.2 `cooperative_search_env.py`

这是当前项目最重要的文件，是主环境。

主要内容：

- `WeakCommConfig`：主环境配置，继承 `WangConfig`。
- `WeakCommBeliefGraphEnv`：动态弱通信协同搜索与跟踪环境。
- `dwa_orca_search_policy()`：用于评估和 demo 的启发式控制策略。
- `run_weak_comm_demo()`：主环境启发式 demo。

它在基础 TPM 搜索之上增加：

- 动态目标。
- 动态障碍。
- UAV 间弱通信图。
- 局部信念图 `local_q`。
- 全局融合信念图 `q_map`。
- 机会式邻居信念融合。
- DWA/ORCA-like 安全动作过滤。
- 动态时间步长。
- 搜索、发现、跟踪、完成、避障、通信相关指标。

当前通信实现的重点不是信道选择，而是通信丢失建模：

- UAV 之间先根据距离判断是否在 `comm_radius` 内。
- 距离越远，丢包概率越大。
- 干扰机越强，丢包概率越大。
- 每一步还存在随机断联。
- 通信成功时，UAV 可以融合邻居的局部信念图。

通信图由 `_comm_graph()` 构造。若 `last_graph[i, j] > 0`，表示 UAV `i` 当前可以接收 UAV `j` 的信息。

当前 `step_joint()` 不再要求策略输出信道动作。它只接收搜索动作：

```python
result = env.step_joint(search_actions)
```

返回值仍保留一些 `spectrum_*` 字段，是为了兼容旧输出表格；但这些字段现在表示通信质量统计，不代表 UAV 正在学习选择信道。

### 3.3 `marl_trainers.py`

这个文件实现强化学习训练器。

主要内容：

- `TrainConfig`：训练参数。
- `MLP`：普通多层感知机。
- `GraphAttentionBlock`：图注意力基本块。
- `GraphAttentionEncoder`：多层 GAT 编码器。
- `GraphActor`：MAPPO actor。
- `GraphQNetwork`：QMIX 中每个智能体的 Q 网络。
- `MAPPOTrainer`：集中训练、分散执行的 MAPPO 训练器。
- `QMIXTrainer`：基于 mixing network 的 QMIX 训练器。
- `MixingNetwork`：QMIX 总 Q 值混合网络。
- `capture_env_frame()`：记录 episode 场景轨迹，用于输出 SVG。

当前训练器的步数逻辑已经改为：

```python
for _ in range(self.cfg.max_steps or env.cfg.search_steps):
```

`TrainConfig.max_steps` 默认是 0，因此如果用户在环境里设置：

```python
search_steps = 300
```

训练就会使用环境的 `search_steps`，不会被训练器里的硬编码默认值覆盖。

注意：文件中仍保留 `DualMAPPOTrainer` 类作为历史扩展代码，但主训练入口已经不再暴露 `dual-mappo`，因为当前研究重点不是频谱信道动作学习。

### 3.4 `train_cooperative_marl.py`

这是单随机种子的主训练入口。

支持：

- `--algo mappo`
- `--algo qmix`
- `--episodes`
- `--search-steps`
- `--n-uavs`
- `--n-targets`
- `--n-obstacles`
- `--grid-size`
- `--comm-radius`
- `--target-speed`
- `--obstacle-speed`
- `--uav-speed`
- `--decision-dt`
- `--no-gat`

输出：

- JSON 训练历史。
- CSV 训练曲线数据。
- SVG 训练曲线。
- 最后若干 episode 的场景轨迹 SVG。

示例：

```bash
python ./repro/train_cooperative_marl.py --algo mappo --episodes 500 --search-steps 300
python ./repro/train_cooperative_marl.py --algo qmix --episodes 500 --search-steps 300
```

如果当前 Python 环境没有安装 PyTorch，会报：

```text
PyTorch is not installed. Install torch to run MAPPO/QMIX training.
```

### 3.5 `train_multi_seed_cooperative_marl.py`

这是多随机种子训练入口，用于生成更稳定的均值和方差结果。

它会对每个 seed 分别训练，并聚合：

- reward 均值/标准差。
- coverage 均值/标准差。
- discovery/tracking/completion 均值/标准差。
- collisions 均值/标准差。
- collision_rate 均值/标准差。

示例：

```bash
python ./repro/train_multi_seed_cooperative_marl.py --algo mappo --episodes 500 --seeds 11,23,41,59,83
python ./repro/train_multi_seed_cooperative_marl.py --algo qmix --episodes 500 --seeds 11,23,41,59,83
```

适合用于论文中的稳定实验结果，而不是单次随机训练曲线。

### 3.6 `evaluate_ablation_cases.py`

这个文件做主环境机制消融。

它使用启发式 `dwa_orca_search_policy()`，不是 MARL 训练结果。用途是验证环境机制本身是否生效。

典型 case 包括：

- 完整模型。
- 全通信。
- 无通信。
- 无动态目标。
- 静态障碍。
- 无避障。
- DWA only。
- DWA + ORCA-like。

输出：

- 每个 case 的 summary。
- 每步历史 CSV。
- 通信历史 CSV。
- 覆盖率、完成率、碰撞等 SVG。
- 轨迹图和目标/障碍运动图。
- 分析报告。

示例：

```bash
python ./repro/evaluate_ablation_cases.py --seeds 11,23,41
```

### 3.7 `evaluate_environment_sensitivity.py`

这个文件做环境参数敏感性实验。

它调用 `evaluate_ablation_cases.py` 中的 `run_case()`，对不同环境参数取值进行对比。

典型敏感性参数包括：

- 通信半径。
- 障碍数量。
- 障碍速度。
- UAV 数量。
- 目标数量。

输出：

- 敏感性 summary CSV。
- 敏感性 summary JSON。
- 指标变化 SVG。

示例：

```bash
python ./repro/evaluate_environment_sensitivity.py --seeds 11,23,41
python ./repro/evaluate_environment_sensitivity.py --quick
```

### 3.8 `evaluate_marl_sensitivity.py`

这个文件做 MARL 训练超参数敏感性实验。

它会对 MAPPO 或 QMIX 在不同训练参数下运行短训练或完整训练，并聚合末尾窗口指标。

可用于比较：

- GAT head 数量。
- GAT 层数。
- dropout。
- hidden_dim。
- 不同算法。

示例：

```bash
python ./repro/evaluate_marl_sensitivity.py --algo mappo --episodes 200 --seeds 11,23
python ./repro/evaluate_marl_sensitivity.py --algo both --quick
```

需要 PyTorch。

### 3.9 `benchmark_avoidance_layer.py`

这个文件专门评估避障层。

它比较：

- 不启用避障。
- DWA 风格障碍过滤。
- DWA + ORCA-like UAV 间互避。

统计指标包括：

- collisions。
- collision_rate。
- min_clearance。
- mean_clearance。
- path_length。
- completion_rate。
- 通信质量指标。

示例：

```bash
python ./repro/benchmark_avoidance_layer.py --seeds 11,23,41,59,83
```

### 3.10 `run_reference_heuristic_demos.py`

这个文件运行参考机制 demo。

它会分别调用：

- `tpm_search_base_env.py` 中的基础 TPM 搜索 demo。
- `graph_tracking_reference_env.py` 中的图跟踪参考 demo。
- `cooperative_search_env.py` 中的主环境启发式 demo。

用途：

- 快速检查环境是否能运行。
- 生成基础图示。
- 展示各模块机制。

它不应该被当作完整论文复现实验，也不代表 MARL 训练性能。

示例：

```bash
python ./repro/run_reference_heuristic_demos.py
```

### 3.11 `graph_tracking_reference_env.py`

这个文件是图跟踪机制参考环境。

它包含：

- `ZhaoConfig`。
- `ZhaoGraphTrackingEnv`。
- `graph_heuristic_policy()`。
- `run_zhao_heuristic_demo()`。

它的作用是保留动态图观测、目标吸引、障碍排斥、智能体间安全距离等参考机制。当前主环境没有直接继承它，而是把相关思想融合进 `cooperative_search_env.py` 和 `marl_trainers.py`。

### 3.12 `run_legacy_reproduction_wrapper.py`

这是兼容旧入口的轻量包装脚本。

它内部调用：

```python
from run_reference_heuristic_demos import main
```

新实验不建议使用它。保留它只是为了避免旧命令完全失效。

## 4. 主环境核心机制

### 4.1 状态和观测

每架 UAV 的观测来自 `agent_observation_vectors()`，由两部分组成：

- `graph_features()`：位置、航向、邻居信息、障碍信息、目标信息、TPM 不确定性、任务阶段、动态时间步等。
- `action_mask()`：当前可选动作 mask。

集中 critic 或 QMIX mixing network 使用 `state_vector()`，它拼接所有 UAV 的观测以及全局统计量。

### 4.2 TPM 目标概率图

TPM 表示每个格子存在目标的概率。

当前默认先验：

```python
tpm_prior_base = 0.5
```

表示全图未知。

检测更新使用 log-odds：

```python
hit_delta = log(pf / pd)
miss_delta = log((1 - pf) / (1 - pd))
```

每个 UAV 维护自己的 `local_q[i]`。通信成功时，邻居的局部信念可以被融合。融合后的共享信念图为 `q_map`。

覆盖率定义：

```python
p = log_odds_to_prob(q_map)
decided = (p <= theta0) | (p >= theta1)
coverage = mean(decided)
```

默认：

```python
theta0 = 0.05
theta1 = 0.95
```

因此只有当某个格子的概率足够接近“确定无目标”或“确定有目标”时，才算覆盖。初始概率 0.5 不会被算作已覆盖。

### 4.3 动态弱通信

当前通信模块不要求策略选择信道。

通信质量由以下因素决定：

- UAV 间距离。
- `comm_radius`。
- 基础随机丢包 `packet_loss_base`。
- 干扰机造成的丢包增强 `packet_loss_jammed`。
- 距离造成的丢包增强 `packet_loss_distance`。
- 每一步随机采样。

边 `i <- j` 表示 UAV `i` 能接收 UAV `j` 的信息。若通信成功，权重近似为：

```python
weight = exp(-graph_beta * distance) / (1 + belief_age[j])
```

距离越远、信息越旧，权重越低。

### 4.4 动态目标

目标有连续位置和速度：

- 未完成目标会移动。
- 边界处反弹。
- 被发现后进入跟踪阶段。
- 连续跟踪达到 `target_completion_steps` 后，该目标完成。
- 完成目标速度置零。

任务阶段由已发现目标比例和当前跟踪状态决定：

- `search`。
- `tracking`。

### 4.5 动态障碍

障碍物有连续位置和速度：

- 默认启用动态障碍。
- 障碍物在边界反弹。
- 障碍物与目标过近时会反向，避免目标和障碍长期重叠。

障碍数量和速度由：

```python
n_obstacles
obstacle_speed
```

控制。

### 4.6 DWA/ORCA-like 安全层

`action_mask(i)` 会对每架 UAV 的候选动作进行过滤。

基础动作 mask 先处理：

- 越界。
- 最大转向约束。
- 是否允许悬停。

然后安全层处理：

- 当前障碍物安全距离。
- 预测障碍物安全距离。
- UAV 间安全距离。

障碍过滤：

```python
static_clearance = obstacle_radius + 0.75 * safe_radius
predicted_clearance = obstacle_radius + safe_radius
```

UAV 间 ORCA-like 过滤：

```python
distance_to_other_uav < safe_radius
```

如果所有移动动作都被过滤，环境会放回一个预测 clearance 最大的动作，以避免 UAV 完全卡死。这意味着极端拥挤场景下仍可能出现真实碰撞，因此真实碰撞惩罚仍然需要保留。

### 4.7 碰撞和安全指标

当前已经区分真实碰撞和安全层拦截。

返回指标：

| 指标 | 含义 |
|---|---|
| `collisions` | 真实碰撞总数 |
| `obstacle_conflicts` | UAV 进入障碍半径 |
| `uav_collisions` | UAV-UAV 距离小于等于真实碰撞半径 |
| `pair_conflicts` | UAV-UAV 距离小于安全半径，但不一定真实碰撞 |
| `invalid_actions` | 策略选择了被安全层拦截的危险动作 |

当前定义：

```python
collisions = obstacle_conflicts + uav_collisions
```

其中：

```python
obstacle_conflict: distance(UAV, obstacle) <= obstacle_radius
uav_collision: distance(UAV_i, UAV_j) <= uav_collision_radius
pair_conflict: distance(UAV_i, UAV_j) < safe_radius
```

默认：

```python
obstacle_radius = 1.25
uav_collision_radius = 0.8
safe_radius = 1.6
```

### 4.8 奖励函数

主环境奖励包含：

- TPM 不确定性下降。
- 新目标发现奖励。
- 当前跟踪奖励。
- 持续跟踪奖励。
- 连续跟踪进度奖励。
- 目标完成奖励。
- 编队/护航式跟踪奖励。
- 朝目标靠近奖励。
- 目标丢失惩罚。
- 被安全层拦截动作惩罚。
- UAV 间安全距离冲突惩罚。
- 真实障碍碰撞惩罚。
- 原地不动平滑惩罚。

当前安全相关惩罚已经拆开：

```python
- invalid_action_penalty * invalid_action_count
- pair_penalty * pair_conflicts
- obstacle_penalty * obstacle_conflicts
```

默认：

```python
invalid_action_penalty = 1.0
pair_penalty = 4.0
obstacle_penalty = 8.0
```

注意：`uav_collisions` 目前计入 `collisions` 指标，但奖励惩罚通过 `pair_penalty * pair_conflicts` 覆盖 UAV 间近距离风险。若后续希望真实 UAV-UAV 碰撞比一般安全距离冲突惩罚更重，可以新增 `uav_collision_penalty`。

## 5. 推荐运行方式

### 5.1 快速检查参考 demo

```bash
python ./repro/run_reference_heuristic_demos.py
```

该命令不需要 PyTorch。

### 5.2 运行主环境消融实验

```bash
python ./repro/evaluate_ablation_cases.py --seeds 11,23,41
```

该命令使用启发式控制策略，用于分析环境机制。

### 5.3 训练 MAPPO

```bash
python ./repro/train_cooperative_marl.py --algo mappo --episodes 500 --search-steps 300
```

### 5.4 训练 QMIX

```bash
python ./repro/train_cooperative_marl.py --algo qmix --episodes 500 --search-steps 300
```

### 5.5 多随机种子训练

```bash
python ./repro/train_multi_seed_cooperative_marl.py --algo mappo --episodes 500 --seeds 11,23,41,59,83
python ./repro/train_multi_seed_cooperative_marl.py --algo qmix --episodes 500 --seeds 11,23,41,59,83
```

### 5.6 避障层 benchmark

```bash
python ./repro/benchmark_avoidance_layer.py --seeds 11,23,41,59,83
```

### 5.7 环境参数敏感性实验

```bash
python ./repro/evaluate_environment_sensitivity.py --seeds 11,23,41
```

### 5.8 MARL 超参数敏感性实验

```bash
python ./repro/evaluate_marl_sensitivity.py --algo both --quick
```

该命令需要 PyTorch。

## 6. 常用参数

以下参数可通过训练脚本或评估脚本传入：

| 参数 | 含义 |
|---|---|
| `--n-uavs` | UAV 数量 |
| `--n-targets` | 目标数量 |
| `--n-obstacles` | 障碍数量 |
| `--grid-size` | 栅格地图尺寸 |
| `--search-steps` | episode 最大步数 |
| `--target-speed` | 动态目标速度 |
| `--obstacle-speed` | 动态障碍速度 |
| `--uav-speed` | UAV 速度 |
| `--decision-dt` | 决策时间步 |
| `--comm-radius` | 通信半径 |
| `--tpm-prior-base` | TPM 全图基础先验概率，默认 0.5 |
| `--enable-target-prior` | 开启目标附近高斯先验 |
| `--tpm-target-prior-strength` | 目标高斯先验强度，默认 0.72 |
| `--tpm-target-prior-sigma` | 目标高斯先验宽度，默认 0.75 |
| `--allow-hover` | 允许悬停动作 |
| `--no-normalize-diagonal-speed` | 不归一化对角动作速度 |
| `--disable-dynamic-dt` | 关闭动态时间步 |
| `--no-gat` | 训练时不使用 GAT 编码器 |

`search_steps` 的权威来源是环境配置。训练器默认 `max_steps=0`，会自动使用：

```python
env.cfg.search_steps
```

因此修改环境或命令行中的 `search_steps` 会真实影响训练 episode 长度。

## 7. 输出文件

默认输出位于 `outputs/`。

常见输出包括：

- `*.json`：训练历史或实验结果。
- `*.csv`：可复分析表格。
- `*.svg`：训练曲线、指标图、轨迹图。
- `*_last3_scenes/`：训练最后几个 episode 的场景可视化。

建议论文分析时优先使用：

- 多随机种子均值和标准差。
- `coverage`。
- `target_discovery_rate`。
- `target_tracking_rate`。
- `target_completion_rate`。
- `collisions`。
- `invalid_actions`。
- `pair_conflicts`。
- `avg_graph_density` 或通信成功率。

不要只看 `total_reward`，因为 reward 是多项加权组合，单独解释容易误导。

## 8. 当前实现边界

当前版本已经完成：

- 弱通信建模从“信道选择”转为“距离 + 干扰 + 随机丢包”。
- `search_steps` 不再由训练器硬编码覆盖。
- `collisions` 只表示真实碰撞。
- `invalid_actions` 与真实碰撞分开统计。
- UAV-UAV 真实碰撞计入 `collisions`。
- TPM 初始先验改为全图未知概率 0.5。
- 文件名按功能重命名。

仍可继续改进：

- 给 UAV-UAV 真实碰撞单独加入更重的 `uav_collision_penalty`。
- 将通信质量指标命名从 `spectrum_*` 进一步改为 `communication_*`，目前为了兼容旧 CSV/SVG 输出暂时保留旧字段名。
- 清理或迁移历史 reference demo，使主项目更轻。
- 安装 PyTorch 后重新跑 MAPPO/QMIX 的完整训练验证。
- 将说明书进一步拆分为“运行手册”和“算法说明”两个文档。

## 9. 推荐论文实验顺序

建议按以下顺序推进：

1. 运行 `run_reference_heuristic_demos.py`，确认环境机制正常。
2. 运行 `evaluate_ablation_cases.py`，验证弱通信、无通信、全通信、避障等机制差异。
3. 运行 `benchmark_avoidance_layer.py`，单独分析 DWA/ORCA-like 安全层。
4. 安装 PyTorch 后运行 `train_cooperative_marl.py`，做短训练 sanity check。
5. 运行 `train_multi_seed_cooperative_marl.py`，得到多 seed 均值和方差。
6. 运行 `evaluate_environment_sensitivity.py`，分析通信半径、障碍数量等环境参数。
7. 运行 `evaluate_marl_sensitivity.py`，分析 GAT 和训练超参数。