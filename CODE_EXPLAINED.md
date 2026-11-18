# Radar5 多机追逃：代码详解（场景 · 约束 · 问题 · 思路 · 算法）

| 字段 | 值 |
|------|----|
| 文档版本 | 2026-11-18 |
| 仓库 | [`kuangyinying039/uav-swarm`](https://github.com/kuangyinying039/uav-swarm) |
| 正式仿真标签 | `isaac_dual_physx_closed` |
| 当前正式捕获球 | `target_diameter=0.5` → `capture_radius=1.0`（见 `configs/pursuit_v2/*_physx_closed.json`） |
| 历史对照 | Hybrid `isaac_dual_postlc`；旧正式表曾用 d=0.75，**勿与现行 d=0.5 混排** |

配套文档：[`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md)（配置与评测表）· [`SIM2REAL.md`](SIM2REAL.md)（PX4 契约）· [`docs/IMPROVEMENT_ROADMAP.md`](docs/IMPROVEMENT_ROADMAP.md)（改进排队）

---

## 1. 仿真场景设定（先读这一节）

### 1.1 任务一句话

在 **25 m × 25 m** 城市网格、**5 栋建筑遮挡** 下，**3 架四旋翼追捕机** 协同包围 **1 架同型号逃逸机**；成功条件不是「单机贴靠」，而是 **至少 2 架活跃追捕机同时进入捕获球**（双机距离捕获）。

### 1.2 场景几何与角色

| 项目 | 设定 | 工程含义 |
|------|------|----------|
| 网格 | `grid_size=25` | 与真机/SITL 可部署的局部任务区同量级 |
| 建筑 | 固定 5 栋，尺寸/高度随难度阶梯加压 | 强制 LOS 遮挡与绕障，不是空旷点质量追逃 |
| 追捕机 | 3× Hummingbird 级四旋翼（PhysX 关节体） | 与逃逸机同平台，避免「仿真专用超机动追捕」 |
| 逃逸机 | 1× 同型号；策略 `evader_policy=occlusion` | 主动躲进建筑阴影，而不是随机游走 |
| 初始交接 | 三角队形交接带约 8–10 m | 模拟雷达/航迹交接后的近距追击段，而非远距搜索 |
| 高度带 | 约 1–12 m，初始约 5 m | 贴合低空城市飞行包线 |

### 1.3 难度阶梯（五级）

公共语义：`building_count=5`，`evader_policy=occlusion`，奖励 v5，策略观测 v2。

加压轴（随 `nominal → medium → hard → severe → extreme` 升高）：

1. 逃逸水平/垂直速度；
2. 建筑尺寸与高度；
3. 交接距离；
4. LiDAR 量程与探测概率。

**训练默认只在 hard**（或 AEG 在 medium/hard/severe 间自适应）；跨级评测同时检验难度与分布偏移。

### 1.4 两套物理域（必须分开报）

| 标签 | 追捕机 | 逃逸机 | 用途 |
|------|--------|--------|------|
| `isaac_dual_postlc`（Hybrid） | PhysX + Lee 跟踪 | **运动学/可传送位姿** | 历史对照；算力更轻 |
| `isaac_dual_physx_closed` | PhysX + Lee | **同型号 PhysX，禁止 Radar5 传送** | **正式仿真**；与真机动力学更对齐 |

正式路径下：PhysX 损坏应截断 episode，不得静默退回运动学；GUI 恢复标记 `physics_recovered` 的局不进正式统计。

---

## 2. 现实工程约束（本仓库刻意「变难」的地方）

下面这些不是为了刷分，而是为了让仿真结论能接到 PX4 / MAVROS / 真机。

### 2.1 控制与接口约束

| 约束 | 实现 | 为什么重要 |
|------|------|------------|
| **5 Hz 决策** | `decision_dt=0.2` | 与 offboard 设点频率一致，禁止仿真里用 50–100 Hz「作弊」 |
| **动作语义** | `(n,4)∈[-1,1]` → 世界系 \(v_x,v_y,v_z\) + 偏航角速率 | 与 MAVROS `setpoint_raw/local`（速度+偏航率）同构 |
| **速度盒** | hard 约水平 1.48 m/s、垂直 0.8 m/s 量级 | 贴近小型多旋翼安全包线，而非竞速机 |
| **内环** | Lee 速度跟踪 + 加速度/倾角限制 | 高层策略看不到力矩级自由度，避免「仿真动作真机做不到」 |

### 2.2 感知与信息约束

| 约束 | 实现 | 为什么重要 |
|------|------|------------|
| **禁止目标真值进策略** | 只允许 KF–CI 融合航迹 | 真机没有 simulator GT |
| **MID-360 几何 LiDAR** | 水平 360°、垂直约 −7°~52°、探测概率 &lt; 1 | 复现 Livox 类固态雷达 FOV 与漏检 |
| **建筑 LOS 遮挡** | 几何射线 / 可选 native LiDAR | 遮挡是主失败模式之一 |
| **通信邻接** | 同伴状态与共享航迹受图约束 | 不是全连接上帝视角 |

### 2.3 任务成功定义约束

实现：[`capture_geometry.py`](capture_geometry.py)

1. ≥ `capture_required_uavs`（默认 **2**）架**活跃**机进入 `capture_radius`；
2. 现行双捕获阶梯多用 `capture_mode=single_distance`（无角度门）；可选 `angular_enclose`；
3. **最近距离 `min_distance` 仅诊断**，永不单独判成功——避免「一架贴脸、两架在外」被算赢。

当前正式配置：`target_diameter=0.5` → `capture_radius=1.0`。相对旧 d=0.75 表，捕获球更小、更接近「近距合围」工程要求；**绝对捕获率不可跨口径对比**。

### 2.4 安全与部署约束

- 机间分离、建筑净空、高度/地理围栏：仿真奖励惩罚 + 真机 [`real_flight_safety.py`](real_flight_safety.py)；
- 首飞推荐 `ResidualPolicy = clip(expert + gain·rl)`，而不是裸 RL；
- 观测栈 [`real_flight_obs.py`](real_flight_obs.py) 只吃里程计 + 融合航迹。

---

## 3. 我们遇到的核心问题

### 问题 A：Hybrid 高捕获率无法直接代表真机

Hybrid 下逃逸机可运动学传送，追捕机却走 PhysX。结果是：

- FRPN 等「直接响应」经典法在 Hybrid hard 上极强（约 80%）；
- 一旦逃逸机也进入 PhysX 闭环，相对优势重排，**MPC 成为 hard 最强经典**；
- 若把 Hybrid 数字和 PhysX-closed / SITL 写进同一张表，会产生虚假结论。

**对策**：双域分表；正式结论只认 `isaac_dual_physx_closed`。

### 问题 B：纯 RL 在 PhysX-closed 上「顶不过教师」（O2O / 超越教师失败）

在 MPC 教师 + 全物理闭环 + 较小捕获球后，常见现象：

- DAgger / BC 能接近教师；
- 继续 MATD3-A6 / MAPPO 在线 RL 后，hard 捕获率仍低于经典 MPC；
- 表现为策略漂移、探索破坏已学会的合围时序（文献里常称 offline-to-online 坍塌）。

**对策**（多线并行，见第 4 节）：

1. 加强 BC / actor-anchor / 专家回放比例；
2. 残差学习（RL 只学相对教师的修正）；
3. 课程 AEG（难度带自适应）；
4. EPN 预测特征缓解部分可观测；
5. 更强教师 NMPC（缩短「教师天花板」与任务几何的差距）。

### 问题 C：部分可观测 + 遮挡导致「追影」

遮挡下航迹陈旧，策略若只看当前相对位置，容易：

- 冲进建筑死角；
- 三机挤同一侧，无法形成双机合围；
- 对逃逸机动的相位滞后。

**对策**：共享 KF–CI 航迹；图观测区分 self / track / peer / building；EPN 用历史航迹预测短时相对未来（训练用特权未来做 MSE，推理仍因果）。

### 问题 D：双机捕获的时序与角色分配

单机 PN/APF 天然优化「我离目标最近」，但任务要 **两架同时进入球**。没有角色分化时会出现：

- 全员追最近点 → 堆叠与碰撞；
- 一架进入、另外两架过远 → 永不成功。

**对策**：MPC/NMPC 显式分配 interceptor（最近 k 架）与 flanker；奖励 v5 使用「最近 k 架势函数平均」的团队捕获势。

### 问题 E：仿真可视化 / 评测口径污染

- 深色建筑 + 深色机身 → GUI 难诊断；
- physx_closed 下逃逸真机 mesh 被雷达球挡住；
- 混用 d=0.5 / d=0.75、hybrid / closed、残差 / 纯 RL。

**对策**：浅色半透明建筑（仅 GUI）、隐藏逃逸雷达球渲染、文档与目录强制分标签。

---

## 4. 解决思路总览

```text
工程契约冻结（5 Hz · 速度盒 · 航迹观测 · 双机捕获）
        │
        ▼
PhysX-closed 正式域（3+1 全关节体）
        │
        ├─ 经典教师：APF / FRPN / MPC / NMPC
        │     └─ 采集成功（+近失误）示范
        │
        ├─ 模仿：BC → DAgger（held-out 选优）
        │
        ├─ 在线 RL：MATD3-A6 / MAPPO+（HGAT 图策略）
        │     ├─ 可选 Residual（专家 + gain·RL）
        │     ├─ 可选 EPN（逃逸短时预测特征）
        │     └─ 可选 AEG（medium/hard/severe 自适应）
        │
        └─ 五级阶梯评测 + PX4 同构部署栈
```

原则：

1. **先对齐植物与接口，再谈算法分数**；
2. **教师与 RL 共享同一局部航迹接口**，保证可迁移；
3. **一次只改一条主因**（残差 / O2O / 课程 / 奖励），避免混杂实验；
4. **正式表与敏感性对照分目录、分 JSON**。

---

## 5. 算法原理（按模块）

### 5.1 感知与跟踪

[`target_tracker.py`](target_tracker.py) + [`pursuit_lidar.py`](pursuit_lidar.py)

- 几何 MID-360：距离门 + 垂直 FOV + 建筑遮挡 + 伯努利漏检；
- 每机常数速度 KF；多机 **新鲜度感知协方差交叉（CI）** 融合；
- 策略输入是融合均值/不确定性相关特征，不是 GT。

### 5.2 异构图观测（HGAT）

[`pursuit_graph_encoder.py`](pursuit_graph_encoder.py)，当前 `PURSUIT_GRAPH_VERSION = 4`

实体类型：

| 节点 | 维度要点 |
|------|----------|
| self | 自身运动学 / 高度 / 边界等 |
| target track | 8 维航迹特征 + **EPN 未来相对 xyz（5 步×3）** |
| peer | 相对同伴 |
| building | 已知建筑盒特征 |

注意力在实体间融合后进入 actor / critic。旧 checkpoint 若 graph version 不一致会拒绝加载，避免静默维度错位。

### 5.3 奖励 v5

[`pursuit_rewards.py`](pursuit_rewards.py)

- 折扣一致势函数（终端清零，减轻「塑形与折扣不一致」）；
- 个体接近势 + **双机团队捕获势**（对最近 `required_uavs` 架势取平均）；
- 合围进度、可见性进度；
- 有界的边界 / 建筑 / 机间接近惩罚。

设计意图：把「双机同时进入」写进塑形，而不是只奖励单机距离。

### 5.4 经典基线 / 教师

[`pursuit_baselines_3d.py`](pursuit_baselines_3d.py)

| 方法 | 原理要点 | 典型角色 |
|------|----------|----------|
| **APF** | 吸引势 + 斥力 | 弱对照 |
| **FRPN** | 向量比例导引 + 直接响应 | Hybrid 上最强经典 |
| **MPC** | 短时域协同：预测目标 → 最近 k 架拦截点、其余侧翼；候选速度安全筛选 | PhysX-closed 上强教师 |
| **NMPC** | 在同一速度参考植物上做非线性滚动时域优化（一阶滞后 + 加速度盒）；保持 interceptor/flanker 分配；L-BFGS-B / 投影坐标下降；只执行第一拍 | 更贴「带惯性的速度指令」 |

NMPC **刻意不**做成力矩级 SO(3) 最优控制：因为策略/教师真正下发的是速度参考，力矩环在 Lee/PhysX 内；优化错误植物会「仿真里很强、接口上用不上」。

角色标签：`teacher_role=0` 拦截机，`1` 侧翼机，用于演示数据集与分析。

### 5.5 MATD3-A6

[`pursuit/algorithms/matd3.py`](pursuit/algorithms/matd3.py) + 训练入口 [`train_isaac_pursuit.py`](train_isaac_pursuit.py)

要点：

- Twin team critic、延迟策略更新、目标策略平滑（TD3 族）；
- 演示 BC / TD3+BC、critic return clip、**actor anchor**（防止离教师太远）；
- 专家回放比例与 success-only / 近失误混合可配，用于抑制 O2O 坍塌。

### 5.6 MAPPO+ 与 DAgger

[`train_pursuit_with_demos.py`](train_pursuit_with_demos.py) · [`marl_trainers.py`](marl_trainers.py)

流水线：

```text
MPC/NMPC 成功示范采集
  → BC 预训练
  → DAgger 多轮纠偏（agent 动作 ↔ 教师标签）
  → MAPPO 在线（可加 reference KL / warm-start）
  → held-out 验证选优
```

与 MATD3 共用同一 HGAT actor 观测，便于对照「off-policy vs on-policy」在同一任务上的表现。

### 5.7 EPN（Evader Prediction Network）

[`pursuit_epn.py`](pursuit_epn.py)

灵感来自多机追逃中的短时轨迹预测网：

- 输入：每机最近 \(H=10\) 步因果特征（进度、航迹相对位姿/速度、同伴相对位置；缺失填 −1）；
- 网络：LSTM → FC → tanh，输出未来 \(H_f=5\) 步归一化相对 xyz；
- 训练：用特权未来位置构造 MSE（**仅损失可见**）；
- 推理：预测写入 `env.epn_predicted_rel`，拼进 target 节点；**策略仍不读 GT**。

目的：在遮挡与噪声下给策略「提前量」，减轻纯反应式追击。

### 5.8 AEG（离散自适应环境生成）

[`pursuit_aeg.py`](pursuit_aeg.py)

不同于连续任务向量采样，本仓库 AEG **只在已有 ladder 配置**（默认 medium/hard/severe）中调度，以保住「5 栋建筑」等工程合同：

1. 先补齐样本不足的级别；
2. 优先选择滚动捕获率落在 \([R_{\min}, R_{\max}]\)（默认 0.5–0.9）的级别——既不太难学崩，也不太简单无效；
3. 全局捕获率过高则偏难，过低则偏易。

### 5.9 残差策略与真机栈

[`residual_policy.py`](residual_policy.py) · [`real_flight_node.py`](real_flight_node.py) · [`px4_offboard_bridge.py`](px4_offboard_bridge.py)

\[
a = \mathrm{clip}(a_{\mathrm{expert}} + g · a_{\mathrm{rl}})
\]

首飞用小 \(g\)；安全层在陈旧里程计、过近同伴、高度/围栏违规时切 hold。这与「论文主表纯 RL 公平对比」是两条线：主表可关残差，部署默认开残差。

---

## 6. 代码地图（从哪里读起）

| 路径 | 职责 |
|------|------|
| `configs/pursuit_v2/*.json` | 五级难度与 physx_closed 变体 |
| `isaac_pursuit_backend.py` | Isaac/PhysX 后端、GUI 表现、闭环开关 |
| `quadrotor_pursuit_env.py` | 逻辑环境、捕获、奖励步进 |
| `pursuit_baselines_3d.py` | APF/FRPN/MPC/NMPC |
| `pursuit_graph_encoder.py` | 图观测 v4（含 EPN 槽位） |
| `pursuit_epn.py` / `pursuit_aeg.py` | 预测头 / 课程调度 |
| `train_isaac_pursuit.py` | Isaac 训练入口（`--epn` `--aeg` `--physx-closed-loop`） |
| `train_pursuit_with_demos.py` | DAgger/MAPPO trainer |
| `evaluate_isaac_pursuit.py` | 阶梯评测（含 `nmpc`） |
| `capture_geometry.py` | 唯一捕获判据 |
| `results/*_eval_summary.json` | 冻结捕获率摘要 |

录屏/回放辅助（可选）：`record_isaac_pursuit_*.py`、`replay_isaac_pursuit_traj.py`、`export_pursuit_traj_*.py`。

---

## 7. 结果口径（防混淆）

1. **Hybrid** 与 **PhysX-closed** 分表；  
2. **d=0.75 历史正式表** 与 **现行 d=0.5 正式配置** 分表；  
3. **纯 RL** 与 **残差 RL** 分表；  
4. 汇报句式建议：

> 正式仿真为 Isaac PhysX-closed（3 追捕 + 1 逃逸同关节体）、几何 MID-360、双机距离捕获、`target_diameter=0.5`、5 Hz 世界系速度指令；Hybrid 与旧 d=0.75 数字仅作对照。

数值细节与 Wilson CI 见：

- `results/physx_closed_eval_summary.json`（历史 d=0.75 闭环表）
- `results/physx_closed_d05_baseline_eval_summary.json`（d=0.5 基线）
- `results/hybrid_postlc_eval_summary.json`（Hybrid 对照）

---

## 8. 当前代码相对上一版的主要增量（2026-11）

1. **NMPC 基线**：速度参考植物上的非线性滚动优化，评测/残差教师可选 `nmpc`；  
2. **EPN**：图观测升级到 v4，训练/评测可挂逃逸短时预测；  
3. **AEG**：离散难度带课程，接入 `train_isaac_pursuit.py`；  
4. **MPC 角色评测加固**：明确 interceptor/flanker 分配单测；  
5. **GUI**：浅色半透明建筑、physx_closed 下隐藏逃逸雷达球渲染，便于诊断遮挡与合围。

---

## 9. 最小复现命令

```bash
# 克隆
git clone https://github.com/kuangyinying039/uav-swarm.git
cd uav-swarm

# PhysX-closed 主流水线（环境变量见各 queue/run 脚本头）
bash queue_physx_closed_mpc_pipeline.sh

# 带 EPN + AEG 的 MAPPO 示例（需已配置 Isaac）
python train_isaac_pursuit.py --algo mappo --physx-closed-loop --epn --aeg

# 评测 NMPC
python evaluate_isaac_pursuit.py --method nmpc --physx-closed-loop --levels hard

# 真机栈干跑（无 ROS）
python real_flight_node.py --dry-run --duration 4 --track-xyz 0,0,2
```

---

## 10. 一句话总结

我们把「3 打 1 双机合围」放进 **带建筑遮挡、MID-360 感知、5 Hz 速度 offboard、同平台 PhysX 逃逸机、双机捕获球** 的工程约束里；先用 **MPC/NMPC 教师 + DAgger** 建立可部署行为，再用 **HGAT-MATD3/MAPPO**，并可选 **EPN / AEG / 残差** 去补部分可观测、课程与 O2O 稳定性——所有正式数字必须按物理域与捕获球口径分表阅读。
