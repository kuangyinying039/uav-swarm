# 技术说明书：Radar5 多机追逃（Hybrid vs PhysX-closed）

本文档冻结**当前仓库**的配置口径、算法方案与评测表，避免与历史 hybrid / 旧口径混用。

| 字段 | 值 |
|------|----|
| 文档版本 | 2026-10-27 |
| 任务 | 3 追 1 四旋翼 · 双机距离捕获 |
| 正式仿真标签 | `isaac_dual_physx_closed` |
| 历史对照标签 | `isaac_dual_postlc`（**仅基线，禁止与正式表混排**） |

配套契约：[`SIM2REAL.md`](SIM2REAL.md) · 结果摘要：[`results/`](results/) · 真机阶梯：[`REAL_FLIGHT_CHECKLIST.md`](REAL_FLIGHT_CHECKLIST.md)

---

## 1. 两套仿真域：必须分开报

| 项目 | Hybrid `isaac_dual_postlc` | PhysX-closed `isaac_dual_physx_closed` |
|------|---------------------------|----------------------------------------|
| 角色 | **历史 / 对照基线** | **当前正式仿真表** |
| 追捕机 | PhysX 关节体 + Lee 跟踪 | 同左 |
| 逃逸机 | **运动学 / 可传送位姿** | **同型号 PhysX 关节体**（禁止 Radar5 传送） |
| 配置目录 | `configs/pursuit_v2/radar5_dual_{level}.json` | `configs/pursuit_v2/radar5_dual_{level}_physx_closed.json` |
| `target_diameter` | **0.5** → `capture_radius=1.0` | **0.75** → `capture_radius=1.5` |
| 追捕水平速度上限（hard） | 1.4 m/s | **1.48 m/s** |
| 传感（评测默认） | 几何 MID-360 + LOS | 同左（`--no-native-lidar`） |
| 训练域 | hard only | hard only |
| 评测种子 | `7000000` 起，每级 100 局 | 同左 |
| 结果文件 | `results/hybrid_postlc_eval_summary.json` | `results/physx_closed_eval_summary.json` |

**禁止**：把 hybrid 捕获率与 physx_closed / SITL / 真机写进同一张排名表。

**可选对照（非正式）**：`radar5_dual_extreme_physx_closed_d05.json` 仅在 extreme 上把直径改回 0.5，用于敏感性；见结果里的 `extreme_diameter_contrast_d05`。

---

## 2. 任务与接口（两域共享语义）

### 2.1 捕获成功

实现：[`capture_geometry.py`](capture_geometry.py)

1. 至少 `capture_required_uavs=2` 架**活跃**追捕机进入 `capture_radius`
2. 当前双捕获阶梯：`capture_mode=single_distance`（无角度门）
3. `min_distance`（最近机距离）**仅诊断**，永不单独判成功

### 2.2 动作（5 Hz）

- 形状 `(n_agents, 4)`，归一化到 `[-1, 1]`
- 映射：世界系 \(v_x,v_y,v_z\) + 偏航角速率
- `decision_dt=0.2`，与 PX4 offboard 契约一致（见 `SIM2REAL.md`）

### 2.3 观测（策略因果）

策略只允许：

- 本机 / 同伴里程计估计
- **融合航迹**（KF + 新鲜度感知 CI），禁止目标仿真真值
- 已知建筑几何特征、通信邻接

---

## 3. 难度阶梯

五级：`nominal → medium → hard → severe → extreme`。  
公共设定：`building_count=5`，`evader_policy=occlusion`，`execution_reward_version=5`，`policy_observation_version=2`。

加压轴（随级别升高）：目标水平/垂直速度、建筑尺寸与高度、交接距离、LiDAR 量程与探测概率。

训练只在 **hard**；跨级分数同时反映难度与分布偏移。

PhysX-closed 相对 hybrid 的额外差异（全级别）：更大捕获球（直径 0.75）、略高追捕速度上限，以及逃逸机的全物理闭环。

---

## 4. 技术方案（当前算法栈）

### 4.1 感知与跟踪

- 几何 LiDAR（默认）模拟 MID-360 FOV；可选 native LiDAR（GPU 管线常不可用）
- 分布式常数速度 KF；多机航迹新鲜度感知 CI 融合（[`target_tracker.py`](target_tracker.py)）
- 经典基线与 RL **共用同一套局部航迹接口**

### 4.2 奖励（execution reward v5）

[`pursuit_rewards.py`](pursuit_rewards.py)：折扣一致势函数（终端清零）

- 个体接近 + **双机团队捕获势**（最近 `required_uavs` 架势平均）
- 合围进度、可见性进度
- 有界边界 / 建筑 / 机间接近惩罚

### 4.3 经典教师 / 基线

[`pursuit_baselines_3d.py`](pursuit_baselines_3d.py) + [`pursuit_guidance.py`](pursuit_guidance.py)

| 方法 | 作用 |
|------|------|
| APF | 吸引 + 斥力对照（通常很弱） |
| FRPN | 向量比例导引 + 直接响应；hybrid 上最强经典 |
| MPC | 短时域协同高层制导：双捕获时为最近 k 机分配拦截点，其余侧翼；路由 + 候选安全筛选 |

### 4.4 图策略

[`pursuit_graph_encoder.py`](pursuit_graph_encoder.py)：异构实体（self / target track / peer / building）+ 注意力融合。

- **MATD3-A6**：Twin team critic、延迟更新、目标策略平滑；演示 BC / TD3+BC / critic return clip / actor anchor 等增强
- **MAPPO+**：同编码器；与 DAgger 初始化衔接

### 4.5 训练流水线（PhysX-closed 正式路径）

```text
collect (MPC, success-only)
  → BC + DAgger (held-out 选优)
  → MATD3-A6 + MAPPO+（hard_physx_closed）
  → 五级阶梯评测
```

入口脚本：

- `queue_physx_closed_mpc_pipeline.sh` — 采集 → DAgger → 训练 → 评测
- `run_physx_closed_train.sh` / `run_physx_closed_eval.sh`
- `queue_physx_closed_resume_train.sh` — 从 `latest.pt` 续训（支持更稀验证）

Hybrid 历史路径曾同时跑 **FRPN / MPC** 两套教师；PhysX-closed **本轮正式 RL 表仅 MPC 教师**（FRPN 教师闭环重训未纳入本表）。

### 4.6 残差 / 部署

- 训练主表默认 **纯 RL**（`RESIDUAL_BASE=none`），与经典基线公平对比
- 首飞 / SITL：`ResidualPolicy` = `clip(expert + gain * rl)`（[`residual_policy.py`](residual_policy.py)）
- 真机栈：`real_flight_{obs,safety,node}.py` + `px4_offboard_bridge.py`

---

## 5. 实验结果摘要（勿混表）

评测协议：每方法 × 每级 **100** 局，seed 自 `7000000`；指标为捕获率（含 Wilson 95% CI，详见 JSON）。

### 5.1 Hybrid `isaac_dual_postlc`（对照）

Hard（训练域）要点：

| 方法 | Hard 捕获率 |
|------|------------:|
| MATD3-A6+FRPN | **0.81** |
| FRPN | 0.80 |
| MATD3-A6+MPC | 0.74 |
| MAPPO+FRPN | 0.73 |
| MPC / MAPPO+MPC | 0.63 |
| APF | 0.06 |

结论（hybrid）：**FRPN 热启显著优于 MPC**；A6+FRPN 略超教师并降低碰撞。完整矩阵见 `results/hybrid_postlc_eval_summary.json`。

### 5.2 PhysX-closed（正式）

Hard 要点（直径 0.75，MPC 教师 RL）：

| 方法 | Hard 捕获率 |
|------|------------:|
| MPC（经典） | **0.73** |
| FRPN（经典） | 0.66 |
| MATD3-A6+MPC | 0.62 |
| MAPPO+MPC | 0.58 |
| APF | 0.30 |

全阶梯捕获率（PhysX-closed）：

| 方法 | nominal | medium | hard | severe | extreme |
|------|--------:|-------:|-----:|-------:|--------:|
| APF | 0.46 | 0.38 | 0.30 | 0.19 | 0.17 |
| FRPN | 0.87 | 0.71 | 0.66 | 0.40 | 0.38 |
| MPC | 0.85 | 0.77 | **0.73** | 0.46 | 0.54 |
| MAPPO+MPC | 0.76 | 0.66 | 0.58 | 0.43 | 0.39 |
| MATD3-A6+MPC | 0.70 | 0.63 | 0.62 | 0.41 | 0.42 |

结论（physx_closed）：

1. 纯物理 + 更大捕获球后，**经典 MPC 成为 hard 最强**；FRPN 相对 hybrid 优势不再成立。  
2. 当前 **MPC 教师 RL 未超过教师**（A6 0.62 / MAPPO 0.58 vs MPC 0.73）——与 hybrid 上 A6+FRPN 超教师的叙事不同，**不得直接对比两个域的绝对捕获率**。  
3. Extreme 上 MPC 仍较强（0.54）；RL 有一定迁移但未反超。

### 5.3 Extreme 直径敏感性（非正式）

在 physx_closed 动力学下仅改 `target_diameter=0.5`（半径 1.0）评 extreme：

| 方法 | 正式 extreme (d=0.75) | 对照 d=0.5 |
|------|----------------------:|-----------:|
| APF | 0.17 | 0.03 |
| FRPN | 0.38 | 0.21 |
| MPC | 0.54 | （评测进行中 / 见结果 JSON 更新） |

说明捕获球缩小会明显压低极端难度成功率；**正式表仍以 d=0.75 为准**。

---

## 6. 配置清单（防混淆速查）

```text
configs/pursuit_v2/
  radar5_dual_{nominal,medium,hard,severe,extreme}.json     # hybrid 阶梯，diameter=0.5
  radar5_dual_*_physx_closed.json                            # 正式闭环，diameter=0.75
  radar5_dual_extreme_physx_closed_d05.json                  # 仅对照，diameter=0.5
  radar5_dual_hard_physx_closed_collect.json                 # 采集用 hard 变体
```

运行产物默认在 `$UAV_PROJECT_RUNS`（本机常见 `/srv/data/projects/kyy/runs`）：

```text
isaac_dual_postlc/train/isaac_dual_eval_*          # hybrid 原始评测
isaac_dual_physx_closed/train/isaac_dual_eval_*    # closed 原始评测
isaac_dual_physx_closed/train/isaac_dual_A6_mpc_*  # A6 训练
isaac_dual_physx_closed/train/isaac_dual_mappo_plus_mpc_*
```

权重（`.pt`）**不进 Git**；仓库只保留算法、配置与压缩结果 JSON。

---

## 7. 复现命令（PhysX-closed）

```bash
# 全流水线（MPC 教师）
bash queue_physx_closed_mpc_pipeline.sh

# 仅评测（已有 ckpt）
NUM_ENVS=10 TEACHER=mpc bash run_physx_closed_eval.sh

# 续训 A6 后评测
bash queue_physx_closed_resume_train.sh
```

依赖：Isaac Sim 4.1 + OmniDrones 环境；`ISAACSIM_PATH` / conda 见各 `run_*.sh` 头部。

---

## 8. 已知差异与后续

| 项 | 状态 |
|----|------|
| Hybrid FRPN 教师 RL | 已完成（对照表） |
| PhysX-closed MPC 教师 RL | 已完成（正式表） |
| PhysX-closed FRPN 教师重训 | **未纳入本轮正式表**；若重跑需新目录名，勿覆盖 MPC 表 |
| Extreme d=0.5 对照 | 敏感性实验；非正式 |
| 真机 / SITL | 契约与阶梯已就绪；数字未并入本文结果表 |

汇报或写论文时建议固定句式：

> Hybrid 结果仅作历史对照；正式仿真结论以 `isaac_dual_physx_closed`、`target_diameter=0.75`、几何 LiDAR、双机 `single_distance` 捕获为准。
