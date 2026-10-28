# PhysX-closed RL 改进路线（与实验排队）

## 当前状态（2026-10-28）

正式 ladder 配置已切到 **`target_diameter=0.5`**（`capture_radius=1.0`）。勿与旧 d=0.75 表混比。

| 项 | 状态 |
|----|------|
| 主线入口 | `queue_physx_closed_teacher_enhance.sh`（采集+近失误 → DAgger → A6∥MAPPO residual → 锁定评测） |
| Stage1 采集 | 完成 kept=240（succ≈155 + near-miss）；`hard_mpc_prior.pt` **未 finalize**（仅有 `.progress.pt`） |
| Stage2 DAgger | 完成；best=`dagger_round_02` val **0.575**；`pi0_frozen.pt` 已存 |
| Stage3 residual 并行训 | **阻塞**：缺 `hard_mpc_prior.pt` |
| 基线评测 d=0.5 | `queue_physx_closed_baseline_eval_d05.sh`；摘要见 `results/physx_closed_d05_baseline_eval_summary.json` |
| hard 捕获（d=0.5） | APF **0.04** / FRPN **0.46** / MPC **0.53** / π0 **0.47**（相对 d=0.75 的 0.30/0.66/0.73 明显更难） |

旧入口 `queue_physx_closed_residual_improve.sh` / `queue_physx_closed_d05_full_retrain.sh` 已转向 teacher-enhance 主线。

## 历史：残差 improve 队列（参考）

入口：`queue_physx_closed_residual_improve.sh`（**不覆盖**纯 RL 正式目录）。

| 步 | 内容 | 目录后缀 |
|----|------|----------|
| 1 | Eval-only：A6 `best` vs `final`、冻结 DAgger、MAPPO `best` vs `final`（hard×100） | `train/isaac_dual_eval_ckpt_compare_hard/` |
| 2 | 主实验：残差 MPC，A6 + MAPPO，200 ep | `*_residual_mpc_ep200` |
| 3 | 残差策略 hard 评测 | `*_eval_*_residual_ep200_*` |
| 4 | 可选 FRPN 残差（默认关，`SKIP_FRPN_RESIDUAL=0` 开启） | `*_residual_frpn_ep200` |

残差训练时自动启用收紧 O2O（`train_isaac_pursuit.py`：更低探索、更高 actor-anchor 下限、关闭 success-only replay）。

日志：`$UAV_PROJECT_RUNS/isaac_dual_physx_closed/logs/residual_improve_pipeline.log`

## 何时做 B（O2O 配方）与 C（课程/数据）

| 阶段 | 触发条件 | 做什么 |
|------|----------|--------|
| **B 轻量（已嵌入残差跑）** | 与残差主实验同时 | 探索↓、anchor 保留↑、混入失败回放 |
| **B 加重** | 残差 200ep 仍明显低于教师，或 val 仍随 ep 下滑 | 再提高 `td3_bc_alpha` / anchor；更短预算+更密 val；考虑 OVM 式冻结 Q 目标 |
| **C 课程/数据** | 残差 hard 已 ≥ 教师或接近后，extreme/severe 仍差 | medium↔hard 混合；演示保留失败纠偏；多步逃逸预测特征 |
| **改奖励** | 残差+O2O 后仍平台期，且失败模式可归因（贴靠不合围、过撞、丢航迹） | 再动 reward-v5；**不要与残差首轮同时改**，避免混杂 |

## 奖励要不要现在改？

**首轮不要。** 当前主因是 O2O 坍塌与「纯 RL 硬顶教师」，不是奖励完全失效（DAgger/MPC 已能到 ~0.7+）。  
若残差跑通后仍：双机时序差、碰撞高、遮挡丢跟，再针对性加项（失败模式 shaping），并单独开 `reward_v6` 配置以免污染正式表。
