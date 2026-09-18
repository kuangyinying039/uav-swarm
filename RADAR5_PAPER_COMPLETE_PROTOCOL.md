# Radar5 围捕论文：完整实验协议（重跑版）

本文件是**从零重跑论文实验**的唯一入口说明。编排脚本：
`run_radar5_paper_suite.py`。旧文档 `RADAR5_PAPER_EXPERIMENT_MANUAL.md` /
`RADAR5_EXPERIMENTS.md` 仍可参考，但默认训练域与命令以本协议为准。

所有命令在源码目录（如 `~/xieton/repro`）执行；相对路径 `outputs/...`
写入相邻的 `~/xieton/outputs/...`。

---

## 0. 关于「要不要从 Medium 开训」

**结论：正式重跑默认从 Medium 开训，不要把“超过 MPC”当成成功标准。**

| 选项 | 优点 | 风险 |
|---|---|---|
| 只在 Nominal 训 | 任务更易、学习曲线好看 | MPC 校准捕获率≈100%，主表几乎不可能“赢”专家，读者易误判失败 |
| **在 Medium 训（推荐）** | MPC≈95%，规则基线与学习策略有区分空间；Hard 仍是更难零样本 | 必须在 Medium 上重采 transition / demo / DAgger |
| 在 Hard 训 | 最难场景内分布 | 样本噪声大、捕获稀疏，训练不稳定 |

论文口径（写进方法/实验节）：

1. **主训练域 = Medium**。
2. **Nominal / Hard = 同一 checkpoint 的迁移测试**（Nominal 更易，Hard 更难）。
3. **MPC 是示范教师与专家上界**；学习方法使用 MPC 数据，成立条件是：超过 APF/FRPN、接近 MPC，并在推理耗时 / 可见度 / 安全性上给出额外证据。
4. 旧的 Nominal seed11 试跑可保留为附录敏感性，**不能**与 Medium 主结果混进同一主表。

---

## 1. 论文必须回答的问题 → 必做实验矩阵

| RQ | 问题 | 必做实验 | 产出 |
|---|---|---|---|
| RQ1 | 有限雷达 + 遮挡规避下，学习策略能否提高捕获、保持可见、安全飞行？ | 主表：APF / FRPN / MPC / MAPPO / HGAT-MATD3-Opt，在 Medium（训练域）上 200-seed | 主表 |
| RQ2 | HGAT 相对 MLP 是否有贡献？ | 消融 A0 vs A1（scratch） | 消融表 |
| RQ3 | DAgger/prior、critic 预热、BC 保持是否改善 MATD3？ | A2 / A3 / A5；曲线看 `demo_bc_loss`、`critic_pretrain.json` | 消融表 + 诊断 |
| RQ4 | 训练域策略能否迁移到更易/更难环境？ | 同一 ckpt 测 Nominal / Medium / Hard | 难度图 |
| RQ5 | 可见度奖励是否必要？ | A4 vs A5，**测试仍用标准难度配置** | 消融表 |

### 1.1 主对比（正文必须）

五种方法：`APF`、`FRPN`、`MPC`、`MAPPO`、`HGAT-MATD3-Opt`。

报告指标：捕获率（Wilson 或多种子分层 bootstrap CI）、平均删失步数、team visibility、UAV visibility、safety correction magnitude、intervention-free episode rate、每步策略耗时（`--workers 1` 另跑）。

### 1.2 消融（正文至少三组；算力够则 A0–A5 全报）

| ID | 设置 | 对照问题 |
|---|---|---|
| A0 | MATD3-MLP scratch | 无图、无专家下限 |
| A1 | HGAT-MATD3 scratch | HGAT 表示 |
| A2 | DAgger+prior，无 BC、无 critic 预热 | 仅专家初始化 |
| A3 | 完整方法，无 critic 预热 | critic 预热 |
| A4 | 完整方法，无可见度奖励（专用 prior） | 可见度目标 |
| A5 | HGAT-MATD3-Opt | 主方法 |

资源紧时正文保留 **A0/A1、A3/A5、A4/A5**；A2 放附录。

### 1.3 不要再扩的内容

- BC / 单独 DAgger 作为主表方法
- 多个 MAPPO 变体并列（正文只留 warmstart 最强对照）
- 在 `7000000...` 测试集上调 Medium/Hard 参数
- 单一种子的正式结论

---

## 2. 冻结配置与种子

训练配置：`configs/pursuit_v2/radar5_benchmark_medium.json`  
无可见度消融：`configs/pursuit_v2/radar5_ablation_no_visibility_medium.json`  
测试配置：`radar5_benchmark_{nominal,medium,hard}.json`

| 用途 | Seed 段 |
|---|---|
| 示范 / transition | `2000000...` |
| DAgger 修正 | `5000000...` |
| 训练 | base 11/12/13 → `1100000...` 等 |
| 验证（选 best_capture） | `4000000...4000049` |
| 正式测试 | `7000000...7000199`（200 场景，全方法共用） |

---

## 3. 一键编排（推荐）

先自检：

```bash
cd ~/xieton/repro
git pull --ff-only origin main
python -m compileall -q .
python -m unittest \
  test_pursuit_scenarios test_velocity_lidar test_pursuit_matd3 \
  test_compare_pursuit_difficulty test_aggregate_pursuit_training_seeds \
  test_run_radar5_paper_suite
```

只打印完整命令（不执行）：

```bash
python run_radar5_paper_suite.py --stage all --train-level medium --dry-run
```

按阶段执行（可断点续跑；每阶段成功后再进下一阶段）：

```bash
# 1) Medium 上重采 MPC transition、成功 demo、DAgger，以及 A4 专用 prior
python run_radar5_paper_suite.py --stage assets --train-level medium

# 2) 三档规则基线 200-seed
python run_radar5_paper_suite.py --stage baselines --train-level medium

# 3) 主方法：先只跑 seed11 冒烟，再 seeds=11,12,13
python run_radar5_paper_suite.py --stage train_main --train-level medium --seeds 11
# 确认 validation 捕获率不崩、critic_pretrain.json 正常后：
python run_radar5_paper_suite.py --stage train_main --train-level medium --seeds 12,13

# 4) 消融（可先 seed11）
python run_radar5_paper_suite.py --stage train_ablation --train-level medium --seeds 11,12,13

# 5) 正式测试（主方法三档）
python run_radar5_paper_suite.py --stage evaluate --train-level medium --seeds 11,12,13

# 5b) 消融表：仅在训练域 Medium 上评估 A0–A4
python run_radar5_paper_suite.py --stage evaluate_ablation --train-level medium --seeds 11,12,13

# 6) 聚合与五方法难度图
python run_radar5_paper_suite.py --stage aggregate --train-level medium --seeds 11,12,13
python run_radar5_paper_suite.py --stage aggregate_ablation --train-level medium --seeds 11,12,13

# 7) 定性轨迹
python run_radar5_paper_suite.py --stage qualitative --train-level medium --seeds 11
```

说明：`--stage all` 会按顺序包含 assets → baselines → train_main → train_ablation → evaluate → evaluate_ablation → aggregate → aggregate_ablation → qualitative。算力紧张时请分阶段跑，先 `train_main --seeds 11` 冒烟。

输出前缀默认 `radar5_medium_*`。若坚持 Nominal 开训：

```bash
python run_radar5_paper_suite.py --stage all --train-level nominal --dry-run
```

---

## 4. 阶段与产物对照

| 阶段 | 关键产物 |
|---|---|
| assets | `radar5_medium_mpc_transitions200.pt`、`..._mpc_success50.pt`、`..._dagger/best_imitation.pt`、`..._mpc_transitions200_no_visibility.pt` |
| baselines | `radar5_medium_benchmark_{nominal,medium,hard}_baselines_200.json` |
| train_main | `..._hgat_matd3_opt_seed{11,12,13}/`、`..._mappo_dagger_replay_seed*/` |
| train_ablation | `..._ablation_a0_*` … `a4_*` |
| evaluate | `..._eval_*_{nominal,medium,hard}/evaluation.json` |
| aggregate | `..._aggregate_*.json`、`..._difficulty_five_methods/` |
| qualitative | `..._qualitative/` |

每个训练目录保留：`best_capture.pt`、`training.csv`、`validation.json`、`critic_pretrain.json`（若有）、Git commit 记录。

---

## 5. 实现注意（重跑必读）

1. **Transition / demo / DAgger 必须与训练域同一 `env-config`。** Medium 开训不能复用旧 Nominal `radar5_mpc_transitions200.pt`。
2. **MATD3 的 `--env-config` 优先生效**（不再被 `--actor-init` 里的旧环境静默覆盖）。跨难度只允许权重兼容的维度契约。
3. **MAPPO 使用 `--checkpoint` 时不要再传 `--env-config`**（脚本已处理）。
4. **A4 使用独立 no-visibility prior**；测试仍用标准 `radar5_benchmark_*.json`。
5. 正式捕获率多种子结果用 `aggregate_pursuit_training_seeds.py` 的**分层 bootstrap**，不要把 3×200 轨迹当 600 独立样本。

---

## 6. 论文图表清单（交稿前勾选）

- [ ] 主表（Medium 训练域）：五方法 × 捕获/可见/安全/耗时
- [ ] 难度图：Nominal / Medium / Hard × 五方法
- [ ] 学习曲线：验证捕获率 vs 环境步（三种子均值±标准差）
- [ ] 消融表：A0–A5（或精简三组）
- [ ] 定性图：训练域成功 + Hard 迁移案例（建筑、可见区间、捕获时刻）
- [ ] 记录 commit、配置哈希、checkpoint 路径、硬件与框架版本
