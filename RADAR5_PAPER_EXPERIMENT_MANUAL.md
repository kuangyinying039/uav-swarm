# Radar5 围捕论文实验说明书

> **重跑请改用** [`RADAR5_PAPER_COMPLETE_PROTOCOL.md`](RADAR5_PAPER_COMPLETE_PROTOCOL.md)
> 与脚本 `run_radar5_paper_suite.py`。默认训练域已改为 **Medium**；MPC 为专家上界而非必须超越的目标。
> 下文保留 Nominal 口径的历史说明，仅作对照。

本说明书用于冻结环境、训练、模型选择、正式测试、统计与作图。所有命令从
`~/xieton/repro` 执行；命令中的 `outputs/...` 实际写入 `~/xieton/outputs/...`。

## 1. 研究问题与实验口径

论文应回答四个问题：

1. 在有限雷达感知和遮挡规避目标下，学习策略能否提高捕获率、保持目标可见并安全飞行？
2. HGAT 相比普通 MLP 表示是否有贡献？
3. DAgger/专家回放、critic 预热和行为保持是否能改善 MATD3 的训练稳定性？
4. 只在 Nominal 场景训练的策略能否零样本泛化到 Medium 和 Hard？

难度曲线统一比较五种方法：`APF`、`FRPN`、`MPC`、`MAPPO` 和
`HGAT-MATD3-Opt`。MPC 是示范教师和专家上界。由于学习方法使用 MPC 数据，论文不应把
“必须超过 MPC”设为成立条件；合理目标是超过 APF/FRPN、接近 MPC，并在推理耗时、泛化、
可见度或安全性上给出额外证据。

## 2. 冻结的环境定义

三个场景保持 3 架无人机、1 个目标、5 栋建筑、25 m × 25 m 平面、1–12 m 高度、
300 步时限、连续速度/偏航角速度动作及相同捕获条件。捕获条件为任意一架有效无人机与
目标的三维距离不超过 1.0 m，持续 1 步。

| 参数 | Nominal | Medium | Hard |
|---|---:|---:|---:|
| 初始三角编队到目标距离 | 6.0–8.0 m | 7.5–9.5 m | 9.5–11.0 m |
| 目标水平速度上限 | 1.82 m/s | 2.15 m/s | 2.35 m/s |
| 目标垂直速度上限 | 1.04 m/s | 1.20 m/s | 1.30 m/s |
| 建筑边长 | 2.0–4.5 m | 2.5–5.0 m | 3.0–5.5 m |
| 建筑高度 | 3.0–10.0 m | 4.0–11.0 m | 5.0–11.5 m |
| 雷达单次检出概率 | 0.90 | 0.85 | 0.80 |

配置文件：

- `configs/pursuit_v2/radar5_benchmark_nominal.json`
- `configs/pursuit_v2/radar5_benchmark_medium.json`
- `configs/pursuit_v2/radar5_benchmark_hard.json`

正式训练只使用 Nominal。难度实验将同一个 Nominal checkpoint 分别放入三档环境测试，
属于零样本环境迁移；不能在 Medium/Hard 测试结果出来后继续选择 checkpoint 或调环境。

### 2.1 无人机信息

MID-360 替代模型水平视场为 360°，垂直视场为 −7°～52°，有效量程 0.2–12 m，扫描率
10 Hz。建筑会阻断视线；测量含距离相关噪声，并按三档概率随机漏检。策略输入使用各无人机
自身状态、本地 KF/CI 目标航迹、通信可达队友状态和已知建筑几何。Actor 不读取目标真值。
集中式 critic 读取所有策略可见观测与无人机刚体状态的联合状态，不额外读取目标真值。

### 2.2 目标信息

目标以 0.10 m 标准差的高斯噪声观测所有有效无人机位置，并用相邻观测估计其速度；目标
知道建筑几何，可从侧面、远侧和屋顶候选点中选择遮挡航路。它不知道无人机未来动作。
目标感知目前不受无人机雷达量程和建筑遮挡限制，因此是信息较强的规避者。论文方法部分应
明确这一不对称设定。

## 3. 数据划分与统计单位

- 示范数据：`2000000...`；只用于 BC/DAgger/专家回放。
- DAgger 修正：`5000000...`。
- 训练：base seed 11、12、13；代码生成互不重叠的 `1100000...`、`1200000...`、
  `1300000...` 场景。
- 验证：`4000000...4000049`；只用于选 `best_capture.pt`。
- 难度校准：`6000000...6000039`；环境冻结后不再使用。
- 正式测试：`7000000...7000199`；所有方法、所有难度使用相同的 200 个场景。

每个学习方法至少训练 3 个随机种子。单一固定策略的捕获率可报告 Wilson 95% CI；多个
训练种子的正式结果使用训练种子与测试场景两层 bootstrap。不要把 3 × 200 条轨迹直接当成
600 个独立样本。

## 4. 执行顺序

### 4.1 更新和自检

```bash
cd ~/xieton/repro
git pull --ff-only origin cursor/hgat-matd3-f36a
python -m compileall -q .
python -m unittest test_pursuit_scenarios test_velocity_lidar test_pursuit_matd3 \
  test_compare_pursuit_difficulty test_aggregate_pursuit_training_seeds
```

确认数据存在：

```bash
ls -lh ~/xieton/outputs/radar5_mpc_transitions200.pt
ls -lh ~/xieton/outputs/radar5_dagger/best_imitation.pt
ls -lh ~/xieton/outputs/radar5_dagger/corrections_round_02.pt
```

只有环境版本、观测版本、动作含义或奖励版本改变时才重采示范。此次 MATD3 优化不改变这些
契约，所以现有 transition 和 DAgger checkpoint 可以继续使用。

### 4.2 先做单种子试运行

新 MATD3 加入两项可独立关闭的稳定化机制：先用专家回放训练 twin critic，再在在线阶段把
行为噪声从 0.10 降到 0.03。critic 预热不更新 actor。

```bash
python train_pursuit_matd3.py train \
  --env-config configs/pursuit_v2/radar5_benchmark_nominal.json \
  --actor-init outputs/radar5_dagger/best_imitation.pt \
  --prior outputs/radar5_mpc_transitions200.pt \
  --prior-fraction 0.5 \
  --warmup-policy actor \
  --critic-pretrain-updates 5000 \
  --actor-lr 5e-5 --critic-lr 3e-4 \
  --exploration-std 0.10 --exploration-final-std 0.03 \
  --exploration-decay-steps 150000 \
  --demo-bc-weight 1.0 --demo-bc-final-weight 0.05 \
  --demo-bc-decay-steps 150000 \
  --episodes 3000 --seed 11 \
  --validation-seed 4000000 --validation-episodes 50 --validation-interval 100 \
  --step-checkpoint-interval 10000 \
  --out outputs/radar5_hgat_matd3_opt_seed11
```

先检查 `critic_pretrain.json`、`validation.json`、`training.csv` 和
`reward_capture.svg`。只有命令能够完整结束、Q 值不发散、验证捕获率没有持续下降后，才运行
seed 12 和 13。

### 4.3 正式训练 HGAT-MATD3

```bash
for SEED in 11 12 13; do
  python train_pursuit_matd3.py train \
    --env-config configs/pursuit_v2/radar5_benchmark_nominal.json \
    --actor-init outputs/radar5_dagger/best_imitation.pt \
    --prior outputs/radar5_mpc_transitions200.pt \
    --prior-fraction 0.5 --warmup-policy actor \
    --critic-pretrain-updates 5000 \
    --actor-lr 5e-5 --critic-lr 3e-4 \
    --exploration-std 0.10 --exploration-final-std 0.03 \
    --exploration-decay-steps 150000 \
    --demo-bc-weight 1.0 --demo-bc-final-weight 0.05 \
    --demo-bc-decay-steps 150000 \
    --episodes 3000 --seed "$SEED" \
    --validation-seed 4000000 --validation-episodes 50 --validation-interval 100 \
    --step-checkpoint-interval 10000 \
    --out "outputs/radar5_hgat_matd3_opt_seed${SEED}"
done
```

已完成的旧 `radar5_hgat_matd3_dagger_bc_seed11` 是算法优化前的试验，可保留作消融，不能替代
上述三个正式种子。

### 4.4 正式训练 MAPPO 对照

MAPPO 使用相同 DAgger 初始化和修正回放。若旧 seed 11 的训练确实为 3000 回合、环境为
Nominal、验证集为当前冻结集合，可保留；否则三个种子统一重跑。

```bash
for SEED in 11 12 13; do
  python train_pursuit_with_demos.py train \
    --checkpoint outputs/radar5_dagger/best_imitation.pt \
    --demos outputs/radar5_mpc_success50.pt \
    --aux-demos outputs/radar5_dagger/corrections_round_02.pt \
    --training-profile warmstart \
    --episodes 3000 --seed "$SEED" \
    --validation-seed 4000000 --validation-episodes 50 --validation-interval 100 \
    --out "outputs/radar5_mappo_dagger_replay_seed${SEED}"
done
```

### 4.5 三个规则基线

```bash
for LEVEL in nominal medium hard; do
  python evaluate_3d_baselines.py \
    --env-config "configs/pursuit_v2/radar5_benchmark_${LEVEL}.json" \
    --methods apf frpn mpc --seed-start 7000000 --seed-count 200 \
    --workers 8 --out "outputs/benchmark_${LEVEL}_baselines_200.json"
done
```

策略耗时的正式比较需要 `--workers 1` 单独运行，以免多进程竞争污染延迟。

### 4.6 同一 checkpoint 跨三档测试

HGAT-MATD3：

```bash
for SEED in 11 12 13; do
  for LEVEL in nominal medium hard; do
    python train_pursuit_matd3.py evaluate \
      --checkpoint "outputs/radar5_hgat_matd3_opt_seed${SEED}/best_capture.pt" \
      --eval-env-config "configs/pursuit_v2/radar5_benchmark_${LEVEL}.json" \
      --eval-seed 7000000 --eval-episodes 200 --methods matd3 \
      --out "outputs/eval_hgat_matd3_opt_seed${SEED}_${LEVEL}"
  done
done
```

MAPPO：

```bash
for SEED in 11 12 13; do
  for LEVEL in nominal medium hard; do
    python train_pursuit_with_demos.py evaluate \
      --checkpoint "outputs/radar5_mappo_dagger_replay_seed${SEED}/best_capture.pt" \
      --eval-env-config "configs/pursuit_v2/radar5_benchmark_${LEVEL}.json" \
      --eval-seed 7000000 --eval-episodes 200 --methods mappo \
      --out "outputs/eval_mappo_seed${SEED}_${LEVEL}"
  done
done
```

### 4.7 聚合训练种子并绘制五方法难度图

```bash
for LEVEL in nominal medium hard; do
  python aggregate_pursuit_training_seeds.py \
    --input "outputs/eval_hgat_matd3_opt_seed11_${LEVEL}/evaluation.json" \
    --input "outputs/eval_hgat_matd3_opt_seed12_${LEVEL}/evaluation.json" \
    --input "outputs/eval_hgat_matd3_opt_seed13_${LEVEL}/evaluation.json" \
    --label HGAT_MATD3_OPT \
    --out "outputs/aggregate_hgat_matd3_opt_${LEVEL}.json"

  python aggregate_pursuit_training_seeds.py \
    --input "outputs/eval_mappo_seed11_${LEVEL}/evaluation.json" \
    --input "outputs/eval_mappo_seed12_${LEVEL}/evaluation.json" \
    --input "outputs/eval_mappo_seed13_${LEVEL}/evaluation.json" \
    --label MAPPO \
    --out "outputs/aggregate_mappo_${LEVEL}.json"
done
```

```bash
python compare_pursuit_difficulty.py \
  --require-methods APF FRPN MPC MAPPO HGAT_MATD3_OPT \
  --input Nominal:APF=outputs/benchmark_nominal_baselines_200.json \
  --input Medium:APF=outputs/benchmark_medium_baselines_200.json \
  --input Hard:APF=outputs/benchmark_hard_baselines_200.json \
  --input Nominal:FRPN=outputs/benchmark_nominal_baselines_200.json \
  --input Medium:FRPN=outputs/benchmark_medium_baselines_200.json \
  --input Hard:FRPN=outputs/benchmark_hard_baselines_200.json \
  --input Nominal:MPC=outputs/benchmark_nominal_baselines_200.json \
  --input Medium:MPC=outputs/benchmark_medium_baselines_200.json \
  --input Hard:MPC=outputs/benchmark_hard_baselines_200.json \
  --input Nominal:MAPPO=outputs/aggregate_mappo_nominal.json \
  --input Medium:MAPPO=outputs/aggregate_mappo_medium.json \
  --input Hard:MAPPO=outputs/aggregate_mappo_hard.json \
  --input Nominal:HGAT_MATD3_OPT=outputs/aggregate_hgat_matd3_opt_nominal.json \
  --input Medium:HGAT_MATD3_OPT=outputs/aggregate_hgat_matd3_opt_medium.json \
  --input Hard:HGAT_MATD3_OPT=outputs/aggregate_hgat_matd3_opt_hard.json \
  --out-dir outputs/radar5_difficulty_five_methods
```

`--require-methods` 会拒绝缺少任一难度或任一方法的图，避免再次只画两条曲线。

## 5. 必须完成的消融实验

正式消融至少使用 seed 11、12、13；可先用 seed 11 筛查命令和趋势。

| 编号 | 实验 | 改动 | 回答的问题 |
|---|---|---|---|
| A0 | MATD3-MLP scratch | `--no-gat`、无专家数据 | 普通 MATD3 下限 |
| A1 | HGAT-MATD3 scratch | 移除 `--no-gat` | HGAT 表示是否有效 |
| A2 | DAgger + prior，无保持 | BC 权重 0、critic 预热 0 | 专家初始化本身的效果 |
| A3 | 完整方法，无 critic 预热 | `--critic-pretrain-updates 0` | critic 预热贡献 |
| A4 | 完整方法，无可见度奖励 | 使用 no-visibility 配置 | 可见度目标贡献 |
| A5 | 完整 HGAT-MATD3-Opt | 全部机制 | 主方法 |

无可见度奖励配置为：
`configs/pursuit_v2/radar5_ablation_no_visibility.json`。训练后仍在标准 Nominal 配置上测试。

若计算资源有限，论文正文保留 A0/A1、A3/A5、A4/A5 三组最直接的对照；A2 放补充材料。
探索衰减、prior fraction、BC 权重网格属于超参数敏感性，可只用 seed 11 在验证集选择，不能用
`7000000...` 测试集选择。

## 6. 论文必须报告的结果

### 6.1 主表

Nominal 上比较 APF、FRPN、MPC、MAPPO、HGAT-MATD3-Opt。报告：

- 捕获率及 95% CI；
- 平均删失步数；
- team visibility ratio；
- UAV visibility ratio；
- safety correction magnitude；
- intervention-free episode rate；
- 每步策略计算时间。

成功回合捕获步数会受到“只统计成功者”的选择偏差，只作为辅助指标。单目标下
`target_visibility_rate` 与 `team_visibility_ratio` 重复，正文只保留后者。

### 6.2 泛化图

五种方法在 Nominal/Medium/Hard 的捕获率、team visibility 和平均删失步数。图题应写清：
学习策略只在 Nominal 训练，三档测试使用同一 checkpoint 和同一测试 seed。

### 6.3 学习曲线

每个学习方法绘制环境步数横轴上的验证捕获率、删失步数和回报，显示三个训练种子的均值和
标准差。训练回报不能代替 held-out 验证捕获率。

### 6.4 消融表

报告 A0–A5 的 Nominal 捕获率、可见度、安全性和训练种子标准差。critic 预热实验同时报告
`critic_pretrain.json` 的 TD error 和 Q disagreement；行为保持实验报告 `demo_bc_loss`。

### 6.5 定性结果

选择固定测试 seed，各展示一个成功、遮挡后重捕获和失败案例。轨迹图必须标记建筑、目标、
三架无人机、雷达可见/不可见区间和捕获时刻。案例用于解释机制，不能替代统计结果。

## 7. 可以移出正文或停止继续扩展的实验

以下已有结果可保留在补充材料，不再作为主表独立方法：

- BC、DAgger 单独结果；
- MAPPO-BC、MAPPO-DAgger、MAPPO-DAggerReplay 三者全部并列；正文只保留最强且预先确定的
  MAPPO 对照；
- MATD3-Prior、MATD3-BC-Prior、MATD3-DAgger-Prior 等旧实现；
- 单个训练种子的正式结论；
- `mean_visibility_ratio`、`target_visibility_rate` 和 `team_visibility_ratio` 的重复报告；
- 在正式测试 seed 上反复修改 Medium/Hard 参数的结果。

旧输出先归档，不必立即删除；主结果目录只保留代码 commit、配置、checkpoint、
`training.csv`、`validation.json`、`evaluation.json` 和绘图源数据。

## 8. 是否需要重跑

必须重跑：

- 优化后的 HGAT-MATD3 三个训练种子；
- MAPPO 缺失的训练种子；
- 两个学习方法在三档环境的统一 200-seed 测试；
- 新增的 critic 预热和无可见度奖励消融。

可以复用：

- 当前 v3/v4/v2 契约下的 MPC transition 数据；
- 当前 DAgger `best_imitation.pt` 和对应 corrections；
- 已冻结配置上的 APF/FRPN/MPC 200-seed 结果；
- 满足同一环境、训练回合、验证集合和当前代码版本的 MAPPO checkpoint。

任何正式表格都应记录 Git commit、Python/PyTorch/CUDA 版本、GPU、训练 seed、验证 seed、测试
seed、配置文件哈希和 checkpoint 路径。
