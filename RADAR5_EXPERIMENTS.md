# Radar5 论文实验流程

所有命令从 `~/xieton/repro` 执行。相对输出路径 `outputs/...` 会写到相邻的
`~/xieton/outputs/...`，不会写进源码仓库。

## 1. 更新与自检

```bash
git pull --ff-only origin cursor/hgat-matd3-f36a
python -m compileall -q .
python -m unittest test_pursuit_scenarios test_velocity_lidar test_pursuit_matd3
```

三档评估配置保持捕获条件、无人机数量、建筑数量、观测维度和动作维度一致：

- Nominal：`configs/pursuit_v2/radar5_benchmark_nominal.json`
- Medium：`configs/pursuit_v2/radar5_benchmark_medium.json`
- Hard：`configs/pursuit_v2/radar5_benchmark_hard.json`

Nominal 与原来的 `previous_models_current_radar.json` 参数相同。Medium 和 Hard
只增大初始距离、目标速度、建筑尺度，并降低单次雷达检出概率。

| 参数 | Nominal | Medium | Hard |
|---|---:|---:|---:|
| 三角编队到目标距离 (m) | 6.0–8.0 | 7.5–9.5 | 9.5–11.0 |
| 目标水平速度上限 (m/s) | 1.82 | 2.15 | 2.35 |
| 目标垂直速度上限 (m/s) | 1.04 | 1.20 | 1.30 |
| 建筑边长范围 (m) | 2.0–4.5 | 2.5–5.0 | 3.0–5.5 |
| 建筑高度范围 (m) | 3.0–10.0 | 4.0–11.0 | 5.0–11.5 |
| 单次雷达检出概率 | 0.90 | 0.85 | 0.80 |

难度参数只使用 `6000000..6000039` 校准 seed 调整一次，然后冻结。40 回合 MPC
校准结果为 Nominal 100%、Medium 95%、Hard 77.5%；对应平均删失步数为
51.7、69.1、120.1。`7000000` 段 seed 只用于冻结后的正式测试，不能根据正式测试
结果继续调环境。40 回合校准比例只是难度检查，最终论文仍报告 100–300 回合结果及
置信区间。

## 2. 固定规则基线

三档都使用相同的 100 个测试 seed：

```bash
python evaluate_3d_baselines.py --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --methods apf frpn mpc --seed-start 7000000 --seed-count 200 --workers 8 --out outputs/benchmark_nominal_baselines_200.json
python evaluate_3d_baselines.py --env-config configs/pursuit_v2/radar5_benchmark_medium.json --methods apf frpn mpc --seed-start 7000000 --seed-count 200 --workers 8 --out outputs/benchmark_medium_baselines_200.json
python evaluate_3d_baselines.py --env-config configs/pursuit_v2/radar5_benchmark_hard.json --methods apf frpn mpc --seed-start 7000000 --seed-count 200 --workers 8 --out outputs/benchmark_hard_baselines_200.json
```

APF、FRPN 和 MPC 都只读取各无人机本地 KF/CI 航迹，不读取目标真值。MPC 是专家
上界，论文中应同时报告 learned policy 与 APF/FRPN 的比较，以及相对 MPC 的差距。

## 3. 补齐纯 RL 与 HGAT 消融

MAPPO scratch：

```bash
python train_pursuit_with_demos.py train --training-profile plain --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --bc-updates 0 --episodes 3000 --seed 11 --validation-seed 4000000 --validation-episodes 10 --validation-interval 200 --out outputs/radar5_mappo_scratch_seed11
```

普通 MATD3 scratch（MLP actor，用于证明 HGAT 的贡献）：

```bash
python train_pursuit_matd3.py train --no-gat --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --prior-fraction 0 --warmup-policy random --episodes 3000 --seed 11 --validation-seed 4000000 --validation-episodes 10 --validation-interval 200 --step-checkpoint-interval 10000 --out outputs/radar5_matd3_mlp_scratch_seed11
```

HGAT-MATD3 scratch：

```bash
python train_pursuit_matd3.py train --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --prior-fraction 0 --warmup-policy random --episodes 3000 --seed 11 --validation-seed 4000000 --validation-episodes 10 --validation-interval 200 --step-checkpoint-interval 10000 --out outputs/radar5_hgat_matd3_scratch_seed11
```

论文正式统计将上述命令分别用 `--seed 11`、`--seed 12`、`--seed 13` 运行。不同
训练 seed 的输出目录必须分开。

## 4. 诊断 DAgger 到 MATD3 的退化

先直接评估导入 MATD3 前的 DAgger actor：

```bash
python train_pursuit_matd3.py evaluate --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --actor-init outputs/radar5_dagger/best_imitation.pt --eval-seed 7000000 --eval-episodes 100 --methods matd3 --out outputs/eval_dagger_actor_before_matd3
```

然后运行“DAgger 初始化 + prior replay，但不加 BC”的纯 HGAT-MATD3。预训练 actor
在 warmup 阶段继续控制无人机，避免 8000 步随机策略覆盖其访问分布：

```bash
python train_pursuit_matd3.py train --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --actor-init outputs/radar5_dagger/best_imitation.pt --prior outputs/radar5_mpc_transitions200.pt --prior-fraction 0.5 --warmup-policy actor --demo-bc-weight 0 --episodes 3000 --seed 11 --validation-seed 4000000 --validation-episodes 50 --validation-interval 50 --step-checkpoint-interval 10000 --out outputs/radar5_hgat_matd3_dagger_init_seed11
```

训练会自动生成 `initial.pt`，并在 `validation.json` 写入 `episode=0, env_steps=0`
的训练前结果（`pre_rl=true`）。该次验证只作 DAgger 上限参考，**不参与**
`best_capture.pt` 竞选，避免 warm-start 永久占据最优 checkpoint。正式报告应分开写
`initial.pt`（模仿上限）与 `best_capture.pt`（RL 后最优）。`checkpoints/step_*.pt`
用于画 0、10k、20k……环境步的性能曲线。

## 5. 专家保持版本：HGAT-MATD3 + decaying BC

该实验显式增加只作用于 prior 行的行为克隆损失。权重从 1.0 线性下降到 0.05，
纯 MATD3 对照仍使用 `--demo-bc-weight 0`。

```bash
python train_pursuit_matd3.py train --env-config configs/pursuit_v2/radar5_benchmark_nominal.json --actor-init outputs/radar5_dagger/best_imitation.pt --prior outputs/radar5_mpc_transitions200.pt --prior-fraction 0.5 --warmup-policy actor --demo-bc-weight 1.0 --demo-bc-final-weight 0.05 --demo-bc-decay-steps 150000 --episodes 3000 --seed 11 --validation-seed 4000000 --validation-episodes 10 --validation-interval 200 --step-checkpoint-interval 10000 --out outputs/radar5_hgat_matd3_dagger_bc_seed11
```

`history.json` 和 `training.csv` 会额外记录 `actor_rl_loss`、`demo_bc_loss`、
`demo_bc_weight`、`target_q_mean`、`td_abs_mean`、`q_disagreement` 和
`prior_batch_fraction`。如果捕获率从 step 0 开始立即下降，同时 BC drift 增大，说明
actor 被 critic 更新破坏；如果 TD error 持续很大，则先处理 critic/奖励尺度。

## 6. 统一测试 checkpoint

MATD3 示例：

```bash
python train_pursuit_matd3.py evaluate --checkpoint outputs/radar5_hgat_matd3_dagger_bc_seed11/best_capture.pt --eval-env-config configs/pursuit_v2/radar5_benchmark_nominal.json --eval-seed 7000000 --eval-episodes 200 --methods matd3 --out outputs/eval_hgat_matd3_dagger_bc_nominal
python train_pursuit_matd3.py evaluate --checkpoint outputs/radar5_hgat_matd3_dagger_bc_seed11/best_capture.pt --eval-env-config configs/pursuit_v2/radar5_benchmark_medium.json --eval-seed 7000000 --eval-episodes 200 --methods matd3 --out outputs/eval_hgat_matd3_dagger_bc_medium
python train_pursuit_matd3.py evaluate --checkpoint outputs/radar5_hgat_matd3_dagger_bc_seed11/best_capture.pt --eval-env-config configs/pursuit_v2/radar5_benchmark_hard.json --eval-seed 7000000 --eval-episodes 200 --methods matd3 --out outputs/eval_hgat_matd3_dagger_bc_hard
```

MAPPO 示例；checkpoint 已保存训练环境，因此不再传 `--env-config`：

```bash
python train_pursuit_with_demos.py evaluate --checkpoint outputs/radar5_mappo_dagger_replay/best_capture.pt --eval-env-config configs/pursuit_v2/radar5_benchmark_hard.json --eval-seed 7000000 --eval-episodes 200 --methods mappo --out outputs/eval_mappo_dagger_replay_hard
```

绘制难度曲线：

```bash
python compare_pursuit_difficulty.py \
  --input Nominal:MPC=outputs/benchmark_nominal_baselines_200.json \
  --input Medium:MPC=outputs/benchmark_medium_baselines_200.json \
  --input Hard:MPC=outputs/benchmark_hard_baselines_200.json \
  --input Nominal:HGAT_MATD3_BC=outputs/eval_hgat_matd3_dagger_bc_nominal/evaluation.json \
  --input Medium:HGAT_MATD3_BC=outputs/eval_hgat_matd3_dagger_bc_medium/evaluation.json \
  --input Hard:HGAT_MATD3_BC=outputs/eval_hgat_matd3_dagger_bc_hard/evaluation.json \
  --out-dir outputs/radar5_difficulty_comparison
```

正式表格优先报告 capture rate（Wilson 95% CI）、mean censored steps、team visibility、
UAV visibility、building occlusion、collision、safety correction magnitude、P90 safety
interventions 和 intervention-free episode rate。单目标情况下 `target_visibility_rate` 与
`team_visibility_ratio` 数学上相同，论文只保留后者。

## 7. 双机捕获 + Extreme + Hard 混合训练（对照协议）

旧 `radar5_benchmark_*` 仍是单机捕获，保留作历史对照。新协议使用：

| 参数 | Dual Nominal | Dual Medium | Dual Hard | Dual Extreme |
|---|---:|---:|---:|---:|
| `capture_required_uavs` | 2 | 2 | 2 | 2 |
| 三角初始距 (m) | 6–8 | 7.5–9.5 | 9.5–11 | 11–13 |
| 目标水平速 (m/s) | 1.82 | 2.15 | 2.35 | 2.60 |
| 雷达检出概率 | 0.90 | 0.85 | 0.80 | 0.55 |
| 雷达量程 (m) | 12 | 12 | 12 | 9 |

配置：`configs/pursuit_v2/radar5_dual_{nominal,medium,hard,extreme}.json`。
观测/动作维度与旧 Radar5 一致，可零样本跨档评估；但捕获成功标签变了，**必须重采**
MPC prior / DAgger（不要直接复用旧 `radar5_mpc_transitions200.pt` 的成功统计）。

### 7.1 重采双机 Hard 专家数据

```bash
python train_pursuit_matd3.py collect-transitions \
  --env-config configs/pursuit_v2/radar5_dual_hard.json \
  --teacher mpc --demo-seed 2000000 --rollouts 200 \
  --dataset outputs/radar5_dual_mpc_transitions200.pt
```

### 7.2 Hard 混合训练（Nominal/Medium/Hard 循环；验证用 Hard）

A5 / HGAT-MATD3-Opt：

```bash
python train_pursuit_matd3.py train \
  --mix-env-configs \
    configs/pursuit_v2/radar5_dual_nominal.json \
    configs/pursuit_v2/radar5_dual_medium.json \
    configs/pursuit_v2/radar5_dual_hard.json \
  --actor-init outputs/radar5_dagger/best_imitation.pt \
  --prior outputs/radar5_dual_mpc_transitions200.pt \
  --prior-fraction 0.5 --warmup-policy actor \
  --critic-pretrain-updates 5000 \
  --demo-bc-weight 1.0 --demo-bc-final-weight 0.05 --demo-bc-decay-steps 150000 \
  --episodes 3000 --seed 11 \
  --validation-seed 4000000 --validation-episodes 50 --validation-interval 100 \
  --step-checkpoint-interval 10000 \
  --out outputs/radar5_dual_hgat_matd3_opt_mix_seed11
```

MAPPO（同样 Hard 混合；若无双机 DAgger，可先用 plain 或旧 warm-start 权重试跑）：

```bash
python train_pursuit_with_demos.py train \
  --training-profile plain \
  --mix-env-configs \
    configs/pursuit_v2/radar5_dual_nominal.json \
    configs/pursuit_v2/radar5_dual_medium.json \
    configs/pursuit_v2/radar5_dual_hard.json \
  --bc-updates 0 --episodes 3000 --seed 11 \
  --validation-seed 4000000 --validation-episodes 50 --validation-interval 100 \
  --out outputs/radar5_dual_mappo_mix_seed11
```

`--mix-env-configs` 按 episode 轮转；`factory()` / 验证 / checkpoint 的 primary 环境是**列表最后一项（Hard）**。

### 7.3 四档评估：MPC / MAPPO / A5

```bash
for DIFF in nominal medium hard extreme; do
  python evaluate_3d_baselines.py \
    --env-config configs/pursuit_v2/radar5_dual_${DIFF}.json \
    --methods apf frpn mpc --seed-start 7000000 --seed-count 200 --workers 8 \
    --out outputs/dual_benchmark_${DIFF}_baselines_200.json
done

for DIFF in nominal medium hard extreme; do
  python train_pursuit_matd3.py evaluate \
    --checkpoint outputs/radar5_dual_hgat_matd3_opt_mix_seed11/best_capture.pt \
    --eval-env-config configs/pursuit_v2/radar5_dual_${DIFF}.json \
    --eval-seed 7000000 --eval-episodes 200 --methods matd3 \
    --out outputs/eval_dual_hgat_matd3_opt_${DIFF}
  python train_pursuit_with_demos.py evaluate \
    --checkpoint outputs/radar5_dual_mappo_mix_seed11/best_capture.pt \
    --eval-env-config configs/pursuit_v2/radar5_dual_${DIFF}.json \
    --eval-seed 7000000 --eval-episodes 200 --methods mappo \
    --out outputs/eval_dual_mappo_${DIFF}
done

python compare_pursuit_difficulty.py \
  --input Nominal:MPC=outputs/dual_benchmark_nominal_baselines_200.json \
  --input Medium:MPC=outputs/dual_benchmark_medium_baselines_200.json \
  --input Hard:MPC=outputs/dual_benchmark_hard_baselines_200.json \
  --input Extreme:MPC=outputs/dual_benchmark_extreme_baselines_200.json \
  --input Nominal:MAPPO=outputs/eval_dual_mappo_nominal/evaluation.json \
  --input Medium:MAPPO=outputs/eval_dual_mappo_medium/evaluation.json \
  --input Hard:MAPPO=outputs/eval_dual_mappo_hard/evaluation.json \
  --input Extreme:MAPPO=outputs/eval_dual_mappo_extreme/evaluation.json \
  --input Nominal:HGAT_MATD3_OPT=outputs/eval_dual_hgat_matd3_opt_nominal/evaluation.json \
  --input Medium:HGAT_MATD3_OPT=outputs/eval_dual_hgat_matd3_opt_medium/evaluation.json \
  --input Hard:HGAT_MATD3_OPT=outputs/eval_dual_hgat_matd3_opt_hard/evaluation.json \
  --input Extreme:HGAT_MATD3_OPT=outputs/eval_dual_hgat_matd3_opt_extreme/evaluation.json \
  --require-methods MPC MAPPO HGAT_MATD3_OPT \
  --out-dir outputs/radar5_dual_difficulty_comparison
```

建议先用 `--seed-count 40` / `--eval-episodes 40` 做校准冒烟，确认 Dual Extreme 上 MPC
捕获率明显低于 Dual Nominal，再跑满 200。
