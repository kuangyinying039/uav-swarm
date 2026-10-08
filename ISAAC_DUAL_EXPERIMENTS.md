# Isaac 双机捕获迁移实验

本协议是独立于旧运动学实验的新基准。学习方法只在 **Dual Medium** 上训练，每个训练 seed 产生一个 checkpoint；Nominal、Medium、Hard、Extreme 仅加载该 checkpoint 测试，不在测试档微调或续训。协议细节以 `isaac_dual_experiment_manifest.json` 为准。

## 固定设置

- 环境：3 架 Hummingbird，1 个目标，5 栋建筑；双机进入 1 m 捕获球后成功。
- 决策频率：5 Hz，`decision_dt=0.2 s`；物理步 `physics_dt=0.01 s`；每回合最多 300 决策步，即 60 仿真秒。
- 正式训练：每个算法/消融 3000 回合，training seeds 11、12、13。
- 四档测试：所有方法使用相同的 seed 7000000–7000199；不改变 checkpoint。
- 几何 LiDAR fallback 默认开启。Isaac native LiDAR 暂不用于该基准。
- 旧单机捕获 v4 checkpoint、运动学 checkpoint、运动学 prior 不允许用于本协议。

## 环境启动

以下命令先设置为用户自己的可写 Isaac clone 环境。不要再 source 项目里会切回只读环境的 `activate.sh`。

```bash
unset PYTHONPATH LD_LIBRARY_PATH CARB_APP_PATH EXP_PATH ISAAC_PATH ISAACSIM_PATH PYTHONSTARTUP
source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "$HOME/.conda/envs/omnidrones-isaac41-writable"
export ISAACSIM_PATH=/opt/lab/apps/isaac-sim-4.1.0
export OMNIDRONES_SOURCE=/opt/lab/src/OmniDrones-isaac41
source "$ISAACSIM_PATH/setup_conda_env.sh"
cd /srv/data/projects/kyy/xieton/repro
```

先做轻量 smoke：

```bash
python train_isaac_pursuit.py mappo \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --out "$HOME/runs/isaac_dual_smoke" \
  --episodes 1 --max-steps 3 --seed 11
```

只有确认 `final.pt` 和 `history.json` 正常生成后，才开始正式数据收集/训练。

## PhysX 专家数据和 DAgger 初始化

MATD3 A2–A5 和 MAPPO DAgger 使用此 Isaac 内收集的 MPC 专家轨迹；不能传入旧运动学 prior。A4 的 prior 需在 no-visibility Medium 配置中单独采集。

```bash
python collect_isaac_pursuit_data.py \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --teacher mpc --demo-seed 2000000 --rollouts 200 \
  --dataset "$HOME/runs/isaac_dual_data/medium_mpc_prior.pt"

python pretrain_isaac_pursuit_actor.py \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --seed 11 --demo-seed 2000000 \
  --bc-episodes 50 --bc-updates 1500 \
  --dagger-rounds 2 --dagger-episodes 40 --dagger-updates 800 \
  --out "$HOME/runs/isaac_dual_data/dagger_seed11"

python collect_isaac_pursuit_data.py \
  --env-config configs/pursuit_v2/radar5_dual_ablation_no_visibility_medium.json \
  --teacher mpc --demo-seed 2100000 --rollouts 200 \
  --dataset "$HOME/runs/isaac_dual_data/medium_no_visibility_prior.pt"
```

每个 MATD3 seed 应用独立 seed 生成 actor initialization（至少 BC/DAgger RNG 不同）；可复用固定 MPC prior，但必须记录数据集 checksum、PhysX 后端标记、环境配置和 teacher seed。对严格统计，建议也按训练 seed 独立重采 prior。

## 训练方法

A0：MATD3-MLP scratch。A1：HGAT-MATD3 scratch。A2：Isaac DAgger actor + 50% Isaac prior，不使用 BC 和 critic 预训练。A3：A2 加衰减 BC，不做 critic 预训练。A4：A5 但关掉 visibility reward，并使用对应 no-visibility PhysX prior。A5：完整 HGAT-MATD3，含 critic 预训练和衰减 BC。

先跑一个 seed 的 smoke，可将正式参数覆盖为 `--episodes 1 --max-steps 3`、减小 batch；这只是执行检查，不能作为算法结果。正式命令例子（seed 11）：

```bash
python train_isaac_pursuit.py mappo \
  --variant mappo_dagger \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --actor-init "$HOME/runs/isaac_dual_data/dagger_seed11/best_imitation.pt" \
  --demos "$HOME/runs/isaac_dual_data/dagger_seed11/expert_episodes.pt" \
  --aux-demos "$HOME/runs/isaac_dual_data/dagger_seed11/correction_episodes.pt" \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_mappo_dagger_seed11"

python train_isaac_pursuit.py matd3 --variant A0 \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A0_seed11"

python train_isaac_pursuit.py matd3 --variant A1 \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A1_seed11"

python train_isaac_pursuit.py matd3 --variant A2 \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --actor-init "$HOME/runs/isaac_dual_data/dagger_seed11/best_imitation.pt" \
  --prior "$HOME/runs/isaac_dual_data/medium_mpc_prior.pt" \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A2_seed11"

python train_isaac_pursuit.py matd3 --variant A3 \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --actor-init "$HOME/runs/isaac_dual_data/dagger_seed11/best_imitation.pt" \
  --prior "$HOME/runs/isaac_dual_data/medium_mpc_prior.pt" \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A3_seed11"

python train_isaac_pursuit.py matd3 --variant A4 \
  --env-config configs/pursuit_v2/radar5_dual_ablation_no_visibility_medium.json \
  --actor-init "$HOME/runs/isaac_dual_data/dagger_seed11/best_imitation.pt" \
  --prior "$HOME/runs/isaac_dual_data/medium_no_visibility_prior.pt" \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A4_seed11"

python train_isaac_pursuit.py matd3 --variant A5 \
  --env-config configs/pursuit_v2/radar5_dual_medium.json \
  --actor-init "$HOME/runs/isaac_dual_data/dagger_seed11/best_imitation.pt" \
  --prior "$HOME/runs/isaac_dual_data/medium_mpc_prior.pt" \
  --episodes 3000 --seed 11 --batch-size 256 --hidden-dim 128 \
  --out "$HOME/runs/isaac_dual_A5_seed11"
```

对上面每个方法，把 seed 改成 12、13 并使用完全独立的输出目录和 DAgger 初始化目录。当前训练器是**一个 Isaac 场景串行采样**，不是 batched Isaac；3000 回合 × 300 步约 90 万决策步/seed，先测吞吐与长期稳定性再提交三个 seed 全量任务。

## 四档 frozen-checkpoint 测试

每个 checkpoint 单独调用一次。默认测试 200 个共同 seeds，并在同一 Isaac App 里依次跑四档；评估脚本不会加载 optimizer 或执行训练更新，也会拒绝未标记 `isaac_physx` 的旧 checkpoint。

```bash
python evaluate_isaac_pursuit.py --method mappo \
  --checkpoint "$HOME/runs/isaac_dual_mappo_dagger_seed11/final.pt" \
  --seed-start 7000000 --seed-count 200 \
  --out "$HOME/runs/isaac_dual_eval_mappo_seed11"

python evaluate_isaac_pursuit.py --method matd3 \
  --checkpoint "$HOME/runs/isaac_dual_A5_seed11/final.pt" \
  --seed-start 7000000 --seed-count 200 \
  --out "$HOME/runs/isaac_dual_eval_A5_seed11"

for METHOD in apf frpn mpc; do
  python evaluate_isaac_pursuit.py --method "$METHOD" \
    --seed-start 7000000 --seed-count 200 \
    --out "$HOME/runs/isaac_dual_eval_${METHOD}"
done
```

四档结果分别在输出目录的 `nominal.json`、`medium.json`、`hard.json`、`extreme.json`，同目录 `summary.json`/`summary.csv` 汇总；每档包含 Wilson 95% capture-rate 区间。正式论文统计还应按 3 个训练 seeds 与 200 个测试场景做分层 bootstrap，不能把所有轨迹当作独立样本。
