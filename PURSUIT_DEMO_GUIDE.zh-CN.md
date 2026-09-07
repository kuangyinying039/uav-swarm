> 最新场景为 v2：目标速度上限高于追捕机、随机初始布局和带响应限制的预测逃逸。请以 [v2 运行说明](PURSUIT_SCENARIO_V2.zh-CN.md) 为准，以下旧版命令和假设保留供历史参考。

> 主实验更新：默认采用持续带噪声的位置观测和主动逃逸目标，不再把 MID-360 漏检作为主实验难点。`pursuit_target_observable=true` 表示理想化的观测可用性，不是雷达物理直视。双方位置观测噪声默认标准差 0.10 m；目标根据带噪声的追捕机位置选择逃逸方向，避开建筑，不再以建筑遮挡收益为目标。基准和 MAPPO 使用相同配置。雷达参数文件仅用于单独的感知验证，不要给主实验传入它。
>
> 新数据使用 `outputs/pursuit_game_demos.pt`，新训练输出使用 `outputs/pursuit_game_seed11`。旧 MID-360 示范与新观测假设不同，采集和训练必须一致；旧检查点仍按其记录的配置运行，不会自动改成主实验。不要将新旧结果直接混比。

# MAPPO 围捕成功示范训练

目标是提高独立测试的捕获率。新增入口 `train_pursuit_with_demos.py`，搜索环境、搜索奖励、离散搜索策略不变。保持捕获网几何、逃逸策略、动力学、安全控制和基准控制器不变。

## 诊断

- 原连续策略直接预测 log std 后硬截断到 `[-5, -0.5]`。四维 Gaussian 的熵上限是 `4 × (0.5 × log(2πe) - 0.5) = 3.675754`，与截图 3.676 一致。处于上限之外的分支没有截断梯度，难以学会减小探索。
- 负回报不代表 PPO 没更新。现有围捕任务超时扣 50，每步扣 0.05，并有通信、碰撞和控制成本。训练末尾随机动作的 50 回合捕获率，不能直接等同于确定性策略的 100 回合独立测试。
- 日志 `losses` 原来是丢失目标次数，已在围捕分支改名为 `target_losses`，并显示近 50 回合捕获率。
- 截图 FRPN 的高回报与低捕获率并存。当前本地奖励代码已以捕获、超时为主要事件，旧图可能来自不同版本。必须重新评估，不能混用旧回报表。
- 当前环境成功条件是任意一架无人机的机体固定捕获网命中目标，不是三机同时达到角度条件。本次没有降低成功判定难度。

## 改动

1. MPC 专家采集完整成功轨迹，只读取已有局部融合航迹及共享队友状态。失败回合不作为成功示范；保留所有尝试的种子与结果。
2. 先行为克隆预训练，再在线 MAPPO。示范按回合均衡采样，动作标签为归一化的世界系速度与偏航角速率，与策略接口一致。
3. 在线前半程加入从 0.1 衰减到 0 的辅助模仿损失，减轻遗忘；PPO 概率比和 GAE 始终只使用本轮在线轨迹，不把专家经验伪装成 on-policy 数据。
4. 连续探索改为独立、可训练的四维平滑有界 log std，初始标准差 0.2。原始 MAPPO 的硬截断在新策略正常工作点内不再造成梯度死区。熵系数从 0.001 衰减至 0.0001，关闭依赖搜索覆盖率的不确定性熵放大。
5. `gamma=0.995`、`gae_lambda=0.97` 延长奖励传播。300 步后的终奖折扣从约 0.049 提高到 0.222。这是待消融的围捕训练设置，不是已证实的最优参数。
6. 每 50 回合用固定验证集评估，按“捕获率优先、含超时的平均步数次之”保存 `best_capture.pt`，同时保留 `pretrained.pt`、`latest.pt`、`final.pt`。测试集不得参与模型选择。
7. 结果只输出围捕、跟踪、安全和训练诊断；删除覆盖率、发现率、频谱、重访与搜索权重等结果列。失败捕获时间用 null，成功耗时与含超时回合长度分别展示。

当前采用独立的成功示范行为克隆与可选教师纠偏，随后进行 on-policy MAPPO 微调；示范不混入 PPO 回放。

## 运行

在仓库根目录使用已安装 PyTorch 和 NumPy 的 Python。正式实验推荐有 CUDA 的训练环境；本机新建的 `tmp/pursuit-venv` 仅用于 CPU 功能验证。

```bash
# 1. 原始环境收集 20 条成功示范，最多尝试 100 回合。
python repro/train_pursuit_with_demos.py collect --demo-successes 10 --demo-attempts 100 --demos outputs/pursuit_game_demos.pt

# 2. 成功先验预训练 + MAPPO 在线微调。
python repro/train_pursuit_with_demos.py train --demos outputs/pursuit_game_demos.pt --bc-updates 1000 --episodes 1000 --seed 11 --out outputs/pursuit_demo_seed11

# 3. 最佳模型与全部基准同配置、同 100 个未见种子测试。
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/pursuit_demo_seed11/best_capture.pt --eval-seed 3000000 --eval-episodes 100 --out outputs/pursuit_demo_seed11/test

# 4. 接续此新训练入口的检查点，episodes 是目标总回合数。
python repro/train_pursuit_with_demos.py train --checkpoint outputs/pursuit_demo_seed11/latest.pt --demos outputs/pursuit_game_demos.pt --episodes 1500 --seed 11 --out outputs/pursuit_demo_seed11
```

有不同环境参数时，用 `--env-config path/to/config.json` 指定 `QuadrotorPursuitConfig` 的 JSON 字段覆盖；采集和训练必须相同。评估自动读取检查点环境，不接受另行覆盖。默认示范种子从 2000000 起，训练为 `seed*100000 + episode`，验证从 4000000 起，测试从 3000000 起，入口检查集合交叠。

旧 MAPPO 检查点的连续策略结构不同，不能直接装入新策略继续训练；需要从新入口预训练。但 `evaluate` 支持旧的直接连续动作 MAPPO 检查点，方便用同一评估协议重新测量旧模型。旧检查点没有完整的种子清单时，需自行确认测试集未被训练使用。原搜索命令仍可使用。

## 输出

- `training.csv`、`history.json`：仅围捕结果及优化诊断。
- `summary.json`：末尾 50 回合训练捕获率，成功回合耗时，包含失败超时的平均长度。
- `capture_learning.svg`：捕获率、成功耗时、含超时长度、最终捕获网距离、碰撞、控制指令通过安全检查的比例六图。
- `index.html`：学习曲线和最后三回合三维围捕回放；对应 SVG 可用于报告。最后三回合不是挑选的成功案例。
- `validation.json`、`best_capture.pt`：验证记录与按捕获率选择的模型。
- 测试目录 `evaluation.json`：逐种子、逐方法的成功标志、步数、回报及奖励分项。
- 测试目录 `evaluation.svg`：捕获率及 95% Wilson 区间。区间反映测试种子采样的不确定性，不能代替多个训练种子的重复实验。

## 建议消融与验收

至少分别使用训练种子 11、59、83，并在相同未见测试种子上对比：原 MAPPO、新连续探索无示范（`--bc-updates 0 --demo-weight 0`）、仅预训练（`--demo-weight 0`）、预训练加衰减模仿（默认）。测试基准也必须通过新评估入口重跑。主要指标是捕获率，同时报告碰撞与含超时耗时，不能通过忽略失败回合得到更快的结论。

20 条示范是起始设置。如果 `pretrained.pt` 的独立验证捕获率仍明显低于 MPC，优先扩大到 50–100 条覆盖不同建筑、目标高度和接近方向的成功示范，增加预训练，并检查误差，而不是盲目延长低成功率 PPO 训练。若预训练已有较好捕获率而在线阶段下降，优先检查验证曲线、KL、模仿权重和探索标准差，再考虑更长的模仿保留期。

功能测试通过不代表捕获率已超过 62%。正式结果需要上述同协议、多训练种子评估。小样本端到端试跑只用于检查流程和输出。
