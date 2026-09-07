# 围捕场景 v2：更快目标、随机初始布局和预测逃逸

本版本只修改围捕任务。搜索环境不变，不运行正式训练。所有运动参数均为**待实测标定的仿真假设**，并非已验证的实机指标。

## 文献依据及未照搬的部分

- [OPEN，III-A、IV-C、V-A](https://arxiv.org/html/2409.15866v3)：研究三架追捕机追捕更快目标，实验中追捕机 1.0 m/s、目标 1.3 m/s；随机化场景并标定动力学。这里参考速度比与场景多样性，保留本项目的速度/偏航角速率接口。本文的目标预测模块增强的是追捕方，不能据此称本项目的预测逃逸为 OPEN 复现。
- [DualCL，III-A、V-D](https://arxiv.org/html/2312.12255v2)：随机生成双方初始位置，启发式基准在训练场景上调参。其捕获半径课程不适用于本项目的固定捕获装置，因此不采用。
- [Decentralized Multi-Agent Pursuit，III、V-A](https://arxiv.org/html/2010.08193v2)：使用无噪声无遮挡位置观测，并按目标/追捕机速度比分组评估。这里有外部定位/目标广播，因此保留持续带噪声位置观测；不把它称为雷达物理可见率。

## 当前默认值

| 参数 | 追捕机 | 目标 |
|---|---:|---:|
| 水平速度上限 | 1.40 m/s | 1.82 m/s |
| 垂直速度上限 | 0.80 m/s | 1.04 m/s |
| 水平加速度上限 | 2.50 m/s² | 2.50 m/s² |
| 垂直加速度上限 | 2.00 m/s² | 2.00 m/s² |
| 速度响应时间常数 | 0.35 s | 0.35 s |

目标速度上限为追捕机的 1.3 倍；从静止加速、转向或避障时，实际速度不保证每一刻都更快。没有加入未经测量的纯通信延迟。目标控制器使用一阶响应与加速度限制；追捕方继续使用原速度响应模型。实际参数应由带载飞行日志替换。

初始布局按种子选择同侧追击、交会、随机分散三种之一；追捕机与目标初始三维距离为 8–16 m。随机化朝向及高度差，检查场地边界、建筑和机间间隔，双方从静止开始。并不保证初始形成包围。狭小地图或过大距离导致无法采样时显式报错，不偷偷回退到近距离三角形。

捕获网半径 1.0 m、下挂距离 0.45 m、轴向容差 0.30 m、任意一架有效网接触即成功，均不变。这些是现有配置值，并非本次重新测量的装置尺寸。

## 逃逸与信息一致性

目标根据带噪声的追捕机位置历史估计速度，检查约 1.2 s（6×0.2 s）候选轨迹上的接近风险，预测追捕机匀速运动和网中心位置。它不知道追捕机将要选择的动作，也没有读取 MAPPO 权重。预测器是本项目规则设计，不是论文算法的严格复现。候选轨迹包含停止指令；残余地图碰撞由停止保护处理，单独记录 `evader_safety_interventions`。保护触发是简化模型局限，应在正式统计中检查，不能视为正常加速度受限机动。

MPC 的每架无人机用本机融合目标航迹分配角色，不能直接汇总其他无人机的私有目标估计。策略输入补充同样的已知建筑地图和经通信可用的队友位置/速度，围捕特征不再使用搜索模块中的目标真值距离统计。默认全通信；弱通信不应声称与默认实验具有同等信息条件。

APF 接近点补偿下挂网高度，编队偏置改为 0.5 m，使其目标与网捕判定一致。这是公开的任务适配，不是为了削弱基准；基准控制增益仍须在独立验证集上合理调参。

## 参数分组

`configs/pursuit_v2/` 提供 nominal、speed_ratio_1_1、speed_ratio_1_5、slower_response、lower_acceleration 五组仿真配置。后两组分别只改变追捕机响应、加速度，避免混淆因素。所有组都保持目标速度上限更高，捕获装置不变。训练和示范采集必须传入同一配置；评估默认恢复检查点记录的配置，不支持通过 `--env-config` 静默替换。跨组性能需要分别训练/评估，目前没有实现冻结同一检查点的跨配置泛化入口。

## 在 Linux VSCode 终端运行

同步代码后在项目根目录的 torch 环境运行，以下命令全部为单行。旧示范/检查点由于输入结构和动力学变化被拒绝；不删除旧文件。

```bash
python repro/evaluate_3d_baselines.py --methods apf frpn mpc --seeds $(seq 3000000 3000099) --steps 300 --workers 4 --out outputs/pursuit_v2_baselines_100.json
```

```bash
python repro/train_pursuit_with_demos.py collect --teacher mpc --demo-successes 10 --demo-attempts 100 --demos outputs/pursuit_game_v2_demos.pt
```

```bash
python repro/train_pursuit_with_demos.py train --demos outputs/pursuit_game_v2_demos.pt --bc-updates 1000 --demo-weight 0.1 --episodes 1000 --validation-interval 50 --validation-episodes 5 --torch-threads 1 --seed 11 --out outputs/pursuit_v2_seed11
```

```bash
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/pursuit_v2_seed11/best_capture.pt --methods mappo apf frpn mpc --eval-seed 3000000 --eval-episodes 100 --out outputs/pursuit_v2_seed11/test
```

保留奖励/捕获率双轴图 `reward_capture.svg`、六宫格诊断、三维回放。基准逐回合 JSON/CSV 和统一评估 JSON 增加初始布局、距离及目标保护次数。短测只验证代码和约束，不用其捕获率决定是否“足够难”，也不能据此保证 MAPPO 超过基准。
