> 本文的 MID-360 视场配置现在用于单独的感知验证。围捕主实验默认持续带噪声观测与主动逃逸，参见 [当前主实验指南](PURSUIT_DEMO_GUIDE.zh-CN.md)。主实验不要加载雷达配置文件；这不是雷达探测性能保证。

# MID-360 与速度/偏航角速率围捕训练

本配置对应用户确认的实机接口：`vx, vy, vz, yaw_rate`。策略不输出推力或力矩，姿态和推力由实机飞控处理。搜索环境、搜索奖励和二维搜索融合路径不变。

## 控制与加速

旧策略原本就输出速度/偏航角速率参考，昂贵的是环境内部推力/力矩 NMPC 的大量候选轨迹积分。

- 围捕只有一套执行链：速度与偏航角速率指令 → 带响应滞后、加速度和角速率限制的简化飞控。内层推力/力矩 NMPC、候选积分器及其参数已删除。
- `QuadrotorPursuitConfig()` 默认就是 MID-360；没有 `execution_mode` / `sensor_mode` 开关，不需要指定配置文件启用。
- 这是低阶闭环模型。姿态与平移响应近似，基准必须在相同环境下重新评估，不能直接比较旧 NMPC/FOV 的捕获率。
- 三维六状态 CI 融合保留权重网格和行列式目标，只对胜出的信息矩阵求逆。二维搜索的原计算分支保留。
- 日志新增 `env=...s ppo=...s`，结果记录 `environment_seconds`、`ppo_update_seconds`。回合总时间还包含策略推理、图构造、验证和保存。

坐标约定：仿真使用 z 向上世界坐标和 FLU 机体系；速度为世界系速度，yaw_rate 为偏航角变化率，不是倾斜时的机体 z 角速率。接 PX4 时必须明确本地地图与 ENU/NED 的对齐。如果地图采用 ENU，转 NED 时速度为 `[vy, vx, -vz]`，偏航角速率取反。这里没有实现或启动 ROS/PX4 实机发布器。

## MID-360 参数与限制

根据 [Livox 官方规格](https://www.livoxtech.com/mid-360/specs)：水平视场 360°，垂直视场 **−7° 至 +52°**，典型帧率 10 Hz，近距盲区 0.1 m；0.1–0.2 m 数据精度不保证。本配置使用 0.2 m 作为可用近距阈值，而非宣称硬件盲区是 0.2 m。

`configs/pursuit_velocity_lidar.json` 已设置对应视场和帧率，并关闭 `pursuit_target_observable`。雷达模式不允许重新开启全局可见绕过。安装姿态、无人机姿态、距离、视场、建筑遮挡、扫描时刻及漏检共同决定测量；漏检时只预测或融合队友航迹，不补发目标真值测量。

下面是**待实测标定的建模参数，不是 MID-360 的官方性能保证**：

- `lidar_target_detection_range=12`：小型被围捕目标可被检测/关联的距离。官方对不同反射率目标的 40/70 m 测距规格不能直接作为小型无人机可靠识别距离。
- `lidar_detection_probability=0.9`、位置噪声 0.08 m 和随距离增长项：点云检测中心的模拟误差，不等于单个激光点测距精度。
- `lidar_mount_xyz`、`lidar_mount_rpy_deg`：暂为零，表示与机体参考点重合且正装；需要填写实际安装外参。
- 速度响应 0.35 s、姿态响应 0.2 s、加速度和倾角上限：需要用装载雷达和捕获装置后的飞行日志标定。

这是**点云检测结果级仿真**，没有生成每秒 20 万个点，也没有实现原始点云目标分类、数据关联、机体自遮挡、扫描畸变及处理延迟。已知目标的测量关联在仿真中是理想化的。实机还需要点云预处理、目标检测/关联、带时间戳跟踪，以及飞机自身定位和坐标变换。现有建筑安全检查与图特征仍假设已知地图；不应将其表述为仅靠实时雷达完成未知地图避障。

默认动作周期为 0.2 s（5 Hz），每个策略时刻最多使用一次最新扫描测量，不逐帧处理全部 10 Hz 点云；低于策略频率的雷达会按设定周期更新，重复读取观测不重复抽样。`handoff_initial_track=true` 保留搜索阶段提供的初始噪声航迹，不代表随后持续接收外部目标位置。

MID-360 正装时向下仅有 7° 视场。下挂捕获网接近目标时可能产生感知盲区，应依靠队友保持观测、短时预测，或根据实际结构评估安装姿态/补充传感器。配置支持外参调整，但没有擅自扩大垂直视场或自动改变安装角度。

## VSCode 终端命令

在项目根目录、原来的 torch 环境中运行。以下命令不会被本次代码修改自动执行。

先做短程环境计时，不训练：

```bash
python repro/benchmark_pursuit_speed.py --steps 8 --out outputs/mid360_speed.json
```

新控制/感知配置与旧示范的环境不同，使用新文件采集，避免覆盖已有成果。先收集 5 条成功轨迹：

```bash
python repro/train_pursuit_with_demos.py collect --demo-successes 5 --demo-attempts 30 --demos outputs/pursuit_mid360_demos.pt
```

用新配置训练：

```bash
python repro/train_pursuit_with_demos.py train --demos outputs/pursuit_mid360_demos.pt --bc-updates 1000 --demo-weight 0.1 --episodes 1000 --validation-interval 50 --validation-episodes 5 --seed 11 --out outputs/mid360_seed11
```

按检查点记录的新环境评估全部基准：

```bash
python repro/train_pursuit_with_demos.py evaluate --checkpoint outputs/mid360_seed11/best_capture.pt --eval-episodes 100 --out outputs/mid360_seed11/test
```

训练完成后的 `index.html` 展示学习曲线和围捕回放。回放蓝线表示安装坐标系下的雷达视场边界，不是真实点云；结果用 `controller_feasible_rate`，表示动作通过边界和障碍检查的比例，不代表优化器求解成功。

旧 NMPC/FOV 检查点与示范现在会被明确拒绝，不会自动改写环境后继续训练。文件不会被删除。正在运行的 Python 进程不会热加载修改。

围捕统一使用 `train_pursuit_with_demos.py`；两个原通用训练入口只保留搜索任务，搜索算法和环境不变。多种子围捕可以在终端依次运行该入口，分别设置 `--seed` 与 `--out`。

30 小时可训练的回合数需要由本机端到端计时估算：约 `108000 / [每个训练回合秒数 + 每50回合验证总秒数/50]`，另外扣除预训练与初始验证时间。环境计时不包含 PPO、推理和保存；本次只做短测，不启动正式训练。
