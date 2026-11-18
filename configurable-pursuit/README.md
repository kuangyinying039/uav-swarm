# 并行无人机追逃环境

用于在 Isaac Sim 中搭建并行无人机导航与追逃任务：一架逃逸机 E、可配置数量的追捕机 P、静态障碍 S 和背景动态障碍 D。环境负责物理、场景、观测、奖励与终止；使用者通过外部策略控制 E，或直接控制全部飞机。

## 三步快速开始

在 Bash 中进入本项目目录。服务器需已安装 Isaac Sim 4.1、CUDA 兼容驱动和 OmniDrones。

```bash
# 1. 复制到个人工作目录；缓存、配置与结果在自己的副本产生
mkdir -p "$HOME/work"
cp -a /srv/data/projects/shared/PEG-zht/environments/configurable-pursuit "$HOME/work/configurable-pursuit"
cd "$HOME/work/configurable-pursuit"
# 2. 在个人副本激活依赖
source ./activate.sh
# 3. GPU 无其他计算任务时，执行并行环境短程检查
python -m parallel.examples.run --mode check --num-envs 1024 --steps 12 \
  --output runtime/first-check
```

`runtime/first-check/result.json` 中 `status=passed` 表示这次短程检查通过；失败查看同目录 `failure.txt`。输出目录必须是新目录。此命令执行固定动作与重置检查，不训练模型。不要直接在公共副本生成缓存或改配置；多用户使用 GPU 前先协调。

## 目录导航

|位置|内容|
|---|---|
|`activate.sh`|选择已安装的依赖环境，不安装软件|
|`parallel/config/environment.json`|物理、任务、奖励、场景和观测|
|`parallel/config/pursuer.json`|P 数量、出生参数、四种规则策略及采样概率|
|`parallel/config/visualization.json`|颜色、相机、显示与录像|
|`parallel/env/`|数量布局、采样、物理、观测、奖励、安全与公开接口|
|`parallel/strategies/`|四种独立 P 策略|
|`parallel/examples/run.py`|资源预检、短程运行与吞吐检查|
|`parallel/examples/demo.py`|单局真实物理演示与相机录屏|
|`parallel/examples/evader_baselines.py`|用于演示的 hold、goal、goal_avoid 规则动作|
|`parallel/tools/runtime_entry.py`|子进程退出码与结果完整性检查|
|`SHA256SUMS`|发布文件校验清单|

## 依赖与启动

需要 Linux、NVIDIA GPU、Isaac Sim 4.1、OmniDrones 对应源码与运行依赖（Python、PyTorch、OmegaConf 等）。物理后端要求 `cuda:0` 与 GPU pipeline；CPU 几何规则可以单独导入，但不替代真实物理仿真。录像另需 ffmpeg 的 libx264 编码器。

`activate.sh` 自动选择服务器 `/opt/lab` 或当前用户已安装的兼容部署。其他安装位置可显式指定：

```bash
export OMNIDRONES_ACTIVATE=/path/to/activate-omnidrones.sh
export OMNIDRONES_SOURCE=/path/to/OmniDrones
source ./activate.sh
```

激活脚本必须设置相应 Python、Isaac 和依赖路径。`OMNIDRONES_SOURCE` 中应存在 `examples/demo.yaml`。启动入口也支持 `--omnidrones-source`。物理运行的可写缓存位于个人副本的 `runtime/isaac-kit/`；不同项目不要并发共享同一缓存。

## 三份配置

配置加载器会校验完整结构、参数类型及未知字段。建议从默认 JSON 复制后修改，通过参数传入，保留公共默认值。

|配置|主要字段与语义|
|---|---|
|`environment.json` 的 `simulation`|Hummingbird、LeePositionController、dt、planar/free_3d、固定高度、E 最大速度、并行场景间距、headless|
|`environment.json` 的 `scene`|地图边界、E 半径、起终点、S 数量和半径、D 数量与速度、采样预算|
|`environment.json` 的 `task`|目标判定半径、最大回合步数、终止开关及物理异常阈值|
|`environment.json` 的 `reward`|各项开关与系数、距离/势能推进、停滞判定|
|`environment.json` 的 `observation`|P 观测字段/范围/槽数、激光范围、D 观测范围|
|`pursuer.json` 的 `spawn`|P 数量、半径、速度、出生区域及净空|
|`pursuer.json` 的 `common`|探测范围、责任半径、巡逻、搜索时长|
|`pursuer.json` 的 `strategies`|策略名称、权重与独立参数|
|`visualization.json`|障碍/标记 RGB、显示高度、相机位置/视角/分辨率、输出目录、fps|

默认 `dt=1/62` 秒、固定高度 1.5 米、E 最大速度 2 米/秒；回合上限 3200 步。dt 必须是 Isaac 能表达的整数频率倒数。默认奖励为到达 +80、被捕/各类碰撞/物理失败/超时 −20、距离推进系数 8，并含目标速度、逐步与停滞项；最终奖励由实际传入配置决定。外部算法应明确保存有效配置。

### 数量与观测容量

E 固定为 1；P、D 为非负整数，S 为正整数。P/D 数量在构造前确定，整个并行批次相同，不支持运行中增减或逐局不同数量。P=0 为无捕获导航，D=0 为无背景动态障碍。数量增加受地图采样密度、显存和计算资源限制，采样耗尽会明确失败。

飞机轴是 `[E,P]`，场景动态对象轴是 `[D,P]`。所有真实 P 参与捕获，所有真实 D 参与碰撞。观测容量与真实数量独立：P 默认按身份保留前三槽，超出槽数的 P 不可见；不足补零并设置 mask。D 保留最近三个可见对象的槽位，不随真实 D 数改变。

生成 P=5、D=7、P 观测槽=5 的配置：

```bash
mkdir -p config-local
python - <<'PY'
import json
from pathlib import Path
from parallel.env.config_loader import load_settings
from parallel.strategies.config import load_config
e = load_settings()
p = load_config()[0]
e['scene']['dynamic_obstacles']['count'] = 7
p['spawn']['count'] = 5
e['observation']['pursuers']['max_count'] = 5
for name, value in [('environment', e), ('pursuer', p)]:
    Path(f'config-local/{name}.json').write_text(json.dumps(value, indent=2))
PY
python -m parallel.examples.run --mode check --num-envs 1024 --steps 12 \
  --environment-config config-local/environment.json \
  --pursuer-config config-local/pursuer.json --output runtime/counts-check
```

### 四种 P 策略

每局重置时按权重为各 P 采样策略，默认 `direct_chase=0.3`、`cut_front=0.3`、`follow_pressure=0.1`、`choke_block=0.3`。这些是四种控制方法；巡逻/追击/搜索/返回则是状态机状态。

|名称|行为|
|---|---|
|`direct_chase`|朝 E 当前方向直接追击|
|`cut_front`|在 E 前方部署截前目标|
|`follow_pressure`|采用后侧/侧向偏移持续施压|
|`choke_block`|依据通道和路线部署堵截；没有适合通道时使用配置回退|

`sampling_seed` 控制策略采样；各 `params` 控制对应方法。`ScriptedPursuerWrapper` 使用这份配置，不覆盖为单一追击策略。安全层处理静态障碍与北侧边界；不保证 P-P 或 P-D 全面避碰。

## 输入输出与自己的算法

先启动 Isaac app，再导入物理环境。`ParallelEnv` 接收全部飞机的世界 XY 速度 `[N,1+P,2]`（米/秒）；`ScriptedPursuerWrapper` 将 P 交给规则策略，仅接收 E 的局部归一化动作 `[N,2]`。局部坐标系在每局起点生成：+x 沿“起点 → 终点”，+y 为其左侧；整个回合保持不变，不随 E 的位置或机体航向转动。wrapper 把动作限制到单位圆并按 E 最大速度映射到世界系；例如 `[1,0]` 表示沿该局固定前向以最大速度飞行，世界 XY 速度应传给底层 `ParallelEnv`。

```python
import torch
from omegaconf import OmegaConf
from parallel.env.dependencies import select_source
from parallel.env.config_loader import load_settings, configure_isaac_app
from parallel.strategies.config import load_config

settings = load_settings()  # 也可传 JSON 路径或完整字典
pursuer = load_config()[0]
source = select_source()
from omni_drones import init_simulation_app
cfg = OmegaConf.load(source / 'examples/demo.yaml')
configure_isaac_app(cfg, settings['simulation'])
app = init_simulation_app(cfg)
from parallel.env.od_env import ParallelEnv
from parallel.env.wrapper import ScriptedPursuerWrapper
base = ParallelEnv(cfg, seeds=range(1024),
                   environment_config=settings, pursuer_config=pursuer)
env = ScriptedPursuerWrapper(base)
try:
    obs = env.observe()
    # 将下面的零动作替换成自己的 policy(obs)。
    action = torch.zeros(env.n, 2, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    ids = (terminated | truncated).nonzero().flatten()
    if ids.numel():
        obs = env.reset(ids)
finally:
    env.close()
    app.close()
```

|接口|返回或约束|
|---|---|
|`observe()`|浮点观测 `[N,O]`；O 随观测配置变化|
|`observation_layout` / `observation_scale`|分段字段、索引、缩放定义；不要写死观测宽度|
|`step(action)`|obs、reward `[N]`、terminated `[N]`、truncated `[N]`、info|
|`state()` / `state_spec`|`[N,1+P,13]`：位置3、wxyz四元数4、线速度3、角速度3；扣除并行场景偏移|
|`motor_state()`|`[N,1+P,4]` 旋翼状态|
|`reset(ids=None, resample=True, seeds=None)`|整批或 int64 索引子集重置；可指定 seed；`resample=False` 恢复该局起点场景|
|`info`|成功、捕获、各类碰撞、超时、物理异常、奖励分项等诊断|
|`close()`|释放环境引用，可重复调用；Isaac app 由调用方关闭|

`step` 不自动重置完成的回合；先消费终局信息，再重置完成索引。`resample=False` 是起点恢复，不承诺任意中途物理状态精确回放。更改观测槽位或字段时，自有模型的输入也必须相容。

## 演示与录像

```bash
# 单局真实物理，规则 goal_avoid 控制 E；不是训练模型
python -m parallel.examples.demo --headless --record --episodes 1 --seed 41
# 可见窗口；相机跟随 E、顶视图
python -m parallel.examples.demo --no-headless --view top --follow-evader
```

可加入 `--environment-config`、`--pursuer-config`、`--visualization-config`；`--max-steps` 可做短演示。录像默认写个人副本 `parallel/examples/render/` 下唯一子目录；每局 MP4 与有效配置/终局摘要一起保存。使用同一份 P 配置初始化场景与 wrapper，核对有效策略概率和实际策略分配，再解释录像。

公开示例视频位于项目根目录的 [`results/pursuit-showcase/`](../../results/pursuit-showcase/)。[overview.mp4 九宫格](../../results/pursuit-showcase/overview.mp4) 展示共 9 局：5 个成功、1 个环境碰撞、2 个被捕、1 个超时；同目录还有 9 个单局视频，可单独查看细节。案例选自已有场景并在真实环境重新运行，使用保存的物理状态经 Isaac 相机渲染；它们用于展示代表行为，不是随机成功率样本。P 的实际策略名与终局显示在画面中，说明见视频目录 README。

服务器上的完整观看路径：

```text
/srv/data/projects/shared/PEG-zht/results/pursuit-showcase/overview.mp4
```

把环境单独复制到个人目录后，视频相对链接可能不在副本内；仍可直接到上述公共路径观看，无需模型权重。环境示例中的规则 E 演示用于学习操作，视频中的外部训练策略表现不能由规则演示命令复现。

## 验证范围与限制

默认配置以及多组 P/D 数量已进行 CPU 场景/观测/重置检查和 1024 并行短程真实物理检查，包括 P/D 为零和 P=5、D=7 等组合。默认配置的场景、随机数、动态对象运动、重置与观测已核对一致性。短程检查覆盖有限采样与运行时间，不能推断任意数量、任意密度或长期训练都稳定。

固定高度平面模式使用真实刚体/旋翼动力学与理想高度约束，碰撞采用平面扫掠圆几何。自由三维可配置但尚无同等完整验收。大数量或高密度可能采样失败/显存不足；P 安全控制没有全面多体避碰保证。并行入口的 `check`/`bench` 是有界检查，限制 N≤1024，check≤1024 步、bench≤256 步；外部训练循环可按公开环境接口自行组织，不应把该检查入口当正式训练器。
