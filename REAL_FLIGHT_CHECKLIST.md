# 真机实验阶梯（PX4 offboard + 同型号逃逸机）

与仿真共用 [`capture_geometry.py`](capture_geometry.py) 与 [`SIM2REAL.md`](SIM2REAL.md)。  
**禁止**把 `isaac_dual_postlc`（hybrid）数字与本清单结果混表。

## 前置

- [ ] 4 架同型号机架固件 PX4，MAVROS 命名空间 `/uav0`…`/uav3`（0–2 追，3 逃）
- [ ] Offboard 已解锁；地理围栏与遥控接管就绪
- [ ] SITL `run_px4_sitl_encircle.sh` 静止合围通过
- [ ] 相对定位（RTK / UWB / 视觉）误差已标定并写入日志 metadata

## 阶梯

### 1. 单机速度跟踪
- 脚本：`python px4_offboard_bridge.py` 交互或 `spin_setpoint_rate` 圆周
- 通过标准：水平速度跟踪 RMSE < 0.25 m/s，无触发 failsafe

### 2. 双机相对定位
- `/uav0` + `/uav1` 保持 4–8 m；记录通信时延与相对误差
- 通过标准：相对水平 RMSE < 0.4 m（或与捕获半径匹配的预算）

### 3. 三追一逃 — 静止合围
- 逃逸机悬停；追捕策略 = FRPN/MPC 速度 + 可选 RL 残差（默认残差增益 0.3）
- 成功：`FleetOffboardBridge.evaluate_capture_now()` 双机捕获定义
- 记录：capture、kth_gap、min_distance（诊断）、相对定位误差

### 4. 慢速脚本逃逸
- 逃逸按折线 / 圆周 ≤ 0.6×`target_speed`
- 同上捕获定义；至少 20 局报告 capture_rate

### 5. 全策略逃逸
- 逃逸也跑 occlusion / 对抗策略；全场最高难度仅在 3+4 通过后进行

## 日志字段（每局 JSONL）

```json
{
  "tag": "real",
  "protocol": "physx_closed_capture_v1",
  "capture_success": false,
  "close_count": 0,
  "angular_span_deg": 0.0,
  "kth_gap": 1.2,
  "min_distance": 0.8,
  "relative_localization_rmse_m": 0.25,
  "residual_gain": 0.3,
  "physics_recovered": false
}
```

## 安全

- 任何单机 `min_distance` 诊断接近 0 但未达双机捕获时，**不得**宣称任务成功
- PhysX corrupt / failsafe → 截断，不计成功
- 在线节点默认启用 [`real_flight_safety.py`](real_flight_safety.py)：里程计超时、同伴过近、高度越界、水平围栏 → **速度置零悬停**
- 决策时钟固定 **5 Hz**（`DecisionClock` / `decision_dt=0.2`），与训练一致

## 软件入口

| 步骤 | 命令 |
|------|------|
| 离线冒烟 | `python real_flight_node.py --dry-run --duration 4 --track-xyz 0,0,2` |
| 单测 | `python -m unittest test_real_flight_stack.py test_px4_offboard_bridge.py` |
| SITL 合围 | `bash run_px4_sitl_encircle.sh` 通过后，再挂 MAVROS 跑 `real_flight_node.py` |
| 残差首飞 | `--teacher frpn --checkpoint <physx_closed ckpt> --residual-gain 0.3` |
