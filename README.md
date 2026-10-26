# UAV Swarm — Radar5 Pursuit (Isaac PhysX → PX4)

Multi-UAV 3-vs-1 quadrotor pursuit with shared dual-capture geometry, Isaac
PhysX closed-loop training, and a PX4 / MAVROS offboard path for SITL and
real flight.

## Scope

| Layer | Role |
|-------|------|
| **Task** | Radar5 KF–CI tracks, dual-capture reward v5, HGAT-MATD3 / MAPPO+ |
| **Physics** | `physx_closed`: 3 pursuers + 1 evader as PhysX articulations (Lee tracking) |
| **Sensing (sim)** | Geometric MID-360 + LOS by default (`--no-native-lidar`); native LiDAR optional |
| **Deploy** | 5 Hz world velocity + yaw-rate → MAVROS `setpoint_raw/local` (ENU↔NED) |

Hybrid `isaac_dual_postlc` results are **baselines only** — do not mix with
`physx_closed` or real-flight tables. See [`SIM2REAL.md`](SIM2REAL.md).

## Repository layout

```
configs/pursuit_v2/     # difficulty ladder + physx_closed JSON
pursuit/                # algorithms, trainers, datasets
isaac_pursuit_backend.py
train_isaac_pursuit.py / evaluate_isaac_pursuit.py
collect_isaac_pursuit_data.py / pretrain_isaac_pursuit_actor.py
queue_physx_closed_*.sh / run_physx_closed_{train,eval}.sh
px4_offboard_bridge.py / residual_policy.py / capture_geometry.py
real_flight_{obs,safety,node}.py
```

Training artifacts (`.pt`, logs) live under `$UAV_PROJECT_RUNS` (default
`/srv/data/projects/kyy/runs`) and are **not** committed.

## Quick start

```bash
# PhysX-closed collect → DAgger → MATD3-A6 + MAPPO+ → ladder eval
bash queue_physx_closed_mpc_pipeline.sh

# Offline real-flight stack smoke (no ROS)
python real_flight_node.py --dry-run --duration 4 --track-xyz 0,0,2
python -m unittest test_real_flight_stack.py test_px4_offboard_bridge.py
```

Real-flight ladder: [`REAL_FLIGHT_CHECKLIST.md`](REAL_FLIGHT_CHECKLIST.md).

## Contracts

- **Actions**: `(n_agents, 4)` in `[-1, 1]` → world \(v_x,v_y,v_z\), yaw rate @ **5 Hz**
- **Capture**: ≥ `capture_required_uavs` (default 2) inside `capture_radius` — [`capture_geometry.py`](capture_geometry.py)
- **Obs (real)**: odometry + fused tracks only; never simulator target GT

## License / notes

Research code for the Radar5 pursuit experiments. Isaac Sim / OmniDrones
runtime paths are environment-specific; adjust `ISAACSIM_PATH` and conda env
in the `run_*.sh` / `queue_*.sh` scripts before launching.
