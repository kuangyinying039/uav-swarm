# UAV Swarm — Radar5 Pursuit (Isaac PhysX → PX4)

Multi-UAV 3-vs-1 quadrotor pursuit with shared dual-capture geometry, Isaac
PhysX closed-loop training, and a PX4 / MAVROS offboard path for SITL and
real flight.

## Read this first

| Document | Purpose |
|----------|---------|
| **[`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md)** | Current configs, algorithm stack, and **hybrid vs physx_closed** result tables (anti-confusion) |
| [`SIM2REAL.md`](SIM2REAL.md) | Action / obs / capture contract for PX4 |
| [`results/`](results/) | Frozen compact eval summaries (JSON) |
| [`REAL_FLIGHT_CHECKLIST.md`](REAL_FLIGHT_CHECKLIST.md) | Real-flight ladder |

**Do not mix** `isaac_dual_postlc` (hybrid baseline) capture rates with
`isaac_dual_physx_closed` (formal pure-PhysX) or real-flight numbers.

| Tag | Physics | Capture ball (`target_diameter`) | Role |
|-----|---------|----------------------------------:|------|
| `isaac_dual_postlc` | PhysX pursuers + **kinematic** evader | 0.5 | Historical baseline only |
| `isaac_dual_physx_closed` | **4× PhysX** articulations | **0.75** | Formal sim table |

## Scope

| Layer | Role |
|-------|------|
| **Task** | Radar5 KF–CI tracks, dual-capture reward v5, HGAT-MATD3 / MAPPO+ |
| **Physics** | `physx_closed`: 3 pursuers + 1 evader as PhysX articulations (Lee tracking) |
| **Sensing (sim)** | Geometric MID-360 + LOS by default (`--no-native-lidar`); native LiDAR optional |
| **Deploy** | 5 Hz world velocity + yaw-rate → MAVROS `setpoint_raw/local` (ENU↔NED) |

This repository keeps **simulation, training, evaluation, docs, and real-flight
deployment code**. Large checkpoints (`.pt`) stay under `$UAV_PROJECT_RUNS`.

## Repository layout

```
TECHNICAL_SPEC.md       # configs + methods + dual result tables
SIM2REAL.md
results/                # compact held-out capture summaries
configs/pursuit_v2/     # dual difficulty ladder (+ *_physx_closed, optional *_d05)
pursuit/                # algorithms, trainers, datasets
isaac_pursuit_backend.py
train_isaac_pursuit.py / evaluate_isaac_pursuit.py
collect_isaac_pursuit_data.py / pretrain_isaac_pursuit_actor.py
queue_physx_closed_*.sh / run_physx_closed_{train,eval}.sh
px4_offboard_bridge.py / residual_policy.py / capture_geometry.py
real_flight_{obs,safety,node}.py
scripts/                # closed-loop probes + extreme d=0.5 contrast
```

## Quick start

```bash
# PhysX-closed collect → DAgger → MATD3-A6 + MAPPO+ → ladder eval
bash queue_physx_closed_mpc_pipeline.sh

# Offline real-flight stack smoke (no ROS)
python real_flight_node.py --dry-run --duration 4 --track-xyz 0,0,2
python -m unittest test_real_flight_stack.py test_px4_offboard_bridge.py
```

## Contracts

- **Actions**: `(n_agents, 4)` in `[-1, 1]` → world \(v_x,v_y,v_z\), yaw rate @ **5 Hz**
- **Capture**: ≥ `capture_required_uavs` (default 2) inside `capture_radius` — [`capture_geometry.py`](capture_geometry.py)
- **Obs (real)**: odometry + fused tracks only; never simulator target GT

## Headline numbers (see TECHNICAL_SPEC for full matrices)

- **Hybrid hard (baseline):** MATD3-A6+FRPN **81%**, FRPN **80%**
- **PhysX-closed hard (formal):** MPC **73%**, MATD3-A6+MPC **62%**, MAPPO+MPC **58%**

## License / notes

Research code for the Radar5 pursuit experiments. Isaac Sim / OmniDrones
runtime paths are environment-specific; adjust `ISAACSIM_PATH` and conda env
in the `run_*.sh` / `queue_*.sh` scripts before launching.
