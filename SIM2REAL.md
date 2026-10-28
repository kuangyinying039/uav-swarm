# Sim-to-Real Contract (PhysX closed + PX4)

This document freezes the interface between Isaac training and PX4 offboard flight.
Hybrid `isaac_dual_postlc` results are a **baseline only**; do not mix them with
`physx_closed` or real-flight tables.

## Fleet

| Role | Count | Platform |
|------|------:|----------|
| Pursuers | 3 | Same airframe (sim: Hummingbird; real: PX4 multicopter) |
| Evader | 1 | Same airframe as pursuers |

## Action interface (5 Hz)

Policy output shape: `(n_agents, 4)` normalized to `[-1, 1]`:

| Index | Meaning | Scale |
|------:|---------|-------|
| 0 | world \(v_x\) | `max_horizontal_velocity` (hard default 1.4 m/s) |
| 1 | world \(v_y\) | same |
| 2 | world \(v_z\) | `max_vertical_velocity` (hard default 0.8 m/s) |
| 3 | yaw rate | `max_reference_yaw_rate` |

Mapping helpers live in
`omnidrones/experiments/radar5_frozen_transfer_v1/sim2real_contract.py`.

PX4 offboard (MAVROS / MAVLink `SET_POSITION_TARGET_LOCAL_NED`):

- type mask: velocity + yaw rate (ignore position / accel)
- frame: `LOCAL_NED` (bridge converts ENU/world ↔ NED)
- send rate: **5 Hz** (`decision_dt = 0.2`), hold command between ticks
- topics (MAVROS): `~setpoint_raw/local`, odometry from `~local_position/odom`

## Observation interface

Policy may only consume:

- own / peer estimated states from local odometry (noisy)
- target **track** estimates from the fused tracker (never simulator GT)
- building / occupancy features derived from onboard sensing
- communication adjacency / shared tracks

Forbidden in policy inputs: target world pose GT, other agents' privileged PhysX state.

## Capture definition

Shared module: `capture_geometry.py`.

Formal success (must match sim eval, SITL, and real logs):

1. At least `capture_required_uavs` (default **2**) *active* pursuers inside `capture_radius`.
2. Optional angular gate when `capture_mode` is not `single_distance`
   (`capture_angular_span_deg`, default 100°).

`min_distance` (nearest UAV) is **diagnostic only** and never ends an episode.

`radar5_dual_*` hybrid configs keep `capture_mode=single_distance` (k-distance).
`physx_closed` configs may enable `angular_enclose` — report which mode was used.

Formal ladder uses **`target_diameter=0.5`** → `capture_radius = 2 × diameter = 1.0 m`
(see `quadrotor_pursuit_env.py`). Do not mix 0.75-diameter tables with 0.5 tables.

## Physics authority (`physx_closed`)

- All four vehicles are PhysX articulations with Lee velocity tracking.
- Evader pose is read from PhysX only; Radar5 must not teleport the target.
- Headless PhysX corruption → truncate episode; do not continue on kinematics.
- GUI recoveries set `physics_recovered=True` and are excluded from formal stats.

## Labels / directories

| Tag | Meaning |
|-----|---------|
| `isaac_dual_postlc` | Hybrid: PhysX pursuers + kinematic evader (historical) |
| `isaac_dual_physx_closed` | Pure PhysX 3+1 closed loop |
| `px4_sitl` / `real` | Offboard flight logs using the same capture API |

## Deployment recommendation

First flights: classical guidance / FRPN velocity + optional RL residual.
Disable residual for ablations only after SITL 3+1 stationary encirclement passes.

## Onboard stack (fills previous gaps)

| Gap | Module | Role |
|-----|--------|------|
| Obs | [`real_flight_obs.py`](real_flight_obs.py) | Odom + fused tracks → shadow env → training-dim `observe_search()` |
| Safety | [`real_flight_safety.py`](real_flight_safety.py) | Stale odom / peer proximity / altitude / geofence → hold |
| 5 Hz node | [`real_flight_node.py`](real_flight_node.py) | `DecisionClock` + expert / `ResidualPolicy` → `FleetOffboardBridge` |
| Residual | [`residual_policy.py`](residual_policy.py) | `clip(expert + gain * rl)` |

Dry-run (no ROS)::

```bash
python real_flight_node.py --dry-run --duration 4 --teacher frpn --track-xyz 0,0,2
```

Live residual (MAVROS namespaces already up)::

```bash
python real_flight_node.py --checkpoint path/to/actor.pt --residual-gain 0.3 --log outputs/real_flight.jsonl
```

Track input: call `RealFlightController.set_shared_track(xyz, vel)` or pass a
`track_provider` callable that returns per-UAV `TargetTrackEstimate`s. Do **not**
feed simulator / mocap target GT into the policy path in formal logs.
