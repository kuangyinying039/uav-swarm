#!/usr/bin/env bash
# PX4 SITL smoke for 1v1 then placeholder notes for 3+1.
# Requires: PX4 SITL + MAVROS already running with namespaces /uav0 /uav3 (minimal)
# or /uav0../uav3 for full fleet.
set -euo pipefail
ROOT=/srv/data/projects/kyy/xieton/repro
cd "$ROOT"
source /opt/lab/apps/miniforge3-26.7.2/etc/profile.d/conda.sh
conda activate "${HOME}/.conda/envs/omnidrones-isaac41-writable" || true
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python - <<'PY'
"""1v1 velocity tracking + synthetic capture check without ROS if unavailable."""
import numpy as np
from px4_offboard_bridge import (
    FleetOffboardBridge,
    OffboardLimits,
    Px4OffboardBridge,
    policy_action_to_world_velocity,
    spin_setpoint_rate,
)

limits = OffboardLimits()
# Offline path always works: validate mapping + capture geometry.
fleet = FleetOffboardBridge(
    ["/uav0", "/uav1", "/uav2", "/uav3"],
    limits=limits,
    capture_radius=1.0,
    capture_required_uavs=2,
)
# Place two pursuers inside radius opposite the origin target.
fleet.vehicles[0]._pose.xyz = np.array([0.5, 0.0, 1.5])
fleet.vehicles[1]._pose.xyz = np.array([-0.5, 0.0, 1.5])
fleet.vehicles[2]._pose.xyz = np.array([4.0, 0.0, 1.5])
fleet.vehicles[3]._pose.xyz = np.array([0.0, 0.0, 1.5])
cap = fleet.evaluate_capture_now()
assert cap["success"], cap
print("[sitl-offline] dual capture geometry OK", cap)

bridge = Px4OffboardBridge("/uav0", limits=limits, publisher=None)
cmds = []

def circle_action():
    t = len(cmds) * limits.decision_dt
    a = np.array([0.5 * np.cos(t), 0.5 * np.sin(t), 0.0, 0.2], dtype=float)
    cmds.append(policy_action_to_world_velocity(a, limits))
    return a

spin_setpoint_rate(bridge, circle_action, hz=5.0, duration_s=2.0)
assert len(cmds) >= 8
print(f"[sitl-offline] streamed {len(cmds)} setpoints @5Hz")
print("[sitl-offline] SUCCESS — attach MAVROS namespaces to exercise live SITL")
PY
