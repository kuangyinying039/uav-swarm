import os
import sys


# Use the project import layout the same way the runtime expects.
ROOT = "/srv/data/projects/kyy"
OMNIDRONES = "/srv/data/projects/kyy/omnidrones"
EXPERIMENTS = "/srv/data/projects/kyy/omnidrones/experiments"
for entry in (ROOT, OMNIDRONES, EXPERIMENTS):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np

from omnidrones.experiments.radar5_frozen_transfer_v1.quadrotor_pursuit_env import QuadrotorPursuitConfig
from xieton.repro.isaac_pursuit_backend import IsaacPursuitEnv


cfg = QuadrotorPursuitConfig(
    n_uavs=3,
    search_steps=10,
    decision_dt=0.2,
    initial_altitude=2.0,
    formation_spacing=2.0,
    lidar_min_range=0.2,
    lidar_target_detection_range=12.0,
    lidar_horizontal_fov_deg=360.0,
    lidar_vertical_min_deg=-25.0,
    lidar_vertical_max_deg=25.0,
    target_diameter=0.5,
    building_count=0,
)

print("[smoke] creating env")
env = IsaacPursuitEnv(cfg, seed=0, headless=True)

try:
    print("[smoke] observe_search")
    obs = env.observe_search()
    print("[smoke] obs_type", type(obs).__name__)
    print("[smoke] obs_keys", list(obs.keys())[:10])
    print("[smoke] obs_dim", env.obs_dim(), "state_dim", env.state_dim())

    action = np.zeros((3, 4), dtype=np.float32)
    print("[smoke] step_joint")
    result = env.step_joint(action)
    print("[smoke] reward", result.get("reward"))
    print("[smoke] terminated", result.get("terminated"))
    print("[smoke] truncated", result.get("truncated"))

    print("[smoke] success")
finally:
    try:
        env.close()
    except Exception as exc:
        print("[smoke] close error", type(exc).__name__, exc)
    os._exit(0)
