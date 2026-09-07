"""Target-detection surrogate for a body-mounted 3-D lidar, not raw point clouds."""
import math
import numpy as np

try:
    from pursuit_kinematics import quaternion_to_rotation, quaternion_from_yaw
except ImportError:
    from .pursuit_kinematics import quaternion_to_rotation, quaternion_from_yaw


def sensor_pose(env, agent):
    cfg = env.cfg
    if hasattr(env, 'quadrotor_states') and not getattr(env, '_resetting_quad_state', False):
        state = env.quadrotor_states[agent]
        position, rotation = state[:3], quaternion_to_rotation(state[6:10])
    else:
        position = np.r_[env.positions[agent], env.altitudes[agent]]
        rotation = quaternion_to_rotation(quaternion_from_yaw(float(env.headings[agent])*math.pi/4))
    roll, pitch, yaw = np.deg2rad(cfg.lidar_mount_rpy_deg)
    cx, sx, cy, sy, cz, sz = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    mount = np.array([[cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx],
                      [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx], [-sy, cy*sx, cy*cx]])
    return position + rotation@np.asarray(cfg.lidar_mount_xyz), rotation@mount


def lidar_visibility(env):
    cfg = env.cfg
    visible = np.zeros((cfg.n_uavs, cfg.n_targets), dtype=bool)
    targets = np.column_stack([env.dynamic_targets, env.target_altitudes])
    for agent in range(cfg.n_uavs):
        if env.disabled_uavs[agent]:
            continue
        origin, rotation = sensor_pose(env, agent)
        local = (targets-origin)@rotation
        ranges = np.linalg.norm(local, axis=1)
        azimuth = np.rad2deg(np.arctan2(local[:, 1], local[:, 0]))
        elevation = np.rad2deg(np.arctan2(local[:, 2], np.linalg.norm(local[:, :2], axis=1)))
        candidates = ((ranges >= cfg.lidar_min_range) & (ranges <= cfg.lidar_target_detection_range)
                      & (np.abs(azimuth) <= cfg.lidar_horizontal_fov_deg/2)
                      & (elevation >= cfg.lidar_vertical_min_deg) & (elevation <= cfg.lidar_vertical_max_deg))
        for target in np.flatnonzero(candidates):
            visible[agent, target] = env.has_line_of_sight_3d(origin, targets[target])
    return visible
