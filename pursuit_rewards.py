"""Bounded geometric rewards for the distance-capture task only."""
import numpy as np


def geometry_features(positions, target, active, cfg):
    delta = np.asarray(positions) - np.asarray(target)
    distances = np.linalg.norm(delta, axis=1)
    # Inside the capture sphere there is no further approach incentive.
    log_distance = np.log(np.maximum(distances, cfg.capture_radius) / cfg.capture_radius)
    proximity = np.where(active, 1.0 / (1.0 + log_distance), 0.0).mean()
    # Horizontal encirclement is useful only near the target and at its altitude.
    # Three vehicles cannot enclose a volume in full 3-D.
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    eligible = np.asarray(active) & (horizontal > 1e-6)
    enclosure = 0.0
    if np.count_nonzero(eligible) >= 3:
        angles = np.sort(np.arctan2(delta[eligible, 1], delta[eligible, 0]))
        max_gap = np.max(np.diff(np.r_[angles, angles[0]+2*np.pi]))
        angular = np.clip((2*np.pi-max_gap)/(2*np.pi*(1-1/len(angles))), 0, 1)
        near = np.exp(-np.mean(np.maximum(distances[eligible]-cfg.capture_radius, 0))/cfg.encirclement_distance_scale)
        altitude = np.exp(-np.mean(np.abs(delta[eligible, 2]))/cfg.encirclement_height_scale)
        enclosure = float(angular * near * altitude)
    return log_distance, float(proximity), enclosure


def obstacle_cost(positions, active, buildings, heights, cfg):
    """3-D distance to solid building boxes, less UAV clearance radius."""
    risk = np.zeros(len(positions))
    for rect, height in zip(buildings, heights):
        lower, upper = np.array([rect[0], rect[1], 0.]), np.array([rect[2], rect[3], height])
        outside = np.maximum(np.maximum(lower-positions, positions-upper), 0.)
        clearance = np.maximum(np.linalg.norm(outside, axis=1)-cfg.quadrotor_clearance, 0.)
        penetration = np.maximum(cfg.obstacle_reward_safe_distance-clearance, 0.)
        risk = np.maximum(risk, np.log1p(penetration/cfg.obstacle_reward_log_scale)/
                          np.log1p(cfg.obstacle_reward_safe_distance/cfg.obstacle_reward_log_scale))
    return -cfg.obstacle_proximity_weight * float(np.where(active, risk, 0.).mean())


def shaped_rewards(previous, current, active_before, active_after, cfg):
    old_log, _, old_enclosure = previous
    new_log, proximity, enclosure = current
    # Lost vehicles never earn progress by disappearing from the minimum.
    valid = np.asarray(active_before) & np.asarray(active_after)
    progress = np.where(valid, np.clip(old_log-new_log, -1., 1.), 0.)
    return {
        "individual_approach": cfg.individual_approach_weight * float(progress.mean()),
        "target_proximity": cfg.target_proximity_weight * proximity,
        "encirclement_progress": cfg.encirclement_progress_weight * (enclosure-old_enclosure),
    }, progress
