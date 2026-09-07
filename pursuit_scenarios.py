"""Randomized pursuit starts and acceleration-limited predictive evasion.

Parameters are simulation assumptions pending loaded-airframe identification.
The evader observes noisy positions, not the pursuer's future commands.
"""
import numpy as np


def initialize_formation(env):
    c = env.cfg
    rng = np.random.default_rng(c.seed + 44021)
    layout = ('same_side', 'crossing', 'dispersed')[int(rng.integers(3))]
    env.initial_layout = layout
    env.evader_safety_interventions = 0
    for _ in range(1500):
        margin = min(c.initial_distance_max, c.grid_size*.3)
        target = np.r_[rng.uniform(margin, c.grid_size-margin, 2),
                       rng.uniform(c.min_altitude+1, c.max_altitude-1)]
        if env._point_inside_building_prism(target, margin=.3):
            continue
        heading = rng.uniform(-np.pi, np.pi)
        if layout == 'same_side':
            angles = heading + rng.uniform(-np.pi/6, np.pi/6, c.n_uavs)
        elif layout == 'crossing':
            angles = heading + np.arange(c.n_uavs)*np.pi + rng.uniform(-.3, .3, c.n_uavs)
        else:
            angles = rng.uniform(-np.pi, np.pi, c.n_uavs)
        distances = rng.uniform(c.initial_distance_min, c.initial_distance_max, c.n_uavs)
        dz = rng.uniform(-1., 1., c.n_uavs)
        radius = np.sqrt(distances**2-dz**2)
        positions = target + np.column_stack([radius*np.cos(angles), radius*np.sin(angles), dz])
        lower, upper = [0.5, .5, c.min_altitude], [c.grid_size-.5, c.grid_size-.5, c.max_altitude]
        if np.any(positions < lower) or np.any(positions > upper):
            continue
        if any(env._point_inside_building_prism(p, margin=c.quadrotor_clearance) for p in positions):
            continue
        pair_distance = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
        pair_distance[np.diag_indices(c.n_uavs)] = np.inf
        if np.min(pair_distance) < max(1.5, 2*c.quadrotor_clearance):
            continue
        env.dynamic_targets[0], env.target_altitudes[0] = target[:2], target[2]
        env.positions, env.altitudes = positions[:, :2].copy(), positions[:, 2].copy()
        env.headings = rng.integers(0, 8, c.n_uavs)
        # Both sides start at rest; a higher speed limit does not mean teleporting to it.
        env.target_velocity[:] = 0.
        env.target_vertical_velocity[:] = 0.
        env.initial_distances = distances.tolist()
        env.target_grid = env._target_occupancy()
        return
    raise ValueError('Unable to sample collision-free initial geometry; check map and distance limits')


def velocity_step(velocity, desired, dt, cfg):
    """Stable first-order velocity response bounded by acceleration and speed."""
    velocity, desired = np.asarray(velocity), np.asarray(desired).copy()
    desired[:2] *= min(1., cfg.target_speed/max(np.linalg.norm(desired[:2]), 1e-9))
    desired[2] = np.clip(desired[2], -cfg.target_vertical_speed, cfg.target_vertical_speed)
    change = (desired-velocity)*(1-np.exp(-dt/cfg.evader_response_time))
    change[:2] *= min(1., cfg.evader_horizontal_acceleration*dt/max(np.linalg.norm(change[:2]), 1e-9))
    change[2] = np.clip(change[2], -cfg.evader_vertical_acceleration*dt, cfg.evader_vertical_acceleration*dt)
    return velocity+change


def command_velocity(direction, cfg):
    direction = np.asarray(direction)
    horizontal = direction[:2]/max(np.linalg.norm(direction[:2]), 1e-9)
    return np.r_[horizontal*cfg.target_speed, np.clip(direction[2], -1, 1)*cfg.target_vertical_speed]


def safe_segment(env, start, end):
    c = env.cfg
    return (np.all(end >= [.2, .2, c.min_altitude])
            and np.all(end <= [c.grid_size-.2, c.grid_size-.2, c.max_altitude])
            and not env._point_inside_building_prism(end, margin=.2)
            and env.has_line_of_sight_3d(start, end))


def predictive_escape(env, target_index):
    c = env.cfg
    dt = float(getattr(env, 'current_decision_dt', c.decision_dt))
    observed = np.column_stack([env.positions, env.altitudes])
    observed = observed + env._evader_sensor_rng.normal(0, c.evader_position_noise_std, observed.shape)
    if env._evader_previous_observation is not None:
        measured = (observed-env._evader_previous_observation)/dt
        measured[:, :2] *= np.minimum(1., c.max_horizontal_velocity/np.maximum(np.linalg.norm(measured[:, :2], axis=1, keepdims=True), 1e-9))
        measured[:, 2] = np.clip(measured[:, 2], -c.max_vertical_velocity, c.max_vertical_velocity)
        env._evader_estimated_velocity = .7*env._evader_estimated_velocity + .3*measured
    env._evader_previous_observation = observed
    active = ~env.disabled_uavs
    peers, peer_velocity = observed[active], env._evader_estimated_velocity[active]
    target = np.r_[env.dynamic_targets[target_index], env.target_altitudes[target_index]]
    current_velocity = np.r_[env.target_velocity[target_index], env.target_vertical_velocity[target_index]]
    best, best_score = np.zeros(3), -np.inf
    directions = [np.zeros(3)]
    directions += [np.array([np.cos(a), np.sin(a), z]) for a in np.linspace(-np.pi, np.pi, 24, endpoint=False) for z in (-.8, 0., .8)]
    for direction in directions:
        position, velocity = target.copy(), current_velocity.copy()
        desired = command_velocity(direction, c)
        clearance = np.inf
        for k in range(c.evader_prediction_steps):
            next_velocity = velocity_step(velocity, desired, dt, c)
            trial = position+.5*(velocity+next_velocity)*dt
            if not safe_segment(env, position, trial):
                break
            future_peers = peers + peer_velocity*((k+1)*dt)
            # Risk near predicted UAV centres; no access to future policy actions.
            if len(peers):
                clearance = min(clearance, float(np.min(np.linalg.norm(future_peers-trial, axis=1))))
            position, velocity = trial, next_velocity
        else:
            clearance = clearance if np.isfinite(clearance) else 0.
            terminal_clearance = float(np.min(np.linalg.norm(future_peers-position, axis=1))) if len(peers) else 0.
            score = clearance + .35*terminal_clearance - .01*np.linalg.norm(desired-current_velocity)
            if score > best_score:
                best, best_score = direction, score
    return best


def advance_evader(env, target_index, direction, dt):
    c = env.cfg
    start = np.r_[env.dynamic_targets[target_index], env.target_altitudes[target_index]]
    velocity = np.r_[env.target_velocity[target_index], env.target_vertical_velocity[target_index]]
    next_velocity = velocity_step(velocity, command_velocity(direction, c), dt, c)
    proposed = start+.5*(velocity+next_velocity)*dt
    if not safe_segment(env, start, proposed):
        # Residual map-contact guard, not a planned instantaneous reversal/climb.
        proposed, next_velocity = start, np.zeros(3)
        env.evader_safety_interventions = getattr(env, 'evader_safety_interventions', 0)+1
    env.dynamic_targets[target_index], env.target_altitudes[target_index] = proposed[:2], proposed[2]
    env.target_velocity[target_index], env.target_vertical_velocity[target_index] = next_velocity[:2], next_velocity[2]
