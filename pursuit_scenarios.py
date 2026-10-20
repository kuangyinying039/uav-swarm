"""Randomized same-side triangle starts and acceleration-limited predictive evasion.

Parameters are simulation assumptions pending loaded-airframe identification.
The evader observes noisy positions, not the pursuer's future commands.
"""
import numpy as np


def initialize_formation(env):
    """Place a 3-UAV triangle on one side of the target with opening lidar LOS."""
    c = env.cfg
    rng = np.random.default_rng(c.seed + 44021)
    env.initial_layout = "same_side_triangle"
    env.evader_safety_interventions = 0
    n_uavs = c.n_uavs
    spacing = float(getattr(c, "formation_spacing", 2.2))
    sensor_horizon = float(getattr(c, "lidar_target_detection_range", c.initial_distance_max)) - 1.25
    distance_max = min(c.initial_distance_max, sensor_horizon)
    distance_min = min(c.initial_distance_min, distance_max - 0.5)
    if distance_min <= 0 or distance_min >= distance_max:
        distance_min, distance_max = max(2.5, 0.45 * sensor_horizon), max(3.0, sensor_horizon)
    vertex_radius = spacing / np.sqrt(3.0)

    try:
        from pursuit_lidar import lidar_visibility
    except ImportError:
        from .pursuit_lidar import lidar_visibility

    for _ in range(2500):
        margin = min(distance_max, c.grid_size * 0.28)
        target = np.r_[rng.uniform(margin, c.grid_size - margin, 2),
                       rng.uniform(c.min_altitude + 1.2, c.max_altitude - 1.2)]
        if env._point_inside_building_prism(target, margin=0.3):
            continue
        heading = float(rng.uniform(-np.pi, np.pi))
        radial = np.array([np.cos(heading), np.sin(heading)])
        distance = float(rng.uniform(distance_min, distance_max))
        centroid_xy = target[:2] + radial * distance
        # First vertex faces the target; the other two complete an equilateral triangle.
        positions = np.zeros((n_uavs, 3), dtype=float)
        for agent in range(n_uavs):
            angle = heading + np.pi + agent * (2.0 * np.pi / max(n_uavs, 1))
            positions[agent, :2] = centroid_xy + vertex_radius * np.array([np.cos(angle), np.sin(angle)])
            positions[agent, 2] = target[2] + rng.uniform(-0.25, 0.25)
        lower = np.array([0.5, 0.5, c.min_altitude])
        upper = np.array([c.grid_size - 0.5, c.grid_size - 0.5, c.max_altitude])
        if np.any(positions < lower) or np.any(positions > upper):
            continue
        if np.any((positions[:, :2] - target[:2]) @ radial <= 0.4):
            continue
        if any(env._point_inside_building_prism(point, margin=c.quadrotor_clearance) for point in positions):
            continue
        pair_distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        pair_distance[np.diag_indices(n_uavs)] = np.inf
        if np.min(pair_distance) < max(1.5, 2 * c.quadrotor_clearance):
            continue
        env.dynamic_targets[0], env.target_altitudes[0] = target[:2], target[2]
        env.positions, env.altitudes = positions[:, :2].copy(), positions[:, 2].copy()
        yaws = np.arctan2(target[1] - env.positions[:, 1], target[0] - env.positions[:, 0])
        env.initial_yaws = yaws
        env.headings = np.mod(np.rint(yaws / (np.pi / 4.0)).astype(int), 8)
        env.target_velocity[:] = 0.0
        env.target_vertical_velocity[:] = 0.0
        env.initial_distances = np.linalg.norm(positions - target[None, :], axis=1).tolist()
        env.target_grid = env._target_occupancy()
        if hasattr(env, "quadrotor_states") and env.quadrotor_states.shape[0] == n_uavs:
            if hasattr(env, "_initialize_quadrotor_states"):
                env._initialize_quadrotor_states()
        if not np.all(lidar_visibility(env)[:, 0]):
            continue
        return
    raise ValueError("Unable to sample a same-side triangle with lidar line-of-sight; check map and range")


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
    direction = np.asarray(direction, dtype=float).reshape(-1)
    # Occlusion / search helpers may emit planar directions; pad to 3-D.
    if direction.size == 2:
        direction = np.r_[direction, 0.0]
    elif direction.size < 2:
        direction = np.zeros(3, dtype=float)
    horizontal = direction[:2] / max(np.linalg.norm(direction[:2]), 1e-9)
    return np.r_[horizontal * cfg.target_speed, np.clip(direction[2], -1, 1) * cfg.target_vertical_speed]


def safe_segment(env, start, end):
    c = env.cfg
    return (np.all(end >= [.2, .2, c.min_altitude])
            and np.all(end <= [c.grid_size-.2, c.grid_size-.2, c.max_altitude])
            and not env._point_inside_building_prism(end, margin=.2)
            and env.has_line_of_sight_3d(start, end))


def _integrate_evader_step(start, velocity, desired, dt, cfg):
    next_velocity = velocity_step(velocity, desired, dt, cfg)
    proposed = np.asarray(start, dtype=float) + 0.5 * (np.asarray(velocity, dtype=float) + next_velocity) * dt
    return proposed, next_velocity


def resolve_evader_motion(env, start, velocity, desired, dt):
    """Advance one evader step. A heading into a building slides along it.

    Returns ``(position, velocity, status)`` with status ``clear``, ``slide``,
    or ``stopped``. Stopping is only the fallback when no tangent step fits.
    """
    c = env.cfg
    start = np.asarray(start, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    desired = np.asarray(desired, dtype=float)
    proposed, next_velocity = _integrate_evader_step(start, velocity, desired, dt, c)
    if safe_segment(env, start, proposed):
        return proposed, next_velocity, "clear"
    # The lagged response cannot turn off a wall in one step. Drop the
    # component that entered the obstacle and keep a tangent (or a short retreat).
    blocked = proposed - start
    horizontal = blocked[:2]
    horizontal_norm = float(np.linalg.norm(horizontal))
    options = []
    if horizontal_norm > 1e-8:
        normal = horizontal / horizontal_norm
        tangent_speed = velocity[:2] - normal * float(np.dot(velocity[:2], normal))
        if float(np.linalg.norm(tangent_speed)) > 0.3:
            options.append(tangent_speed)
        sideways = np.array([-normal[1], normal[0]])
        options.append(sideways * c.target_speed)
        options.append(-sideways * c.target_speed)
        options.append(-normal * min(c.target_speed, 1.0))
    for horizontal_velocity in options:
        trial = np.array([horizontal_velocity[0], horizontal_velocity[1], 0.0], dtype=float)
        speed = float(np.linalg.norm(trial[:2]))
        if speed > c.target_speed:
            trial[:2] *= c.target_speed / speed
        for scale in (1.0, 0.5, 0.25):
            end = start + trial * scale * dt
            moved = float(np.linalg.norm(end[:2] - start[:2]))
            if moved > 1e-3 and safe_segment(env, start, end):
                return end, trial * scale, "slide"
    return start.copy(), np.zeros(3), "stopped"


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
    hidden_now = 0
    nearest_now = 0.0
    if len(peers):
        hidden_now = sum(not env.has_line_of_sight_3d(peer, target) for peer in peers)
        nearest_now = float(np.min(np.linalg.norm(peers - target, axis=1)))
    # Inside this radius a waiting evader is pressured to move. Farther out it may wait.
    react_distance = float(getattr(c, "evader_repulsion_radius", 10.0))
    closing_speed = 0.0
    if len(peers):
        offset = peers - target
        distance = np.maximum(np.linalg.norm(offset, axis=1), 1e-6)
        closing_speed = float(np.max(np.sum(-peer_velocity * offset, axis=1) / distance))
        closing_speed = max(0.0, closing_speed)
    best, best_score = np.zeros(3), -np.inf
    directions = [np.zeros(3)]
    directions += [np.array([np.cos(a), np.sin(a), z]) for a in np.linspace(-np.pi, np.pi, 24, endpoint=False) for z in (-.8, 0., .8)]
    for direction in directions:
        position, velocity = target.copy(), current_velocity.copy()
        desired = command_velocity(direction, c)
        clearance = np.inf
        hidden = hidden_now
        moved_steps = 0
        future_peers = peers
        for k in range(c.evader_prediction_steps):
            nxt, velocity, status = resolve_evader_motion(env, position, velocity, desired, dt)
            if status == "stopped":
                break
            moved_steps += 1
            future_peers = peers + peer_velocity*((k+1)*dt)
            if len(peers):
                clearance = min(clearance, float(np.min(np.linalg.norm(future_peers-nxt, axis=1))))
                hidden = sum(not env.has_line_of_sight_3d(peer, nxt) for peer in future_peers)
            position = nxt
        hovering = float(np.linalg.norm(desired[:2])) < 0.25 * c.target_speed
        if moved_steps == 0 and not hovering:
            continue
        clearance = clearance if np.isfinite(clearance) else nearest_now
        terminal_clearance = float(np.min(np.linalg.norm(future_peers-position, axis=1))) if len(peers) else 0.
        # Staying hidden adds nothing. A pursuer that is close or closing makes waiting costly.
        hover_penalty = 0.0
        if hovering:
            hover_penalty = 0.35 * max(0.0, react_distance - nearest_now) + 1.2 * closing_speed
        score = (clearance + .35*terminal_clearance
                 + float(getattr(c, "occlusion_blocked_weight", 1.2))*(hidden - hidden_now)
                 - .01*np.linalg.norm(desired-current_velocity)
                 - hover_penalty)
        if score > best_score:
            best, best_score = direction, score
    return best


def _pressure_escape_directions(env, start, target_index):
    """Fallback headings when a cover waypoint wedges the evader under pressure."""
    c = env.cfg
    active = ~np.asarray(env.disabled_uavs, dtype=bool)
    pursuers = np.column_stack([env.positions[active], env.altitudes[active]])
    if len(pursuers) == 0:
        return [np.array([1.0, 0.0, 0.0])]
    delta = start[None, :] - pursuers
    distances = np.linalg.norm(delta, axis=1)
    weights = 1.0 / np.maximum(distances, 0.5) ** 2
    repulsion = np.sum(delta / np.maximum(distances[:, None], 1e-8) * weights[:, None], axis=0)
    repulsion = repulsion / max(float(np.linalg.norm(repulsion)), 1e-8)
    nearest = float(np.min(distances))
    options = [
        np.array([repulsion[0], repulsion[1], 1.0]),
        repulsion,
        np.array([repulsion[0], repulsion[1], 0.85]),
        np.array([repulsion[0], repulsion[1], -0.55]),
        np.array([-repulsion[1], repulsion[0], 0.55]),
        np.array([repulsion[1], -repulsion[0], 0.55]),
        np.array([-repulsion[0], -repulsion[1], 0.75]),
    ]
    # Dense ring so a soft corner cannot kill every candidate.
    for angle in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        options.append(np.array([np.cos(angle), np.sin(angle), 0.35]))
        options.append(np.array([np.cos(angle), np.sin(angle), 0.9]))
    if nearest < float(getattr(c, "evader_repulsion_radius", 10.0)):
        options.insert(0, np.array([repulsion[0], repulsion[1], 1.0]))
    return options


def advance_evader(env, target_index, direction, dt):
    c = env.cfg
    start = np.r_[env.dynamic_targets[target_index], env.target_altitudes[target_index]]
    velocity = np.r_[env.target_velocity[target_index], env.target_vertical_velocity[target_index]]
    proposed, next_velocity, status = resolve_evader_motion(
        env, start, velocity, command_velocity(direction, c), dt)
    active = ~np.asarray(env.disabled_uavs, dtype=bool)
    pursuers = np.column_stack([env.positions[active], env.altitudes[active]])
    nearest = (
        float(np.min(np.linalg.norm(pursuers - start[None, :], axis=1)))
        if len(pursuers)
        else float("inf")
    )
    moved = float(np.linalg.norm(proposed - start))
    # Hover/"clear" with ~0 displacement under pressure is the freeze-to-capture bug.
    # Treat it like a hard stop and force leave-corner escape.
    react = float(getattr(c, "evader_repulsion_radius", 10.0))
    stalled = moved < max(0.02, 0.08 * c.target_speed * dt)
    if status == "stopped" or (nearest < react and stalled):
        if status == "stopped":
            env.evader_safety_interventions = getattr(env, "evader_safety_interventions", 0) + 1
        if nearest < react:
            for escape in _pressure_escape_directions(env, start, target_index):
                trial_pos, trial_vel, trial_status = resolve_evader_motion(
                    env, start, velocity, command_velocity(escape, c), dt
                )
                trial_moved = float(np.linalg.norm(trial_pos - start))
                if trial_status != "stopped" and trial_moved >= max(0.02, 0.08 * c.target_speed * dt):
                    proposed, next_velocity, status = trial_pos, trial_vel, trial_status
                    if hasattr(env, "occlusion_waypoints"):
                        env.occlusion_waypoints[target_index] = proposed + escape * max(
                            c.target_speed, 1.0
                        )
                        env.occlusion_waypoint_dwell[target_index] = 0
                    break
    env.dynamic_targets[target_index], env.target_altitudes[target_index] = proposed[:2], proposed[2]
    env.target_velocity[target_index], env.target_vertical_velocity[target_index] = next_velocity[:2], next_velocity[2]
