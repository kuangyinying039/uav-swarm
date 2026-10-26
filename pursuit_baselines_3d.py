"""Three-dimensional pursuit baselines for the velocity/yaw-rate pursuit environment.

All baselines use exactly the same per-agent local 6-D track produced by the
3-D KF and freshness-aware CI; target truth is never exposed to their guidance
law. Reliable teammate-state sharing gives each controller all UAV positions
for collision avoidance and role allocation. They output only a velocity/yaw-
rate reference. All methods use the same lagged, acceleration-limited flight
controller and configured observation model. MPC here is high-level guidance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is not None:
        return _unit(fallback)
    return np.zeros_like(vector)


def _wrap(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _reference_action(env, agent: int, desired_velocity: np.ndarray) -> np.ndarray:
    c = env.cfg
    velocity = np.asarray(desired_velocity, dtype=float).copy()
    horizontal = float(np.linalg.norm(velocity[:2]))
    if horizontal > c.max_horizontal_velocity:
        velocity[:2] *= c.max_horizontal_velocity / horizontal
    velocity[2] = np.clip(velocity[2], -c.max_vertical_velocity, c.max_vertical_velocity)
    speed_xy = float(np.linalg.norm(velocity[:2]))
    current_yaw = float(2.0 * math.atan2(env.quadrotor_states[agent, 9], env.quadrotor_states[agent, 6]))
    desired_yaw = current_yaw if speed_xy < 1e-6 else math.atan2(velocity[1], velocity[0])
    yaw_rate = np.clip(
        2.0 * _wrap(desired_yaw - current_yaw),
        -c.max_reference_yaw_rate,
        c.max_reference_yaw_rate,
    )
    return np.array(
        [
            velocity[0] / c.max_horizontal_velocity,
            velocity[1] / c.max_horizontal_velocity,
            velocity[2] / c.max_vertical_velocity,
            yaw_rate / c.max_reference_yaw_rate,
        ],
        dtype=float,
    )


def _local_track(env, agent: int) -> tuple[np.ndarray, np.ndarray]:
    """Return only agent-local fused position/velocity estimates."""
    track = env.track_memory[agent][0]
    if not track.initialized:
        # No privileged fallback: without a valid local track, keep station.
        return env.quadrotor_states[agent, :3].copy(), np.zeros(3, dtype=float)
    return track.mean[:3].copy(), track.mean[3:6].copy()


@dataclass
class ArtificialPotentialField3D:
    """Attraction to the target plus 3-D peer/building/boundary repulsion."""

    attraction_gain: float = 1.0
    peer_repulsion_gain: float = 1.6
    building_repulsion_gain: float = 2.2
    boundary_repulsion_gain: float = 1.2
    influence_radius: float = 3.0
    formation_offset: float = 0.5

    name: str = "apf_3d"

    def reset(self, env) -> None:
        pass

    def actions(self, env) -> np.ndarray:
        positions = env.quadrotor_states[:, :3]
        actions = []
        for i, position in enumerate(positions):
            target, _ = _local_track(env, i)
            radial = _unit(position - target, np.array([1.0, 0.0, 0.0]))
            # Stagger desired approach bearings to encourage encirclement.
            bearing = 2.0 * math.pi * i / max(env.cfg.n_uavs, 1)
            offset = self.formation_offset * np.array([math.cos(bearing), math.sin(bearing), 0.25 * (i - 1)])
            desired_point = target + offset
            field = self.attraction_gain * _unit(desired_point - position)

            for j, peer in enumerate(positions):
                if i == j or not env.policy_peer_mask(i)[j]:
                    continue
                delta = position - peer
                distance = float(np.linalg.norm(delta))
                if distance < self.influence_radius:
                    field += self.peer_repulsion_gain * _unit(delta) * (
                        1.0 / max(distance, 0.25) - 1.0 / self.influence_radius
                    )

            for rect, height in zip(env.buildings, env.building_heights):
                x0, y0, x1, y1 = rect
                closest = np.array(
                    [np.clip(position[0], x0, x1), np.clip(position[1], y0, y1), np.clip(position[2], 0.0, height)]
                )
                delta = position - closest
                distance = float(np.linalg.norm(delta))
                if distance < self.influence_radius:
                    field += self.building_repulsion_gain * _unit(delta, radial) * (
                        1.0 / max(distance, 0.25) - 1.0 / self.influence_radius
                    )

            lower = np.array([0.0, 0.0, env.cfg.min_altitude])
            upper = np.array([env.cfg.grid_size, env.cfg.grid_size, env.cfg.max_altitude])
            for axis in range(3):
                low_distance = position[axis] - lower[axis]
                high_distance = upper[axis] - position[axis]
                if low_distance < self.influence_radius:
                    field[axis] += self.boundary_repulsion_gain / max(low_distance, 0.25)
                if high_distance < self.influence_radius:
                    field[axis] -= self.boundary_repulsion_gain / max(high_distance, 0.25)
            desired_velocity = _unit(field) * env.cfg.max_horizontal_velocity
            actions.append(_reference_action(env, i, desired_velocity))
        return np.asarray(actions)


@dataclass
class FastResponseProportionalNavigation3D:
    """Vector proportional navigation with direct-pursuit response injection."""

    navigation_constant: float = 4.0
    response_gain: float = 1.1
    acceleration_limit: float = 2.5
    name: str = "frpn_3d"

    def reset(self, env) -> None:
        pass

    def actions(self, env) -> np.ndarray:
        dt = float(getattr(env, "current_decision_dt", env.cfg.decision_dt))
        actions = []
        for i, state in enumerate(env.quadrotor_states):
            target_position, target_velocity = _local_track(env, i)
            relative_position = target_position - state[:3]
            distance = max(float(np.linalg.norm(relative_position)), 1e-6)
            los = relative_position / distance
            relative_velocity = target_velocity - state[3:6]
            closing_speed = max(-float(np.dot(relative_velocity, los)), 0.0)
            los_rate = np.cross(relative_position, relative_velocity) / (distance * distance)
            lateral_acceleration = (
                self.navigation_constant * max(closing_speed, 0.25) * np.cross(los_rate, los)
            )
            response_acceleration = self.response_gain * (
                target_velocity + env.cfg.max_horizontal_velocity * los - state[3:6]
            )
            acceleration = lateral_acceleration + response_acceleration
            norm = float(np.linalg.norm(acceleration))
            if norm > self.acceleration_limit:
                acceleration *= self.acceleration_limit / norm
            desired_velocity = state[3:6] + acceleration * dt
            actions.append(_reference_action(env, i, desired_velocity))
        return np.asarray(actions)


@dataclass
class CooperativeGuidanceMPC3D:
    """Kinematic receding-horizon guidance above the common velocity-response flight model.

    With ``capture_required_uavs == 1`` (legacy), the closest pursuer intercepts the
    predicted target while teammates take lateral blocking points.  When the
    environment needs multi-UAV capture (``capture_required_uavs >= 2``), the
    ``k`` closest teammates all receive intercept goals inside the capture
    sphere so the teacher matches the evaluation success rule.
    """

    horizon: int = 8
    flank_radius: float = 2.5
    terminal_weight: float = 3.0
    separation_weight: float = 1.5
    building_weight: float = 4.0
    smoothness_weight: float = 0.25
    # Fraction of capture_radius used to stagger multi-intercept goals (dual+).
    multi_intercept_stagger_fraction: float = 0.35
    name: str = "cooperative_guidance_mpc_3d"

    def reset(self, env) -> None:
        self.previous_velocity = np.zeros((env.cfg.n_uavs, 3), dtype=float)

    @staticmethod
    def _building_clearance(point: np.ndarray, env) -> float:
        clearance = float("inf")
        for rect, height in zip(env.buildings, env.building_heights):
            x0, y0, x1, y1 = rect
            closest = np.array(
                [np.clip(point[0], x0, x1), np.clip(point[1], y0, y1), np.clip(point[2], 0.0, height)]
            )
            clearance = min(clearance, float(np.linalg.norm(point - closest)))
        return clearance

    def _assign_goals(self, env, tracks, positions, predicted_horizon_dt: float):
        """Return per-agent goals, roles, and predicted intercept points."""
        c = env.cfg
        n_uavs = c.n_uavs
        required = max(1, int(getattr(c, "capture_required_uavs", 1)))
        capture_radius = float(getattr(c, "capture_radius", max(2.0 * getattr(c, "target_diameter", 0.5), 1e-6)))
        stagger = min(
            self.multi_intercept_stagger_fraction * capture_radius,
            0.45 * capture_radius,
        )
        teacher_role = np.zeros(n_uavs, dtype=np.int64)
        teacher_goal = np.zeros((n_uavs, 3), dtype=np.float32)
        predicted_target = np.zeros((n_uavs, 3), dtype=np.float32)
        goals = [None] * n_uavs
        for i in range(n_uavs):
            target_position, target_velocity = tracks[i]
            visible_peers = np.flatnonzero(env.policy_peer_mask(i))
            if not len(visible_peers):
                visible_peers = np.array([i], dtype=int)
            order = sorted(
                visible_peers.tolist(),
                key=lambda agent: float(np.linalg.norm(positions[agent] - target_position)),
            )
            interceptors = order[: min(required, len(order))]
            predicted = target_position + target_velocity * predicted_horizon_dt
            predicted_target[i] = predicted
            if i in interceptors:
                rank = interceptors.index(i)
                if required <= 1:
                    goal = predicted
                else:
                    angle = 2.0 * math.pi * rank / max(required, 1)
                    goal = predicted + stagger * np.array(
                        [math.cos(angle), math.sin(angle), 0.0], dtype=float
                    )
                teacher_role[i] = 0
            else:
                flankers = [agent for agent in order if agent not in interceptors]
                flank_rank = flankers.index(i)
                speed_xy = float(np.linalg.norm(target_velocity[:2]))
                evader_heading = (
                    math.atan2(target_velocity[1], target_velocity[0]) if speed_xy > 1e-6 else 0.0
                )
                angle = evader_heading + math.pi / 2.0 + flank_rank * math.pi
                goal = predicted + self.flank_radius * np.array(
                    [math.cos(angle), math.sin(angle), 0.25], dtype=float
                )
                teacher_role[i] = 1
            goals[i] = np.asarray(goal, dtype=float)
            teacher_goal[i] = goals[i]
        return goals, teacher_role, teacher_goal, predicted_target, required

    def plan(self, env) -> dict:
        """Return proposed actions plus interceptor/flanker labels for offline datasets.

        ``actions`` is identical to the historical ``actions()`` contract.
        """
        if not hasattr(self, "previous_velocity"):
            self.reset(env)
        c = env.cfg
        dt = float(getattr(env, "current_decision_dt", c.decision_dt))
        positions = env.quadrotor_states[:, :3]
        tracks = [_local_track(env, i) for i in range(c.n_uavs)]
        n_uavs = c.n_uavs
        goals, teacher_role, teacher_goal, predicted_target, required = self._assign_goals(
            env, tracks, positions, self.horizon * dt
        )
        # Multi-UAV capture needs teammates closer than the legacy flank geometry.
        separation_weight = (
            0.35 * self.separation_weight if required >= 2 else self.separation_weight
        )
        min_peer_distance = 0.45 if required >= 2 else 0.2
        actions = []
        candidate_velocity = np.zeros((n_uavs, 11, 3), dtype=np.float32)
        selected_velocity = np.zeros((n_uavs, 3), dtype=np.float32)
        for i, state in enumerate(env.quadrotor_states):
            target_position, target_velocity = tracks[i]
            visible_peers = np.flatnonzero(env.policy_peer_mask(i))
            goal = goals[i]
            direct = _unit(goal - state[:3])
            tangent = _unit(np.array([-direct[1], direct[0], 0.0]))
            candidates = []
            for speed_fraction in (0.45, 0.75, 1.0):
                for lateral in (-0.35, 0.0, 0.35):
                    direction = _unit(direct + lateral * tangent, direct)
                    candidates.append(direction * c.max_horizontal_velocity * speed_fraction)
            candidates.extend([np.zeros(3), target_velocity.copy()])

            best_velocity, best_cost = candidates[0], float("inf")
            for velocity in candidates:
                velocity = np.asarray(velocity, dtype=float).copy()
                horizontal = np.linalg.norm(velocity[:2])
                if horizontal > c.max_horizontal_velocity:
                    velocity[:2] *= c.max_horizontal_velocity / horizontal
                velocity[2] = np.clip(velocity[2], -c.max_vertical_velocity, c.max_vertical_velocity)
                endpoint = state[:3] + velocity * (self.horizon * dt)
                cost = self.terminal_weight * float(np.sum((endpoint - goal) ** 2))
                cost += self.smoothness_weight * float(np.sum((velocity - self.previous_velocity[i]) ** 2))
                peer_ids = [j for j in visible_peers if j != i]
                if peer_ids:
                    peer_distance = np.linalg.norm(positions[peer_ids] - endpoint, axis=1)
                    cost += separation_weight / max(float(np.min(peer_distance)), min_peer_distance) ** 2
                clearance = self._building_clearance(endpoint, env)
                if np.isfinite(clearance):
                    cost += self.building_weight / max(clearance, 0.2) ** 2
                if cost < best_cost:
                    best_cost, best_velocity = cost, velocity
            self.previous_velocity[i] = best_velocity
            actions.append(_reference_action(env, i, best_velocity))
            candidate_velocity[i] = np.stack(candidates[:11])
            selected_velocity[i] = best_velocity
        return {
            "actions": np.asarray(actions),
            "teacher_role": teacher_role,
            "teacher_goal": teacher_goal,
            "predicted_target": predicted_target,
            "candidate_velocity": candidate_velocity,
            "selected_velocity": selected_velocity,
            "capture_required_uavs": int(required),
        }

    def actions(self, env) -> np.ndarray:
        return self.plan(env)["actions"]


BASELINES_3D = {
    "apf": ArtificialPotentialField3D,
    "frpn": FastResponseProportionalNavigation3D,
    "mpc": CooperativeGuidanceMPC3D,
}
