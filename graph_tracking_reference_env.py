"""Core-mechanism scaffold inspired by Zhao et al. 2025.

The code captures the continuous CSTDU setting, distance-based GAT aggregation
formulas, tracking/collision rewards, and decentralized execution API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class ZhaoConfig:
    n_uavs: int = 3
    n_targets: int = 1
    n_obstacles: int = 4
    world_size: float = 400.0
    perception_radius: float = 50.0
    obstacle_radius: float = 20.0
    agent_safe_radius: float = 12.0
    dt: float = 1.0
    max_speed: float = 12.0
    max_steps: int = 180
    seed: int = 11
    beta_agent: float = 0.04
    beta_obstacle: float = 0.08
    beta_target: float = 0.04


class ZhaoGraphTrackingEnv:
    def __init__(self, cfg: ZhaoConfig | None = None):
        self.cfg = cfg or ZhaoConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.reset()

    def reset(self) -> dict:
        c = self.cfg
        self.t = 0
        self.uav_pos = self.rng.uniform(20, c.world_size - 20, size=(c.n_uavs, 2))
        self.uav_vel = self.rng.normal(0, 1, size=(c.n_uavs, 2))
        self.target_pos = self.rng.uniform(40, c.world_size - 40, size=(c.n_targets, 2))
        angles = self.rng.uniform(0, 2 * math.pi, size=c.n_targets)
        self.target_vel = np.column_stack([np.cos(angles), np.sin(angles)]) * 4.0
        self.obstacles = self.rng.uniform(45, c.world_size - 45, size=(c.n_obstacles, 2))
        return self.observe()

    def observe(self) -> dict:
        return {
            "uav_pos": self.uav_pos.copy(),
            "uav_vel": self.uav_vel.copy(),
            "target_pos": self.target_pos.copy(),
            "target_vel": self.target_vel.copy(),
            "obstacles": self.obstacles.copy(),
            "gat_features": self.gat_features(),
        }

    def neighbor_sets(self, k: int) -> dict:
        c = self.cfg
        pa = self.uav_pos[k]
        agent_dist = np.linalg.norm(self.uav_pos - pa, axis=1)
        target_dist = np.linalg.norm(self.target_pos - pa, axis=1)
        obstacle_dist = np.linalg.norm(self.obstacles - pa, axis=1)
        return {
            "agents": [i for i, d in enumerate(agent_dist) if i != k and d <= c.perception_radius],
            "targets": [i for i, d in enumerate(target_dist) if d <= c.perception_radius],
            "obstacles": [i for i, d in enumerate(obstacle_dist) if d <= c.perception_radius],
        }

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        if len(scores) == 0:
            return scores
        shifted = scores - np.max(scores)
        exp = np.exp(shifted)
        return exp / exp.sum()

    def gat_features(self) -> np.ndarray:
        c = self.cfg
        out = np.zeros((c.n_uavs, 9), dtype=float)
        for k in range(c.n_uavs):
            sets = self.neighbor_sets(k)
            pa = self.uav_pos[k]
            parts = []
            for name, points, beta in [
                ("agents", self.uav_pos, c.beta_agent),
                ("obstacles", self.obstacles, c.beta_obstacle),
                ("targets", self.target_pos, c.beta_target),
            ]:
                idx = sets[name]
                if not idx:
                    parts.append(np.zeros(3))
                    continue
                rel = points[idx] - pa
                dist = np.linalg.norm(rel, axis=1)
                weights = self._softmax(np.exp(-beta * dist))
                agg = (weights[:, None] * rel).sum(axis=0)
                parts.append(np.array([agg[0] / c.world_size, agg[1] / c.world_size, len(idx) / 10.0]))
            out[k] = np.concatenate(parts)
        return out

    def step(self, accelerations: np.ndarray) -> dict:
        c = self.cfg
        accelerations = np.asarray(accelerations, dtype=float)
        self.uav_vel += accelerations * c.dt
        speed = np.linalg.norm(self.uav_vel, axis=1, keepdims=True)
        self.uav_vel = np.where(speed > c.max_speed, self.uav_vel / np.maximum(speed, 1e-6) * c.max_speed, self.uav_vel)
        self.uav_pos += self.uav_vel * c.dt
        self.uav_pos = np.clip(self.uav_pos, 0.0, c.world_size)
        self._move_targets()
        reward, metrics = self.reward()
        self.t += 1
        done = metrics["success"] or self.t >= c.max_steps
        return {"obs": self.observe(), "reward": reward, "done": done, "metrics": metrics}

    def _move_targets(self) -> None:
        c = self.cfg
        self.target_pos += self.target_vel * c.dt
        for j in range(c.n_targets):
            for axis in range(2):
                if self.target_pos[j, axis] < 0 or self.target_pos[j, axis] > c.world_size:
                    self.target_vel[j, axis] *= -1
            self.target_pos[j] = np.clip(self.target_pos[j], 0.0, c.world_size)

    def reward(self) -> tuple[float, dict]:
        c = self.cfg
        d_ut = np.linalg.norm(self.uav_pos[:, None, :] - self.target_pos[None, :, :], axis=2)
        target_reward = np.maximum(0.0, 1.0 + (c.perception_radius - d_ut) / c.perception_radius).sum()
        d_uo = np.linalg.norm(self.uav_pos[:, None, :] - self.obstacles[None, :, :], axis=2)
        obstacle_penalty = np.maximum(0.0, (c.obstacle_radius - d_uo) / c.obstacle_radius).sum()
        d_uu = np.linalg.norm(self.uav_pos[:, None, :] - self.uav_pos[None, :, :], axis=2)
        pair_penalty = 0.0
        for i in range(c.n_uavs):
            for j in range(i):
                pair_penalty += max(0.0, (c.agent_safe_radius - d_uu[i, j]) / c.agent_safe_radius)
        reward = float(target_reward - 4.0 * obstacle_penalty - 2.0 * pair_penalty)
        success = bool(np.all(np.min(d_ut, axis=0) < c.perception_radius / 2.0))
        return reward, {
            "target_reward": float(target_reward),
            "obstacle_penalty": float(obstacle_penalty),
            "agent_penalty": float(pair_penalty),
            "min_target_distance": float(np.min(d_ut)),
            "success": success,
        }


def graph_heuristic_policy(env: ZhaoGraphTrackingEnv) -> np.ndarray:
    c = env.cfg
    acc = np.zeros((c.n_uavs, 2), dtype=float)
    for k in range(c.n_uavs):
        pos = env.uav_pos[k]
        to_targets = env.target_pos - pos
        d_targets = np.linalg.norm(to_targets, axis=1)
        visible = np.where(d_targets <= c.perception_radius)[0]
        if len(visible):
            desired = to_targets[visible[np.argmin(d_targets[visible])]]
        else:
            angle = 2 * math.pi * (k / max(c.n_uavs, 1)) + 0.05 * env.t
            center_bias = np.array([math.cos(angle), math.sin(angle)]) * c.perception_radius
            desired = (np.array([c.world_size / 2, c.world_size / 2]) + center_bias) - pos
        norm = np.linalg.norm(desired)
        if norm > 1e-6:
            acc[k] += desired / norm * 2.0
        for obstacle in env.obstacles:
            rel = pos - obstacle
            d = np.linalg.norm(rel)
            if d < c.obstacle_radius * 2.2 and d > 1e-6:
                acc[k] += rel / d * (4.0 * (1.0 - d / (c.obstacle_radius * 2.2)))
        for j in range(c.n_uavs):
            if j == k:
                continue
            rel = pos - env.uav_pos[j]
            d = np.linalg.norm(rel)
            if d < c.agent_safe_radius * 2.0 and d > 1e-6:
                acc[k] += rel / d * 2.0
    return np.clip(acc, -3.0, 3.0)


def run_zhao_heuristic_demo(seed: int = 11, n_uavs: int = 3, n_targets: int = 1) -> dict:
    """Run a non-learning demo of Zhao-style graph tracking mechanics."""
    obstacle_count = 4 if n_uavs <= 3 else 8
    world = 400.0 if n_uavs <= 3 else 1000.0
    perception = 50.0 if n_uavs <= 3 else 160.0
    env = ZhaoGraphTrackingEnv(
        ZhaoConfig(
            seed=seed,
            n_uavs=n_uavs,
            n_targets=n_targets,
            n_obstacles=obstacle_count,
            world_size=world,
            perception_radius=perception,
            max_steps=220,
        )
    )
    history = []
    for _ in range(env.cfg.max_steps):
        action = graph_heuristic_policy(env)
        result = env.step(action)
        history.append(
            {
                "step": env.t,
                "reward": result["reward"],
                "min_target_distance": result["metrics"]["min_target_distance"],
                "success": result["metrics"]["success"],
                "uav_pos": env.uav_pos.copy().tolist(),
                "target_pos": env.target_pos.copy().tolist(),
            }
        )
        if result["done"]:
            break
    return {
        "paper": "Zhao et al. 2025",
        "experiment_kind": "heuristic_core_mechanism_demo",
        "note": "Uses a hand-coded graph heuristic policy; not a full MARL reproduction.",
        "steps": env.t,
        "success": history[-1]["success"],
        "final_min_target_distance": history[-1]["min_target_distance"],
        "history": history,
        "obstacles": env.obstacles.tolist(),
    }


def run_zhao_demo(seed: int = 11, n_uavs: int = 3, n_targets: int = 1) -> dict:
    """Backward-compatible alias for the heuristic demo."""
    return run_zhao_heuristic_demo(seed, n_uavs, n_targets)
