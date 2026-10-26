"""Onboard observation pipeline: odometry + tracks → training-compatible obs.

Injects fleet ENU estimates into a shadow ``QuadrotorPursuitEnv`` so
``observe_search()`` produces the same ``agent_observations`` / ``hetero_graph``
layout used in Isaac training. Never feeds simulator GT; tracks come from the
caller (fused radar / vision / shared CI).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from pursuit_kinematics import quaternion_from_yaw
from px4_offboard_bridge import VehiclePose


@dataclass
class TargetTrackEstimate:
    """Per-agent fused target track in world ENU (z-up)."""

    mean_xyz: np.ndarray  # (3,)
    mean_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initialized: bool = False
    stamp_s: float = 0.0
    measurement_variance: float = 0.25


@dataclass
class FleetEstimates:
    """3 pursuers (+ optional shared target track broadcast)."""

    vehicles: Sequence[VehiclePose]
    tracks: Sequence[TargetTrackEstimate] | None = None
    buildings: np.ndarray | None = None  # (N, 4) xy lower/upper or empty
    tick: int = 0


def pose_to_quadrotor_state(pose: VehiclePose) -> np.ndarray:
    """Pack ENU pose into the env's 13-D rigid-body state (wxyz quat)."""
    state = np.zeros(13, dtype=float)
    state[:3] = np.asarray(pose.xyz, dtype=float).reshape(3)
    state[3:6] = np.asarray(pose.xyz_vel, dtype=float).reshape(3)
    state[6:10] = quaternion_from_yaw(float(pose.yaw))
    state[10:13] = [0.0, 0.0, float(pose.yaw_rate)]
    return state


class OnboardObsBuilder:
    """Shadow env kept in sync with onboard estimates; builds policy obs."""

    def __init__(self, env_cfg=None, *, seed: int = 0):
        from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv

        if env_cfg is None:
            env_cfg = QuadrotorPursuitConfig(
                seed=int(seed),
                building_count=0,
                handoff_initial_track=False,
                pursuit_target_observable=False,
            )
        self.env = QuadrotorPursuitEnv(env_cfg)
        self.env.reset()
        # First flights: full mesh peer visibility (comm/relocalization assumed).
        n = int(self.env.cfg.n_uavs)
        self.env.last_graph = np.ones((n, n), dtype=float) - np.eye(n)
        self._link_ok = np.ones(n, dtype=bool)

    @property
    def n_uavs(self) -> int:
        return int(self.env.cfg.n_uavs)

    def set_peer_graph(self, adjacency: np.ndarray) -> None:
        adj = np.asarray(adjacency, dtype=float)
        if adj.shape != (self.n_uavs, self.n_uavs):
            raise ValueError(f"adjacency must be ({self.n_uavs},{self.n_uavs})")
        self.env.last_graph = adj.copy()

    def set_link_ok(self, link_ok: Sequence[bool]) -> None:
        flags = np.asarray(list(link_ok), dtype=bool).reshape(-1)
        if flags.shape[0] != self.n_uavs:
            raise ValueError("link_ok length must match n_uavs")
        self._link_ok = flags
        self.env.disconnected_uavs = ~flags

    def inject(self, estimates: FleetEstimates) -> None:
        """Write vehicle + track estimates into the shadow env (no physics)."""
        if len(estimates.vehicles) < self.n_uavs:
            raise ValueError(f"need at least {self.n_uavs} vehicle poses")
        for i in range(self.n_uavs):
            self.env.quadrotor_states[i] = pose_to_quadrotor_state(estimates.vehicles[i])
        self.env.positions = self.env.quadrotor_states[:, :2].copy()
        self.env.altitudes = self.env.quadrotor_states[:, 2].copy()
        self.env.vertical_velocities = self.env.quadrotor_states[:, 5].copy()
        yaws = [
            float(2.0 * math.atan2(q[3], q[0]))
            for q in self.env.quadrotor_states[:, 6:10]
        ]
        # Discrete heading buckets used by legacy planar fields only.
        self.env.headings = np.asarray(
            [int(((yaw % (2 * math.pi)) / (math.pi / 4)) % 8) for yaw in yaws],
            dtype=int,
        )
        self.env.t = int(estimates.tick)
        self.env.disconnected_uavs = ~self._link_ok

        tracks = estimates.tracks
        if tracks is None:
            return
        if len(tracks) < self.n_uavs:
            raise ValueError("tracks must cover every pursuer (broadcast OK)")
        timestamp = int(estimates.tick)
        for i in range(self.n_uavs):
            est = tracks[i]
            slot = self.env.track_memory[i][0]
            if not est.initialized:
                slot.initialized = False
                continue
            xyz = np.asarray(est.mean_xyz, dtype=float).reshape(3)
            vel = np.asarray(est.mean_vel, dtype=float).reshape(3)
            var = float(est.measurement_variance)
            if not slot.initialized:
                slot.initialize(xyz, timestamp, var, velocity_variance=4.0)
            else:
                slot.update_xyz(xyz, timestamp, var, velocity_variance=4.0)
            slot.mean[3:6] = vel
            if hasattr(self.env, "target_belief_mean"):
                self.env.target_belief_mean[i, 0] = xyz
                self.env.target_belief_velocity[i, 0] = vel

    def observe(self, estimates: FleetEstimates | None = None) -> dict:
        if estimates is not None:
            self.inject(estimates)
        return self.env.observe_search()


def broadcast_track(
    mean_xyz: np.ndarray,
    mean_vel: np.ndarray | None = None,
    *,
    n_uavs: int = 3,
    stamp_s: float = 0.0,
    variance: float = 0.25,
) -> list[TargetTrackEstimate]:
    """Same fused track shared to all pursuers (central fusion / leader broadcast)."""
    vel = np.zeros(3) if mean_vel is None else np.asarray(mean_vel, dtype=float)
    base = TargetTrackEstimate(
        mean_xyz=np.asarray(mean_xyz, dtype=float).reshape(3),
        mean_vel=vel.reshape(3),
        initialized=True,
        stamp_s=float(stamp_s),
        measurement_variance=float(variance),
    )
    return [
        TargetTrackEstimate(
            mean_xyz=base.mean_xyz.copy(),
            mean_vel=base.mean_vel.copy(),
            initialized=True,
            stamp_s=base.stamp_s,
            measurement_variance=base.measurement_variance,
        )
        for _ in range(n_uavs)
    ]
