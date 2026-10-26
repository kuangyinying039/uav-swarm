"""5 Hz real-flight / SITL inference loop.

Pipeline per tick:
  odom (+ optional track topic) → OnboardObsBuilder → expert / ResidualPolicy
  → SafetyFilter → FleetOffboardBridge.send_actions

ROS is optional: without rospy the loop still runs against injected poses
(unit tests / dry-run). With MAVROS namespaces the bridge publishes NED
velocity + yaw-rate setpoints.

Example (SITL / dry-run)::

    python real_flight_node.py --dry-run --duration 4

Example (live residual)::

    python real_flight_node.py --checkpoint path/to/mappo.pt --residual-gain 0.3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from px4_offboard_bridge import (
    FleetOffboardBridge,
    OffboardLimits,
    VehiclePose,
)
from real_flight_obs import (
    FleetEstimates,
    OnboardObsBuilder,
    TargetTrackEstimate,
    broadcast_track,
)
from real_flight_safety import SafetyFilter, SafetyLimits
from residual_policy import ResidualPolicy


@dataclass
class DecisionClock:
    """Fixed-rate decision clock aligned with ``decision_dt`` (default 5 Hz)."""

    hz: float = 5.0

    def __post_init__(self) -> None:
        if self.hz <= 0:
            raise ValueError("hz must be positive")
        self.period = 1.0 / float(self.hz)
        self._tick = 0
        self._next_deadline = time.monotonic()

    @property
    def tick(self) -> int:
        return self._tick

    def sleep_to_next(self) -> float:
        """Sleep until the next decision instant; returns scheduled stamp (monotonic).

        Tick 0 fires immediately; subsequent ticks keep a fixed ``1/hz`` grid and
        catch up if a previous decision overran.
        """
        now = time.monotonic()
        if self._tick == 0:
            self._next_deadline = now
        else:
            while self._next_deadline <= now:
                self._next_deadline += self.period
        delay = self._next_deadline - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        stamp = self._next_deadline
        self._next_deadline += self.period
        self._tick += 1
        return stamp


def load_rl_action_fn(checkpoint: str | Path, device: str = "cpu") -> Callable:
    """Load MAPPO/MATD3 actor; returns ``fn(obs) -> (n_agents, 4)`` numpy."""
    import torch

    from evaluate_isaac_pursuit import load_policy
    from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv

    payload = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    method = str(payload.get("method", payload.get("algorithm", "mappo"))).lower()
    if "matd3" in method:
        method = "matd3"
    else:
        method = "mappo"
    # Shadow env only needed to size the trainer; obs come from OnboardObsBuilder.
    env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(**payload["env_config"]))
    action_fn, _trainer = load_policy(method, str(checkpoint), env, device)

    def _fn(obs, env_index: int = 0):
        actions = action_fn(obs)
        return np.asarray(actions, dtype=np.float32)

    return _fn


def make_expert_fn(shadow_env, teacher: str = "frpn") -> Callable:
    from pursuit_baselines_3d import BASELINES_3D

    key = teacher.lower()
    if key not in BASELINES_3D:
        raise ValueError(f"unknown teacher {teacher}; choose {sorted(BASELINES_3D)}")
    policy = BASELINES_3D[key]()
    policy.reset(shadow_env)

    def _fn(obs, env_index: int = 0):
        return np.asarray(policy.actions(shadow_env), dtype=np.float32)

    return _fn


class RealFlightController:
    """Owns obs builder, policy, safety, and offboard bridge."""

    def __init__(
        self,
        bridge: FleetOffboardBridge,
        *,
        obs_builder: OnboardObsBuilder | None = None,
        policy_fn: Callable | None = None,
        safety: SafetyFilter | None = None,
        clock: DecisionClock | None = None,
        track_provider: Callable[[], Sequence[TargetTrackEstimate] | None] | None = None,
        log_path: str | Path | None = None,
    ):
        self.bridge = bridge
        self.obs_builder = obs_builder or OnboardObsBuilder()
        self.policy_fn = policy_fn
        self.safety = safety or SafetyFilter()
        self.clock = clock or DecisionClock(hz=1.0 / bridge.vehicles[0].limits.decision_dt)
        self.track_provider = track_provider
        self._shared_track: TargetTrackEstimate | None = None
        self.log_path = Path(log_path) if log_path else None
        self._log_fh = None
        self.refresh_pose_stamps = False  # dry-run: keep odom age fresh without ROS

    def set_shared_track(
        self,
        xyz: np.ndarray,
        vel: np.ndarray | None = None,
        *,
        stamp_s: float | None = None,
    ) -> None:
        self._shared_track = TargetTrackEstimate(
            mean_xyz=np.asarray(xyz, dtype=float).reshape(3),
            mean_vel=np.zeros(3) if vel is None else np.asarray(vel, dtype=float).reshape(3),
            initialized=True,
            stamp_s=float(time.time() if stamp_s is None else stamp_s),
        )

    def _tracks_for_tick(self) -> list[TargetTrackEstimate] | None:
        if self.track_provider is not None:
            tracks = self.track_provider()
            if tracks is not None:
                return list(tracks)
        if self._shared_track is not None and self._shared_track.initialized:
            return broadcast_track(
                self._shared_track.mean_xyz,
                self._shared_track.mean_vel,
                n_uavs=self.obs_builder.n_uavs,
                stamp_s=self._shared_track.stamp_s,
            )
        return None

    def step_once(self, *, now_s: float | None = None) -> dict:
        now = float(time.time() if now_s is None else now_s)
        if self.refresh_pose_stamps:
            for v in self.bridge.vehicles:
                v._pose.stamp_s = now
        poses = [v.pose for v in self.bridge.vehicles]
        estimates = FleetEstimates(
            vehicles=poses[: self.obs_builder.n_uavs],
            tracks=self._tracks_for_tick(),
            tick=self.clock.tick,
        )
        obs = self.obs_builder.observe(estimates)
        if self.policy_fn is None:
            actions = np.zeros((self.obs_builder.n_uavs, 4), dtype=np.float64)
        else:
            actions = np.asarray(self.policy_fn(obs), dtype=np.float64)
        # Evader slot: hold unless caller extends policy_fn to 4 agents.
        if actions.shape[0] < len(self.bridge.vehicles):
            pad = np.zeros((len(self.bridge.vehicles) - actions.shape[0], 4))
            actions = np.vstack([actions, pad])
        filtered, report = self.safety.apply(
            actions,
            poses,
            now_s=now,
            n_pursuers=self.obs_builder.n_uavs,
        )
        cmds = self.bridge.send_actions(filtered)
        record = {
            "tick": self.clock.tick,
            "stamp_s": now,
            "hold": report.hold,
            "reasons": report.reasons,
            "peer_min_distance_m": report.peer_min_distance_m,
            "actions": filtered.tolist(),
            "cmds_world": [c.tolist() for c in cmds],
        }
        if self.bridge.has_evader:
            try:
                record["capture"] = self.bridge.evaluate_capture_now()
            except Exception as exc:  # pragma: no cover - pose may be unset
                record["capture_error"] = str(exc)
        self._append_log(record)
        return record

    def run(self, *, duration_s: float = 10.0) -> list[dict]:
        end = time.time() + float(duration_s)
        records = []
        while time.time() < end:
            self.clock.sleep_to_next()
            records.append(self.step_once())
        return records

    def _append_log(self, record: dict) -> None:
        if self.log_path is None:
            return
        if self._log_fh is None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("a", encoding="utf-8")
        self._log_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    def close(self) -> None:
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


def build_controller_from_args(args) -> RealFlightController:
    namespaces = [ns.strip() for ns in args.namespaces.split(",") if ns.strip()]
    limits = OffboardLimits(
        max_horizontal_velocity=args.max_h_vel,
        max_vertical_velocity=args.max_v_vel,
        max_yaw_rate=args.max_yaw_rate,
        decision_dt=1.0 / args.hz,
    )
    bridge = FleetOffboardBridge(
        namespaces,
        limits=limits,
        capture_radius=args.capture_radius,
        capture_required_uavs=args.capture_required_uavs,
    )
    refresh_stamps = False
    if args.dry_run:
        # Seed synthetic poses so safety / obs path is exercisable offline.
        refresh_stamps = True
        for i, v in enumerate(bridge.vehicles):
            angle = 2.0 * np.pi * i / max(len(bridge.vehicles), 1)
            v._pose = VehiclePose(
                xyz=np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle), 2.0]),
                yaw=angle,
                xyz_vel=np.zeros(3),
                stamp_s=time.time(),
            )

    obs_builder = OnboardObsBuilder(seed=args.seed)
    expert = make_expert_fn(obs_builder.env, teacher=args.teacher)
    rl_fn = None
    if args.checkpoint:
        rl_fn = load_rl_action_fn(args.checkpoint, device=args.device)
    if args.residual_gain > 0.0 and rl_fn is not None:
        policy_fn = ResidualPolicy(expert, rl_fn, residual_gain=args.residual_gain)
    elif rl_fn is not None and args.residual_gain <= 0.0:
        policy_fn = rl_fn
    else:
        policy_fn = expert

    safety = SafetyFilter(
        SafetyLimits(
            max_odom_age_s=args.max_odom_age,
            min_peer_distance_m=args.min_peer_distance,
            min_altitude_m=args.min_altitude,
            max_altitude_m=args.max_altitude,
            geofence_xy_m=args.geofence_xy,
        )
    )
    controller = RealFlightController(
        bridge,
        obs_builder=obs_builder,
        policy_fn=policy_fn,
        safety=safety,
        clock=DecisionClock(hz=args.hz),
        log_path=args.log,
    )
    controller.refresh_pose_stamps = refresh_stamps
    if args.track_xyz:
        xyz = np.asarray([float(x) for x in args.track_xyz.split(",")], dtype=float)
        controller.set_shared_track(xyz)
    return controller


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Radar5 5 Hz PX4 offboard inference node")
    p.add_argument("--namespaces", default="/uav0,/uav1,/uav2,/uav3")
    p.add_argument("--hz", type=float, default=5.0)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--teacher", default="frpn", choices=("frpn", "apf", "mpc"))
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--residual-gain", type=float, default=0.3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="No ROS; synthetic poses")
    p.add_argument("--log", default=None)
    p.add_argument("--track-xyz", default=None, help="Shared track x,y,z for dry-run")
    p.add_argument("--max-h-vel", type=float, default=1.4)
    p.add_argument("--max-v-vel", type=float, default=0.8)
    p.add_argument("--max-yaw-rate", type=float, default=1.4)
    p.add_argument("--max-odom-age", type=float, default=0.5)
    p.add_argument("--min-peer-distance", type=float, default=1.2)
    p.add_argument("--min-altitude", type=float, default=0.8)
    p.add_argument("--max-altitude", type=float, default=11.0)
    p.add_argument("--geofence-xy", type=float, default=20.0)
    p.add_argument("--capture-radius", type=float, default=1.0)
    p.add_argument("--capture-required-uavs", type=int, default=2)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    controller = build_controller_from_args(args)
    try:
        if args.dry_run and args.track_xyz is None:
            controller.set_shared_track(np.array([0.0, 0.0, 2.0]))
        records = controller.run(duration_s=args.duration)
        holds = sum(1 for r in records if r["hold"])
        print(
            f"[real-flight] ticks={len(records)} holds={holds} "
            f"hz={args.hz} teacher={args.teacher} residual_gain={args.residual_gain}"
        )
        return 0
    finally:
        controller.close()


if __name__ == "__main__":
    raise SystemExit(main())
