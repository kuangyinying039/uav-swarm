"""PX4 / MAVROS offboard bridge for the frozen sim2real action contract.

Maps Radar5 policy actions ``(..., agents, 4)`` (world vx,vy,vz,yaw_rate) to
MAVLink ``SET_POSITION_TARGET_LOCAL_NED`` velocity + yaw-rate setpoints at 5 Hz.

Dependencies are optional so unit tests run without ROS:
  - ``rospy`` + ``mavros_msgs`` for live MAVROS
  - or a user-supplied ``publisher`` duck type with ``publish(msg)``

See SIM2REAL.md for topic names and capture rules.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from capture_geometry import angular_span_from_cfg, evaluate_capture


@dataclass
class OffboardLimits:
    max_horizontal_velocity: float = 1.4
    max_vertical_velocity: float = 0.8
    max_yaw_rate: float = 1.4
    decision_dt: float = 0.2


@dataclass
class VehiclePose:
    """ENU / world-frame pose used by the policy side."""

    xyz: np.ndarray  # shape (3,)
    yaw: float
    xyz_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rate: float = 0.0
    stamp_s: float = 0.0


def enu_velocity_to_ned(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    """ROS ENU (x east, y north, z up) → NED (x north, y east, z down)."""
    return float(vy), float(vx), float(-vz)


def ned_velocity_to_enu(vn: float, ve: float, vd: float) -> tuple[float, float, float]:
    return float(ve), float(vn), float(-vd)


def policy_action_to_world_velocity(
    action: np.ndarray,
    limits: OffboardLimits | None = None,
) -> np.ndarray:
    """Normalize ``(4,)`` or ``(N,4)`` action → world velocity + yaw rate."""
    limits = limits or OffboardLimits()
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    single = a.ndim == 1
    if single:
        a = a.reshape(1, 4)
    if a.shape[-1] != 4:
        raise ValueError("action last dim must be 4")
    out = np.zeros((a.shape[0], 4), dtype=np.float64)
    out[:, 0] = a[:, 0] * limits.max_horizontal_velocity
    out[:, 1] = a[:, 1] * limits.max_horizontal_velocity
    out[:, 2] = a[:, 2] * limits.max_vertical_velocity
    out[:, 3] = a[:, 3] * limits.max_yaw_rate
    horiz = np.linalg.norm(out[:, :2], axis=1, keepdims=True)
    out[:, :2] *= np.minimum(1.0, limits.max_horizontal_velocity / np.maximum(horiz, 1e-12))
    return out[0] if single else out


class Px4OffboardBridge:
    """One multicopter offboard channel (pursuer or evader)."""

    def __init__(
        self,
        vehicle_ns: str = "/uav0",
        *,
        limits: OffboardLimits | None = None,
        publisher: Callable | None = None,
        frame_id: str = "map",
    ):
        self.vehicle_ns = vehicle_ns.rstrip("/")
        self.limits = limits or OffboardLimits()
        self.frame_id = frame_id
        self._publisher = publisher
        self._last_cmd: np.ndarray | None = None
        self._pose = VehiclePose(xyz=np.zeros(3), yaw=0.0)
        if publisher is None:
            self._try_init_mavros()

    def _try_init_mavros(self) -> None:
        try:
            import rospy
            from mavros_msgs.msg import PositionTarget
            from geometry_msgs.msg import PoseStamped, TwistStamped
        except ImportError:
            self._publisher = None
            return

        if not rospy.core.is_initialized():
            rospy.init_node("radar5_px4_offboard", anonymous=True)
        topic = f"{self.vehicle_ns}/mavros/setpoint_raw/local"
        self._PositionTarget = PositionTarget
        self._publisher = rospy.Publisher(topic, PositionTarget, queue_size=10)
        rospy.Subscriber(
            f"{self.vehicle_ns}/mavros/local_position/pose",
            PoseStamped,
            self._on_pose,
            queue_size=1,
        )
        rospy.Subscriber(
            f"{self.vehicle_ns}/mavros/local_position/velocity_local",
            TwistStamped,
            self._on_vel,
            queue_size=1,
        )

    def _on_pose(self, msg) -> None:
        p = msg.pose.position
        q = msg.pose.orientation
        # yaw from ENU quaternion
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._pose.xyz = np.array([p.x, p.y, p.z], dtype=np.float64)
        self._pose.yaw = math.atan2(siny_cosp, cosy_cosp)
        self._pose.stamp_s = time.time()

    def _on_vel(self, msg) -> None:
        v = msg.twist.linear
        self._pose.xyz_vel = np.array([v.x, v.y, v.z], dtype=np.float64)
        self._pose.yaw_rate = float(msg.twist.angular.z)

    @property
    def pose(self) -> VehiclePose:
        return self._pose

    def send_policy_action(self, action: np.ndarray) -> np.ndarray:
        """Send one 5 Hz setpoint; returns world [vx,vy,vz,yaw_rate]."""
        cmd = policy_action_to_world_velocity(action, self.limits)
        self._last_cmd = cmd
        if self._publisher is None:
            return cmd
        vn, ve, vd = enu_velocity_to_ned(cmd[0], cmd[1], cmd[2])
        msg = self._PositionTarget()
        msg.header.stamp = self._now_stamp()
        msg.header.frame_id = self.frame_id
        # Ignore position & accel; use velocity + yaw rate.
        msg.type_mask = (
            self._PositionTarget.IGNORE_PX
            | self._PositionTarget.IGNORE_PY
            | self._PositionTarget.IGNORE_PZ
            | self._PositionTarget.IGNORE_AFX
            | self._PositionTarget.IGNORE_AFY
            | self._PositionTarget.IGNORE_AFZ
            | self._PositionTarget.IGNORE_YAW
        )
        msg.coordinate_frame = self._PositionTarget.FRAME_LOCAL_NED
        msg.velocity.x = vn
        msg.velocity.y = ve
        msg.velocity.z = vd
        msg.yaw_rate = float(cmd[3])
        self._publisher.publish(msg)
        return cmd

    def _now_stamp(self):
        try:
            import rospy

            return rospy.Time.now()
        except Exception:
            return None

    def hold_zero(self) -> None:
        self.send_policy_action(np.zeros(4, dtype=np.float64))


class FleetOffboardBridge:
    """3 pursuers + optional 1 evader sharing the capture_geometry contract."""

    def __init__(
        self,
        vehicle_namespaces: Sequence[str],
        *,
        limits: OffboardLimits | None = None,
        capture_radius: float = 1.0,
        capture_required_uavs: int = 2,
        capture_mode: str = "single_distance",
        capture_angular_span_deg: float = 100.0,
    ):
        if len(vehicle_namespaces) < 3:
            raise ValueError("need at least 3 pursuer namespaces")
        self.vehicles = [
            Px4OffboardBridge(ns, limits=limits) for ns in vehicle_namespaces
        ]
        self.n_pursuers = 3
        self.has_evader = len(vehicle_namespaces) >= 4
        self.capture_radius = float(capture_radius)
        self.capture_required_uavs = int(capture_required_uavs)
        cfg = type(
            "C",
            (),
            {
                "capture_mode": capture_mode,
                "capture_angular_span_deg": capture_angular_span_deg,
            },
        )()
        self._angular_span_deg = angular_span_from_cfg(cfg)

    def send_actions(self, actions: np.ndarray) -> list[np.ndarray]:
        """``actions`` shape (3,4) pursuers, or (4,4) with trailing evader."""
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 4:
            raise ValueError("actions must be (N,4)")
        cmds = []
        for i in range(min(a.shape[0], len(self.vehicles))):
            cmds.append(self.vehicles[i].send_policy_action(a[i]))
        return cmds

    def evaluate_capture_now(self) -> dict:
        pursuers = np.stack([v.pose.xyz for v in self.vehicles[: self.n_pursuers]])
        if not self.has_evader:
            raise RuntimeError("evader vehicle required for capture eval")
        target = self.vehicles[3].pose.xyz
        result = evaluate_capture(
            pursuers,
            target,
            radius=self.capture_radius,
            required_uavs=self.capture_required_uavs,
            angular_span_deg=self._angular_span_deg,
        )
        return result._asdict()


def spin_setpoint_rate(bridge: Px4OffboardBridge, action_fn, *, hz: float = 5.0, duration_s: float = 10.0):
    """Hold offboard stream at ``hz`` (PX4 requires continuous setpoints)."""
    period = 1.0 / hz
    end = time.time() + duration_s
    while time.time() < end:
        t0 = time.time()
        bridge.send_policy_action(np.asarray(action_fn(), dtype=np.float64))
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)
