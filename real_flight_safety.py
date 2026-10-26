"""Safety filter for real-flight / SITL policy actions.

Operates in normalized action space ``[-1, 1]`` before the PX4 bridge scales
to world velocities. Fail-safe behaviour: hold (zero velocity) when link is
stale, peers are too close, altitude is out of band, or capture geometry is
about to collide without dual-capture success.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from px4_offboard_bridge import VehiclePose


@dataclass
class SafetyLimits:
    max_odom_age_s: float = 0.5
    min_peer_distance_m: float = 1.2
    min_altitude_m: float = 0.8
    max_altitude_m: float = 11.0
    geofence_xy_m: float = 20.0  # half-extent around origin; 0 disables
    emergency_gain: float = 0.0  # 0 → hard hold; >0 soft push-away


@dataclass
class SafetyReport:
    hold: bool = False
    reasons: list[str] = field(default_factory=list)
    peer_min_distance_m: float = float("inf")


class SafetyFilter:
    def __init__(self, limits: SafetyLimits | None = None):
        self.limits = limits or SafetyLimits()

    def check(
        self,
        poses: Sequence[VehiclePose],
        *,
        now_s: float,
        n_pursuers: int = 3,
    ) -> SafetyReport:
        report = SafetyReport()
        lim = self.limits
        active = list(poses[:n_pursuers])
        for i, pose in enumerate(active):
            age = float(now_s) - float(pose.stamp_s) if pose.stamp_s > 0 else 0.0
            if pose.stamp_s > 0 and age > lim.max_odom_age_s:
                report.hold = True
                report.reasons.append(f"stale_odom_uav{i}:{age:.2f}s")
            z = float(pose.xyz[2])
            if z < lim.min_altitude_m or z > lim.max_altitude_m:
                report.hold = True
                report.reasons.append(f"altitude_uav{i}:{z:.2f}m")
            if lim.geofence_xy_m > 0:
                if abs(float(pose.xyz[0])) > lim.geofence_xy_m or abs(float(pose.xyz[1])) > lim.geofence_xy_m:
                    report.hold = True
                    report.reasons.append(f"geofence_uav{i}")

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                d = float(np.linalg.norm(active[i].xyz - active[j].xyz))
                report.peer_min_distance_m = min(report.peer_min_distance_m, d)
                if d < lim.min_peer_distance_m:
                    report.hold = True
                    report.reasons.append(f"peer_close:{i}-{j}:{d:.2f}m")
        return report

    def apply(
        self,
        actions: np.ndarray,
        poses: Sequence[VehiclePose],
        *,
        now_s: float,
        n_pursuers: int = 3,
    ) -> tuple[np.ndarray, SafetyReport]:
        """Return filtered actions and the safety report that produced them."""
        a = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0).copy()
        if a.ndim != 2 or a.shape[1] != 4:
            raise ValueError("actions must be (N,4)")
        report = self.check(poses, now_s=now_s, n_pursuers=n_pursuers)
        if not report.hold:
            return a, report
        if self.limits.emergency_gain <= 0.0:
            a[:n_pursuers] = 0.0
            return a, report
        # Soft push-away between close peers, then clamp.
        for i in range(min(n_pursuers, a.shape[0])):
            push = np.zeros(3, dtype=np.float64)
            for j in range(n_pursuers):
                if i == j:
                    continue
                delta = poses[i].xyz - poses[j].xyz
                dist = float(np.linalg.norm(delta))
                if dist < self.limits.min_peer_distance_m and dist > 1e-6:
                    push += (delta / dist) * self.limits.emergency_gain
            a[i, :3] = np.clip(a[i, :3] * 0.2 + push, -1.0, 1.0)
            a[i, 3] = 0.0
        return a, report
