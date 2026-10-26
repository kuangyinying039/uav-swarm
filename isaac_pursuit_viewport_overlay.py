"""Isaac Viewport overlays for dual-capture pursuit (debug draw).

Capture sphere + lidar range rings. Trails are thin; we deliberately do **not**
draw large points that look like fake sphere drones — pursuers are Hummingbird
USD articulations.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Sequence

import numpy as np


PURSUER_TRAIL_COLORS = (
    (0.15, 0.90, 1.00, 1.0),  # cyan
    (0.35, 1.00, 0.45, 1.0),  # mint
    (1.00, 0.40, 0.90, 1.0),  # magenta
)
EVADER_TRAIL_COLOR = (0.95, 0.15, 0.15, 1.0)


def horizontal_circle_points(
    center: Sequence[float],
    radius: float,
    *,
    segments: int = 48,
) -> list[tuple[float, float, float]]:
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    radius = float(radius)
    points = []
    for index in range(segments):
        angle = 2.0 * np.pi * index / segments
        points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle), cz))
    return points


def closed_polyline_segments(
    points: Sequence[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    if len(points) < 2:
        return [], []
    starts = list(points)
    ends = list(points[1:]) + [points[0]]
    return starts, ends


def open_polyline_segments(
    points: Sequence[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    if len(points) < 2:
        return [], []
    return list(points[:-1]), list(points[1:])


def sphere_wireframe_segments(
    center: Sequence[float],
    radius: float,
    *,
    segments: int = 36,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Three orthogonal great-circle rings approximating a sphere."""
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    radius = float(radius)
    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    for axis in ("xy", "xz", "yz"):
        ring = []
        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            c, s = np.cos(angle), np.sin(angle)
            if axis == "xy":
                ring.append((cx + radius * c, cy + radius * s, cz))
            elif axis == "xz":
                ring.append((cx + radius * c, cy, cz + radius * s))
            else:
                ring.append((cx, cy + radius * c, cz + radius * s))
        s, e = closed_polyline_segments(ring)
        starts.extend(s)
        ends.extend(e)
    return starts, ends


class PursuitViewportOverlay:
    """Per-decision-step debug draw for Radar5 Isaac dual capture."""

    def __init__(
        self,
        env,
        *,
        trail_len: int = 80,
        enabled: bool = True,
        show_radar_rings: bool = True,
        show_target_wireframe: bool = True,
        show_body_markers: bool = False,
        follow_camera: bool = False,
        camera_view: str = "side",
    ):
        self.env = env
        self.trail_len = max(2, int(trail_len))
        self.enabled = bool(enabled)
        self.show_radar_rings = bool(show_radar_rings)
        # Capture-radius wireframe on the target (keep); UAV wireframe spheres off by default.
        self.show_target_wireframe = bool(show_target_wireframe)
        self.show_body_markers = bool(show_body_markers)
        self.follow_camera = bool(follow_camera)
        self.camera_view = str(camera_view or "side")
        self._draw = None
        self._pursuer_trails: list[Deque[tuple[float, float, float]]] = [
            deque(maxlen=self.trail_len) for _ in range(int(env.cfg.n_uavs))
        ]
        self._evader_trail: Deque[tuple[float, float, float]] = deque(maxlen=self.trail_len)
        if self.enabled:
            self._acquire_draw()

    def _acquire_draw(self) -> None:
        try:
            from omni.isaac.core.utils.extensions import enable_extension
            from omni.isaac.debug_draw import _debug_draw

            enable_extension("omni.isaac.debug_draw")
            self._draw = _debug_draw.acquire_debug_draw_interface()
        except Exception as exc:  # pragma: no cover - depends on Kit
            print(f"[viewport-overlay] debug_draw unavailable: {exc}", flush=True)
            self._draw = None
            self.enabled = False

    def reset(self) -> None:
        for trail in self._pursuer_trails:
            trail.clear()
        self._evader_trail.clear()
        if self._draw is not None:
            self._draw.clear_lines()
            if hasattr(self._draw, "clear_points"):
                self._draw.clear_points()

    def _world_origin(self) -> np.ndarray:
        if hasattr(self.env, "_world_origin"):
            return np.asarray(self.env._world_origin(0), dtype=float).reshape(3)
        return np.zeros(3, dtype=float)

    def update(self, result: dict | None = None) -> str:
        """Redraw overlays from the live env state. Returns a HUD status line."""
        origin = self._world_origin()
        physical = np.asarray(self.env._physical_state(), dtype=float)
        # Pose-replay / PhysX recovery: prefer Radar5 bridge kinematics when
        # Direct-GPU PhysX rows are non-finite so trails still follow the chase.
        if not np.isfinite(physical).all():
            physical = np.asarray(self.env.bridge.env.quadrotor_states, dtype=float)
        target = np.asarray(self.env._target_position(), dtype=float) + origin
        world_uavs = physical[:, :3] + origin[None, :]
        for index, trail in enumerate(self._pursuer_trails):
            trail.append(tuple(map(float, world_uavs[index])))
        self._evader_trail.append(tuple(map(float, target[:3])))

        gap = float(result["minimum_capture_gap"]) if result and "minimum_capture_gap" in result else float(
            self.env.minimum_capture_gap()
        )
        captured = bool(result.get("capture_success", False)) if result else bool(self.env.captured)
        visible = 0
        try:
            mask = np.asarray(self.env.bridge.env.direct_visibility_mask(), dtype=bool)
            visible = int(np.count_nonzero(mask))
        except Exception:
            visible = -1
        hud = (
            f"Radar5 dual capture | step={int(self.env.t)} | "
            f"gap={gap:.2f}m | captured={int(captured)} | "
            f"visible={visible}/{int(self.env.cfg.n_uavs)}"
        )

        if self.follow_camera and hasattr(self.env, "follow_camera"):
            try:
                self.env.follow_camera(view=self.camera_view)
            except Exception:
                pass

        if not self.enabled or self._draw is None:
            return hud

        self._draw.clear_lines()
        if hasattr(self._draw, "clear_points"):
            self._draw.clear_points()

        starts: list[tuple[float, float, float]] = []
        ends: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float, float]] = []
        sizes: list[float] = []

        def add_segments(seg_starts, seg_ends, color, size):
            for a, b in zip(seg_starts, seg_ends):
                starts.append(a)
                ends.append(b)
                colors.append(color)
                sizes.append(size)

        # Target capture-radius wireframe only (UAV wireframes stay optional / default off).
        radius = float(self.env.cfg.capture_radius)
        if self.show_target_wireframe:
            cap_color = (0.25, 0.98, 0.45, 1.0) if captured else (0.95, 0.20, 0.20, 1.0)
            s, e = sphere_wireframe_segments(target, radius, segments=32)
            add_segments(s, e, cap_color, 3.0)
        if self.show_body_markers:
            marker_r = max(0.85, 0.9 * radius)
            for index in range(int(self.env.cfg.n_uavs)):
                color = PURSUER_TRAIL_COLORS[index % len(PURSUER_TRAIL_COLORS)]
                s, e = sphere_wireframe_segments(world_uavs[index], marker_r, segments=28)
                add_segments(s, e, (*color[:3], 1.0), 3.0)

        # UAV lidar: full bright ring only when that UAV has clear LOS to target;
        # occluded → dim ring + no detection beam (matches geometric MID-360 LOS).
        if self.show_radar_rings:
            range_m = float(self.env.cfg.lidar_target_detection_range)
            los_mask = None
            try:
                from pursuit_lidar import lidar_visibility

                los_mask = np.asarray(lidar_visibility(self.env.bridge.env), dtype=bool).reshape(-1)
            except Exception:
                try:
                    los_mask = np.asarray(self.env.bridge.env.direct_visibility_mask(), dtype=bool).reshape(-1)
                except Exception:
                    los_mask = None
            for index in range(int(self.env.cfg.n_uavs)):
                center = world_uavs[index]
                color = PURSUER_TRAIL_COLORS[index % len(PURSUER_TRAIL_COLORS)]
                clear = True if los_mask is None else bool(los_mask[index])
                ring_alpha = 0.95 if clear else 0.25
                ring = horizontal_circle_points(center, range_m, segments=48)
                rs, re = closed_polyline_segments(ring)
                add_segments(rs, re, (*color[:3], ring_alpha), 2.8 if clear else 1.4)
                if clear:
                    # Detection beam UAV → target (only when not building-occluded).
                    add_segments([tuple(map(float, center))], [tuple(map(float, target[:3]))], (*color[:3], 0.9), 2.2)
                else:
                    # Short stub toward target to show "blocked".
                    toward = np.asarray(target[:3], dtype=float) - np.asarray(center, dtype=float)
                    norm = float(np.linalg.norm(toward))
                    if norm > 1e-6:
                        stub = np.asarray(center, dtype=float) + toward / norm * min(1.5, 0.15 * norm)
                        add_segments(
                            [tuple(map(float, center))],
                            [tuple(map(float, stub))],
                            (0.95, 0.35, 0.25, 0.8),
                            2.0,
                        )

        # Trails only — no big points (those looked like fake sphere drones).
        for index, trail in enumerate(self._pursuer_trails):
            s, e = open_polyline_segments(list(trail))
            color = PURSUER_TRAIL_COLORS[index % len(PURSUER_TRAIL_COLORS)]
            add_segments(s, e, color, 2.0)
        s, e = open_polyline_segments(list(self._evader_trail))
        add_segments(s, e, EVADER_TRAIL_COLOR, 2.4)

        if starts:
            self._draw.draw_lines(starts, ends, colors, sizes)

        app = getattr(self.env, "simulation_app", None)
        if app is not None and hasattr(app, "update"):
            try:
                app.update()
            except Exception:
                pass
        return hud
