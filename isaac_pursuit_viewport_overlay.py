"""Isaac Viewport overlays for dual-capture pursuit (debug draw).

Capture sphere, radar range rings, and short trajectory trails are drawn each
decision step. These are task overlays on top of the Kit Viewport — not OmniDrones
defaults. Safe to import without Isaac; the draw interface is acquired lazily.
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Sequence

import numpy as np


def horizontal_circle_points(
    center: Sequence[float],
    radius: float,
    *,
    segments: int = 48,
) -> list[tuple[float, float, float]]:
    """Polyline samples for a horizontal circle centered at ``center``."""
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
    """Convert a closed polyline into debug-draw start/end segment lists."""
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
    """Three orthogonal great-circle rings approximating a capture sphere."""
    cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))
    radius = float(radius)
    starts: list[tuple[float, float, float]] = []
    ends: list[tuple[float, float, float]] = []
    rings = []
    # XY (horizontal), XZ, YZ
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
        rings.append(ring)
    for ring in rings:
        s, e = closed_polyline_segments(ring)
        starts.extend(s)
        ends.extend(e)
    return starts, ends


class PursuitViewportOverlay:
    """Per-decision-step debug draw for Radar5 Isaac dual capture."""

    def __init__(self, env, *, trail_len: int = 40, enabled: bool = True):
        self.env = env
        self.trail_len = max(2, int(trail_len))
        self.enabled = bool(enabled)
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

    def update(self, result: dict | None = None) -> str:
        """Redraw overlays from the live env state. Returns a HUD status line."""
        physical = np.asarray(self.env._physical_state(), dtype=float)
        target = np.asarray(self.env._target_position(), dtype=float)
        for index, trail in enumerate(self._pursuer_trails):
            trail.append(tuple(map(float, physical[index, :3])))
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

        # Capture sphere wireframe (yellow / green when captured).
        cap_color = (0.2, 0.95, 0.35, 1.0) if captured else (0.95, 0.85, 0.15, 1.0)
        s, e = sphere_wireframe_segments(target, float(self.env.cfg.capture_radius))
        add_segments(s, e, cap_color, 2.0)

        # Per-pursuer radar range ring at each UAV altitude.
        range_m = float(self.env.cfg.lidar_target_detection_range)
        for index in range(int(self.env.cfg.n_uavs)):
            center = physical[index, :3]
            ring = horizontal_circle_points(center, range_m, segments=40)
            s, e = closed_polyline_segments(ring)
            add_segments(s, e, (0.95, 0.25, 0.25, 0.85), 1.0)

        # Trails.
        for trail in self._pursuer_trails:
            s, e = open_polyline_segments(list(trail))
            add_segments(s, e, (0.95, 0.2, 0.2, 1.0), 2.5)
        s, e = open_polyline_segments(list(self._evader_trail))
        add_segments(s, e, (0.2, 0.45, 0.95, 1.0), 2.5)

        if starts:
            self._draw.draw_lines(starts, ends, colors, sizes)

        # Emphasize current UAV / target positions as points.
        points = [tuple(map(float, physical[i, :3])) for i in range(physical.shape[0])]
        points.append(tuple(map(float, target[:3])))
        point_colors = [(0.95, 0.2, 0.2, 1.0)] * physical.shape[0] + [(0.95, 0.55, 0.1, 1.0)]
        point_sizes = [12.0] * physical.shape[0] + [16.0]
        if hasattr(self._draw, "draw_points"):
            self._draw.draw_points(points, point_colors, point_sizes)

        # Keep the Kit UI pumping when a window is open.
        app = getattr(self.env, "simulation_app", None)
        if app is not None and hasattr(app, "update"):
            try:
                app.update()
            except Exception:
                pass
        return hud
