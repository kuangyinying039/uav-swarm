"""五栋静态盒子楼，对应原来的建筑棱柱。

底面 3–5 米，高度 5–11.5 米，楼与楼之间至少空 2 米。
视线被楼挡住的条件和原来的三维遮挡一致：线段穿过底面，且该段高度不高于楼顶。
"""

from __future__ import annotations

import math

import numpy as np
import torch

COUNT = 5
MIN_SIZE = 3.0
MAX_SIZE = 5.0
MIN_HEIGHT = 5.0
MAX_HEIGHT = 11.5
GAP = 2.0
EDGE_MARGIN = 1.0
PROTECT = 0.8


def _separated(rect, other, gap: float) -> bool:
    return (
        rect[2] + gap <= other[0]
        or other[2] + gap <= rect[0]
        or rect[3] + gap <= other[1]
        or other[3] + gap <= rect[1]
    )


def _covers_point(rect, point, margin: float) -> bool:
    return (
        rect[0] - margin <= point[0] <= rect[2] + margin
        and rect[1] - margin <= point[1] <= rect[3] + margin
    )


def sample_buildings(rng: np.random.Generator, start: np.ndarray, goal: np.ndarray, bounds: tuple[float, float]) -> list[tuple[float, float, float, float, float]]:
    """返回 (x0, y0, x1, y1, height)。起点和目标点周围留出保护距离。"""
    half_x, half_y = bounds
    protected = (np.asarray(start, dtype=float), np.asarray(goal, dtype=float))
    buildings: list[tuple[float, float, float, float, float]] = []
    for _ in range(COUNT * 80):
        if len(buildings) >= COUNT:
            break
        width = float(rng.uniform(MIN_SIZE, MAX_SIZE))
        depth = float(rng.uniform(MIN_SIZE, MAX_SIZE))
        x_lo = -half_x + EDGE_MARGIN
        y_lo = -half_y + EDGE_MARGIN
        x_hi = half_x - EDGE_MARGIN - width
        y_hi = half_y - EDGE_MARGIN - depth
        if x_hi <= x_lo or y_hi <= y_lo:
            break
        x0 = float(rng.uniform(x_lo, x_hi))
        y0 = float(rng.uniform(y_lo, y_hi))
        rect = (x0, y0, x0 + width, y0 + depth)
        if any(_covers_point(rect, point, PROTECT) for point in protected):
            continue
        if any(not _separated(rect, other, GAP) for other in buildings):
            continue
        height = float(rng.uniform(MIN_HEIGHT, MAX_HEIGHT))
        buildings.append((*rect, height))
    if len(buildings) < COUNT:
        raise RuntimeError("Unable to place five separated buildings")
    return buildings


def point_inside(point: np.ndarray, building: tuple[float, float, float, float, float], margin: float = 0.0) -> bool:
    x0, y0, x1, y1, height = building
    return _covers_point((x0, y0, x1, y1), point, margin) and float(point[2]) <= height + margin


def segment_blocked(start: np.ndarray, end: np.ndarray, buildings: list[tuple[float, float, float, float, float]]) -> bool:
    """线段在水平面上穿过楼的底面，并且这一段不高于楼顶。"""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for x0, y0, x1, y1, height in buildings:
        dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
        clips = ((-dx, start[0] - x0), (dx, x1 - start[0]), (-dy, start[1] - y0), (dy, y1 - start[1]))
        lo, hi = 0.0, 1.0
        missed = False
        for pi, qi in clips:
            if abs(pi) < 1e-12:
                if qi < 0.0:
                    missed = True
                    break
                continue
            ratio = qi / pi
            if pi < 0.0:
                lo = max(lo, ratio)
            else:
                hi = min(hi, ratio)
            if lo > hi:
                missed = True
                break
        if missed:
            continue
        z_lo = float(start[2] + lo * (end[2] - start[2]))
        z_hi = float(start[2] + hi * (end[2] - start[2]))
        if min(z_lo, z_hi) <= height:
            return True
    return False


def building_delta(points: torch.Tensor, buildings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """点到楼体表面的外向向量和距离。points [N,P,3]，buildings [N,B,5]。"""
    x0, y0, x1, y1, height = buildings.unbind(-1)
    px = points[:, :, None, 0]
    py = points[:, :, None, 1]
    pz = points[:, :, None, 2]
    closest_x = torch.minimum(torch.maximum(px, x0[:, None, :]), x1[:, None, :])
    closest_y = torch.minimum(torch.maximum(py, y0[:, None, :]), y1[:, None, :])
    closest_z = torch.minimum(torch.maximum(pz, torch.zeros_like(pz)), height[:, None, :])
    closest = torch.stack((closest_x, closest_y, closest_z), dim=-1)
    delta = points[:, :, None, :] - closest
    distance = torch.linalg.vector_norm(delta, dim=-1)
    center = torch.stack(((x0 + x1) * 0.5, (y0 + y1) * 0.5, height * 0.5), dim=-1)
    outward = points[:, :, None, :] - center[:, None, :, :]
    outward = outward / torch.linalg.vector_norm(outward, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = torch.where(distance.unsqueeze(-1) < 1e-3, outward, delta / distance.clamp_min(1e-6).unsqueeze(-1))
    return direction, distance


def building_repulsion(points: torch.Tensor, buildings: torch.Tensor, gain: float, influence: float = 3.0) -> torch.Tensor:
    direction, distance = building_delta(points, buildings)
    active = distance < influence
    magnitude = 1.0 / distance.clamp_min(0.25) - 1.0 / influence
    return (gain * direction * magnitude.unsqueeze(-1) * active.unsqueeze(-1)).sum(-2)


def footprint_push(xy: torch.Tensor, buildings: torch.Tensor, influence: float = 2.0) -> torch.Tensor:
    """逃逸机水平势场对楼底面的排斥。xy [N,2]。"""
    lifted = torch.cat((xy, torch.full_like(xy[:, :1], 1.0)), dim=-1)
    direction, distance = building_delta(lifted[:, None, :], buildings)
    flat = direction[..., :2]
    flat = flat / torch.linalg.vector_norm(flat, dim=-1, keepdim=True).clamp_min(1e-6)
    strength = (1.0 - distance / influence).clamp(0.0, 2.0)
    return (flat * strength.unsqueeze(-1)).sum(-2).squeeze(1)


def prism_blocks(start: torch.Tensor, end: torch.Tensor, buildings: torch.Tensor) -> torch.Tensor:
    """start/end [N,P,3]，buildings [N,B,5]。返回 [N,P]。"""
    x0, y0, x1, y1, height = buildings.unbind(-1)
    sx, sy, sz = start.unbind(-1)
    ex, ey, ez = end.unbind(-1)
    dx = ex[:, :, None] - sx[:, :, None]
    dy = ey[:, :, None] - sy[:, :, None]
    sx = sx[:, :, None]
    sy = sy[:, :, None]
    sz = sz[:, :, None]
    ez = ez[:, :, None]
    x0, y0, x1, y1, height = x0[:, None, :], y0[:, None, :], x1[:, None, :], y1[:, None, :], height[:, None, :]
    p = torch.stack((-dx, dx, -dy, dy), dim=-1)
    q = torch.stack((sx - x0, x1 - sx, sy - y0, y1 - sy), dim=-1)
    lo = torch.zeros_like(dx)
    hi = torch.ones_like(dx)
    valid = torch.ones_like(dx, dtype=torch.bool)
    for k in range(4):
        pk, qk = p[..., k], q[..., k]
        parallel = pk.abs() < 1e-8
        valid = valid & ~(parallel & (qk < 0))
        ratio = qk / torch.where(parallel, torch.ones_like(pk), pk)
        entering = pk < 0
        lo = torch.where(parallel, lo, torch.where(entering, torch.maximum(lo, ratio), lo))
        hi = torch.where(parallel, hi, torch.where(~entering, torch.minimum(hi, ratio), hi))
        valid = valid & (lo <= hi)
    z_lo = sz + lo * (ez - sz)
    z_hi = sz + hi * (ez - sz)
    blocked = valid & (torch.minimum(z_lo, z_hi) <= height)
    return blocked.any(-1)
