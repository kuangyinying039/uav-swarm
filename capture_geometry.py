"""Shared dual-UAV capture geometry for training, eval, replay, and PX4.

Success semantics
-----------------
* At least ``required_uavs`` *active* pursuers lie within ``radius`` of the target.
* If ``angular_span_deg > 0``, those close pursuers' horizontal bearing span must
  also be at least that many degrees (largest complementary gap on the circle).
* If ``angular_span_deg <= 0``, only the distance / count rule applies
  (Radar5 ``single_distance`` dual-capture protocol).

``min_distance`` / nearest-UAV range is diagnostic only and never implies success.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch always present in training envs
    torch = None


class CaptureResult(NamedTuple):
    success: bool
    close_count: int
    angular_span_deg: float
    kth_gap: float
    min_distance: float


def _as_bool_mask(active_mask, n: int) -> np.ndarray:
    if active_mask is None:
        return np.ones(n, dtype=bool)
    mask = np.asarray(active_mask, dtype=bool).reshape(-1)
    if mask.shape[0] != n:
        raise ValueError(f"active_mask length {mask.shape[0]} != n_uavs {n}")
    return mask


def horizontal_angular_span_deg(positions_xy: np.ndarray, target_xy: np.ndarray) -> float:
    """Largest complementary gap on the unit circle → enclosed horizontal span (deg)."""
    pts = np.asarray(positions_xy, dtype=np.float64).reshape(-1, 2)
    tgt = np.asarray(target_xy, dtype=np.float64).reshape(2)
    if pts.shape[0] < 2:
        return 0.0
    delta = pts - tgt[None, :]
    angles = np.sort(np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi))
    wrapped = np.r_[angles, angles[0] + 2.0 * np.pi]
    largest_gap = float(np.max(np.diff(wrapped)))
    span = 2.0 * np.pi - largest_gap
    return float(np.rad2deg(span))


def evaluate_capture(
    pursuer_xyz: np.ndarray,
    target_xyz: np.ndarray,
    *,
    radius: float,
    required_uavs: int = 2,
    angular_span_deg: float = 0.0,
    active_mask=None,
) -> CaptureResult:
    """Scalar capture check used by Isaac, kinematic env, rollout, and PX4."""
    pos = np.asarray(pursuer_xyz, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_xyz, dtype=np.float64).reshape(3)
    n = pos.shape[0]
    if n < 1:
        raise ValueError("need at least one pursuer")
    if radius <= 0:
        raise ValueError("radius must be positive")
    required = int(required_uavs)
    if required < 1:
        raise ValueError("required_uavs must be >= 1")

    active = _as_bool_mask(active_mask, n)
    distances = np.linalg.norm(pos - tgt[None, :], axis=1)
    min_distance = float(distances[active].min()) if np.any(active) else float("inf")

    close = active & (distances <= float(radius))
    close_count = int(np.count_nonzero(close))
    close_idx = np.flatnonzero(close)

    if np.count_nonzero(active) < required:
        kth_gap = float("inf")
    else:
        kth = float(np.partition(distances[active], required - 1)[required - 1])
        kth_gap = float(max(kth - float(radius), 0.0))

    span = 0.0
    if close_count >= 2:
        span = horizontal_angular_span_deg(pos[close_idx, :2], tgt[:2])

    distance_ok = close_count >= required
    if float(angular_span_deg) > 0.0:
        success = bool(distance_ok and span >= float(angular_span_deg))
    else:
        success = bool(distance_ok)

    return CaptureResult(
        success=success,
        close_count=close_count,
        angular_span_deg=span,
        kth_gap=kth_gap if np.isfinite(kth_gap) else 1e9,
        min_distance=min_distance if np.isfinite(min_distance) else 1e9,
    )


def evaluate_capture_batch(
    pursuer_xyz,
    target_xyz,
    *,
    radius: float,
    required_uavs: int = 2,
    angular_span_deg: float = 0.0,
    active_mask=None,
):
    """Batch API.

    NumPy: pursuer ``(E, N, 3)``, target ``(E, 3)`` → dict of length-E arrays.
    Torch: same shapes on any device → dict of 1-D tensors.
    """
    if torch is not None and isinstance(pursuer_xyz, torch.Tensor):
        return _evaluate_capture_batch_torch(
            pursuer_xyz,
            target_xyz,
            radius=radius,
            required_uavs=required_uavs,
            angular_span_deg=angular_span_deg,
            active_mask=active_mask,
        )

    pos = np.asarray(pursuer_xyz, dtype=np.float64)
    tgt = np.asarray(target_xyz, dtype=np.float64)
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("pursuer_xyz must have shape (E, N, 3)")
    if tgt.shape != (pos.shape[0], 3):
        raise ValueError("target_xyz must have shape (E, 3)")
    e, n, _ = pos.shape
    if active_mask is None:
        active = np.ones((e, n), dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool).reshape(e, n)

    success = np.zeros(e, dtype=bool)
    close_count = np.zeros(e, dtype=np.int32)
    span = np.zeros(e, dtype=np.float64)
    kth_gap = np.zeros(e, dtype=np.float64)
    min_distance = np.zeros(e, dtype=np.float64)
    for i in range(e):
        result = evaluate_capture(
            pos[i],
            tgt[i],
            radius=radius,
            required_uavs=required_uavs,
            angular_span_deg=angular_span_deg,
            active_mask=active[i],
        )
        success[i] = result.success
        close_count[i] = result.close_count
        span[i] = result.angular_span_deg
        kth_gap[i] = result.kth_gap
        min_distance[i] = result.min_distance
    return {
        "success": success,
        "close_count": close_count,
        "angular_span_deg": span,
        "kth_gap": kth_gap,
        "min_distance": min_distance,
    }


def _evaluate_capture_batch_torch(
    pursuer_xyz,
    target_xyz,
    *,
    radius: float,
    required_uavs: int,
    angular_span_deg: float,
    active_mask,
):
    pos = pursuer_xyz
    tgt = target_xyz
    if pos.ndim != 3 or pos.shape[-1] != 3:
        raise ValueError("pursuer_xyz must have shape (E, N, 3)")
    e, n, _ = pos.shape
    if tgt.shape != (e, 3):
        raise ValueError("target_xyz must have shape (E, 3)")
    if active_mask is None:
        active = torch.ones(e, n, dtype=torch.bool, device=pos.device)
    else:
        active = active_mask.to(dtype=torch.bool, device=pos.device).reshape(e, n)

    # Fall back to per-env scalar for angular span correctness (N is tiny).
    success = torch.zeros(e, dtype=torch.bool, device=pos.device)
    close_count = torch.zeros(e, dtype=torch.int32, device=pos.device)
    span = torch.zeros(e, dtype=pos.dtype, device=pos.device)
    kth_gap = torch.zeros(e, dtype=pos.dtype, device=pos.device)
    min_distance = torch.zeros(e, dtype=pos.dtype, device=pos.device)
    pos_np = pos.detach().cpu().numpy()
    tgt_np = tgt.detach().cpu().numpy()
    active_np = active.detach().cpu().numpy()
    for i in range(e):
        result = evaluate_capture(
            pos_np[i],
            tgt_np[i],
            radius=radius,
            required_uavs=required_uavs,
            angular_span_deg=angular_span_deg,
            active_mask=active_np[i],
        )
        success[i] = result.success
        close_count[i] = result.close_count
        span[i] = result.angular_span_deg
        kth_gap[i] = result.kth_gap
        min_distance[i] = result.min_distance
    return {
        "success": success,
        "close_count": close_count,
        "angular_span_deg": span,
        "kth_gap": kth_gap,
        "min_distance": min_distance,
    }


def angular_span_from_cfg(cfg) -> float:
    """Return span threshold for capture; 0 disables the angular gate."""
    mode = str(getattr(cfg, "capture_mode", "single_distance") or "single_distance")
    if mode in ("single_distance", "distance", "k_distance"):
        # Dual-distance protocol used by radar5_dual_*: count only.
        return 0.0
    return float(getattr(cfg, "capture_angular_span_deg", 0.0) or 0.0)
