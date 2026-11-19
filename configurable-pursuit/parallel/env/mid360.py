"""机载 MID-360 的几何检测，以及同侧三角形出生。

量程、视场、扫描频率和检测概率沿用 hard 闭环配置。位置噪声沿用
机载模型：0.08 米加上每米 0.005 米。水平视场 360°，垂直 -7° 到 52°。
当前场景的障碍是贯穿高度的圆柱和运动圆盘，不是建筑棱柱。
队友位置和场地边界仍直接共享。目标和障碍走雷达。
制导和刹车都只使用本机航迹，没扫到的圆柱不进入期望速度，也不进入刹车。
"""

from __future__ import annotations

import math

import numpy as np
import torch

MIN_RANGE = 0.2
MAX_RANGE = 16.0
HORIZONTAL_FOV_DEG = 360.0
VERTICAL_MIN_DEG = -7.0
VERTICAL_MAX_DEG = 52.0
SCAN_HZ = 10.0
DETECTION_PROBABILITY = 0.95
NOISE_STD = 0.08
NOISE_PER_METER = 0.005
# 单次回波打在柱面上，量不出半径。航迹按场景里最大的柱半径外推圆心。
ASSUMED_RADIUS = 0.55
# 单次测距差分会把 0.1 米噪声放大成数米每秒。平滑并限在逃逸机速度附近，
# 丢检后先停外推，再丢掉航迹，避免追着幽灵点飞出量程。
TRACK_VEL_ALPHA = 0.35
TRACK_SPEED_H = 2.2
TRACK_SPEED_V = 1.2
TRACK_COAST_S = 0.8
TRACK_DROP_S = 2.5
FORMATION_MIN = 8.0
FORMATION_MAX = 10.0
CLEARANCE = 0.45


def _body_local(delta: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1], delta[2]], dtype=float)


def cylinders_block(start: np.ndarray, end: np.ndarray, cylinders: list[tuple[float, float, float]]) -> bool:
    """线段的水平投影是否穿过竖直圆柱。圆柱贯穿飞行高度，从上面绕不过去。"""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    for x, y, radius in cylinders:
        rel = start[:2] - np.array([x, y], dtype=float)
        planar = float(delta[0] ** 2 + delta[1] ** 2)
        if planar <= 1e-12:
            if float(np.dot(rel, rel)) > radius * radius:
                continue
            t_lo, t_hi = 0.0, 1.0
        else:
            linear = 2.0 * float(delta[0] * rel[0] + delta[1] * rel[1])
            constant = float(np.dot(rel, rel) - radius * radius)
            disc = linear * linear - 4.0 * planar * constant
            if disc < 0.0:
                continue
            root = math.sqrt(disc)
            t1 = (-linear - root) / (2.0 * planar)
            t2 = (-linear + root) / (2.0 * planar)
            t_lo = max(min(t1, t2), 0.0)
            t_hi = min(max(t1, t2), 1.0)
            if t_hi < t_lo:
                continue
        return True
    return False


def geometric_visible(origin: np.ndarray, target: np.ndarray, yaw: float, buildings: list[tuple[float, float, float, float, float]]) -> bool:
    """距离、机体方位角、俯仰角和建筑遮挡都通过才算看见。"""
    from parallel.env.buildings import point_inside, segment_blocked

    local = _body_local(np.asarray(target, dtype=float) - np.asarray(origin, dtype=float), yaw)
    distance = float(np.linalg.norm(local))
    azimuth = math.degrees(math.atan2(local[1], local[0]))
    elevation = math.degrees(math.atan2(local[2], max(math.hypot(local[0], local[1]), 1e-12)))
    if not MIN_RANGE <= distance <= MAX_RANGE:
        return False
    if abs(azimuth) > HORIZONTAL_FOV_DEG / 2.0:
        return False
    if not VERTICAL_MIN_DEG <= elevation <= VERTICAL_MAX_DEG:
        return False
    return not segment_blocked(origin, target, buildings) and not any(point_inside(origin, building) for building in buildings)


def one_side_triangle(rng: np.random.Generator, target: np.ndarray, buildings: list[tuple[float, float, float, float, float]], bounds: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """三架追捕机放在目标同一侧，三维距离落在 8–10 米，并且开局就被 MID-360 看见。"""
    radii = np.linspace(FORMATION_MIN + 0.25, FORMATION_MAX - 0.25, 4)
    angle_offsets = (0.0, -math.pi / 18.0, math.pi / 18.0, -math.pi / 9.0, math.pi / 9.0)
    half_x, half_y = bounds
    target = np.asarray(target, dtype=float)

    def candidates(expected_angle: float):
        for radius in radii:
            for angle_offset in angle_offsets:
                angle = expected_angle + angle_offset
                for dz in (0.0, 0.35, -0.35):
                    horizontal = math.sqrt(max(radius * radius - dz * dz, 0.0))
                    point = target + np.array([
                        horizontal * math.cos(angle),
                        horizontal * math.sin(angle),
                        dz,
                    ])
                    if abs(point[0]) > half_x - 0.5 or abs(point[1]) > half_y - 0.5:
                        continue
                    if not 1.0 <= point[2] <= 12.0:
                        continue
                    from parallel.env.buildings import point_inside
                    if any(point_inside(point, building, CLEARANCE) for building in buildings):
                        continue
                    yaw = math.atan2(target[1] - point[1], target[0] - point[0])
                    if geometric_visible(point, target, yaw, buildings):
                        yield point

    base_angle = float(rng.uniform(-math.pi, math.pi))
    anchor = next(iter(candidates(base_angle)), None)
    if anchor is None:
        anchor = next(
            (point for angle in np.linspace(-math.pi, math.pi, 37)[:-1] for point in candidates(float(angle))),
            None,
        )
    if anchor is None:
        raise RuntimeError("Unable to place a visible one-side triangle")
    selected = [anchor]
    anchor_angle = math.atan2(anchor[1] - target[1], anchor[0] - target[0])
    for offset in (-math.pi / 5.0, math.pi / 5.0):
        expected = anchor_angle + offset
        point = next(
            (
                candidate
                for candidate in candidates(expected)
                if all(np.linalg.norm(candidate - other) >= 2.0 * CLEARANCE for other in selected)
            ),
            None,
        )
        if point is None:
            point = next(
                (
                    candidate
                    for angle in np.linspace(expected - math.pi / 8.0, expected + math.pi / 8.0, 17)
                    for candidate in candidates(float(angle))
                    if all(np.linalg.norm(candidate - other) >= 2.0 * CLEARANCE for other in selected)
                ),
                None,
            )
        if point is None:
            raise RuntimeError("Unable to place the other two vertices of the one-side triangle")
        selected.append(point)
    positions = np.asarray(selected, dtype=float)
    yaws = np.arctan2(target[1] - positions[:, 1], target[0] - positions[:, 0])
    return positions, yaws


def one_side_visible(rng: np.random.Generator, target: np.ndarray, cylinders: list[tuple[float, float, float]], bounds: tuple[float, float], preferred_angle: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """交接三角：8–10 米、开局看得见，另外两架在锚点左右各 36 度。

    给出 preferred_angle 时，三架都放在该方向一侧，也就是逃逸机朝向目标的一侧。
    这一侧被圆柱挡住就失败，不改到目标的背面。
    """
    radii = np.linspace(FORMATION_MIN + 0.25, FORMATION_MAX - 0.25, 4)
    angle_offsets = (0.0, -math.pi / 18.0, math.pi / 18.0, -math.pi / 9.0, math.pi / 9.0)
    half_x, half_y = bounds
    target = np.asarray(target, dtype=float)
    body = 0.3

    def clear(point: np.ndarray) -> bool:
        for x, y, radius in cylinders:
            if math.hypot(point[0] - x, point[1] - y) <= radius + body + CLEARANCE:
                return False
        return True

    def sees(point: np.ndarray) -> bool:
        yaw = math.atan2(target[1] - point[1], target[0] - point[0])
        local = _body_local(target - point, yaw)
        distance = float(np.linalg.norm(local))
        elevation = math.degrees(math.atan2(local[2], max(math.hypot(local[0], local[1]), 1e-12)))
        if not MIN_RANGE <= distance <= MAX_RANGE:
            return False
        if not VERTICAL_MIN_DEG <= elevation <= VERTICAL_MAX_DEG:
            return False
        return not cylinders_block(point, target, cylinders)

    def candidates(expected_angle: float):
        for radius in radii:
            for angle_offset in angle_offsets:
                angle = expected_angle + angle_offset
                for dz in (0.0, 0.35, -0.35):
                    horizontal = math.sqrt(max(radius * radius - dz * dz, 0.0))
                    point = target + np.array([
                        horizontal * math.cos(angle),
                        horizontal * math.sin(angle),
                        dz,
                    ])
                    if abs(point[0]) > half_x - 0.5 or abs(point[1]) > half_y - 0.5:
                        continue
                    if not 1.0 <= point[2] <= 12.0 or not clear(point) or not sees(point):
                        continue
                    yield point

    if preferred_angle is None:
        base_angle = float(rng.uniform(-math.pi, math.pi))
        sweep = np.linspace(-math.pi, math.pi, 37, endpoint=False)
    else:
        base_angle = float(preferred_angle)
        sweep = np.linspace(base_angle - math.pi / 2.0, base_angle + math.pi / 2.0, 25)
    anchor = next(iter(candidates(base_angle)), None)
    if anchor is None:
        anchor = next(
            (point for angle in sweep for point in candidates(float(angle))),
            None,
        )
    if anchor is None:
        raise RuntimeError("Unable to place a visible one-side formation among cylinders")
    selected = [anchor]
    anchor_angle = math.atan2(anchor[1] - target[1], anchor[0] - target[0])
    for offset in (-math.pi / 5.0, math.pi / 5.0):
        expected = anchor_angle + offset
        point = next(
            (
                candidate
                for candidate in candidates(expected)
                if all(np.linalg.norm(candidate - other) >= 2.0 * CLEARANCE for other in selected)
            ),
            None,
        )
        if point is None:
            point = next(
                (
                    candidate
                    for angle in np.linspace(expected - math.pi / 8.0, expected + math.pi / 8.0, 17)
                    for candidate in candidates(float(angle))
                    if all(np.linalg.norm(candidate - other) >= 2.0 * CLEARANCE for other in selected)
                ),
                None,
            )
        if point is None:
            raise RuntimeError("Unable to place a visible one-side formation among cylinders")
        selected.append(point)
    positions = np.asarray(selected, dtype=float)
    yaws = np.arctan2(target[1] - positions[:, 1], target[0] - positions[:, 0])
    return positions, yaws


def _torch_blocked(start: torch.Tensor, end: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """start/end [N,P,3]，圆柱 [N,S,2] 与 [N,S]。圆柱贯穿高度，只看水平投影。返回 [N,P]。"""
    delta = end - start
    rel = start[:, :, None, :2] - centers[:, None, :, :]
    planar = delta[..., 0].square() + delta[..., 1].square()
    linear = 2.0 * (delta[:, :, None, :2] * rel).sum(-1)
    constant = rel.square().sum(-1) - radii[:, None, :].square()
    disc = linear.square() - 4.0 * planar[:, :, None] * constant
    denom = (2.0 * planar[:, :, None]).clamp_min(1e-12)
    root = disc.clamp_min(0).sqrt()
    t1 = (-linear - root) / denom
    t2 = (-linear + root) / denom
    t_lo = torch.minimum(t1, t2).clamp(0.0, 1.0)
    t_hi = torch.maximum(t1, t2).clamp(0.0, 1.0)
    crossing = (disc >= 0) & (t_hi >= t_lo) & (planar[:, :, None] > 1e-12)
    vertical = (planar[:, :, None] <= 1e-12) & (rel.square().sum(-1) <= radii[:, None, :].square())
    return (crossing | vertical).any(-1)


def geometric_visible_batch(origin: torch.Tensor, target: torch.Tensor, yaw: torch.Tensor, buildings: torch.Tensor) -> torch.Tensor:
    """origin/yaw [N,P,*]，target [N,3]，buildings [N,B,5]。返回 [N,P]。"""
    from parallel.env.buildings import prism_blocks

    delta = target[:, None, :] - origin
    cosine, sine = yaw.cos(), yaw.sin()
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    local_z = delta[..., 2]
    distance = torch.sqrt(local_x.square() + local_y.square() + local_z.square())
    azimuth = torch.atan2(local_y, local_x) * (180.0 / math.pi)
    elevation = torch.atan2(local_z, torch.hypot(local_x, local_y).clamp_min(1e-6)) * (180.0 / math.pi)
    in_range = (distance >= MIN_RANGE) & (distance <= MAX_RANGE)
    in_fov = (azimuth.abs() <= HORIZONTAL_FOV_DEG / 2.0) & (elevation >= VERTICAL_MIN_DEG) & (elevation <= VERTICAL_MAX_DEG)
    blocked = prism_blocks(origin, target[:, None, :].expand_as(origin), buildings) if buildings.shape[1] else torch.zeros_like(in_range)
    return in_range & in_fov & ~blocked


class Mid360Tracks:
    """每架追捕机一份目标航迹。新扫描 10 Hz 一次，中间按上次速度外推。"""

    def __init__(self, envs: int, pursuers: int, device: torch.device):
        self.position = torch.zeros(envs, pursuers, 3, device=device)
        self.velocity = torch.zeros(envs, pursuers, 3, device=device)
        self.valid = torch.zeros(envs, pursuers, dtype=torch.bool, device=device)
        self.visible = torch.zeros_like(self.valid)
        self.measured = torch.zeros(envs, pursuers, 3, device=device)
        self.time = torch.zeros(envs, device=device)
        self.steps = torch.zeros(envs, dtype=torch.long, device=device)
        self.scan = torch.full((envs,), -1, dtype=torch.long, device=device)
        self.last_measure = torch.zeros(envs, pursuers, device=device)
        self.obstacle_count = 0
        self.obstacle_center = torch.zeros(envs, pursuers, 0, 2, device=device)
        self.obstacle_measured = torch.zeros_like(self.obstacle_center)
        self.obstacle_velocity = torch.zeros_like(self.obstacle_center)
        self.obstacle_radius = torch.zeros(envs, pursuers, 0, device=device)
        self.obstacle_valid = torch.zeros(envs, pursuers, 0, dtype=torch.bool, device=device)
        self.obstacle_last = torch.zeros(envs, pursuers, 0, device=device)

    def reset(self, ids: torch.Tensor, state: torch.Tensor, disabled: torch.Tensor, buildings: torch.Tensor, cylinders=None) -> None:
        self.time[ids] = 0
        self.steps[ids] = 0
        self.scan[ids] = -1
        self.valid[ids] = False
        self.velocity[ids] = 0
        self.position[ids] = state[ids, 1:, :3]
        if self.obstacle_count:
            self.obstacle_valid[ids] = False
            self.obstacle_velocity[ids] = 0
            self.obstacle_last[ids] = 0
        self._measure(ids, state, disabled, buildings, opening=True, scan_index=torch.zeros(ids.shape, dtype=torch.long, device=ids.device), cylinders=cylinders)

    def advance(self, state: torch.Tensor, disabled: torch.Tensor, dt: float, buildings: torch.Tensor, cylinders=None) -> None:
        active = self.valid & ~disabled
        self.position = torch.where(active.unsqueeze(-1), self.position + self.velocity * dt, self.position)
        if self.obstacle_count:
            moving = self.obstacle_valid.unsqueeze(-1)
            self.obstacle_center = torch.where(moving, self.obstacle_center + self.obstacle_velocity * dt, self.obstacle_center)
        self.time = self.time + dt
        self.steps = self.steps + 1
        period = max(1, int(round(1.0 / (dt * SCAN_HZ))))
        scan = torch.div(self.steps, period, rounding_mode="floor")
        fresh = scan != self.scan
        if bool(fresh.any()):
            chosen = fresh.nonzero(as_tuple=False).flatten()
            self._measure(chosen, state, disabled, buildings, opening=False, scan_index=scan[chosen], cylinders=cylinders)
        age = self.time[:, None] - self.last_measure
        coast = self.valid & (age > TRACK_COAST_S)
        self.velocity = torch.where(coast.unsqueeze(-1), torch.zeros_like(self.velocity), self.velocity)
        self.valid = self.valid & (age <= TRACK_DROP_S)
        if self.obstacle_count:
            obstacle_age = self.time[:, None, None] - self.obstacle_last
            obstacle_coast = self.obstacle_valid & (obstacle_age > TRACK_COAST_S)
            self.obstacle_velocity = torch.where(obstacle_coast.unsqueeze(-1), torch.zeros_like(self.obstacle_velocity), self.obstacle_velocity)
            self.obstacle_valid = self.obstacle_valid & (obstacle_age <= TRACK_DROP_S)

    def _measure(self, ids: torch.Tensor, state: torch.Tensor, disabled: torch.Tensor, buildings: torch.Tensor, opening: bool, scan_index: torch.Tensor, cylinders=None) -> None:
        if not ids.numel():
            return
        from parallel.env.radar5_contract import body_yaw

        origin = state[ids, 1:, :3]
        yaw = body_yaw(state[ids, 1:, 3:7])
        target = state[ids, 0, :3]
        visible = geometric_visible_batch(origin, target, yaw, buildings[ids])
        if cylinders is not None:
            centers, radii = cylinders
            centers, radii = centers[ids], radii[ids]
            if centers.shape[1]:
                visible = visible & ~_torch_blocked(
                    origin, target[:, None, :].expand_as(origin), centers, radii
                )
        visible = visible & ~disabled[ids]
        if opening:
            detected = visible
        else:
            detected = visible & (torch.rand(visible.shape, device=visible.device) < DETECTION_PROBABILITY)
        distance = torch.linalg.vector_norm(target[:, None, :] - origin, dim=-1).clamp_min(1e-6)
        sigma = NOISE_STD + NOISE_PER_METER * distance
        noise = torch.randn(origin.shape, device=origin.device, dtype=origin.dtype) * sigma.unsqueeze(-1)
        measurement = target[:, None, :] + noise
        elapsed = (self.time[ids, None] - self.last_measure[ids]).clamp_min(1.0 / SCAN_HZ)
        had = self.valid[ids]
        raw = (measurement - self.measured[ids]) / elapsed.unsqueeze(-1)
        horizontal = torch.linalg.vector_norm(raw[..., :2], dim=-1, keepdim=True).clamp_min(1e-6)
        raw = torch.cat((
            raw[..., :2] * torch.clamp(TRACK_SPEED_H / horizontal, max=1.0),
            raw[..., 2:3].clamp(-TRACK_SPEED_V, TRACK_SPEED_V),
        ), dim=-1)
        blended = (1.0 - TRACK_VEL_ALPHA) * self.velocity[ids] + TRACK_VEL_ALPHA * raw
        velocity = torch.where(had.unsqueeze(-1), blended, torch.zeros_like(raw))
        self.visible[ids] = visible
        self.measured[ids] = torch.where(detected.unsqueeze(-1), measurement, self.measured[ids])
        self.position[ids] = torch.where(detected.unsqueeze(-1), measurement, self.position[ids])
        self.velocity[ids] = torch.where(detected.unsqueeze(-1), velocity, self.velocity[ids])
        self.valid[ids] = self.valid[ids] | detected
        self.scan[ids] = scan_index
        self.last_measure[ids] = torch.where(detected, self.time[ids, None], self.last_measure[ids])
        if cylinders is not None and cylinders[0].shape[1]:
            self._measure_obstacles(ids, origin, disabled[ids], cylinders[0][ids], cylinders[1][ids], opening)

    def _allocate_obstacles(self, count: int) -> None:
        if count == self.obstacle_count:
            return
        envs, pursuers = self.position.shape[:2]
        device = self.position.device
        self.obstacle_count = count
        self.obstacle_center = torch.zeros(envs, pursuers, count, 2, device=device)
        self.obstacle_measured = torch.zeros_like(self.obstacle_center)
        self.obstacle_velocity = torch.zeros_like(self.obstacle_center)
        self.obstacle_radius = torch.zeros(envs, pursuers, count, device=device)
        self.obstacle_valid = torch.zeros(envs, pursuers, count, dtype=torch.bool, device=device)
        self.obstacle_last = torch.zeros(envs, pursuers, count, device=device)

    def _measure_obstacles(self, ids, origin, disabled, centers, radii, opening: bool) -> None:
        """每架只保留自己扫到的圆柱。回波是近侧面的点，半径用场景上限，不记真值。"""
        self._allocate_obstacles(int(centers.shape[1]))
        visible = cylinder_visible(origin, centers, radii) & ~disabled.unsqueeze(-1)
        detected = visible if opening else visible & (torch.rand(visible.shape, device=visible.device) < DETECTION_PROBABILITY)
        offset = centers[:, None, :, :] - origin[:, :, None, :2]
        distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(1e-6)
        direction = offset / distance.unsqueeze(-1)
        surface = (distance - radii[:, None, :]).clamp_min(0.0)
        hit = centers[:, None, :, :] - direction * radii[:, None, :, None]
        sigma = NOISE_STD + NOISE_PER_METER * surface
        noisy = hit + torch.randn(detected.shape + (2,), device=origin.device, dtype=origin.dtype) * sigma.unsqueeze(-1)
        bearing = noisy - origin[:, :, None, :2]
        bearing = bearing / torch.linalg.vector_norm(bearing, dim=-1, keepdim=True).clamp_min(1e-6)
        measured = noisy + bearing * ASSUMED_RADIUS
        elapsed = (self.time[ids, None, None] - self.obstacle_last[ids]).clamp_min(1.0 / SCAN_HZ)
        had = self.obstacle_valid[ids]
        raw = (measured - self.obstacle_measured[ids]) / elapsed.unsqueeze(-1)
        speed = torch.linalg.vector_norm(raw, dim=-1, keepdim=True).clamp_min(1e-6)
        raw = raw * torch.clamp(TRACK_SPEED_H / speed, max=1.0)
        blended = (1.0 - TRACK_VEL_ALPHA) * self.obstacle_velocity[ids] + TRACK_VEL_ALPHA * raw
        velocity = torch.where(had.unsqueeze(-1), blended, torch.zeros_like(raw))
        detected_c = detected.unsqueeze(-1)
        self.obstacle_measured[ids] = torch.where(detected_c, measured, self.obstacle_measured[ids])
        self.obstacle_center[ids] = torch.where(detected_c, measured, self.obstacle_center[ids])
        self.obstacle_velocity[ids] = torch.where(detected_c, velocity, self.obstacle_velocity[ids])
        self.obstacle_radius[ids] = torch.where(
            detected,
            torch.full_like(self.obstacle_radius[ids], ASSUMED_RADIUS),
            self.obstacle_radius[ids],
        )
        self.obstacle_valid[ids] = self.obstacle_valid[ids] | detected
        self.obstacle_last[ids] = torch.where(detected, self.time[ids, None, None], self.obstacle_last[ids])


def cylinder_visible(origin: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    """圆柱近侧面是否落进量程，且没有被另一根贯穿柱挡住。origin [N,P,3]，圆柱 [N,S,*]。"""
    count = int(centers.shape[1])
    if count == 0:
        return torch.zeros(origin.shape[:2] + (0,), dtype=torch.bool, device=origin.device)
    offset = centers[:, None, :, :] - origin[:, :, None, :2]
    distance = torch.linalg.vector_norm(offset, dim=-1).clamp_min(1e-6)
    surface = (distance - radii[:, None, :]).clamp_min(0.0)
    in_range = (surface >= MIN_RANGE) & (surface <= MAX_RANGE)
    direction = offset / distance.unsqueeze(-1)
    near = centers[:, None, :, :] - direction * radii[:, None, :, None]
    start = origin[:, :, None, :2].expand_as(near)
    delta = near - start
    relative = start[:, :, :, None, :] - centers[:, None, None, :, :]
    planar = delta.square().sum(-1).clamp_min(1e-12)
    linear = 2.0 * (delta[:, :, :, None, :] * relative).sum(-1)
    constant = relative.square().sum(-1) - radii[:, None, None, :].square()
    disc = linear.square() - 4.0 * planar[:, :, :, None] * constant
    denom = (2.0 * planar[:, :, :, None]).clamp_min(1e-12)
    root = disc.clamp_min(0.0).sqrt()
    t_lo = torch.minimum((-linear - root) / denom, (-linear + root) / denom)
    t_hi = torch.maximum((-linear - root) / denom, (-linear + root) / denom)
    overlap = (disc >= 0.0) & (t_lo < 0.98) & (t_hi > 0.02)
    eye = torch.eye(count, device=origin.device, dtype=torch.bool).view(1, 1, count, count)
    blocked = (overlap & ~eye).any(-1)
    return in_range & ~blocked


def sensed_cylinders(env):
    """每架追捕机自己的障碍航迹：圆心、半径、速度、是否仍有效、是否在动。

    没有雷达时退回整张地图，供真值感知使用。有雷达时没扫到的圆柱不出现。
    """
    state = env.state()
    agents = state.shape[1] - 1
    tracker = getattr(env, "mid360", None)
    if tracker is not None:
        if tracker.obstacle_count and tracker.obstacle_center.shape[1] == agents:
            moving = torch.linalg.vector_norm(tracker.obstacle_velocity, dim=-1) > 0.3
            return (
                tracker.obstacle_center,
                tracker.obstacle_radius,
                tracker.obstacle_velocity,
                tracker.obstacle_valid,
                moving,
            )
        empty_center = state.new_zeros(state.shape[0], agents, 0, 2)
        empty_radius = state.new_zeros(state.shape[0], agents, 0)
        empty_valid = torch.zeros(state.shape[0], agents, 0, dtype=torch.bool, device=state.device)
        return empty_center, empty_radius, empty_center.clone(), empty_valid, empty_valid
    centers, radii = obstacle_cylinders(env)
    motion = torch.zeros(*centers.shape, device=centers.device, dtype=centers.dtype)
    static_count = int(env.rules.static.shape[1])
    dynamic = torch.zeros(centers.shape[:2], dtype=torch.bool, device=centers.device)
    if env.rules.bg.shape[1]:
        motion[:, static_count:] = env.rules.bgvel
        dynamic[:, static_count:] = True
    count = centers.shape[0]
    return (
        centers[:, None].expand(count, agents, -1, -1),
        radii[:, None].expand(count, agents, -1),
        motion[:, None].expand(count, agents, -1, -1),
        torch.ones(count, agents, centers.shape[1], dtype=torch.bool, device=centers.device),
        dynamic[:, None].expand(count, agents, -1),
    )


def obstacle_cylinders(env) -> tuple[torch.Tensor, torch.Tensor]:
    static = env.rules.static
    centers = [static[..., :2]]
    radii = [static[..., 2]]
    if env.rules.bg.shape[1]:
        centers.append(env.rules.bg)
        radii.append(env.rules.radii[:, env.rules.layout.dynamic_objects])
    return torch.cat(centers, 1), torch.cat(radii, 1)


def sensed_target(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """追捕机各自的目标位置、速度和是否已有航迹。没有雷达时退回仿真真值。

    有雷达时不读仿真真值。某架没有航迹时，采用队友里最新的一份机载航迹。
    """
    state = env.state()
    own = state[:, 1:, :3]
    tracker = getattr(env, "mid360", None)
    if tracker is None:
        valid = torch.ones(own.shape[:2], dtype=torch.bool, device=own.device)
        return state[:, :1, :3].expand_as(own), state[:, :1, 7:10].expand_as(own), valid
    fresh = tracker.valid
    age = tracker.time[:, None] - tracker.last_measure
    hold = (tracker.last_measure > 0) & ~fresh & (age <= 8.0)
    position = tracker.position
    velocity = torch.where(fresh.unsqueeze(-1), tracker.velocity, torch.zeros_like(tracker.velocity))
    usable = fresh | hold
    age_fresh = age.masked_fill(~fresh, 1.0e6)
    donor = age_fresh.argmin(dim=1)
    index = donor.view(-1, 1, 1).expand(-1, 1, 3)
    shared = fresh.any(dim=1)
    missing = ~usable & shared[:, None]
    position = torch.where(missing.unsqueeze(-1), position.gather(1, index), position)
    velocity = torch.where(missing.unsqueeze(-1), velocity.gather(1, index), velocity)
    usable = usable | missing
    position = torch.where(usable.unsqueeze(-1), position, own)
    velocity = torch.where(usable.unsqueeze(-1), velocity, torch.zeros_like(own))
    return position, velocity, usable
