"""APF、FRPN 和协同 MPC 在 PEG 并行环境里的批量制导。

没有机载航迹时直接使用仿真里的位置和速度。圆柱和运动障碍计入排斥。
输出是世界系速度和偏航角速度。
"""

from __future__ import annotations

import math

import torch

from parallel.env.buildings import building_delta, building_repulsion
from parallel.env.geometry import segment_clearance
from parallel.env.mid360 import sensed_target
from parallel.env.radar5_contract import (
    BODY_CONTACT,
    BODY_RADIUS,
    EVADER_VERTICAL,
    MAX_YAW_RATE,
    PURSUER_HORIZONTAL,
    PURSUER_VERTICAL,
    body_yaw,
)
from parallel.examples.classical import cbf_qp, mpc_velocity as receding_mpc, orca_velocity
from parallel.training.mappo import evader_world_velocity

INFLUENCE = 3.0
DECISION_DT = 0.2
MIN_ALTITUDE = 1.0
MAX_ALTITUDE = 12.0
DODGE_RANGE = 1.6
PEER_PUSH = 0.8
BRAKE_ACCEL = 2.5
RESPONSE_TIME = 0.35
SAFETY_BUFFER = 0.25


def _unit(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6)


def _yaw_rate(velocity_xy: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    yaw = body_yaw(quat)
    desired = torch.atan2(velocity_xy[..., 1], velocity_xy[..., 0])
    error = torch.atan2(torch.sin(desired - yaw), torch.cos(desired - yaw))
    moving = torch.linalg.vector_norm(velocity_xy, dim=-1) > 1e-6
    error = torch.where(moving, error, torch.zeros_like(error))
    return (2.0 * error).clamp(-MAX_YAW_RATE, MAX_YAW_RATE)


def keep_in_band(velocity: torch.Tensor, height: torch.Tensor, limit: float) -> torch.Tensor:
    """垂直速度用来拉开高度，同时拉回 5 米附近，避免贴地滑行。"""
    limited = velocity.clone()
    restore = (5.0 - height).clamp(-limit, limit)
    vertical = 0.55 * limited[..., 2] + 0.45 * restore
    vertical = torch.where(height < 2.0, vertical.clamp(min=0.7), vertical)
    vertical = torch.where(height > 10.0, vertical.clamp(max=-0.7), vertical)
    limited[..., 2] = vertical.clamp(-limit, limit)
    return limited


def clip_speed(velocity: torch.Tensor, horizontal, vertical) -> torch.Tensor:
    limited = velocity.clone()
    speed = torch.linalg.vector_norm(limited[..., :2], dim=-1, keepdim=True)
    cap = horizontal if torch.is_tensor(horizontal) else speed.new_tensor(float(horizontal))
    while cap.ndim < speed.ndim:
        cap = cap.unsqueeze(-1)
    up = vertical if torch.is_tensor(vertical) else limited.new_tensor(float(vertical))
    while up.ndim < limited[..., 2].ndim:
        up = up.unsqueeze(-1)
    limited[..., :2] = limited[..., :2] * torch.clamp(cap / speed.clamp_min(1e-6), max=1.0)
    limited[..., 2] = limited[..., 2].clamp(-up, up)
    return limited


def clip_pursuer_velocity(velocity: torch.Tensor) -> torch.Tensor:
    return clip_speed(velocity, PURSUER_HORIZONTAL, PURSUER_VERTICAL)


def _as_rows(limit, count: int, device, dtype) -> torch.Tensor:
    value = limit if torch.is_tensor(limit) else torch.as_tensor(float(limit), device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype).reshape(-1)
    if value.numel() == 1:
        value = value.expand(count)
    return value


def _brake_cylinders(env, agents: int):
    """刹车和制导共用本机障碍航迹。调用方不是追捕机时退回地图。"""
    from parallel.env.mid360 import sensed_cylinders

    centers, radii, motion, valid, _moving = sensed_cylinders(env)
    if centers.shape[1] != agents:
        centers, radii = obstacle_bank(env)
        motion = torch.zeros(*centers.shape, device=centers.device, dtype=centers.dtype)
        static_count = int(env.rules.static.shape[1])
        if env.rules.bg.shape[1]:
            motion[:, static_count:] = env.rules.bgvel
        return centers.unsqueeze(1), radii.unsqueeze(1), motion.unsqueeze(1), None
    return centers, radii, motion, valid


def _intent_cylinders(env, points: torch.Tensor):
    """期望速度只看最近的 3 条有效航迹，和学员图里的障碍节点一致。刹车仍使用全部航迹。"""
    centers, radii, motion, valid = _brake_cylinders(env, points.shape[1])
    keep = 3
    if valid is None or centers.shape[2] <= keep:
        return centers, radii, motion, valid
    gap = torch.linalg.vector_norm(points[:, :, None, :2] - centers[..., :2], dim=-1)
    gap = gap.masked_fill(~valid, 1.0e6)
    index = torch.topk(gap, keep, dim=-1, largest=False).indices
    centers = centers.gather(2, index.unsqueeze(-1).expand(*index.shape, centers.shape[-1]))
    radii = radii.gather(2, index)
    if motion.ndim == centers.ndim:
        motion = motion.gather(2, index.unsqueeze(-1).expand(*index.shape, motion.shape[-1]))
    else:
        motion = motion.gather(2, index)
    valid = valid.gather(2, index)
    return centers, radii, motion, valid


def shield_velocity(velocity: torch.Tensor, points: torch.Tensor, env, *, peers: bool, horizontal, vertical) -> torch.Tensor:
    """半空间刹车预算，再保证这一拍的扫掠不进柱体和队友。

    规则与原先的 project_velocity 相同：允许的靠近速度由刹车距离决定。
    逃逸机不是约束，捕获接触仍然允许。
    """
    limited = clip_speed(velocity, horizontal, vertical)
    normals, clearances, offsets = [], [], []
    centers, radii, motion, valid = _brake_cylinders(env, points.shape[1])
    if centers.shape[2]:
        delta = points[:, :, None, :2] - centers
        radial = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        outward = delta / radial.unsqueeze(-1)
        normal = torch.cat((outward, torch.zeros_like(points[:, :, None, 2:3]).expand(*outward.shape[:3], 1)), dim=-1)
        clearance = radial - radii - BODY_RADIUS
        if valid is not None:
            clearance = torch.where(valid, clearance, torch.full_like(clearance, 1.0e3))
        normals.append(normal)
        clearances.append(clearance)
        offsets.append((outward * motion).sum(-1))
    half_x = float(env.rules.cfg["boundary_half_extent_x"])
    half_y = float(env.rules.cfg["boundary_half_extent_y"])
    wall = (
        ((1.0, 0.0, 0.0), points[..., 0] + half_x - BODY_RADIUS),
        ((-1.0, 0.0, 0.0), half_x - BODY_RADIUS - points[..., 0]),
        ((0.0, 1.0, 0.0), points[..., 1] + half_y - BODY_RADIUS),
        ((0.0, -1.0, 0.0), half_y - BODY_RADIUS - points[..., 1]),
        ((0.0, 0.0, 1.0), points[..., 2] - MIN_ALTITUDE),
        ((0.0, 0.0, -1.0), MAX_ALTITUDE - points[..., 2]),
    )
    for direction, gap in wall:
        normals.append(points.new_tensor(direction).view(1, 1, 1, 3).expand(*points.shape[:2], 1, 3))
        clearances.append(gap.unsqueeze(-1))
        offsets.append(torch.zeros_like(gap).unsqueeze(-1))
    if peers and points.shape[1] > 1:
        delta = points[:, :, None, :] - points[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        normal = delta / distance.unsqueeze(-1)
        eye = torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
        normals.append(normal)
        clearances.append(torch.where(eye, torch.full_like(distance, 10.0), distance - BODY_CONTACT))
        offsets.append(torch.zeros_like(distance))
    if normals:
        normal = torch.cat(normals, dim=2)
        clearance = torch.cat(clearances, dim=2)
        offset = torch.cat(offsets, dim=2)
        delay = RESPONSE_TIME + DECISION_DT
        gap = (clearance - SAFETY_BUFFER).clamp(min=0.0)
        allowed = torch.sqrt((BRAKE_ACCEL * delay) ** 2 + 2.0 * BRAKE_ACCEL * gap) - BRAKE_ACCEL * delay
        push = ((SAFETY_BUFFER - clearance).clamp(min=0.0) / delay).clamp(max=0.3)
        lower = offset - allowed + push
        for _ in range(4):
            closing = (normal * limited[:, :, None, :]).sum(-1)
            violation = (lower - closing).clamp(min=0.0)
            limited = limited + (violation.unsqueeze(-1) * normal).sum(2)
            limited = clip_speed(limited, horizontal, vertical)
    limited = _clear_columns(limited, points, centers, radii, valid, horizontal, vertical)
    if peers:
        limited = _clear_peers(limited, points, horizontal, vertical)
    limited = _stay_inside(limited, points, env)
    return clip_speed(limited, horizontal, vertical)


def _stay_inside(velocity, points, env):
    """柱体修正不能把飞机推出边界。"""
    half_x = float(env.rules.cfg["boundary_half_extent_x"]) - BODY_RADIUS - SAFETY_BUFFER
    half_y = float(env.rules.cfg["boundary_half_extent_y"]) - BODY_RADIUS - SAFETY_BUFFER
    limited = velocity.clone()
    end = points + limited * DECISION_DT
    for index, low, high in ((0, -half_x, half_x), (1, -half_y, half_y), (2, MIN_ALTITUDE + SAFETY_BUFFER, MAX_ALTITUDE - SAFETY_BUFFER)):
        over = end[..., index] - high
        under = low - end[..., index]
        limited[..., index] = torch.where(over > 0, (high - points[..., index]) / DECISION_DT, limited[..., index])
        limited[..., index] = torch.where(under > 0, (low - points[..., index]) / DECISION_DT, limited[..., index])
    return limited


def _clear_columns(velocity, points, centers, radii, valid, horizontal, vertical):
    if centers.shape[2] == 0:
        return velocity
    limited = velocity
    for _ in range(4):
        end = points + limited * DECISION_DT
        gap = segment_clearance(
            points[:, :, None, :2] - centers,
            end[:, :, None, :2] - centers,
            radii + BODY_RADIUS,
        )
        if valid is not None:
            gap = gap.masked_fill(~valid, float("inf"))
        worst = gap.argmin(dim=-1)
        gap_n = gap.gather(-1, worst.unsqueeze(-1)).squeeze(-1)
        if bool((gap_n >= SAFETY_BUFFER).all()):
            break
        chosen = centers.gather(2, worst.unsqueeze(-1).unsqueeze(-1).expand(*worst.shape, 1, 2)).squeeze(2)
        outward = _unit(torch.cat((points[..., :2] - chosen, torch.zeros_like(points[..., 2:3])), dim=-1))
        inward = (limited * outward).sum(-1, keepdim=True)
        limited = limited - inward.clamp(max=0.0) * outward
        limited = limited + ((SAFETY_BUFFER - gap_n).clamp(min=0.0) / DECISION_DT).unsqueeze(-1) * outward
        limited = clip_speed(limited, horizontal, vertical)
    return limited


def _clear_peers(velocity, points, horizontal, vertical):
    limited = velocity
    count = points.shape[1]
    eye = torch.eye(count, device=points.device, dtype=torch.bool)
    for _ in range(3):
        delta = points[:, :, None, :] - points[:, None, :, :]
        moved = delta[..., :2] + (limited[:, :, None, :2] - limited[:, None, :, :2]) * DECISION_DT
        gap = segment_clearance(delta[..., :2], moved, BODY_CONTACT)
        gap = gap.masked_fill(eye, float("inf"))
        worst = gap.argmin(dim=-1)
        gap_n = gap.gather(-1, worst.unsqueeze(-1)).squeeze(-1)
        if bool((gap_n >= SAFETY_BUFFER).all()):
            break
        chosen = delta.gather(2, worst.view(*worst.shape, 1, 1).expand(*worst.shape, 1, 3)).squeeze(2)
        outward = _unit(chosen)
        inward = (limited * outward).sum(-1, keepdim=True)
        limited = limited - inward.clamp(max=0.0) * outward
        limited = limited + ((SAFETY_BUFFER - gap_n).clamp(min=0.0) / DECISION_DT).unsqueeze(-1) * outward
        limited = clip_speed(limited, horizontal, vertical)
    return limited


def shield_pursuer_command(env, command: torch.Tensor) -> torch.Tensor:
    """学习器输出也过同一道安全投影，再交给机体。"""
    state = env.state()
    safe = shield_velocity(
        command[..., :3],
        state[:, 1:, :3],
        env,
        peers=True,
        horizontal=PURSUER_HORIZONTAL,
        vertical=PURSUER_VERTICAL,
    )
    shielded = command.clone()
    shielded[..., :3] = safe
    return shielded


def pack_command(velocity: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    velocity = clip_pursuer_velocity(velocity)
    command = torch.zeros(*velocity.shape[:-1], 4, device=velocity.device, dtype=velocity.dtype)
    command[..., :3] = velocity
    command[..., 3] = _yaw_rate(velocity[..., :2], quat)
    return command


def evader_command(env) -> torch.Tensor:
    """水平方向用 goal_avoid；垂直方向躲开仍在飞行的追捕机，并留在飞行高度内。"""
    state = env.state()
    horizontal = evader_world_velocity(env)
    evader = state[:, 0]
    others = state[:, 1:, :3]
    alive = ~env.disabled[:, 1:]
    delta = evader[:, None, :3] - others
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    influence = (1.0 - distance / 8.0).clamp(0.0, 1.0)
    influence = torch.where(alive, influence, torch.zeros_like(influence))
    vertical = (delta[..., 2] / distance * influence).sum(-1)
    level = alive & (distance < 8.0) & (delta[..., 2].abs() < 0.4)
    limit = getattr(env, "evader_vertical", None)
    if limit is None:
        limit = vertical.new_tensor(EVADER_VERTICAL)
    else:
        limit = limit.to(device=vertical.device, dtype=vertical.dtype)
    vertical = vertical + level.sum(-1) * limit
    command = torch.zeros(env.n, 4, device=state.device, dtype=state.dtype)
    command[:, :2] = horizontal
    command[:, 2] = vertical.clamp(-limit, limit)
    command[:, :3] = keep_in_band(command[:, :3], evader[:, 2], limit)
    horizontal_limit = getattr(env, "evader_horizontal", None)
    if horizontal_limit is None:
        horizontal_limit = float(env.rules.settings["simulation"]["evader_speed_limit"])
    command[:, :3] = shield_velocity(
        command[:, None, :3],
        evader[:, None, :3],
        env,
        peers=False,
        horizontal=horizontal_limit,
        vertical=limit,
    )[:, 0]
    command[:, 3] = _yaw_rate(command[:, :2], evader[:, 3:7])
    return command


def obstacle_bank(env) -> tuple[torch.Tensor, torch.Tensor]:
    static = env.rules.static
    centers = [static[..., :2]]
    radii = [static[..., 2]]
    if env.rules.bg.shape[1]:
        centers.append(env.rules.bg)
        radii.append(env.rules.radii[:, env.rules.layout.dynamic_objects])
    return torch.cat(centers, 1), torch.cat(radii, 1)


def intercept_time(origin: torch.Tensor, target: torch.Tensor, velocity: torch.Tensor, speed: float) -> torch.Tensor:
    """到达匀速目标所需时间。追不上时返回一个很大的数。"""
    relative = target - origin
    vv = (velocity * velocity).sum(-1)
    rr = (relative * relative).sum(-1)
    rv = (relative * velocity).sum(-1)
    quadratic = vv - speed * speed
    linear = 2.0 * rv
    disc = linear.square() - 4.0 * quadratic * rr
    far = torch.full_like(rr, 1.0e6)
    parallel = quadratic.abs() < 1e-4
    t_parallel = torch.where(rv < -1e-4, -rr / linear.clamp(max=-1e-4), rr.sqrt() / speed)
    denom = (2.0 * quadratic).where(~parallel, torch.ones_like(quadratic))
    root = disc.clamp(min=0.0).sqrt()
    earliest = torch.minimum((-linear - root) / denom, (-linear + root) / denom)
    later = torch.maximum((-linear - root) / denom, (-linear + root) / denom)
    positive = torch.where(earliest > 0.05, earliest, later)
    solved = torch.where((disc >= 0) & (positive > 0.05), positive, far)
    return torch.where(parallel, torch.where(t_parallel > 0.05, t_parallel, far), solved)


def aim_points(env) -> torch.Tensor:
    """两架去拦截预测点，最远的一架提前堵在目标航路上。"""
    state = env.state()
    pursuers = state[:, 1:, :3]
    evader, evader_velocity, valid = sensed_target(env)
    travel = intercept_time(pursuers, evader, evader_velocity, PURSUER_HORIZONTAL)
    lead_cap = 2.0 if getattr(env, "mid360", None) is not None else 8.0
    lead = evader + evader_velocity * travel.clamp(max=lead_cap).unsqueeze(-1)
    height = float(env.rules.settings["simulation"]["target_height"])
    goal = torch.cat((env.rules.goal, env.rules.goal.new_full((env.n, 1), height)), dim=-1)
    block = goal[:, None, :] - 1.6 * _unit(goal[:, None, :] - evader)
    distance = torch.linalg.vector_norm(pursuers - evader, dim=-1)
    blocker = torch.zeros_like(distance, dtype=torch.bool)
    blocker.scatter_(1, distance.argmax(dim=-1, keepdim=True), True)
    reachable = (travel < 12.0) & valid & ~blocker
    aim = torch.where(reachable.unsqueeze(-1), lead, block)
    agents = pursuers.shape[1]
    index = torch.arange(agents, device=pursuers.device, dtype=pursuers.dtype)
    bearing = 2.0 * math.pi * index / max(agents, 1)
    offset = 0.4 * torch.stack((bearing.cos(), bearing.sin(), torch.zeros_like(bearing)), dim=-1)
    offset = offset * (~blocker).unsqueeze(-1).to(offset.dtype)
    # 全队都没有航迹时在已知目标点周围散开搜索，飞进 16 米量程后再锁定。
    ring = 6.0 * torch.stack((bearing.cos(), bearing.sin(), torch.zeros_like(bearing)), dim=-1)
    search = goal[:, None, :] + ring
    return torch.where(valid.unsqueeze(-1), aim + offset, search)


def dodge_obstacles(velocity: torch.Tensor, points: torch.Tensor, env) -> torch.Tensor:
    """机身还没贴到柱面时就把水平速度转开。只使用本机航迹。"""
    centers, radii, _motion, valid = _brake_cylinders(env, points.shape[1])
    if centers.shape[2] == 0:
        return velocity
    horizontal = points[:, :, None, :2] - centers
    radial = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-6)
    gap = radial - radii - BODY_RADIUS
    if valid is not None:
        gap = gap.masked_fill(~valid, float("inf"))
    nearest = gap.argmin(dim=-1)
    gap_n = gap.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
    outward_xy = horizontal.gather(2, nearest.unsqueeze(-1).unsqueeze(-1).expand(*nearest.shape, 1, 2)).squeeze(2)
    outward_xy = outward_xy / radial.gather(-1, nearest.unsqueeze(-1)).clamp_min(1e-6)
    outward = torch.cat((outward_xy, torch.zeros_like(points[..., 2:3])), dim=-1)
    outward = _unit(outward)
    closing = (velocity * outward).sum(-1) < 0.0
    weight = ((DODGE_RANGE - gap_n) / DODGE_RANGE).clamp(0.0, 1.0)
    weight = torch.where(closing | (gap_n < 0.7), weight, 0.25 * weight)
    escape = torch.cat((outward[..., :2] * PURSUER_HORIZONTAL, velocity[..., 2:3]), dim=-1)
    return (1.0 - weight.unsqueeze(-1)) * velocity + weight.unsqueeze(-1) * escape


def separate_peers(velocity: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """在机身接触之前把两架追捕机推开。"""
    count = points.shape[1]
    if count < 2:
        return velocity
    delta = points[:, :, None, :] - points[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    eye = torch.eye(count, device=points.device, dtype=torch.bool)
    distance = distance.masked_fill(eye, float("inf"))
    nearest = distance.argmin(dim=-1)
    gap = distance.gather(-1, nearest.unsqueeze(-1)).squeeze(-1) - BODY_CONTACT
    direction = _unit(delta.gather(2, nearest.view(nearest.shape[0], count, 1, 1).expand(*points.shape[:2], 1, 3)).squeeze(2))
    weight = ((PEER_PUSH - gap) / PEER_PUSH).clamp(0.0, 1.0)
    escape = torch.cat((direction[..., :2] * PURSUER_HORIZONTAL, velocity[..., 2:3]), dim=-1)
    return (1.0 - weight.unsqueeze(-1)) * velocity + weight.unsqueeze(-1) * escape


def cylinder_delta(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """点到贯穿高度的竖直圆柱的外向向量和水平距离。points [N,P,3]，圆柱 [N,S,*]。"""
    xy = points[..., :2]
    horizontal = xy[:, :, None, :] - centers[:, None, :, :]
    radial = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-6)
    inward = torch.clamp(radii[:, None, :] / radial, max=1.0)
    closest_xy = centers[:, None, :, :] + horizontal * inward.unsqueeze(-1)
    closest_z = points[:, :, None, 2].expand_as(radial)
    closest = torch.cat((closest_xy, closest_z.unsqueeze(-1)), dim=-1)
    delta = points[:, :, None, :] - closest
    distance = torch.linalg.vector_norm(delta, dim=-1)
    outward_xy = horizontal / radial.unsqueeze(-1)
    outward = torch.cat((outward_xy, torch.zeros_like(closest_z).unsqueeze(-1)), dim=-1)
    direction = torch.where(distance.unsqueeze(-1) < 1e-3, outward, delta / distance.clamp_min(1e-6).unsqueeze(-1))
    return direction, distance


def _local_cylinder_repulsion(points, centers, radii, valid, gain: float) -> torch.Tensor:
    """centers/radii/valid 为 [N,P,S,*]，只排斥本机航迹里的圆柱。"""
    if centers.shape[2] == 0:
        return torch.zeros_like(points)
    horizontal = points[:, :, None, :2] - centers
    radial = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-6)
    distance = radial - radii
    outward = horizontal / radial.unsqueeze(-1)
    direction = torch.cat((outward, torch.zeros_like(points[:, :, None, 2:3]).expand(*outward.shape[:3], 1)), dim=-1)
    active = distance < INFLUENCE
    if valid is not None:
        active = active & valid
    magnitude = (1.0 / distance.clamp_min(0.25) - 1.0 / INFLUENCE).masked_fill(~active, 0.0)
    return (gain * direction * magnitude.unsqueeze(-1)).sum(2)


def cylinder_repulsion(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, gain: float) -> torch.Tensor:
    direction, distance = cylinder_delta(points, centers, radii)
    active = distance < INFLUENCE
    magnitude = 1.0 / distance.clamp_min(0.25) - 1.0 / INFLUENCE
    return (gain * direction * magnitude.unsqueeze(-1) * active.unsqueeze(-1)).sum(-2)


def boundary_repulsion(points: torch.Tensor, env, gain: float = 1.2) -> torch.Tensor:
    lower = points.new_tensor([-env.rules.cfg["boundary_half_extent_x"], -env.rules.cfg["boundary_half_extent_y"], MIN_ALTITUDE])
    upper = points.new_tensor([env.rules.cfg["boundary_half_extent_x"], env.rules.cfg["boundary_half_extent_y"], MAX_ALTITUDE])
    field = torch.zeros_like(points)
    low = points - lower
    high = upper - points
    field = field + torch.where(low < INFLUENCE, gain / low.clamp_min(0.25), torch.zeros_like(low))
    field = field - torch.where(high < INFLUENCE, gain / high.clamp_min(0.25), torch.zeros_like(high))
    return field


def peer_repulsion(points: torch.Tensor, gain: float = 1.6) -> torch.Tensor:
    delta = points[:, :, None, :] - points[:, None, :, :]
    distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    eye = torch.eye(points.shape[1], device=points.device, dtype=torch.bool)
    active = (distance < INFLUENCE) & ~eye
    direction = delta / distance.unsqueeze(-1)
    direction = direction.masked_fill(eye.unsqueeze(-1), 0.0)
    magnitude = 1.0 / distance.clamp_min(0.25) - 1.0 / INFLUENCE
    magnitude = magnitude.masked_fill(~active, 0.0)
    return (gain * direction * magnitude.unsqueeze(-1)).sum(2)


def apf_velocity(env) -> torch.Tensor:
    state = env.state()
    pursuers = state[:, 1:, :3]
    field = _unit(aim_points(env) - pursuers)
    field = field + peer_repulsion(pursuers, gain=1.2)
    centers, radii, _motion, valid = _intent_cylinders(env, pursuers)
    if centers.shape[2]:
        field = field + _local_cylinder_repulsion(pursuers, centers, radii + BODY_RADIUS, valid, gain=1.8)
    field = field + boundary_repulsion(pursuers, env)
    return _unit(field) * PURSUER_HORIZONTAL


def _preferred_velocity(env) -> torch.Tensor:
    state = env.state()
    return _unit(aim_points(env) - state[:, 1:, :3]) * PURSUER_HORIZONTAL


def orca_guidance(env) -> torch.Tensor:
    state = env.state()
    points = state[:, 1:, :3]
    centers, radii, motion, valid = _intent_cylinders(env, points)
    alive = ~env.disabled[:, 1:]
    present = valid if valid is not None else torch.ones(centers.shape[:3], dtype=torch.bool, device=centers.device)
    solved = orca_velocity(
        points,
        state[:, 1:, 7:10],
        _preferred_velocity(env),
        alive,
        centers,
        radii,
        motion,
        present,
    )
    return keep_in_band(solved, points[..., 2], PURSUER_VERTICAL)


def cbf_guidance(env) -> torch.Tensor:
    state = env.state()
    points = state[:, 1:, :3]
    centers, radii, motion, valid = _intent_cylinders(env, points)
    solved = cbf_qp(_preferred_velocity(env), points, centers, radii, motion, valid)
    return keep_in_band(solved, points[..., 2], PURSUER_VERTICAL)


def mpc_guidance(env, previous: torch.Tensor) -> torch.Tensor:
    state = env.state()
    points = state[:, 1:, :3]
    preferred = _preferred_velocity(env)
    centers, radii, motion, valid = _intent_cylinders(env, points)
    solved = receding_mpc(points, aim_points(env), preferred, previous, centers, radii, motion, valid)
    previous.copy_(solved)
    return keep_in_band(solved, points[..., 2], PURSUER_VERTICAL)


def shield_heuristic_velocity(env) -> torch.Tensor:
    """拦截点直飞。绕柱和间隔交给后面的安全投影，不再靠势场碰运气。"""
    state = env.state()
    return _unit(aim_points(env) - state[:, 1:, :3]) * PURSUER_HORIZONTAL


def frpn_velocity(env) -> torch.Tensor:
    state = env.state()
    pursuers = state[:, 1:, :3]
    own_velocity = state[:, 1:, 7:10]
    relative = aim_points(env) - pursuers
    distance = torch.linalg.vector_norm(relative, dim=-1, keepdim=True).clamp_min(1e-6)
    los = relative / distance
    relative_velocity = -own_velocity
    closing = torch.clamp(-(relative_velocity * los).sum(-1, keepdim=True), min=0.0)
    los_rate = torch.cross(relative, relative_velocity, dim=-1) / distance.square()
    lateral = 4.0 * torch.clamp(closing, min=0.25) * torch.cross(los_rate, los, dim=-1)
    response = 1.1 * (PURSUER_HORIZONTAL * los - own_velocity)
    acceleration = lateral + response
    limit = torch.linalg.vector_norm(acceleration, dim=-1, keepdim=True).clamp_min(1e-6)
    acceleration = acceleration * torch.clamp(2.5 / limit, max=1.0)
    return own_velocity + acceleration * DECISION_DT


def mpc_velocity(env, previous: torch.Tensor) -> torch.Tensor:
    state = env.state()
    pursuers = state[:, 1:, :3]
    own_count = pursuers.shape[1]
    _, evader_velocity, _ = sensed_target(env)
    goal = aim_points(env)
    direct = _unit(goal - pursuers)
    tangent = _unit(torch.stack((-direct[..., 1], direct[..., 0], torch.zeros_like(direct[..., 2])), dim=-1))
    speed = pursuers.new_tensor([0.45, 0.75, 1.0])
    lateral = pursuers.new_tensor([-1.0, -0.6, -0.25, 0.0, 0.25, 0.6, 1.0])
    direction = _unit(direct[:, :, None, None, :] + lateral[None, None, None, :, None] * tangent[:, :, None, None, :])
    velocity = direction * (PURSUER_HORIZONTAL * speed)[None, None, :, None, None]
    velocity = velocity.reshape(env.n, own_count, int(speed.numel() * lateral.numel()), 3)
    zeros = torch.zeros(env.n, own_count, 1, 3, device=state.device, dtype=state.dtype)
    hold = evader_velocity[:, :, None, :]
    samples = [velocity, zeros, hold]
    centers, radii, _motion, valid = _brake_cylinders(env, own_count)
    if centers.shape[2]:
        horizontal = pursuers[:, :, None, :2] - centers
        radial = torch.linalg.vector_norm(horizontal, dim=-1).clamp_min(1e-6)
        gap = radial - radii - BODY_RADIUS
        if valid is not None:
            gap = gap.masked_fill(~valid, float("inf"))
        nearest = gap.argmin(dim=-1)
        outward_xy = horizontal.gather(2, nearest.unsqueeze(-1).unsqueeze(-1).expand(*nearest.shape, 1, 2)).squeeze(2)
        outward_xy = outward_xy / radial.gather(-1, nearest.unsqueeze(-1)).clamp_min(1e-6)
        escape = _unit(torch.cat((outward_xy, torch.zeros_like(pursuers[..., 2:3])), dim=-1))
        escape = torch.cat((escape[..., :2] * PURSUER_HORIZONTAL, torch.zeros_like(escape[..., 2:3])), dim=-1)
        samples.append(escape.unsqueeze(2))
    velocity = clip_pursuer_velocity(torch.cat(samples, dim=2))
    sample_count = velocity.shape[2]
    endpoint = pursuers[:, :, None, :] + velocity * (8 * DECISION_DT)
    cost = 3.0 * (endpoint - goal[:, :, None, :]).square().sum(-1)
    cost = cost + 0.25 * (velocity - previous[:, :, None, :]).square().sum(-1)
    peer = torch.linalg.vector_norm(pursuers[:, None, :, None, :] - endpoint[:, :, None, :, :], dim=-1)
    eye = torch.eye(own_count, device=state.device, dtype=torch.bool)
    peer = peer.masked_fill(eye[:, :, None], float("inf"))
    nearest_peer = peer.amin(dim=2).clamp_min(0.45)
    cost = cost + (0.35 * 1.5) / nearest_peer.square()
    if env.rules.buildings.shape[1]:
        _, clearance = building_delta(endpoint.reshape(env.n, own_count * sample_count, 3), env.rules.buildings)
        clearance = clearance.reshape(env.n, own_count, sample_count, -1).amin(-1)
        cost = cost + 4.0 / clearance.clamp_min(0.2).square()
    if centers.shape[2]:
        swept = []
        for horizon in (DECISION_DT, 0.8, 1.6):
            future = pursuers[:, :, None, :] + velocity * horizon
            gap = segment_clearance(
                pursuers[:, :, None, None, :2] - centers[:, :, None, :, :],
                future[:, :, :, None, :2] - centers[:, :, None, :, :],
                radii[:, :, None, :] + BODY_RADIUS,
            )
            if valid is not None:
                gap = gap.masked_fill(~valid[:, :, None, :], float("inf"))
            swept.append(gap.amin(-1))
        clearance = torch.stack(swept, dim=0)
        immediate = clearance[0]
        later = clearance[1:].amin(0)
        cost = torch.where(immediate >= SAFETY_BUFFER, cost, cost + 1.0e4 - immediate)
        cost = cost + torch.where(later >= SAFETY_BUFFER, torch.zeros_like(cost), 40.0 / later.clamp_min(0.05))
    best = cost.argmin(dim=-1)
    chosen = velocity.gather(2, best[..., None, None].expand(env.n, own_count, 1, 3)).squeeze(2)
    previous.copy_(chosen)
    return chosen


class PursuerGuidance:
    def __init__(self, method: str, envs: int, device: torch.device):
        if method not in ("apf", "frpn", "orca", "cbf", "mpc"):
            raise ValueError(method)
        self.method = method
        self.previous = torch.zeros(envs, 3, 3, device=device)

    def reset(self) -> None:
        self.previous.zero_()

    def propose(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """过滤前的意图速度，以及过完安全投影后真正下发的指令。"""
        state = env.state()
        points = state[:, 1:, :3]
        if self.method in ("apf", "frpn"):
            raw = apf_velocity(env) if self.method == "apf" else frpn_velocity(env)
            velocity = keep_in_band(raw, points[..., 2], PURSUER_VERTICAL)
            preferred = pack_command(velocity, state[:, 1:, 3:7])
            executed = pack_command(
                shield_velocity(velocity, points, env, peers=True, horizontal=PURSUER_HORIZONTAL, vertical=PURSUER_VERTICAL),
                state[:, 1:, 3:7],
            )
            return preferred, executed
        if self.method == "orca":
            velocity = orca_guidance(env)
        elif self.method == "cbf":
            velocity = cbf_guidance(env)
        else:
            velocity = mpc_guidance(env, self.previous)
        preferred = pack_command(velocity, state[:, 1:, 3:7])
        executed = pack_command(
            shield_velocity(velocity, points, env, peers=True, horizontal=PURSUER_HORIZONTAL, vertical=PURSUER_VERTICAL),
            state[:, 1:, 3:7],
        )
        return preferred, executed

    def command(self, env) -> torch.Tensor:
        return self.propose(env)[1]


def body_contact(state: torch.Tensor, radius: float = 0.6) -> torch.Tensor:
    """任意两架机体中心距不大于两倍半径。"""
    distance = torch.cdist(state[:, :, :3], state[:, :, :3])
    eye = torch.eye(state.shape[1], device=state.device, dtype=torch.bool)
    distance = distance.masked_fill(eye, float("inf"))
    return distance.amin(dim=(-2, -1)) <= radius
