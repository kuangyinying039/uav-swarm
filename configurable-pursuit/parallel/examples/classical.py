"""ORCA、控制屏障二次规划和滚动时域 MPC。

障碍地图用仿真里的圆心、半径和运动速度。目标点由调用方传入，可以是雷达航迹。
三个方法都不再套一层固定轮数的半空间投影。
"""

from __future__ import annotations

import torch

from parallel.env.radar5_contract import BODY_CONTACT, BODY_RADIUS, PURSUER_HORIZONTAL, PURSUER_VERTICAL

DECISION_DT = 0.2
ORCA_HORIZON = 2.0
CBF_BUFFER = 0.25
CBF_BRAKE = 2.5
CBF_DELAY = 0.55
MPC_HORIZON = 8
MPC_ITERS = 28


def project_halfspaces(desired: torch.Tensor, normals: torch.Tensor, lower: torch.Tensor, iterations: int = 48, step: float = 0.45) -> torch.Tensor:
    """投影到 n·v >= lower。对偶梯度求最小修正，变量维数就是速度维数。"""
    if normals.numel() == 0:
        return desired
    lam = torch.zeros_like(lower)
    velocity = desired
    for _ in range(iterations):
        slack = lower - (normals * velocity.unsqueeze(-2)).sum(-1)
        lam = (lam + step * slack).clamp(min=0.0)
        velocity = desired + (lam.unsqueeze(-1) * normals).sum(-2)
    return velocity


def clip_velocity(velocity: torch.Tensor, horizontal: float = PURSUER_HORIZONTAL, vertical: float = PURSUER_VERTICAL) -> torch.Tensor:
    limited = velocity.clone()
    speed = torch.linalg.vector_norm(limited[..., :2], dim=-1, keepdim=True).clamp_min(1e-6)
    limited[..., :2] = limited[..., :2] * torch.clamp(horizontal / speed, max=1.0)
    limited[..., 2] = limited[..., 2].clamp(-vertical, vertical)
    return limited


def _unit2(vector: torch.Tensor) -> torch.Tensor:
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True).clamp_min(1e-6)


def orca_lines(
    relative_position: torch.Tensor,
    relative_velocity: torch.Tensor,
    radius: torch.Tensor,
    inv_horizon: float,
    inv_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RVO2 的 ORCA 线。返回线方向和相对速度需要加上的修正 u。"""
    dist_sq = relative_position.square().sum(-1, keepdim=True).clamp_min(1e-8)
    while radius.ndim < dist_sq.ndim:
        radius = radius.unsqueeze(-1)
    radius = radius.expand(dist_sq.shape)
    radius_sq = radius.square()
    colliding = dist_sq <= radius_sq
    inv = torch.where(colliding, relative_position.new_tensor(inv_step), relative_position.new_tensor(inv_horizon))
    w = relative_velocity - inv * relative_position
    w_len_sq = w.square().sum(-1, keepdim=True).clamp_min(1e-8)
    w_len = w_len_sq.sqrt()
    unit_w = w / w_len
    dot = (w * relative_position).sum(-1, keepdim=True)
    cutoff = (~colliding) & (dot < 0) & (dot.square() > radius_sq * w_len_sq)
    u_cut = (radius * inv - w_len) * unit_w
    direction_cut = torch.stack((unit_w[..., 1], -unit_w[..., 0]), dim=-1)
    leg = (dist_sq - radius_sq).clamp_min(0.0).sqrt()
    left = torch.stack((
        relative_position[..., 0] * leg.squeeze(-1) - relative_position[..., 1] * radius.squeeze(-1),
        relative_position[..., 0] * radius.squeeze(-1) + relative_position[..., 1] * leg.squeeze(-1),
    ), dim=-1) / dist_sq
    right = torch.stack((
        relative_position[..., 0] * leg.squeeze(-1) + relative_position[..., 1] * radius.squeeze(-1),
        -relative_position[..., 0] * radius.squeeze(-1) + relative_position[..., 1] * leg.squeeze(-1),
    ), dim=-1) / dist_sq
    det = relative_position[..., 0] * w[..., 1] - relative_position[..., 1] * w[..., 0]
    direction_leg = torch.where((det > 0).unsqueeze(-1), left, -right)
    dot_leg = (relative_velocity * direction_leg).sum(-1, keepdim=True)
    u_leg = dot_leg * direction_leg - relative_velocity
    use_cut = colliding | cutoff
    return torch.where(use_cut, direction_cut, direction_leg), torch.where(use_cut, u_cut, u_leg)


def orca_velocity(
    position: torch.Tensor,
    velocity: torch.Tensor,
    preferred: torch.Tensor,
    peer_mask: torch.Tensor,
    obstacle_center: torch.Tensor,
    obstacle_radius: torch.Tensor,
    obstacle_velocity: torch.Tensor,
    obstacle_mask: torch.Tensor,
) -> torch.Tensor:
    """水平面 ORCA。队友各承担一半，圆柱和运动圆盘由追捕机全部避开。

    position/velocity/preferred 为 [N,A,3]。peer_mask [N,A] 为 True 表示该机还在飞。
    障碍 [N,S,2]、[N,S]、[N,S,2]，obstacle_mask [N,S]。
    """
    n, agents, _ = position.shape
    horizontal = position[..., :2]
    current = velocity[..., :2]
    preferred_xy = preferred[..., :2]
    normals = []
    lower = []
    if agents > 1:
        rel = horizontal[:, None, :, :] - horizontal[:, :, None, :]
        rel_vel = current[:, :, None, :] - current[:, None, :, :]
        radius = position.new_full(rel.shape[:-1], BODY_CONTACT)
        direction, correction = orca_lines(rel, rel_vel, radius, 1.0 / ORCA_HORIZON, 1.0 / DECISION_DT)
        normal = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
        point = current[:, :, None, :] + 0.5 * correction
        bound = (normal * point).sum(-1)
        eye = torch.eye(agents, device=position.device, dtype=torch.bool)
        valid = peer_mask[:, None, :] & peer_mask[:, :, None] & ~eye
        normals.append(normal.reshape(n, agents, -1, 2))
        lower.append(bound.reshape(n, agents, -1).masked_fill(~valid.reshape(n, agents, -1), -1.0e3))
    if obstacle_center.shape[-2]:
        if obstacle_center.ndim == 3:
            rel = obstacle_center[:, None, :, :] - horizontal[:, :, None, :]
            rel_vel = current[:, :, None, :] - obstacle_velocity[:, None, :, :]
            radius = obstacle_radius[:, None, :] + BODY_RADIUS
            present = obstacle_mask[:, None, :] & peer_mask[:, :, None]
        else:
            rel = obstacle_center - horizontal[:, :, None, :]
            rel_vel = current[:, :, None, :] - obstacle_velocity
            radius = obstacle_radius + BODY_RADIUS
            present = obstacle_mask & peer_mask[:, :, None]
        direction, correction = orca_lines(rel, rel_vel, radius, 1.0 / ORCA_HORIZON, 1.0 / DECISION_DT)
        normal = torch.stack((-direction[..., 1], direction[..., 0]), dim=-1)
        point = current[:, :, None, :] + correction
        bound = (normal * point).sum(-1)
        normals.append(normal)
        lower.append(bound.masked_fill(~present, -1.0e3))
    if not normals:
        solved = preferred_xy
    else:
        solved = project_halfspaces(preferred_xy, torch.cat(normals, 2), torch.cat(lower, 2), iterations=36, step=0.4)
        speed = torch.linalg.vector_norm(solved, dim=-1, keepdim=True).clamp_min(1e-6)
        solved = solved * torch.clamp(PURSUER_HORIZONTAL / speed, max=1.0)
    return torch.cat((solved, preferred[..., 2:3]), dim=-1)


def cbf_halfspaces(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, motion: torch.Tensor, valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """速度半空间：允许的靠近速度由刹车距离决定。centers 为 [N,S,2] 或每机 [N,A,S,2]。"""
    obstacle_count = centers.shape[-2]
    if obstacle_count == 0:
        empty_n = points.new_zeros(points.shape[0], points.shape[1], 0, 3)
        empty_l = points.new_zeros(points.shape[0], points.shape[1], 0)
        return empty_n, empty_l
    if centers.ndim == 3:
        delta = points[:, :, None, :2] - centers[:, None, :, :]
        radii = radii[:, None, :]
        motion = motion[:, None, :, :]
    else:
        delta = points[:, :, None, :2] - centers
    radial = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
    outward = delta / radial.unsqueeze(-1)
    normal = torch.cat((outward, torch.zeros_like(points[:, :, None, 2:3]).expand(*outward.shape[:3], 1)), dim=-1)
    clearance = radial - radii - BODY_RADIUS
    if valid is not None:
        clearance = torch.where(valid, clearance, torch.full_like(clearance, 1.0e3))
    closing_offset = (outward * motion).sum(-1)
    gap = (clearance - CBF_BUFFER).clamp(min=0.0)
    allowed = torch.sqrt((CBF_BRAKE * CBF_DELAY) ** 2 + 2.0 * CBF_BRAKE * gap) - CBF_BRAKE * CBF_DELAY
    push = ((CBF_BUFFER - clearance).clamp(min=0.0) / CBF_DELAY).clamp(max=0.3)
    lower = closing_offset - allowed + push
    return normal, lower


def cbf_qp(preferred: torch.Tensor, points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor, motion: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """min ||v - v_pref||^2，约束为障碍半空间、队友间隔和高度边界。"""
    normals, lower = cbf_halfspaces(points, centers, radii, motion, valid)
    agents = points.shape[1]
    pieces_n = [normals]
    pieces_l = [lower]
    if agents > 1:
        delta = points[:, :, None, :] - points[:, None, :, :]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        normal = delta / distance.unsqueeze(-1)
        eye = torch.eye(agents, device=points.device, dtype=torch.bool)
        clearance = torch.where(eye, torch.full_like(distance, 10.0), distance - BODY_CONTACT)
        gap = (clearance - CBF_BUFFER).clamp(min=0.0)
        allowed = torch.sqrt((CBF_BRAKE * CBF_DELAY) ** 2 + 2.0 * CBF_BRAKE * gap) - CBF_BRAKE * CBF_DELAY
        push = ((CBF_BUFFER - clearance).clamp(min=0.0) / CBF_DELAY).clamp(max=0.3)
        pieces_n.append(normal)
        pieces_l.append(-allowed + push)
    floor = points[..., 2] - 1.0
    ceiling = 12.0 - points[..., 2]
    wall_n = points.new_zeros(points.shape[0], agents, 2, 3)
    wall_n[..., 0, 2] = 1.0
    wall_n[..., 1, 2] = -1.0
    wall_l = torch.stack((
        -(floor - CBF_BUFFER).clamp(min=0.0) / CBF_DELAY,
        -(ceiling - CBF_BUFFER).clamp(min=0.0) / CBF_DELAY,
    ), dim=-1)
    pieces_n.append(wall_n)
    pieces_l.append(wall_l)
    velocity = project_halfspaces(preferred, torch.cat(pieces_n, 2), torch.cat(pieces_l, 2))
    return clip_velocity(velocity)


def mpc_velocity(
    points: torch.Tensor,
    aim: torch.Tensor,
    preferred: torch.Tensor,
    previous: torch.Tensor,
    centers: torch.Tensor,
    radii: torch.Tensor,
    motion: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """三架联立的速度模型 MPC。时域 1.6 秒，每步 0.2 秒，二次代价用原始对偶迭代求解。"""
    n, agents, _ = points.shape
    horizon = MPC_HORIZON
    dt = DECISION_DT
    velocity = preferred.unsqueeze(2).expand(n, agents, horizon, 3).clone()
    time = dt * torch.arange(1, horizon + 1, device=points.device, dtype=points.dtype).view(1, 1, horizon, 1, 1)
    eye = torch.eye(agents, device=points.device, dtype=torch.bool).view(1, agents, agents, 1)
    aim_weight = 0.35
    smooth_weight = 0.15
    for _ in range(MPC_ITERS):
        velocity = clip_velocity(velocity)
        position = points.unsqueeze(2) + dt * torch.cumsum(velocity, dim=2)
        grad = 2.0 * (velocity - preferred.unsqueeze(2))
        delta_v = torch.cat((
            velocity[:, :, :1, :] - previous.unsqueeze(2),
            velocity[:, :, 1:, :] - velocity[:, :, :-1, :],
        ), dim=2)
        grad = grad + (2.0 * smooth_weight) * delta_v
        grad[:, :, :-1, :] = grad[:, :, :-1, :] - (2.0 * smooth_weight) * delta_v[:, :, 1:, :]
        error = position - aim.unsqueeze(2)
        suffix = torch.flip(torch.cumsum(torch.flip(error, dims=(2,)), dim=2), dims=(2,))
        grad = grad + (2.0 * aim_weight * dt) * suffix
        velocity = velocity - 0.08 * grad
        predicted = position[..., :2]
        if centers.shape[-2]:
            if centers.ndim == 3:
                relative = predicted.unsqueeze(3) - centers[:, None, None, :, :] - motion[:, None, None, :, :] * time
                radius = radii[:, None, None, :]
            else:
                relative = predicted.unsqueeze(3) - centers[:, :, None, :, :] - motion[:, :, None, :, :] * time
                radius = radii[:, :, None, :]
            distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1e-6)
            outward = relative / distance.unsqueeze(-1)
            gap = distance - radius - BODY_RADIUS
            violation = (CBF_BUFFER - gap).clamp(min=0.0)
            if valid is not None:
                violation = violation.masked_fill(~valid.unsqueeze(2), 0.0)
            push = ((violation / dt).unsqueeze(-1) * outward).sum(3)
            push = torch.flip(torch.cumsum(torch.flip(push, dims=(2,)), dim=2), dims=(2,))
            velocity = torch.cat((velocity[..., :2] + 0.5 * push, velocity[..., 2:3]), dim=-1)
        if agents > 1:
            relative = predicted[:, :, None, :, :] - predicted[:, None, :, :, :]
            distance = torch.linalg.vector_norm(relative, dim=-1).clamp_min(1e-6)
            normal = relative / distance.unsqueeze(-1)
            gap = distance - BODY_CONTACT
            violation = (CBF_BUFFER - gap).clamp(min=0.0).masked_fill(eye, 0.0)
            push = ((violation / dt).unsqueeze(-1) * normal).sum(2)
            push = torch.flip(torch.cumsum(torch.flip(push, dims=(2,)), dim=2), dims=(2,))
            velocity = torch.cat((velocity[..., :2] + 0.35 * push, velocity[..., 2:3]), dim=-1)
    return clip_velocity(velocity[:, :, 0, :])
