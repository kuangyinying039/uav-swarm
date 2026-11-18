"""Radar5 追捕契约：世界系三维速度加偏航角速度，以及团队势函数奖励。

动作四路一起给出，互不由对方换算。水平速度按模长再截一次，
垂直速度和偏航角速度各自截断。奖励按一次决策结算，不按物理子步累加。
"""

from __future__ import annotations

import math

import torch

PURSUER_HORIZONTAL = 1.48
PURSUER_VERTICAL = 0.8
EVADER_VERTICAL = 0.88
MAX_YAW_RATE = 1.4
DECISION_DT = 0.2
PHYSICS_DT = 0.02
DECISION_STEPS = 300
CAPTURE_RADIUS = 1.0
CAPTURE_REQUIRED = 2
BODY_RADIUS = 0.3
BODY_CONTACT = 0.6
# 追捕机彼此停在这个距离外；对逃逸机则停在捕获球内、机身外。
PURSUER_CLEARANCE = 1.2
EVADER_CLEARANCE = 0.8
BRAKE_HORIZON = 0.4
PEER_SLIDE = 0.5
PEER_SLIDE_RANGE = 2.0
COLLISION_RADIUS = 0.8
APPROACH_SCALE = 8.0
APPROACH_WEIGHT = 4.0
COOPERATIVE_WEIGHT = 3.0
SEPARATION_RANGE = 2.0
OBSTACLE_INFLUENCE = 1.5
SEPARATION_WEIGHT = 0.5
OBSTACLE_WEIGHT = 0.5
GAMMA = 0.995
TEAM_WEIGHT = 0.1
CAPTURE_BONUS = 6.0
TIME_PENALTY = 0.0
COLLISION_PENALTY = 8.0
DISTANCE_COEF = 0.1
DISTANCE_SCALE = 10.0


def physics_substeps(dt: float = PHYSICS_DT) -> int:
    """一次 0.2 s 决策对应的物理步数。50 Hz 时正好 10 步。"""
    return max(1, int(round(DECISION_DT / dt)))


def body_yaw(quat: torch.Tensor) -> torch.Tensor:
    """四元数 wxyz 的偏航角，弧度，范围 [-pi, pi]。"""
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _broadcast_limit(limit, reference: torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(limit):
        return limit.to(device=reference.device, dtype=reference.dtype).reshape(reference.shape[0], *([1] * (reference.ndim - 1)))
    return reference.new_tensor(float(limit))


def _limit_horizontal(velocity_xy: torch.Tensor, limit) -> torch.Tensor:
    speed = torch.linalg.vector_norm(velocity_xy, dim=-1, keepdim=True)
    scale = torch.clamp(_broadcast_limit(limit, velocity_xy) / speed.clamp_min(1e-12), max=1.0)
    return velocity_xy * scale


def clip_velocity_yaw(action: torch.Tensor, evader_horizontal, evader_vertical=None) -> tuple[torch.Tensor, torch.Tensor]:
    """把 [N, aircraft, 4] 的 (vx, vy, vz, yaw_rate) 截到契约限幅。

    索引 0 是逃逸机，其余是追捕机。逃逸机限幅可以是标量，也可以是逐环境向量。
    返回世界系速度 [N, aircraft, 3] 和偏航角速度。
    """
    if action.shape[-1] != 4:
        raise ValueError(f"Expected (vx, vy, vz, yaw_rate), got {tuple(action.shape)}")
    vel = torch.empty(*action.shape[:-1], 3, device=action.device, dtype=action.dtype)
    vel[:, :1, :2] = _limit_horizontal(action[:, :1, :2], evader_horizontal)
    vel[:, 1:, :2] = _limit_horizontal(action[:, 1:, :2], PURSUER_HORIZONTAL)
    vertical = EVADER_VERTICAL if evader_vertical is None else _broadcast_limit(evader_vertical, action[:, :1, 2])
    vel[:, :1, 2] = action[:, :1, 2].clamp(-vertical, vertical)
    vel[:, 1:, 2] = action[:, 1:, 2].clamp(-PURSUER_VERTICAL, PURSUER_VERTICAL)
    yaw_rate = action[..., 3].clamp(-MAX_YAW_RATE, MAX_YAW_RATE)
    return vel, yaw_rate


def pursuer_distance(state: torch.Tensor) -> torch.Tensor:
    """各追捕机到逃逸机的三维距离 [N, P]。"""
    return torch.linalg.vector_norm(state[:, 1:, :3] - state[:, :1, :3], dim=-1)


def living_pair_contact(state: torch.Tensor, active: torch.Tensor, radius: float = BODY_CONTACT) -> torch.Tensor:
    """仍在飞行的飞机对。返回上三角 [N, aircraft, aircraft]，失效机体不参与。"""
    distance = torch.cdist(state[..., :3], state[..., :3])
    count = state.shape[-2]
    upper = torch.triu(torch.ones(count, count, device=state.device, dtype=torch.bool), diagonal=1)
    hit = (distance <= radius) & upper
    hit = hit & active.unsqueeze(-1) & active.unsqueeze(-2)
    return hit


def colliding_aircraft(state: torch.Tensor, active: torch.Tensor | None = None, radius: float = BODY_CONTACT) -> torch.Tensor:
    """中心距不大于两倍机体半径、且双方都还在飞行的飞机。返回 [N, aircraft]。"""
    if active is None:
        active = torch.ones(state.shape[:-1], dtype=torch.bool, device=state.device)
    pairs = living_pair_contact(state, active, radius)
    return pairs.any(-1) | pairs.any(-2)


def separate_velocities(
    velocity: torch.Tensor,
    position: torch.Tensor,
    actual_velocity: torch.Tensor,
    active: torch.Tensor,
    evader_horizontal: float,
) -> torch.Tensor:
    """按实际接近速度提前刹车，把活着的飞机留在机身距离之外。"""
    commanded = velocity
    adjusted = velocity.clone()
    count = position.shape[1]
    for i in range(count):
        for j in range(i + 1, count):
            both = active[:, i] & active[:, j]
            delta = position[:, i] - position[:, j]
            distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
            direction = delta / distance.unsqueeze(-1)
            keep = EVADER_CLEARANCE if i == 0 or j == 0 else PURSUER_CLEARANCE
            limit = ((distance - keep) / BRAKE_HORIZON).clamp_min(0.0)
            for own, other, outward in ((i, j, direction), (j, i, -direction)):
                command_closing = -(commanded[:, own] * outward).sum(-1)
                actual_closing = -(actual_velocity[:, own] * outward).sum(-1)
                desired = torch.minimum(command_closing, limit) - (actual_closing - limit).clamp_min(0.0)
                correction = outward * (command_closing - desired).unsqueeze(-1)
                correction = torch.where(both.unsqueeze(-1), correction, torch.zeros_like(correction))
                adjusted[:, own] = adjusted[:, own] + correction
            if i != 0 and j != 0:
                tangent = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
                slide = torch.where(distance < PEER_SLIDE_RANGE, torch.full_like(distance, PEER_SLIDE), torch.zeros_like(distance))
                slide = torch.where(both, slide, torch.zeros_like(slide)).unsqueeze(-1)
                adjusted[:, i, :2] = adjusted[:, i, :2] + tangent * slide
                adjusted[:, j, :2] = adjusted[:, j, :2] - tangent * slide
    adjusted[:, :1, :2] = _limit_horizontal(adjusted[:, :1, :2], float(evader_horizontal))
    adjusted[:, 1:, :2] = _limit_horizontal(adjusted[:, 1:, :2], PURSUER_HORIZONTAL)
    adjusted[:, :1, 2] = adjusted[:, :1, 2].clamp(-EVADER_VERTICAL, EVADER_VERTICAL)
    adjusted[:, 1:, 2] = adjusted[:, 1:, 2].clamp(-PURSUER_VERTICAL, PURSUER_VERTICAL)
    return adjusted


def _mean_active(score: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    count = active.sum(-1).clamp_min(1).to(score.dtype)
    present = active.any(-1)
    return torch.where(present, score.masked_fill(~active, 0).sum(-1) / count, torch.zeros_like(count))


def _nearest_gap(xy: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
    gap = torch.linalg.vector_norm(xy[:, :, None, :] - centers[:, None, :, :], dim=-1)
    return (gap - radii[:, None, :] - BODY_RADIUS).amin(-1)


def team_potential(
    state: torch.Tensor,
    active: torch.Tensor | None = None,
    static: torch.Tensor | None = None,
    dynamic: torch.Tensor | None = None,
    dynamic_radii: torch.Tensor | None = None,
) -> torch.Tensor:
    """全体仍在飞行的追捕机共享一个势。

    接近项按各机距离平均，权重高于间隔和障碍，接触后饱和。
    协同围捕项在多架从不同方向靠近时升高，单架或挤在同一侧时为零。
    间隔项在机身接触到 SEPARATION_RANGE 之间升高。
    障碍项在柱面外 OBSTACLE_INFLUENCE 内升高。
    """
    distance = pursuer_distance(state)
    if active is None:
        active = torch.ones_like(distance, dtype=torch.bool)
    approach = torch.exp(-torch.clamp(distance - BODY_CONTACT, min=0.0) / APPROACH_SCALE)
    approach_mean = _mean_active(approach, active)
    offset = state[:, 1:, :3] - state[:, :1, :3]
    direction = offset / distance.clamp_min(1e-6).unsqueeze(-1)
    weight = approach.masked_fill(~active, 0)
    mean_direction = (weight.unsqueeze(-1) * direction).sum(1) / weight.sum(-1, keepdim=True).clamp_min(1e-6)
    enclosure = (1.0 - torch.linalg.vector_norm(mean_direction, dim=-1)).clamp(0, 1)
    enclosure = torch.where(active.sum(-1) >= 2, enclosure, torch.zeros_like(enclosure))
    potential = APPROACH_WEIGHT * approach_mean + COOPERATIVE_WEIGHT * enclosure * approach_mean
    pos = state[:, 1:, :3]
    count = pos.shape[1]
    if count >= 2:
        pair_dist = torch.linalg.vector_norm(pos[:, :, None, :] - pos[:, None, :, :], dim=-1)
        upper = torch.triu(torch.ones(count, count, device=state.device, dtype=torch.bool), diagonal=1)
        pair = active[:, :, None] & active[:, None, :] & upper
        separation = ((pair_dist - BODY_CONTACT) / (SEPARATION_RANGE - BODY_CONTACT)).clamp(0, 1)
        pair_count = pair.sum((-2, -1)).clamp_min(1).to(separation.dtype)
        pair_mean = separation.masked_fill(~pair, 0).sum((-2, -1)) / pair_count
        potential = potential + SEPARATION_WEIGHT * torch.where(pair.any((-2, -1)), pair_mean, torch.zeros_like(pair_mean))
    gaps = []
    xy = state[:, 1:, :2]
    if static is not None and static.shape[1] > 0:
        gaps.append(_nearest_gap(xy, static[..., :2], static[..., 2]))
    if dynamic is not None and dynamic_radii is not None and dynamic.shape[1] > 0:
        gaps.append(_nearest_gap(xy, dynamic, dynamic_radii))
    if gaps:
        clearance = gaps[0] if len(gaps) == 1 else torch.minimum(gaps[0], gaps[1])
        obstacle = (clearance / OBSTACLE_INFLUENCE).clamp(0, 1)
        potential = potential + OBSTACLE_WEIGHT * _mean_active(obstacle, active)
    return potential


def dual_capture(state: torch.Tensor, active: torch.Tensor | None = None) -> torch.Tensor:
    """至少两架仍在飞行的追捕机进入捕获球。已经相碰失效的不计。"""
    inside = pursuer_distance(state) <= CAPTURE_RADIUS
    if active is not None:
        inside = inside & active
    return inside.sum(-1) >= CAPTURE_REQUIRED


def min_peer_distance(state: torch.Tensor) -> torch.Tensor:
    """追捕机之间的最小三维距离。不足两架时为 +inf。"""
    pos = state[:, 1:, :3]
    if pos.shape[1] < 2:
        return torch.full((state.shape[0],), math.inf, device=state.device, dtype=state.dtype)
    delta = pos[:, :, None, :] - pos[:, None, :, :]
    dist = torch.linalg.vector_norm(delta, dim=-1)
    dist = dist.masked_fill(torch.eye(pos.shape[1], device=state.device, dtype=torch.bool), math.inf)
    return dist.amin(dim=(-2, -1))


def decision_reward(
    before: torch.Tensor,
    after: torch.Tensor,
    *,
    captured: torch.Tensor,
    truncated: torch.Tensor,
    terminated: torch.Tensor,
    collision: torch.Tensor,
    active_before: torch.Tensor | None = None,
    active_after: torch.Tensor | None = None,
    static: torch.Tensor | None = None,
    dynamic_before: torch.Tensor | None = None,
    dynamic_after: torch.Tensor | None = None,
    dynamic_radii: torch.Tensor | None = None,
) -> torch.Tensor:
    """团队势的差分塑形，加上捕获奖金、碰撞惩罚和每步时间惩罚。

    塑形项是 TEAM_WEIGHT * (gamma * Φ' - Φ)。终局的势记为 0。
    超时不再另扣。碰撞仍扣 COLLISION_PENALTY。
    """
    old = team_potential(before, active_before, static, dynamic_before, dynamic_radii)
    new = team_potential(after, active_after, static, dynamic_after, dynamic_radii)
    terminal = captured | truncated | terminated
    new = torch.where(terminal, torch.zeros_like(new), new)
    reward = TEAM_WEIGHT * (GAMMA * new - old) - TIME_PENALTY
    distance = pursuer_distance(before)
    if active_before is None:
        mean_distance = distance.mean(-1)
    else:
        mean_distance = distance.masked_fill(~active_before, 0).sum(-1) / active_before.sum(-1).clamp_min(1)
    reward = reward - DISTANCE_COEF * (mean_distance / DISTANCE_SCALE)
    reward = reward + CAPTURE_BONUS * captured.float()
    reward = reward - COLLISION_PENALTY * collision.float()
    return reward
