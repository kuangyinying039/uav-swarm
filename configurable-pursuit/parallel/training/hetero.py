"""局部异构图观测。

目标、队友和最近圆柱各自成一类节点。没有航迹时目标相对位置填 -5，
和“目标就在身边”分开。编码器对三类节点分别做注意力，再和自身特征相加。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from parallel.env.mid360 import sensed_target
from parallel.env.radar5_contract import (
    BODY_RADIUS,
    PURSUER_HORIZONTAL,
    PURSUER_VERTICAL,
    body_yaw,
)

BLIND = -5.0
K_OBSTACLES = 3
SELF_DIM = 10
TARGET_DIM = 22
PEER_DIM = 7
OBSTACLE_DIM = 6
HIDDEN = 128


def graph_observation(state: torch.Tensor, env, forecast: torch.Tensor | None = None) -> torch.Tensor:
    """返回 [N, P, SELF+TARGET+(P-1)*PEER+K*OBSTACLE]。"""
    pursuers = state[:, 1:]
    n, count, _ = pursuers.shape
    position = pursuers[..., :3]
    velocity = pursuers[..., 7:10]
    yaw = body_yaw(pursuers[..., 3:7]).unsqueeze(-1) / torch.pi
    alive = (~env.disabled[:, 1:]).unsqueeze(-1).to(position.dtype)
    if getattr(env, "mid360", None) is not None:
        target, target_velocity, valid = sensed_target(env)
    else:
        target = state[:, :1, :3].expand_as(position)
        target_velocity = state[:, :1, 7:10].expand_as(position)
        valid = torch.ones(n, count, dtype=torch.bool, device=state.device)
    scale = position.new_tensor([20.0, 20.0, 12.0])
    relative = (position - target) / scale
    relative = torch.where(valid.unsqueeze(-1), relative, torch.full_like(relative, BLIND))
    speed_scale = position.new_tensor([1.52, 1.52, 0.88])
    target_velocity = torch.where(valid.unsqueeze(-1), target_velocity / speed_scale, torch.zeros_like(target_velocity))
    if forecast is None:
        forecast = position.new_zeros(n, count, 15)
    forecast = torch.where(valid.unsqueeze(-1), forecast, torch.zeros_like(forecast))
    detected = valid.unsqueeze(-1).to(position.dtype)
    if count != 3:
        raise RuntimeError("共享动作网络按 3 架追捕机编码角色")
    own_scale = position.new_tensor([PURSUER_HORIZONTAL, PURSUER_HORIZONTAL, PURSUER_VERTICAL])
    identity = torch.eye(count, device=position.device, dtype=position.dtype).expand(n, count, count)
    self_node = torch.cat((velocity / own_scale, yaw, position[..., 2:3] / 12.0, alive, detected, identity), dim=-1)
    target_node = torch.cat((relative, target_velocity, forecast, detected), dim=-1)
    peer_node = _peers(position, velocity, alive.squeeze(-1), scale, own_scale)
    obstacle_node = _obstacles(position, env)
    return torch.cat((self_node, target_node, peer_node, obstacle_node), dim=-1)


def _peers(position, velocity, alive, scale, own_scale):
    count = position.shape[1]
    relative = position[:, None, :, :] - position[:, :, None, :]
    distance = torch.linalg.vector_norm(relative, dim=-1)
    eye = torch.eye(count, device=position.device, dtype=torch.bool)
    distance = distance.masked_fill(eye, 1.0e6)
    order = distance.argsort(dim=-1)[..., : max(count - 1, 1)]
    chosen = relative.gather(2, order.unsqueeze(-1).expand(*order.shape, 3)) / scale
    peer_velocity = velocity[:, None].expand_as(relative).gather(2, order.unsqueeze(-1).expand(*order.shape, 3))
    peer_alive = alive[:, None].expand(alive.shape[0], count, count).gather(2, order).unsqueeze(-1)
    return torch.cat((chosen, peer_velocity / own_scale, peer_alive), dim=-1).flatten(-2)


def _obstacles(position, env):
    from parallel.env.mid360 import sensed_cylinders

    n, count, _ = position.shape
    centers, radii, _motion, valid, dynamic = sensed_cylinders(env)
    slots = K_OBSTACLES
    blank = position.new_zeros(n, count, slots, OBSTACLE_DIM)
    if centers.shape[1] != count or centers.shape[2] == 0:
        return blank.flatten(-2)
    delta = position[:, :, None, :2] - centers
    gap = torch.linalg.vector_norm(delta, dim=-1) - radii - BODY_RADIUS
    gap = gap.masked_fill(~valid, 1.0e6)
    take = min(slots, int(centers.shape[2]))
    gaps, index = torch.topk(gap, take, dim=-1, largest=False)
    chosen = delta.gather(2, index.unsqueeze(-1).expand(n, count, take, 2))
    radius = radii.gather(2, index)
    kind = dynamic.gather(2, index).to(position.dtype)
    seen = gaps < 1.0e5
    node = torch.cat((
        chosen / 20.0,
        (gaps / 5.0).unsqueeze(-1),
        (radius / 2.0).unsqueeze(-1),
        kind.unsqueeze(-1),
        seen.unsqueeze(-1).to(position.dtype),
    ), dim=-1)
    node = node.masked_fill(~seen.unsqueeze(-1), 0.0)
    blank[:, :, :take] = node
    return blank.flatten(-2)


def observation_slices(n_pursuers: int = 3) -> dict[str, tuple[int, int, int]]:
    """名字 -> (起点, 终点, 每个节点的宽度)。"""
    cursor = 0
    parts = {}
    for name, width, count in (
        ("self", SELF_DIM, 1),
        ("target", TARGET_DIM, 1),
        ("peer", PEER_DIM, max(n_pursuers - 1, 1)),
        ("obstacle", OBSTACLE_DIM, K_OBSTACLES),
    ):
        parts[name] = (cursor, cursor + width * count, width)
        cursor += width * count
    return parts


class EntityAttention(nn.Module):
    def __init__(self, entity_dim: int, hidden: int = HIDDEN, heads: int = 2):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.proj = nn.Linear(entity_dim, hidden)
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, query: torch.Tensor, entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        entity = F.relu(self.proj(entities))
        batch = query.shape[:-1]
        query_h = self.q(query).reshape(*batch, self.heads, self.head_dim)
        key = self.k(entity).reshape(*entity.shape[:-1], self.heads, self.head_dim)
        value = self.v(entity).reshape(*entity.shape[:-1], self.heads, self.head_dim)
        score = (query_h.unsqueeze(-3) * key).sum(-1) / self.head_dim**0.5
        score = score.masked_fill(~mask.unsqueeze(-1), -1.0e4)
        weight = torch.softmax(score, dim=-2) * mask.unsqueeze(-1).to(score.dtype)
        weight = weight / weight.sum(dim=-2, keepdim=True).clamp_min(1.0e-6)
        context = (weight.unsqueeze(-1) * value).sum(-3)
        context = context.reshape(*batch, self.heads * self.head_dim)
        present = mask.any(dim=-1, keepdim=True).to(context.dtype)
        return self.out(F.relu(context)) * present


class HeteroEncoder(nn.Module):
    def __init__(self, obs_dim: int, n_pursuers: int = 3, hidden: int = HIDDEN):
        super().__init__()
        slices = observation_slices(n_pursuers)
        expected = slices["obstacle"][1]
        if int(obs_dim) != expected:
            raise ValueError(f"graph observation width is {expected}, received {obs_dim}")
        self.slices = slices
        self.self_proj = nn.Linear(SELF_DIM, hidden)
        self.target = EntityAttention(TARGET_DIM, hidden)
        self.peer = EntityAttention(PEER_DIM, hidden)
        self.obstacle = EntityAttention(OBSTACLE_DIM, hidden)
        self.fuse = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, hidden))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        parts = {}
        masks = {}
        for name, (start, stop, width) in self.slices.items():
            block = obs[..., start:stop].reshape(*obs.shape[:-1], -1, width)
            parts[name] = block
            if name == "obstacle":
                masks[name] = block[..., -1] > 0.5
            elif name == "peer":
                masks[name] = block[..., -1] > 0.5
            else:
                masks[name] = torch.ones(*block.shape[:-1], dtype=torch.bool, device=obs.device)
        self_h = F.relu(self.self_proj(parts["self"].squeeze(-2)))
        target_h = self.target(self_h, parts["target"], masks["target"])
        peer_h = self.peer(self_h, parts["peer"], masks["peer"])
        obstacle_h = self.obstacle(self_h, parts["obstacle"], masks["obstacle"])
        fused = self.fuse(torch.cat((self_h, target_h, peer_h, obstacle_h), dim=-1))
        return self_h + fused
