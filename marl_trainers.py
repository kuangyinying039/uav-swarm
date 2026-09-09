"""PyTorch MAPPO and QMIX adapters for the weak-communication environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import time
from typing import Callable

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch to run MAPPO/QMIX training.")


def seed_everything(seed: int) -> None:
    """Seed training-side RNGs before creating networks or sampling actions."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainConfig:
    episodes: int = 500
    max_steps: int = 0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.20
    lr: float = 3e-4
    actor_lr: float | None = None
    critic_lr: float | None = None
    lr_end_factor: float = 0.10
    hidden_dim: int = 128
    update_epochs: int = 5
    batch_size: int = 32
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_coef_end: float = 0.002
    adaptive_entropy: bool = True
    uncertainty_entropy_gain: float = 1.0
    continuous_log_std_max: float = -0.5
    continuous_initial_std: float = 0.4
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    normalize_returns: bool = False
    reward_scale: float = 1.0
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    target_update: int = 10
    use_gat: bool = True
    use_hetero_entities: bool = True
    use_search_weights: bool = False
    gat_heads: int = 4
    gat_layers: int = 2
    dropout: float = 0.0
    device: str = "auto"
    log_interval: int = 10
    checkpoint_interval: int = 100
    checkpoint_path: str = ""
    target_kl: float = 0.03
    use_assignment_head: bool = False
    assignment_entropy_coef: float = 0.005


class MLP(nn.Module if nn else object):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128):
        require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class FlatMLP(MLP):
    """Plain per-UAV MLP accepting (and deliberately ignoring) graph inputs."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, positive_output: bool = False):
        super().__init__(in_dim, out_dim, hidden_dim)
        self.positive_output = positive_output

    def forward(self, x, adjacency=None, hetero_graph=None):
        output = super().forward(x)
        return F.softplus(output) + 0.2 if self.positive_output else output


class GraphAttentionBlock(nn.Module if nn else object):
    """Three-channel heterogeneous semantic GAT block.

    The graph nodes are still UAV decision nodes, but messages are separated
    into peer, target, and obstacle semantic relations before semantic fusion.
    """

    def __init__(self, hidden_dim: int = 128, heads: int = 2, dropout: float = 0.05):
        require_torch()
        super().__init__()
        self.relations = ("peer", "target", "obstacle")
        self.q_layers = nn.ModuleDict(
            {rel: nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)]) for rel in self.relations}
        )
        self.k_layers = nn.ModuleDict(
            {rel: nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)]) for rel in self.relations}
        )
        self.v_layers = nn.ModuleDict(
            {rel: nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)]) for rel in self.relations}
        )
        self.edge_gates = nn.ModuleDict(
            {
                rel: nn.Sequential(
                    nn.Linear(hidden_dim * 2 + 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for rel in self.relations
            }
        )
        self.relation_out = nn.ModuleDict({rel: nn.Linear(hidden_dim * heads, hidden_dim) for rel in self.relations})
        self.semantic_score = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1, bias=False))
        self.out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        self.hidden_dim = hidden_dim

    def _relation_signal(self, obs, relation: str):
        if relation == "peer":
            return obs[..., 6:7] if obs.shape[-1] > 6 else torch.zeros(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)
        if relation == "target":
            return obs[..., 13:14] if obs.shape[-1] > 13 else torch.zeros(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)
        if obs.shape[-1] > 9:
            return 1.0 - torch.clamp(obs[..., 9:10], 0.0, 1.0)
        return torch.zeros(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)

    def forward(self, h, obs, semantic_contexts, adjacency=None):
        pos = obs[..., :2]
        dist = torch.cdist(pos, pos)
        eye = torch.eye(dist.shape[-1], device=dist.device, dtype=torch.bool)
        if adjacency is not None:
            allowed = adjacency.to(device=dist.device, dtype=torch.bool)
            if allowed.dim() == 2:
                allowed = allowed.unsqueeze(0)
            allowed = allowed | eye
        relation_outputs = []
        hi = h.unsqueeze(-2).expand(*h.shape[:-2], h.shape[-2], h.shape[-2], h.shape[-1])
        for relation in self.relations:
            rel_context = semantic_contexts[relation]
            rel_h = h + rel_context
            hj = rel_h.unsqueeze(-3).expand_as(hi)
            signal = self._relation_signal(obs, relation)
            signal_i = signal.unsqueeze(-2).expand(*dist.shape, 1)
            signal_j = signal.unsqueeze(-3).expand(*dist.shape, 1)
            edge_features = torch.cat([hi, hj, dist.unsqueeze(-1), signal_i, signal_j], dim=-1)
            gate = torch.sigmoid(self.edge_gates[relation](edge_features)).squeeze(-1)
            gate = gate.masked_fill(eye, 1.0)
            distance_scale = 1.5 if relation == "peer" else 0.7
            adjacency_bias = torch.log(torch.clamp(gate, 1e-4, 1.0)) - distance_scale * dist
            head_outputs = []
            for q_layer, k_layer, v_layer in zip(self.q_layers[relation], self.k_layers[relation], self.v_layers[relation]):
                q = q_layer(h)
                k = k_layer(rel_h)
                v = v_layer(rel_h)
                logits = torch.matmul(q, k.transpose(-1, -2)) / (self.hidden_dim ** 0.5)
                if adjacency is not None:
                    logits = logits.masked_fill(~allowed, -1e9)
                weights = torch.softmax(logits + adjacency_bias, dim=-1)
                head_outputs.append(torch.matmul(weights, v))
            relation_outputs.append(F.relu(self.relation_out[relation](torch.cat(head_outputs, dim=-1))))
        relation_stack = torch.stack(relation_outputs, dim=-2)
        semantic_weights = torch.softmax(self.semantic_score(relation_stack).squeeze(-1), dim=-1).unsqueeze(-1)
        fused = torch.sum(semantic_weights * relation_stack, dim=-2)
        encoded = self.dropout(F.relu(self.out(torch.cat([h, fused], dim=-1))))
        return self.norm(h + encoded)


class EntityCrossAttention(nn.Module if nn else object):
    """Relation-specific attention from UAV nodes to local entity nodes."""

    def __init__(self, node_dim: int, hidden_dim: int, heads: int, dropout: float):
        require_torch()
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.q = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.k = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.v = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(heads)])
        self.out = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, uav_h, entity_nodes, entity_mask):
        entity_h = F.relu(self.node_proj(entity_nodes))
        outputs = []
        valid = entity_mask.to(dtype=uav_h.dtype)
        for q_layer, k_layer, v_layer in zip(self.q, self.k, self.v):
            q = q_layer(uav_h).unsqueeze(-2)
            k = k_layer(entity_h)
            v = v_layer(entity_h)
            scores = torch.sum(q * k, dim=-1) / (self.hidden_dim ** 0.5)
            scores = scores.masked_fill(~entity_mask, -1e9)
            weights = torch.softmax(scores, dim=-1) * valid
            weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
            outputs.append(torch.sum(weights.unsqueeze(-1) * v, dim=-2))
        context = self.dropout(F.relu(self.out(torch.cat(outputs, dim=-1))))
        has_entity = entity_mask.any(dim=-1, keepdim=True).to(dtype=context.dtype)
        return context * has_entity


class GraphAttentionEncoder(nn.Module if nn else object):
    """Heterogeneous UAV-target-belief-obstacle graph encoder."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 128,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.05,
        use_peer_messages: bool = True,
    ):
        require_torch()
        super().__init__()
        self.obs_dim = obs_dim
        self.use_peer_messages = use_peer_messages
        # Environment observations consist of 32 semantic features followed by
        # the action mask.  This keeps the encoder compatible with the original
        # 9-action planar environment and the experimental 11-action 3-D mode.
        self.action_mask_dim = obs_dim - 32
        if self.action_mask_dim < 1:
            raise ValueError(f"Invalid observation dimension {obs_dim}: no action-mask features.")
        self.semantic_dim = obs_dim - self.action_mask_dim
        if self.semantic_dim < 14:
            raise ValueError(f"Graph observation requires at least 14 semantic features, got {self.semantic_dim}.")
        self.self_proj = nn.Linear(4, hidden_dim)
        self.agent_proj = nn.Linear(3, hidden_dim)
        self.obstacle_proj = nn.Linear(3, hidden_dim)
        self.target_proj = nn.Linear(4, hidden_dim)
        self.task_proj = nn.Linear(self.semantic_dim - 14, hidden_dim)
        self.mask_proj = nn.Linear(self.action_mask_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.type_fuse = nn.Linear(hidden_dim * 6, hidden_dim)
        self.blocks = nn.ModuleList([GraphAttentionBlock(hidden_dim, heads, dropout) for _ in range(layers)])
        self.target_attention = nn.ModuleList([EntityCrossAttention(13, hidden_dim, heads, dropout) for _ in range(layers)])
        self.frontier_attention = nn.ModuleList([EntityCrossAttention(8, hidden_dim, heads, dropout) for _ in range(layers)])
        self.obstacle_attention = nn.ModuleList([EntityCrossAttention(8, hidden_dim, heads, dropout) for _ in range(layers)])

    def _typed_projection(self, obs):
        if obs.shape[-1] < self.semantic_dim:
            h = F.relu(self.obs_proj(obs))
            return h, {"peer": h, "target": h, "obstacle": h}
        self_feat = obs[..., 0:4]
        agent_feat = obs[..., 4:7]
        obstacle_feat = obs[..., 7:10]
        target_feat = obs[..., 10:14]
        task_feat = obs[..., 14:self.semantic_dim]
        mask_feat = obs[..., self.semantic_dim:]
        if mask_feat.shape[-1] == 0:
            mask_feat = torch.zeros(*obs.shape[:-1], 1, device=obs.device, dtype=obs.dtype)
        parts = [
            F.relu(self.self_proj(self_feat)),
            F.relu(self.agent_proj(agent_feat)),
            F.relu(self.obstacle_proj(obstacle_feat)),
            F.relu(self.target_proj(target_feat)),
            F.relu(self.task_proj(task_feat)),
            F.relu(self.mask_proj(mask_feat)),
        ]
        h = F.relu(self.type_fuse(torch.cat(parts, dim=-1)))
        contexts = {
            "peer": parts[1],
            "target": parts[3] + parts[4],
            "obstacle": parts[2] + parts[5],
        }
        return h, contexts

    def forward(self, obs, adjacency=None, hetero_graph=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)
            if adjacency is not None and adjacency.dim() == 2:
                adjacency = adjacency.unsqueeze(0)
            if hetero_graph is not None:
                hetero_graph = {key: value.unsqueeze(0) for key, value in hetero_graph.items()}
            squeeze = True
        else:
            squeeze = False
        if not self.use_peer_messages:
            # A controlled graph-message-passing ablation: retain the shared
            # encoder, parameter count, typed projections, and heterogeneous
            # entity attention, while allowing only each UAV's self edge.
            n_agents = obs.shape[-2]
            adjacency = torch.eye(n_agents, dtype=torch.bool, device=obs.device)
            adjacency = adjacency.unsqueeze(0).expand(obs.shape[0], -1, -1)
        h, semantic_contexts = self._typed_projection(obs)
        for layer_idx, block in enumerate(self.blocks):
            if hetero_graph is not None:
                track_nodes = hetero_graph.get("track_nodes", hetero_graph["target_nodes"])
                track_mask = hetero_graph.get("track_mask", hetero_graph["target_mask"])
                target_h = self.target_attention[layer_idx](h, track_nodes, track_mask)
                frontier_h = torch.zeros_like(target_h)
                if "frontier_nodes" in hetero_graph:
                    frontier_h = self.frontier_attention[layer_idx](
                        h, hetero_graph["frontier_nodes"], hetero_graph["frontier_mask"]
                    )
                obstacle_h = self.obstacle_attention[layer_idx](h, hetero_graph["obstacle_nodes"], hetero_graph["obstacle_mask"])
                semantic_contexts["target"] = F.relu(semantic_contexts["target"] + target_h + frontier_h)
                semantic_contexts["obstacle"] = F.relu(semantic_contexts["obstacle"] + obstacle_h)
            h = block(h, obs, semantic_contexts, adjacency)
            semantic_contexts = {key: F.relu(value + h) for key, value in semantic_contexts.items()}
        encoded = h
        return encoded.squeeze(0) if squeeze else encoded


class GraphActor(nn.Module if nn else object):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.05,
        use_peer_messages: bool = True,
        assignment_slots: int = 0,
    ):
        require_torch()
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            obs_dim, hidden_dim, heads, layers, dropout, use_peer_messages
        )
        self.head = nn.Linear(hidden_dim, action_dim)
        self.assignment_slots = int(assignment_slots)
        self.target_assignment_slots = max(self.assignment_slots - 1, 0)
        if self.assignment_slots > 0:
            self.assignment_query = nn.Linear(hidden_dim, hidden_dim)
            self.assignment_key = nn.Linear(13, hidden_dim)
            self.search_assignment = nn.Parameter(torch.zeros(hidden_dim))
            self.assignment_motion_fuse = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, obs, adjacency=None, hetero_graph=None):
        return self.head(self.encoder(obs, adjacency, hetero_graph))

    def encode_with_assignment(self, obs, adjacency=None, hetero_graph=None):
        encoded = self.encoder(obs, adjacency, hetero_graph)
        if self.assignment_slots <= 0 or hetero_graph is None:
            raise RuntimeError("Assignment head requires split heterogeneous track nodes")
        tracks = hetero_graph.get("track_nodes", hetero_graph["target_nodes"])
        track_mask = hetero_graph.get("track_mask", hetero_graph["target_mask"])
        if tracks.shape[-2] < self.target_assignment_slots:
            raise ValueError(
                "Heterogeneous graph has fewer track slots than the assignment head"
            )
        tracks = tracks[..., : self.target_assignment_slots, :]
        track_mask = track_mask[..., : self.target_assignment_slots]
        query = self.assignment_query(encoded)
        keys = self.assignment_key(tracks)
        target_logits = torch.einsum("...h,...th->...t", query, keys) / np.sqrt(keys.shape[-1])
        target_logits = target_logits.masked_fill(~track_mask, -1e9)
        search_logit = torch.einsum("...h,h->...", query, self.search_assignment).unsqueeze(-1)
        return encoded, keys, torch.cat([target_logits, search_logit], dim=-1)

    def motion_for_assignment(self, encoded, track_keys, assignments):
        search_shape = (*track_keys.shape[:-2], 1, track_keys.shape[-1])
        search_key = self.search_assignment.reshape(
            *((1,) * (len(search_shape) - 2)), 1, -1
        ).expand(search_shape)
        task_keys = torch.cat([track_keys, search_key], dim=-2)
        gather_index = assignments.long().unsqueeze(-1).unsqueeze(-1).expand(
            *assignments.shape, 1, task_keys.shape[-1]
        )
        selected = torch.gather(task_keys, -2, gather_index).squeeze(-2)
        conditioned = F.relu(self.assignment_motion_fuse(torch.cat([encoded, selected], dim=-1)))
        return self.head(conditioned)

    def forward_with_assignment(self, obs, adjacency=None, hetero_graph=None, assignments=None):
        encoded, keys, logits = self.encode_with_assignment(obs, adjacency, hetero_graph)
        if assignments is None:
            assignments = torch.argmax(logits, dim=-1)
        return self.motion_for_assignment(encoded, keys, assignments), logits


class GraphWeightActor(nn.Module if nn else object):
    """GAT actor producing Dirichlet concentrations for adaptive J1..J5 weights."""

    def __init__(
        self,
        obs_dim: int,
        objective_dim: int = 5,
        hidden_dim: int = 128,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.05,
        use_peer_messages: bool = True,
    ):
        require_torch()
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            obs_dim, hidden_dim, heads, layers, dropout, use_peer_messages
        )
        self.head = nn.Linear(hidden_dim, objective_dim)

    def forward(self, obs, adjacency=None, hetero_graph=None):
        raw = self.head(self.encoder(obs, adjacency, hetero_graph))
        return F.softplus(raw) + 0.2


class GraphQNetwork(nn.Module if nn else object):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        heads: int = 2,
        layers: int = 2,
        dropout: float = 0.05,
        use_peer_messages: bool = True,
    ):
        require_torch()
        super().__init__()
        self.encoder = GraphAttentionEncoder(
            obs_dim, hidden_dim, heads, layers, dropout, use_peer_messages
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs, adjacency=None, hetero_graph=None):
        return self.head(self.encoder(obs, adjacency, hetero_graph))


def tensorize_heterogeneous_graph(graph: dict, device=None) -> dict:
    return {
        key: torch.as_tensor(
            value,
            dtype=torch.bool if key.endswith("_mask") else torch.float32,
            device=device,
        )
        for key, value in graph.items()
        if key.endswith("_nodes") or key.endswith("_mask") or key == "uav_xyz"
    }


def episode_task_metrics(env) -> dict[str, float]:
    """Return logging metrics with discovery-only and tracking semantics aligned."""
    discovery_rate = float(np.mean(env.found_targets))
    pursuit_mode = hasattr(env, "captured")
    discovery_only = env.cfg.target_completion_steps <= 0
    tracking_rate = (
        0.0
        if discovery_only
        else float(np.mean(env.tracked_targets | env.completed_targets))
    )
    completion_rate = (
        float(env.captured)
        if pursuit_mode
        else discovery_rate
        if discovery_only
        else float(np.mean(env.completed_targets))
    )
    return {
        "target_discovery_rate": discovery_rate,
        "target_tracking_rate": tracking_rate,
        "target_completion_rate": completion_rate,
        "belief_decisiveness": float(env.belief_decisiveness()),
        "belief_brier_score": float(env.belief_brier_score()),
        "task_success": float(
            bool(env.captured)
            if pursuit_mode
            else discovery_rate >= 1.0
            if discovery_only
            else completion_rate >= 1.0
        ),
    }


def _tolist(value):
    if hasattr(value, "tolist"):
        return value.copy().tolist()
    if isinstance(value, (list, tuple)):
        return [
            item.tolist() if hasattr(item, "tolist") else list(item) if isinstance(item, tuple) else item
            for item in value
        ]
    return value


def _action_list(value):
    array = np.asarray(value)
    return array.astype(float).tolist() if array.ndim > 1 else array.tolist()


def capture_env_frame(env, result: dict, actions: list[int]) -> dict:
    frame = {
        "step": int(getattr(env, "t", 0)),
        "grid_size": int(getattr(getattr(env, "cfg", None), "grid_size", 25)),
        "actions": _action_list(actions),
        "raw_actions": _action_list(result.get("raw_actions", actions)),
        "executed_actions": _action_list(result.get("executed_actions", actions)),
        "action_agreement_rate": float(result.get("action_agreement_rate", 1.0)),
        "reward": float(result.get("reward", 0.0)),
        "coverage": float(result.get("coverage", env.coverage() if hasattr(env, "coverage") else 0.0)),
        "collisions": int(result.get("collisions", 0)),
        "lost_tracks": int(result.get("lost_tracks", 0)),
        "graph_density": float(result.get("graph_density", 0.0)),
        "outage_rate": float(result.get("outage_rate", 0.0)),
        "disabled_rate": float(result.get("disabled_rate", 0.0)),
        "active_uav_count": int(result.get("active_uav_count", 0)),
        "new_cells": int(result.get("new_cells", 0)),
        "revisit_gain": float(result.get("revisit_gain", 0.0)),
        "stagnant_agents": int(result.get("stagnant_agents", 0)),
        "positions": _tolist(env.positions) if hasattr(env, "positions") else [],
        "headings": _tolist(env.headings) if hasattr(env, "headings") else [],
        "altitudes": _tolist(env.altitudes) if hasattr(env, "altitudes") else [],
        "target_altitudes": _tolist(env.target_altitudes) if hasattr(env, "target_altitudes") else [],
        "buildings": _tolist(env.buildings) if hasattr(env, "buildings") else [],
        "building_heights": _tolist(env.building_heights) if hasattr(env, "building_heights") else [],
        "comm_adjacency": _tolist(getattr(env, "last_graph", [])),
        "direct_visibility": _tolist(env.direct_visibility_mask()) if hasattr(env, "direct_visibility_mask") else [],
        "capture_success": float(result.get("capture_success", 0.0)),
        "capture_hold_count": int(result.get("capture_hold_count", 0)),
        "capture_close_uavs": int(result.get("capture_close_uavs", 0)),
        "capture_angular_span_deg": float(result.get("capture_angular_span_deg", 0.0)),
        "minimum_capture_gap": float(result.get("minimum_capture_gap", float("nan"))),
        "continuous_safety_interventions": int(result.get("continuous_safety_interventions", 0)),
        "target_reacquired": float(result.get("target_reacquired", 0.0)),
        "target_lost": float(result.get("target_lost", 0.0)),
    }
    if hasattr(env, "minimum_capture_gap"):
        frame["initial_layout"] = env.initial_layout
        frame["initial_distances"] = env.initial_distances
        frame["execution_mode"] = "velocity_yaw_rate"
        frame["sensor_mode"] = "noisy_game_observation" if env.cfg.pursuit_target_observable else "lidar"
        frame["controller_feasible_rate"] = float(result.get("controller_feasible_rate", 0.0))
        for key in ('desired_velocities', 'executed_velocity_references', 'safety_reasons',
                    'safety_correction_rate', 'safety_correction_magnitude', 'emergency_stop_rate'):
            if key in result:
                frame[key] = result[key]
        frame["body_angular_rates"] = _tolist(env.quadrotor_states[:, 10:13])
        frame["lidar_model"] = env.cfg.lidar_model
        try:
            from pursuit_lidar import sensor_pose
        except ImportError:
            from .pursuit_lidar import sensor_pose
        frame["lidar_poses"] = [] if env.cfg.pursuit_target_observable else [{"origin": _tolist(sensor_pose(env, i)[0]),
                                 "rotation": _tolist(sensor_pose(env, i)[1])} for i in range(env.cfg.n_uavs)]
        frame["lidar_detection_ratio"] = float(result.get("lidar_detection_ratio", 0.0))
        frame["lidar_horizontal_fov_deg"] = env.cfg.lidar_horizontal_fov_deg
        frame["lidar_range"] = env.cfg.lidar_target_detection_range
        frame["lidar_vertical_fov_deg"] = [env.cfg.lidar_vertical_min_deg, env.cfg.lidar_vertical_max_deg]
        frame["capture_radius"] = float(env.cfg.capture_radius)
    if hasattr(env, "target_belief_mean"):
        frame["target_belief_mean"] = _tolist(env.target_belief_mean)
        frame["target_belief_covariance"] = _tolist(env.target_belief_covariance)
        frame["target_belief_timestamp"] = _tolist(env.target_belief_timestamp)
    if hasattr(env, "dynamic_targets"):
        frame["targets"] = _tolist(env.dynamic_targets)
    elif hasattr(env, "targets"):
        frame["targets"] = _tolist(env.targets)
    else:
        frame["targets"] = []
    frame["obstacles"] = _tolist(env.obstacles) if hasattr(env, "obstacles") else []
    frame["completed_targets"] = (
        _tolist(env.completed_targets.astype(int)) if hasattr(env, "completed_targets") else [0] * len(frame["targets"])
    )
    frame["found_targets"] = _tolist(env.found_targets.astype(int)) if hasattr(env, "found_targets") else []
    frame["tracked_targets"] = _tolist(env.tracked_targets.astype(int)) if hasattr(env, "tracked_targets") else []
    frame["disconnected_uavs"] = (
        _tolist(env.disconnected_uavs.astype(int)) if hasattr(env, "disconnected_uavs") else []
    )
    frame["disabled_uavs"] = _tolist(env.disabled_uavs.astype(int)) if hasattr(env, "disabled_uavs") else []
    frame["crashed_uavs"] = _tolist(env.crashed_uavs.astype(int)) if hasattr(env, "crashed_uavs") else []
    frame["stagnation_steps"] = _tolist(env.stagnation_steps) if hasattr(env, "stagnation_steps") else []
    return frame


class MAPPOTrainer:
    def __init__(self, env_factory: Callable, cfg: TrainConfig | None = None):
        require_torch()
        self.env_factory = env_factory
        self.cfg = cfg or TrainConfig()
        requested_device = str(self.cfg.device).lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        self.device = torch.device(requested_device)
        self.episode_traces = []
        self.start_episode = 0
        self.restored_history = []
        probe = env_factory()
        self.n_agents = probe.cfg.n_uavs
        self.obs_dim = probe.obs_dim()
        self.state_dim = probe.state_dim()
        self.continuous_actions = hasattr(probe, "continuous_action_dim")
        self.action_dim = (
            int(probe.continuous_action_dim())
            if self.continuous_actions
            else int(probe.action_dim()) if hasattr(probe, "action_dim") else 9
        )
        self.objective_dim = 5
        if self.continuous_actions and self.cfg.use_search_weights:
            raise ValueError("Continuous actions and legacy search-weight actions are mutually exclusive.")
        if self.cfg.use_assignment_head and self.cfg.use_search_weights:
            raise ValueError("Assignment and legacy search-weight heads are mutually exclusive.")
        if self.cfg.use_assignment_head and not self.cfg.use_gat:
            raise ValueError("Assignment head requires the heterogeneous graph encoder.")
        self.assignment_slots = int(probe.cfg.n_targets + 1) if self.cfg.use_assignment_head else 0
        policy_out_dim = (
            self.objective_dim
            if self.cfg.use_search_weights
            else self.action_dim * 2 if self.continuous_actions else self.action_dim
        )
        if self.cfg.use_gat:
            actor_cls = GraphWeightActor if self.cfg.use_search_weights else GraphActor
            actor_kwargs = {}
            if actor_cls is GraphActor:
                actor_kwargs["assignment_slots"] = self.assignment_slots
            self.actor = actor_cls(
                self.obs_dim, policy_out_dim, self.cfg.hidden_dim,
                self.cfg.gat_heads, self.cfg.gat_layers, self.cfg.dropout,
                **actor_kwargs,
            )
        else:
            self.actor = FlatMLP(
                self.obs_dim,
                policy_out_dim,
                self.cfg.hidden_dim,
                positive_output=self.cfg.use_search_weights,
            )
        self.actors = nn.ModuleList([self.actor])
        # A shared centralized critic is the correct value baseline for the
        # shared team reward and avoids N redundant, mutually drifting critics.
        self.critic = MLP(self.state_dim, 1, self.cfg.hidden_dim)
        self.critics = nn.ModuleList([self.critic])
        self.actors.to(self.device)
        self.critic.to(self.device)
        self.actor_optim = torch.optim.Adam(self.actors.parameters(), lr=self.cfg.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.validation_callback = None
        self.checkpoint_env_config = None
        self.best_validation_score = -float("inf")
        self.algorithm_name = "mappo"

    def save_checkpoint(self, path, env_config: dict | None = None, episode: int | None = None, history=None) -> None:
        payload = {
            "format_version": 4,
            "algorithm": self.algorithm_name,
            "train_config": asdict(self.cfg),
            "env_config": env_config or {},
            "actors": self.actors.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_agents": self.n_agents,
            "continuous_actions": self.continuous_actions,
            "action_dim": self.action_dim,
            "assignment_slots": self.assignment_slots,
            "episode": int(self.start_episode if episode is None else episode),
            "history": list(history or []),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if payload.get("algorithm") != self.algorithm_name:
            raise ValueError(
                f"Expected a {self.algorithm_name.upper()} checkpoint, "
                f"got {payload.get('algorithm')!r}."
            )
        if int(payload.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError(
                "Checkpoint observation dimension does not match the current environment; "
                "train a fresh policy for the 53-D full-quadrotor task."
            )
        if int(payload.get("action_dim", self.action_dim)) != self.action_dim:
            raise ValueError("Checkpoint action dimension does not match the current environment.")
        if bool(payload.get("continuous_actions", self.continuous_actions)) != self.continuous_actions:
            raise ValueError("Checkpoint discrete/continuous action type does not match the current environment.")
        if int(payload.get("assignment_slots", 0)) != self.assignment_slots:
            raise ValueError("Checkpoint assignment-head structure does not match the current trainer.")
        self.actors.load_state_dict(payload["actors"])
        self.critic.load_state_dict(payload["critic"])
        if "actor_optimizer" in payload:
            self.actor_optim.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optim.load_state_dict(payload["critic_optimizer"])
        self.start_episode = int(payload.get("episode", 0))
        self.restored_history = list(payload.get("history", []))
        return payload

    def _make_env(self, episode: int):
        try:
            return self.env_factory(episode)
        except TypeError:
            return self.env_factory()

    def train(self) -> list[dict]:
        history = list(self.restored_history)
        started_at = time.perf_counter()
        print(
            f"[{self.algorithm_name.upper()}] device={self.device}, episodes={self.start_episode + 1}..{self.cfg.episodes}, "
            f"batched_ppo=True",
            flush=True,
        )
        for episode in range(self.start_episode, self.cfg.episodes):
            episode_started_at = time.perf_counter()
            env = self._make_env(episode)
            obs = env.observe_search()
            rollout = []
            trace = []
            total_reward = 0.0
            total_collisions = 0
            total_uav_collisions = 0
            total_obstacle_conflicts = 0
            total_pair_conflicts = 0
            outage_rates = []
            disabled_rates = []
            safety_intervention_rates = []
            action_agreement_rates = []
            total_new_cells = 0
            total_revisit_gain = 0.0
            stagnant_agent_rates = []
            total_new_crashes = 0
            first_detection_step = None
            target_loss_count = 0
            target_reacquisition_count = 0
            last_target_loss_step = None
            target_reacquisition_delays = []
            direct_visibility_steps = 0
            capture_time = 0
            track_position_rmse_steps = []
            track_velocity_rmse_steps = []
            track_nees_steps = []
            track_nees_consistency_steps = []
            track_age_steps = []
            valid_track_rate_steps = []
            assignment_search_rates = []
            assignment_switch_rates = []
            total_track_messages = 0
            total_track_communication_bytes = 0
            total_active_uav_exposure = 0
            mpc_feasible_rates = []
            total_path_length = 0.0
            total_control_smoothness_cost = 0.0
            episode_search_weights = []
            rollout_limit = self.cfg.max_steps or env.cfg.search_steps
            for step_index in range(rollout_limit):
                obs_vec = torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=self.device)
                adjacency = torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=self.device)
                hetero_graph = tensorize_heterogeneous_graph(obs["hetero_graph"], self.device) if self.cfg.use_gat and self.cfg.use_hetero_entities else None
                state = torch.as_tensor(obs["state_vector"], dtype=torch.float32, device=self.device)
                masks = torch.as_tensor(obs["action_masks"], dtype=torch.bool, device=self.device)
                actions, policy_samples, logps = [], [], []
                assignment_samples = None
                assignment_logps = None
                if self.cfg.use_assignment_head:
                    encoded_policy, assignment_keys, assignment_logits = self.actor.encode_with_assignment(
                        obs_vec, adjacency, hetero_graph
                    )
                    assignment_dist = torch.distributions.Categorical(logits=assignment_logits)
                    assignment_samples = assignment_dist.sample()
                    assignment_logps = assignment_dist.log_prob(assignment_samples)
                    all_policy_outputs = self.actor.motion_for_assignment(
                        encoded_policy, assignment_keys, assignment_samples
                    )
                    if hasattr(env, "set_assignments"):
                        env.set_assignments(assignment_samples.detach().cpu().numpy())
                else:
                    all_policy_outputs = self.actor(obs_vec, adjacency, hetero_graph) if self.actor is not None else None
                value = self._value_estimate(obs_vec, state)
                if self.continuous_actions:
                    means, log_stds = torch.chunk(all_policy_outputs, 2, dim=-1)
                    log_stds = torch.clamp(
                        log_stds, -5.0, self.cfg.continuous_log_std_max
                    )
                    distributions = torch.distributions.Normal(means, torch.exp(log_stds))
                    raw_samples = distributions.rsample()
                    squashed = torch.tanh(raw_samples)
                    corrected_logps = (
                        distributions.log_prob(raw_samples)
                        - torch.log(torch.clamp(1.0 - squashed.pow(2), min=1e-6))
                    ).sum(dim=-1)
                    actions = squashed.detach().cpu().numpy()
                    policy_samples = [raw_samples[i].detach() for i in range(self.n_agents)]
                    if assignment_logps is not None:
                        corrected_logps = corrected_logps + assignment_logps
                    logps = [corrected_logps[i] for i in range(self.n_agents)]
                elif self.cfg.use_search_weights:
                    sampled_weights = []
                    for i in range(self.n_agents):
                        concentration = (
                            all_policy_outputs[i]
                            if all_policy_outputs is not None
                            else F.softplus(self.actors[i](obs_vec[i])) + 0.2
                        )
                        concentration = torch.clamp(concentration, min=0.05, max=100.0)
                        dist = torch.distributions.Dirichlet(concentration)
                        sample = dist.sample()
                        sampled_weights.append(sample)
                        policy_samples.append(sample.detach())
                        logps.append(dist.log_prob(sample))
                    weight_array = torch.stack(sampled_weights).detach().cpu().numpy()
                    actions = env.actions_from_search_weights(weight_array).astype(int).tolist()
                    episode_search_weights.append(weight_array)
                else:
                    for i in range(self.n_agents):
                        logits = all_policy_outputs[i] if all_policy_outputs is not None else self.actors[i](obs_vec[i])
                        logits = logits.masked_fill(~masks[i], -1e9)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample()
                        actions.append(int(action.item()))
                        policy_samples.append(action.detach())
                        logps.append(dist.log_prob(action))
                    if assignment_logps is not None:
                        logps = [logps[i] + assignment_logps[i] for i in range(self.n_agents)]
                result = env.step_joint(
                    np.asarray(actions, dtype=float if self.continuous_actions else int)
                )
                reward = float(result["reward"])
                total_reward += reward
                total_collisions += int(result.get("collisions", 0))
                total_uav_collisions += int(result.get("uav_collisions", 0))
                total_obstacle_conflicts += int(result.get("obstacle_conflicts", 0))
                total_pair_conflicts += int(result.get("pair_conflicts", 0))
                outage_rates.append(float(result.get("outage_rate", 0.0)))
                disabled_rates.append(float(result.get("disabled_rate", 0.0)))
                safety_intervention_rates.append(float(result.get("safety_intervention_rate", 0.0)))
                action_agreement_rates.append(float(result.get("action_agreement_rate", 1.0)))
                total_new_cells += int(result.get("new_cells", 0))
                total_revisit_gain += float(result.get("revisit_gain", 0.0))
                stagnant_agent_rates.append(float(result.get("stagnant_agents", 0)) / max(self.n_agents, 1))
                total_new_crashes += int(result.get("new_crashes", 0))
                directly_visible = bool(result.get("direct_target_visible", 0.0))
                direct_visibility_steps += int(directly_visible)
                if directly_visible and first_detection_step is None:
                    first_detection_step = step_index + 1
                target_loss_count += int(bool(result.get("target_lost", 0.0)))
                target_reacquisition_count += int(bool(result.get("target_reacquired", 0.0)))
                if bool(result.get("target_lost", 0.0)):
                    last_target_loss_step = step_index + 1
                if bool(result.get("target_reacquired", 0.0)) and last_target_loss_step is not None:
                    target_reacquisition_delays.append(
                        step_index + 1 - last_target_loss_step
                    )
                    last_target_loss_step = None
                if bool(result.get("capture_success", 0.0)) and capture_time == 0:
                    capture_time = step_index + 1
                for key, destination in (
                    ("track_position_rmse", track_position_rmse_steps),
                    ("track_velocity_rmse", track_velocity_rmse_steps),
                    ("mean_track_nees", track_nees_steps),
                    ("track_nees_consistency_rate", track_nees_consistency_steps),
                    ("mean_track_age", track_age_steps),
                    ("valid_track_rate", valid_track_rate_steps),
                    ("assignment_search_rate", assignment_search_rates),
                    ("assignment_switch_rate", assignment_switch_rates),
                ):
                    metric_value = float(result.get(key, np.nan))
                    if np.isfinite(metric_value):
                        destination.append(metric_value)
                total_track_messages += int(result.get("track_messages", 0))
                total_track_communication_bytes += int(
                    result.get("track_communication_bytes", 0)
                )
                total_active_uav_exposure += int(
                    result.get("active_uav_exposure", self.n_agents)
                )
                if "mpc_feasible_rate" in result:
                    mpc_feasible_rates.append(float(result["mpc_feasible_rate"]))
                total_path_length += float(result.get("step_path_length", 0.0))
                total_control_smoothness_cost += float(
                    result.get("control_smoothness_cost", 0.0)
                )
                next_state = torch.as_tensor(
                    result["obs"]["state_vector"], dtype=torch.float32, device=self.device
                )
                terminated = bool(result.get("terminated", False))
                truncated = bool(result.get("truncated", False))
                if step_index + 1 >= rollout_limit and not terminated:
                    truncated = True
                rollout.append(
                    (
                        obs_vec,
                        adjacency,
                        hetero_graph,
                        state,
                        masks,
                        actions,
                        policy_samples,
                        torch.stack(logps).detach(),
                        value.detach(),
                        reward * self.cfg.reward_scale,
                        next_state,
                        terminated,
                        truncated,
                        1.0 - float(result.get("coverage", env.coverage())),
                        torch.as_tensor(
                            result["obs"]["agent_observations"],
                            dtype=torch.float32,
                            device=self.device,
                        ),
                        assignment_samples.detach() if assignment_samples is not None else None,
                    )
                )
                trace.append(capture_env_frame(env, result, actions))
                obs = result["obs"]
                if terminated or truncated:
                    break
            update_stats = self._update(rollout, episode)
            mean_search_weights = (
                np.mean(np.concatenate(episode_search_weights, axis=0), axis=0)
                if episode_search_weights
                else np.full(self.objective_dim, np.nan)
            )
            self.episode_traces.append({"episode": episode + 1, "frames": trace})
            self.episode_traces = self.episode_traces[-3:]
            discovery_only = env.cfg.target_completion_steps <= 0
            task_progress = episode_task_metrics(env)
            history.append(
                {
                    "episode": episode + 1,
                    "reward": total_reward,
                    "coverage": env.coverage(),
                    **task_progress,
                    "collisions": total_collisions,
                    "uav_collisions": total_uav_collisions,
                    "obstacle_conflicts": total_obstacle_conflicts,
                    "pair_conflicts": total_pair_conflicts,
                    "collision_rate": float(total_collisions / max(env.t, 1)),
                    "outage_rate": float(np.mean(outage_rates)) if outage_rates else 0.0,
                    "disabled_rate": float(np.mean(disabled_rates)) if disabled_rates else 0.0,
                    "safety_intervention_rate": float(np.mean(safety_intervention_rates)) if safety_intervention_rates else 0.0,
                    "action_agreement_rate": float(np.mean(action_agreement_rates)) if action_agreement_rates else 1.0,
                    "new_cells": total_new_cells,
                    "revisit_gain": total_revisit_gain,
                    "stagnant_agent_rate": float(np.mean(stagnant_agent_rates)) if stagnant_agent_rates else 0.0,
                    "new_crashes": total_new_crashes,
                    "capture_success": float(result.get("capture_success", 0.0)),
                    "capture_hold_count": int(result.get("capture_hold_count", 0)),
                    "capture_close_uavs": int(result.get("capture_close_uavs", 0)),
                    "capture_angular_span_deg": float(result.get("capture_angular_span_deg", 0.0)),
                    "direct_target_visible": float(result.get("direct_target_visible", 0.0)),
                    # In post-detection pursuit this is the first direct onboard
                    # contact after the upstream track handoff. Keep the old
                    # key as a compatibility alias for existing CSV readers.
                    "first_onboard_contact_time": int(first_detection_step or (rollout_limit + 1)),
                    "first_detection_time": int(first_detection_step or (rollout_limit + 1)),
                    "capture_time": int(capture_time or (rollout_limit + 1)),
                    "target_loss_count": int(target_loss_count),
                    "target_reacquisition_count": int(target_reacquisition_count),
                    "mean_reacquisition_time": float(np.mean(target_reacquisition_delays))
                    if target_reacquisition_delays
                    else np.nan,
                    "continuous_visibility_ratio": float(direct_visibility_steps / max(env.t, 1)),
                    "track_position_rmse": float(
                        np.sqrt(np.mean(np.square(track_position_rmse_steps)))
                        if track_position_rmse_steps
                        else np.nan
                    ),
                    "track_velocity_rmse": float(
                        np.sqrt(np.mean(np.square(track_velocity_rmse_steps)))
                        if track_velocity_rmse_steps
                        else np.nan
                    ),
                    "mean_track_nees": float(np.mean(track_nees_steps)) if track_nees_steps else np.nan,
                    "track_nees_consistency_rate": float(np.mean(track_nees_consistency_steps))
                    if track_nees_consistency_steps
                    else np.nan,
                    "mean_track_age": float(np.mean(track_age_steps)) if track_age_steps else np.nan,
                    "valid_track_rate": float(np.mean(valid_track_rate_steps))
                    if valid_track_rate_steps
                    else 0.0,
                    "track_messages": int(total_track_messages),
                    "track_communication_bytes": int(total_track_communication_bytes),
                    "assignment_search_rate": float(np.mean(assignment_search_rates))
                    if assignment_search_rates
                    else 1.0,
                    "assignment_switch_rate": float(np.mean(assignment_switch_rates))
                    if assignment_switch_rates
                    else 0.0,
                    "collision_exposure_rate": float(
                        total_collisions / max(total_active_uav_exposure, 1)
                    ),
                    "mpc_feasible_rate": float(np.mean(mpc_feasible_rates))
                    if mpc_feasible_rates
                    else np.nan,
                    "path_length": float(total_path_length),
                    "control_smoothness_cost": float(total_control_smoothness_cost),
                    "weight_target": float(mean_search_weights[0]),
                    "weight_exploration": float(mean_search_weights[1]),
                    "weight_revisit": float(mean_search_weights[2]),
                    "weight_connectivity": float(mean_search_weights[3]),
                    "weight_safety": float(mean_search_weights[4]),
                    "steps": env.t,
                    **update_stats,
                }
            )
            if self.continuous_actions and hasattr(self, "finalize_pursuit_row"):
                history[-1] = self.finalize_pursuit_row(history[-1])
            completed = episode + 1
            episode_seconds = time.perf_counter() - episode_started_at
            if completed == 1 or completed % max(1, self.cfg.log_interval) == 0 or completed == self.cfg.episodes:
                elapsed = time.perf_counter() - started_at
                run_count = completed - self.start_episode
                eta = elapsed / max(run_count, 1) * max(self.cfg.episodes - completed, 0)
                row = history[-1]
                pursuit_mode = hasattr(env, "captured")
                if pursuit_mode:
                    rmse = row.get("track_position_rmse", float("nan"))
                    task_metrics = (
                        f"capture={row['capture_success']:.0f} "
                        f"first_obs_step={row['first_onboard_contact_time']} "
                        f"visibility={row['continuous_visibility_ratio']:.3f} "
                        f"target_losses={row['target_loss_count']} "
                        f"capture_rate_50={np.mean([r['capture_success'] for r in history[-50:]]):.3f} "
                        f"track_rmse={rmse:.3f}"
                    )
                elif discovery_only:
                    task_metrics = (
                        f"discovery={row['target_discovery_rate']:.3f} "
                        f"success={row['task_success']:.0f}"
                    )
                else:
                    task_metrics = (
                        f"discovery={row['target_discovery_rate']:.3f} "
                        f"tracking={row['target_tracking_rate']:.3f} "
                        f"completion={row['target_completion_rate']:.3f} "
                        f"track_gap={row['target_discovery_rate'] - row['target_tracking_rate']:.3f}"
                    )
                auxiliary = "" if pursuit_mode else f"coverage={row['coverage']:.3f} "
                if pursuit_mode and "environment_seconds" in row:
                    auxiliary = f"env={row['environment_seconds']:.2f}s ppo={row['ppo_update_seconds']:.2f}s "
                print(
                    f"[MAPPO] episode={completed}/{self.cfg.episodes} reward={row['reward']:.2f} "
                    f"{auxiliary}{task_metrics} "
                    f"entropy={row.get('entropy', float('nan')):.3f} "
                    f"time={episode_seconds:.2f}s eta={eta / 3600:.2f}h",
                    flush=True,
                )
            if self.cfg.checkpoint_path and (
                completed % max(1, self.cfg.checkpoint_interval) == 0 or completed == self.cfg.episodes
            ):
                self.save_checkpoint(self.cfg.checkpoint_path, episode=completed, history=history)
        self.start_episode = self.cfg.episodes
        self.restored_history = history
        return history

    def _value_estimate(self, obs_vec, state):
        return self.critic(state).squeeze()

    def _update(self, rollout, episode: int) -> dict:
        if not rollout:
            return {}
        rewards = torch.tensor([item[9] for item in rollout], dtype=torch.float32, device=self.device)
        # Rollout layout: [..., value, reward, next_state, terminated, truncated].
        # Only a true terminal state suppresses bootstrapping; a time-limit
        # truncation must bootstrap from the post-action next_state.
        terminated_flags = torch.tensor(
            [float(item[11]) for item in rollout], dtype=torch.float32, device=self.device
        )
        old_values = torch.stack([item[8] for item in rollout]).float()
        advantages = torch.zeros_like(rewards)
        gae = torch.tensor(0.0, device=self.device)
        next_value = torch.tensor(0.0, device=self.device)
        # Standard GAE(lambda), with terminal masking. A rollout that reaches
        # the configured horizon is treated as a time-limit truncation and is
        # bootstrapped from the final observation.
        if not bool(terminated_flags[-1]):
            final_transition = rollout[-1]
            next_value = self.critic(final_transition[10]).detach().squeeze()
        for t in reversed(range(len(rollout))):
            nonterminal = 1.0 - terminated_flags[t]
            delta = rewards[t] + self.cfg.gamma * next_value * nonterminal - old_values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_value = old_values[t]
        returns = advantages + old_values
        if self.cfg.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        if self.cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        progress = episode / max(self.cfg.episodes - 1, 1)
        lr_factor = 1.0 - progress * (1.0 - self.cfg.lr_end_factor)
        entropy_coef = self.cfg.entropy_coef + progress * (self.cfg.entropy_coef_end - self.cfg.entropy_coef)
        rollout_uncertainty = float(
            np.clip(np.mean([item[13] for item in rollout]), 0.0, 1.0)
        )
        if self.cfg.adaptive_entropy:
            entropy_coef *= 1.0 + self.cfg.uncertainty_entropy_gain * rollout_uncertainty
        current_lr = self.cfg.lr * lr_factor
        for optimizer in (self.actor_optim, self.critic_optim):
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        return_variance = torch.var(returns, unbiased=False)
        explained_variance = 1.0 - torch.var(returns - old_values, unbiased=False) / torch.clamp(return_variance, min=1e-8)
        loss_values, policy_values, value_values, entropy_values, kl_values, clip_values = [], [], [], [], [], []
        actor_grad_values, critic_grad_values = [], []
        early_stop = False
        for _ in range(max(1, self.cfg.update_epochs)):
            for indices in torch.randperm(len(rollout)).split(max(1, self.cfg.batch_size)):
                idx_list = [int(item) for item in indices]
                obs_batch = torch.stack([rollout[idx][0] for idx in idx_list])
                adjacency_batch = torch.stack([rollout[idx][1] for idx in idx_list])
                state_batch = torch.stack([rollout[idx][3] for idx in idx_list])
                mask_batch = torch.stack([rollout[idx][4] for idx in idx_list])
                old_logps = torch.stack([rollout[idx][7] for idx in idx_list])
                sample_batch = torch.stack(
                    [torch.stack(rollout[idx][6]) for idx in idx_list]
                )
                hetero_batch = None
                if self.cfg.use_gat and self.cfg.use_hetero_entities:
                    hetero_batch = {
                        key: torch.stack([rollout[idx][2][key] for idx in idx_list])
                        for key in rollout[idx_list[0]][2]
                    }
                assignment_logits = None
                if self.actor is not None:
                    if self.cfg.use_assignment_head:
                        policy_outputs, assignment_logits = self.actor.forward_with_assignment(
                            obs_batch,
                            adjacency_batch,
                            hetero_batch,
                            torch.stack([rollout[idx][15] for idx in idx_list]).long(),
                        )
                    else:
                        policy_outputs = self.actor(obs_batch, adjacency_batch, hetero_batch)
                else:
                    policy_outputs = torch.stack(
                        [self.actors[i](obs_batch[:, i]) for i in range(self.n_agents)], dim=1
                    )
                if self.continuous_actions:
                    means, log_stds = torch.chunk(policy_outputs, 2, dim=-1)
                    log_stds = torch.clamp(
                        log_stds, -5.0, self.cfg.continuous_log_std_max
                    )
                    dist = torch.distributions.Normal(means, torch.exp(log_stds))
                    squashed = torch.tanh(sample_batch)
                    new_logps = (
                        dist.log_prob(sample_batch)
                        - torch.log(torch.clamp(1.0 - squashed.pow(2), min=1e-6))
                    ).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                elif self.cfg.use_search_weights:
                    concentration = policy_outputs if self.actor is not None else F.softplus(policy_outputs) + 0.2
                    concentration = torch.clamp(concentration, min=0.05, max=100.0)
                    dist = torch.distributions.Dirichlet(concentration)
                    new_logps = dist.log_prob(sample_batch)
                    entropy = dist.entropy().mean()
                else:
                    logits = policy_outputs.masked_fill(~mask_batch, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logps = dist.log_prob(sample_batch.long())
                    entropy = dist.entropy().mean()
                assignment_entropy = torch.tensor(0.0, device=self.device)
                if self.cfg.use_assignment_head:
                    assignment_batch = torch.stack([rollout[idx][15] for idx in idx_list]).long()
                    assignment_dist = torch.distributions.Categorical(logits=assignment_logits)
                    new_logps = new_logps + assignment_dist.log_prob(assignment_batch)
                    assignment_entropy = assignment_dist.entropy().mean()
                ratio = torch.exp(torch.clamp(new_logps - old_logps, min=-20.0, max=20.0))
                clipped = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                adv = advantages[indices.to(self.device)].detach().unsqueeze(-1)
                pg = -torch.min(ratio * adv, clipped * adv).mean()
                predicted_values = self.critic(state_batch).squeeze(-1)
                vf = F.mse_loss(predicted_values, returns[indices.to(self.device)])
                actor_loss = (
                    pg
                    - entropy_coef * entropy
                    - self.cfg.assignment_entropy_coef * assignment_entropy
                )
                critic_loss = self.cfg.value_coef * vf
                if self.continuous_actions and hasattr(self, "pursuit_demo_loss"):
                    actor_loss = actor_loss + self.pursuit_demo_loss(episode)
                loss = actor_loss + critic_loss

                # The policy and centralized critic have independent networks.
                # Backpropagate and clip them separately so a large value-loss
                # gradient cannot suppress the policy gradient through a joint
                # global-norm clip.
                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actors.parameters(), self.cfg.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm
                )
                self.actor_optim.step()
                self.critic_optim.step()
                loss_values.append(float(loss.detach()))
                policy_values.append(float(pg.detach()))
                value_values.append(float(vf.detach()))
                entropy_values.append(float(entropy.detach()))
                actor_grad_values.append(float(actor_grad_norm.detach()))
                critic_grad_values.append(float(critic_grad_norm.detach()))
                approx_kl = float((old_logps - new_logps).mean().detach())
                kl_values.append(approx_kl)
                clip_values.append(float((torch.abs(ratio - 1.0) > self.cfg.clip_eps).float().mean().detach()))
                if self.cfg.target_kl > 0 and approx_kl > 1.5 * self.cfg.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break
        return {
            "policy_loss": float(np.mean(policy_values)),
            "value_loss": float(np.mean(value_values)),
            "entropy": float(np.mean(entropy_values)),
            "learning_rate": float(self.actor_optim.param_groups[0]["lr"]),
            "actor_grad_norm": float(np.mean(actor_grad_values)) if actor_grad_values else 0.0,
            "critic_grad_norm": float(np.mean(critic_grad_values)) if critic_grad_values else 0.0,
            "entropy_coef": float(entropy_coef),
            "rollout_uncertainty": rollout_uncertainty,
            "approx_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "ppo_early_stop": float(early_stop),
            "clip_fraction": float(np.mean(clip_values)) if clip_values else 0.0,
            "explained_variance": float(explained_variance.detach()),
        }


class IPPOTrainer(MAPPOTrainer):
    """Parameter-shared IPPO with decentralized per-agent value estimation.

    The actor and critic consume only each UAV's local observation. The shared
    team reward is replicated across agents, but neither network receives the
    centralized state or peer graph. This makes IPPO a controlled baseline for
    measuring the benefit of CTDE and relational encoding.
    """

    def __init__(self, env_factory: Callable, cfg: TrainConfig | None = None):
        cfg = cfg or TrainConfig()
        cfg.use_gat = False
        cfg.use_hetero_entities = False
        super().__init__(env_factory, cfg)
        self.algorithm_name = "ippo"
        self.critic = MLP(self.obs_dim, 1, self.cfg.hidden_dim).to(self.device)
        self.critics = nn.ModuleList([self.critic])
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)

    def _value_estimate(self, obs_vec, state):
        del state
        return self.critic(obs_vec).squeeze(-1)

    def _update(self, rollout, episode: int) -> dict:
        if not rollout:
            return {}
        rewards = torch.tensor(
            [item[9] for item in rollout], dtype=torch.float32, device=self.device
        ).unsqueeze(-1).expand(-1, self.n_agents)
        terminated = torch.tensor(
            [float(item[11]) for item in rollout],
            dtype=torch.float32,
            device=self.device,
        )
        old_values = torch.stack([item[8] for item in rollout]).float()
        advantages = torch.zeros_like(old_values)
        gae = torch.zeros(self.n_agents, device=self.device)
        next_value = torch.zeros(self.n_agents, device=self.device)
        if not bool(terminated[-1]):
            next_value = self.critic(rollout[-1][14]).detach().squeeze(-1)
        for t in reversed(range(len(rollout))):
            nonterminal = 1.0 - terminated[t]
            delta = (
                rewards[t]
                + self.cfg.gamma * next_value * nonterminal
                - old_values[t]
            )
            gae = (
                delta
                + self.cfg.gamma
                * self.cfg.gae_lambda
                * nonterminal
                * gae
            )
            advantages[t] = gae
            next_value = old_values[t]
        returns = advantages + old_values
        if self.cfg.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (
                returns.std(unbiased=False) + 1e-8
            )
        if self.cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        progress = episode / max(self.cfg.episodes - 1, 1)
        lr_factor = 1.0 - progress * (1.0 - self.cfg.lr_end_factor)
        entropy_coef = self.cfg.entropy_coef + progress * (
            self.cfg.entropy_coef_end - self.cfg.entropy_coef
        )
        rollout_uncertainty = float(
            np.clip(np.mean([item[13] for item in rollout]), 0.0, 1.0)
        )
        if self.cfg.adaptive_entropy:
            entropy_coef *= (
                1.0
                + self.cfg.uncertainty_entropy_gain * rollout_uncertainty
            )
        current_lr = self.cfg.lr * lr_factor
        for optimizer in (self.actor_optim, self.critic_optim):
            for group in optimizer.param_groups:
                group["lr"] = current_lr

        return_variance = torch.var(returns, unbiased=False)
        explained_variance = 1.0 - torch.var(
            returns - old_values, unbiased=False
        ) / torch.clamp(return_variance, min=1e-8)
        policy_values, value_values, entropy_values = [], [], []
        actor_grad_values, critic_grad_values, kl_values, clip_values = (
            [],
            [],
            [],
            [],
        )
        early_stop = False
        for _ in range(max(1, self.cfg.update_epochs)):
            for indices in torch.randperm(len(rollout)).split(
                max(1, self.cfg.batch_size)
            ):
                idx_list = [int(item) for item in indices]
                obs_batch = torch.stack(
                    [rollout[idx][0] for idx in idx_list]
                )
                mask_batch = torch.stack(
                    [rollout[idx][4] for idx in idx_list]
                )
                old_logps = torch.stack(
                    [rollout[idx][7] for idx in idx_list]
                )
                sample_batch = torch.stack(
                    [torch.stack(rollout[idx][6]) for idx in idx_list]
                )
                policy_outputs = self.actor(obs_batch)
                if self.cfg.use_search_weights:
                    concentration = torch.clamp(
                        policy_outputs, min=0.05, max=100.0
                    )
                    dist = torch.distributions.Dirichlet(concentration)
                    new_logps = dist.log_prob(sample_batch)
                    entropy = dist.entropy().mean()
                else:
                    logits = policy_outputs.masked_fill(~mask_batch, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logps = dist.log_prob(sample_batch.long())
                    entropy = dist.entropy().mean()
                ratio = torch.exp(
                    torch.clamp(new_logps - old_logps, min=-20.0, max=20.0)
                )
                clipped = torch.clamp(
                    ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps
                )
                adv = advantages[indices.to(self.device)].detach()
                pg = -torch.min(ratio * adv, clipped * adv).mean()
                predicted_values = self.critic(obs_batch).squeeze(-1)
                vf = F.mse_loss(
                    predicted_values, returns[indices.to(self.device)]
                )
                actor_loss = pg - entropy_coef * entropy
                critic_loss = self.cfg.value_coef * vf

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                actor_loss.backward()
                critic_loss.backward()
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actors.parameters(), self.cfg.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.cfg.max_grad_norm
                )
                self.actor_optim.step()
                self.critic_optim.step()

                policy_values.append(float(pg.detach()))
                value_values.append(float(vf.detach()))
                entropy_values.append(float(entropy.detach()))
                actor_grad_values.append(float(actor_grad_norm.detach()))
                critic_grad_values.append(float(critic_grad_norm.detach()))
                approx_kl = float((old_logps - new_logps).mean().detach())
                kl_values.append(approx_kl)
                clip_values.append(
                    float(
                        (
                            torch.abs(ratio - 1.0) > self.cfg.clip_eps
                        )
                        .float()
                        .mean()
                        .detach()
                    )
                )
                if (
                    self.cfg.target_kl > 0
                    and approx_kl > 1.5 * self.cfg.target_kl
                ):
                    early_stop = True
                    break
            if early_stop:
                break
        return {
            "policy_loss": float(np.mean(policy_values)),
            "value_loss": float(np.mean(value_values)),
            "entropy": float(np.mean(entropy_values)),
            "learning_rate": float(self.actor_optim.param_groups[0]["lr"]),
            "actor_grad_norm": float(np.mean(actor_grad_values)),
            "critic_grad_norm": float(np.mean(critic_grad_values)),
            "entropy_coef": float(entropy_coef),
            "rollout_uncertainty": rollout_uncertainty,
            "approx_kl": float(np.mean(kl_values)),
            "ppo_early_stop": float(early_stop),
            "clip_fraction": float(np.mean(clip_values)),
            "explained_variance": float(explained_variance.detach()),
        }


class DualMAPPOTrainer:
    """MAPPO trainer for search under outage-based weak communication."""

    def __init__(self, env_factory: Callable, cfg: TrainConfig | None = None):
        require_torch()
        self.env_factory = env_factory
        self.cfg = cfg or TrainConfig()
        self.episode_traces = []
        probe = env_factory()
        self.n_agents = probe.cfg.n_uavs
        self.obs_dim = probe.obs_dim()
        self.state_dim = probe.state_dim()
        self.search_action_dim = 9
        if self.cfg.use_gat:
            self.search_actor = GraphActor(
                self.obs_dim,
                self.search_action_dim,
                self.cfg.hidden_dim,
                self.cfg.gat_heads,
                self.cfg.gat_layers,
                self.cfg.dropout,
            )
            self.search_actors = nn.ModuleList([self.search_actor])
        else:
            self.search_actor = None
            self.search_actors = nn.ModuleList([MLP(self.obs_dim, self.search_action_dim, self.cfg.hidden_dim) for _ in range(self.n_agents)])
        self.critics = nn.ModuleList([MLP(self.state_dim, 1, self.cfg.hidden_dim) for _ in range(self.n_agents)])
        params = list(self.search_actors.parameters()) + list(self.critics.parameters())
        self.optim = torch.optim.Adam(params, lr=self.cfg.lr)

    def train(self) -> list[dict]:
        history = []
        for episode in range(self.cfg.episodes):
            env = self.env_factory()
            obs = env.observe_search()
            rollout = []
            trace = []
            total_reward = 0.0
            total_collisions = 0
            total_uav_collisions = 0
            total_obstacle_conflicts = 0
            total_pair_conflicts = 0
            spectrum_success = []
            for _ in range(self.cfg.max_steps or env.cfg.search_steps):
                obs_vec = torch.tensor(obs["agent_observations"], dtype=torch.float32)
                adjacency = torch.tensor(obs["comm_adjacency"], dtype=torch.bool)
                state = torch.tensor(obs["state_vector"], dtype=torch.float32)
                search_masks = torch.tensor(obs["action_masks"], dtype=torch.bool)
                search_actions, search_logps, values = [], [], []
                all_search_logits = self.search_actor(obs_vec, adjacency) if self.search_actor is not None else None
                for i in range(self.n_agents):
                    search_logits = all_search_logits[i] if all_search_logits is not None else self.search_actors[i](obs_vec[i])
                    search_logits = search_logits.masked_fill(~search_masks[i], -1e9)
                    search_dist = torch.distributions.Categorical(logits=search_logits)
                    search_action = search_dist.sample()
                    search_actions.append(int(search_action.item()))
                    search_logps.append(search_dist.log_prob(search_action))
                    values.append(self.critics[i](state).squeeze())
                result = env.step_joint(np.asarray(search_actions, dtype=int))
                reward = float(result["reward"])
                total_reward += reward
                total_collisions += int(result.get("collisions", 0))
                total_uav_collisions += int(result.get("uav_collisions", 0))
                total_obstacle_conflicts += int(result.get("obstacle_conflicts", 0))
                total_pair_conflicts += int(result.get("pair_conflicts", 0))
                spectrum_success.append(float(result.get("spectrum_success_rate", 0.0)))
                rollout.append(
                    (
                        obs_vec,
                        adjacency,
                        state,
                        search_masks,
                        search_actions,
                        torch.stack(search_logps).detach(),
                        torch.stack(values).detach(),
                        reward * self.cfg.reward_scale,
                        result["done"],
                    )
                )
                trace.append(capture_env_frame(env, result, search_actions))
                obs = result["obs"]
                if result["done"]:
                    break
            self._update(rollout)
            self.episode_traces.append({"episode": episode + 1, "frames": trace})
            self.episode_traces = self.episode_traces[-3:]
            history.append(
                {
                    "episode": episode + 1,
                    "reward": total_reward,
                    "spectrum_reward": 0.0,
                    "spectrum_success_rate": float(np.mean(spectrum_success)) if spectrum_success else 0.0,
                    "coverage": env.coverage(),
                    "target_discovery_rate": float(np.mean(env.found_targets)),
                    "target_tracking_rate": float(np.mean(env.tracked_targets | env.completed_targets)),
                    "target_completion_rate": float(np.mean(env.completed_targets)),
                    "collisions": total_collisions,
                    "uav_collisions": total_uav_collisions,
                    "obstacle_conflicts": total_obstacle_conflicts,
                    "pair_conflicts": total_pair_conflicts,
                    "collision_rate": float(total_collisions / max(env.t, 1)),
                    "steps": env.t,
                }
            )
        return history

    def _update(self, rollout) -> None:
        if not rollout:
            return
        returns = []
        running = 0.0
        for *_, reward, done in reversed(rollout):
            running = reward + self.cfg.gamma * running * (1.0 - float(done))
            returns.append(running)
        returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
        if self.cfg.normalize_returns and returns.numel() > 1:
            returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
        old_values = torch.stack([item[6] for item in rollout])
        advantages = returns[:, None] - old_values
        if self.cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        for _ in range(max(1, self.cfg.update_epochs)):
            losses = []
            for idx, item in enumerate(rollout):
                (
                    obs_vec,
                    adjacency,
                    state,
                    search_masks,
                    search_actions,
                    old_search_logps,
                    _values,
                    _reward,
                    _done,
                ) = item
                ret = returns[idx]
                all_search_logits = self.search_actor(obs_vec, adjacency) if self.search_actor is not None else None
                for i in range(self.n_agents):
                    search_logits = all_search_logits[i] if all_search_logits is not None else self.search_actors[i](obs_vec[i])
                    search_logits = search_logits.masked_fill(~search_masks[i], -1e9)
                    search_dist = torch.distributions.Categorical(logits=search_logits)
                    search_action = torch.tensor(search_actions[i])
                    search_logp = search_dist.log_prob(search_action)
                    value = self.critics[i](state).squeeze()
                    adv = advantages[idx, i].detach()
                    search_ratio = torch.exp(search_logp - old_search_logps[i])
                    search_pg = -torch.min(
                        search_ratio * adv,
                        torch.clamp(search_ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv,
                    )
                    vf = F.mse_loss(value, ret)
                    entropy = search_dist.entropy()
                    losses.append(search_pg + self.cfg.value_coef * vf - self.cfg.entropy_coef * entropy)
            loss = torch.stack(losses).mean()
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.search_actors.parameters()) + list(self.critics.parameters()),
                self.cfg.max_grad_norm,
            )
            self.optim.step()


class MixingNetwork(nn.Module if nn else object):
    def __init__(self, n_agents: int, state_dim: int, hidden_dim: int):
        require_torch()
        super().__init__()
        self.hyper_w1 = nn.Linear(state_dim, n_agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim

    def forward(self, agent_qs, states):
        bs = agent_qs.shape[0]
        w1 = torch.abs(self.hyper_w1(states)).view(bs, self.n_agents, self.hidden_dim)
        b1 = self.hyper_b1(states).view(bs, 1, self.hidden_dim)
        hidden = F.elu(torch.bmm(agent_qs.view(bs, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(states)).view(bs, self.hidden_dim, 1)
        b2 = self.hyper_b2(states).view(bs, 1, 1)
        return (torch.bmm(hidden, w2) + b2).view(bs)


class QMIXTrainer:
    def __init__(self, env_factory: Callable, cfg: TrainConfig | None = None):
        require_torch()
        self.env_factory = env_factory
        self.cfg = cfg or TrainConfig()
        self.episode_traces = []
        probe = env_factory()
        self.n_agents = probe.cfg.n_uavs
        self.obs_dim = probe.obs_dim()
        self.state_dim = probe.state_dim()
        self.action_dim = 9
        if self.cfg.use_gat:
            self.qnet = GraphQNetwork(
                self.obs_dim,
                self.action_dim,
                self.cfg.hidden_dim,
                self.cfg.gat_heads,
                self.cfg.gat_layers,
                self.cfg.dropout,
            )
            self.target_qnet = GraphQNetwork(
                self.obs_dim,
                self.action_dim,
                self.cfg.hidden_dim,
                self.cfg.gat_heads,
                self.cfg.gat_layers,
                self.cfg.dropout,
            )
        else:
            self.qnet = FlatMLP(self.obs_dim, self.action_dim, self.cfg.hidden_dim)
            self.target_qnet = FlatMLP(self.obs_dim, self.action_dim, self.cfg.hidden_dim)
        self.qnets = nn.ModuleList([self.qnet])
        self.target_qnets = nn.ModuleList([self.target_qnet])
        self.mixer = MixingNetwork(self.n_agents, self.state_dim, self.cfg.hidden_dim)
        self.target_mixer = MixingNetwork(self.n_agents, self.state_dim, self.cfg.hidden_dim)
        self.target_qnets.load_state_dict(self.qnets.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optim = torch.optim.Adam(list(self.qnets.parameters()) + list(self.mixer.parameters()), lr=self.cfg.lr)
        self.start_episode = 0
        self.restored_history = []

    def _make_env(self, episode: int):
        """Create a distinct deterministic scenario for each training episode."""
        try:
            return self.env_factory(episode)
        except TypeError:
            return self.env_factory()

    def save_checkpoint(self, path, env_config: dict | None = None, episode: int | None = None, history=None) -> None:
        """Save all online/target networks and optimizer state for QMIX."""
        payload = {
            "format_version": 1,
            "algorithm": "qmix",
            "train_config": asdict(self.cfg),
            "env_config": env_config or {},
            "qnets": self.qnets.state_dict(),
            "target_qnets": self.target_qnets.state_dict(),
            "mixer": self.mixer.state_dict(),
            "target_mixer": self.target_mixer.state_dict(),
            "optimizer": self.optim.state_dict(),
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_agents": self.n_agents,
            "episode": int(self.start_episode if episode is None else episode),
            "history": list(history or []),
        }
        torch.save(payload, path)

    def load_checkpoint(self, path) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != "qmix":
            raise ValueError(f"Expected a QMIX checkpoint, got {payload.get('algorithm')!r}.")
        if int(payload.get("obs_dim", self.obs_dim)) != self.obs_dim:
            raise ValueError("Checkpoint observation dimension does not match the current environment.")
        self.qnets.load_state_dict(payload["qnets"])
        self.target_qnets.load_state_dict(payload["target_qnets"])
        self.mixer.load_state_dict(payload["mixer"])
        self.target_mixer.load_state_dict(payload["target_mixer"])
        if "optimizer" in payload:
            self.optim.load_state_dict(payload["optimizer"])
        self.start_episode = int(payload.get("episode", 0))
        self.restored_history = list(payload.get("history", []))
        return payload

    def train(self) -> list[dict]:
        history = list(self.restored_history)
        for episode in range(self.start_episode, self.cfg.episodes):
            env = self._make_env(episode)
            obs = env.observe_search()
            transitions = []
            trace = []
            total_reward = 0.0
            total_collisions = 0
            total_uav_collisions = 0
            total_obstacle_conflicts = 0
            total_pair_conflicts = 0
            outage_rates = []
            disabled_rates = []
            eps = max(self.cfg.epsilon_end, self.cfg.epsilon_start - episode / max(self.cfg.episodes, 1))
            for _ in range(self.cfg.max_steps or env.cfg.search_steps):
                obs_vec = torch.tensor(obs["agent_observations"], dtype=torch.float32)
                adjacency = torch.tensor(obs["comm_adjacency"], dtype=torch.bool)
                hetero_graph = tensorize_heterogeneous_graph(obs["hetero_graph"]) if self.cfg.use_gat and self.cfg.use_hetero_entities else None
                masks = obs["action_masks"]
                actions = []
                with torch.no_grad():
                    all_q = self.qnet(obs_vec, adjacency, hetero_graph) if self.qnet is not None else None
                for i in range(self.n_agents):
                    if np.random.random() < eps:
                        actions.append(int(np.random.choice(np.flatnonzero(masks[i]))))
                    else:
                        with torch.no_grad():
                            q = all_q[i] if all_q is not None else self.qnets[i](obs_vec[i])
                            q[~torch.tensor(masks[i], dtype=torch.bool)] = -1e9
                            actions.append(int(torch.argmax(q).item()))
                state = obs["state_vector"]
                result = env.step_joint(np.asarray(actions, dtype=int))
                transitions.append(
                    (
                        obs["agent_observations"],
                        obs["comm_adjacency"],
                        obs["hetero_graph"],
                        state,
                        masks,
                        actions,
                        result["reward"],
                        result["obs"],
                        result["done"],
                    )
                )
                trace.append(capture_env_frame(env, result, actions))
                obs = result["obs"]
                total_reward += float(result["reward"])
                total_collisions += int(result.get("collisions", 0))
                total_uav_collisions += int(result.get("uav_collisions", 0))
                total_obstacle_conflicts += int(result.get("obstacle_conflicts", 0))
                total_pair_conflicts += int(result.get("pair_conflicts", 0))
                outage_rates.append(float(result.get("outage_rate", 0.0)))
                disabled_rates.append(float(result.get("disabled_rate", 0.0)))
                if result["done"]:
                    break
            self._update(transitions)
            self.episode_traces.append({"episode": episode + 1, "frames": trace})
            self.episode_traces = self.episode_traces[-3:]
            if (episode + 1) % self.cfg.target_update == 0:
                self.target_qnets.load_state_dict(self.qnets.state_dict())
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            task_progress = episode_task_metrics(env)
            history.append(
                {
                    "episode": episode + 1,
                    "reward": total_reward,
                    "coverage": env.coverage(),
                    **task_progress,
                    "collisions": total_collisions,
                    "uav_collisions": total_uav_collisions,
                    "obstacle_conflicts": total_obstacle_conflicts,
                    "pair_conflicts": total_pair_conflicts,
                    "collision_rate": float(total_collisions / max(env.t, 1)),
                    "outage_rate": float(np.mean(outage_rates)) if outage_rates else 0.0,
                    "disabled_rate": float(np.mean(disabled_rates)) if disabled_rates else 0.0,
                    "steps": env.t,
                }
            )
            completed = episode + 1
            if self.cfg.checkpoint_path and (
                completed % max(1, self.cfg.checkpoint_interval) == 0 or completed == self.cfg.episodes
            ):
                self.save_checkpoint(self.cfg.checkpoint_path, episode=completed, history=history)
        return history

    def _update(self, transitions) -> None:
        losses = []
        for obs_arr, adjacency_arr, hetero_arr, state_arr, _masks, actions, reward, next_obs, done in transitions:
            obs = torch.tensor(obs_arr, dtype=torch.float32)
            adjacency = torch.tensor(adjacency_arr, dtype=torch.bool)
            hetero_graph = tensorize_heterogeneous_graph(hetero_arr) if self.cfg.use_gat and self.cfg.use_hetero_entities else None
            state = torch.tensor(state_arr[None, :], dtype=torch.float32)
            next_obs_arr = torch.tensor(next_obs["agent_observations"], dtype=torch.float32)
            next_adjacency = torch.tensor(next_obs["comm_adjacency"], dtype=torch.bool)
            next_hetero = tensorize_heterogeneous_graph(next_obs["hetero_graph"]) if self.cfg.use_gat and self.cfg.use_hetero_entities else None
            next_state = torch.tensor(next_obs["state_vector"][None, :], dtype=torch.float32)
            chosen_qs, target_qs = [], []
            all_q = self.qnet(obs, adjacency, hetero_graph) if self.qnet is not None else None
            with torch.no_grad():
                all_tq = self.target_qnet(next_obs_arr, next_adjacency, next_hetero) if self.target_qnet is not None else None
            for i in range(self.n_agents):
                q_values = all_q[i] if all_q is not None else self.qnets[i](obs[i])
                q = q_values[actions[i]]
                chosen_qs.append(q)
                with torch.no_grad():
                    tq = all_tq[i] if all_tq is not None else self.target_qnets[i](next_obs_arr[i])
                    mask = torch.tensor(next_obs["action_masks"][i], dtype=torch.bool)
                    tq = tq.masked_fill(~mask, -1e9).max()
                    target_qs.append(tq)
            qtot = self.mixer(torch.stack(chosen_qs)[None, :], state)
            with torch.no_grad():
                target_qtot = self.target_mixer(torch.stack(target_qs)[None, :], next_state)
                y = torch.tensor([reward], dtype=torch.float32) + self.cfg.gamma * target_qtot * (1.0 - float(done))
            losses.append(F.mse_loss(qtot, y))
        loss = torch.stack(losses).mean()
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.qnets.parameters()) + list(self.mixer.parameters()), self.cfg.max_grad_norm)
        self.optim.step()
