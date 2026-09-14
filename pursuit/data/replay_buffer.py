"""Joint-team replay for centralized MATD3 critics."""
from __future__ import annotations

import numpy as np
import torch

from marl_trainers import tensorize_heterogeneous_graph


GRAPH_KEYS = (
    "self_nodes", "target_nodes", "target_mask", "peer_nodes", "peer_mask",
    "building_nodes", "building_mask", "uav_xyz", "active_mask",
)


def _allocate_graph(capacity, example):
    storage = {}
    for key in GRAPH_KEYS:
        value = np.asarray(example[key])
        storage[key] = np.empty((capacity, *value.shape), dtype=value.dtype)
    return storage


def _write_graph(storage, index, graph):
    for key in GRAPH_KEYS:
        storage[key][index] = np.asarray(graph[key])


def _read_graph(storage, indices):
    return {key: storage[key][indices] for key in GRAPH_KEYS}


class JointReplayBuffer:
    def __init__(self, capacity, seed=0):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.size = 0
        self.cursor = 0
        self._ready = False

    def __len__(self):
        return self.size

    def _ensure(self, row):
        if self._ready:
            return
        n_agents, action_dim = np.asarray(row["actions"]).shape
        obs_dim = np.asarray(row["agent_observations"]).shape[-1]
        state_dim = np.asarray(row["state"]).shape[-1]
        self.agent_observations = np.empty((self.capacity, n_agents, obs_dim), dtype=np.float32)
        self.next_agent_observations = np.empty_like(self.agent_observations)
        self.comm_adjacency = np.empty((self.capacity, n_agents, n_agents), dtype=bool)
        self.next_comm_adjacency = np.empty_like(self.comm_adjacency)
        self.state = np.empty((self.capacity, state_dim), dtype=np.float32)
        self.next_state = np.empty_like(self.state)
        self.actions = np.empty((self.capacity, n_agents, action_dim), dtype=np.float32)
        self.reward = np.empty(self.capacity, dtype=np.float32)
        self.terminated = np.empty(self.capacity, dtype=np.float32)
        self.active = np.empty((self.capacity, n_agents), dtype=np.float32)
        self.next_active = np.empty_like(self.active)
        self.graph = _allocate_graph(self.capacity, row["hetero_graph"])
        self.next_graph = _allocate_graph(self.capacity, row["next_hetero_graph"])
        self._ready = True

    def add(self, row):
        self._ensure(row)
        index = self.cursor
        self.agent_observations[index] = row["agent_observations"]
        self.next_agent_observations[index] = row["next_agent_observations"]
        self.comm_adjacency[index] = row["comm_adjacency"]
        self.next_comm_adjacency[index] = row["next_comm_adjacency"]
        self.state[index] = row["state"]
        self.next_state[index] = row["next_state"]
        self.actions[index] = row["actions"]
        self.reward[index] = row["reward"]
        self.terminated[index] = float(row["terminated"])
        self.active[index] = np.asarray(row["active"], dtype=np.float32)
        self.next_active[index] = np.asarray(row["next_active"], dtype=np.float32)
        _write_graph(self.graph, index, row["hetero_graph"])
        _write_graph(self.next_graph, index, row["next_hetero_graph"])
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_dataset(self, rows):
        for row in rows:
            self.add(row)

    def sample_indices(self, batch_size):
        if self.size < 1:
            raise ValueError("Cannot sample from an empty replay buffer")
        return self.rng.integers(0, self.size, size=batch_size)

    def gather(self, indices, device):
        return pack_batch(
            observations=self.agent_observations[indices],
            next_observations=self.next_agent_observations[indices],
            adjacency=self.comm_adjacency[indices],
            next_adjacency=self.next_comm_adjacency[indices],
            graph=_read_graph(self.graph, indices),
            next_graph=_read_graph(self.next_graph, indices),
            state=self.state[indices],
            next_state=self.next_state[indices],
            actions=self.actions[indices],
            reward=self.reward[indices],
            terminated=self.terminated[indices],
            active=self.active[indices],
            next_active=self.next_active[indices],
            device=device,
        )

    def sample(self, batch_size, device):
        return self.gather(self.sample_indices(batch_size), device)


def pack_batch(observations, next_observations, adjacency, next_adjacency, graph, next_graph,
               state, next_state, actions, reward, terminated, active, next_active, device):
    return {
        "obs": torch.as_tensor(observations, dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(next_observations, dtype=torch.float32, device=device),
        "adjacency": torch.as_tensor(adjacency, dtype=torch.bool, device=device),
        "next_adjacency": torch.as_tensor(next_adjacency, dtype=torch.bool, device=device),
        "graph": tensorize_heterogeneous_graph(graph, device),
        "next_graph": tensorize_heterogeneous_graph(next_graph, device),
        "state": torch.as_tensor(state, dtype=torch.float32, device=device),
        "next_state": torch.as_tensor(next_state, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.float32, device=device),
        "reward": torch.as_tensor(reward, dtype=torch.float32, device=device),
        "terminated": torch.as_tensor(terminated, dtype=torch.float32, device=device),
        "active": torch.as_tensor(active, dtype=torch.float32, device=device),
        "next_active": torch.as_tensor(next_active, dtype=torch.float32, device=device),
    }


def mix_batches(prior, online, batch_size, prior_fraction, device):
    """Symmetric RLPD-style sampling from two joint buffers."""
    if not 0.0 <= prior_fraction <= 1.0:
        raise ValueError("prior_fraction must be in [0, 1]")
    if prior_fraction == 0:
        return online.sample(batch_size, device)
    if prior_fraction == 1:
        return prior.sample(batch_size, device)
    n_prior = int(round(batch_size * prior_fraction))
    n_prior = min(max(n_prior, 1), batch_size - 1) if 0 < prior_fraction < 1 else n_prior
    n_online = batch_size - n_prior
    prior_batch = prior.gather(prior.sample_indices(n_prior), device)
    online_batch = online.gather(online.sample_indices(n_online), device)
    merged = {}
    for key in prior_batch:
        if key in ("graph", "next_graph"):
            merged[key] = {
                name: torch.cat((prior_batch[key][name], online_batch[key][name]), dim=0)
                for name in prior_batch[key]
            }
        else:
            merged[key] = torch.cat((prior_batch[key], online_batch[key]), dim=0)
    return merged
