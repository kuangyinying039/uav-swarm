"""Evader Prediction Network (EPN) for MAPPO partial-observability relief.

Adapted from thu-uav Multi-UAV-pursuit-evasion ``TP_net``: an LSTM predicts a
short future relative trajectory of the evader from per-UAV track history.
Privileged future positions are used only as an MSE training target; inference
and policy observations never read simulator ground truth.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    nn = None

# Keep in sync with pursuit_graph_encoder.EPN_FUTURE_STEPS / TARGET_DIM layout.
EPN_HISTORY_STEPS = 10
EPN_FUTURE_STEPS = 5
EPN_MISSING_FILL = -1.0


def epn_input_dim(n_uavs: int) -> int:
    """progress + track xyz/vel + peer relative xyz slots (incl. self zero)."""
    return 1 + 3 + 3 + 3 * int(n_uavs)


def epn_prediction_dim(future_steps: int = EPN_FUTURE_STEPS) -> int:
    return 3 * int(future_steps)


@dataclass
class EPNConfig:
    history_steps: int = EPN_HISTORY_STEPS
    future_steps: int = EPN_FUTURE_STEPS
    hidden_dim: int = 64
    lr: float = 1e-4
    epochs: int = 1
    missing_fill: float = EPN_MISSING_FILL
    n_uavs: int = 3


class EvaderPredictionNet(nn.Module if nn else object):
    """LSTM → FC → tanh; output is flattened future relative xyz in [-1, 1]."""

    def __init__(self, input_dim: int, future_steps: int = EPN_FUTURE_STEPS, hidden_dim: int = 64):
        if torch is None:
            raise RuntimeError("PyTorch is required for EvaderPredictionNet")
        super().__init__()
        self.input_dim = int(input_dim)
        self.future_steps = int(future_steps)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = epn_prediction_dim(self.future_steps)
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        # x: [B, T, D]
        h0 = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        c0 = torch.zeros(1, x.size(0), self.hidden_dim, device=x.device, dtype=x.dtype)
        out, _ = self.lstm(x, (h0, c0))
        return torch.tanh(self.fc(out[:, -1, :]))

    def predict_relative(self, x):
        """Return ``[B, future, 3]`` normalized relative positions."""
        flat = self.forward(x)
        return flat.reshape(x.size(0), self.future_steps, 3)


def build_epn_frame(env, agent: int, *, missing_fill: float = EPN_MISSING_FILL) -> np.ndarray:
    """One causal EPN frame for ``agent`` from track / peer estimates only."""
    cfg = env.cfg
    n_uavs = int(cfg.n_uavs)
    frame = np.full(epn_input_dim(n_uavs), float(missing_fill), dtype=np.float32)
    progress = max(0.0, 1.0 - float(getattr(env, "t", 0)) / max(int(cfg.search_steps), 1))
    frame[0] = progress
    if hasattr(env, "quadrotor_states") and env.quadrotor_states.shape[0] > agent:
        position = np.asarray(env.quadrotor_states[agent, :3], dtype=float)
    else:
        position = np.array(
            [*np.asarray(env.positions[agent], dtype=float), float(env.altitudes[agent])],
            dtype=float,
        )
    z_scale = max(float(cfg.max_altitude), 1e-6)
    grid = max(float(cfg.grid_size), 1e-6)
    target_speed = max(float(cfg.target_speed), float(cfg.target_vertical_speed), 1e-6)

    track = None
    if hasattr(env, "track_memory"):
        track = env.track_memory[agent][0]
    if track is not None and track.initialized and len(track.mean) >= 6:
        relative = np.asarray(track.mean[:3], dtype=float) - position
        frame[1:3] = relative[:2] / grid
        frame[3] = relative[2] / z_scale
        frame[4:7] = np.asarray(track.mean[3:6], dtype=float) / target_speed

    disabled = np.asarray(getattr(env, "disabled_uavs", np.zeros(n_uavs)), dtype=bool)
    if hasattr(env, "last_graph"):
        peer_mask = np.asarray(env.last_graph[agent] > 0, dtype=bool) & ~disabled
    else:
        peer_mask = np.zeros(n_uavs, dtype=bool)
    peer_mask[agent] = False
    for peer in np.flatnonzero(peer_mask):
        if hasattr(env, "quadrotor_states"):
            peer_xyz = np.asarray(env.quadrotor_states[peer, :3], dtype=float)
        else:
            peer_xyz = np.array(
                [*np.asarray(env.positions[peer], dtype=float), float(env.altitudes[peer])],
                dtype=float,
            )
        delta = peer_xyz - position
        slot = 7 + 3 * int(peer)
        frame[slot : slot + 2] = delta[:2] / grid
        frame[slot + 2] = delta[2] / z_scale
    # Self slot stays at missing_fill / zeros intentionally (aligned with THU padding).
    return frame


def privileged_evader_xyz(env) -> np.ndarray:
    """Training-only ground-truth evader position (never for policy inputs)."""
    if hasattr(env, "dynamic_targets") and hasattr(env, "target_altitudes"):
        return np.array(
            [*np.asarray(env.dynamic_targets[0], dtype=float)[:2], float(env.target_altitudes[0])],
            dtype=np.float64,
        )
    raise AttributeError("environment has no privileged evader pose")


def agent_xyz(env, agent: int) -> np.ndarray:
    if hasattr(env, "quadrotor_states") and env.quadrotor_states.shape[0] > agent:
        return np.asarray(env.quadrotor_states[agent, :3], dtype=np.float64).copy()
    return np.array(
        [*np.asarray(env.positions[agent], dtype=float), float(env.altitudes[agent])],
        dtype=np.float64,
    )


def normalize_relative_xyz(relative: np.ndarray, cfg) -> np.ndarray:
    grid = max(float(cfg.grid_size), 1e-6)
    z_scale = max(float(cfg.max_altitude), 1e-6)
    out = np.asarray(relative, dtype=np.float64).reshape(-1, 3).copy()
    out[:, :2] /= grid
    out[:, 2] /= z_scale
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def future_relative_labels(
    agent_positions: np.ndarray,
    evader_positions: np.ndarray,
    cfg,
    *,
    future_steps: int = EPN_FUTURE_STEPS,
) -> np.ndarray:
    """Build ``[T, n_agents, future, 3]`` tanh-scale labels from privileged xyz.

    ``agent_positions``: ``[T, n_agents, 3]``, ``evader_positions``: ``[T, 3]``.
    Near episode end, missing futures repeat the last known evader pose.
    """
    agent_positions = np.asarray(agent_positions, dtype=np.float64)
    evader_positions = np.asarray(evader_positions, dtype=np.float64)
    t_steps, n_agents, _ = agent_positions.shape
    labels = np.zeros((t_steps, n_agents, future_steps, 3), dtype=np.float32)
    for t in range(t_steps):
        for k in range(future_steps):
            src = min(t + k + 1, t_steps - 1)
            relative = evader_positions[src][None, :] - agent_positions[t]
            labels[t, :, k, :] = normalize_relative_xyz(relative, cfg)
    return labels


class EPNRuntime:
    """Per-logic-env history + shared network weights for observe-time injection."""

    def __init__(self, epn: EvaderPredictionNet, cfg: EPNConfig, device: str = "cpu"):
        self.epn = epn
        self.cfg = cfg
        self.device = device
        self.histories = [
            deque(maxlen=int(cfg.history_steps)) for _ in range(int(cfg.n_uavs))
        ]
        self.last_input = None  # [n_uavs, hist, dim]
        self.last_prediction = np.zeros(
            (int(cfg.n_uavs), int(cfg.future_steps), 3), dtype=np.float32
        )

    def reset(self) -> None:
        for hist in self.histories:
            hist.clear()
        self.last_input = None
        self.last_prediction.fill(0.0)

    def _padded_history(self, agent: int) -> np.ndarray:
        hist = list(self.histories[agent])
        dim = epn_input_dim(self.cfg.n_uavs)
        if not hist:
            return np.full((self.cfg.history_steps, dim), self.cfg.missing_fill, dtype=np.float32)
        stacked = np.stack(hist, axis=0)
        if stacked.shape[0] >= self.cfg.history_steps:
            return stacked[-self.cfg.history_steps :]
        pad = np.full(
            (self.cfg.history_steps - stacked.shape[0], dim),
            self.cfg.missing_fill,
            dtype=np.float32,
        )
        return np.concatenate([pad, stacked], axis=0)

    @torch.no_grad()
    def update(self, env) -> np.ndarray:
        """Append current frames, run EPN, return ``[n_uavs, future, 3]``."""
        if int(getattr(env, "t", 0) or 0) == 0:
            for hist in self.histories:
                hist.clear()
        for agent in range(self.cfg.n_uavs):
            self.histories[agent].append(
                build_epn_frame(env, agent, missing_fill=self.cfg.missing_fill)
            )
        batch = np.stack([self._padded_history(a) for a in range(self.cfg.n_uavs)], axis=0)
        self.last_input = batch.astype(np.float32)
        tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
        was_training = self.epn.training
        self.epn.eval()
        pred = self.epn.predict_relative(tensor).detach().cpu().numpy().astype(np.float32)
        if was_training:
            self.epn.train()
        self.last_prediction = pred
        env.epn_predicted_rel = pred
        return pred


def attach_epn_runtime(env, runtime: EPNRuntime) -> None:
    """Share EPN weights with a logic env and wrap ``observe_search`` once."""
    env.epn_runtime = runtime
    env.epn_predicted_rel = runtime.last_prediction
    if getattr(env, "_epn_observe_wrapped", False):
        return
    original = env.observe_search

    def observe_with_epn():
        runtime.update(env)
        return original()

    env.observe_search = observe_with_epn
    env._epn_observe_wrapped = True


def attach_epn_to_vector_env(vec_env, runtime: EPNRuntime) -> None:
    """Attach the same runtime to every Isaac bridge / clone logic env."""
    if hasattr(vec_env, "bridges"):
        for bridge in vec_env.bridges:
            attach_epn_runtime(bridge.env, runtime)
        return
    if hasattr(vec_env, "envs"):
        for env in vec_env.envs:
            attach_epn_runtime(env, runtime)
        return
    attach_epn_runtime(vec_env, runtime)


def make_epn(cfg: EPNConfig, device: str = "cpu") -> EvaderPredictionNet:
    net = EvaderPredictionNet(
        input_dim=epn_input_dim(cfg.n_uavs),
        future_steps=cfg.future_steps,
        hidden_dim=cfg.hidden_dim,
    )
    return net.to(device)


def update_epn_from_rollout(
    epn: EvaderPredictionNet,
    optimizer,
    epn_inputs: Sequence[np.ndarray],
    agent_positions: Sequence[np.ndarray],
    evader_positions: Sequence[np.ndarray],
    cfg_env,
    epn_cfg: EPNConfig,
    *,
    device: str = "cpu",
) -> float:
    """Supervised MSE on tanh-scale future relative positions. Returns scalar loss."""
    if epn_inputs is None or len(epn_inputs) == 0:
        return 0.0
    inputs = np.asarray(epn_inputs, dtype=np.float32)
    if inputs.ndim == 3:
        # Single-step [n_agents, hist, dim] → add time axis.
        inputs = inputs[None, ...]
    agents = np.asarray(agent_positions, dtype=np.float64)
    if agents.ndim == 2:
        agents = agents[None, ...]
    evaders = np.asarray(evader_positions, dtype=np.float64)
    if evaders.ndim == 1:
        evaders = evaders[None, ...]
    labels = future_relative_labels(
        agents, evaders, cfg_env, future_steps=epn_cfg.future_steps
    )
    # Flatten agents into batch: [T*n_agents, hist, dim] / [T*n_agents, future*3]
    t_steps, n_agents = inputs.shape[:2]
    x = torch.as_tensor(
        inputs.reshape(t_steps * n_agents, inputs.shape[2], inputs.shape[3]),
        dtype=torch.float32,
        device=device,
    )
    y = torch.as_tensor(
        labels.reshape(t_steps * n_agents, -1),
        dtype=torch.float32,
        device=device,
    )
    epn.train()
    last_loss = 0.0
    for _ in range(max(int(epn_cfg.epochs), 1)):
        pred = epn(x)
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().item())
    return last_loss
