"""逃逸机短时预测网络。

历史里的目标位置和速度来自机载航迹。没有航迹时这一帧的目标通道记为缺失。
未来真值只用于 MSE 监督，不进入策略观测。输出是未来若干步的归一化相对位置。
"""

from __future__ import annotations

import torch
from torch import nn

HISTORY_STEPS = 10
FUTURE_STEPS = 5
MISSING = -1.0
XY_SCALE = 21.0
Z_SCALE = 12.0


def radar_track(env):
    """有雷达时返回机载目标航迹。没有雷达时让预测网络退回仿真真值。"""
    if getattr(env, "mid360", None) is None:
        return None, None, None
    from parallel.env.mid360 import sensed_target

    return sensed_target(env)


def input_dim(n_pursuers: int = 3) -> int:
    return 1 + 3 + 3 + 3 * int(n_pursuers)


class EvaderPredictionNet(nn.Module):
    def __init__(self, n_pursuers: int = 3, future_steps: int = FUTURE_STEPS, hidden_dim: int = 64):
        super().__init__()
        self.n_pursuers = int(n_pursuers)
        self.future_steps = int(future_steps)
        self.hidden_dim = int(hidden_dim)
        self.lstm = nn.LSTM(input_dim(n_pursuers), self.hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 3 * self.future_steps)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        batch = sequence.shape[0]
        hidden = torch.zeros(1, batch, self.hidden_dim, device=sequence.device, dtype=sequence.dtype)
        cell = torch.zeros_like(hidden)
        output, _ = self.lstm(sequence, (hidden, cell))
        return torch.tanh(self.fc(output[:, -1, :]))

    def predict(self, sequence: torch.Tensor) -> torch.Tensor:
        flat = self.forward(sequence)
        return flat.reshape(sequence.shape[0], self.future_steps, 3)


class BatchEPN:
    """并行环境上的历史缓存。history 形状 [N, P, T, D]。"""

    def __init__(self, envs: int, n_pursuers: int, device: torch.device):
        self.envs = int(envs)
        self.n_pursuers = int(n_pursuers)
        self.device = device
        self.net = EvaderPredictionNet(n_pursuers).to(device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        dim = input_dim(n_pursuers)
        self.history = torch.full((envs, n_pursuers, HISTORY_STEPS, dim), MISSING, device=device)
        self.dim = dim

    def reset(self, ids: torch.Tensor) -> None:
        if ids.numel():
            self.history[ids] = MISSING

    def _frame(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        target: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pursuers = state[:, 1:, :3]
        if target is None:
            target = state[:, :1, :3].expand(-1, self.n_pursuers, -1)
            target_velocity = state[:, :1, 7:10].expand(-1, self.n_pursuers, -1)
            valid = torch.ones(state.shape[0], self.n_pursuers, dtype=torch.bool, device=state.device)
        relative = target - pursuers
        velocity = target_velocity - state[:, 1:, 7:10]
        scale = relative.new_tensor([XY_SCALE, XY_SCALE, Z_SCALE])
        frame = relative.new_zeros(state.shape[0], self.n_pursuers, self.dim)
        frame[..., 0] = torch.where(valid, progress[:, None], torch.full_like(progress[:, None], MISSING))
        frame[..., 1:4] = torch.where(valid.unsqueeze(-1), (relative / scale).clamp(-1, 1), torch.zeros_like(relative))
        frame[..., 4:7] = torch.where(valid.unsqueeze(-1), (velocity / scale).clamp(-1, 1), torch.zeros_like(velocity))
        peer = (pursuers[:, None, :, :] - pursuers[:, :, None, :]) / scale
        eye = torch.eye(self.n_pursuers, device=state.device, dtype=torch.bool)
        peer = peer.masked_fill(eye.view(1, self.n_pursuers, self.n_pursuers, 1), 0)
        frame[..., 7:] = peer.clamp(-1, 1).reshape(state.shape[0], self.n_pursuers, -1)
        return frame

    @torch.no_grad()
    def forecast(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        target: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """下一步会看到的预测。不写入历史，供离策略算法取 next_obs。"""
        frame = self._frame(state, progress, target, target_velocity, valid)
        history = torch.roll(self.history, shifts=-1, dims=2).clone()
        history[:, :, -1] = frame
        flat = history.reshape(self.envs * self.n_pursuers, HISTORY_STEPS, self.dim)
        self.net.eval()
        prediction = self.net.predict(flat)
        return prediction.reshape(self.envs, self.n_pursuers, FUTURE_STEPS * 3)

    @torch.no_grad()
    def observe(
        self,
        state: torch.Tensor,
        progress: torch.Tensor,
        target: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        frame = self._frame(state, progress, target, target_velocity, valid)
        self.history = torch.roll(self.history, shifts=-1, dims=2)
        self.history[:, :, -1] = frame
        flat = self.history.reshape(self.envs * self.n_pursuers, HISTORY_STEPS, self.dim)
        self.net.eval()
        prediction = self.net.predict(flat)
        return prediction.reshape(self.envs, self.n_pursuers, FUTURE_STEPS * 3)

    def update(
        self,
        sequences: torch.Tensor,
        pursuer_xyz: torch.Tensor,
        evader_xyz: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> float:
        """sequences [T,N,P,H,D]，位置 [T,N,P,3] / [T,N,3]，episode_ids [T,N]。"""
        steps = sequences.shape[0]
        if steps <= FUTURE_STEPS:
            return 0.0
        usable = steps - FUTURE_STEPS
        same = torch.ones(usable, sequences.shape[1], dtype=torch.bool, device=sequences.device)
        for shift in range(1, FUTURE_STEPS + 1):
            same &= episode_ids[:usable] == episode_ids[shift:shift + usable]
        if not bool(same.any()):
            return 0.0
        chosen_t, chosen_n = same.nonzero(as_tuple=True)
        sequence = sequences[chosen_t, chosen_n].reshape(-1, HISTORY_STEPS, self.dim)
        origin = pursuer_xyz[chosen_t, chosen_n]
        future = []
        scale = origin.new_tensor([XY_SCALE, XY_SCALE, Z_SCALE])
        for shift in range(1, FUTURE_STEPS + 1):
            relative = evader_xyz[chosen_t + shift, chosen_n].unsqueeze(1) - origin
            future.append((relative / scale).clamp(-1, 1))
        target = torch.cat(future, dim=-1)
        self.net.train()
        prediction = self.net(sequence)
        loss = nn.functional.mse_loss(prediction, target.reshape(prediction.shape[0], -1))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())
