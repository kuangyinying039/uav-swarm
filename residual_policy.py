"""Expert velocity + RL residual for safer first flights / SITL.

``final = clip(expert + residual_gain * rl_action)`` in normalized action space.
"""

from __future__ import annotations

import numpy as np


class ResidualPolicy:
    def __init__(self, expert_fn, rl_fn, *, residual_gain: float = 0.3):
        if not 0.0 <= float(residual_gain) <= 1.0:
            raise ValueError("residual_gain must be in [0, 1]")
        self.expert_fn = expert_fn
        self.rl_fn = rl_fn
        self.residual_gain = float(residual_gain)

    def __call__(self, obs, env_index: int = 0) -> np.ndarray:
        expert = np.asarray(self.expert_fn(obs, env_index=env_index), dtype=np.float32)
        if self.residual_gain <= 0.0 or self.rl_fn is None:
            return np.clip(expert, -1.0, 1.0)
        residual = np.asarray(self.rl_fn(obs, env_index=env_index), dtype=np.float32)
        if expert.shape != residual.shape:
            raise ValueError(f"expert {expert.shape} vs residual {residual.shape}")
        return np.clip(expert + self.residual_gain * residual, -1.0, 1.0)
