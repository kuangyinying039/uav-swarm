"""Residual / hybrid action composition for beating strong classical teachers.

Motivation
----------
Open-source pursuit stacks that reliably beat pure classical guidance almost
never train end-to-end RL *against* a strong teacher on the same interface.
Common patterns we adopt here:

1. **Hybrid / residual control** (MattxDave/pursuit-evasion-rl; hierarchical
   RL+MPC papers): classical law provides the bulk of the command; RL only
   outputs a correction. This preserves the teacher's Medium capture rate as a
   floor and lets RL spend capacity on failure modes (occlusion, dual-capture
   geometry, Extreme OOD).

2. **Curriculum over fixed Dual-Medium** (thu-uav/Multi-UAV-pursuit-evasion
   Adaptive Environment Generator): train across a difficulty mix so Extreme
   transfer does not rely on a single seed's Midium local optimum.

3. **Evader prediction features** (same THU repo): feed short-horizon predicted
   target positions into the policy (we already have KF velocity; residual
   training should keep that path).

Modes
-----
- ``blend``: a = clip((1-w)·a_base + w·a_rl)   — works with *existing* full-action
  MAPPO/MATD3 checkpoints (no retrain).
- ``residual``: a = clip(a_base + scale·δ)     — for policies trained as deltas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pursuit_baselines_3d import BASELINES_3D


@dataclass
class ResidualComposeConfig:
    base: str = "mpc"
    mode: str = "blend"  # blend | residual
    weight: float = 0.35  # blend: RL weight; residual: delta scale
    # Anneal RL weight so early rollouts keep a classical floor, then RL can
    # dominate and *exceed* the teacher (residual policy learning pattern).
    weight_final: float | None = None
    weight_warmup_steps: int = 0
    weight_decay_steps: int = 100_000
    clip: float = 1.0

    def weight_at(self, env_steps: int) -> float:
        start = float(self.weight)
        end = start if self.weight_final is None else float(self.weight_final)
        if self.weight_final is None or int(self.weight_decay_steps) <= 0:
            return start
        progress_steps = max(int(env_steps) - int(self.weight_warmup_steps), 0)
        progress = min(progress_steps / max(int(self.weight_decay_steps), 1), 1.0)
        return start + progress * (end - start)


def compose_actions(
    base_actions,
    rl_actions,
    cfg: ResidualComposeConfig,
    *,
    weight: float | None = None,
) -> np.ndarray:
    base = np.asarray(base_actions, dtype=np.float32)
    rl = np.asarray(rl_actions, dtype=np.float32)
    if base.shape != rl.shape:
        raise ValueError(f"base/rl action shape mismatch: {base.shape} vs {rl.shape}")
    w = float(cfg.weight if weight is None else weight)
    if cfg.mode == "blend":
        mixed = (1.0 - w) * base + w * rl
    elif cfg.mode == "residual":
        mixed = base + w * rl
    else:
        raise ValueError(f"unknown residual mode: {cfg.mode}")
    return np.clip(mixed, -float(cfg.clip), float(cfg.clip)).astype(np.float32)


def make_base_policy(name: str):
    if name not in BASELINES_3D:
        raise KeyError(f"unknown base policy {name}; expected one of {sorted(BASELINES_3D)}")
    return BASELINES_3D[name]()


def wrap_action_fn(rl_action_fn, env, cfg: ResidualComposeConfig):
    """Return an action_fn(obs) that mixes a classical base with the RL policy."""
    teacher = make_base_policy(cfg.base)

    def action_fn(observation):
        rl = np.asarray(rl_action_fn(observation), dtype=np.float32)
        base = np.asarray(teacher.actions(env), dtype=np.float32)
        return compose_actions(base, rl, cfg)

    action_fn.residual_cfg = cfg  # type: ignore[attr-defined]
    action_fn.base_policy = teacher  # type: ignore[attr-defined]
    return action_fn


def residualize_demo_actions(demo_actions, base_actions, scale: float) -> np.ndarray:
    """Convert full teacher actions into residual labels: δ = (a - a_base) / scale."""
    if scale <= 1e-8:
        raise ValueError("residual scale must be positive")
    demo = np.asarray(demo_actions, dtype=np.float32)
    base = np.asarray(base_actions, dtype=np.float32)
    return np.clip((demo - base) / float(scale), -1.0, 1.0).astype(np.float32)
