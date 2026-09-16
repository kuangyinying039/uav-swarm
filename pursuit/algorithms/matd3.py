"""MATD3 update: twin team critics, delayed actor, target policy smoothing."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class Matd3Config:
    gamma: float = 0.995
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    policy_delay: int = 2
    target_noise: float = 0.1
    noise_clip: float = 0.2
    exploration_std: float = 0.12
    exploration_final_std: float = 0.12
    exploration_decay_steps: int = 150_000
    batch_size: int = 256
    replay_size: int = 500_000
    warmup_steps: int = 8_000
    warmup_policy: str = "random"
    hidden_dim: int = 128
    critic_hidden: int = 256
    critic_layers: int = 3
    gat_heads: int = 4
    gat_layers: int = 2
    use_gat: bool = True
    max_grad_norm: float = 1.0
    utd: int = 1
    prior_fraction: float = 0.0
    demo_bc_weight: float = 0.0
    demo_bc_final_weight: float = 0.0
    demo_bc_decay_steps: int = 100_000
    critic_pretrain_updates: int = 0
    device: str = "auto"
    episodes: int = 3000
    checkpoint_interval: int = 50
    validation_interval: int = 50
    step_checkpoint_interval: int = 10_000


def flatten_actions(actions):
    return actions.reshape(actions.shape[0], -1)


def soft_update(target, source, tau):
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(source_param.data, alpha=tau)


def update_matd3(trainer, batch, update_actor):
    cfg = trainer.cfg
    with torch.no_grad():
        next_means = trainer.actor_target(batch["next_obs"], batch["next_adjacency"], batch["next_graph"])
        noise = torch.randn_like(next_means) * cfg.target_noise
        noise = noise.clamp(-cfg.noise_clip, cfg.noise_clip)
        next_actions = (next_means.tanh() + noise).clamp(-1.0, 1.0)
        next_actions = next_actions * batch["next_active"].unsqueeze(-1)
        next_q1, next_q2 = trainer.critic_target(batch["next_state"], flatten_actions(next_actions))
        target_q = batch["reward"] + cfg.gamma * (1.0 - batch["terminated"]) * torch.min(next_q1, next_q2)

    executed_actions = batch["actions"] * batch["active"].unsqueeze(-1)
    q1, q2 = trainer.critic(batch["state"], flatten_actions(executed_actions))
    critic_loss = ((q1 - target_q).square() + (q2 - target_q).square()).mean()
    trainer.critic_optim.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = nn.utils.clip_grad_norm_(trainer.critic.parameters(), cfg.max_grad_norm)
    trainer.critic_optim.step()

    stats = {
        "critic_loss": float(critic_loss.detach()),
        "q1_mean": float(q1.detach().mean()),
        "q2_mean": float(q2.detach().mean()),
        "target_q_mean": float(target_q.detach().mean()),
        "td_abs_mean": float(
            0.5 * ((q1.detach() - target_q).abs().mean() + (q2.detach() - target_q).abs().mean())
        ),
        "q_disagreement": float((q1.detach() - q2.detach()).abs().mean()),
        "actor_loss": None,
        "actor_rl_loss": None,
        "demo_bc_loss": None,
        "demo_bc_weight": 0.0,
        "prior_batch_fraction": float(batch.get("is_prior", torch.zeros_like(batch["reward"], dtype=torch.bool)).float().mean()),
        "actor_grad_norm": None,
        "critic_grad_norm": float(critic_grad),
        "updated_actor": False,
    }
    if not update_actor:
        return stats

    proposed = trainer.actor(batch["obs"], batch["adjacency"], batch["graph"]).tanh()
    proposed = proposed * batch["active"].unsqueeze(-1)
    actor_rl_loss = -trainer.critic.q1_only(batch["state"], flatten_actions(proposed)).mean()
    actor_loss = actor_rl_loss
    is_prior = batch.get("is_prior")
    demo_bc_loss = None
    demo_bc_weight = 0.0
    if cfg.demo_bc_weight > 0.0 and is_prior is not None and bool(is_prior.any()):
        prior_proposed = proposed[is_prior]
        prior_actions = executed_actions[is_prior]
        prior_active = batch["active"][is_prior].unsqueeze(-1)
        denominator = prior_active.sum().clamp_min(1.0) * proposed.shape[-1]
        demo_bc_loss = (((prior_proposed - prior_actions).square()) * prior_active).sum() / denominator
        progress = min(float(trainer.env_steps) / max(float(cfg.demo_bc_decay_steps), 1.0), 1.0)
        demo_bc_weight = cfg.demo_bc_weight + progress * (
            cfg.demo_bc_final_weight - cfg.demo_bc_weight
        )
        actor_loss = actor_loss + demo_bc_weight * demo_bc_loss
    trainer.actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_grad = nn.utils.clip_grad_norm_(trainer.actor.parameters(), cfg.max_grad_norm)
    trainer.actor_optim.step()
    soft_update(trainer.actor_target, trainer.actor, cfg.tau)
    soft_update(trainer.critic_target, trainer.critic, cfg.tau)
    stats["actor_loss"] = float(actor_loss.detach())
    stats["actor_rl_loss"] = float(actor_rl_loss.detach())
    stats["demo_bc_loss"] = None if demo_bc_loss is None else float(demo_bc_loss.detach())
    stats["demo_bc_weight"] = float(demo_bc_weight)
    stats["actor_grad_norm"] = float(actor_grad)
    stats["updated_actor"] = True
    return stats
