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
    # When true, online and expert batches keep capture successes only.
    success_replay_only: bool = False
    demo_bc_weight: float = 0.0
    demo_bc_final_weight: float = 0.0
    demo_bc_decay_steps: int = 100_000
    # TD3+BC (Fujimoto & Gu, 2021): scale RL term by alpha/|Q| and BC toward batch actions.
    td3_bc_alpha: float = 0.0
    bc_on_batch: bool = False
    # B3C-style critic target clipping (AAMAS 2026): clamp bootstrap targets by dataset max return.
    critic_return_clip: float | None = None
    # Offline-to-online actor anchor (ANCHOR / WSRL-inspired): stay near frozen init policy.
    actor_anchor_weight: float = 0.0
    actor_anchor_final_weight: float = 0.0
    actor_anchor_decay_steps: int = 100_000
    critic_pretrain_updates: int = 0
    # After this many capture-rate rollbacks, stop and keep best_capture.pt (P0 early stop).
    max_safeguard_rollbacks: int = 3
    device: str = "auto"
    episodes: int = 3000
    max_steps: int = 0
    checkpoint_interval: int = 50
    validation_interval: int = 50
    step_checkpoint_interval: int = 10_000
    # Sample-efficiency protocol: save frozen snapshots at these completed episode counts.
    episode_budget_checkpoints: tuple = ()


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
        clip_value = cfg.critic_return_clip
        if clip_value is None:
            clip_value = getattr(trainer, "critic_return_clip", None)
        if clip_value is not None:
            target_q = torch.minimum(target_q, torch.as_tensor(float(clip_value), device=target_q.device))

    executed_actions = batch["actions"] * batch["active"].unsqueeze(-1)
    q1, q2 = trainer.critic(batch["state"], flatten_actions(executed_actions))
    critic_loss = ((q1 - target_q).square() + (q2 - target_q).square()).mean()
    trainer.critic_optim.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_grad = nn.utils.clip_grad_norm_(trainer.critic.parameters(), cfg.max_grad_norm)
    trainer.critic_optim.step()
    trainer.critic_optim.zero_grad(set_to_none=True)

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
        "actor_anchor_loss": None,
        "actor_anchor_weight": 0.0,
        "td3_bc_lambda": None,
        "prior_batch_fraction": float(batch.get("is_prior", torch.zeros_like(batch["reward"], dtype=torch.bool)).float().mean()),
        "actor_grad_norm": None,
        "critic_grad_norm": float(critic_grad),
        "updated_actor": False,
    }
    if not update_actor:
        return stats

    proposed = trainer.actor(batch["obs"], batch["adjacency"], batch["graph"]).tanh()
    proposed = proposed * batch["active"].unsqueeze(-1)
    critic_parameters = tuple(trainer.critic.parameters())
    for parameter in critic_parameters:
        parameter.requires_grad_(False)
    try:
        q_pi = trainer.critic.q1_only(batch["state"], flatten_actions(proposed))
        if float(cfg.td3_bc_alpha) > 0.0:
            # Fujimoto & Gu TD3+BC: balance Q maximization against BC via |Q| normalization.
            td3_bc_lambda = float(cfg.td3_bc_alpha) / q_pi.abs().mean().detach().clamp_min(1e-6)
            actor_rl_loss = -(td3_bc_lambda * q_pi).mean()
            stats["td3_bc_lambda"] = float(td3_bc_lambda.detach()) if torch.is_tensor(td3_bc_lambda) else float(td3_bc_lambda)
        else:
            actor_rl_loss = -q_pi.mean()
            td3_bc_lambda = None
    finally:
        for parameter in critic_parameters:
            parameter.requires_grad_(True)
    actor_loss = actor_rl_loss
    is_prior = batch.get("is_prior")
    demo_bc_loss = None
    demo_bc_weight = 0.0
    use_bc = float(cfg.demo_bc_weight) > 0.0 or float(cfg.demo_bc_final_weight) > 0.0
    if use_bc:
        if cfg.bc_on_batch or is_prior is None:
            bc_mask = torch.ones(proposed.shape[0], dtype=torch.bool, device=proposed.device)
        else:
            bc_mask = is_prior
        if bool(bc_mask.any()):
            bc_proposed = proposed[bc_mask]
            bc_actions = executed_actions[bc_mask]
            bc_active = batch["active"][bc_mask].unsqueeze(-1)
            denominator = bc_active.sum().clamp_min(1.0) * proposed.shape[-1]
            demo_bc_loss = (((bc_proposed - bc_actions).square()) * bc_active).sum() / denominator
            progress = min(float(trainer.env_steps) / max(float(cfg.demo_bc_decay_steps), 1.0), 1.0)
            demo_bc_weight = cfg.demo_bc_weight + progress * (
                cfg.demo_bc_final_weight - cfg.demo_bc_weight
            )
            actor_loss = actor_loss + demo_bc_weight * demo_bc_loss

    anchor = getattr(trainer, "actor_anchor", None)
    actor_anchor_loss = None
    actor_anchor_weight = 0.0
    if anchor is not None and (
        float(cfg.actor_anchor_weight) > 0.0 or float(cfg.actor_anchor_final_weight) > 0.0
    ):
        with torch.no_grad():
            anchor_actions = anchor(batch["obs"], batch["adjacency"], batch["graph"]).tanh()
            anchor_actions = anchor_actions * batch["active"].unsqueeze(-1)
        active = batch["active"].unsqueeze(-1)
        denominator = active.sum().clamp_min(1.0) * proposed.shape[-1]
        actor_anchor_loss = (((proposed - anchor_actions).square()) * active).sum() / denominator
        progress = min(float(trainer.env_steps) / max(float(cfg.actor_anchor_decay_steps), 1.0), 1.0)
        actor_anchor_weight = cfg.actor_anchor_weight + progress * (
            cfg.actor_anchor_final_weight - cfg.actor_anchor_weight
        )
        actor_loss = actor_loss + actor_anchor_weight * actor_anchor_loss

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
    stats["actor_anchor_loss"] = None if actor_anchor_loss is None else float(actor_anchor_loss.detach())
    stats["actor_anchor_weight"] = float(actor_anchor_weight)
    stats["actor_grad_norm"] = float(actor_grad)
    stats["updated_actor"] = True
    return stats
