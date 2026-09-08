"""Pursuit-only PPO with rollout likelihood checks and transactional KL guard.

The guard uses analytic KL of the unsquashed Normal on every rollout state.
Tanh is a bijection, so this is also the policy KL before the environment's
safety filter. PPO always scores the *proposed* action, not the filtered one.
"""
from copy import deepcopy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(values, active):
    return (values*active).sum()/active.sum().clamp_min(1.)


def action_log_prob(dist, samples):
    # Match the collection contract exactly; the Jacobian is constant for
    # stored samples and cancels in the PPO ratio.
    return (dist.log_prob(samples)-torch.log((1.-samples.tanh().square()).clamp_min(1e-6))).sum(-1)


def squashed_entropy(dist):
    sample = dist.rsample()
    log_jacobian = 2.*(math.log(2.)-sample-F.softplus(-2.*sample))
    return -(dist.log_prob(sample)-log_jacobian).sum(-1)


def guarded_actor_step(actor, optimizer, assess_kl, target_kl, retries=5):
    """Rollback BOTH weights and Adam moments; retry the same gradient at lower LR."""
    weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    moments = deepcopy(optimizer.state_dict())
    rates = [group['lr'] for group in optimizer.param_groups]
    for attempt in range(retries+1):
        try:
            optimizer.step()
            with torch.no_grad():
                kl = float(assess_kl())
        except Exception:
            # OOM/invalid-distribution failures must not leave a half-applied
            # transaction behind. Surface the error after restoring state.
            actor.load_state_dict(weights)
            optimizer.load_state_dict(moments)
            raise
        if math.isfinite(kl) and kl <= 1.5*target_kl:
            return True, kl, attempt
        actor.load_state_dict(weights)
        optimizer.load_state_dict(moments)
        for group, rate in zip(optimizer.param_groups, rates):
            group['lr'] = rate*(.5**(attempt+1))
    # A failed transaction leaves the policy AND optimizer exactly unchanged.
    optimizer.load_state_dict(moments)
    with torch.no_grad():
        return False, float(assess_kl()), retries+1


def update_pursuit(trainer, rollout, episode):
    if not rollout:
        return {}
    cfg, device = trainer.cfg, trainer.device
    if cfg.target_kl <= 0:
        raise ValueError('Pursuit PPO requires a positive target_kl')
    obs = torch.stack([r[0] for r in rollout]).detach()
    adjacency = torch.stack([r[1] for r in rollout]).detach()
    graph = {k: torch.stack([r[2][k] for r in rollout]).detach() for k in rollout[0][2]} if rollout[0][2] is not None else None
    states = torch.stack([r[3] for r in rollout]).detach()
    samples = torch.stack([torch.stack(r[6]) for r in rollout]).detach()
    stored_logps = torch.stack([r[7] for r in rollout]).detach()
    old_values = torch.stack([r[8] for r in rollout]).detach().reshape(-1)
    rewards = torch.tensor([r[9] for r in rollout], dtype=torch.float32, device=device)
    terminated = torch.tensor([r[11] for r in rollout], dtype=torch.float32, device=device)
    active = (obs[..., 20] < .5).to(obs.dtype)  # v2 pursuit disabled-UAV channel

    def distribution(indices=None):
        vectors = obs if indices is None else obs[indices]
        edges = adjacency if indices is None else adjacency[indices]
        entities = graph if indices is None or graph is None else {k: v[indices] for k, v in graph.items()}
        means, log_std = trainer.actor(vectors, edges, entities).chunk(2, -1)
        return torch.distributions.Normal(means, log_std.exp())

    with torch.no_grad():
        old_dist = distribution()
        parity = (action_log_prob(old_dist, samples)-stored_logps).abs().max().item()
        if not math.isfinite(parity) or parity > 2e-3:
            raise RuntimeError(f'PPO collection/update likelihood mismatch before update: {parity:.6g}')
        advantages = torch.zeros_like(rewards)
        next_value = torch.zeros((), device=device) if terminated[-1] else trainer.critic(rollout[-1][10]).squeeze()
        gae = torch.zeros((), device=device)
        for t in reversed(range(len(rollout))):
            delta = rewards[t]+cfg.gamma*next_value*(1.-terminated[t])-old_values[t]
            gae = delta+cfg.gamma*cfg.gae_lambda*(1.-terminated[t])*gae
            advantages[t] = gae
            next_value = old_values[t]
        returns = advantages+old_values
        if cfg.normalize_advantages and advantages.numel() > 1:
            advantages = (advantages-advantages.mean())/advantages.std(unbiased=False).clamp_min(1e-8)
        return_variance = returns.var(unbiased=False)
        explained = 1.-(returns-old_values).var(unbiased=False)/return_variance.clamp_min(1e-8)

    progress = episode/max(cfg.episodes-1, 1)
    lr = cfg.lr*(1.-progress*(1.-cfg.lr_end_factor))
    entropy_coef = cfg.entropy_coef+progress*(cfg.entropy_coef_end-cfg.entropy_coef)
    for optimizer in (trainer.actor_optim, trainer.critic_optim):
        for group in optimizer.param_groups:
            group['lr'] = lr

    def policy_kl():
        return masked_mean(torch.distributions.kl_divergence(old_dist, distribution()).sum(-1), active)

    metrics = {k: [] for k in ('policy_loss', 'value_loss', 'entropy', 'actor_grad_norm', 'critic_grad_norm', 'clip_fraction')}
    rejected = 0
    accepted = 0
    stop_actor = False
    last_kl = 0.
    for _ in range(max(cfg.update_epochs, 1)):
        for indices in torch.randperm(len(rollout), device=device).split(max(cfg.batch_size, 1)):
            # Critic updates continue after actor early stopping. Huber loss
            # limits terminal-reward outliers without changing value units.
            prediction = trainer.critic(states[indices]).squeeze(-1)
            value_loss = F.smooth_l1_loss(prediction, returns[indices])
            trainer.critic_optim.zero_grad(set_to_none=True)
            (cfg.value_coef*value_loss).backward()
            critic_grad = nn.utils.clip_grad_norm_(trainer.critic.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(critic_grad):
                raise FloatingPointError('Non-finite pursuit critic gradient')
            trainer.critic_optim.step()
            metrics['value_loss'].append(value_loss.item())
            metrics['critic_grad_norm'].append(float(critic_grad))
            if stop_actor or not active[indices].any():
                continue
            dist = distribution(indices)
            log_ratio = action_log_prob(dist, samples[indices])-stored_logps[indices]
            ratio = log_ratio.clamp(-20., 20.).exp()
            surrogate = torch.minimum(ratio*advantages[indices, None],
                                      ratio.clamp(1.-cfg.clip_eps, 1.+cfg.clip_eps)*advantages[indices, None])
            policy_loss = -masked_mean(surrogate, active[indices])
            entropy = masked_mean(squashed_entropy(dist), active[indices])
            loss = policy_loss-entropy_coef*entropy+trainer.pursuit_demo_loss(episode)
            trainer.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(trainer.actor.parameters(), cfg.max_grad_norm)
            if not torch.isfinite(actor_grad):
                raise FloatingPointError('Non-finite pursuit actor gradient')
            success, last_kl, backtracks = guarded_actor_step(trainer.actor, trainer.actor_optim, policy_kl, cfg.target_kl)
            rejected += backtracks
            accepted += int(success)
            stop_actor = not success or last_kl >= cfg.target_kl
            metrics['policy_loss'].append(policy_loss.item())
            metrics['entropy'].append(entropy.item())
            metrics['actor_grad_norm'].append(float(actor_grad))
            metrics['clip_fraction'].append(float(masked_mean(((ratio-1.).abs() > cfg.clip_eps).float(), active[indices])))
    with torch.no_grad():
        final_dist = distribution()
        saturation = masked_mean((final_dist.mean.tanh().abs() > .95).float().mean(-1), active)
        std = final_dist.stddev.mean().item()
    return {**{k: float(np.mean(v)) if v else 0. for k, v in metrics.items()},
            'learning_rate': trainer.actor_optim.param_groups[0]['lr'], 'entropy_coef': entropy_coef,
            'approx_kl': last_kl, 'exact_policy_kl': last_kl, 'pre_update_logprob_error': parity,
            'ppo_early_stop': float(stop_actor), 'ppo_backtracks': rejected, 'ppo_accepted_steps': accepted,
            'action_mean_saturation': float(saturation), 'policy_std': std, 'explained_variance': float(explained)}
