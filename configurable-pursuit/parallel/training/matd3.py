"""MATD3 on the same PEG parallel plant as MAPPO.

Three pursuers share one deterministic actor. Twin critics score the joint
state and the joint executed action. The evader, capture, reward, safety
shield, evader prediction network and adaptive curriculum are the MAPPO ones.
Behavior cloning and the frozen actor anchor update on their own backward,
so the value loss cannot shrink them.

Run from environments/configurable-pursuit after source ./activate.sh:

    python -m parallel.training.matd3 --num-envs 128 --episodes 5000 --output runtime/matd3-peg-5000
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from torch import nn

from parallel.env.radar5_contract import (
    BODY_CONTACT,
    COLLISION_PENALTY,
    GAMMA,
    MAX_YAW_RATE,
    PURSUER_HORIZONTAL,
    PURSUER_VERTICAL,
    TIME_PENALTY,
    physics_substeps,
)
from parallel.training.mappo import (
    _apply_level,
    _build_app,
    _evaluate_controller,
    imitation_phase,
    reference_scale,
    _evaluate_batches,
    SuccessSteps,
    _stack_rows,
    still_training,
    bc_stalled,
    command_to_unit,
    hold_decision,
    joint_state,
    project_unit,
    pursuer_observation,
    scene_observation,
    unit_to_command,
)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        from parallel.training.hetero import HeteroEncoder
        self.encoder = HeteroEncoder(obs_dim)
        self.head = nn.Linear(128, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return project_unit(torch.tanh(self.head(self.encoder(obs))))


class TwinCritic(nn.Module):
    def __init__(self, state_dim: int, joint_action_dim: int):
        super().__init__()
        in_dim = int(state_dim) + int(joint_action_dim)
        self.q1 = self._mlp(in_dim)
        self.q2 = self._mlp(in_dim)

    @staticmethod
    def _mlp(in_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state: torch.Tensor, joint_action: torch.Tensor):
        features = torch.cat((state, joint_action), dim=-1)
        return self.q1(features).squeeze(-1), self.q2(features).squeeze(-1)

    def q1_only(self, state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat((state, joint_action), dim=-1)).squeeze(-1)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for destination, origin in zip(target.parameters(), source.parameters()):
            destination.mul_(1.0 - tau)
            destination.add_(origin, alpha=tau)


class Replay:
    def __init__(self, capacity: int, n_pursuers: int, obs_dim: int, state_dim: int, action_dim: int, device):
        self.capacity = int(capacity)
        self.device = device
        self.n_pursuers = n_pursuers
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        shape = {
            "obs": (n_pursuers, obs_dim),
            "next_obs": (n_pursuers, obs_dim),
            "state": (state_dim,),
            "next_state": (state_dim,),
            "action": (n_pursuers, action_dim),
            "teacher": (n_pursuers, action_dim),
            "active": (n_pursuers,),
            "next_active": (n_pursuers,),
            "reward": (),
            "done": (),
        }
        self.data = {
            name: torch.zeros((self.capacity, *dims), device=device)
            for name, dims in shape.items()
        }
        self.pos = 0
        self.size = 0
        self.demo = None
        self.demo_size = 0

    def freeze_demo(self) -> None:
        """复制当前教师演示。在线缓存继续滚动，模仿采样仍保留这一份。"""
        if self.demo is not None or self.size < 8:
            return
        self.demo_size = self.size
        self.demo = {name: values[: self.size].detach().clone() for name, values in self.data.items()}

    def add(self, batch: dict) -> None:
        count = int(batch["reward"].shape[0])
        if count > self.capacity:
            raise ValueError("replay insert is larger than the buffer")
        first = min(count, self.capacity - self.pos)
        self._write(self.pos, batch, 0, first)
        rest = count - first
        if rest:
            self._write(0, batch, first, count)
        self.pos = (self.pos + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def _write(self, destination: int, batch: dict, start: int, stop: int) -> None:
        width = stop - start
        for name, values in batch.items():
            self.data[name][destination:destination + width] = values[start:stop]

    def sample(self, batch_size: int) -> dict:
        index = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {name: values[index] for name, values in self.data.items()}

    def sample_imitation(self, batch_size: int) -> dict:
        if self.demo is None:
            return self.sample(batch_size)
        demo_count = max(batch_size // 2, 1)
        online = self.sample(batch_size - demo_count)
        demo_index = torch.randint(0, self.demo_size, (demo_count,), device=self.device)
        return {
            name: torch.cat((self.demo[name][demo_index], online[name]), 0)
            for name in self.data
        }


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    weight = active.unsqueeze(-1)
    denominator = weight.sum().clamp_min(1.0) * prediction.shape[-1]
    return ((prediction - target).square() * weight).sum() / denominator


def update_matd3(
    actor,
    actor_target,
    critic,
    critic_target,
    actor_optim,
    critic_optim,
    anchor_optim,
    batch,
    *,
    anchor,
    update_actor: bool,
    gamma: float,
    tau: float,
    target_noise: float,
    noise_clip: float,
    bc_weight: float,
    anchor_weight: float,
    td3_bc_alpha: float,
) -> dict:
    active = batch["active"]
    next_active = batch["next_active"]
    executed = batch["action"] * active.unsqueeze(-1)
    with torch.no_grad():
        next_action = actor_target(batch["next_obs"])
        noise = (torch.randn_like(next_action) * target_noise).clamp(-noise_clip, noise_clip)
        next_action = project_unit(next_action + noise) * next_active.unsqueeze(-1)
        next_q1, next_q2 = critic_target(batch["next_state"], next_action.flatten(1))
        target_q = batch["reward"] + gamma * (1.0 - batch["done"]) * torch.minimum(next_q1, next_q2)
    q1, q2 = critic(batch["state"], executed.flatten(1))
    critic_loss = (q1 - target_q).square().mean() + (q2 - target_q).square().mean()
    critic_optim.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
    critic_optim.step()
    stats = {
        "critic_loss": float(critic_loss.detach()),
        "actor_loss": None,
        "bc_mse": None,
        "anchor_mse": None,
    }
    if not update_actor:
        soft_update(critic_target, critic, tau)
        return stats

    proposed = actor(batch["obs"]) * active.unsqueeze(-1)
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    q_pi = critic.q1_only(batch["state"], proposed.flatten(1))
    scale = td3_bc_alpha / q_pi.abs().mean().detach().clamp_min(1e-6)
    actor_loss = -(scale * q_pi).mean()
    anchor_mse = None
    if anchor is not None and anchor_weight > 0.0:
        with torch.no_grad():
            frozen = anchor(batch["obs"])
        anchor_mse = masked_mse(proposed, frozen, active)
        actor_loss = actor_loss + anchor_weight * anchor_mse
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    actor_optim.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    actor_optim.step()

    bc_mse = masked_mse(actor(batch["obs"]), batch["teacher"], active)
    if bc_weight > 0.0:
        anchor_optim.zero_grad(set_to_none=True)
        (bc_weight * bc_mse).backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        anchor_optim.step()
    soft_update(actor_target, actor, tau)
    soft_update(critic_target, critic, tau)
    stats["actor_loss"] = float(actor_loss.detach())
    stats["bc_mse"] = float(bc_mse.detach())
    stats["anchor_mse"] = None if anchor_mse is None else float(anchor_mse.detach())
    return stats


def _weights(update: int, warmup: int, span: int) -> float:
    progress = min(max(update - warmup, 0) / max(span, 1), 1.0)
    return max(1.0 * (1.0 - progress) + 0.35 * progress, 0.35)


def train(args):
    from parallel.examples.radar5_guidance import PursuerGuidance, evader_command, shield_pursuer_command
    from parallel.training.aeg import AEGConfig, DiscreteAEG, LEVEL_SPEED, LayoutCurriculum, reset_with_curriculum
    from parallel.training.hetero import graph_observation
    from parallel.training.epn import FUTURE_STEPS, BatchEPN, radar_track

    app, env = _build_app(
        args.num_envs, args.seed, args.output, args.scene, args.perception,
        resume=bool(getattr(args, "resume", False)),
    )
    device = env.device
    try:
        state = env.state()
        n_pursuers = state.shape[1] - 1
        if n_pursuers != 3:
            raise RuntimeError(f"Radar5 expects 3 pursuers, found {n_pursuers}")
        static_count = int(env.rules.static.shape[1])
        dynamic_count = int(env.rules.bg.shape[1])
        if args.scene == "simple":
            evader_level = "medium"
            horizontal, vertical = LEVEL_SPEED[evader_level]
            env.evader_horizontal = torch.full((env.n,), horizontal, device=device)
            env.evader_vertical = torch.full((env.n,), vertical, device=device)
        else:
            if (static_count, dynamic_count) != (16, 3):
                raise RuntimeError(f"Expected the PEG scene, found {static_count} static and {dynamic_count} dynamic")
            evader_level = "aeg"
            aeg = DiscreteAEG(AEGConfig(cold_start_level="medium"))
            aeg.set_seed(args.seed)
            level_of = [aeg.cfg.cold_start_level] * env.n
            env.evader_horizontal = torch.empty(env.n, device=device)
            env.evader_vertical = torch.empty(env.n, device=device)
            for index, level in enumerate(level_of):
                _apply_level(env, index, level)
        teacher = None if args.teacher == "none" else PursuerGuidance(args.teacher, env.n, device)
        forecaster = BatchEPN(env.n, n_pursuers, device)
        max_steps = int(env.rules.settings["task"]["max_episode_steps"])
        episode_ids = torch.zeros(env.n, dtype=torch.long, device=device)
        next_seed = int(args.seed) * 100_000
        episode_seed = list(range(args.seed, args.seed + env.n))
        layout = LayoutCurriculum()
        layout.set_seed(args.seed)
        obs_dim = graph_observation(state, env, torch.zeros(env.n, n_pursuers, 15, device=device)).shape[-1]
        state_dim = n_pursuers * obs_dim
        action_dim = 4
        actor = Actor(obs_dim, action_dim).to(device)
        actor_target = copy.deepcopy(actor).eval()
        for parameter in actor_target.parameters():
            parameter.requires_grad_(False)
        critic = TwinCritic(state_dim, n_pursuers * action_dim).to(device)
        critic_target = copy.deepcopy(critic).eval()
        for parameter in critic_target.parameters():
            parameter.requires_grad_(False)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
        anchor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        replay = Replay(args.replay_size, n_pursuers, obs_dim, state_dim, action_dim, device)
        success_replay = Replay(524_288, n_pursuers, obs_dim, state_dim, action_dim, device)
        val_replay = Replay(65_536, n_pursuers, obs_dim, state_dim, action_dim, device)
        successes = SuccessSteps(env.n)
        anchor = None
        teacher_capture = 0.0
        rollbacks = 0
        rollback_streak = 0
        best_capture = -1.0
        best_snap = None
        matched = False
        accepted_actor = None
        teacher_report = None
        reverts = 0
        history = []
        gradient_steps = 0

        def emit(message: str) -> None:
            try:
                print(message, flush=True)
            except BrokenPipeError:
                pass
            with (args.output / "train.log").open("a") as handle:
                handle.write(message + "\n")

        def features(current, progress, forecast):
            return graph_observation(current, env, forecast)

        def observe(current, progress):
            return features(current, progress, forecaster.observe(current, progress, *radar_track(env)))

        def pursue_actor(_state, progress):
            return shield_pursuer_command(env, unit_to_command(actor(observe(_state, progress))))

        def pursue_teacher(_state, _progress):
            return teacher.command(env)

        def gate() -> dict:
            nonlocal matched, accepted_actor, teacher_report, anchor
            saved_h = env.evader_horizontal.clone()
            saved_v = env.evader_vertical.clone()
            if evader_level == "aeg":
                horizontal, vertical = LEVEL_SPEED["medium"]
                env.evader_horizontal.fill_(horizontal)
                env.evader_vertical.fill_(vertical)
            seeds = list(range(8_000_000, 8_000_000 + int(env.n)))
            actor_report = _evaluate_controller(
                env, forecaster, observe, evader_command, max_steps, pursue_actor, seeds
            )
            held_teacher = _evaluate_controller(
                env, forecaster, observe, evader_command, max_steps, pursue_teacher, seeds
            )
            teacher.previous.zero_()
            env.evader_horizontal.copy_(saved_h)
            env.evader_vertical.copy_(saved_v)
            if actor_report["capture_rate"] + 1e-6 >= held_teacher["capture_rate"] - 0.02:
                matched = True
            teacher_report = held_teacher
            emit(
                f"[matd3] gate actor={actor_report['capture_rate']:.3f} "
                f"collisions={actor_report['static_collisions']}+{actor_report['dynamic_collisions']} "
                f"teacher={held_teacher['capture_rate']:.3f} "
                f"teacher_collisions={held_teacher['static_collisions']}+{held_teacher['dynamic_collisions']} "
                f"matched={int(matched)}"
            )
            return actor_report

        completed = 0
        update = 0
        dagger_origin = None
        env_steps = 0
        rl_at_steps = None
        reference_actor = None
        low_streak = 0
        best_monitor = -1.0
        critic_warmup = 0
        if bool(getattr(args, "resume", False)):
            ckpt_path = args.output / "matd3.pt"
            if not ckpt_path.is_file():
                raise FileNotFoundError(f"--resume needs {ckpt_path}")
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            actor.load_state_dict(payload["actor"])
            actor_target.load_state_dict(payload.get("actor_target") or payload["actor"])
            critic.load_state_dict(payload["critic"])
            critic_target.load_state_dict(payload.get("critic_target") or payload["critic"])
            if payload.get("epn") is not None:
                forecaster.net.load_state_dict(payload["epn"])
            history = list(payload.get("history") or [])
            hist_path = args.output / "history.json"
            if hist_path.is_file():
                disk_history = json.loads(hist_path.read_text())
                if len(disk_history) >= len(history):
                    history = list(disk_history)
            if history:
                update = int(history[-1].get("update") or -1) + 1
                completed = int(history[-1].get("episodes") or 0)
                stored = history[-1].get("env_steps")
                env_steps = int(stored) if stored is not None else update * args.rollout_steps * args.num_envs
                critic_warmup = 4
            emit(f"[matd3] resume from {ckpt_path} steps={env_steps} update={update} episodes={completed}")
        rl_at = None
        extend_note = False
        bc_trace: list[float] = []
        stalled = False
        last_bc = 1.0e9
        rl_ready = False
        init_capture = None
        def fresh_critic() -> None:
            nonlocal critic, critic_target, critic_optim
            critic = TwinCritic(state_dim, n_pursuers * action_dim).to(device)
            critic_target = copy.deepcopy(critic).eval()
            for parameter in critic_target.parameters():
                parameter.requires_grad_(False)
            critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

        def snapshot() -> dict:
            return {
                "actor": {key: value.detach().cpu().clone() for key, value in actor.state_dict().items()},
                "actor_target": {key: value.detach().cpu().clone() for key, value in actor_target.state_dict().items()},
                "critic": {key: value.detach().cpu().clone() for key, value in critic.state_dict().items()},
                "critic_target": {key: value.detach().cpu().clone() for key, value in critic_target.state_dict().items()},
            }

        def restore(snap: dict) -> None:
            actor.load_state_dict(snap["actor"])
            actor_target.load_state_dict(snap["actor_target"])
            critic.load_state_dict(snap["critic"])
            critic_target.load_state_dict(snap["critic_target"])
            actor_optim.state.clear()
            critic_optim.state.clear()
            anchor_optim.state.clear()

        def scored(seed_start: int, episodes: int) -> dict:
            saved_h = env.evader_horizontal.clone()
            saved_v = env.evader_vertical.clone()
            if evader_level == "aeg":
                horizontal, vertical = LEVEL_SPEED["medium"]
                env.evader_horizontal.fill_(horizontal)
                env.evader_vertical.fill_(vertical)
            report = _evaluate_batches(
                env, forecaster, observe, evader_command, max_steps, pursue_actor, seed_start, episodes,
            )
            if teacher is not None:
                teacher.previous.zero_()
            env.evader_horizontal.copy_(saved_h)
            env.evader_vertical.copy_(saved_v)
            return report

        while still_training(env_steps, args.imitation_max_steps, args.rl_steps, teacher is not None, rl_at_steps):
            sequence_buf, pursuer_buf, evader_buf, episode_buf = [], [], [], []
            captures = 0
            finished_steps = 0
            reward_sum = 0.0
            reward_count = 0
            if teacher is None:
                beta, phase = 0.0, "rl"
            else:
                beta, started, phase = imitation_phase(
                    update, args.bc_warmup, args.bc_max, args.dagger_decay, dagger_origin, stalled, last_bc <= args.dagger_fit,
                )
                if started is not None and dagger_origin is None:
                    success_replay.freeze_demo()
                    dagger_origin = started
                    emit(
                        f"[matd3] dagger steps={env_steps} bc={last_bc:.4f} "
                        f"rows={successes.kept} failure_rows={successes.failure_rows}"
                    )
            if phase == "rl" and teacher is not None and not rl_ready:
                if best_snap is not None:
                    actor.load_state_dict(best_snap["actor"])
                actor_target.load_state_dict(actor.state_dict())
                fresh_critic()
                replay.pos = 0
                replay.size = 0
                anchor = copy.deepcopy(actor).eval()
                for parameter in anchor.parameters():
                    parameter.requires_grad_(False)
                actor_optim.state.clear()
                anchor_optim.state.clear()
                rl_at_steps = env_steps
                rl_ready = True
                critic_warmup = 0
                opening = scored(9_000_000, args.confirm_episodes)
                init_capture = opening["capture_rate"]
                best_capture = init_capture
                best_snap = snapshot()
                emit(f"[matd3] rl start steps={env_steps} init_capture={init_capture:.3f}")
            if teacher is not None and rl_at_steps is None and env_steps >= args.imitation_max_steps and not extend_note:
                extend_note = True
                emit(f"[matd3] imitation step ceiling steps={env_steps} bc={last_bc:.4f}")
            imitate = phase == "bc"
            anchor_weight = reference_scale(env_steps, rl_at_steps, args.rl_steps) if phase == "rl" and anchor is not None else 0.0
            for _ in range(args.rollout_steps):
                current = env.state()
                progress = 1.0 - env.rules.steps.float() / max_steps
                obs = observe(current, progress)
                with torch.no_grad():
                    unit = actor(obs)
                    if phase == "rl":
                        unit = project_unit(unit + torch.randn_like(unit) * args.exploration_std)
                    env_steps += int(env.n)
                    preferred = executed = None
                    if teacher is not None:
                        preferred, executed = teacher.propose(env)
                    use_teacher = torch.rand(env.n, device=device) < beta if teacher is not None else torch.zeros(env.n, dtype=torch.bool, device=device)
                    learner_cmd = shield_pursuer_command(env, unit_to_command(unit))
                command = torch.zeros(env.n, current.shape[1], action_dim, device=device)
                command[:, 0] = evader_command(env)
                if teacher is None:
                    command[:, 1:] = learner_cmd
                else:
                    command[:, 1:] = torch.where(use_teacher.view(env.n, 1, 1), executed, learner_cmd)
                alive = (~env.disabled[:, 1:]).float()
                team_reward, done, captured, _info, _outcome = hold_decision(env, command)
                after = env.state()
                after_progress = 1.0 - env.rules.steps.float() / max_steps
                next_obs = features(after, after_progress, forecaster.forecast(after, after_progress, *radar_track(env)))
                next_alive = (~env.disabled[:, 1:]).float()
                teacher_unit = command_to_unit(preferred) if preferred is not None else unit
                chosen_intent = torch.where(use_teacher.view(env.n, 1, 1), teacher_unit, unit) if preferred is not None else unit
                replay.add({
                    "obs": obs.detach(),
                    "next_obs": next_obs.detach(),
                    "state": obs.detach().flatten(1),
                    "next_state": next_obs.detach().flatten(1),
                    "action": chosen_intent.detach(),
                    "teacher": teacher_unit.detach(),
                    "active": alive.detach(),
                    "next_active": next_alive.detach(),
                    "reward": team_reward.detach(),
                    "done": done.float().detach(),
                })
                if teacher is not None:
                    for index in range(env.n):
                        successes.push_env(index, {
                            "obs": obs[index].detach(),
                            "next_obs": next_obs[index].detach(),
                            "state": obs[index].detach().flatten(),
                            "next_state": next_obs[index].detach().flatten(),
                            "action": chosen_intent[index].detach(),
                            "teacher": teacher_unit[index].detach(),
                            "active": alive[index].detach(),
                            "next_active": next_alive[index].detach(),
                            "reward": team_reward[index].detach(),
                            "done": done[index].float().detach(),
                        })
                sequence_buf.append(forecaster.history.detach().clone())
                pursuer_buf.append(current[:, 1:, :3].detach())
                evader_buf.append(current[:, 0, :3].detach())
                episode_buf.append(episode_ids.clone())
                captures += int(captured.sum())
                reward_sum += float(team_reward.sum())
                reward_count += int(team_reward.numel())
                finished = done.nonzero(as_tuple=False).flatten()
                finished_steps += int(finished.numel())
                if finished.numel():
                    if teacher is not None:
                        successes.commit(
                            finished,
                            captured,
                            lambda index, rows: (val_replay if index % 8 == 0 else success_replay).add(_stack_rows(rows, tuple(rows[0]))),
                        )
                    completed += int(finished.numel())
                    episode_ids[finished] += 1
                    forecaster.reset(finished)
                    if teacher is not None:
                        teacher.previous[finished] = 0
                    if evader_level == "aeg":
                        for index in finished.tolist():
                            aeg.observe(level_of[index], bool(captured[index].item()))
                            level_of[index] = aeg.next_level()
                            _apply_level(env, index, level_of[index])
                    next_seed = reset_with_curriculum(
                        env, finished, episode_seed, next_seed, captured, _outcome, layout
                    )
            critic_loss = None
            actor_loss = None
            bc_weight = 0.0
            bc_mse = None
            anchor_mse = None
            anchor_weight = 0.0 if phase != "rl" or anchor is None or critic_warmup < 4 else reference_scale(env_steps, rl_at_steps, args.rl_steps)
            if phase == "rl" and replay.size >= args.batch_size:
                actor_updates = 0
                for step in range(args.updates_per_rollout):
                    batch = replay.sample(args.batch_size)
                    do_actor = critic_warmup >= 4 and ((gradient_steps + 1) % args.policy_delay == 0)
                    stats = update_matd3(
                        actor, actor_target, critic, critic_target,
                        actor_optim, critic_optim, anchor_optim, batch,
                        anchor=anchor if phase == "rl" else None,
                        update_actor=do_actor,
                        gamma=GAMMA,
                        tau=args.tau,
                        target_noise=args.target_noise,
                        noise_clip=args.noise_clip,
                        bc_weight=0.0,
                        anchor_weight=anchor_weight,
                        td3_bc_alpha=args.td3_bc_alpha,
                    )
                    gradient_steps += 1
                    critic_loss = stats["critic_loss"]
                    if do_actor:
                        actor_updates += 1
                        actor_loss = stats["actor_loss"]
                        bc_mse = stats["bc_mse"]
                        anchor_mse = stats["anchor_mse"]
                critic_warmup += 1
            if teacher is not None and phase != "rl" and success_replay.size >= 8:
                bc_weight = 1.0
                bc_mse = _behavior_clone(
                    actor, anchor_optim, success_replay, min(1024, success_replay.size),
                    epochs=args.bc_steps if phase == "bc" else 16,
                )
                if val_replay.size >= 8:
                    held = val_replay.sample(min(1024, val_replay.size))
                    with torch.no_grad():
                        bc_mse = float(masked_mse(actor(held["obs"]), held["teacher"], held["active"]))
                last_bc = bc_mse
                bc_trace.append(last_bc)
                if phase == "bc" and not stalled and bc_stalled(bc_trace, args.dagger_fit):
                    stalled = True
                    emit(
                        f"[matd3] bc plateau val_mse={last_bc:.4f} best={min(bc_trace):.4f} "
                        f"rows={successes.kept} failure_rows={successes.failure_rows}"
                    )
            epn_mse = forecaster.update(
                torch.stack(sequence_buf),
                torch.stack(pursuer_buf),
                torch.stack(evader_buf),
                torch.stack(episode_buf),
            )
            row = {
                "update": update,
                "episodes": completed,
                "finished": finished_steps,
                "captures": captures,
                "capture_rate": captures / finished_steps if finished_steps else 0.0,
                "mean_reward": reward_sum / reward_count if reward_count else 0.0,
                "epn_mse": epn_mse,
                "evader_level": evader_level,
                "static_obstacles": static_count,
                "dynamic_obstacles": dynamic_count,
                "teacher": args.teacher,
                "phase": phase,
                "dagger_beta": beta,
                "bc_weight": bc_weight,
                "anchor_weight": anchor_weight,
                "bc_mse": bc_mse,
                "anchor_mse": anchor_mse,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "rollbacks": rollbacks,
                "replay": replay.size,
                "aeg": evader_level == "aeg",
            }
            if evader_level == "aeg":
                rates = aeg.stats()["capture_rates"]
                row["aeg_level"] = aeg.current_level
                row["aeg_rates"] = {name: None if value != value else value for name, value in rates.items()}
            history.append(row)
            (args.output / "history.json").write_text(json.dumps(history, indent=2))
            emit(
                f"[matd3] update={update} episodes={completed}/{args.episodes} "
                f"phase={row['phase']} capture_rate={row['capture_rate']:.3f} reward={row['mean_reward']:.3f} "
                f"epn_mse={epn_mse:.4f} "
                f"critic={critic_loss if critic_loss is not None else 0:.3f} "
                f"bc={bc_mse if bc_mse is not None else 0:.4f} successes={successes.kept} "
                f"anchor={anchor_mse if anchor_mse is not None else 0:.4f} "
                f"aeg={row.get('aeg_level', evader_level)} beta={beta:.2f} replay={replay.size} matched={int(matched)}"
            )
            if (update + 1) % args.gate_interval == 0:
                report = scored(8_000_000, env.n)
                emit(
                    f"[matd3] monitor capture={report['capture_rate']:.3f} "
                    f"episodes={report['episodes']} collisions={report['static_collisions']}+{report['dynamic_collisions']}"
                )
                dropped = phase == "rl" and init_capture is not None and report["capture_rate"] + 0.08 < init_capture
                low_streak = low_streak + 1 if dropped else 0
                if low_streak >= 2 and best_snap is not None:
                    check = scored(9_000_000, args.confirm_episodes)
                    if check["capture_rate"] + 0.08 < init_capture:
                        restore(best_snap)
                        rollbacks += 1
                        low_streak = 0
                        emit(
                            f"[matd3] restore capture={check['capture_rate']:.3f} "
                            f"init={init_capture:.3f} best={best_capture:.3f}"
                        )
                    else:
                        low_streak = 0
                elif report["capture_rate"] > best_monitor:
                    best_monitor = report["capture_rate"]
                    check = scored(9_000_000, args.confirm_episodes)
                    if check["capture_rate"] > best_capture:
                        best_capture = check["capture_rate"]
                        best_snap = snapshot()
                        emit(f"[matd3] confirmed capture={best_capture:.3f} phase={phase} steps={env_steps}")
            update += 1
            if update % 10 == 0:
                torch.save(_payload(actor, critic, actor_target, critic_target, forecaster, history, args), args.output / "matd3.pt")
        torch.save(_payload(actor, critic, actor_target, critic_target, forecaster, history, args), args.output / "matd3.pt")
        emit(f"[matd3] wrote {args.output / 'matd3.pt'} episodes={completed}")
        if best_snap is not None:
            restore(best_snap)
        if evader_level == "aeg":
            for index in range(env.n):
                _apply_level(env, index, "medium")
        actor_report = _evaluate_batches(
            env, forecaster, observe, evader_command, max_steps, pursue_actor, 10_000_000, args.final_episodes,
        )
        held_teacher = None
        if teacher is not None:
            held_teacher = _evaluate_batches(
                env, forecaster, observe, evader_command, max_steps, pursue_teacher, 10_000_000, args.final_episodes,
            )
        evaluation = {
            "algorithm": "matd3",
            **actor_report,
            "teacher": args.teacher,
            "perception": args.perception,
            "dagger": True,
            "teacher_eval": held_teacher,
            "control": "learner mean action, no teacher in the action, safety shield on",
            "eval_evader_level": "medium" if evader_level == "aeg" else evader_level,
            "checkpoint": "best capture",
            "best_capture": best_capture,
            "matched": matched,
            "epn": True,
            "aeg": evader_level == "aeg",
        }
        (args.output / "eval.json").write_text(json.dumps(evaluation, indent=2))
        teacher_rate = 0.0 if held_teacher is None else held_teacher["capture_rate"]
        emit(
            f"[matd3] eval capture_rate={evaluation['capture_rate']:.3f} "
            f"captures={evaluation['captures']}/{evaluation['episodes']} "
            f"collisions={evaluation['static_collisions']}+{evaluation['dynamic_collisions']} "
            f"teacher={teacher_rate:.3f}"
        )
    except BaseException:
        import traceback
        (args.output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        env.close()
        app.close()


def _behavior_clone(actor, optim, replay: Replay, batch_size: int, epochs: int) -> float:
    mse = 0.0
    for _epoch in range(epochs):
        batch = replay.sample_imitation(batch_size)
        loss = masked_mse(actor(batch["obs"]), batch["teacher"], batch["active"])
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optim.step()
        mse = float(loss.detach())
    return mse


def _evaluate(env, actor, forecaster, observe, evader_command, shield_pursuer_command, max_steps) -> torch.Tensor:
    device = env.device
    ids = torch.arange(env.n, device=device)
    forecaster.reset(ids)
    env.reset(ids, seeds=list(range(8_000_000, 8_000_000 + int(env.n))))
    decisions = max(int(max_steps) // physics_substeps(float(env.rules.settings["simulation"]["dt"])), 1)
    pending = torch.ones(env.n, dtype=torch.bool, device=device)
    captured = torch.zeros(env.n, dtype=torch.bool, device=device)
    for _ in range(decisions):
        if not bool(pending.any()):
            break
        current = env.state()
        progress = 1.0 - env.rules.steps.float() / max_steps
        with torch.no_grad():
            command = torch.zeros(env.n, current.shape[1], 4, device=device)
            command[:, 0] = evader_command(env)
            command[:, 1:] = shield_pursuer_command(env, unit_to_command(actor(observe(current, progress))))
            command[~pending] = 0
        _reward, done, got, _info, _outcome = hold_decision(env, command)
        captured |= got & pending
        pending &= ~done
    return captured


def _payload(actor, critic, actor_target, critic_target, forecaster, history, args) -> dict:
    return {
        "algorithm": "matd3",
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "epn": forecaster.net.state_dict(),
        "history": history,
        "contract": {
            "action": ["vx", "vy", "vz", "yaw_rate"],
            "pursuer_horizontal": PURSUER_HORIZONTAL,
            "pursuer_vertical": PURSUER_VERTICAL,
            "yaw_rate": MAX_YAW_RATE,
            "gamma": GAMMA,
            "capture_radius": BODY_CONTACT,
            "collision_penalty": COLLISION_PENALTY,
            "time_penalty": TIME_PENALTY,
            "epn": True,
            "aeg": args.scene == "full",
            "teacher": None if args.teacher == "none" else args.teacher,
            "scene": args.scene,
            "exploration_std": args.exploration_std,
            "bc_floor": 0.35,
            "anchor_floor": 0.35,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--episodes", type=int, default=5000, help="保留参数，停止条件改用环境步数")
    parser.add_argument("--imitation-max-steps", type=int, default=600_000)
    parser.add_argument("--rl-steps", type=int, default=5_000_000)
    parser.add_argument("--confirm-episodes", type=int, default=1024)
    parser.add_argument("--final-episodes", type=int, default=4096)
    parser.add_argument("--max-episodes", type=int, default=20000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--updates-per-rollout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200_000)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--target-noise", type=float, default=0.1)
    parser.add_argument("--noise-clip", type=float, default=0.2)
    parser.add_argument("--exploration-std", type=float, default=0.08)
    parser.add_argument("--td3-bc-alpha", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--scene", choices=("full", "simple"), default="full")
    parser.add_argument("--teacher", choices=("none", "apf", "frpn", "orca", "cbf", "mpc"), default="apf")
    parser.add_argument("--bc-warmup", type=int, default=20)
    parser.add_argument("--bc-max", type=int, default=48)
    parser.add_argument("--bc-steps", type=int, default=64)
    parser.add_argument("--gate-interval", type=int, default=8)
    parser.add_argument("--dagger-warmup", type=int, default=8)
    parser.add_argument("--dagger-decay", type=int, default=24)
    parser.add_argument("--dagger-fit", type=float, default=0.35)
    parser.add_argument("--dagger-latest", type=int, default=32, help="保留参数，不再用来强制进入混合")
    parser.add_argument("--perception", choices=("truth", "mid360"), default="mid360")
    parser.add_argument("--resume", action="store_true", help="从输出目录里的 matd3.pt 继续")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 1024:
        parser.error("num-envs must be 1..1024")
    if args.episodes < 1 or args.policy_delay < 1 or args.batch_size < 1:
        parser.error("episodes, policy-delay and batch-size must be positive")
    if args.resume and not (args.output / "matd3.pt").is_file():
        parser.error(f"--resume requires {args.output / 'matd3.pt'}")
    started = time.time()
    train(args)
    print(f"[matd3] elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
