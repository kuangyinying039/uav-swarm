"""HGAT actor + twin team critics trained with MATD3 / prior replay."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json

import numpy as np
import torch
from torch import nn

from marl_trainers import FlatMLP, tensorize_heterogeneous_graph
from pursuit.algorithms.matd3 import Matd3Config, soft_update, update_matd3
from pursuit.data.replay_buffer import JointReplayBuffer, mix_batches
from pursuit.data.transition_dataset import flatten_transitions, make_transition
from pursuit.models.twin_critic import TwinCentralizedQ
from pursuit_graph_encoder import PURSUIT_GRAPH_VERSION, PursuitGraphActor
from pursuit_lidar import accumulate_visibility, finalize_visibility
from pursuit_training_output import clean_history_row, write_reward_capture_chart
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def resolve_device(requested):
    name = str(requested).lower()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


def observation_tensors(obs, device):
    return (
        torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=device),
        torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=device),
        tensorize_heterogeneous_graph(obs["hetero_graph"], device),
    )


def mean_actor_state_from_checkpoint(payload):
    if "actor" in payload and payload.get("algorithm") == "hgat_matd3":
        return payload["actor"]
    raw = payload.get("actors")
    if raw is None:
        raise ValueError("Checkpoint has neither a MATD3 actor nor a MAPPO actors bundle")
    for prefix in ("0.mean_actor.", "mean_actor."):
        stripped = {key[len(prefix):]: value for key, value in raw.items() if key.startswith(prefix)}
        if stripped:
            return stripped
    raise ValueError("Could not find PursuitGraphActor weights (mean_actor) in the checkpoint")


class Matd3Trainer:
    def __init__(self, env_factory, cfg: Matd3Config, seed=0):
        probe = env_factory()
        if not isinstance(probe, QuadrotorPursuitEnv):
            raise ValueError("MATD3 trainer is restricted to quadrotor pursuit")
        if not np.isclose(cfg.gamma, probe.cfg.reward_gamma, atol=1e-10):
            raise ValueError("MATD3 gamma must equal environment reward_gamma")
        if cfg.utd < 1 or cfg.policy_delay < 1:
            raise ValueError("UTD and policy delay must be positive")
        if cfg.warmup_policy not in ("random", "actor"):
            raise ValueError("warmup_policy must be 'random' or 'actor'")
        if cfg.demo_bc_weight < 0 or cfg.demo_bc_final_weight < 0:
            raise ValueError("demo BC weights must be non-negative")
        if min(cfg.exploration_std, cfg.exploration_final_std) < 0:
            raise ValueError("exploration standard deviations must be non-negative")
        if cfg.exploration_decay_steps < 1 or cfg.critic_pretrain_updates < 0:
            raise ValueError("exploration decay steps must be positive and critic pretraining non-negative")
        self.env_factory = env_factory
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        self.n_agents = probe.cfg.n_uavs
        self.obs_dim = probe.obs_dim()
        self.state_dim = probe.state_dim()
        self.action_dim = int(probe.continuous_action_dim())
        self.environment_config = asdict(probe.cfg)
        if cfg.use_gat:
            self.actor = PursuitGraphActor(
                self.obs_dim, self.action_dim, cfg.hidden_dim, cfg.gat_heads, cfg.gat_layers,
                dropout=0.0, spatial_scale=probe.cfg.grid_size,
            ).to(self.device)
            output_layer = self.actor.head
        else:
            self.actor = FlatMLP(self.obs_dim, self.action_dim, cfg.hidden_dim).to(self.device)
            output_layer = self.actor.net[-1]
        nn.init.orthogonal_(output_layer.weight, gain=0.01)
        nn.init.zeros_(output_layer.bias)
        self.actor_target = deepcopy(self.actor).to(self.device).eval()
        for parameter in self.actor_target.parameters():
            parameter.requires_grad_(False)
        joint_dim = self.n_agents * self.action_dim
        self.critic = TwinCentralizedQ(self.state_dim, joint_dim, cfg.critic_hidden, cfg.critic_layers).to(self.device)
        self.critic_target = deepcopy(self.critic).to(self.device).eval()
        for parameter in self.critic_target.parameters():
            parameter.requires_grad_(False)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.online_replay = JointReplayBuffer(cfg.replay_size, seed=seed)
        self.prior_replay = JointReplayBuffer(cfg.replay_size, seed=seed + 1)
        self.env_steps = 0
        self.total_updates = 0
        self.start_episode = 0
        self.history = []
        self.validation_records = []
        self.best_capture_score = (-1.0, -float("inf"), -float("inf"))
        self.best_actor_state = None
        self.output_directory = None
        self.validation_seeds = []
        self.train_seed_bank = []
        self.validation_seed_bank = []
        self.demo_metadata = {}
        self.algorithm_name = "hgat_matd3" if cfg.use_gat else "matd3"

    def load_actor_weights(self, state_dict):
        self.actor.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)

    def load_prior(self, dataset):
        rows = flatten_transitions(dataset)
        if len(rows) > self.prior_replay.capacity:
            self.prior_replay = JointReplayBuffer(len(rows), seed=1)
        self.prior_replay.add_dataset(rows)
        self.demo_metadata["prior_transitions"] = len(rows)
        self.demo_metadata["prior_episodes"] = len(dataset["episodes"])
        self.demo_metadata["prior_successes"] = sum(1 for row in dataset["report"] if row["success"])
        self.demo_metadata["prior_teacher"] = dataset.get("teacher")

    @torch.no_grad()
    def deterministic_action(self, obs):
        vectors, adjacency, graph = observation_tensors(obs, self.device)
        return self.actor(vectors, adjacency, graph).tanh().cpu().numpy()

    @torch.no_grad()
    def explore_action(self, obs, rng):
        action = self.deterministic_action(obs)
        noise = rng.normal(0.0, self.current_exploration_std(), size=action.shape)
        return np.clip(action + noise, -1.0, 1.0).astype(np.float32)

    def current_exploration_std(self):
        """Linearly anneal behavior noise after replay warmup."""
        progress_steps = max(self.env_steps - self.cfg.warmup_steps, 0)
        progress = min(progress_steps / max(self.cfg.exploration_decay_steps, 1), 1.0)
        return float(
            self.cfg.exploration_std
            + progress * (self.cfg.exploration_final_std - self.cfg.exploration_std)
        )

    def pretrain_critic(self, updates=None):
        """Fit the twin critics on expert replay before changing a warm-start actor."""
        requested = self.cfg.critic_pretrain_updates if updates is None else int(updates)
        completed = int(self.demo_metadata.get("critic_pretrain_updates_completed", 0))
        remaining = max(requested - completed, 0)
        if remaining == 0:
            return {}
        if len(self.prior_replay) < 1:
            raise ValueError("critic pretraining requires a non-empty prior replay buffer")
        totals = {}
        last = {}
        for _ in range(remaining):
            batch = self.prior_replay.sample(self.cfg.batch_size, self.device)
            batch["is_prior"] = torch.ones(
                self.cfg.batch_size, dtype=torch.bool, device=self.device
            )
            last = update_matd3(self, batch, update_actor=False)
            self.total_updates += 1
            if self.total_updates % self.cfg.policy_delay == 0:
                soft_update(self.critic_target, self.critic, self.cfg.tau)
            for key in ("critic_loss", "target_q_mean", "td_abs_mean", "q_disagreement"):
                totals[key] = totals.get(key, 0.0) + float(last[key])
        completed += remaining
        self.demo_metadata["critic_pretrain_updates_completed"] = completed
        summary = {
            "updates": remaining,
            "completed_updates": completed,
            **{f"mean_{key}": value / remaining for key, value in totals.items()},
            **{f"final_{key}": float(last[key]) for key in totals},
        }
        return summary

    def save_checkpoint(self, path, episode=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "algorithm": self.algorithm_name,
            "train_config": asdict(self.cfg),
            "env_config": self.environment_config,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optim.state_dict(),
            "critic_optimizer": self.critic_optim.state_dict(),
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "n_agents": self.n_agents,
            "action_dim": self.action_dim,
            "episode": int(self.start_episode if episode is None else episode),
            "env_steps": self.env_steps,
            "total_updates": self.total_updates,
            "history": [clean_history_row(row) for row in self.history],
            "validation_records": self.validation_records,
            "best_capture_score": self.best_capture_score,
            "best_actor_state": self.best_actor_state,
            "pursuit_graph_version": PURSUIT_GRAPH_VERSION if self.cfg.use_gat else None,
            "train_seed_bank": list(self.train_seed_bank),
            "validation_seed_bank": list(self.validation_seed_bank),
            "demo_metadata": self.demo_metadata,
        }
        torch.save(payload, path)

    def load_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("algorithm") != self.algorithm_name:
            raise ValueError(f"Expected {self.algorithm_name}, got {payload.get('algorithm')!r}")
        if self.cfg.use_gat and payload.get("pursuit_graph_version") != PURSUIT_GRAPH_VERSION:
            raise ValueError("Checkpoint actor graph version does not match PursuitGraphActor")
        for name, expected in (
            ("obs_dim", self.obs_dim),
            ("state_dim", self.state_dim),
            ("n_agents", self.n_agents),
            ("action_dim", self.action_dim),
        ):
            if int(payload[name]) != expected:
                raise ValueError(f"Checkpoint {name} does not match the current environment")
        self.actor.load_state_dict(payload["actor"])
        self.actor_target.load_state_dict(payload["actor_target"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        self.actor_optim.load_state_dict(payload["actor_optimizer"])
        self.critic_optim.load_state_dict(payload["critic_optimizer"])
        self.start_episode = int(payload.get("episode", 0))
        self.env_steps = int(payload.get("env_steps", 0))
        self.total_updates = int(payload.get("total_updates", 0))
        self.history = list(payload.get("history", []))
        self.validation_records = list(payload.get("validation_records", []))
        self.best_capture_score = tuple(payload.get("best_capture_score", self.best_capture_score))
        self.best_actor_state = payload.get("best_actor_state")
        self.demo_metadata = dict(payload.get("demo_metadata", {}))
        return payload

    def _sample_batch(self):
        if self.cfg.prior_fraction > 0 and len(self.prior_replay) < 1:
            raise ValueError("prior_fraction > 0 but the prior buffer is empty")
        if len(self.online_replay) < 1:
            raise ValueError("online replay is empty")
        return mix_batches(
            self.prior_replay, self.online_replay, self.cfg.batch_size, self.cfg.prior_fraction, self.device
        )

    def _maybe_update(self):
        if self.env_steps < self.cfg.warmup_steps:
            return {}
        if len(self.online_replay) < self.cfg.batch_size:
            return {}
        last = {}
        for local in range(self.cfg.utd):
            update_actor = ((self.total_updates + 1) % self.cfg.policy_delay) == 0
            last = update_matd3(self, self._sample_batch(), update_actor)
            self.total_updates += 1
        return last

    def train(self, rng):
        from pursuit.eval import evaluate_policy

        print(
            f"[HGAT-MATD3] device={self.device} episodes={self.start_episode + 1}..{self.cfg.episodes} "
            f"prior_fraction={self.cfg.prior_fraction} utd={self.cfg.utd}",
            flush=True,
        )
        for episode in range(self.start_episode, self.cfg.episodes):
            env = self.env_factory(episode)
            obs = env.observe_search()
            total_reward = 0.0
            collisions = 0
            correction = []
            visibility = []
            components = {}
            closest = float(env.minimum_capture_gap())
            last_stats = {}
            visibility_totals = {}
            for step in range(env.cfg.search_steps):
                active = ~env.disabled_uavs
                if self.env_steps < self.cfg.warmup_steps and self.cfg.warmup_policy == "random":
                    action = rng.uniform(-1.0, 1.0, size=(self.n_agents, self.action_dim)).astype(np.float32)
                else:
                    action = self.explore_action(obs, rng)
                result = env.step_joint(action)
                row = make_transition(obs, action, result, {}, active, ~env.disabled_uavs)
                self.online_replay.add(row)
                self.env_steps += 1
                last_stats = self._maybe_update() or last_stats
                if (
                    self.output_directory
                    and self.cfg.step_checkpoint_interval > 0
                    and self.env_steps % self.cfg.step_checkpoint_interval == 0
                ):
                    self.save_checkpoint(
                        self.output_directory / "checkpoints" / f"step_{self.env_steps:09d}.pt",
                        episode=episode,
                    )
                total_reward += float(result["reward"])
                collisions += int(result.get("collisions", 0))
                correction.append(float(result.get("safety_correction_rate", 0.0)))
                visibility.append(float(result.get("target_visibility_rate", 0.0)))
                closest = min(closest, float(result.get("minimum_capture_gap", closest)))
                accumulate_visibility(visibility_totals, result)
                for key, value in result["reward_components"].items():
                    if key in ("pursuit", "estimation"):
                        continue
                    components[key] = components.get(key, 0.0) + float(value)
                obs = result["obs"]
                if result["terminated"] or result["truncated"]:
                    break
            captured = bool(result.get("capture_success", False))
            history_row = {
                "episode": episode + 1,
                "reward": total_reward,
                "capture_success": float(captured),
                "capture_time": int(result.get("capture_time", step + 1)) if captured else None,
                "steps": step + 1,
                "collisions": collisions,
                "minimum_capture_gap": env.minimum_capture_gap(),
                "closest_capture_gap": closest,
                "safety_correction_rate": float(np.mean(correction)) if correction else 0.0,
                "target_visibility_rate": float(np.mean(visibility)) if visibility else 0.0,
                "reward_components": components,
                **finalize_visibility(visibility_totals, step + 1),
                "controller_feasible_rate": float(result.get("controller_feasible_rate", 1.0)),
                "critic_loss": last_stats.get("critic_loss"),
                "actor_loss": last_stats.get("actor_loss"),
                "actor_rl_loss": last_stats.get("actor_rl_loss"),
                "demo_bc_loss": last_stats.get("demo_bc_loss"),
                "demo_bc_weight": last_stats.get("demo_bc_weight"),
                "q1_mean": last_stats.get("q1_mean"),
                "q2_mean": last_stats.get("q2_mean"),
                "target_q_mean": last_stats.get("target_q_mean"),
                "td_abs_mean": last_stats.get("td_abs_mean"),
                "q_disagreement": last_stats.get("q_disagreement"),
                "prior_batch_fraction": last_stats.get("prior_batch_fraction"),
                "replay_size": len(self.online_replay),
                "prior_size": len(self.prior_replay),
                "env_steps": self.env_steps,
                "matd3_updates": self.total_updates,
                "exploration_std": self.current_exploration_std(),
                "utd": self.cfg.utd,
                "actor_grad_norm": last_stats.get("actor_grad_norm"),
                "critic_grad_norm": last_stats.get("critic_grad_norm"),
            }
            self.history.append(clean_history_row(history_row))
            if self.output_directory and (episode + 1 == 1 or (episode + 1) % 10 == 0):
                print(
                    f"[matd3] episode={episode + 1} capture={captured} steps={step + 1} "
                    f"return={total_reward:.2f} replay={len(self.online_replay)} "
                    f"updates={self.total_updates} critic={last_stats.get('critic_loss')}",
                    flush=True,
                )
                write_reward_capture_chart(self.output_directory / "reward_capture.svg", self.history)
            if self.output_directory and (episode + 1) % self.cfg.checkpoint_interval == 0:
                self.save_checkpoint(self.output_directory / "latest.pt", episode=episode + 1)
            if self.output_directory and self.validation_seeds and (episode + 1) % self.cfg.validation_interval == 0:
                self.validate(episode + 1, evaluate_policy)
        if self.output_directory:
            self.save_checkpoint(self.output_directory / "final.pt", episode=self.cfg.episodes)
        return self.history

    def validate(self, episode, evaluate_policy):
        self.actor.eval()
        result = evaluate_policy(
            lambda obs, env: self.deterministic_action(obs),
            QuadrotorPursuitConfig(**self.environment_config),
            self.validation_seeds,
            method_name="matd3",
        )
        self.actor.train()
        metrics = result["summary"]["matd3"]
        score = (metrics["capture_rate"], -metrics["mean_censored_steps"], metrics["mean_return"])
        self.validation_records.append({"episode": episode, "env_steps": self.env_steps, **metrics})
        if score > self.best_capture_score:
            self.best_capture_score = score
            self.best_actor_state = {key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()}
            self.save_checkpoint(self.output_directory / "best_capture.pt", episode=episode)
        (self.output_directory / "validation.json").write_text(
            json.dumps(self.validation_records, indent=2), encoding="utf-8"
        )
        print(
            f"[validation] episode={episode} capture_rate={metrics['capture_rate']:.3f} "
            f"return={metrics['mean_return']:.2f}",
            flush=True,
        )
