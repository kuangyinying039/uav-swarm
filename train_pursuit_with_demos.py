"""Frozen on-policy MAPPO pursuit baseline (BC / retention / DAgger reproduction).

Do not extend this file with off-policy MATD3. New pursuit RL lives in
``train_pursuit_matd3.py`` and the ``pursuit/`` package. Successful teacher
trajectories here remain BC-only; they are never PPO data. The cooperative
search environment and discrete search trainers are left unchanged.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marl_trainers import (
    MAPPOTrainer, TrainConfig, FlatMLP, seed_everything,
    tensorize_heterogeneous_graph, capture_env_frame, episode_task_metrics,
)
from pursuit_graph_encoder import (
    PURSUIT_GRAPH_VERSION, PursuitGraphActor, pursuit_graph_from_flat_observation,
)
from pursuit_baselines_3d import BASELINES_3D
from pursuit_env_mix import load_pursuit_config, make_pursuit_env_factory, mix_manifest
from pursuit_lidar import accumulate_visibility, finalize_visibility
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_training_output import clean_history_row, write_pursuit_outputs, write_evaluation_chart, write_reward_capture_chart
from pursuit_ppo import update_pursuit
from artifact_paths import artifact_path, default_output


SEED_BANK_VERSION = 1


def load_seed_bank(path):
    """Load a newline/whitespace-delimited seed bank with strict validation."""
    if path is None:
        return []
    path = Path(path)
    try:
        tokens = path.read_text(encoding="utf-8-sig").split()
        seeds = [int(token) for token in tokens]
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read integer seeds from {path}: {error}") from error
    if not seeds:
        raise ValueError(f"Seed bank is empty: {path}")
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"Seed bank contains duplicate seeds: {path}")
    return seeds


def seed_for_episode(bank, episode, base_seed):
    """Visit every bank entry once per cycle, reshuffling deterministically."""
    if not bank:
        return int(base_seed) * 100000 + int(episode)
    cycle, offset = divmod(int(episode), len(bank))
    rng = np.random.default_rng(int(base_seed) + 100003 * cycle)
    return int(np.asarray(bank, dtype=np.int64)[rng.permutation(len(bank))[offset]])


def ensure_disjoint_seed_banks(named_banks):
    """Reject leakage between any nonempty train/validation/evaluation banks."""
    names = list(named_banks)
    for index, left_name in enumerate(names):
        left = set(named_banks[left_name])
        for right_name in names[index + 1:]:
            overlap = left.intersection(named_banks[right_name])
            if overlap:
                sample = sorted(overlap)[:5]
                raise ValueError(
                    f"{left_name} and {right_name} seed banks overlap "
                    f"({len(overlap)} seeds; first: {sample})"
                )


def resolve_auxiliary_path(aux_demos, no_aux_demos):
    """Resolve correction replay only from an explicit command-line request."""
    if aux_demos is not None and no_aux_demos:
        raise ValueError("--aux-demos and --no-aux-demos cannot be used together")
    return None if no_aux_demos or aux_demos is None else Path(aux_demos)


class PursuitActor(nn.Module):
    """Shared means with trainable, smoothly bounded per-action exploration."""

    def __init__(self, mean_actor, action_dim, upper=-0.5, initial_std=0.4):
        super().__init__()
        self.mean_actor = mean_actor
        self.lower, self.upper = math.log(0.05), upper
        initial = math.log(initial_std)
        if not self.lower < initial < upper:
            raise ValueError("Initial log standard deviation must lie inside its bounds")
        fraction = (initial - self.lower) / (upper - self.lower)
        self.std_parameter = nn.Parameter(torch.full((action_dim,), math.log(fraction / (1 - fraction))))

    def forward(self, obs, adjacency=None, hetero_graph=None):
        means = self.mean_actor(obs, adjacency, hetero_graph)
        log_std = self.lower + (self.upper - self.lower) * torch.sigmoid(self.std_parameter)
        return torch.cat((means, log_std.expand_as(means)), dim=-1)

    @torch.no_grad()
    def set_std(self, std):
        """Set exploration without depending on a checkpoint's old parameterization."""
        log_std = math.log(float(std))
        if not self.lower < log_std < self.upper:
            raise ValueError("Policy standard deviation must lie strictly inside its bounds")
        fraction = (log_std - self.lower) / (self.upper - self.lower)
        self.std_parameter.fill_(math.log(fraction / (1.0 - fraction)))


class PursuitDemoTrainer(MAPPOTrainer):
    def __init__(self, env_factory, cfg):
        probe = env_factory()
        required_env_members = (
            "cfg", "observe_search", "step_joint", "minimum_capture_gap",
            "obs_dim", "state_dim", "continuous_action_dim",
        )
        missing_env_members = [name for name in required_env_members if not hasattr(probe, name)]
        if missing_env_members:
            raise TypeError(
                "Pursuit trainer environment is missing required members: "
                f"{missing_env_members}"
            )
        if cfg.use_assignment_head or cfg.use_search_weights:
            raise ValueError("Pursuit demonstration training requires direct motion actions")
        if not math.isclose(cfg.gamma, probe.cfg.reward_gamma, abs_tol=1e-10):
            raise ValueError("PPO gamma must equal environment reward_gamma for potential shaping")
        if cfg.normalize_returns:
            raise ValueError("Per-rollout return standardization changes critic units; use unnormalized returns")
        # PPO must recompute the same policy likelihood before any update.
        # Fresh dropout masks invalidate that ratio; exploration comes from Normal.
        cfg = replace(cfg, dropout=0.0)
        super().__init__(env_factory, cfg)
        mean_actor = (
            PursuitGraphActor(self.obs_dim, self.action_dim, cfg.hidden_dim, cfg.gat_heads,
                              cfg.gat_layers, cfg.dropout, probe.cfg.grid_size)
            if cfg.use_gat else FlatMLP(self.obs_dim, self.action_dim, cfg.hidden_dim)
        )
        # Official MAPPO uses a small (0.01) orthogonal action-head gain.
        # A default random head on LayerNorm features created a large initial
        # common-direction bias in the failed run.
        output_head = mean_actor.head if cfg.use_gat else mean_actor.net[-1]
        nn.init.orthogonal_(output_head.weight, gain=0.01)
        nn.init.zeros_(output_head.bias)
        self.actor = PursuitActor(
            mean_actor, self.action_dim, cfg.continuous_log_std_max, cfg.continuous_initial_std
        ).to(self.device)
        self.actors = nn.ModuleList([self.actor])
        self.actor_base_lr = cfg.actor_lr if cfg.actor_lr is not None else cfg.lr
        self.critic_base_lr = cfg.critic_lr if cfg.critic_lr is not None else cfg.lr
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_base_lr, eps=1e-5)
        self.critic = nn.Sequential(nn.LayerNorm(self.state_dim), self.critic).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.critic_base_lr, eps=1e-5)
        self.algorithm_name = "mappo_pursuit_3d_graph_v5"
        self.environment_config = asdict(probe.cfg)
        self.demo_metadata = {}
        self.demo_episodes = []
        self.correction_episodes = []
        self.demo_rng = np.random.default_rng(0)
        self.demo_weight = 0.1
        self.demo_weight_floor = 0.02
        self.correction_fraction = 0.5
        self.reference_actor = None
        self.reference_kl_weight = 0.0
        self.validation_seeds = []
        self.validation_interval = 50
        self.validation_records = []
        self.pursuit_history = []
        self.best_capture_score = (-1.0, -float("inf"), -float("inf"))
        self.best_actor_state = None
        self.safeguard_drop = 0.10
        self.safeguard_patience = 2
        self.safeguard_bad_validations = 0
        self.safeguard_rollbacks = 0
        self.max_safeguard_rollbacks = 3
        self.early_stop = False
        self.early_stop_reason = None
        self.min_actor_lr = 1e-5
        self.output_directory = None
        self.train_seed_bank = []
        self.validation_seed_bank = []
        self.seed_bank_version = SEED_BANK_VERSION
        self.residual_cfg = None
        self._residual_teacher = None
        self._residual_env_steps = 0

    def set_residual(self, cfg):
        """Compose policy deltas/actions with a classical base before env.step.

        For on-policy MAPPO prefer ``mode='residual'`` so the buffer stores δ
        while the environment receives ``base + w·δ``. Blend mode is mainly for
        eval-time hybrids; if used in training it still keeps PPO log-probs on
        the pre-compose policy sample (approximate).
        """
        from pursuit_residual import ResidualComposeConfig, make_base_policy

        if cfg is None:
            self.residual_cfg = None
            self._residual_teacher = None
            self.demo_metadata.pop("residual", None)
            return
        if not isinstance(cfg, ResidualComposeConfig):
            cfg = ResidualComposeConfig(**cfg)
        self.residual_cfg = cfg
        self._residual_teacher = make_base_policy(cfg.base)
        self.demo_metadata["residual"] = {
            "base": cfg.base,
            "mode": cfg.mode,
            "weight": float(cfg.weight),
            "weight_final": None if cfg.weight_final is None else float(cfg.weight_final),
        }

    def execution_actions(self, policy_actions, env):
        if self.residual_cfg is None or self._residual_teacher is None:
            return policy_actions
        from pursuit_residual import compose_actions

        logic = getattr(env, "env", env)
        if hasattr(env, "bridges") and len(getattr(env, "bridges", [])) == 1:
            logic = env.bridges[0].env
        base = np.asarray(self._residual_teacher.actions(logic), dtype=np.float32)
        weight = self.residual_cfg.weight_at(self._residual_env_steps)
        self._residual_env_steps += 1
        return compose_actions(base, policy_actions, self.residual_cfg, weight=weight)

    def reset_actor_optimizer(self):
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.actor_base_lr, eps=1e-5)

    def capture_reference_policy(self):
        """Freeze the DAgger policy, not the previous PPO iterate."""
        self.reference_actor = deepcopy(self.actor).to(self.device).eval()
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)

    def demo_loss_weight(self, episode):
        if not (self.demo_episodes or self.correction_episodes):
            return 0.0
        progress = min(max(float(episode) / max(self.cfg.episodes - 1, 1), 0.0), 1.0)
        floor = min(max(self.demo_weight_floor, 0.0), self.demo_weight)
        return floor + (self.demo_weight - floor) * (1.0 - progress)

    def _make_env(self, episode):
        self.active_env = super()._make_env(episode)
        self.environment_seconds = 0.0
        self.controller_rates = []
        self.safety_metrics = {k: [] for k in ('safety_correction_rate', 'safety_correction_magnitude', 'emergency_stop_rate')}
        self.safety_reason_counts = {}
        self.reward_totals = {}
        self.closest_capture_gap = float("inf")
        self.lidar_rates = []
        self.visibility_totals = {}
        self.visibility_steps = 0
        if not getattr(self.active_env, "_pursuit_timing_wrapped", False):
            original_step = self.active_env.step_joint

            def timed_step(actions):
                started = time.perf_counter()
                result = original_step(actions)
                self.environment_seconds += time.perf_counter() - started
                accumulate_visibility(self.visibility_totals, result)
                self.visibility_steps += 1
                for key, value in result.get("reward_components", {}).items():
                    if key not in ("pursuit", "estimation"):
                        self.reward_totals[key] = self.reward_totals.get(key, 0.0) + float(value)
                self.closest_capture_gap = min(self.closest_capture_gap, result.get("minimum_capture_gap", float("inf")))
                self.controller_rates.append(result.get("controller_feasible_rate", 0.0))
                for key in self.safety_metrics:
                    self.safety_metrics[key].append(result.get(key, 0.0))
                for reasons in result.get("safety_reasons", []):
                    for reason in reasons:
                        self.safety_reason_counts[reason] = self.safety_reason_counts.get(reason, 0) + 1
                if "lidar_detection_ratio" in result:
                    self.lidar_rates.append(result["lidar_detection_ratio"])
                return result

            self.active_env.step_joint = timed_step
            self.active_env._pursuit_timing_wrapped = True
        return self.active_env

    def _update(self, rollout, episode):
        started = time.perf_counter()
        result = update_pursuit(self, rollout, episode)
        result["ppo_update_seconds"] = time.perf_counter() - started
        return result

    @staticmethod
    def _logic_env_at(vec_env, index: int):
        if hasattr(vec_env, "envs"):
            return vec_env.envs[index]
        if hasattr(vec_env, "bridges"):
            return vec_env.bridges[index].env
        raise TypeError("vector env must expose envs[] or bridges[].env")

    @staticmethod
    def _empty_slot_diagnostics():
        return {
            "environment_seconds": 0.0,
            "controller_rates": [],
            "safety_metrics": {
                key: []
                for key in (
                    "safety_correction_rate",
                    "safety_correction_magnitude",
                    "emergency_stop_rate",
                )
            },
            "safety_reason_counts": {},
            "reward_totals": {},
            "closest_capture_gap": float("inf"),
            "lidar_rates": [],
            "visibility_totals": {},
            "visibility_steps": 0,
            "total_reward": 0.0,
            "total_collisions": 0,
            "total_uav_collisions": 0,
            "total_obstacle_conflicts": 0,
            "total_pair_conflicts": 0,
            "outage_rates": [],
            "disabled_rates": [],
            "safety_intervention_rates": [],
            "action_agreement_rates": [],
            "total_new_cells": 0,
            "total_revisit_gain": 0.0,
            "stagnant_agent_rates": [],
            "total_new_crashes": 0,
            "first_detection_step": None,
            "target_loss_count": 0,
            "target_reacquisition_count": 0,
            "last_target_loss_step": None,
            "target_reacquisition_delays": [],
            "direct_visibility_steps": 0,
            "capture_time": 0,
            "track_position_rmse_steps": [],
            "track_velocity_rmse_steps": [],
            "track_nees_steps": [],
            "track_nees_consistency_steps": [],
            "track_age_steps": [],
            "valid_track_rate_steps": [],
            "assignment_search_rates": [],
            "assignment_switch_rates": [],
            "total_track_messages": 0,
            "total_track_communication_bytes": 0,
            "total_active_uav_exposure": 0,
            "mpc_feasible_rates": [],
            "total_path_length": 0.0,
            "total_control_smoothness_cost": 0.0,
            "trace": [],
        }

    def _accumulate_vector_diagnostics(self, diag, result, *, step_index: int, env_seconds: float):
        diag["environment_seconds"] += float(env_seconds)
        accumulate_visibility(diag["visibility_totals"], result)
        diag["visibility_steps"] += 1
        for key, value in result.get("reward_components", {}).items():
            if key not in ("pursuit", "estimation"):
                diag["reward_totals"][key] = diag["reward_totals"].get(key, 0.0) + float(value)
        diag["closest_capture_gap"] = min(
            diag["closest_capture_gap"],
            float(result.get("minimum_capture_gap", float("inf"))),
        )
        diag["controller_rates"].append(result.get("controller_feasible_rate", 0.0))
        for key in diag["safety_metrics"]:
            diag["safety_metrics"][key].append(result.get(key, 0.0))
        for reasons in result.get("safety_reasons", []):
            for reason in reasons:
                diag["safety_reason_counts"][reason] = diag["safety_reason_counts"].get(reason, 0) + 1
        if "lidar_detection_ratio" in result:
            diag["lidar_rates"].append(result["lidar_detection_ratio"])

        reward = float(result["reward"])
        diag["total_reward"] += reward
        diag["total_collisions"] += int(result.get("collisions", 0))
        diag["total_uav_collisions"] += int(result.get("uav_collisions", 0))
        diag["total_obstacle_conflicts"] += int(result.get("obstacle_conflicts", 0))
        diag["total_pair_conflicts"] += int(result.get("pair_conflicts", 0))
        diag["outage_rates"].append(float(result.get("outage_rate", 0.0)))
        diag["disabled_rates"].append(float(result.get("disabled_rate", 0.0)))
        diag["safety_intervention_rates"].append(float(result.get("safety_intervention_rate", 0.0)))
        diag["action_agreement_rates"].append(float(result.get("action_agreement_rate", 1.0)))
        diag["total_new_cells"] += int(result.get("new_cells", 0))
        diag["total_revisit_gain"] += float(result.get("revisit_gain", 0.0))
        diag["stagnant_agent_rates"].append(
            float(result.get("stagnant_agents", 0)) / max(self.n_agents, 1)
        )
        diag["total_new_crashes"] += int(result.get("new_crashes", 0))
        directly_visible = bool(result.get("direct_target_visible", 0.0))
        diag["direct_visibility_steps"] += int(directly_visible)
        if directly_visible and diag["first_detection_step"] is None:
            diag["first_detection_step"] = step_index + 1
        diag["target_loss_count"] += int(bool(result.get("target_lost", 0.0)))
        diag["target_reacquisition_count"] += int(bool(result.get("target_reacquired", 0.0)))
        if bool(result.get("target_lost", 0.0)):
            diag["last_target_loss_step"] = step_index + 1
        if bool(result.get("target_reacquired", 0.0)) and diag["last_target_loss_step"] is not None:
            diag["target_reacquisition_delays"].append(
                step_index + 1 - diag["last_target_loss_step"]
            )
            diag["last_target_loss_step"] = None
        if bool(result.get("capture_success", 0.0)) and diag["capture_time"] == 0:
            diag["capture_time"] = step_index + 1
        for key, destination_key in (
            ("track_position_rmse", "track_position_rmse_steps"),
            ("track_velocity_rmse", "track_velocity_rmse_steps"),
            ("mean_track_nees", "track_nees_steps"),
            ("track_nees_consistency_rate", "track_nees_consistency_steps"),
            ("mean_track_age", "track_age_steps"),
            ("valid_track_rate", "valid_track_rate_steps"),
            ("assignment_search_rate", "assignment_search_rates"),
            ("assignment_switch_rate", "assignment_switch_rates"),
        ):
            metric_value = float(result.get(key, np.nan))
            if np.isfinite(metric_value):
                diag[destination_key].append(metric_value)
        diag["total_track_messages"] += int(result.get("track_messages", 0))
        diag["total_track_communication_bytes"] += int(
            result.get("track_communication_bytes", 0)
        )
        diag["total_active_uav_exposure"] += int(
            result.get("active_uav_exposure", self.n_agents)
        )
        if "mpc_feasible_rate" in result:
            diag["mpc_feasible_rates"].append(float(result["mpc_feasible_rate"]))
        diag["total_path_length"] += float(result.get("step_path_length", 0.0))
        diag["total_control_smoothness_cost"] += float(
            result.get("control_smoothness_cost", 0.0)
        )
        return reward

    def _sample_continuous_step(self, obs, logic_env):
        """Sample one continuous MAPPO step; returns (executed, rollout_fields)."""
        obs_vec = torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=self.device)
        adjacency = torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=self.device)
        hetero_graph = (
            tensorize_heterogeneous_graph(obs["hetero_graph"], self.device)
            if self.cfg.use_gat and self.cfg.use_hetero_entities
            else None
        )
        state = torch.as_tensor(obs["state_vector"], dtype=torch.float32, device=self.device)
        masks = torch.as_tensor(obs["action_masks"], dtype=torch.bool, device=self.device)
        all_policy_outputs = self.actor(obs_vec, adjacency, hetero_graph)
        value = self._value_estimate(obs_vec, state)
        means, log_stds = torch.chunk(all_policy_outputs, 2, dim=-1)
        log_stds = torch.clamp(log_stds, -5.0, self.cfg.continuous_log_std_max)
        distributions = torch.distributions.Normal(means, torch.exp(log_stds))
        raw_samples = distributions.rsample()
        squashed = torch.tanh(raw_samples)
        corrected_logps = (
            distributions.log_prob(raw_samples)
            - torch.log(torch.clamp(1.0 - squashed.pow(2), min=1e-6))
        ).sum(dim=-1)
        actions = squashed.detach().cpu().numpy()
        policy_samples = [raw_samples[i].detach() for i in range(self.n_agents)]
        logps = [corrected_logps[i] for i in range(self.n_agents)]
        executed = np.asarray(actions, dtype=float)
        if hasattr(self, "execution_actions"):
            executed = np.asarray(self.execution_actions(executed, logic_env), dtype=executed.dtype)
        return executed, {
            "obs_vec": obs_vec,
            "adjacency": adjacency,
            "hetero_graph": hetero_graph,
            "state": state,
            "masks": masks,
            "actions": actions,
            "policy_samples": policy_samples,
            "logps": logps,
            "value": value,
        }

    def _append_rollout_transition(self, rollout, fields, transition_result, *, truncated_override=False):
        next_obs = transition_result["obs"]
        next_state = torch.as_tensor(
            next_obs["state_vector"], dtype=torch.float32, device=self.device
        )
        terminated = bool(transition_result.get("terminated", False))
        truncated = bool(transition_result.get("truncated", False)) or bool(truncated_override)
        reward = float(transition_result["reward"]) * self.cfg.reward_scale
        rollout.append(
            (
                fields["obs_vec"],
                fields["adjacency"],
                fields["hetero_graph"],
                fields["state"],
                fields["masks"],
                fields["actions"],
                fields["policy_samples"],
                torch.stack(fields["logps"]).detach(),
                fields["value"].detach(),
                reward,
                next_state,
                terminated,
                truncated,
                1.0 - float(transition_result.get("coverage", 0.0)),
                torch.as_tensor(
                    next_obs["agent_observations"],
                    dtype=torch.float32,
                    device=self.device,
                ),
                None,
            )
        )
        return terminated, truncated

    def _apply_slot_diagnostics_to_self(self, diag, logic_env):
        self.active_env = logic_env
        self.environment_seconds = diag["environment_seconds"]
        self.controller_rates = diag["controller_rates"]
        self.safety_metrics = diag["safety_metrics"]
        self.safety_reason_counts = diag["safety_reason_counts"]
        self.reward_totals = diag["reward_totals"]
        self.closest_capture_gap = diag["closest_capture_gap"]
        self.lidar_rates = diag["lidar_rates"]
        self.visibility_totals = diag["visibility_totals"]
        self.visibility_steps = diag["visibility_steps"]

    def _history_row_from_slot(
        self,
        diag,
        logic_env,
        result,
        update_stats,
        *,
        episode,
        rollout_limit,
        steps=None,
        env_already_reset=False,
    ):
        steps = int(steps if steps is not None else getattr(logic_env, "t", 0) or 0)
        if steps <= 0:
            steps = int(len(diag.get("trace") or []) or 1)
        captured = float(result.get("capture_success", 0.0))
        if env_already_reset:
            # Auto-reset already replaced the live env; use terminal flags only.
            task_progress = {
                "target_discovery_rate": captured,
                "target_tracking_rate": captured,
                "target_completion_rate": captured,
                "belief_decisiveness": float("nan"),
                "belief_brier_score": float("nan"),
                "task_success": captured,
            }
            coverage = float(result.get("coverage", 0.0))
        else:
            task_progress = episode_task_metrics(logic_env)
            coverage = float(logic_env.coverage()) if hasattr(logic_env, "coverage") else 0.0
        row = {
            "episode": episode,
            "reward": diag["total_reward"],
            "coverage": coverage,
            **task_progress,
            "collisions": diag["total_collisions"],
            "uav_collisions": diag["total_uav_collisions"],
            "obstacle_conflicts": diag["total_obstacle_conflicts"],
            "pair_conflicts": diag["total_pair_conflicts"],
            "collision_rate": float(diag["total_collisions"] / max(steps, 1)),
            "outage_rate": float(np.mean(diag["outage_rates"])) if diag["outage_rates"] else 0.0,
            "disabled_rate": float(np.mean(diag["disabled_rates"])) if diag["disabled_rates"] else 0.0,
            "safety_intervention_rate": (
                float(np.mean(diag["safety_intervention_rates"]))
                if diag["safety_intervention_rates"]
                else 0.0
            ),
            "action_agreement_rate": (
                float(np.mean(diag["action_agreement_rates"]))
                if diag["action_agreement_rates"]
                else 1.0
            ),
            "new_cells": diag["total_new_cells"],
            "revisit_gain": diag["total_revisit_gain"],
            "stagnant_agent_rate": (
                float(np.mean(diag["stagnant_agent_rates"])) if diag["stagnant_agent_rates"] else 0.0
            ),
            "new_crashes": diag["total_new_crashes"],
            "capture_success": float(result.get("capture_success", 0.0)),
            "capture_hold_count": int(result.get("capture_hold_count", 0)),
            "capture_close_uavs": int(result.get("capture_close_uavs", 0)),
            "capture_angular_span_deg": float(result.get("capture_angular_span_deg", 0.0)),
            "direct_target_visible": float(result.get("direct_target_visible", 0.0)),
            "first_onboard_contact_time": int(diag["first_detection_step"] or (rollout_limit + 1)),
            "first_detection_time": int(diag["first_detection_step"] or (rollout_limit + 1)),
            "capture_time": int(diag["capture_time"] or (rollout_limit + 1)),
            "target_loss_count": int(diag["target_loss_count"]),
            "target_reacquisition_count": int(diag["target_reacquisition_count"]),
            "mean_reacquisition_time": (
                float(np.mean(diag["target_reacquisition_delays"]))
                if diag["target_reacquisition_delays"]
                else np.nan
            ),
            "continuous_visibility_ratio": float(
                diag["direct_visibility_steps"] / max(steps, 1)
            ),
            "track_position_rmse": float(
                np.sqrt(np.mean(np.square(diag["track_position_rmse_steps"])))
                if diag["track_position_rmse_steps"]
                else np.nan
            ),
            "track_velocity_rmse": float(
                np.sqrt(np.mean(np.square(diag["track_velocity_rmse_steps"])))
                if diag["track_velocity_rmse_steps"]
                else np.nan
            ),
            "mean_track_nees": (
                float(np.mean(diag["track_nees_steps"])) if diag["track_nees_steps"] else np.nan
            ),
            "track_nees_consistency_rate": (
                float(np.mean(diag["track_nees_consistency_steps"]))
                if diag["track_nees_consistency_steps"]
                else np.nan
            ),
            "mean_track_age": (
                float(np.mean(diag["track_age_steps"])) if diag["track_age_steps"] else np.nan
            ),
            "valid_track_rate": (
                float(np.mean(diag["valid_track_rate_steps"]))
                if diag["valid_track_rate_steps"]
                else 0.0
            ),
            "track_messages": int(diag["total_track_messages"]),
            "track_communication_bytes": int(diag["total_track_communication_bytes"]),
            "assignment_search_rate": (
                float(np.mean(diag["assignment_search_rates"]))
                if diag["assignment_search_rates"]
                else 1.0
            ),
            "assignment_switch_rate": (
                float(np.mean(diag["assignment_switch_rates"]))
                if diag["assignment_switch_rates"]
                else 0.0
            ),
            "collision_exposure_rate": float(
                diag["total_collisions"] / max(diag["total_active_uav_exposure"], 1)
            ),
            "mpc_feasible_rate": (
                float(np.mean(diag["mpc_feasible_rates"]))
                if diag["mpc_feasible_rates"]
                else np.nan
            ),
            "path_length": float(diag["total_path_length"]),
            "control_smoothness_cost": float(diag["total_control_smoothness_cost"]),
            "weight_target": float("nan"),
            "weight_exploration": float("nan"),
            "weight_revisit": float("nan"),
            "weight_connectivity": float("nan"),
            "weight_safety": float("nan"),
            "steps": steps,
            "num_envs": int(getattr(self, "_vector_num_envs", 1) or 1),
            **update_stats,
        }
        return row

    def _soft_reset_vector_slot(self, vec_env, observations, index, *, seed):
        if hasattr(vec_env, "_reset_env_indices"):
            vec_env._reset_env_indices([index], seeds=[int(seed)])
            observations[index] = vec_env.bridges[index].observe()
            return
        if hasattr(vec_env, "envs"):
            env = vec_env.envs[index]
            env.cfg.seed = int(seed)
            env.rng = np.random.default_rng(int(seed))
            env.reset()
            observations[index] = env.observe_search()
            return
        raise TypeError("vector env cannot soft-reset individual slots")

    def _flush_vector_batch(
        self, batch, history, *, completed, target, num_envs, started_at, episode_wall
    ):
        """Run on-policy PPO for trajectories collected under one frozen policy."""
        for item in batch:
            if completed >= target:
                break
            completed += 1
            update_stats = self._update(item["rollout"], completed - 1)
            self.episode_traces.append(
                {"episode": completed, "frames": item["diag"]["trace"], "mode": "3d"}
            )
            self.episode_traces = self.episode_traces[-3:]
            self._apply_slot_diagnostics_to_self(item["diag"], item["logic_env"])
            self._finalize_use_closest_gap = bool(item["env_already_reset"])
            self._finalize_disabled_count = item["disabled_count"]
            if item["env_already_reset"] and item["terminal_gap"] < float("inf"):
                self.closest_capture_gap = min(self.closest_capture_gap, item["terminal_gap"])
            row = self._history_row_from_slot(
                item["diag"],
                item["logic_env"],
                item["capture_result"],
                update_stats,
                episode=completed,
                rollout_limit=item["rollout_limit"],
                steps=item["episode_steps"],
                env_already_reset=item["env_already_reset"],
            )
            row = self.finalize_pursuit_row(row)
            if item["env_already_reset"] and np.isfinite(item["terminal_gap"]):
                row["minimum_capture_gap"] = item["terminal_gap"]
            self._finalize_use_closest_gap = False
            self._finalize_disabled_count = None
            history.append(row)
            callback = getattr(self, "on_vector_episode_complete", None)
            if callable(callback):
                callback(completed)
            if getattr(self, "early_stop", False):
                print(
                    f"[MAPPO-VEC] stopped early at episode={completed}: "
                    f"{getattr(self, 'early_stop_reason', 'capture_collapse')}",
                    flush=True,
                )
                break

            episode_seconds = time.perf_counter() - episode_wall
            if (
                completed == 1
                or completed % max(1, self.cfg.log_interval) == 0
                or completed >= target
            ):
                elapsed = time.perf_counter() - started_at
                run_count = max(completed - int(getattr(self, "_vector_start_episode", 0)), 1)
                eta = elapsed / run_count * max(target - completed, 0)
                rmse = row.get("track_position_rmse", float("nan"))
                reward = float(row.get("reward") if row.get("reward") is not None else 0.0)
                capture = float(row.get("capture_success") or 0.0)
                visibility = float(row.get("continuous_visibility_ratio") or 0.0)
                print(
                    f"[MAPPO] episode={completed}/{target} reward={reward:.2f} "
                    f"env={float(row.get('environment_seconds') or 0.0):.2f}s "
                    f"ppo={float(row.get('ppo_update_seconds') or 0.0):.2f}s "
                    f"capture={capture:.0f} "
                    f"first_obs_step={row.get('first_onboard_contact_time')} "
                    f"visibility={visibility:.3f} "
                    f"target_losses={row.get('target_loss_count')} "
                    f"capture_rate_50={np.mean([float(r.get('capture_success') or 0.0) for r in history[-50:]]):.3f} "
                    f"track_rmse={float(rmse) if np.isfinite(rmse) else float('nan'):.3f} "
                    f"entropy={float(row.get('entropy') if row.get('entropy') is not None else float('nan')):.3f} "
                    f"num_envs={num_envs} "
                    f"time={episode_seconds:.2f}s eta={eta / 3600:.2f}h",
                    flush=True,
                )

            if self.cfg.checkpoint_path and (
                completed % max(1, self.cfg.checkpoint_interval) == 0 or completed >= target
            ):
                self.save_checkpoint(self.cfg.checkpoint_path, episode=completed, history=history)
            budgets = tuple(
                int(x) for x in getattr(self.cfg, "episode_budget_checkpoints", ()) or ()
            )
            if self.cfg.checkpoint_path and budgets and completed in budgets:
                budget_dir = Path(self.cfg.checkpoint_path).resolve().parent / "budgets"
                budget_dir.mkdir(parents=True, exist_ok=True)
                budget_path = budget_dir / f"ep{completed:04d}.pt"
                self.save_checkpoint(str(budget_path), episode=completed, history=history)
                print(f"[MAPPO] budget checkpoint → {budget_path}", flush=True)
            if completed >= target:
                break
        return completed

    def train_vectorized(self, vec_env, *, target_episodes=None):
        """Collect from Isaac/kinematic vector envs with on-policy PPO.

        All ``num_envs`` scenes step in lockstep under a frozen policy (using
        ``info['final_obs']`` on auto-reset). When the first scene finishes an
        episode, that trajectory is PPO-updated once; every other in-flight or
        simultaneously-finished trajectory is discarded and soft-reset so
        likelihood parity stays valid. PhysX still runs N-way in parallel for
        most of each episode wall-clock.
        """
        if not self.continuous_actions:
            raise ValueError("train_vectorized currently supports continuous pursuit MAPPO only")
        target = int(self.cfg.episodes if target_episodes is None else target_episodes)
        completed = int(self.start_episode)
        self._vector_start_episode = completed
        history = list(self.restored_history)
        base_seed = int(getattr(self, "seed", None) or self.environment_config.get("seed", 0))
        num_envs = int(vec_env.num_envs)
        self._vector_num_envs = num_envs
        observations = vec_env.reset(
            seeds=[base_seed * 100_000 + completed + i for i in range(num_envs)]
        )
        print(
            f"[MAPPO-VEC] device={self.device} num_envs={num_envs} "
            f"episodes={completed + 1}..{target}",
            flush=True,
        )
        started_at = time.perf_counter()
        step_limit = int(self.cfg.max_steps or 0)
        if step_limit <= 0:
            step_limit = int(getattr(getattr(vec_env, "cfg", None), "search_steps", 0) or 0)
        local_steps = np.zeros(num_envs, dtype=np.int32)
        rollouts = [[] for _ in range(num_envs)]
        diagnostics = [self._empty_slot_diagnostics() for _ in range(num_envs)]
        pending_fields = [None] * num_envs
        finished_batch = []
        next_episode_seed = completed

        while completed < target:
            episode_wall = time.perf_counter()
            actions = []
            for index in range(num_envs):
                logic_env = self._logic_env_at(vec_env, index)
                executed, fields = self._sample_continuous_step(observations[index], logic_env)
                pending_fields[index] = fields
                actions.append(executed)
            actions = np.stack(actions, axis=0)
            step_started = time.perf_counter()
            results = vec_env.step_joint(actions)
            vector_seconds = time.perf_counter() - step_started
            per_env_seconds = vector_seconds / max(num_envs, 1)

            for index, result in enumerate(results):
                logic_env = self._logic_env_at(vec_env, index)
                fields = pending_fields[index]
                transition_result = dict(result)
                info = result.get("info") or {}
                if result.get("_episode_done") and info.get("final_obs") is not None:
                    transition_result["obs"] = info["final_obs"]
                    if "final_terminated" in info:
                        transition_result["terminated"] = bool(info["final_terminated"])
                        transition_result["truncated"] = bool(info.get("final_truncated", False))
                    else:
                        captured = bool(
                            info.get("capture_success", result.get("capture_success", False))
                        )
                        transition_result["terminated"] = captured
                        transition_result["truncated"] = not captured
                local_steps[index] += 1
                force_done = step_limit > 0 and int(local_steps[index]) >= step_limit
                truncated_override = bool(
                    force_done and not bool(transition_result.get("terminated", False))
                )
                self._accumulate_vector_diagnostics(
                    diagnostics[index],
                    transition_result,
                    step_index=int(local_steps[index]) - 1,
                    env_seconds=per_env_seconds,
                )
                terminated, truncated = self._append_rollout_transition(
                    rollouts[index],
                    fields,
                    transition_result,
                    truncated_override=truncated_override,
                )
                diagnostics[index]["trace"].append(
                    capture_env_frame(logic_env, transition_result, fields["actions"])
                )

                if result.get("_episode_done") or force_done or terminated or truncated:
                    capture_result = dict(transition_result)
                    capture_result["capture_success"] = bool(
                        info.get(
                            "capture_success",
                            transition_result.get(
                                "capture_success", result.get("capture_success", False)
                            ),
                        )
                    )
                    episode_steps = int(local_steps[index])
                    env_already_reset = bool(
                        result.get("_episode_done") and info.get("final_obs") is not None
                    )
                    terminal_gap = float(
                        transition_result.get(
                            "minimum_capture_gap",
                            diagnostics[index]["closest_capture_gap"],
                        )
                    )
                    finished_batch.append(
                        {
                            "rollout": rollouts[index],
                            "diag": diagnostics[index],
                            "logic_env": logic_env,
                            "capture_result": capture_result,
                            "episode_steps": episode_steps,
                            "rollout_limit": step_limit or episode_steps,
                            "env_already_reset": env_already_reset,
                            "terminal_gap": terminal_gap,
                            "disabled_count": (
                                int(np.count_nonzero(getattr(logic_env, "disabled_uavs", [])))
                                if not env_already_reset
                                else int(result.get("disabled_count", 0) or 0)
                            ),
                        }
                    )
                    rollouts[index] = []
                    diagnostics[index] = self._empty_slot_diagnostics()
                    local_steps[index] = 0
                    next_episode_seed += 1
                    if force_done and not result.get("_episode_done"):
                        self._soft_reset_vector_slot(
                            vec_env,
                            observations,
                            index,
                            seed=base_seed * 100_000 + next_episode_seed + index,
                        )
                    else:
                        observations[index] = result["obs"]
                else:
                    observations[index] = result["obs"]

            if not finished_batch or completed >= target:
                continue

            # One on-policy update per flush. Remaining finished/partial rollouts
            # were collected under the pre-update policy and must be dropped.
            to_flush = finished_batch[:1]
            finished_batch = []
            completed = self._flush_vector_batch(
                to_flush,
                history,
                completed=completed,
                target=target,
                num_envs=num_envs,
                started_at=started_at,
                episode_wall=episode_wall,
            )
            for index in range(num_envs):
                rollouts[index] = []
                diagnostics[index] = self._empty_slot_diagnostics()
                local_steps[index] = 0
                next_episode_seed += 1
                self._soft_reset_vector_slot(
                    vec_env,
                    observations,
                    index,
                    seed=base_seed * 100_000 + next_episode_seed + index,
                )

        self.start_episode = target
        self.restored_history = history
        self.pursuit_history = list(history)
        return history

    def finalize_pursuit_row(self, row):
        if self.episode_traces:
            self.episode_traces[-1].update(mode="3d")
        row = clean_history_row(row)
        row["environment_seconds"] = self.environment_seconds
        row["controller_feasible_rate"] = float(np.mean(self.controller_rates)) if self.controller_rates else None
        row.update({key: float(np.mean(values)) for key, values in self.safety_metrics.items()})
        row['safety_reason_counts'] = dict(self.safety_reason_counts)
        if self.lidar_rates:
            row["lidar_detection_ratio"] = float(np.mean(self.lidar_rates))
        row["target_observation_ratio"] = row.get("continuous_visibility_ratio")
        row.update(finalize_visibility(self.visibility_totals, self.visibility_steps))
        if getattr(self, "_finalize_use_closest_gap", False):
            gap = self.closest_capture_gap
            row["minimum_capture_gap"] = (
                float(gap) if np.isfinite(gap) else float(self.active_env.minimum_capture_gap())
            )
        else:
            row["minimum_capture_gap"] = self.active_env.minimum_capture_gap()
        row["demo_loss_weight"] = self.demo_loss_weight(row["episode"] - 1)
        row["reference_kl_weight"] = self.reference_kl_weight if self.reference_actor is not None else 0.0
        row["safeguard_rollbacks"] = self.safeguard_rollbacks
        row["reward_components"] = dict(self.reward_totals)
        row["closest_capture_gap"] = self.closest_capture_gap
        disabled_override = getattr(self, "_finalize_disabled_count", None)
        row["disabled_uavs_final"] = (
            int(disabled_override)
            if disabled_override is not None
            else int(np.count_nonzero(self.active_env.disabled_uavs))
        )
        if self.output_directory:
            diagnostic = {"episode": row["episode"], "reward": row["reward"],
                          "component_sum": sum(self.reward_totals.values()),
                          "components": self.reward_totals,
                          "closest_capture_gap": self.closest_capture_gap,
                          "final_capture_gap": row["minimum_capture_gap"],
                          "controller_feasible_rate": row["controller_feasible_rate"],
                          "disabled_uavs_final": row["disabled_uavs_final"]}
            diagnostic.update({key: row[key] for key in (*self.safety_metrics, 'safety_reason_counts',
                               'exact_policy_kl', 'pre_update_logprob_error', 'action_mean_saturation', 'ppo_backtracks')})
            with (self.output_directory / "reward_diagnostics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(diagnostic) + "\n")
            if row["episode"] == 1 or row["episode"] % 10 == 0:
                terms = " ".join(f"{key}={value:.2f}" for key, value in self.reward_totals.items())
                print(f"[reward-breakdown] episode={row['episode']} {terms} closest_capture_gap={self.closest_capture_gap:.3f} disabled={row['disabled_uavs_final']}", flush=True)
                print(f"[pursuit-health] episode={row['episode']} kl={row['exact_policy_kl']:.5f} "
                      f"saturation={row['action_mean_saturation']:.3f} correction={row['safety_correction_rate']:.3f} "
                      f"emergency={row['emergency_stop_rate']:.3f}", flush=True)
        self.pursuit_history.append(row)
        if self.output_directory and (row["episode"] == 1 or row["episode"] % 50 == 0):
            write_reward_capture_chart(self.output_directory / "reward_capture.svg", self.pursuit_history)
        if self.validation_seeds and row["episode"] % self.validation_interval == 0:
            self.validate_capture(row["episode"])
        return row

    def validate_capture(self, episode):
        if getattr(self, "physical_metrics_fn", None) is not None and self.validation_seeds:
            metrics = self.physical_metrics_fn()
        else:
            result = evaluate(self, QuadrotorPursuitConfig(**self.environment_config), self.validation_seeds, ["mappo"])
            metrics = result["summary"]["mappo"]
        score = (metrics["capture_rate"], -metrics["mean_censored_steps"], metrics['mean_return'])
        self.validation_records.append({"episode": episode, **metrics})
        if score > self.best_capture_score:
            self.best_capture_score = score
            self.best_actor_state = {
                key: value.detach().cpu().clone() for key, value in self.actor.state_dict().items()
            }
            self.safeguard_bad_validations = 0
            self.save_checkpoint(self.output_directory / "best_capture.pt", episode=episode, history=self.pursuit_history)
        elif metrics["capture_rate"] < self.best_capture_score[0] - self.safeguard_drop:
            self.safeguard_bad_validations += 1
            if self.best_actor_state is not None and self.safeguard_bad_validations >= self.safeguard_patience:
                old_actor_lr = self.actor_base_lr
                self.actor.load_state_dict(self.best_actor_state)
                self.actor_base_lr = max(self.actor_base_lr * 0.5, self.min_actor_lr)
                self.reset_actor_optimizer()
                self.safeguard_rollbacks += 1
                self.safeguard_bad_validations = 0
                self.validation_records[-1]["actor_rollback"] = True
                self.validation_records[-1]["rollback_reason"] = "capture_rate_degradation"
                self.validation_records[-1]["actor_lr_before_rollback"] = old_actor_lr
                self.validation_records[-1]["actor_lr_after_rollback"] = self.actor_base_lr
                self.save_checkpoint(
                    self.output_directory / "rollback_latest.pt", episode=episode, history=self.pursuit_history
                )
                max_rb = int(getattr(self, "max_safeguard_rollbacks", 0) or 0)
                if max_rb > 0 and self.safeguard_rollbacks >= max_rb:
                    self.early_stop = True
                    self.early_stop_reason = f"capture_rate_degradation_rollbacks>={max_rb}"
                    self.validation_records[-1]["early_stop"] = True
                    self.validation_records[-1]["early_stop_reason"] = self.early_stop_reason
                    print(
                        f"[validation] early_stop after {self.safeguard_rollbacks} rollbacks; "
                        f"keeping best_capture.pt (rate={self.best_capture_score[0]:.3f})",
                        flush=True,
                    )
        else:
            self.safeguard_bad_validations = 0
        (self.output_directory / "validation.json").write_text(json.dumps(self.validation_records, indent=2), encoding="utf-8")

    def pursuit_demo_loss(self, episode):
        weight = self.demo_loss_weight(episode)
        if weight == 0:
            return 0.0
        batch_size = min(self.cfg.batch_size, 32)
        if not self.correction_episodes:
            return weight * cloning_loss(self, self.demo_episodes, batch_size, self.demo_rng)
        correction_count = int(round(batch_size * self.correction_fraction))
        if self.correction_fraction > 0:
            correction_count = max(1, correction_count)
        demo_count = batch_size - correction_count
        losses, counts = [], []
        if demo_count and self.demo_episodes:
            losses.append(cloning_loss(self, self.demo_episodes, demo_count, self.demo_rng))
            counts.append(demo_count)
        if correction_count:
            losses.append(cloning_loss(self, self.correction_episodes, correction_count, self.demo_rng))
            counts.append(correction_count)
        return weight * sum(loss * count for loss, count in zip(losses, counts)) / sum(counts)

    def save_checkpoint(self, path, env_config=None, episode=None, history=None):
        super().save_checkpoint(path, env_config or self.environment_config, episode,
                                [clean_history_row(row) for row in (history or [])])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["demo_metadata"] = self.demo_metadata
        payload["best_capture_score"] = self.best_capture_score
        payload["validation_records"] = self.validation_records
        payload["best_actor_state"] = self.best_actor_state
        payload["reference_actor"] = None if self.reference_actor is None else self.reference_actor.state_dict()
        payload["anti_forgetting"] = {
            "demo_weight_floor": self.demo_weight_floor,
            "correction_fraction": self.correction_fraction,
            "reference_kl_weight": self.reference_kl_weight,
            "safeguard_drop": self.safeguard_drop,
            "safeguard_patience": self.safeguard_patience,
            "safeguard_rollbacks": self.safeguard_rollbacks,
            "min_actor_lr": self.min_actor_lr,
            "actor_base_lr": self.actor_base_lr,
        }
        payload["pursuit_graph_version"] = PURSUIT_GRAPH_VERSION
        payload["train_seed_bank"] = list(self.train_seed_bank)
        payload["validation_seed_bank"] = list(self.validation_seed_bank)
        payload["seed_bank_version"] = self.seed_bank_version
        torch.save(payload, path)

    def load_checkpoint(self, path):
        recorded = torch.load(path, map_location='cpu', weights_only=False)
        load_environment_config(recorded['env_config'])
        if self.cfg.use_gat and recorded.get("pursuit_graph_version") != PURSUIT_GRAPH_VERSION:
            raise ValueError(
                "Checkpoint uses the legacy search HGAT and cannot initialize PursuitGraphActor. "
                "Re-run pretrain/DAgger with the current code; old demo states can be upgraded in memory."
            )
        payload = super().load_checkpoint(path)
        self.demo_metadata = payload.get("demo_metadata", {})
        self.best_capture_score = tuple(payload.get("best_capture_score", self.best_capture_score))
        self.validation_records = payload.get("validation_records", [])
        self.best_actor_state = payload.get("best_actor_state")
        self.pursuit_history = list(self.restored_history)
        self.train_seed_bank = list(payload.get("train_seed_bank", []))
        self.validation_seed_bank = list(payload.get("validation_seed_bank", []))
        self.seed_bank_version = payload.get("seed_bank_version", SEED_BANK_VERSION)
        settings = payload.get("anti_forgetting", {})
        for key in ("demo_weight_floor", "correction_fraction", "reference_kl_weight",
                    "safeguard_drop", "safeguard_patience", "safeguard_rollbacks",
                    "min_actor_lr", "actor_base_lr"):
            if key in settings:
                setattr(self, key, settings[key])
        if payload.get("reference_actor") is not None:
            self.capture_reference_policy()
            self.reference_actor.load_state_dict(payload["reference_actor"])
        return payload

    @torch.no_grad()
    def action(self, obs):
        return deterministic_action(self, obs)


@torch.no_grad()
def deterministic_action(trainer, obs, env=None):
    vectors, adjacency, graph = observation_tensors(obs, trainer.device, trainer.cfg)
    means, _ = trainer.actor(vectors, adjacency, graph).chunk(2, dim=-1)
    action = means.tanh().cpu().numpy()
    if env is not None and hasattr(trainer, "execution_actions"):
        return trainer.execution_actions(action, env)
    return action


def observation_tensors(obs, device, cfg):
    return (
        torch.as_tensor(obs["agent_observations"], dtype=torch.float32, device=device),
        torch.as_tensor(obs["comm_adjacency"], dtype=torch.bool, device=device),
        tensorize_heterogeneous_graph(obs["hetero_graph"], device)
        if cfg.use_gat and cfg.use_hetero_entities else None,
    )


def snapshot(obs, actions, active):
    # All inputs are pre-action observations; no future or target-truth labels.
    return {
        "agent_observations": np.array(obs["agent_observations"], copy=True),
        "comm_adjacency": np.array(obs["comm_adjacency"], copy=True),
        "hetero_graph": {k: np.array(v, copy=True) for k, v in obs["hetero_graph"].items()},
        "actions": np.array(actions, copy=True),
        "active": np.array(active, copy=True),
    }


def collect_demos(env_cfg, first_seed, wanted, attempts, teacher_name, progress_path=None):
    episodes, report = [], []
    for seed in range(first_seed, first_seed + attempts):
        env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
        teacher = BASELINES_3D[teacher_name]()
        teacher.reset(env)
        obs, trajectory = env.observe_search(), []
        for step in range(env.cfg.search_steps):
            action = teacher.actions(env)
            trajectory.append(snapshot(obs, action, ~env.disabled_uavs))
            result = env.step_joint(action)
            obs = result["obs"]
            if (step + 1) % 50 == 0:
                print(f"[demonstrations] seed={seed} step={step+1} capture_gap={env.minimum_capture_gap():.3f}", flush=True)
            if result["terminated"] or result["truncated"]:
                break
        success = bool(result.get("capture_success", False))
        report.append({"seed": seed, "success": success, "steps": len(trajectory)})
        if success:
            episodes.append(trajectory)
        if progress_path is not None:
            progress_path = Path(progress_path)
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            partial = {"env_config": asdict(env_cfg), "teacher": teacher_name, "episodes": episodes,
                       "report": report, "pursuit_graph_version": PURSUIT_GRAPH_VERSION}
            torch.save(partial, progress_path)
            progress_path.with_suffix(".json").write_text(json.dumps({k: v for k, v in partial.items() if k != "episodes"}, indent=2), encoding="utf-8")
        print(f"[demonstrations] seed={seed} success={success} kept={len(episodes)}/{wanted}", flush=True)
        if len(episodes) >= wanted:
            break
    if len(episodes) < wanted:
        raise RuntimeError(f"Only {len(episodes)}/{wanted} successful demonstrations in {attempts} attempts; "
                           "increase --demo-attempts or inspect the teacher under this configuration")
    return {"env_config": asdict(env_cfg), "teacher": teacher_name, "episodes": episodes,
            "report": report, "pursuit_graph_version": PURSUIT_GRAPH_VERSION}


def validate_demos(dataset, env_cfg):
    # Fill newly added defaults for older datasets without changing their mode.
    recorded = asdict(load_environment_config(dataset["env_config"]))
    current = asdict(env_cfg)
    recorded.pop("seed", None)
    current.pop("seed", None)
    if recorded != current:
        raise ValueError("Demonstration environment differs from training; regenerate demos with the same config")
    if not dataset["episodes"] or any(not ep for ep in dataset["episodes"]):
        raise ValueError("Empty demonstration dataset")
    # Legacy snapshots contain the same causal flat track/peer/building data,
    # so upgrade their graph without consulting environment ground truth.
    for episode in dataset["episodes"]:
        for row in episode:
            if not isinstance(row, dict) or "agent_observations" not in row:
                continue
            graph = row.get("hetero_graph", {})
            if graph.get("self_nodes") is None or graph.get("uav_xyz") is None:
                row["hetero_graph"] = pursuit_graph_from_flat_observation(row["agent_observations"], env_cfg)
    dataset["pursuit_graph_version"] = PURSUIT_GRAPH_VERSION


def load_correction_replay(path, original, env_cfg):
    """Load and causally upgrade a corrections file, validating its lineage."""
    corrections = torch.load(path, map_location="cpu", weights_only=False)
    validate_demos(corrections, env_cfg)
    report = corrections.get("correction_report", [])
    original_count = len(corrections["episodes"]) - len(report)
    if not report or original_count < 0:
        raise ValueError("corrections.pt must contain a nonempty correction_report")
    if corrections.get("teacher") != original.get("teacher") or original_count != len(original["episodes"]):
        raise ValueError("corrections.pt was not aggregated from the supplied original demonstrations")
    return corrections, corrections["episodes"][original_count:]


def validate_correction_checkpoint_lineage(checkpoint_metadata, corrections):
    """Prevent pairing a selected DAgger actor with later correction data."""
    if checkpoint_metadata.get("initialization_type") != "dagger":
        return
    expected_round = checkpoint_metadata.get("dagger_round")
    actual_round = corrections.get("dagger_round")
    if actual_round != expected_round:
        raise ValueError(
            f"Checkpoint expects DAgger round {expected_round}, but correction dataset is round {actual_round}"
        )
    expected_seeds = checkpoint_metadata.get("correction_seeds", [])
    actual_seeds = [row["seed"] for row in corrections.get("correction_report", [])]
    if actual_seeds != expected_seeds:
        raise ValueError("Checkpoint and correction dataset have different correction lineage")


def cloning_loss(trainer, episodes, batch_size, rng, source_boundary=None, sampling_stats=None):
    # Episode-balanced sampling avoids weighting slow successes more heavily.
    selected_indices = [int(rng.integers(len(episodes))) for _ in range(batch_size)]
    selected = [episodes[index] for index in selected_indices]
    if source_boundary is not None and sampling_stats is not None:
        correction_count = sum(index >= source_boundary for index in selected_indices)
        sampling_stats["correction_samples"] = sampling_stats.get("correction_samples", 0) + correction_count
        sampling_stats["original_samples"] = sampling_stats.get("original_samples", 0) + batch_size - correction_count
    rows = [ep[int(rng.integers(len(ep)))] for ep in selected]
    obs = {key: np.stack([r[key] for r in rows]) for key in ("agent_observations", "comm_adjacency")}
    obs["hetero_graph"] = {k: np.stack([r["hetero_graph"][k] for r in rows]) for k in rows[0]["hetero_graph"]}
    vectors, adjacency, graph = observation_tensors(obs, trainer.device, trainer.cfg)
    means, _ = trainer.actor(vectors, adjacency, graph).chunk(2, dim=-1)
    target = torch.as_tensor(np.stack([r["actions"] for r in rows]), dtype=torch.float32, device=trainer.device)
    active = torch.as_tensor(np.stack([r["active"] for r in rows]), dtype=torch.float32, device=trainer.device)
    # Bound saturated teacher references before atanh; regress both action
    # and latent mean so near-limit actions do not lose all gradient.
    target = target.clamp(-0.98, 0.98)
    errors = (means.tanh() - target).square().mean(-1)
    errors += 0.05 * (means - torch.atanh(target)).square().mean(-1)
    loss = (errors * active).sum() / active.sum().clamp_min(1)
    return loss


def pretrain(trainer, episodes, updates, batch_size, seed, source_boundary=None,
             sampling_stats=None):
    rng = np.random.default_rng(seed)
    # Separate optimizer: discard supervised Adam moments before online PPO.
    optimizer = torch.optim.Adam(trainer.actor.mean_actor.parameters(), lr=trainer.cfg.lr)
    trainer.actor.train()
    losses = []
    for update in range(updates):
        loss = cloning_loss(
            trainer, episodes, batch_size, rng,
            source_boundary=source_boundary, sampling_stats=sampling_stats,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.actor.parameters(), trainer.cfg.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 100 == 0 or update + 1 == updates:
            print(f"[behavior-cloning] update={update+1}/{updates} loss={np.mean(losses[-100:]):.5f}", flush=True)
    return losses


def pretrain_mixed(trainer, original_episodes, correction_episodes, updates, batch_size,
                   correction_fraction, seed):
    """Supervise from explicit original/correction sources and report sampling."""
    if not 0.0 <= correction_fraction <= 1.0:
        raise ValueError("dagger correction fraction must be in [0, 1]")
    if not original_episodes or not correction_episodes:
        raise ValueError("explicit DAgger mixing requires both original and correction episodes")
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(trainer.actor.mean_actor.parameters(), lr=trainer.cfg.lr)
    trainer.actor.train()
    losses = []
    sampled_original = sampled_correction = 0
    for update in range(updates):
        correction_count = int(round(batch_size * correction_fraction))
        correction_count = min(max(correction_count, 0), batch_size)
        original_count = batch_size - correction_count
        terms, counts = [], []
        if original_count:
            terms.append(cloning_loss(trainer, original_episodes, original_count, rng))
            counts.append(original_count)
            sampled_original += original_count
        if correction_count:
            terms.append(cloning_loss(trainer, correction_episodes, correction_count, rng))
            counts.append(correction_count)
            sampled_correction += correction_count
        loss = sum(term * count for term, count in zip(terms, counts)) / sum(counts)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.actor.parameters(), trainer.cfg.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 100 == 0 or update + 1 == updates:
            print(f"[dagger-supervision] update={update+1}/{updates} "
                  f"loss={np.mean(losses[-100:]):.5f} correction_fraction="
                  f"{sampled_correction / max(sampled_original + sampled_correction, 1):.3f}", flush=True)
    sampled_total = sampled_original + sampled_correction
    return losses, {
        "original_samples": sampled_original,
        "correction_samples": sampled_correction,
        "actual_correction_fraction": sampled_correction / max(sampled_total, 1),
    }


def imitation_score(metrics):
    return (metrics["capture_rate"], -metrics["mean_censored_steps"], metrics["mean_return"])


def write_dagger_validation(out, records, selected_stage, validation_seeds=None):
    payload = {"selection_rule": ["capture_rate:max", "mean_censored_steps:min", "mean_return:max"],
               "selected_stage": selected_stage,
               "validation_seeds": list(validation_seeds or []),
               "records": records}
    (out / "dagger_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def aggregate_corrections(trainer, env_cfg, dataset, rounds, episodes_per_round, updates,
                          batch_size, first_seed, out, correction_fraction=None,
                          validation_seeds=None, validation_records=None, best_score=None,
                          best_stage="bc"):
    """DAgger-style teacher labels on learner-visited states, not PPO replay.

    Failed rollouts are retained as correction states, never labelled successes.
    Teacher and learner use the same existing observation interface.
    """
    aggregate = list(dataset["episodes"])
    report = list(dataset.get("correction_report", []))
    original_count = len(aggregate) - len(report)
    original_episodes = aggregate[:original_count]
    validation_records = [] if validation_records is None else validation_records
    for iteration in range(rounds):
        beta = 0.5 * (1.0 - iteration / max(rounds - 1, 1))
        for index in range(episodes_per_round):
            seed = first_seed + iteration * episodes_per_round + index
            rng = np.random.default_rng(seed)
            env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
            teacher = BASELINES_3D[dataset["teacher"]]()
            teacher.reset(env)
            obs, trajectory = env.observe_search(), []
            trainer.actor.eval()
            for step in range(env.cfg.search_steps):
                label = teacher.actions(env)
                trajectory.append(snapshot(obs, label, ~env.disabled_uavs))
                # Mix whole-team commands to preserve the teacher's coordination.
                action = label if rng.random() < beta else deterministic_action(trainer, obs)
                result = env.step_joint(action)
                obs = result["obs"]
                if result["terminated"] or result["truncated"]:
                    break
            aggregate.append(trajectory)
            record = {"round": iteration + 1, "seed": seed, "teacher_probability": beta,
                      "mixed_rollout_capture": bool(result.get("capture_success", False)),
                      "steps": len(trajectory)}
            report.append(record)
            print(f"[correction-data] {record}", flush=True)
        correction_episodes = aggregate[original_count:]
        if correction_fraction is None:
            sampling = {"original_samples": 0, "correction_samples": 0}
            pretrain(
                trainer, aggregate, updates, batch_size, first_seed + iteration,
                source_boundary=original_count, sampling_stats=sampling,
            )
            sampled_total = sampling["original_samples"] + sampling["correction_samples"]
            mixing = {
                "mode": "aggregate_uniform",
                **sampling,
                "actual_correction_fraction": sampling["correction_samples"] / max(sampled_total, 1),
            }
            print(f"[dagger-supervision] mode=aggregate_uniform actual_correction_fraction="
                  f"{mixing['actual_correction_fraction']:.3f}", flush=True)
        else:
            _, mixing = pretrain_mixed(
                trainer, original_episodes, correction_episodes, updates, batch_size,
                correction_fraction, first_seed + iteration,
            )
            mixing["mode"] = "explicit_source_fraction"
            mixing["requested_correction_fraction"] = correction_fraction
        trainer.demo_metadata["correction_seeds"] = [r["seed"] for r in report]
        round_number = iteration + 1
        corrections_name = f"corrections_round_{round_number:02d}.pt"
        checkpoint_name = f"corrected_round_{round_number:02d}.pt"
        correction_payload = {
            "env_config": asdict(env_cfg), "teacher": dataset["teacher"],
            "episodes": aggregate, "correction_report": report,
            "original_episode_count": original_count,
            "dagger_round": round_number,
            "sampling": mixing,
            "pursuit_graph_version": PURSUIT_GRAPH_VERSION,
        }
        torch.save(correction_payload, out / corrections_name)
        trainer.demo_metadata.update({
            "initialization_type": "dagger",
            "dagger_round": round_number,
            "correction_dataset": corrections_name,
            "correction_updates": round_number * updates,
            "total_supervised_updates": trainer.demo_metadata.get("bc_updates", 0) + round_number * updates,
            "dagger_sampling": mixing,
        })
        trainer.save_checkpoint(out / checkpoint_name, episode=0)
        if validation_seeds:
            validation = evaluate(trainer, env_cfg, validation_seeds, ["mappo"])
            metrics = validation["summary"]["mappo"]
            validation_records.append({
                "stage": f"dagger_round_{round_number:02d}",
                "checkpoint": checkpoint_name,
                "correction_dataset": corrections_name,
                **metrics,
            })
            score = imitation_score(metrics)
            if best_score is None or score > best_score:
                best_score, best_stage = score, f"dagger_round_{round_number:02d}"
                trainer.save_checkpoint(out / "best_imitation.pt", episode=0)
            write_dagger_validation(out, validation_records, best_stage, validation_seeds)
    return aggregate, validation_records, best_score, best_stage


def evaluate(trainer, env_cfg, seeds, methods):
    rows = []
    trainer.actor.eval()
    for seed in seeds:
        for method in methods:
            env = QuadrotorPursuitEnv(replace(env_cfg, seed=seed))
            policy = None if method == "mappo" else BASELINES_3D[method]()
            if policy is not None:
                policy.reset(env)
            obs, total, components = env.observe_search(), 0.0, {}
            collisions, safety, feasible = 0, 0, 0.0
            correction, emergency = 0., 0.
            visibility = 0.0
            policy_compute_seconds = 0.0
            closest_capture_gap = float(env.minimum_capture_gap())
            visibility_totals = {}
            for step in range(env.cfg.search_steps):
                policy_started = time.perf_counter()
                action = deterministic_action(trainer, obs) if policy is None else policy.actions(env)
                policy_compute_seconds += time.perf_counter() - policy_started
                result = env.step_joint(action)
                obs = result["obs"]
                total += float(result["reward"])
                collisions += int(result.get("collisions", 0))
                safety += int(result.get("continuous_safety_interventions", 0))
                feasible += float(result.get("controller_feasible_rate", 0.0))
                correction += float(result.get('safety_correction_rate', 0.))
                emergency += float(result.get('emergency_stop_rate', 0.))
                visibility += float(result.get("target_visibility_rate", 0.0))
                closest_capture_gap = min(
                    closest_capture_gap, float(result.get("minimum_capture_gap", env.minimum_capture_gap()))
                )
                accumulate_visibility(visibility_totals, result)
                if (step + 1) % 100 == 0:
                    print(f"[evaluation] {method} seed={seed} step={step+1}", flush=True)
                for key, value in result["reward_components"].items():
                    if key in ("pursuit", "estimation"):
                        continue  # Aggregate aliases, not additional rewards.
                    components[key] = components.get(key, 0.0) + float(value)
                if result["terminated"] or result["truncated"]:
                    break
            rows.append({"method": method, "seed": seed, "captured": bool(result["capture_success"]),
                         "steps": step + 1, "return": total, "reward_components": components,
                         "initial_layout": env.initial_layout, "initial_distances": env.initial_distances,
                         "evader_safety_interventions": getattr(env, "evader_safety_interventions", 0),
                         "collisions": collisions, "safety_interventions": safety,
                         "mean_policy_compute_ms": 1000.0 * policy_compute_seconds / (step + 1),
                         "controller_feasible_rate": feasible / (step + 1),
                         "safety_correction_rate": correction / (step + 1),
                         "emergency_stop_rate": emergency / (step + 1),
                         "target_visibility_rate": visibility / (step + 1),
                         "closest_capture_gap": closest_capture_gap,
                         "final_capture_gap": env.minimum_capture_gap(),
                         **finalize_visibility(visibility_totals, step + 1)})
            print(f"[evaluation] {method} seed={seed} captured={rows[-1]['captured']} steps={step+1}", flush=True)
    trainer.actor.train()
    summary = {}
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        successes = [r["steps"] for r in group if r["captured"]]
        summary[method] = {"episodes": len(group), "capture_rate": np.mean([r["captured"] for r in group]).item(),
                           "mean_censored_steps": np.mean([r["steps"] for r in group]).item(),
                           "mean_success_steps": float(np.mean(successes)) if successes else None,
                           "mean_return": np.mean([r["return"] for r in group]).item(),
                           "mean_policy_compute_ms": float(np.mean([r["mean_policy_compute_ms"] for r in group])),
                           "mean_closest_capture_gap": float(np.mean([r["closest_capture_gap"] for r in group])),
                           "mean_final_capture_gap": float(np.mean([r["final_capture_gap"] for r in group])),
                           "mean_collisions": float(np.mean([r["collisions"] for r in group])),
                           "mean_safety_interventions": float(np.mean([r["safety_interventions"] for r in group])),
                           "mean_controller_feasible_rate": float(np.mean([r["controller_feasible_rate"] for r in group])),
                           "mean_safety_correction_rate": float(np.mean([r["safety_correction_rate"] for r in group])),
                           "mean_emergency_stop_rate": float(np.mean([r["emergency_stop_rate"] for r in group])),
                           "mean_team_visibility_ratio": float(np.mean([r["team_visibility_ratio"] for r in group])),
                           "mean_uav_visibility_ratio": float(np.mean([r["uav_visibility_ratio"] for r in group])),
                           "mean_building_occlusion_ratio": float(np.mean([r["building_occlusion_ratio"] for r in group])),
                           "mean_n_uavs_seeing_target": float(np.mean([r["mean_n_uavs_seeing_target"] for r in group])),
                           "mean_target_visibility_rate": float(np.mean([r["target_visibility_rate"] for r in group]))}
    return {"env_config": asdict(env_cfg), "rows": rows, "summary": summary}


def load_environment_config(recorded):
    """Reject old physics/observations even when action dimensions happen to agree."""
    if recorded.get("capture_mode") != "single_distance":
        raise ValueError("Capture rule changed to 3-D distance; recollect demonstrations and retrain")
    if recorded.get("policy_observation_version") != 2:
        raise ValueError("Pursuit observation v2 adds target height/velocity and deadline; recollect demonstrations and retrain into a new directory")
    if recorded.get("scenario_version") != 3:
        raise ValueError("Scenario v3 uses same-side triangle lidar starts; recollect demonstrations and retrain into a new directory")
    if recorded.get("task_mode") != "pursuit_quadrotor_3d" or any(
            key in recorded for key in ("execution_mode", "sensor_mode", "nmpc_horizon")):
        raise ValueError("Old NMPC/FOV demonstrations or checkpoint are incompatible. "
                         "Recollect with the current velocity/MID-360 environment into a new file.")
    reward_version = int(recorded.get("execution_reward_version", -1))
    required_uavs = int(recorded.get("capture_required_uavs", 1))
    expected_reward_version = 5 if required_uavs > 1 else 4
    if reward_version != expected_reward_version:
        raise ValueError(
            f"Capture/reward contract requires execution_reward_version={expected_reward_version}; "
            "recollect demos and retrain in a new directory"
        )
    if recorded.get("pursuit_target_observable", False):
        raise ValueError("Pursuit is limited-FOV MID-360 lidar, not globally observable; recollect demonstrations")
    return QuadrotorPursuitConfig(**recorded)


def load_evaluation_environment_config(path, checkpoint_cfg):
    """Use a new scenario for evaluation only when checkpoint tensor shapes match."""
    evaluation_cfg = QuadrotorPursuitConfig(**json.loads(path.read_text(encoding="utf-8")))
    evaluation_cfg.building_state_capacity = max(
        evaluation_cfg.building_state_capacity, evaluation_cfg.building_count
    )
    contract_fields = (
        "task_mode", "action_mode", "scenario_version",
        "policy_observation_version", "execution_reward_version", "capture_mode",
        "capture_required_uavs",
    )
    changed_contracts = [
        field for field in contract_fields
        if getattr(evaluation_cfg, field) != getattr(checkpoint_cfg, field)
    ]
    if changed_contracts:
        raise ValueError(f"Evaluation environment changes checkpoint contracts: {changed_contracts}")
    checkpoint_env = QuadrotorPursuitEnv(checkpoint_cfg)
    evaluation_env = QuadrotorPursuitEnv(evaluation_cfg)
    checkpoint_shape = (
        checkpoint_cfg.n_uavs, checkpoint_cfg.n_targets,
        checkpoint_env.obs_dim(), checkpoint_env.state_dim(),
        checkpoint_env.continuous_action_dim(),
    )
    evaluation_shape = (
        evaluation_cfg.n_uavs, evaluation_cfg.n_targets,
        evaluation_env.obs_dim(), evaluation_env.state_dim(),
        evaluation_env.continuous_action_dim(),
    )
    if evaluation_shape != checkpoint_shape:
        raise ValueError(
            f"Evaluation environment changes checkpoint input/action dimensions: "
            f"checkpoint={checkpoint_shape}, evaluation={evaluation_shape}"
        )
    return evaluation_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "pretrain", "train", "evaluate"])
    parser.add_argument("--training-profile", choices=["auto", "plain", "warmstart"], default="auto")
    parser.add_argument("--env-config", type=Path, help="JSON object of QuadrotorPursuitConfig overrides")
    parser.add_argument(
        "--mix-env-configs",
        nargs="+",
        type=Path,
        help="Cycle these configs during training; validation/primary uses the last one",
    )
    parser.add_argument("--eval-env-config", type=Path,
                        help="Evaluate a checkpoint in a different, dimension-compatible environment")
    parser.add_argument("--demos", type=artifact_path,
                        default=default_output("pursuit_game_v2_demos.pt"))
    parser.add_argument("--aux-demos", type=artifact_path,
                        help="DAgger corrections.pt used by PPO auxiliary replay")
    parser.add_argument("--no-aux-demos", action="store_true",
                        help="Guarantee that no DAgger correction replay is loaded")
    parser.add_argument("--checkpoint", type=artifact_path)
    parser.add_argument("--out", type=artifact_path,
                        default=default_output("pursuit_game_v2_run"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo-seed", type=int, default=2000000)
    parser.add_argument("--demo-successes", type=int, default=10)
    parser.add_argument("--demo-attempts", type=int, default=100)
    parser.add_argument("--teacher", choices=sorted(BASELINES_3D), default="mpc")
    parser.add_argument("--bc-updates", type=int, default=1000)
    parser.add_argument("--demo-weight", type=float, help="Initial auxiliary imitation weight")
    parser.add_argument("--demo-weight-floor", type=float,
                        help="Nonzero anti-forgetting imitation floor")
    parser.add_argument("--correction-fraction", type=float, default=0.5,
                        help="Fraction of auxiliary batches sampled from learner-visited DAgger corrections")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4, help="Deprecated shared LR fallback")
    parser.add_argument("--actor-lr", type=float)
    parser.add_argument("--critic-lr", type=float)
    parser.add_argument("--update-epochs", type=int)
    parser.add_argument("--target-kl", type=float)
    parser.add_argument("--clip-eps", type=float)
    parser.add_argument("--initial-policy-std", type=float)
    parser.add_argument("--max-policy-std", type=float)
    parser.add_argument("--reference-kl-weight", type=float)
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1, help="Small graph batches usually benefit from a single CPU thread")
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--validation-seed", type=int, default=4000000)
    parser.add_argument("--train-seeds-file", type=artifact_path,
                        help="MPC-solvable training seeds, one integer per line")
    parser.add_argument("--validation-seeds-file", type=artifact_path,
                        help="Held-out validation seeds, one integer per line")
    parser.add_argument("--safeguard-drop", type=float, default=0.10)
    parser.add_argument("--safeguard-patience", type=int, default=2)
    parser.add_argument("--min-actor-lr", type=float, default=1e-5)
    parser.add_argument("--eval-seed", type=int, default=3000000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds-file", type=artifact_path,
                        help="Held-out evaluation seeds, one integer per line")
    parser.add_argument("--methods", nargs="+", choices=["mappo", *BASELINES_3D], default=["mappo", "apf", "frpn", "mpc"])
    parser.add_argument("--correction-rounds", type=int, default=0)
    parser.add_argument("--correction-episodes", type=int, default=5)
    parser.add_argument("--correction-updates", type=int, default=500)
    parser.add_argument("--correction-seed", type=int, default=5000000)
    parser.add_argument("--dagger-correction-fraction", type=float,
                        help="Explicit fraction of DAgger supervised samples from corrections; "
                             "omit for legacy uniform aggregate sampling")
    parser.add_argument("--dagger-validation-seed", type=int)
    parser.add_argument("--dagger-validation-episodes", type=int)
    args = parser.parse_args()
    if args.eval_env_config and args.mode != "evaluate":
        parser.error("--eval-env-config is only valid for evaluate")
    try:
        explicit_aux_path = resolve_auxiliary_path(args.aux_demos, args.no_aux_demos)
        requested_train_bank = load_seed_bank(args.train_seeds_file)
        requested_validation_bank = load_seed_bank(args.validation_seeds_file)
        requested_eval_bank = load_seed_bank(args.eval_seeds_file)
    except ValueError as error:
        parser.error(str(error))
    preview = (torch.load(args.checkpoint, map_location="cpu", weights_only=False)
               if args.checkpoint else {})
    checkpoint_warmstart = preview.get("demo_metadata", {}).get("successful_episodes", 0) > 0
    resolved_profile = args.training_profile
    if resolved_profile == "auto":
        resolved_profile = "warmstart" if (
            checkpoint_warmstart or (not args.checkpoint and args.bc_updates > 0 and args.mode in ("pretrain", "train"))
        ) else "plain"
    profile = {
        "plain": dict(actor_lr=1e-4, critic_lr=1e-4, update_epochs=3, target_kl=0.01,
                      clip_eps=0.10, initial_policy_std=0.20, max_policy_std=0.40,
                      demo_weight=0.0, demo_weight_floor=0.0, reference_kl_weight=0.0),
        "warmstart": dict(actor_lr=5e-5, critic_lr=1e-4, update_epochs=2, target_kl=0.0075,
                          clip_eps=0.10, initial_policy_std=0.08, max_policy_std=0.20,
                          demo_weight=0.03, demo_weight_floor=0.01, reference_kl_weight=0.05),
    }[resolved_profile]
    for name, value in profile.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    args.training_profile = resolved_profile
    positive = (args.lr, args.actor_lr, args.critic_lr, args.target_kl,
                args.clip_eps, args.initial_policy_std, args.max_policy_std, args.min_actor_lr)
    nonnegative = (args.demo_weight, args.demo_weight_floor, args.reference_kl_weight, args.safeguard_drop)
    if not all(math.isfinite(value) and value > 0 for value in positive) or args.update_epochs < 1:
        parser.error('learning rates, PPO bounds, policy stds, target-kl and update-epochs must be positive and finite')
    if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
        parser.error('anti-forgetting weights and safeguard-drop must be finite and nonnegative')
    if not 0 <= args.demo_weight_floor <= args.demo_weight or not 0 <= args.correction_fraction <= 1:
        parser.error('Require 0 <= demo-weight-floor <= demo-weight and correction-fraction in [0, 1]')
    if (args.dagger_correction_fraction is not None
            and not 0 <= args.dagger_correction_fraction <= 1):
        parser.error("--dagger-correction-fraction must be in [0, 1]")
    if not 0.05 < args.initial_policy_std < args.max_policy_std or args.safeguard_patience < 1:
        parser.error('Require 0.05 < initial-policy-std < max-policy-std and positive safeguard-patience')
    if args.mode == 'train' and not args.checkpoint and any((args.out/name).exists() for name in ('latest.pt', 'final.pt', 'history.json')):
        parser.error('Output already contains a run; choose a new --out directory or explicitly resume a v3 checkpoint')
    if args.correction_rounds < 0 or min(args.correction_episodes, args.correction_updates) < 1:
        parser.error("Invalid correction counts")
    if args.dagger_validation_episodes is not None and args.dagger_validation_episodes < 1:
        parser.error("--dagger-validation-episodes must be positive")
    if args.correction_rounds and args.mode != "pretrain":
        parser.error("Run corrections using pretrain, then evaluate before PPO")
    if args.mode == "pretrain" and args.checkpoint:
        parser.error("pretrain starts from demonstrations; omit --checkpoint")
    if min(args.episodes, args.batch_size, args.demo_successes, args.demo_attempts, args.eval_episodes) < 1 or args.bc_updates < 0:
        parser.error("Counts must be positive (bc-updates may be zero for ablation)")
    seed_everything(args.seed)
    torch.set_num_threads(max(1, args.torch_threads))
    if args.mix_env_configs and args.mode == "evaluate":
        parser.error("--mix-env-configs is only valid for train/collect/pretrain")
    if args.mix_env_configs:
        if args.env_config:
            parser.error("Use either --env-config or --mix-env-configs, not both")
        mix_paths = list(args.mix_env_configs)
        mix_cfgs = [load_pursuit_config(path) for path in mix_paths]
        env_cfg = mix_cfgs[-1]
    else:
        mix_paths = []
        mix_cfgs = []
        env_cfg = QuadrotorPursuitConfig(**(json.loads(args.env_config.read_text(encoding="utf-8")) if args.env_config else {}))
        env_cfg.building_state_capacity = max(env_cfg.building_state_capacity, env_cfg.building_count)
    if env_cfg.search_steps < 1:
        parser.error("search_steps must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "collect":
        dataset = collect_demos(env_cfg, args.demo_seed, args.demo_successes, args.demo_attempts, args.teacher, args.demos)
        args.demos.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.demos)
        args.demos.with_suffix(".json").write_text(json.dumps({k: v for k, v in dataset.items() if k != "episodes"}, indent=2), encoding="utf-8")
        return
    cfg = TrainConfig(episodes=args.episodes, gamma=env_cfg.reward_gamma, gae_lambda=0.97,
                      hidden_dim=args.hidden_dim, batch_size=args.batch_size, device=args.device,
                      lr=args.lr, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                      update_epochs=args.update_epochs, target_kl=args.target_kl, clip_eps=args.clip_eps,
                      continuous_initial_std=args.initial_policy_std,
                      continuous_log_std_max=math.log(args.max_policy_std),
                      entropy_coef=0.003, entropy_coef_end=0.001, adaptive_entropy=False,
                      use_gat=not args.no_gat, use_hetero_entities=not args.no_gat,
                      checkpoint_path=str(args.out / "latest.pt"))
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if mix_cfgs:
            if args.env_config:
                parser.error("Mixed training already defines the environment; omit --env-config")
        else:
            env_cfg = load_environment_config(payload["env_config"])
            if args.env_config:
                parser.error("Checkpoint already defines the environment; omit --env-config")
        cfg = TrainConfig(**payload["train_config"])
        cfg.device = args.device
        if args.mode == "train":
            cfg.episodes = args.episodes
            cfg.checkpoint_path = str(args.out / "latest.pt")
            cfg.actor_lr = args.actor_lr
            cfg.critic_lr = args.critic_lr
            cfg.update_epochs = args.update_epochs
            cfg.target_kl = args.target_kl
            cfg.clip_eps = args.clip_eps
            cfg.continuous_initial_std = args.initial_policy_std
            cfg.continuous_log_std_max = math.log(args.max_policy_std)
    elif args.mode == "evaluate":
        parser.error("evaluate requires --checkpoint")
    recorded_train_bank = list(preview.get("train_seed_bank", []))
    if (args.checkpoint and args.mode == "train" and int(preview.get("episode", 0)) > 0
            and requested_train_bank and recorded_train_bank != requested_train_bank):
        parser.error("Resume requires the same --train-seeds-file stored in the checkpoint")
    train_seed_bank = requested_train_bank or (recorded_train_bank if args.mode == "train" else [])
    if mix_cfgs:
        factory, primary_cfg = make_pursuit_env_factory(
            mix_cfgs,
            seed_fn=lambda episode: seed_for_episode(train_seed_bank, episode, args.seed),
            primary_index=-1,
        )
        env_cfg = primary_cfg
    else:
        factory = lambda episode=0: QuadrotorPursuitEnv(
            replace(env_cfg, seed=seed_for_episode(train_seed_bank, episode, args.seed))
        )
    trainer = PursuitDemoTrainer(factory, cfg)
    if mix_paths:
        trainer.demo_metadata.update(mix_manifest(mix_paths, env_cfg))
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
        if args.mode == "train" and trainer.start_episode == 0 and trainer.reference_actor is None:
            # BC/DAgger checkpoints contain an untrained exploration parameter.
            trainer.actor.set_std(args.initial_policy_std)
            trainer.reset_actor_optimizer()
    if args.mode == "evaluate":
        seeds = requested_eval_bank or list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
        metadata = getattr(trainer, "demo_metadata", {})
        used = set(metadata.get("demo_seeds", []))
        used.update(metadata.get("correction_seeds", []))
        used.update(metadata.get("training_seeds", []))
        used.update(metadata.get("validation_seeds", []))
        if used.intersection(seeds):
            parser.error("Evaluation seeds overlap demonstration or training seeds")
        evaluation_cfg = (
            load_evaluation_environment_config(args.eval_env_config, env_cfg)
            if args.eval_env_config else env_cfg
        )
        result = evaluate(trainer, evaluation_cfg, seeds, args.methods)
        (args.out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        write_evaluation_chart(args.out / "evaluation.svg", result)
        print(json.dumps(result["summary"], indent=2))
        return
    if not args.checkpoint and args.bc_updates:
        dataset = torch.load(args.demos, map_location="cpu", weights_only=False)
        validate_demos(dataset, env_cfg)
        trainer.demo_metadata = {
            "teacher": dataset["teacher"],
            "demo_seeds": [r["seed"] for r in dataset["report"]],
            "successful_episodes": len(dataset["episodes"]),
            "bc_updates": args.bc_updates,
            "correction_updates": 0,
            "total_supervised_updates": args.bc_updates,
            "initialization_type": "bc",
            "auxiliary_source": "none",
            "dagger_round": 0,
        }
        pretrain_dataset = dataset
        if explicit_aux_path is not None:
            try:
                pretrain_dataset, _ = load_correction_replay(explicit_aux_path, dataset, env_cfg)
            except ValueError as error:
                parser.error(str(error))
        # DAgger is round 1 onward. Round 0 is always a clean BC control using
        # only successful original demonstrations.
        losses = pretrain(trainer, dataset["episodes"], args.bc_updates, args.batch_size, args.seed)
        (args.out / "bc_losses.json").write_text(json.dumps(losses), encoding="utf-8")
        trainer.save_checkpoint(args.out / "pretrained.pt", episode=0)
    if args.mode == "pretrain":
        if not args.bc_updates:
            parser.error("pretrain requires positive --bc-updates")
        if (requested_validation_bank and args.dagger_validation_seed is None
                and args.dagger_validation_episodes is None):
            dagger_validation_seeds = list(requested_validation_bank)
        else:
            dagger_validation_seed = (args.validation_seed if args.dagger_validation_seed is None
                                      else args.dagger_validation_seed)
            dagger_validation_episodes = (args.validation_episodes if args.dagger_validation_episodes is None
                                          else args.dagger_validation_episodes)
            dagger_validation_seeds = list(range(
                dagger_validation_seed, dagger_validation_seed + dagger_validation_episodes
            ))
        trainer.demo_metadata["dagger_validation_seeds"] = dagger_validation_seeds
        # Save again so round-0 lineage includes the dedicated validation bank.
        trainer.save_checkpoint(args.out / "pretrained.pt", episode=0)
        correction_seeds = set(range(
            args.correction_seed,
            args.correction_seed + args.correction_rounds * args.correction_episodes,
        ))
        occupied = set(trainer.demo_metadata.get("demo_seeds", []))
        occupied.update(row["seed"] for row in pretrain_dataset.get("correction_report", []))
        occupied.update(requested_train_bank)
        if correction_seeds.intersection(occupied):
            parser.error("Correction seeds must differ from original demonstration and prior correction seeds")
        if set(dagger_validation_seeds).intersection(occupied | correction_seeds):
            parser.error("DAgger validation seeds overlap demonstrations or correction rounds")
        bc_validation = evaluate(trainer, env_cfg, dagger_validation_seeds, ["mappo"])
        bc_metrics = bc_validation["summary"]["mappo"]
        validation_records = [{
            "stage": "bc", "checkpoint": "pretrained.pt", "correction_dataset": None,
            **bc_metrics,
        }]
        best_score, best_stage = imitation_score(bc_metrics), "bc"
        trainer.save_checkpoint(args.out / "best_imitation.pt", episode=0)
        write_dagger_validation(args.out, validation_records, best_stage, dagger_validation_seeds)
        if args.correction_rounds:
            _, validation_records, best_score, best_stage = aggregate_corrections(
                trainer, env_cfg, pretrain_dataset, args.correction_rounds,
                args.correction_episodes, args.correction_updates,
                args.batch_size, args.correction_seed, args.out,
                correction_fraction=args.dagger_correction_fraction,
                validation_seeds=dagger_validation_seeds,
                validation_records=validation_records,
                best_score=best_score,
                best_stage=best_stage,
            )
        print(f"Initialization complete. Validation selected best_imitation.pt from {best_stage}.")
        return
    if args.demo_weight > 0 and trainer.demo_metadata.get("successful_episodes", 0):
        dataset = torch.load(args.demos, map_location="cpu", weights_only=False)
        validate_demos(dataset, env_cfg)
        if [r["seed"] for r in dataset["report"]] != trainer.demo_metadata["demo_seeds"]:
            parser.error("Resume requires the original demonstration dataset")
        trainer.demo_episodes = dataset["episodes"]
        aux_path = explicit_aux_path
        if aux_path is not None:
            try:
                corrections, trainer.correction_episodes = load_correction_replay(aux_path, dataset, env_cfg)
                validate_correction_checkpoint_lineage(trainer.demo_metadata, corrections)
            except ValueError as error:
                parser.error(str(error))
            if not trainer.correction_episodes:
                parser.error("--aux-demos does not contain learner-visited correction episodes")
            trainer.demo_metadata["aux_demos"] = str(aux_path)
            trainer.demo_metadata["aux_correction_episodes"] = len(trainer.correction_episodes)
    trainer.demo_weight = args.demo_weight
    trainer.demo_weight_floor = args.demo_weight_floor
    trainer.correction_fraction = args.correction_fraction
    has_warm_start = trainer.demo_metadata.get("successful_episodes", 0) > 0
    trainer.reference_kl_weight = args.reference_kl_weight if has_warm_start else 0.0
    trainer.safeguard_drop = args.safeguard_drop
    trainer.safeguard_patience = args.safeguard_patience
    trainer.min_actor_lr = args.min_actor_lr
    trainer.demo_metadata["demo_weight"] = args.demo_weight
    trainer.demo_metadata["demo_weight_floor"] = args.demo_weight_floor
    trainer.demo_metadata["correction_fraction"] = args.correction_fraction
    trainer.demo_metadata["reference_kl_weight"] = args.reference_kl_weight
    trainer.demo_metadata["training_profile"] = args.training_profile
    prior_initialization = trainer.demo_metadata.get("initialization_type")
    if prior_initialization not in ("bc", "dagger"):
        if trainer.demo_metadata.get("correction_seeds"):
            prior_initialization = "dagger"
        elif trainer.demo_metadata.get("successful_episodes", 0):
            prior_initialization = "bc"
        else:
            prior_initialization = "scratch"
    trainer.demo_metadata["initialization_type"] = prior_initialization
    trainer.demo_metadata["auxiliary_source"] = (
        "dagger_corrections" if trainer.correction_episodes
        else "original_demo" if trainer.demo_episodes
        else "none"
    )
    if args.mode == "train" and trainer.reference_kl_weight > 0 and trainer.reference_actor is None:
        trainer.capture_reference_policy()
    if args.checkpoint and trainer.demo_metadata.get("training_base_seed", args.seed) != args.seed:
        parser.error("Resume requires the original --seed")
    trainer.demo_metadata["training_base_seed"] = args.seed
    trainer.train_seed_bank = list(train_seed_bank)
    trainer.seed_bank_version = SEED_BANK_VERSION
    new_seeds = [
        seed_for_episode(train_seed_bank, episode, args.seed)
        for episode in range(trainer.start_episode, cfg.episodes)
    ]
    if set(new_seeds).intersection(trainer.demo_metadata.get("demo_seeds", []) + trainer.demo_metadata.get("correction_seeds", [])):
        parser.error("Online training seeds overlap demonstration seeds")
    trainer.demo_metadata["training_seeds"] = sorted(set(trainer.demo_metadata.get("training_seeds", []) + new_seeds))
    trainer.output_directory = args.out
    if args.validation_interval > 0 and args.validation_episodes > 0:
        trainer.validation_interval = args.validation_interval
        recorded_validation_bank = list(preview.get("validation_seed_bank", []))
        if (args.checkpoint and int(preview.get("episode", 0)) > 0 and requested_validation_bank
                and recorded_validation_bank != requested_validation_bank):
            parser.error("Resume requires the same --validation-seeds-file stored in the checkpoint")
        trainer.validation_seeds = (
            requested_validation_bank or recorded_validation_bank
            or list(range(args.validation_seed, args.validation_seed + args.validation_episodes))
        )
        trainer.validation_seed_bank = list(trainer.validation_seeds) if (
            requested_validation_bank or recorded_validation_bank
        ) else []
        try:
            ensure_disjoint_seed_banks({
                "training": trainer.demo_metadata["training_seeds"],
                "validation": trainer.validation_seeds,
            })
        except ValueError as error:
            parser.error(str(error))
        used = set(trainer.demo_metadata.get("demo_seeds", []) + trainer.demo_metadata.get("correction_seeds", []) + trainer.demo_metadata["training_seeds"])
        if used.intersection(trainer.validation_seeds):
            parser.error("Validation seeds overlap training or demonstrations")
        prior_validation = trainer.demo_metadata.get("validation_seeds", [])
        if args.checkpoint and prior_validation != trainer.validation_seeds:
            trainer.best_capture_score = (-1.0, -float("inf"), -float("inf"))
        trainer.demo_metadata["validation_seeds"] = sorted(set(prior_validation + trainer.validation_seeds))
        trainer.validate_capture(trainer.start_episode)
    history = trainer.train()
    if trainer.validation_seeds and cfg.episodes % trainer.validation_interval:
        trainer.validate_capture(cfg.episodes)
    trainer.save_checkpoint(args.out / "final.pt", episode=cfg.episodes, history=history)
    write_pursuit_outputs(args.out, history, trainer.episode_traces)


if __name__ == "__main__":
    main()
