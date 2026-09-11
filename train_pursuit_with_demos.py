"""Pursuit-only demonstration warm start and on-policy MAPPO fine tuning.

Successful teacher trajectories provide optional behavior-cloning initialization.
Teacher transitions are used only for supervised learning, never as PPO data.
The search environment and discrete search training path are left unchanged.
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
    tensorize_heterogeneous_graph,
)
from pursuit_graph_encoder import (
    PURSUIT_GRAPH_VERSION, PursuitGraphActor, pursuit_graph_from_flat_observation,
)
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_training_output import clean_history_row, write_pursuit_outputs, write_evaluation_chart, write_reward_capture_chart
from pursuit_ppo import update_pursuit


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
        if not isinstance(probe, QuadrotorPursuitEnv):
            raise ValueError("This trainer is restricted to quadrotor pursuit")
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
        self.min_actor_lr = 1e-5
        self.output_directory = None
        self.train_seed_bank = []
        self.validation_seed_bank = []
        self.seed_bank_version = SEED_BANK_VERSION

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
        original_step = self.active_env.step_joint
        def timed_step(actions):
            started = time.perf_counter()
            result = original_step(actions)
            self.environment_seconds += time.perf_counter() - started
            for key, value in result.get("reward_components", {}).items():
                if key not in ("pursuit", "estimation"):
                    self.reward_totals[key] = self.reward_totals.get(key, 0.0) + float(value)
            self.closest_capture_gap = min(self.closest_capture_gap, result.get("minimum_capture_gap", float("inf")))
            self.controller_rates.append(result.get("controller_feasible_rate", 0.0))
            for key in self.safety_metrics:
                self.safety_metrics[key].append(result[key])
            for reasons in result['safety_reasons']:
                for reason in reasons:
                    self.safety_reason_counts[reason] = self.safety_reason_counts.get(reason, 0)+1
            if "lidar_detection_ratio" in result:
                self.lidar_rates.append(result["lidar_detection_ratio"])
            return result
        self.active_env.step_joint = timed_step
        return self.active_env

    def _update(self, rollout, episode):
        started = time.perf_counter()
        result = update_pursuit(self, rollout, episode)
        result["ppo_update_seconds"] = time.perf_counter() - started
        return result

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
        row["minimum_capture_gap"] = self.active_env.minimum_capture_gap()
        row["demo_loss_weight"] = self.demo_loss_weight(row["episode"] - 1)
        row["reference_kl_weight"] = self.reference_kl_weight if self.reference_actor is not None else 0.0
        row["safeguard_rollbacks"] = self.safeguard_rollbacks
        row["reward_components"] = dict(self.reward_totals)
        row["closest_capture_gap"] = self.closest_capture_gap
        row["disabled_uavs_final"] = int(np.count_nonzero(self.active_env.disabled_uavs))
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
def deterministic_action(trainer, obs):
    vectors, adjacency, graph = observation_tensors(obs, trainer.device, trainer.cfg)
    means, _ = trainer.actor(vectors, adjacency, graph).chunk(2, dim=-1)
    return means.tanh().cpu().numpy()


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
            closest_capture_gap = float(env.minimum_capture_gap())
            for step in range(env.cfg.search_steps):
                action = deterministic_action(trainer, obs) if policy is None else policy.actions(env)
                result = env.step_joint(action)
                obs = result["obs"]
                total += float(result["reward"])
                collisions += int(result.get("collisions", 0))
                safety += int(result.get("continuous_safety_interventions", 0))
                feasible += float(result.get("controller_feasible_rate", 0.0))
                correction += float(result.get('safety_correction_rate', 0.))
                emergency += float(result.get('emergency_stop_rate', 0.))
                closest_capture_gap = min(
                    closest_capture_gap, float(result.get("minimum_capture_gap", env.minimum_capture_gap()))
                )
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
                         "controller_feasible_rate": feasible / (step + 1),
                         "safety_correction_rate": correction / (step + 1),
                         "emergency_stop_rate": emergency / (step + 1),
                         "closest_capture_gap": closest_capture_gap,
                         "final_capture_gap": env.minimum_capture_gap()})
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
                           "mean_closest_capture_gap": float(np.mean([r["closest_capture_gap"] for r in group])),
                           "mean_final_capture_gap": float(np.mean([r["final_capture_gap"] for r in group])),
                           "mean_collisions": float(np.mean([r["collisions"] for r in group])),
                           "mean_safety_interventions": float(np.mean([r["safety_interventions"] for r in group])),
                           "mean_controller_feasible_rate": float(np.mean([r["controller_feasible_rate"] for r in group])),
                           "mean_safety_correction_rate": float(np.mean([r["safety_correction_rate"] for r in group])),
                           "mean_emergency_stop_rate": float(np.mean([r["emergency_stop_rate"] for r in group]))}
    return {"env_config": asdict(env_cfg), "rows": rows, "summary": summary}


def load_environment_config(recorded):
    """Reject old physics/observations even when action dimensions happen to agree."""
    if recorded.get("capture_mode") != "single_distance":
        raise ValueError("Capture rule changed to 3-D distance; recollect demonstrations and retrain")
    if recorded.get("policy_observation_version") != 2:
        raise ValueError("Pursuit observation v2 adds target height/velocity and deadline; recollect demonstrations and retrain into a new directory")
    if recorded.get("scenario_version") != 2:
        raise ValueError("Scenario v2 changes initialization and evader dynamics; recollect demonstrations and retrain into a new directory")
    if recorded.get("task_mode") != "pursuit_quadrotor_3d" or any(
            key in recorded for key in ("execution_mode", "sensor_mode", "nmpc_horizon")):
        raise ValueError("Old NMPC/FOV demonstrations or checkpoint are incompatible. "
                         "Recollect with the current velocity/MID-360 environment into a new file.")
    if recorded.get('execution_reward_version') != 3:
        raise ValueError('Safety execution and rewards changed to v3; recollect demos and retrain in a new directory')
    return QuadrotorPursuitConfig(**recorded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["collect", "pretrain", "train", "evaluate"])
    parser.add_argument("--training-profile", choices=["auto", "plain", "warmstart"], default="auto")
    parser.add_argument("--env-config", type=Path, help="JSON object of QuadrotorPursuitConfig overrides")
    parser.add_argument("--demos", type=Path, default=Path("outputs/pursuit_game_v2_demos.pt"))
    parser.add_argument("--aux-demos", type=Path, help="DAgger corrections.pt used by PPO auxiliary replay")
    parser.add_argument("--no-aux-demos", action="store_true",
                        help="Guarantee that no DAgger correction replay is loaded")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/pursuit_game_v2_run"))
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
    parser.add_argument("--train-seeds-file", type=Path,
                        help="MPC-solvable training seeds, one integer per line")
    parser.add_argument("--validation-seeds-file", type=Path,
                        help="Held-out validation seeds, one integer per line")
    parser.add_argument("--safeguard-drop", type=float, default=0.10)
    parser.add_argument("--safeguard-patience", type=int, default=2)
    parser.add_argument("--min-actor-lr", type=float, default=1e-5)
    parser.add_argument("--eval-seed", type=int, default=3000000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seeds-file", type=Path,
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
    env_cfg = QuadrotorPursuitConfig(**(json.loads(args.env_config.read_text(encoding="utf-8")) if args.env_config else {}))
    if env_cfg.search_steps < 1:
        parser.error("search_steps must be positive")
    env_cfg.building_state_capacity = max(env_cfg.building_state_capacity, env_cfg.building_count)
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
    factory = lambda episode=0: QuadrotorPursuitEnv(
        replace(env_cfg, seed=seed_for_episode(train_seed_bank, episode, args.seed))
    )
    trainer = PursuitDemoTrainer(factory, cfg)
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
        result = evaluate(trainer, env_cfg, seeds, args.methods)
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
