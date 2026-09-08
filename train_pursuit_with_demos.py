"""Pursuit-only demonstration warm start and on-policy MAPPO fine tuning.

Successful teacher trajectories provide optional behavior-cloning initialization.
Teacher transitions are used only for supervised learning, never as PPO data.
The search environment and discrete search training path are left unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from marl_trainers import (
    MAPPOTrainer, TrainConfig, GraphActor, FlatMLP, seed_everything,
    tensorize_heterogeneous_graph,
)
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_training_output import clean_history_row, write_pursuit_outputs, write_evaluation_chart, write_reward_capture_chart
from pursuit_ppo import update_pursuit


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
            GraphActor(self.obs_dim, self.action_dim, cfg.hidden_dim, cfg.gat_heads,
                       cfg.gat_layers, cfg.dropout)
            if cfg.use_gat else FlatMLP(self.obs_dim, self.action_dim, cfg.hidden_dim)
        )
        # Official MAPPO uses a small (0.01) orthogonal action-head gain.
        # A default random head on LayerNorm features created a large initial
        # common-direction bias in the failed run.
        output_head = mean_actor.head if cfg.use_gat else mean_actor.net[-1]
        nn.init.orthogonal_(output_head.weight, gain=0.01)
        nn.init.zeros_(output_head.bias)
        self.actor = PursuitActor(mean_actor, self.action_dim, cfg.continuous_log_std_max).to(self.device)
        self.actors = nn.ModuleList([self.actor])
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr, eps=1e-5)
        self.critic = nn.Sequential(nn.LayerNorm(self.state_dim), self.critic).to(self.device)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr, eps=1e-5)
        self.algorithm_name = "mappo_pursuit_safe_v3"
        self.environment_config = asdict(probe.cfg)
        self.demo_metadata = {}
        self.demo_episodes = []
        self.demo_rng = np.random.default_rng(0)
        self.demo_weight = 0.1
        self.validation_seeds = []
        self.validation_interval = 50
        self.validation_records = []
        self.pursuit_history = []
        self.best_capture_score = (-1.0, -float("inf"), -float("inf"))
        self.output_directory = None

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
        row["demo_loss_weight"] = self.demo_weight * max(0.0, 1.0 - (row["episode"] - 1) / max(self.cfg.episodes // 2, 1))
        if not self.demo_episodes:
            row["demo_loss_weight"] = 0.0
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
            self.save_checkpoint(self.output_directory / "best_capture.pt", episode=episode, history=self.pursuit_history)
        (self.output_directory / "validation.json").write_text(json.dumps(self.validation_records, indent=2), encoding="utf-8")

    def pursuit_demo_loss(self, episode):
        weight = self.demo_weight * max(0.0, 1.0 - episode / max(self.cfg.episodes // 2, 1))
        if not self.demo_episodes or weight == 0:
            return 0.0
        return weight * cloning_loss(self, self.demo_episodes, min(self.cfg.batch_size, 32), self.demo_rng)

    def save_checkpoint(self, path, env_config=None, episode=None, history=None):
        super().save_checkpoint(path, env_config or self.environment_config, episode,
                                [clean_history_row(row) for row in (history or [])])
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["demo_metadata"] = self.demo_metadata
        payload["best_capture_score"] = self.best_capture_score
        payload["validation_records"] = self.validation_records
        torch.save(payload, path)

    def load_checkpoint(self, path):
        recorded = torch.load(path, map_location='cpu', weights_only=False)
        load_environment_config(recorded['env_config'])
        payload = super().load_checkpoint(path)
        self.demo_metadata = payload.get("demo_metadata", {})
        self.best_capture_score = tuple(payload.get("best_capture_score", self.best_capture_score))
        self.validation_records = payload.get("validation_records", [])
        self.pursuit_history = list(self.restored_history)
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
            partial = {"env_config": asdict(env_cfg), "teacher": teacher_name, "episodes": episodes, "report": report}
            torch.save(partial, progress_path)
            progress_path.with_suffix(".json").write_text(json.dumps({k: v for k, v in partial.items() if k != "episodes"}, indent=2), encoding="utf-8")
        print(f"[demonstrations] seed={seed} success={success} kept={len(episodes)}/{wanted}", flush=True)
        if len(episodes) >= wanted:
            break
    if len(episodes) < wanted:
        raise RuntimeError(f"Only {len(episodes)}/{wanted} successful demonstrations in {attempts} attempts; "
                           "increase --demo-attempts or inspect the teacher under this configuration")
    return {"env_config": asdict(env_cfg), "teacher": teacher_name, "episodes": episodes, "report": report}


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


def cloning_loss(trainer, episodes, batch_size, rng):
    # Episode-balanced sampling avoids weighting slow successes more heavily.
    selected = [episodes[int(rng.integers(len(episodes)))] for _ in range(batch_size)]
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


def pretrain(trainer, episodes, updates, batch_size, seed):
    rng = np.random.default_rng(seed)
    # Separate optimizer: discard supervised Adam moments before online PPO.
    optimizer = torch.optim.Adam(trainer.actor.mean_actor.parameters(), lr=trainer.cfg.lr)
    trainer.actor.train()
    losses = []
    for update in range(updates):
        loss = cloning_loss(trainer, episodes, batch_size, rng)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainer.actor.parameters(), trainer.cfg.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update + 1) % 100 == 0 or update + 1 == updates:
            print(f"[behavior-cloning] update={update+1}/{updates} loss={np.mean(losses[-100:]):.5f}", flush=True)
    return losses


def aggregate_corrections(trainer, env_cfg, dataset, rounds, episodes_per_round, updates, batch_size, first_seed, out):
    """DAgger-style teacher labels on learner-visited states, not PPO replay.

    Failed rollouts are retained as correction states, never labelled successes.
    Teacher and learner use the same existing observation interface.
    """
    aggregate = list(dataset["episodes"])
    report = []
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
        pretrain(trainer, aggregate, updates, batch_size, first_seed + iteration)
        # Persist each completed round; these are supervised correction labels.
        torch.save({"env_config": asdict(env_cfg), "teacher": dataset["teacher"],
                    "episodes": aggregate, "correction_report": report}, out / "corrections.pt")
        trainer.demo_metadata["correction_seeds"] = [r["seed"] for r in report]
        trainer.save_checkpoint(out / "corrected.pt", episode=0)
    return aggregate


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
    parser.add_argument("--env-config", type=Path, help="JSON object of QuadrotorPursuitConfig overrides")
    parser.add_argument("--demos", type=Path, default=Path("outputs/pursuit_game_v2_demos.pt"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("outputs/pursuit_game_v2_run"))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo-seed", type=int, default=2000000)
    parser.add_argument("--demo-successes", type=int, default=10)
    parser.add_argument("--demo-attempts", type=int, default=100)
    parser.add_argument("--teacher", choices=sorted(BASELINES_3D), default="mpc")
    parser.add_argument("--bc-updates", type=int, default=1000)
    parser.add_argument("--demo-weight", type=float, default=0.1, help="Auxiliary BC weight, decays to zero by halfway through RL")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--no-gat", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1, help="Small graph batches usually benefit from a single CPU thread")
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--validation-seed", type=int, default=4000000)
    parser.add_argument("--eval-seed", type=int, default=3000000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--methods", nargs="+", choices=["mappo", *BASELINES_3D], default=["mappo", "apf", "frpn", "mpc"])
    parser.add_argument("--correction-rounds", type=int, default=0)
    parser.add_argument("--correction-episodes", type=int, default=5)
    parser.add_argument("--correction-updates", type=int, default=500)
    parser.add_argument("--correction-seed", type=int, default=5000000)
    args = parser.parse_args()
    if not (math.isfinite(args.lr) and args.lr > 0 and math.isfinite(args.target_kl) and args.target_kl > 0 and args.update_epochs > 0):
        parser.error('lr, target-kl and update-epochs must be positive and finite')
    if args.mode == 'train' and not args.checkpoint and any((args.out/name).exists() for name in ('latest.pt', 'final.pt', 'history.json')):
        parser.error('Output already contains a run; choose a new --out directory or explicitly resume a v3 checkpoint')
    if args.correction_rounds < 0 or min(args.correction_episodes, args.correction_updates) < 1:
        parser.error("Invalid correction counts")
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
                      lr=args.lr, update_epochs=args.update_epochs, target_kl=args.target_kl, clip_eps=0.1,
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
    elif args.mode == "evaluate":
        parser.error("evaluate requires --checkpoint")
    factory = lambda episode=0: QuadrotorPursuitEnv(replace(env_cfg, seed=args.seed * 100000 + episode))
    trainer = PursuitDemoTrainer(factory, cfg)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    if args.mode == "evaluate":
        seeds = list(range(args.eval_seed, args.eval_seed + args.eval_episodes))
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
        trainer.demo_metadata = {"teacher": dataset["teacher"], "demo_seeds": [r["seed"] for r in dataset["report"]],
                                 "successful_episodes": len(dataset["episodes"]), "bc_updates": args.bc_updates}
        losses = pretrain(trainer, dataset["episodes"], args.bc_updates, args.batch_size, args.seed)
        (args.out / "bc_losses.json").write_text(json.dumps(losses), encoding="utf-8")
        trainer.save_checkpoint(args.out / "pretrained.pt", episode=0)
    if args.mode == "pretrain":
        if not args.bc_updates:
            parser.error("pretrain requires positive --bc-updates")
        if args.correction_rounds:
            correction_seeds = set(range(args.correction_seed, args.correction_seed + args.correction_rounds * args.correction_episodes))
            if correction_seeds.intersection(trainer.demo_metadata.get("demo_seeds", [])):
                parser.error("Correction seeds must differ from original demonstration seeds")
            aggregate_corrections(trainer, env_cfg, dataset, args.correction_rounds,
                                  args.correction_episodes, args.correction_updates,
                                  args.batch_size, args.correction_seed, args.out)
        print("Initialization complete. Evaluate pretrained.pt and corrected.pt (if generated) before PPO.")
        return
    if args.demo_weight > 0 and trainer.demo_metadata.get("successful_episodes", 0):
        dataset = torch.load(args.demos, map_location="cpu", weights_only=False)
        validate_demos(dataset, env_cfg)
        if [r["seed"] for r in dataset["report"]] != trainer.demo_metadata["demo_seeds"]:
            parser.error("Resume requires the original demonstration dataset")
        trainer.demo_episodes = dataset["episodes"]
    trainer.demo_weight = args.demo_weight
    trainer.demo_metadata["demo_weight"] = args.demo_weight
    if args.checkpoint and trainer.demo_metadata.get("training_base_seed", args.seed) != args.seed:
        parser.error("Resume requires the original --seed")
    trainer.demo_metadata["training_base_seed"] = args.seed
    new_seeds = list(range(args.seed * 100000 + trainer.start_episode, args.seed * 100000 + cfg.episodes))
    if set(new_seeds).intersection(trainer.demo_metadata.get("demo_seeds", []) + trainer.demo_metadata.get("correction_seeds", [])):
        parser.error("Online training seeds overlap demonstration seeds")
    trainer.demo_metadata["training_seeds"] = sorted(set(trainer.demo_metadata.get("training_seeds", []) + new_seeds))
    trainer.output_directory = args.out
    if args.validation_interval > 0 and args.validation_episodes > 0:
        trainer.validation_interval = args.validation_interval
        trainer.validation_seeds = list(range(args.validation_seed, args.validation_seed + args.validation_episodes))
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
