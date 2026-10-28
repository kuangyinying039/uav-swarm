"""Train MAPPO or MATD3 against one full three-UAV Isaac Sim scene."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import traceback

import numpy as np
import torch

from artifact_paths import artifact_path
from pursuit.algorithms.matd3 import Matd3Config
from quadrotor_pursuit_env import QuadrotorPursuitConfig

_ISAAC_BACKENDS = frozenset({"isaac_physx", "isaac_physx_closed"})


def _require_isaac_backend(value, what: str) -> None:
    if value not in _ISAAC_BACKENDS:
        raise ValueError(f"{what} must come from Isaac PhysX (got {value!r})")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("mappo", "matd3"))
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Isaac PhysX parallel scenes in one process (GridCloner). "
        "MAPPO and MATD3 both use train_vectorized when >1.",
    )
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_hard.json"))
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--checkpoint", type=artifact_path,
                        help="Resume a full checkpoint produced by the selected trainer")
    parser.add_argument("--actor-init", type=artifact_path,
                        help="Initialize the MATD3 actor from a MAPPO/DAgger/MATD3 checkpoint")
    parser.add_argument("--demos", type=artifact_path,
                        help="Isaac MPC behavior-cloning episodes for MAPPO")
    parser.add_argument("--aux-demos", type=artifact_path,
                        help="Isaac DAgger correction episodes for MAPPO")
    parser.add_argument("--prior", type=artifact_path, help="Isaac-collected transition dataset for MATD3 prior replay")
    parser.add_argument("--prior-fraction", type=float)
    parser.add_argument(
        "--variant",
        choices=("mappo_dagger", "mappo_plus", "A6"),
        help="A6 is pure-actor HGAT-MATD3 (O2O, success replay, capture rollback). "
        "mappo_plus is the MAPPO mainline; mappo_dagger is its DAgger ablation.",
    )
    parser.add_argument(
        "--residual-base",
        choices=("none", "mpc", "frpn", "apf"),
        default=None,
        help="Compose RL with a classical teacher each step (default: none for MAPPO+ and A6)",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.40,
        help="Initial blend weight for RL when --residual-base is set (a=(1-w)·base+w·rl)",
    )
    parser.add_argument(
        "--residual-weight-final",
        type=float,
        default=None,
        help="Optional annealed final RL blend weight when --residual-base is set",
    )
    parser.add_argument(
        "--validation-episodes",
        type=int,
        default=20,
        help="Held-out Hard episodes scored by physical capture for best_capture.pt (0 disables)",
    )
    parser.add_argument(
        "--validation-interval",
        type=int,
        default=100,
        help="Run the held-out physical capture test every this many episodes",
    )
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optional per-episode rollout cap for short validation runs")
    parser.add_argument(
        "--budget-checkpoints",
        default="250,500,750,1000",
        help="Comma-separated episode counts for sample-efficiency snapshots "
             "(empty string disables). Values above --episodes are ignored.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=1)
    from isaac_cli_flags import add_isaac_display_flags

    add_isaac_display_flags(parser, default_headless=True)
    parser.add_argument("--native-lidar", action="store_true",
                        help="Force Isaac PhysX LiDAR on (default ON for --physx-closed-loop)")
    parser.add_argument("--no-native-lidar", action="store_true",
                        help="Force geometric LiDAR fallback even with --physx-closed-loop")
    parser.add_argument(
        "--physx-closed-loop",
        action="store_true",
        help="Pure PhysX 3+1 closed loop (same-model evader drone); see SIM2REAL.md",
    )
    return parser


def _parse_budget_checkpoints(raw, episodes):
    text = str(raw or "").strip()
    if not text:
        return ()
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("budget checkpoints must be positive")
        if value <= int(episodes):
            values.append(value)
    return tuple(sorted(set(values)))


def _scaled_decay_steps(episodes):
    """Keep exploration/BC schedules finishing inside the training budget."""
    # ~100 decision steps/episode is a conservative mean for dual Medium.
    horizon = max(20_000, int(episodes) * 100)
    return int(min(150_000, horizon))


def _load_env_config(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)
    return cfg


def _apply_o2o_stable(values, decay):
    """Offline-to-online package that stops warm-start collapse (A5s / A6).

    P1 defaults: keep BC + actor-anchor alive through training, and keep
    exploration small so DAgger init is not washed out (see hard best≪final).
    """
    slow = max(decay * 2, 200_000)
    values.update(
        warmup_steps=20_000,
        exploration_std=0.04,
        exploration_final_std=0.015,
        exploration_decay_steps=slow,
        demo_bc_weight=1.0,
        demo_bc_final_weight=0.35,
        demo_bc_decay_steps=slow,
        td3_bc_alpha=2.5,
        bc_on_batch=True,
        actor_anchor_weight=1.0,
        actor_anchor_final_weight=0.35,
        actor_anchor_decay_steps=slow,
        prior_fraction=0.5,
        utd=2,
        success_replay_only=True,
        max_safeguard_rollbacks=3,
    )
    return values


def _variant_config(variant, args, env_cfg):
    values = asdict(Matd3Config())
    values.update(
        gamma=env_cfg.reward_gamma,
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
        checkpoint_interval=10,
    )
    values["validation_interval"] = int(args.validation_interval)
    decay = _scaled_decay_steps(args.episodes)
    if variant == "A6":
        values.update(
            use_gat=True,
            actor_lr=5e-5,
            critic_lr=3e-4,
            prior_fraction=0.5,
            warmup_policy="actor",
            critic_pretrain_updates=5000,
            exploration_std=0.10,
            exploration_final_std=0.03,
            exploration_decay_steps=decay,
            demo_bc_weight=0.0,
            demo_bc_final_weight=0.0,
            demo_bc_decay_steps=decay,
            utd=4,
            warmup_steps=15_000,
        )
        _apply_o2o_stable(values, decay)
        required = (args.actor_init is not None and args.prior is not None)
    elif variant.startswith("mappo_"):
        required = True  # checked in main()
    else:
        required = (args.actor_init is None and args.prior is None)
    if not required:
        raise ValueError(f"Variant {variant} received incompatible --actor-init/--prior inputs")
    if args.prior_fraction is not None:
        values["prior_fraction"] = args.prior_fraction
    budgets = _parse_budget_checkpoints(getattr(args, "budget_checkpoints", ""), args.episodes)
    values["episode_budget_checkpoints"] = budgets
    return Matd3Config(**values) if not str(variant).startswith("mappo_") else None


TRAIN_DOMAINS = ("hard",)
EVAL_DOMAINS = ("nominal", "medium", "hard", "severe", "extreme")


def _training_mix(*, physx_closed: bool = False):
    root = Path(__file__).resolve().parent / "configs" / "pursuit_v2"
    suffix = "_physx_closed" if physx_closed else ""
    return [
        _load_env_config(root / f"radar5_dual_{level}{suffix}.json")
        for level in TRAIN_DOMAINS
    ]


def _bind_training_domains(env):
    # Closed-loop envs must mix *_physx_closed ladder JSONs so capture_radius /
    # building_count stay on the same contract as IsaacPursuitEnv.cfg.
    closed = bool(getattr(env, "_physx_closed_loop", False))
    mix = _training_mix(physx_closed=closed)

    def apply(episode):
        cfg = mix[int(episode) % len(mix)]
        env.configure_domain(cfg)
        return cfg

    def apply_medium():
        env.configure_domain(mix[0])
        return mix[0]

    return apply, apply_medium, mix


def _physical_validation_metrics(env, apply_medium, action_fn, seeds):
    """Held-out rollouts on the training level using the physical-capture scorer.

    ``action_fn(obs, env_index=0)`` must return a ``(n_uavs, 4)`` action. With
    ``num_envs>1`` only env 0 is scored; other clones receive policy actions so
    PhysX can advance the shared world step.
    """
    apply_medium()
    captured, returns, steps = [], [], []
    num_envs = int(getattr(env, "num_envs", 1) or 1)
    for seed in seeds:
        if num_envs > 1:
            obs_list = env.reset(seeds=[int(seed) + i for i in range(num_envs)])
        else:
            reset_obs = env.reset(seed=int(seed))
            obs = reset_obs if isinstance(reset_obs, dict) else env.observe_search()
            if not isinstance(obs, dict):
                obs = env.observe_search()
            obs_list = [obs]
        total = 0.0
        result = {"capture_success": False}
        taken = 0
        for taken in range(1, int(env.cfg.search_steps) + 1):
            if num_envs == 1:
                result = env.step_joint(action_fn(obs_list[0], env_index=0))
                obs_list = [result["obs"]]
                reward = float(result["reward"])
                done = bool(result.get("terminated") or result.get("truncated"))
            else:
                actions = np.stack(
                    [action_fn(obs_list[i], env_index=i) for i in range(num_envs)],
                    axis=0,
                )
                results = env.step_joint(actions)
                result = dict(results[0])
                info = result.get("info") or {}
                if result.get("_episode_done") and info.get("final_obs") is not None:
                    # Score the terminal transition, not the auto-reset observation.
                    if "capture_success" in info:
                        result["capture_success"] = bool(info["capture_success"])
                    if "final_terminated" in info:
                        result["terminated"] = bool(info["final_terminated"])
                        result["truncated"] = bool(info.get("final_truncated", False))
                reward = float(result["reward"])
                done = bool(result.get("_episode_done") or result.get("terminated") or result.get("truncated"))
                obs_list = [item["obs"] for item in results]
            total += reward
            if done:
                break
        captured.append(bool(result.get("capture_success", False)))
        returns.append(total)
        steps.append(taken)
    return {
        "capture_rate": float(np.mean(captured)) if captured else 0.0,
        "mean_censored_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "episodes": len(seeds),
        "validation_domain": TRAIN_DOMAINS[0],
        "capture_scorer": "physical",
    }


def _isaac_validation_seeds(args):
    count = int(getattr(args, "validation_episodes", 0) or 0)
    if count <= 0:
        return []
    # Held-out from training seeds (seed*100000 + ep); use a high offset bank.
    base = int(args.seed) * 1_000_000 + 900_000
    return list(range(base, base + count))


def _resolve_residual(args, variant):
    from pursuit_residual import ResidualComposeConfig

    requested = getattr(args, "residual_base", None)
    if requested is None:
        requested = "none"
    if requested in (None, "none"):
        return None
    weight = float(getattr(args, "residual_weight", 0.40))
    weight_final = getattr(args, "residual_weight_final", None)
    # MAPPO on-policy: store δ / apply base+w·δ (see PursuitDemoTrainer.set_residual).
    # Blend is reserved for eval-time hybrids on full-action checkpoints.
    mode = "residual" if str(variant).startswith("mappo") else "blend"
    return ResidualComposeConfig(
        base=requested,
        mode=mode,
        weight=weight,
        weight_final=None if weight_final is None else float(weight_final),
        weight_warmup_steps=10_000,
        weight_decay_steps=max(_scaled_decay_steps(args.episodes), 80_000),
    )


def _run_mappo(args, env):
    from marl_trainers import TrainConfig, seed_everything
    from pursuit_training_output import write_pursuit_outputs
    from train_pursuit_with_demos import PursuitDemoTrainer

    seed_everything(args.seed)
    plus = args.variant == "mappo_plus"
    # PursuitDemoTrainer forbids normalize_returns (it breaks shaped-reward critic units).
    config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=env.cfg.reward_gamma,
        gae_lambda=0.98 if plus else 0.97,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
        actor_lr=5e-5 if plus else 1e-4,
        critic_lr=1e-4,
        update_epochs=5 if plus else 3,
        target_kl=0.006 if plus else 0.01,
        clip_eps=0.08 if plus else 0.10,
        continuous_initial_std=0.06 if plus else 0.20,
        continuous_log_std_max=float(np.log(0.20 if plus else 0.40)),
        entropy_coef=0.0005 if plus else 0.003,
        entropy_coef_end=0.00015 if plus else 0.001,
        adaptive_entropy=False,
        use_gat=True,
        use_hetero_entities=True,
        normalize_returns=False,
        checkpoint_interval=10,
        checkpoint_path=str(args.out / "latest.pt"),
        episode_budget_checkpoints=_parse_budget_checkpoints(args.budget_checkpoints, args.episodes),
    )
    apply_domain, apply_medium, mix = _bind_training_domains(env)
    num_envs = int(getattr(args, "num_envs", 1) or 1)

    def env_factory(episode=0):
        apply_domain(episode)
        if num_envs > 1:
            env.reset(
                seeds=[
                    args.seed * 100_000 + int(episode) + i for i in range(num_envs)
                ]
            )
        else:
            env.reset(seed=args.seed * 100_000 + int(episode))
        return env

    trainer = PursuitDemoTrainer(env_factory, config)
    trainer.seed = int(args.seed)
    trainer.output_directory = args.out
    trainer.validation_seeds = _isaac_validation_seeds(args)
    trainer.validation_seed_bank = list(trainer.validation_seeds)
    trainer.on_vector_episode_complete = apply_domain
    # P1: stronger demo retention + KL anchor so DAgger warm-start survives PPO.
    if args.variant in ("mappo_dagger", "mappo_plus"):
        trainer.demo_weight = 0.45 if plus else 0.15
        trainer.demo_weight_floor = 0.20 if plus else 0.04
    if plus:
        # Keep the frozen DAgger policy as a trust-region / KL anchor (O2O).
        # Residual/MPC blend is applied at Isaac *eval* time so on-policy PPO
        # log-probs stay consistent with DAgger full-action initialization.
        trainer.reference_kl_weight = 0.15
        trainer.safeguard_drop = 0.05
        trainer.safeguard_patience = 2
        trainer.max_safeguard_rollbacks = 3
        residual = _resolve_residual(args, "mappo_plus")
        if residual is not None:
            # Optional train-time residual (explicit --residual-base); default none.
            trainer.set_residual(residual)
            # Residual runs: even tighter KL / demo floor (teacher already in the loop).
            trainer.reference_kl_weight = max(float(trainer.reference_kl_weight), 0.20)
            trainer.demo_weight = max(float(trainer.demo_weight), 0.50)
            trainer.demo_weight_floor = max(float(trainer.demo_weight_floor), 0.25)
    from train_pursuit_with_demos import deterministic_action

    def _validation_action(obs, env_index=0):
        logic = env
        if num_envs > 1 and hasattr(env, "bridges"):
            logic = env.bridges[int(env_index)].env
        return deterministic_action(trainer, obs, env=logic)

    trainer.physical_metrics_fn = lambda: _physical_validation_metrics(
        env,
        apply_medium,
        _validation_action,
        trainer.validation_seeds,
    )
    trainer.demo_metadata["train_domains"] = list(TRAIN_DOMAINS)
    trainer.demo_metadata["execution_backend"] = (
        "isaac_physx_closed" if bool(getattr(args, "physx_closed_loop", False)) else "isaac_physx"
    )
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["experiment_variant"] = args.variant
    trainer.demo_metadata["validation_seeds"] = list(trainer.validation_seeds)
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    resuming = False
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _require_isaac_backend(
            payload.get("demo_metadata", {}).get("execution_backend"),
            "MAPPO resume checkpoint",
        )
        if payload.get("demo_metadata", {}).get("experiment_variant") != args.variant:
            raise ValueError("Cannot resume MAPPO from a different experiment variant")
        trainer.load_checkpoint(args.checkpoint)
        trainer.cfg.episodes = args.episodes
        trainer.cfg.checkpoint_interval = 10
        trainer.cfg.checkpoint_path = str(args.out / "latest.pt")
        resuming = True
        print(f"[mappo] resume episode={trainer.start_episode}/{args.episodes} from {args.checkpoint}", flush=True)
    if args.actor_init and not resuming:
        payload = torch.load(args.actor_init, map_location="cpu", weights_only=False)
        _require_isaac_backend(
            payload.get("demo_metadata", {}).get("execution_backend"),
            "MAPPO actor initialization",
        )
        source_env = QuadrotorPursuitConfig(**payload["env_config"])
        if source_env.capture_required_uavs != 2 or source_env.execution_reward_version != 5:
            raise ValueError("MAPPO actor initialization must use the dual-capture reward-v5 task")
        from pursuit.trainers.matd3_trainer import mean_actor_state_from_checkpoint

        trainer.actor.mean_actor.load_state_dict(mean_actor_state_from_checkpoint(payload))
        trainer.demo_metadata.update(payload.get("demo_metadata", {}))
        trainer.demo_metadata["execution_backend"] = (
            "isaac_physx_closed"
            if bool(getattr(args, "physx_closed_loop", False))
            else "isaac_physx"
        )
        trainer.demo_metadata["actor_init"] = str(args.actor_init)
        if plus or float(getattr(trainer, "reference_kl_weight", 0.0) or 0.0) > 0.0:
            trainer.capture_reference_policy()
    for path, attribute in ((args.demos, "demo_episodes"), (args.aux_demos, "correction_episodes")):
        if path is None:
            continue
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        _require_isaac_backend(
            dataset.get("execution_backend"),
            f"{path} demonstration dataset",
        )
        recorded_config = QuadrotorPursuitConfig(**dataset["env_config"])
        if recorded_config.capture_required_uavs != 2 or recorded_config.execution_reward_version != 5:
            raise ValueError(f"{path} does not use the dual-capture reward-v5 contract")
        expected_config = asdict(env.cfg)
        actual_config = asdict(recorded_config)
        expected_config.pop("seed", None)
        actual_config.pop("seed", None)
        if actual_config != expected_config:
            raise ValueError(f"{path} was collected in a different environment profile")
        setattr(trainer, attribute, dataset["episodes"])
        trainer.demo_metadata[f"{attribute}_path"] = str(path)
    trainer.validation_interval = int(args.validation_interval)
    trainer.demo_metadata["num_envs"] = num_envs
    if trainer.validation_seeds and int(getattr(trainer, "start_episode", 0) or 0) == 0:
        try:
            trainer.validate_capture(0)
        except Exception as error:
            print(f"[mappo] pre-RL validation skipped: {error}", flush=True)
    if num_envs > 1:
        history = trainer.train_vectorized(env)
    else:
        history = trainer.train()
    trainer.save_checkpoint(args.out / "final.pt", episode=trainer.cfg.episodes, history=history)
    write_pursuit_outputs(args.out, history, trainer.episode_traces)
    best = args.out / "best_capture.pt"
    return {
        "algorithm": "mappo",
        "episodes": len(history),
        "num_envs": num_envs,
        "checkpoint": str(best if best.is_file() else args.out / "final.pt"),
        "best_capture": str(best) if best.is_file() else None,
    }


def _run_matd3(args, env):
    from marl_trainers import seed_everything
    from pursuit.algorithms.matd3 import Matd3Config
    from pursuit.trainers.matd3_trainer import Matd3Trainer, mean_actor_state_from_checkpoint

    seed_everything(args.seed)
    variant = args.variant or "A6"
    config = _variant_config(variant, args, env.cfg)
    config.max_steps = int(args.max_steps)
    num_envs = int(getattr(args, "num_envs", 1) or 1)

    apply_domain, apply_medium, mix = _bind_training_domains(env)

    def env_factory(episode=0):
        apply_domain(episode)
        if num_envs > 1:
            env.reset(
                seeds=[
                    args.seed * 100_000 + int(episode) + i for i in range(num_envs)
                ]
            )
        else:
            env.reset(seed=args.seed * 100_000 + int(episode))
        return env

    trainer = Matd3Trainer(env_factory, config, seed=args.seed)

    def _validation_action(obs, env_index=0):
        # Match MAPPO: vectorized Isaac validation passes env_index so residual /
        # teacher composition (if any) reads the correct clone's logic env.
        logic = env
        if num_envs > 1 and hasattr(env, "bridges"):
            logic = env.bridges[int(env_index)].env
        return trainer.deterministic_action(obs, env=logic)

    trainer.physical_metrics_fn = lambda: _physical_validation_metrics(
        env,
        apply_medium,
        _validation_action,
        trainer.validation_seeds,
    )
    trainer.demo_metadata["train_domains"] = list(TRAIN_DOMAINS)
    trainer.demo_metadata["execution_backend"] = (
        "isaac_physx_closed" if bool(getattr(args, "physx_closed_loop", False)) else "isaac_physx"
    )
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    trainer.validation_seeds = _isaac_validation_seeds(args)
    trainer.validation_seed_bank = list(trainer.validation_seeds)
    residual = _resolve_residual(args, variant)
    if residual is not None:
        trainer.set_residual(residual)
        # Residual / blend runs: keep a tighter offline-to-online package so RL
        # corrects the teacher instead of unlearning the DAgger prior (B3C/OVMSE-ish).
        # Zero-init soft residual uses blend α:0→small; keep exploration tiny and
        # allow near-miss / mixed transitions (success_replay_only=False).
        trainer.cfg.exploration_std = min(float(trainer.cfg.exploration_std), 0.03)
        trainer.cfg.exploration_final_std = min(float(trainer.cfg.exploration_final_std), 0.01)
        trainer.cfg.actor_anchor_final_weight = max(
            float(trainer.cfg.actor_anchor_final_weight), 0.35
        )
        trainer.cfg.success_replay_only = False
        trainer.demo_metadata["o2o_residual_tight"] = True
        trainer.demo_metadata["residual_zero_init_blend"] = True
    resuming = False
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        _require_isaac_backend(
            payload.get("demo_metadata", {}).get("execution_backend"),
            "MATD3 resume checkpoint",
        )
        if payload.get("demo_metadata", {}).get("experiment_variant") != args.variant:
            raise ValueError("Cannot resume MATD3 from a different experiment variant")
        trainer.load_checkpoint(args.checkpoint)
        trainer.cfg.episodes = args.episodes
        # CLI validation settings win over whatever was baked into the ckpt era.
        trainer.cfg.validation_interval = int(args.validation_interval)
        trainer.validation_seeds = _isaac_validation_seeds(args)
        trainer.validation_seed_bank = list(trainer.validation_seeds)
        resuming = True
        print(
            f"[matd3] resume episode={trainer.start_episode}/{args.episodes} "
            f"val={len(trainer.validation_seeds)}@{trainer.cfg.validation_interval} "
            f"from {args.checkpoint}",
            flush=True,
        )
    if args.actor_init:
        payload = torch.load(args.actor_init, map_location="cpu", weights_only=False)
        _require_isaac_backend(
            payload.get("demo_metadata", {}).get("execution_backend"),
            "MATD3 actor initialization",
        )
        source_env = QuadrotorPursuitConfig(**payload["env_config"])
        if source_env.capture_required_uavs != 2 or source_env.execution_reward_version != 5:
            raise ValueError("MATD3 actor initialization must use the dual-capture reward-v5 task")
        actor_weights = mean_actor_state_from_checkpoint(payload)
        # Resume must keep the trained weights; only seed the O2O anchor from expert init.
        if resuming:
            trainer.set_actor_anchor(actor_weights)
        else:
            trainer.load_actor_weights(actor_weights)
        trainer.demo_metadata["actor_init"] = str(args.actor_init)
        trainer.demo_metadata["actor_init_algorithm"] = payload.get("algorithm")
    if args.prior:
        from pursuit.data.transition_dataset import load_transition_dataset

        trainer.load_prior(
            load_transition_dataset(
                args.prior,
                env.cfg,
                expected_backend="isaac_physx",
            )
        )
    trainer.demo_metadata["experiment_variant"] = variant
    trainer.demo_metadata["num_envs"] = num_envs
    trainer.demo_metadata["validation_seeds"] = list(trainer.validation_seeds)
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.output_directory = args.out
    rng = np.random.default_rng(args.seed)
    if (not resuming) and trainer.cfg.critic_pretrain_updates > 0:
        pretrain_summary = trainer.pretrain_critic()
        if pretrain_summary:
            (args.out / "critic_pretrain.json").write_text(
                json.dumps(pretrain_summary, indent=2), encoding="utf-8"
            )
            print(f"[critic-pretrain] {json.dumps(pretrain_summary)}", flush=True)
        trainer.save_checkpoint(args.out / "initial.pt", episode=0)
        if trainer.validation_seeds:
            from pursuit.eval import evaluate_policy

            trainer.validate(0, evaluate_policy)
    if num_envs > 1:
        history = trainer.train_vectorized(rng, env)
    else:
        history = trainer.train(rng)
    trainer.save_checkpoint(args.out / "final.pt", episode=trainer.cfg.episodes)
    best = args.out / "best_capture.pt"
    return {
        "algorithm": trainer.algorithm_name,
        "episodes": len(history),
        "env_steps": trainer.env_steps,
        "num_envs": num_envs,
        "checkpoint": str(best if best.is_file() else args.out / "final.pt"),
        "best_capture": str(best) if best.is_file() else None,
        "residual": trainer.demo_metadata.get("residual"),
    }


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.variant is None:
        args.variant = "mappo_plus" if args.algorithm == "mappo" else "A6"
    if args.episodes < 1 or args.hidden_dim < 1 or args.batch_size < 1 or args.max_steps < 0:
        parser.error("episodes, hidden-dim, and batch-size must be positive; max-steps cannot be negative")
    if args.validation_interval < 1 or args.validation_episodes < 0:
        parser.error("--validation-interval must be positive and --validation-episodes cannot be negative")
    if int(getattr(args, "num_envs", 1) or 1) < 1:
        parser.error("--num-envs must be >= 1")
    if args.algorithm == "mappo" and args.variant in ("mappo_dagger", "mappo_plus"):
        if not args.actor_init or not args.demos:
            parser.error(f"{args.variant} requires Isaac-trained --actor-init and --demos")
    if args.algorithm == "mappo" and args.aux_demos and not args.demos:
        parser.error("--aux-demos requires its matching --demos source")
    if args.algorithm == "mappo" and args.prior:
        parser.error("--prior is supported only for MATD3")
    if args.variant and args.algorithm == "matd3" and args.variant.startswith("mappo_"):
        parser.error("mappo_* variants require algorithm=mappo")
    if args.algorithm == "mappo" and args.variant not in ("mappo_dagger", "mappo_plus"):
        parser.error("A6 requires algorithm=matd3")
    if args.residual_weight <= 0 or args.residual_weight > 1:
        parser.error("--residual-weight must be in (0, 1]")
    # Resume may pass --checkpoint with --actor-init/--prior: checkpoint restores
    # trained weights; actor-init only seeds the optional O2O anchor; prior refills
    # the offline buffer (not serialized in checkpoints).
    try:
        budgets = _parse_budget_checkpoints(args.budget_checkpoints, args.episodes)
    except ValueError as error:
        parser.error(str(error))
    args.budget_checkpoints = ",".join(str(x) for x in budgets) if budgets else ""
    env_cfg = _load_env_config(args.env_config)
    allowed_profiles = {f"radar5_dual_{level}.json" for level in EVAL_DOMAINS}
    allowed_profiles |= {f"radar5_dual_{level}_physx_closed.json" for level in EVAL_DOMAINS}
    if args.env_config.name not in allowed_profiles:
        parser.error(
            "Isaac training uses the dual Radar5 profiles "
            + ", ".join(sorted(allowed_profiles))
        )
    if args.algorithm == "matd3" and args.variant == "A6":
        try:
            _variant_config(args.variant, args, env_cfg)
        except ValueError as error:
            parser.error(str(error))
    torch.set_num_threads(max(1, args.torch_threads))
    args.out.mkdir(parents=True, exist_ok=True)
    from isaac_cli_flags import resolve_headless
    from isaac_pursuit_backend import IsaacPursuitEnv

    headless = resolve_headless(args)
    num_envs = int(getattr(args, "num_envs", 1) or 1)
    physx_closed = bool(getattr(args, "physx_closed_loop", False))
    if getattr(args, "no_native_lidar", False):
        native_lidar = False
    elif getattr(args, "native_lidar", False):
        native_lidar = True
    else:
        native_lidar = None  # closed-loop defaults ON inside IsaacPursuitEnv
    env = IsaacPursuitEnv(
        env_cfg,
        seed=args.seed,
        headless=headless,
        native_lidar=native_lidar,
        num_envs=num_envs,
        physx_closed_loop=physx_closed,
    )
    backend_metadata = {
        "backend": "omnidrones_isaac_sim",
        "training_algorithm": args.algorithm,
        "scene_count": num_envs,
        "uavs_per_scene": 3,
        "physx_closed_loop": physx_closed,
        "physics_drones_per_scene": int(getattr(env, "_n_physics_drones", 3)),
        "execution_backend": "isaac_physx_closed" if physx_closed else "isaac_physx",
        "native_physx_lidar": bool(getattr(env, "_native_lidar_enabled", False)),
        "headless": headless,
        "visualize": bool(getattr(args, "visualize", False)),
        "physics_dt": env.physics_dt,
        "decision_dt": env.cfg.decision_dt,
        "training_seed": args.seed,
        "env_config": asdict(env_cfg),
        "experiment_variant": args.variant or ("mappo_plus" if args.algorithm == "mappo" else "A6"),
        "train_domain": list(TRAIN_DOMAINS),
        "validation_domain": TRAIN_DOMAINS[0],
        "capture_scorer": "physical",
        "num_envs": num_envs,
    }
    (args.out / "backend.json").write_text(json.dumps(backend_metadata, indent=2), encoding="utf-8")
    # Isaac SimulationApp.close() may call os._exit(0) and mask a raised error,
    # so track failure explicitly and force a non-zero process status afterward.
    exit_code = 0
    try:
        print(f"[isaac-training] starting algorithm={args.algorithm} episodes={args.episodes}", flush=True)
        result = _run_mappo(args, env) if args.algorithm == "mappo" else _run_matd3(args, env)
        print(json.dumps(result, indent=2))
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            env.close()
        except Exception:
            pass
    if exit_code:
        os._exit(exit_code)


if __name__ == "__main__":
    main()