"""Train MAPPO or MATD3 against one full three-UAV Isaac Sim scene."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback

import numpy as np
import torch

from artifact_paths import artifact_path
from pursuit.algorithms.matd3 import Matd3Config
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("mappo", "matd3"))
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Isaac PhysX parallel scenes in one process (GridCloner). "
        "MATD3 uses train_vectorized when >1; MAPPO stays single-scene for now.",
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
                        help="Enable Isaac PhysX LiDAR; default uses the noisy geometric LiDAR model")
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
    """Offline-to-online package that stops warm-start collapse (A5s / A6)."""
    slow = max(decay * 2, 200_000)
    values.update(
        warmup_steps=20_000,
        exploration_std=0.05,
        exploration_final_std=0.02,
        exploration_decay_steps=slow,
        demo_bc_weight=1.0,
        demo_bc_final_weight=0.5,
        demo_bc_decay_steps=slow,
        td3_bc_alpha=2.5,
        bc_on_batch=True,
        actor_anchor_weight=1.0,
        actor_anchor_final_weight=0.1,
        actor_anchor_decay_steps=slow,
        prior_fraction=0.5,
        utd=2,
        success_replay_only=True,
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


def _training_mix():
    root = Path(__file__).resolve().parent / "configs" / "pursuit_v2"
    return [_load_env_config(root / f"radar5_dual_{level}.json") for level in TRAIN_DOMAINS]


def _bind_training_domains(env):
    mix = _training_mix()

    def apply(episode):
        cfg = mix[int(episode) % len(mix)]
        env.configure_domain(cfg)
        return cfg

    def apply_medium():
        env.configure_domain(mix[0])
        return mix[0]

    return apply, apply_medium, mix


def _physical_validation_metrics(env, apply_medium, action_fn, seeds):
    """Held-out rollouts on the training level using the physical-capture scorer."""
    apply_medium()
    captured, returns, steps = [], [], []
    for seed in seeds:
        env.reset(seed=int(seed))
        obs = env.observe_search()
        total = 0.0
        result = {"capture_success": False}
        taken = 0
        for taken in range(1, int(env.cfg.search_steps) + 1):
            result = env.step_joint(action_fn(obs))
            obs = result["obs"]
            total += float(result["reward"])
            if result.get("terminated") or result.get("truncated"):
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
    return ResidualComposeConfig(
        base=requested,
        mode="blend",
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
        continuous_initial_std=0.08 if plus else 0.20,
        continuous_log_std_max=float(np.log(0.25 if plus else 0.40)),
        entropy_coef=0.001 if plus else 0.003,
        entropy_coef_end=0.0003 if plus else 0.001,
        adaptive_entropy=False,
        use_gat=True,
        use_hetero_entities=True,
        normalize_returns=False,
        checkpoint_interval=10,
        checkpoint_path=str(args.out / "latest.pt"),
        episode_budget_checkpoints=_parse_budget_checkpoints(args.budget_checkpoints, args.episodes),
    )
    apply_domain, apply_medium, mix = _bind_training_domains(env)

    def env_factory(episode=0):
        apply_domain(episode)
        env.reset(seed=args.seed * 100_000 + int(episode))
        return env

    trainer = PursuitDemoTrainer(env_factory, config)
    trainer.output_directory = args.out
    trainer.validation_seeds = _isaac_validation_seeds(args)
    trainer.validation_seed_bank = list(trainer.validation_seeds)
    # Stronger demo retention so DAgger warm-start is not washed out by on-policy noise.
    if args.variant in ("mappo_dagger", "mappo_plus"):
        trainer.demo_weight = 0.35 if plus else 0.15
        trainer.demo_weight_floor = 0.12 if plus else 0.04
    if plus:
        # Keep the frozen DAgger policy as a trust-region / KL anchor (O2O).
        # Residual/MPC blend is applied at Isaac *eval* time so on-policy PPO
        # log-probs stay consistent with DAgger full-action initialization.
        trainer.reference_kl_weight = 0.08
        trainer.safeguard_drop = 0.05
        trainer.safeguard_patience = 2
        residual = _resolve_residual(args, "mappo_plus")
        if residual is not None:
            # Optional train-time residual (explicit --residual-base); default none.
            trainer.set_residual(residual)
    from train_pursuit_with_demos import deterministic_action

    trainer.physical_metrics_fn = lambda: _physical_validation_metrics(
        env,
        apply_medium,
        lambda obs: deterministic_action(trainer, obs, env=env),
        trainer.validation_seeds,
    )
    trainer.demo_metadata["train_domains"] = list(TRAIN_DOMAINS)
    trainer.demo_metadata["execution_backend"] = "isaac_physx"
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["experiment_variant"] = args.variant
    trainer.demo_metadata["validation_seeds"] = list(trainer.validation_seeds)
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    resuming = False
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("Cannot resume MAPPO from a non-Isaac checkpoint")
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
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("MAPPO actor initialization must come from Isaac PhysX")
        source_env = QuadrotorPursuitConfig(**payload["env_config"])
        if source_env.capture_required_uavs != 2 or source_env.execution_reward_version != 5:
            raise ValueError("MAPPO actor initialization must use the dual-capture reward-v5 task")
        from pursuit.trainers.matd3_trainer import mean_actor_state_from_checkpoint

        trainer.actor.mean_actor.load_state_dict(mean_actor_state_from_checkpoint(payload))
        trainer.demo_metadata.update(payload.get("demo_metadata", {}))
        trainer.demo_metadata["execution_backend"] = "isaac_physx"
        trainer.demo_metadata["actor_init"] = str(args.actor_init)
        if plus or float(getattr(trainer, "reference_kl_weight", 0.0) or 0.0) > 0.0:
            trainer.capture_reference_policy()
    for path, attribute in ((args.demos, "demo_episodes"), (args.aux_demos, "correction_episodes")):
        if path is None:
            continue
        dataset = torch.load(path, map_location="cpu", weights_only=False)
        if dataset.get("execution_backend") != "isaac_physx":
            raise ValueError(f"{path} is not an Isaac PhysX demonstration dataset")
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
    if trainer.validation_seeds and int(getattr(trainer, "start_episode", 0) or 0) == 0:
        try:
            trainer.validate_capture(0)
        except Exception as error:
            print(f"[mappo] pre-RL validation skipped: {error}", flush=True)
    history = trainer.train()
    trainer.save_checkpoint(args.out / "final.pt", episode=trainer.cfg.episodes, history=history)
    write_pursuit_outputs(args.out, history, trainer.episode_traces)
    best = args.out / "best_capture.pt"
    return {
        "algorithm": "mappo",
        "episodes": len(history),
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
    trainer.physical_metrics_fn = lambda: _physical_validation_metrics(
        env,
        apply_medium,
        lambda obs: trainer.deterministic_action(obs, env=env),
        trainer.validation_seeds,
    )
    trainer.demo_metadata["train_domains"] = list(TRAIN_DOMAINS)
    trainer.demo_metadata["execution_backend"] = "isaac_physx"
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    trainer.validation_seeds = _isaac_validation_seeds(args)
    trainer.validation_seed_bank = list(trainer.validation_seeds)
    residual = _resolve_residual(args, variant)
    if residual is not None:
        trainer.set_residual(residual)
    resuming = False
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("Cannot resume MATD3 from a non-Isaac checkpoint")
        if payload.get("demo_metadata", {}).get("experiment_variant") != args.variant:
            raise ValueError("Cannot resume MATD3 from a different experiment variant")
        trainer.load_checkpoint(args.checkpoint)
        trainer.cfg.episodes = args.episodes
        resuming = True
    if args.actor_init:
        payload = torch.load(args.actor_init, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("MATD3 actor initialization must be collected/trained in Isaac PhysX")
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
    if int(getattr(args, "num_envs", 1) or 1) < 1:
        parser.error("--num-envs must be >= 1")
    if int(getattr(args, "num_envs", 1) or 1) > 1 and args.algorithm != "matd3":
        parser.error("Isaac --num-envs>1 currently wires through MATD3 train_vectorized only")
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
    env = IsaacPursuitEnv(
        env_cfg,
        seed=args.seed,
        headless=headless,
        native_lidar=args.native_lidar,
        num_envs=num_envs,
    )
    backend_metadata = {
        "backend": "omnidrones_isaac_sim",
        "training_algorithm": args.algorithm,
        "scene_count": num_envs,
        "uavs_per_scene": 3,
        "native_physx_lidar": args.native_lidar and num_envs == 1,
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
    try:
        print(f"[isaac-training] starting algorithm={args.algorithm} episodes={args.episodes}", flush=True)
        result = _run_mappo(args, env) if args.algorithm == "mappo" else _run_matd3(args, env)
        print(json.dumps(result, indent=2))
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()