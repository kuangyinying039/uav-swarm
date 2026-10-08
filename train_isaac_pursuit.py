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
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_medium.json"))
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
    parser.add_argument("--variant", choices=("mappo_scratch", "mappo_dagger", "A0", "A1", "A2", "A3", "A4", "A5"),
                        help="Fixed comparison/ablation recipe; A0-A5 are MATD3 variants")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Optional per-episode rollout cap for short validation runs")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-lidar", action="store_true",
                        help="Enable Isaac PhysX LiDAR; default uses the noisy geometric LiDAR model")
    return parser


def _load_env_config(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)
    return cfg


def _variant_config(variant, args, env_cfg):
    values = asdict(Matd3Config())
    values.update(
        gamma=env_cfg.reward_gamma,
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
    )
    if variant == "A0":
        values.update(use_gat=False, prior_fraction=0.0, warmup_policy="random",
                      critic_pretrain_updates=0, demo_bc_weight=0.0, demo_bc_final_weight=0.0)
        required = (args.actor_init is None and args.prior is None)
    elif variant == "A1":
        values.update(use_gat=True, prior_fraction=0.0, warmup_policy="random",
                      critic_pretrain_updates=0, demo_bc_weight=0.0, demo_bc_final_weight=0.0)
        required = (args.actor_init is None and args.prior is None)
    elif variant in ("A2", "A3", "A4", "A5"):
        pretrain = 5000 if variant in ("A4", "A5") else 0
        bc_start, bc_end = (1.0, 0.05) if variant in ("A3", "A4", "A5") else (0.0, 0.0)
        values.update(
            use_gat=True,
            actor_lr=5e-5,
            critic_lr=3e-4,
            prior_fraction=0.5,
            warmup_policy="actor",
            critic_pretrain_updates=pretrain,
            exploration_std=0.10,
            exploration_final_std=0.03,
            exploration_decay_steps=150000,
            demo_bc_weight=bc_start,
            demo_bc_final_weight=bc_end,
            demo_bc_decay_steps=150000,
        )
        required = (args.actor_init is not None and args.prior is not None)
    elif variant == "mappo_scratch":
        required = (args.actor_init is None and args.prior is None)
    else:
        required = (args.actor_init is None and args.prior is None)
    if not required:
        raise ValueError(f"Variant {variant} received incompatible --actor-init/--prior inputs")
    if args.prior_fraction is not None:
        values["prior_fraction"] = args.prior_fraction
    return Matd3Config(**values) if variant != "mappo_scratch" else None


def _run_mappo(args, env):
    from marl_trainers import TrainConfig, seed_everything
    from pursuit_training_output import write_pursuit_outputs
    from train_pursuit_with_demos import PursuitDemoTrainer

    seed_everything(args.seed)
    config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        gamma=env.cfg.reward_gamma,
        gae_lambda=0.97,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        device=args.device,
        actor_lr=1e-4,
        critic_lr=1e-4,
        update_epochs=3,
        target_kl=0.01,
        clip_eps=0.10,
        continuous_initial_std=0.20,
        continuous_log_std_max=float(np.log(0.40)),
        entropy_coef=0.003,
        entropy_coef_end=0.001,
        adaptive_entropy=False,
        use_gat=True,
        use_hetero_entities=True,
        normalize_returns=False,
        checkpoint_path=str(args.out / "latest.pt"),
    )
    factory_calls = 0

    def env_factory(episode=0):
        nonlocal factory_calls
        if factory_calls >= 2:
            env.reset(seed=args.seed * 100_000 + int(episode))
        factory_calls += 1
        return env

    trainer = PursuitDemoTrainer(env_factory, config)
    trainer.output_directory = args.out
    trainer.validation_seeds = []
    trainer.demo_metadata["execution_backend"] = "isaac_physx"
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["experiment_variant"] = args.variant
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("Cannot resume MAPPO from a non-Isaac checkpoint")
        if payload.get("demo_metadata", {}).get("experiment_variant") != args.variant:
            raise ValueError("Cannot resume MAPPO from a different experiment variant")
        trainer.load_checkpoint(args.checkpoint)
        trainer.cfg.episodes = args.episodes
        trainer.cfg.checkpoint_path = str(args.out / "latest.pt")
    if args.actor_init:
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
    history = trainer.train()
    trainer.save_checkpoint(args.out / "final.pt", episode=trainer.cfg.episodes, history=history)
    write_pursuit_outputs(args.out, history, trainer.episode_traces)
    return {"algorithm": "mappo", "episodes": len(history), "checkpoint": str(args.out / "final.pt")}


def _run_matd3(args, env):
    from marl_trainers import seed_everything
    from pursuit.algorithms.matd3 import Matd3Config
    from pursuit.trainers.matd3_trainer import Matd3Trainer, mean_actor_state_from_checkpoint

    seed_everything(args.seed)
    config = _variant_config(args.variant or "A5", args, env.cfg)

    factory_calls = 0

    def env_factory(episode=0):
        nonlocal factory_calls
        if factory_calls >= 1:
            env.reset(seed=args.seed * 100_000 + int(episode))
        factory_calls += 1
        return env

    trainer = Matd3Trainer(env_factory, config, seed=args.seed)
    trainer.demo_metadata["execution_backend"] = "isaac_physx"
    trainer.demo_metadata["physics_dt"] = float(env.physics_dt)
    trainer.demo_metadata["training_seeds"] = [
        args.seed * 100_000 + episode for episode in range(args.episodes)
    ]
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("Cannot resume MATD3 from a non-Isaac checkpoint")
        if payload.get("demo_metadata", {}).get("experiment_variant") != args.variant:
            raise ValueError("Cannot resume MATD3 from a different experiment variant")
        trainer.load_checkpoint(args.checkpoint)
        trainer.cfg.episodes = args.episodes
    if args.actor_init:
        payload = torch.load(args.actor_init, map_location="cpu", weights_only=False)
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("MATD3 actor initialization must be collected/trained in Isaac PhysX")
        source_env = QuadrotorPursuitConfig(**payload["env_config"])
        if source_env.capture_required_uavs != 2 or source_env.execution_reward_version != 5:
            raise ValueError("MATD3 actor initialization must use the dual-capture reward-v5 task")
        trainer.load_actor_weights(mean_actor_state_from_checkpoint(payload))
        trainer.demo_metadata["actor_init"] = str(args.actor_init)
        trainer.demo_metadata["actor_init_algorithm"] = payload.get("algorithm")
    if args.prior:
        from pursuit.data.transition_dataset import load_transition_dataset

        trainer.load_prior(load_transition_dataset(args.prior, env.cfg, expected_backend="isaac_physx"))
    trainer.demo_metadata["experiment_variant"] = args.variant or "A5"
    args.out.mkdir(parents=True, exist_ok=True)
    trainer.output_directory = args.out
    history = trainer.train(np.random.default_rng(args.seed))
    trainer.save_checkpoint(args.out / "final.pt", episode=trainer.cfg.episodes)
    return {"algorithm": trainer.algorithm_name, "episodes": len(history),
            "env_steps": trainer.env_steps, "checkpoint": str(args.out / "final.pt")}


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.variant is None:
        args.variant = "mappo_scratch" if args.algorithm == "mappo" else "A5"
    if args.episodes < 1 or args.hidden_dim < 1 or args.batch_size < 1 or args.max_steps < 0:
        parser.error("episodes, hidden-dim, and batch-size must be positive; max-steps cannot be negative")
    if args.algorithm == "mappo" and args.variant == "mappo_dagger":
        if not args.actor_init or not args.demos:
            parser.error("mappo_dagger requires Isaac-trained --actor-init and --demos")
    if args.algorithm == "mappo" and args.variant == "mappo_scratch" and any(
        (args.actor_init, args.demos, args.aux_demos)
    ):
        parser.error("mappo_scratch cannot load actor-init or demonstration datasets")
    if args.algorithm == "mappo" and args.aux_demos and not args.demos:
        parser.error("--aux-demos requires its matching --demos source")
    if args.algorithm == "mappo" and args.prior:
        parser.error("--prior is supported only for MATD3")
    if args.variant and args.algorithm == "matd3" and args.variant.startswith("mappo_"):
        parser.error("mappo_scratch requires algorithm=mappo")
    if args.algorithm == "mappo" and args.variant not in ("mappo_scratch", "mappo_dagger"):
        parser.error("A0-A5 variants require algorithm=matd3")
    if args.checkpoint and args.actor_init:
        parser.error("--checkpoint and --actor-init are mutually exclusive")
    env_cfg = _load_env_config(args.env_config)
    expected_profile = (
        "radar5_dual_ablation_no_visibility_medium.json"
        if args.variant == "A4" else "radar5_dual_medium.json"
    )
    if args.env_config.name != expected_profile:
        parser.error(f"This suite variant must train on {expected_profile}")
    if args.variant in ("A4",) and env_cfg.visibility_progress_weight != 0:
        parser.error("A4 requires the dual Medium no-visibility config")
    if args.algorithm == "matd3" and args.variant in ("A0", "A1", "A2", "A3", "A4", "A5"):
        try:
            _variant_config(args.variant, args, env_cfg)
        except ValueError as error:
            parser.error(str(error))
    torch.set_num_threads(max(1, args.torch_threads))
    args.out.mkdir(parents=True, exist_ok=True)
    from isaac_pursuit_backend import IsaacPursuitEnv

    env = IsaacPursuitEnv(
        env_cfg,
        seed=args.seed,
        headless=args.headless,
        native_lidar=args.native_lidar,
    )
    backend_metadata = {
        "backend": "omnidrones_isaac_sim",
        "training_algorithm": args.algorithm,
        "scene_count": 1,
        "uavs_per_scene": 3,
        "native_physx_lidar": args.native_lidar,
        "headless": args.headless,
        "physics_dt": env.physics_dt,
        "decision_dt": env.cfg.decision_dt,
        "training_seed": args.seed,
        "env_config": asdict(env_cfg),
        "experiment_variant": args.variant or ("mappo_scratch" if args.algorithm == "mappo" else "A5"),
        "train_domain": "medium",
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