"""Build an Isaac-trained MPC-BC/DAgger actor for MAPPO and MATD3 warm starts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from marl_trainers import TrainConfig, seed_everything
from pursuit_baselines_3d import CooperativeGuidanceMPC3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig
from train_pursuit_with_demos import (
    PursuitDemoTrainer,
    deterministic_action,
    pretrain,
    pretrain_mixed,
    snapshot,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_medium.json"))
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--demo-seed", type=int, default=2_000_000)
    parser.add_argument("--bc-episodes", type=int, default=50)
    parser.add_argument("--bc-updates", type=int, default=1500)
    parser.add_argument("--dagger-rounds", type=int, default=2)
    parser.add_argument("--dagger-episodes", type=int, default=40)
    parser.add_argument("--dagger-updates", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dagger-correction-fraction", type=float, default=0.5)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-lidar", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.bc_episodes, args.bc_updates, args.batch_size, args.hidden_dim) < 1:
        parser.error("BC episodes/updates, batch size and hidden dimension must be positive")
    if args.dagger_rounds < 0 or args.dagger_episodes < 1 or args.dagger_updates < 1:
        parser.error("DAgger rounds must be non-negative and episode/update counts positive")
    if not 0.0 <= args.dagger_correction_fraction <= 1.0:
        parser.error("--dagger-correction-fraction must be in [0, 1]")

    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    if cfg.capture_required_uavs != 2 or cfg.execution_reward_version != 5:
        parser.error("MPC-BC/DAgger initialization must use dual-capture reward-v5")
    seed_everything(args.seed)
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)

    env = IsaacPursuitEnv(
        cfg, seed=args.demo_seed, headless=args.headless, native_lidar=args.native_lidar
    )
    try:
        train_cfg = TrainConfig(
            episodes=1,
            gamma=cfg.reward_gamma,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_gat=True,
            use_hetero_entities=True,
            continuous_initial_std=0.20,
            continuous_log_std_max=float(np.log(0.40)),
        )

        def env_factory(episode=0):
            env.reset(seed=args.demo_seed + int(episode))
            return env

        trainer = PursuitDemoTrainer(env_factory, train_cfg)
        trainer.demo_metadata.update({
            "execution_backend": "isaac_physx",
            "physics_dt": float(env.physics_dt),
            "initialization_type": "isaac_mpc_dagger",
            "teacher": "mpc",
            "bc_seed": int(args.demo_seed),
        })
        expert_episodes = []
        expert = CooperativeGuidanceMPC3D()
        for episode in range(args.bc_episodes):
            seed = args.demo_seed + episode
            observation = env.reset(seed=seed)
            expert.reset(env)
            trajectory = []
            for _ in range(cfg.search_steps):
                action = expert.actions(env)
                active = ~np.asarray(env.disabled_uavs, dtype=bool)
                trajectory.append(snapshot(observation, action, active))
                transition = env.step_joint(action)
                observation = transition["obs"]
                if transition["terminated"] or transition["truncated"]:
                    break
            expert_episodes.append(trajectory)
            print(f"[isaac-mpc-bc] episode={episode + 1}/{args.bc_episodes} steps={len(trajectory)}", flush=True)
        trainer.demo_episodes = expert_episodes
        pretrain(trainer, expert_episodes, args.bc_updates, args.batch_size, args.seed)

        correction_episodes = []
        correction_report = []
        for round_index in range(args.dagger_rounds):
            beta = 0.5 * (1.0 - round_index / max(args.dagger_rounds - 1, 1))
            round_episodes = []
            trainer.actor.eval()
            for episode in range(args.dagger_episodes):
                seed = args.demo_seed + 1_000_000 + round_index * args.dagger_episodes + episode
                rng = np.random.default_rng(seed)
                observation = env.reset(seed=seed)
                expert.reset(env)
                trajectory = []
                for _ in range(cfg.search_steps):
                    label = expert.actions(env)
                    active = ~np.asarray(env.disabled_uavs, dtype=bool)
                    trajectory.append(snapshot(observation, label, active))
                    learner_action = deterministic_action(trainer, observation)
                    action = label if rng.random() < beta else learner_action
                    transition = env.step_joint(action)
                    observation = transition["obs"]
                    if transition["terminated"] or transition["truncated"]:
                        break
                round_episodes.append(trajectory)
                correction_report.append({
                    "round": round_index + 1,
                    "seed": seed,
                    "teacher_probability": beta,
                    "steps": len(trajectory),
                })
            correction_episodes.extend(round_episodes)
            pretrain_mixed(
                trainer,
                expert_episodes,
                correction_episodes,
                args.dagger_updates,
                args.batch_size,
                args.dagger_correction_fraction,
                args.seed + round_index + 1,
            )
            trainer.demo_metadata["dagger_round"] = round_index + 1
            trainer.demo_metadata["dagger_correction_seeds"] = [
                row["seed"] for row in correction_report
            ]
            print(f"[isaac-dagger] round={round_index + 1}/{args.dagger_rounds} episodes={len(round_episodes)}", flush=True)

        trainer.demo_metadata["demo_seeds"] = list(range(args.demo_seed, args.demo_seed + args.bc_episodes))
        trainer.demo_metadata["correction_report"] = correction_report
        trainer.demo_metadata["dagger_correction_fraction"] = args.dagger_correction_fraction
        torch.save({
            "execution_backend": "isaac_physx",
            "env_config": cfg.__dict__,
            "episodes": expert_episodes,
            "teacher": "mpc",
        }, args.out / "expert_episodes.pt")
        torch.save({
            "execution_backend": "isaac_physx",
            "env_config": cfg.__dict__,
            "episodes": correction_episodes,
            "correction_report": correction_report,
            "dagger_round": args.dagger_rounds,
        }, args.out / "correction_episodes.pt")
        trainer.save_checkpoint(
            args.out / "best_imitation.pt",
            episode=0,
            history=[],
        )
        (args.out / "imitation_manifest.json").write_text(json.dumps({
            "execution_backend": "isaac_physx",
            "environment": cfg.__dict__,
            "bc_episodes": args.bc_episodes,
            "bc_updates": args.bc_updates,
            "dagger_rounds": args.dagger_rounds,
            "dagger_episodes_per_round": args.dagger_episodes,
            "dagger_updates_per_round": args.dagger_updates,
            "correction_report": correction_report,
            "checkpoint": str(args.out / "best_imitation.pt"),
        }, indent=2), encoding="utf-8")
        print(f"[isaac-imitation] checkpoint={args.out / 'best_imitation.pt'}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
