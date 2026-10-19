"""Build an Isaac-trained MPC-BC/DAgger actor for MAPPO and MATD3 warm starts."""
from __future__ import annotations

import argparse
import json
import os
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


def _atomic_torch_save(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_dagger_progress(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        print(f"[isaac-imitation] ignore unreadable progress {path}: {error}", flush=True)
        return {}
    if int(payload.get("version", 0)) != 1:
        print(f"[isaac-imitation] ignore progress with an unexpected version: {path}", flush=True)
        return {}
    return payload


def main():
    parser = build_parser()
    args = parser.parse_args()
    if min(args.bc_episodes, args.bc_updates, args.batch_size, args.hidden_dim) < 1:
        parser.error("BC episodes/updates, batch size and hidden dimension must be positive")
    if args.dagger_rounds < 0 or args.dagger_episodes < 1 or args.dagger_updates < 1:
        parser.error("DAgger rounds must be non-negative and episode/update counts positive")
    if not 0.0 <= args.dagger_correction_fraction <= 1.0:
        parser.error("--dagger-correction-fraction must be in [0, 1]")

    manifest_path = args.out / "imitation_manifest.json"
    if (args.out / "best_imitation.pt").is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            int(manifest.get("bc_episodes", -1)) == args.bc_episodes
            and int(manifest.get("dagger_rounds", -1)) == args.dagger_rounds
            and int(manifest.get("dagger_episodes_per_round", -1)) == args.dagger_episodes
        ):
            print(f"[isaac-imitation] skip existing checkpoint={args.out / 'best_imitation.pt'}", flush=True)
            return

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
        progress_path = args.out / "dagger_progress.pt"
        actor_path = args.out / "dagger_latest.pt"
        progress = _load_dagger_progress(progress_path)
        expert_episodes = list(progress.get("expert_episodes") or [])
        bc_pretrained = bool(progress.get("bc_pretrained"))
        correction_episodes = list(progress.get("correction_episodes") or [])
        correction_report = list(progress.get("correction_report") or [])
        completed_rounds = int(progress.get("completed_dagger_rounds") or 0)
        open_round = progress.get("open_round")
        open_episodes = list(progress.get("open_round_episodes") or [])
        open_report = list(progress.get("open_round_report") or [])
        if expert_episodes or bc_pretrained or completed_rounds or open_episodes:
            print(
                f"[isaac-imitation] resume bc={len(expert_episodes)}/{args.bc_episodes} "
                f"bc_pretrained={bc_pretrained} rounds={completed_rounds}/{args.dagger_rounds} "
                f"open={0 if open_round is None else len(open_episodes)}",
                flush=True,
            )
        if bc_pretrained and actor_path.is_file():
            trainer.load_checkpoint(actor_path)

        def save_progress():
            _atomic_torch_save({
                "version": 1,
                "expert_episodes": expert_episodes,
                "bc_pretrained": bc_pretrained,
                "correction_episodes": correction_episodes,
                "correction_report": correction_report,
                "completed_dagger_rounds": completed_rounds,
                "open_round": open_round,
                "open_round_episodes": open_episodes,
                "open_round_report": open_report,
            }, progress_path)
            if bc_pretrained:
                trainer.save_checkpoint(actor_path, episode=0, history=[])

        expert = CooperativeGuidanceMPC3D()
        for episode in range(len(expert_episodes), args.bc_episodes):
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
            save_progress()
            print(f"[isaac-mpc-bc] episode={episode + 1}/{args.bc_episodes} steps={len(trajectory)}", flush=True)
        trainer.demo_episodes = expert_episodes
        if not bc_pretrained:
            pretrain(trainer, expert_episodes, args.bc_updates, args.batch_size, args.seed)
            bc_pretrained = True
            save_progress()

        for round_index in range(completed_rounds, args.dagger_rounds):
            beta = 0.5 * (1.0 - round_index / max(args.dagger_rounds - 1, 1))
            if open_round != round_index:
                open_round = round_index
                open_episodes = []
                open_report = []
                save_progress()
            trainer.actor.eval()
            while len(open_episodes) < args.dagger_episodes:
                episode = len(open_episodes)
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
                open_episodes.append(trajectory)
                open_report.append({
                    "round": round_index + 1,
                    "seed": seed,
                    "teacher_probability": beta,
                    "steps": len(trajectory),
                })
                save_progress()
            correction_episodes.extend(open_episodes)
            correction_report.extend(open_report)
            open_round = None
            open_episodes = []
            open_report = []
            pretrain_mixed(
                trainer,
                expert_episodes,
                correction_episodes,
                args.dagger_updates,
                args.batch_size,
                args.dagger_correction_fraction,
                args.seed + round_index + 1,
            )
            completed_rounds = round_index + 1
            trainer.demo_metadata["dagger_round"] = completed_rounds
            trainer.demo_metadata["dagger_correction_seeds"] = [
                row["seed"] for row in correction_report
            ]
            save_progress()
            print(f"[isaac-dagger] round={completed_rounds}/{args.dagger_rounds} episodes={args.dagger_episodes}", flush=True)

        trainer.demo_metadata["demo_seeds"] = list(range(args.demo_seed, args.demo_seed + args.bc_episodes))
        trainer.demo_metadata["correction_report"] = correction_report
        trainer.demo_metadata["dagger_correction_fraction"] = args.dagger_correction_fraction
        _atomic_torch_save({
            "execution_backend": "isaac_physx",
            "env_config": cfg.__dict__,
            "episodes": expert_episodes,
            "teacher": "mpc",
        }, args.out / "expert_episodes.pt")
        _atomic_torch_save({
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
