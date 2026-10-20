"""Collect teacher transition data in the Isaac PhysX backend."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit.data.transition_dataset import collect_teacher_transitions, json_sidecar
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_medium.json"))
    parser.add_argument("--teacher", choices=("mpc", "frpn"), default="mpc")
    parser.add_argument("--demo-seed", type=int, default=2_000_000)
    parser.add_argument("--rollouts", type=int, default=200)
    parser.add_argument("--dataset", type=artifact_path, required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-lidar", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    if cfg.capture_required_uavs != 2 or cfg.execution_reward_version != 5:
        parser.error("Expert data must use the dual-capture reward-v5 task")

    env = IsaacPursuitEnv(
        cfg,
        seed=args.demo_seed,
        headless=args.headless,
        native_lidar=args.native_lidar,
    )
    try:
        dataset = collect_teacher_transitions(
            cfg,
            args.demo_seed,
            args.rollouts,
            args.teacher,
            progress_path=args.dataset,
            env_factory=lambda _seed: env,
            execution_backend="isaac_physx",
        )
        args.dataset.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dataset, args.dataset)
        args.dataset.with_suffix(".json").write_text(
            json_sidecar(dataset), encoding="utf-8"
        )
        print(json_sidecar(dataset), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
