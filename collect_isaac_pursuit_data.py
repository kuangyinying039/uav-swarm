"""Collect teacher transition data in the Isaac PhysX backend."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit.data.transition_dataset import (
    collect_teacher_transitions,
    json_sidecar,
    prior_is_complete,
    progress_path_for,
)
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path,
                        default=Path("configs/pursuit_v2/radar5_dual_medium.json"))
    parser.add_argument("--teacher", choices=("mpc", "frpn"), default="mpc")
    parser.add_argument("--demo-seed", type=int, default=2_000_000)
    parser.add_argument(
        "--rollouts",
        type=int,
        default=200,
        help="Target number of episodes stored in the final prior (kept episodes)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
        help="Max teacher rollouts while filling --rollouts (0 → 4× rollouts when "
             "discarding failures, else = rollouts)",
    )
    parser.add_argument("--dataset", type=artifact_path, required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--native-lidar",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use Isaac PhysX range sensors (default: geometric LiDAR fallback)",
    )
    parser.add_argument(
        "--physx-closed-loop",
        action="store_true",
        help="Collect under pure PhysX 3+1 closed loop (sensing defaults to geometric)",
    )
    parser.add_argument(
        "--keep-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, store timeout/failure episodes (pollutes prior; off by default)",
    )
    parser.add_argument(
        "--keep-near-misses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also keep unsuccessful episodes whose closest capture gap ≤ --near-miss-gap",
    )
    parser.add_argument(
        "--near-miss-gap",
        type=float,
        default=1.5,
        help="Closest-gap threshold (m) for --keep-near-misses (default 1.5 ≈ radius+margin)",
    )
    parser.add_argument(
        "--min-successes",
        type=int,
        default=0,
        help="Refuse to finalize unless at least this many successful captures are kept "
             "(0 → equals --rollouts when failures are discarded)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore an existing complete dataset and recollect",
    )
    return parser


def _atomic_finalize(dataset, dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = dataset_path.with_name(dataset_path.name + ".tmp")
    torch.save(dataset, temporary)
    os.replace(temporary, dataset_path)
    sidecar = dataset_path.with_suffix(".json")
    sidecar_tmp = Path(str(sidecar) + ".tmp")
    sidecar_tmp.write_text(json_sidecar(dataset), encoding="utf-8")
    os.replace(sidecar_tmp, sidecar)


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.rollouts < 1:
        parser.error("--rollouts must be positive")
    if args.min_successes < 0:
        parser.error("--min-successes must be >= 0")
    min_successes = int(args.min_successes)
    if min_successes == 0 and not args.keep_failures:
        min_successes = int(args.rollouts)
    if min_successes > args.rollouts and not args.keep_failures:
        parser.error("--min-successes cannot exceed --rollouts when failures are discarded")
    max_attempts = int(args.max_attempts)
    if max_attempts <= 0:
        max_attempts = int(args.rollouts if args.keep_failures else max(args.rollouts * 4, args.rollouts))

    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    if cfg.capture_required_uavs != 2 or cfg.execution_reward_version != 5:
        parser.error("Expert data must use the dual-capture reward-v5 task")

    dataset_path = Path(args.dataset)
    if (
        not args.force
        and prior_is_complete(
            dataset_path,
            min_episodes=args.rollouts,
            min_successes=min_successes,
            allow_failures=bool(args.keep_failures or args.keep_near_misses),
        )
    ):
        print(
            f"[isaac-collect] skip existing clean prior={dataset_path} "
            f"(episodes>={args.rollouts}, successes>={min_successes}, "
            f"keep_failures={args.keep_failures}, keep_near_misses={args.keep_near_misses})",
            flush=True,
        )
        return

    # Never write partial progress into the final artifact path — that is how the
    # previous 75/200 interrupted collect was mistaken for a finished prior.
    progress_path = progress_path_for(dataset_path)
    if dataset_path.is_file() and not prior_is_complete(
        dataset_path,
        min_episodes=args.rollouts,
        min_successes=min_successes,
        allow_failures=bool(args.keep_failures or args.keep_near_misses),
    ):
        polluted = dataset_path.with_name(
            f"{dataset_path.stem}.polluted_incomplete{dataset_path.suffix}"
        )
        print(
            f"[isaac-collect] archiving incomplete/polluted prior {dataset_path} → {polluted}",
            flush=True,
        )
        os.replace(dataset_path, polluted)
        sidecar = dataset_path.with_suffix(".json")
        if sidecar.is_file():
            os.replace(sidecar, polluted.with_suffix(".json"))

    physx_closed = bool(args.physx_closed_loop)
    # Teacher collects stay on geometric LiDAR by default: native range sensors
    # are flaky under long rollouts and previously yielded 0/200 successes.
    env = IsaacPursuitEnv(
        cfg,
        seed=args.demo_seed,
        headless=args.headless,
        native_lidar=bool(args.native_lidar),
        physx_closed_loop=physx_closed,
    )
    try:
        print(
            f"[isaac-collect] target_kept={args.rollouts} min_successes={min_successes} "
            f"max_attempts={max_attempts} keep_failures={args.keep_failures} "
            f"keep_near_misses={args.keep_near_misses} near_miss_gap={args.near_miss_gap} "
            f"progress={progress_path}",
            flush=True,
        )
        dataset = collect_teacher_transitions(
            cfg,
            args.demo_seed,
            args.rollouts,
            args.teacher,
            progress_path=progress_path,
            env_factory=lambda _seed: env,
            execution_backend="isaac_physx_closed" if physx_closed else "isaac_physx",
            keep_failures=bool(args.keep_failures),
            keep_near_misses=bool(args.keep_near_misses),
            near_miss_gap=float(args.near_miss_gap),
            max_attempts=max_attempts,
        )
        n_success = sum(1 for row in dataset.get("report", []) if row.get("success"))
        if n_success < min_successes:
            raise RuntimeError(
                f"teacher collect produced n_successes={n_success} "
                f"< min_successes={min_successes}; refusing to finalize prior"
            )
        if not prior_is_complete(
            dataset,
            min_episodes=args.rollouts,
            min_successes=min_successes,
            allow_failures=bool(args.keep_failures or args.keep_near_misses),
        ):
            raise RuntimeError("collected dataset failed completeness / cleanliness checks")
        _atomic_finalize(dataset, dataset_path)
        print(json_sidecar(dataset), flush=True)
        print(f"[isaac-collect] finalized clean prior={dataset_path}", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
