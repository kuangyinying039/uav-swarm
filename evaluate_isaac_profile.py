"""Evaluate one policy on a single Isaac env profile (oracle / reward / capture ablations)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("apf", "frpn", "mpc", "mappo", "matd3"), required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=artifact_path)
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--tag", default="", help="Optional label written into summary.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-lidar", action="store_true")
    parser.add_argument(
        "--allow-non-dual",
        action="store_true",
        help="Permit single-capture / reward-v4 profiles (baselines only)",
    )
    return parser


def load_policy(method, checkpoint, env, device, *, require_dual=True):
    if method in ("mappo", "matd3"):
        if checkpoint is None:
            raise ValueError("learned policy evaluation requires --checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_env = QuadrotorPursuitConfig(**payload["env_config"])
        if require_dual and (
            saved_env.capture_required_uavs != 2 or saved_env.execution_reward_version != 5
        ):
            raise ValueError("Checkpoint is not from the dual-capture reward-v5 experiment")
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError("Checkpoint is not marked as Isaac PhysX-trained")
        if method == "mappo":
            from marl_trainers import TrainConfig
            from train_pursuit_with_demos import PursuitDemoTrainer, deterministic_action

            config = TrainConfig(**payload["train_config"])
            config.device = device
            trainer = PursuitDemoTrainer(lambda _episode=0: env, config)
            trainer.load_checkpoint(checkpoint)
            trainer.actor.eval()
            return lambda observation: deterministic_action(trainer, observation), trainer
        from pursuit.algorithms.matd3 import Matd3Config
        from pursuit.trainers.matd3_trainer import Matd3Trainer

        config = Matd3Config(**payload["train_config"])
        config.device = device
        trainer = Matd3Trainer(lambda _episode=0: env, config, seed=0)
        trainer.load_checkpoint(checkpoint)
        trainer.actor.eval()
        return lambda observation: trainer.deterministic_action(observation), trainer
    policy = BASELINES_3D[method]()
    return lambda observation: policy.actions(env), policy


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.seed_count < 1 or args.max_steps < 0:
        parser.error("invalid seed-count / max-steps")
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        parser.error("--checkpoint required for learned methods")
    if args.method in ("apf", "frpn", "mpc") and args.checkpoint is not None:
        parser.error("baselines do not accept --checkpoint")

    env_cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    is_dual = env_cfg.capture_required_uavs == 2 and env_cfg.execution_reward_version == 5
    if not is_dual and not args.allow_non_dual:
        parser.error(f"{args.env_config.name} is not dual reward-v5; pass --allow-non-dual for baselines")
    if not is_dual and args.method in ("mappo", "matd3"):
        parser.error("learned dual-trained policies should not be evaluated under single-capture contracts here")

    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    args.out.mkdir(parents=True, exist_ok=True)
    env = IsaacPursuitEnv(env_cfg, seed=seeds[0], headless=args.headless, native_lidar=args.native_lidar)
    rows = []
    try:
        action_fn, policy = load_policy(
            args.method, args.checkpoint, env, args.device, require_dual=is_dual,
        )
        if hasattr(policy, "reset"):
            policy.reset(env)
        for seed in seeds:
            observation = env.reset(seed=seed)
            if hasattr(policy, "reset"):
                policy.reset(env)
            episode_return = 0.0
            collisions = 0
            intervention_sum = 0.0
            closest_gap = env.minimum_capture_gap()
            role_hist = np.zeros(3, dtype=np.int64)
            started = time.perf_counter()
            result = None
            limit = min(env.cfg.search_steps, args.max_steps) if args.max_steps else env.cfg.search_steps
            for step in range(limit):
                if args.method == "mpc" and hasattr(policy, "plan"):
                    plan = policy.plan(env)
                    action = np.asarray(plan["actions"], dtype=np.float32)
                    roles = np.asarray(plan.get("teacher_role", []), dtype=np.int64)
                    for role in roles:
                        if 0 <= int(role) < 3:
                            role_hist[int(role)] += 1
                else:
                    action = np.asarray(action_fn(observation), dtype=np.float32)
                result = env.step_joint(action)
                observation = result["obs"]
                episode_return += float(result["reward"])
                collisions += int(result.get("collisions", 0))
                intervention_sum += float(result.get("safety_correction_rate", 0.0))
                closest_gap = min(closest_gap, float(result["minimum_capture_gap"]))
                if result["terminated"] or result["truncated"]:
                    break
            rows.append({
                "method": args.method,
                "seed": int(seed),
                "captured": bool(result["capture_success"]),
                "steps": step + 1,
                "return": episode_return,
                "collisions": collisions,
                "mean_safety_correction_rate": intervention_sum / (step + 1),
                "closest_capture_gap": closest_gap,
                "final_capture_gap": env.minimum_capture_gap(),
                "teacher_role_counts": role_hist.tolist(),
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(
                f"[profile-eval] tag={args.tag or args.env_config.stem} method={args.method} "
                f"seed={seed} capture={rows[-1]['captured']} steps={step + 1}",
                flush=True,
            )
        if hasattr(policy, "close"):
            policy.close()
    finally:
        env.close()

    successes = int(sum(bool(row["captured"]) for row in rows))
    count = len(rows)
    rate = successes / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator
    summary = {
        "tag": args.tag or args.env_config.stem,
        "method": args.method,
        "env_config": str(args.env_config),
        "episodes": count,
        "capture_rate": rate,
        "capture_rate_wilson_95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
        "mean_censored_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_safety_correction_rate": float(
            np.mean([row["mean_safety_correction_rate"] for row in rows])
        ),
        "teacher_role_totals": np.sum([row["teacher_role_counts"] for row in rows], axis=0).tolist()
        if args.method == "mpc" else None,
    }
    report = {
        "summary": summary,
        "env_config": asdict(env_cfg),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "seeds": seeds,
        "rows": rows,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
