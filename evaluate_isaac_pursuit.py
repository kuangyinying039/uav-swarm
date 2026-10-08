"""Evaluate one frozen policy across the four Isaac dual-capture domains."""
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


LEVELS = ("nominal", "medium", "hard", "extreme")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("apf", "frpn", "mpc", "mappo", "matd3"), required=True)
    parser.add_argument("--checkpoint", type=artifact_path,
                        help="Required for MAPPO/MATD3; never use an old kinematic checkpoint")
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Short diagnostic cap; capped results are marked smoke-only")
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--native-lidar", action="store_true")
    return parser


def load_policy(method, checkpoint, env, device):
    payload = None
    if method in ("mappo", "matd3"):
        if checkpoint is None:
            raise ValueError("learned policy evaluation requires --checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        saved_env = QuadrotorPursuitConfig(**payload["env_config"])
        if saved_env.capture_required_uavs != 2 or saved_env.execution_reward_version != 5:
            raise ValueError("Checkpoint is not from the dual-capture reward-v5 experiment")
        if saved_env.task_mode != "pursuit_quadrotor_3d":
            raise ValueError("Checkpoint is not compatible with the Isaac quadrotor pursuit task")
        if payload.get("demo_metadata", {}).get("execution_backend") != "isaac_physx":
            raise ValueError(
                "Checkpoint is not marked as Isaac PhysX-trained; retrain it in this simulator "
                "instead of reusing a kinematic checkpoint"
            )

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


def evaluate_level(method, checkpoint, level, seeds, out_dir, device, env, max_steps=0):
    config_path = Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{level}.json"
    env_config = QuadrotorPursuitConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    if env_config.capture_required_uavs != 2 or env_config.execution_reward_version != 5:
        raise ValueError(f"{config_path} is not a dual-capture reward-v5 profile")

    env.configure_domain(env_config)
    rows = []
    action_fn, policy = load_policy(method, checkpoint, env, device)
    try:
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
            started = time.perf_counter()
            result = None
            limit = min(env.cfg.search_steps, max_steps) if max_steps else env.cfg.search_steps
            for step in range(limit):
                action = np.asarray(action_fn(observation), dtype=np.float32)
                result = env.step_joint(action)
                observation = result["obs"]
                episode_return += float(result["reward"])
                collisions += int(result.get("collisions", 0))
                intervention_sum += float(result.get("safety_correction_rate", 0.0))
                closest_gap = min(closest_gap, float(result["minimum_capture_gap"]))
                if result["terminated"] or result["truncated"]:
                    break
            if result is None:
                raise RuntimeError("evaluation produced no environment steps")
            rows.append({
                "method": method,
                "level": level,
                "seed": int(seed),
                "captured": bool(result["capture_success"]),
                "steps": step + 1,
                "return": episode_return,
                "collisions": collisions,
                "mean_safety_correction_rate": intervention_sum / (step + 1),
                "closest_capture_gap": closest_gap,
                "final_capture_gap": env.minimum_capture_gap(),
                "smoke_step_cap_hit": bool(
                    max_steps and step + 1 >= limit
                    and not result["terminated"] and not result["truncated"]
                ),
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(
                f"[isaac-eval] method={method} level={level} seed={seed} "
                f"capture={rows[-1]['captured']} steps={step + 1}",
                flush=True,
            )
    finally:
        if hasattr(policy, "close"):
            policy.close()

    successes = int(sum(bool(row["captured"]) for row in rows))
    count = len(rows)
    rate = successes / count
    z = 1.959963984540054
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    half = z * np.sqrt(rate * (1.0 - rate) / count + z * z / (4.0 * count * count)) / denominator

    summary = {
        "method": method,
        "level": level,
        "episodes": len(rows),
        "smoke_only": bool(max_steps),
        "capture_rate": rate,
        "capture_rate_wilson_95": [float(max(0.0, center - half)), float(min(1.0, center + half))],
        "mean_censored_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_success_steps": float(np.mean([row["steps"] for row in rows if row["captured"]]))
        if any(row["captured"] for row in rows) else None,
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_collisions": float(np.mean([row["collisions"] for row in rows])),
        "mean_safety_correction_rate": float(
            np.mean([row["mean_safety_correction_rate"] for row in rows])
        ),
    }
    result = {
        "method": method,
        "level": level,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "seeds": list(map(int, seeds)),
        "env_config": asdict(env_config),
        "summary": summary,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{level}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("a", encoding="utf-8") as stream:
        if stream.tell() == 0:
            stream.write("method,level,episodes,capture_rate,mean_censored_steps,mean_success_steps,mean_return,mean_collisions,mean_safety_correction_rate\n")
        stream.write(",".join(str(summary[key]) for key in (
            "method", "level", "episodes", "capture_rate", "mean_censored_steps",
            "mean_success_steps", "mean_return", "mean_collisions", "mean_safety_correction_rate",
        )) + "\n")
    return result


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.seed_count < 1 or args.max_steps < 0:
        parser.error("--seed-count must be positive and --max-steps nonnegative")
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        parser.error("--checkpoint is required for learned methods")
    if args.method in ("apf", "frpn", "mpc") and args.checkpoint is not None:
        parser.error("baseline methods do not accept --checkpoint")

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    summaries = []
    first_path = Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{args.levels[0]}.json"
    first_cfg = QuadrotorPursuitConfig(**json.loads(first_path.read_text(encoding="utf-8")))
    env = IsaacPursuitEnv(
        first_cfg,
        seed=seeds[0],
        headless=args.headless,
        native_lidar=args.native_lidar,
    )
    try:
        for level in args.levels:
            result = evaluate_level(
                args.method, args.checkpoint, level, seeds, args.out, args.device, env,
                args.max_steps,
            )
            summaries.append(result["summary"])
    finally:
        env.close()
    report = {
        "method": args.method,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "levels": list(args.levels),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "summaries": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
