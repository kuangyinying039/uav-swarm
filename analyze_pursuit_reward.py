"""Read-only episode reward audit for pursuit checkpoints and baselines."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import torch

from marl_trainers import TrainConfig
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from train_pursuit_with_demos import PursuitDemoTrainer, deterministic_action, load_environment_config


AUDIT_COMPONENTS = (
    "capture", "timeout", "time", "individual_approach", "nearest_approach",
    "encirclement_progress", "obstacle_proximity", "boundary_proximity",
    "peer_proximity", "reference_smoothness", "controller_rejection",
    "safety_correction",
)


def run_reward_audit(env_cfg, seeds, trainer=None, baseline=None):
    """Evaluate without changing configuration, rewards, dynamics, or policy weights."""
    if (trainer is None) == (baseline is None):
        raise ValueError("Specify exactly one of trainer or baseline")
    rows = []
    actor_was_training = None
    if trainer is not None:
        actor_was_training = trainer.actor.training
        trainer.actor.eval()
    for seed in seeds:
        env = QuadrotorPursuitEnv(replace(env_cfg, seed=int(seed)))
        policy = BASELINES_3D[baseline]() if baseline is not None else None
        if policy is not None:
            policy.reset(env)
        obs = env.observe_search()
        total_return = 0.0
        closest_gap = float(env.minimum_capture_gap())
        components = {key: 0.0 for key in AUDIT_COMPONENTS}
        correction = emergency = 0.0
        for step in range(env.cfg.search_steps):
            actions = deterministic_action(trainer, obs) if trainer is not None else policy.actions(env)
            result = env.step_joint(actions)
            obs = result["obs"]
            total_return += float(result["reward"])
            closest_gap = min(closest_gap, float(result["minimum_capture_gap"]))
            correction += float(result.get("safety_correction_rate", 0.0))
            emergency += float(result.get("emergency_stop_rate", 0.0))
            for key in AUDIT_COMPONENTS:
                components[key] += float(result.get("reward_components", {}).get(key, 0.0))
            if result["terminated"] or result["truncated"]:
                break
        steps = step + 1
        rows.append({
            "seed": int(seed),
            "captured": bool(result["capture_success"]),
            "steps": steps,
            "total_return": total_return,
            "closest_capture_gap": closest_gap,
            "final_capture_gap": float(env.minimum_capture_gap()),
            "reward_components": components,
            "safety_correction_rate": correction / steps,
            "emergency_stop_rate": emergency / steps,
        })
    if trainer is not None:
        trainer.actor.train(actor_was_training)
    return {
        "env_config": asdict(env_cfg),
        "policy": "checkpoint" if trainer is not None else baseline,
        "rows": rows,
        "summary": {
            name: summarize(group)
            for name, group in (
                ("all", rows),
                ("successful", [row for row in rows if row["captured"]]),
                ("failed", [row for row in rows if not row["captured"]]),
            )
        },
    }


def summarize(rows):
    if not rows:
        return {"episodes": 0}
    return {
        "episodes": len(rows),
        "capture_rate": float(np.mean([row["captured"] for row in rows])),
        "mean_steps": float(np.mean([row["steps"] for row in rows])),
        "mean_total_return": float(np.mean([row["total_return"] for row in rows])),
        "mean_closest_capture_gap": float(np.mean([row["closest_capture_gap"] for row in rows])),
        "mean_final_capture_gap": float(np.mean([row["final_capture_gap"] for row in rows])),
        "mean_safety_correction_rate": float(np.mean([row["safety_correction_rate"] for row in rows])),
        "mean_emergency_stop_rate": float(np.mean([row["emergency_stop_rate"] for row in rows])),
        "mean_reward_components": {
            key: float(np.mean([row["reward_components"][key] for row in rows]))
            for key in AUDIT_COMPONENTS
        },
    }


def write_reward_audit(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "reward_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = [
        "seed", "captured", "steps", "total_return", "closest_capture_gap",
        "final_capture_gap", "safety_correction_rate", "emergency_stop_rate",
        *AUDIT_COMPONENTS,
    ]
    with (out / "reward_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({
                **{key: row[key] for key in fields if key not in AUDIT_COMPONENTS},
                **row["reward_components"],
            })


def load_checkpoint_trainer(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    env_cfg = load_environment_config(payload["env_config"])
    cfg = TrainConfig(**payload["train_config"])
    cfg.device = device
    trainer = PursuitDemoTrainer(lambda episode=0: QuadrotorPursuitEnv(env_cfg), cfg)
    trainer.load_checkpoint(path)
    return env_cfg, trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--baseline", choices=sorted(BASELINES_3D))
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--env-config", type=Path,
                        help="Required for a baseline; checkpoint embeds its environment")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", type=Path, default=Path("outputs/reward_audit"))
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds contains duplicates")
    if args.checkpoint:
        if args.env_config:
            parser.error("Checkpoint embeds its environment; omit --env-config")
        env_cfg, trainer = load_checkpoint_trainer(args.checkpoint, args.device)
        result = run_reward_audit(env_cfg, args.seeds, trainer=trainer)
    else:
        overrides = json.loads(args.env_config.read_text(encoding="utf-8")) if args.env_config else {}
        env_cfg = QuadrotorPursuitConfig(**overrides)
        result = run_reward_audit(env_cfg, args.seeds, baseline=args.baseline)
    write_reward_audit(result, args.out)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
