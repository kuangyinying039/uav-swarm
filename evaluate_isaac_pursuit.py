"""Evaluate one frozen policy across the Isaac dual-capture domains."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from artifact_paths import artifact_path
from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig


LEVELS = ("nominal", "medium", "hard", "severe", "extreme")


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
    parser.add_argument(
        "--skip-existing-levels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse complete per-level JSON in --out (default: on). Speeds up restarts.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print one progress line every N episodes (0 = every episode)",
    )
    parser.add_argument(
        "--residual-base",
        choices=("apf", "frpn", "mpc"),
        default=None,
        help="If set with a learned method, mix classical base + RL (see --residual-mode)",
    )
    parser.add_argument(
        "--residual-mode",
        choices=("blend", "residual"),
        default="blend",
        help="blend: (1-w)·base + w·RL; residual: base + w·δ (for residual-trained ckpts)",
    )
    parser.add_argument(
        "--residual-weight",
        type=float,
        default=0.35,
        help="RL weight (blend) or delta scale (residual); default 0.35",
    )
    from isaac_cli_flags import add_isaac_display_flags

    add_isaac_display_flags(parser, default_headless=True)
    parser.add_argument("--native-lidar", action="store_true")
    return parser


def _read_partial_rows(path: Path, seeds) -> dict:
    """Rows already flushed for this level. A crashed line is ignored."""
    wanted = {int(seed) for seed in seeds}
    found = {}
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        seed = row.get("seed")
        if isinstance(seed, int) and seed in wanted and "captured" in row and "steps" in row:
            found[seed] = row
    return found


def _append_partial_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _level_result_complete(path: Path, seeds) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    want = list(map(int, seeds))
    if list(map(int, payload.get("seeds") or [])) != want:
        return None
    if len(payload.get("rows") or []) != len(want):
        return None
    if "summary" not in payload:
        return None
    return payload


def _rewrite_summary_csv(out_dir: Path, summaries) -> None:
    path = out_dir / "summary.csv"
    with path.open("w", encoding="utf-8") as stream:
        stream.write(
            "method,level,episodes,capture_rate,mean_censored_steps,mean_success_steps,"
            "mean_return,mean_collisions,mean_safety_correction_rate\n"
        )
        for summary in summaries:
            stream.write(
                ",".join(
                    str(summary[key])
                    for key in (
                        "method",
                        "level",
                        "episodes",
                        "capture_rate",
                        "mean_censored_steps",
                        "mean_success_steps",
                        "mean_return",
                        "mean_collisions",
                        "mean_safety_correction_rate",
                    )
                )
                + "\n"
            )


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


def evaluate_level(
    method,
    checkpoint,
    level,
    seeds,
    out_dir,
    device,
    env,
    max_steps=0,
    *,
    log_every=25,
    residual_base=None,
    residual_mode="blend",
    residual_weight=0.35,
):
    config_path = Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{level}.json"
    env_config = QuadrotorPursuitConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    if env_config.capture_required_uavs != 2 or env_config.execution_reward_version != 5:
        raise ValueError(f"{config_path} is not a dual-capture reward-v5 profile")

    env.configure_domain(env_config)
    partial_path = out_dir / f"{level}.partial.jsonl"
    resumed = _read_partial_rows(partial_path, seeds)
    rows = [resumed[int(seed)] for seed in seeds if int(seed) in resumed]
    pending = [int(seed) for seed in seeds if int(seed) not in resumed]
    if resumed:
        print(
            f"[isaac-eval] resume method={method} level={level} "
            f"kept={len(rows)}/{len(seeds)}",
            flush=True,
        )
    action_fn, policy = load_policy(method, checkpoint, env, device)
    method_tag = method
    if residual_base is not None:
        if method not in ("mappo", "matd3"):
            raise ValueError("--residual-base only applies to learned methods")
        from pursuit_residual import ResidualComposeConfig, wrap_action_fn

        action_fn = wrap_action_fn(
            action_fn,
            env,
            ResidualComposeConfig(base=residual_base, mode=residual_mode, weight=residual_weight),
        )
        method_tag = f"{method}+{residual_mode}:{residual_base}@{residual_weight:g}"
    try:
        if hasattr(policy, "reset"):
            policy.reset(env)
        for seed in pending:
            episode_index = len(rows) + 1
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
                "method": method_tag,
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
            _append_partial_row(partial_path, rows[-1])
            if (
                log_every <= 0
                or episode_index == 1
                or episode_index == len(seeds)
                or episode_index % int(log_every) == 0
            ):
                running = sum(1 for row in rows if row["captured"]) / len(rows)
                print(
                    f"[isaac-eval] method={method_tag} level={level} "
                    f"{episode_index}/{len(seeds)} seed={seed} "
                    f"capture={rows[-1]['captured']} steps={step + 1} "
                    f"rate_so_far={running:.3f}",
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
        "method": method_tag,
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
        "residual_base": residual_base,
        "residual_mode": residual_mode if residual_base else None,
        "residual_weight": residual_weight if residual_base else None,
    }
    result = {
        "method": method_tag,
        "level": level,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "seeds": list(map(int, seeds)),
        "env_config": asdict(env_config),
        "summary": summary,
        "rows": rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{level}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    os.replace(tmp_path, final_path)
    partial_path.unlink(missing_ok=True)
    print(
        f"[isaac-eval] done method={method_tag} level={level} "
        f"capture_rate={summary['capture_rate']:.3f} episodes={summary['episodes']}",
        flush=True,
    )
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
    if args.residual_base is not None and args.method not in ("mappo", "matd3"):
        parser.error("--residual-base requires --method mappo or matd3")
    if args.residual_weight < 0.0 or args.residual_weight > 1.5:
        parser.error("--residual-weight should be in [0, 1.5]")

    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    summaries = []
    from isaac_cli_flags import resolve_headless

    pending_levels = []
    for level in args.levels:
        cached = (
            _level_result_complete(args.out / f"{level}.json", seeds)
            if args.skip_existing_levels
            else None
        )
        if cached is not None:
            print(
                f"[isaac-eval] skip existing level={level} "
                f"capture_rate={cached['summary']['capture_rate']:.3f}",
                flush=True,
            )
            summaries.append(cached["summary"])
        else:
            pending_levels.append(level)

    if pending_levels:
        first_path = Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{pending_levels[0]}.json"
        first_cfg = QuadrotorPursuitConfig(**json.loads(first_path.read_text(encoding="utf-8")))
        headless = resolve_headless(args)
        env = IsaacPursuitEnv(
            first_cfg,
            seed=seeds[0],
            headless=headless,
            native_lidar=args.native_lidar,
        )
        try:
            for level in pending_levels:
                result = evaluate_level(
                    args.method,
                    args.checkpoint,
                    level,
                    seeds,
                    args.out,
                    args.device,
                    env,
                    args.max_steps,
                    log_every=args.log_every,
                    residual_base=args.residual_base,
                    residual_mode=args.residual_mode,
                    residual_weight=args.residual_weight,
                )
                summaries.append(result["summary"])
        finally:
            env.close()
    # Keep CSV in level order requested by --levels.
    ordered = []
    by_level = {row["level"]: row for row in summaries}
    for level in args.levels:
        if level in by_level:
            ordered.append(by_level[level])
    summaries = ordered
    _rewrite_summary_csv(args.out, summaries)
    report = {
        "method": args.method,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "residual_base": args.residual_base,
        "residual_mode": args.residual_mode if args.residual_base else None,
        "residual_weight": args.residual_weight if args.residual_base else None,
        "levels": list(args.levels),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "summaries": summaries,
    }
    (args.out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
