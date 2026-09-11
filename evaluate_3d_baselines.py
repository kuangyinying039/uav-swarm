"""Evaluate 3-D APF, fast-response PN, and MPC against the occlusion evader."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import time

import numpy as np

try:
    from pursuit_baselines_3d import BASELINES_3D
    from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
except ImportError:
    from .pursuit_baselines_3d import BASELINES_3D
    from .quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def run_episode(method: str, seed: int, args) -> dict:
    overrides = json.loads(args.env_config.read_text(encoding="utf-8")) if args.env_config else {}
    overrides.update(seed=seed, search_steps=args.steps)
    cfg = QuadrotorPursuitConfig(**overrides)
    env = QuadrotorPursuitEnv(cfg)
    policy = BASELINES_3D[method]()
    policy.reset(env)
    initial_target = np.array([*env.dynamic_targets[0], env.target_altitudes[0]])
    initial_distance = float(np.mean(np.linalg.norm(env.quadrotor_states[:, :3] - initial_target, axis=1)))
    totals = dict(reward=0.0, visible=0.0, lost=0.0, reacquired=0.0, collisions=0.0,
                  interventions=0.0, feasible=0.0, path_length=0.0, correction=0.0, emergency=0.0)
    result = {}
    executed = 0
    for step in range(args.steps):
        result = env.step_joint(policy.actions(env))
        executed = step + 1
        totals["reward"] += float(result["reward"])
        totals["visible"] += float(result.get("direct_target_visible", 0.0))
        totals["lost"] += float(result.get("target_lost", 0.0))
        totals["reacquired"] += float(result.get("target_reacquired", 0.0))
        totals["collisions"] += float(result.get("collisions", 0.0))
        totals["interventions"] += float(result.get("continuous_safety_interventions", 0.0))
        totals["feasible"] += float(result.get("controller_feasible_rate", 0.0))
        totals['correction'] += float(result.get('safety_correction_rate', 0.))
        totals['emergency'] += float(result.get('emergency_stop_rate', 0.))
        totals["path_length"] += float(result.get("step_path_length", 0.0))
        if result.get("terminated") or result.get("truncated"):
            break
    captured = bool(result.get("capture_success", 0.0))
    return {
        "method": method,
        "seed": seed,
        "captured": int(captured),
        "capture_time": int(result.get("capture_time", 0)) if captured else None,
        "censored_time": executed,
        "initial_mean_distance": initial_distance,
        "initial_layout": env.initial_layout,
        "evader_safety_interventions": getattr(env, "evader_safety_interventions", 0),
        "visibility_ratio": totals["visible"] / max(executed, 1),
        "target_loss_count": totals["lost"],
        "reacquisition_count": totals["reacquired"],
        "collision_count": totals["collisions"],
        "safety_interventions": totals["interventions"],
        "controller_feasible_rate": totals["feasible"] / max(executed, 1),
        "safety_correction_rate": totals['correction'] / max(executed, 1),
        "emergency_stop_rate": totals['emergency'] / max(executed, 1),
        "path_length": totals["path_length"],
        "return": totals["reward"],
    }


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for method in sorted({row["method"] for row in rows}):
        group = [row for row in rows if row["method"] == method]
        captured_times = [row["capture_time"] for row in group if row["capture_time"] is not None]
        summary[method] = {
            "episodes": len(group),
            "capture_rate": float(np.mean([row["captured"] for row in group])),
            "mean_capture_time_success_only": float(np.mean(captured_times)) if captured_times else None,
            "mean_censored_time": float(np.mean([row["censored_time"] for row in group])),
            "mean_visibility_ratio": float(np.mean([row["visibility_ratio"] for row in group])),
            "mean_safety_interventions": float(np.mean([row["safety_interventions"] for row in group])),
            "mean_controller_feasible_rate": float(np.mean([row["controller_feasible_rate"] for row in group])),
            "mean_safety_correction_rate": float(np.mean([row["safety_correction_rate"] for row in group])),
            "mean_emergency_stop_rate": float(np.mean([row["emergency_stop_rate"] for row in group])),
            "mean_return": float(np.mean([row["return"] for row in group])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", choices=sorted(BASELINES_3D), default=["apf", "frpn", "mpc"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[23, 37, 51, 71, 89])
    parser.add_argument("--seeds-file", type=Path,
                        help="Whitespace-delimited integer seeds; replaces --seeds")
    parser.add_argument("--seed-start", type=int,
                        help="First seed in a contiguous candidate range")
    parser.add_argument("--seed-count", type=int,
                        help="Number of seeds in --seed-start range")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1, help="Parallel independent episodes.")
    parser.add_argument("--out", type=Path, default=Path("outputs/three_dimensional_baseline_difficulty.json"))
    parser.add_argument("--env-config", type=Path, help="Same calibration overrides as training")
    args = parser.parse_args()
    if (args.seed_start is None) != (args.seed_count is None):
        parser.error("Use --seed-start and --seed-count together")
    if args.seeds_file and args.seed_start is not None:
        parser.error("Use either --seeds-file or --seed-start/--seed-count")
    if args.seed_start is not None:
        if args.seed_count < 1:
            parser.error("--seed-count must be positive")
        args.seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    if args.seeds_file:
        try:
            args.seeds = [int(token) for token in args.seeds_file.read_text(encoding="utf-8-sig").split()]
        except (OSError, ValueError) as error:
            parser.error(f"Cannot read integer seeds from {args.seeds_file}: {error}")
        if not args.seeds:
            parser.error(f"Seed file is empty: {args.seeds_file}")
        if len(args.seeds) != len(set(args.seeds)):
            parser.error(f"Seed file contains duplicates: {args.seeds_file}")
    jobs = [(method, seed) for method in args.methods for seed in args.seeds]
    started_at = time.perf_counter()

    def report_progress(completed: int, row: dict) -> None:
        elapsed = time.perf_counter() - started_at
        eta = elapsed / max(completed, 1) * (len(jobs) - completed)
        print(
            f"[baseline] {completed}/{len(jobs)} "
            f"method={row['method']} seed={row['seed']} "
            f"captured={row['captured']} steps={row['censored_time']} "
            f"elapsed={elapsed / 3600:.2f}h eta={eta / 3600:.2f}h",
            flush=True,
        )

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(run_episode, method, seed, args): (method, seed)
                for method, seed in jobs
            }
            rows = []
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                report_progress(completed, row)
    else:
        rows = []
        for completed, (method, seed) in enumerate(jobs, start=1):
            row = run_episode(method, seed, args)
            rows.append(row)
            report_progress(completed, row)
    method_order = {method: index for index, method in enumerate(args.methods)}
    rows.sort(key=lambda row: (method_order[row["method"]], row["seed"]))
    payload = {"protocol": "3-D velocity/yaw-rate pursuit; identical noisy observation and active escape configuration", "env_config": vars(QuadrotorPursuitConfig(**(json.loads(args.env_config.read_text(encoding="utf-8")) if args.env_config else {}))), "rows": rows, "summary": summarize(rows)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = args.out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
