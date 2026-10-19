"""Evaluate sample-efficiency budget checkpoints on Dual Medium (one Isaac App)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from artifact_paths import artifact_path
from evaluate_isaac_pursuit import evaluate_level
from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("mappo", "matd3"), required=True)
    parser.add_argument("--run-dir", type=artifact_path, required=True,
                        help="Training output dir containing budgets/epXXXX.pt")
    parser.add_argument("--budgets", default="250,500,750,1000")
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--out", type=artifact_path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    if not budgets or args.seed_count < 1:
        parser.error("need positive budgets and seed-count")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    medium = Path(__file__).parent / "configs/pursuit_v2/radar5_dual_medium.json"
    env_cfg = QuadrotorPursuitConfig(**json.loads(medium.read_text(encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    env = IsaacPursuitEnv(env_cfg, seed=seeds[0], headless=args.headless, native_lidar=False)
    summaries = []
    try:
        for budget in budgets:
            ckpt = args.run_dir / "budgets" / f"ep{budget:04d}.pt"
            if not ckpt.is_file():
                # Fall back to final.pt when the last budget equals full training.
                alt = args.run_dir / "final.pt"
                if budget == max(budgets) and alt.is_file():
                    ckpt = alt
                else:
                    print(f"[budget-eval] missing {ckpt}, skip", flush=True)
                    continue
            result = evaluate_level(
                args.method, ckpt, "medium", seeds, args.out / f"ep{budget:04d}", args.device, env, 0,
            )
            row = {"budget_episodes": budget, "checkpoint": str(ckpt), **result["summary"]}
            summaries.append(row)
            print(json.dumps(row, indent=2), flush=True)
    finally:
        env.close()
    report = {
        "method": args.method,
        "run_dir": str(args.run_dir),
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "summaries": summaries,
    }
    (args.out / "budget_curve.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
