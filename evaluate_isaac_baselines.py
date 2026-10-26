"""Evaluate APF/FRPN/MPC in one Isaac app (avoids repeated Kit startups)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from artifact_paths import artifact_path
from evaluate_isaac_pursuit import (
    LEVELS,
    _level_result_complete,
    _rewrite_summary_csv,
    evaluate_level,
)
from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=["apf", "frpn", "mpc"],
                        choices=("apf", "frpn", "mpc"))
    parser.add_argument("--levels", nargs="+", choices=LEVELS, default=list(LEVELS))
    parser.add_argument("--seed-start", type=int, default=7_000_000)
    parser.add_argument("--seed-count", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--out-root", type=artifact_path, required=True,
                        help="Writes isaac_dual_eval_{method}/ under this directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--skip-existing-levels",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--log-every", type=int, default=25)
    from isaac_cli_flags import add_isaac_display_flags

    add_isaac_display_flags(parser, default_headless=True)
    parser.add_argument("--native-lidar", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.seed_count < 1 or args.max_steps < 0:
        raise SystemExit("--seed-count must be positive and --max-steps nonnegative")
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    from isaac_cli_flags import resolve_headless

    # Build work list and skip complete method/level pairs.
    work = []
    cached_by_method = {method: [] for method in args.methods}
    for method in args.methods:
        out_dir = args.out_root / f"isaac_dual_eval_{method}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for level in args.levels:
            cached = (
                _level_result_complete(out_dir / f"{level}.json", seeds)
                if args.skip_existing_levels
                else None
            )
            if cached is not None:
                print(
                    f"[isaac-eval] skip existing method={method} level={level} "
                    f"capture_rate={cached['summary']['capture_rate']:.3f}",
                    flush=True,
                )
                cached_by_method[method].append(cached["summary"])
            else:
                work.append((method, level, out_dir))

    if work:
        first_level = work[0][1]
        first_path = Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{first_level}.json"
        first_cfg = QuadrotorPursuitConfig(**json.loads(first_path.read_text(encoding="utf-8")))
        env = IsaacPursuitEnv(
            first_cfg,
            seed=seeds[0],
            headless=resolve_headless(args),
            native_lidar=args.native_lidar,
        )
        try:
            for method, level, out_dir in work:
                result = evaluate_level(
                    method,
                    None,
                    level,
                    seeds,
                    out_dir,
                    args.device,
                    env,
                    args.max_steps,
                    log_every=args.log_every,
                )
                cached_by_method[method].append(result["summary"])
        finally:
            env.close()

    for method in args.methods:
        out_dir = args.out_root / f"isaac_dual_eval_{method}"
        by_level = {row["level"]: row for row in cached_by_method[method]}
        summaries = [by_level[level] for level in args.levels if level in by_level]
        _rewrite_summary_csv(out_dir, summaries)
        report = {
            "method": method,
            "checkpoint": None,
            "levels": list(args.levels),
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "summaries": summaries,
            "env_config": asdict(QuadrotorPursuitConfig(
                **json.loads(
                    (Path(__file__).parent / "configs/pursuit_v2" / f"radar5_dual_{args.levels[0]}.json")
                    .read_text(encoding="utf-8")
                )
            )),
        }
        (out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"method": method, "summaries": summaries}, indent=2), flush=True)


if __name__ == "__main__":
    main()
