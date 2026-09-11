"""Split MPC-successful baseline episodes into leakage-free scenario banks."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="JSON output produced by evaluate_3d_baselines.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--shuffle-seed", type=int, default=20260910)
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) < 1:
        parser.error("Train, validation, and test counts must all be positive")

    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        parser.error(f"Cannot read baseline JSON {args.input}: {error}")
    successes = [
        row for row in payload.get("rows", [])
        if row.get("method") == "mpc" and bool(row.get("captured"))
    ]
    seeds = [int(row["seed"]) for row in successes]
    if len(seeds) != len(set(seeds)):
        parser.error("Input contains duplicate MPC-success rows")
    needed = args.train + args.validation + args.test
    if len(successes) < needed:
        parser.error(
            f"Only {len(successes)} MPC-successful scenarios, but {needed} are required; "
            "scan a larger candidate range"
        )

    random.Random(args.shuffle_seed).shuffle(successes)
    groups = {
        "train": successes[:args.train],
        "validation": successes[args.train:args.train + args.validation],
        "test": successes[args.train + args.validation:needed],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, rows in groups.items():
        (args.out / f"{name}_seeds.txt").write_text(
            "".join(f"{int(row['seed'])}\n" for row in rows), encoding="utf-8"
        )
        capture_times = [float(row["capture_time"]) for row in rows]
        summaries[name] = {
            "count": len(rows),
            "layout_counts": dict(Counter(str(row.get("initial_layout")) for row in rows)),
            "mean_mpc_capture_time": sum(capture_times) / len(capture_times),
            "min_seed": min(int(row["seed"]) for row in rows),
            "max_seed": max(int(row["seed"]) for row in rows),
        }

    # Difficulty labels are diagnostic only and never change the primary split.
    test_by_time = sorted(groups["test"], key=lambda row: float(row["capture_time"]))
    difficulty = {}
    boundaries = [round(len(test_by_time) * fraction / 3) for fraction in range(4)]
    for index, name in enumerate(("easy", "medium", "hard")):
        rows = test_by_time[boundaries[index]:boundaries[index + 1]]
        (args.out / f"test_{name}_seeds.txt").write_text(
            "".join(f"{int(row['seed'])}\n" for row in rows), encoding="utf-8"
        )
        difficulty[name] = {
            "count": len(rows),
            "capture_time_min": min((float(row["capture_time"]) for row in rows), default=None),
            "capture_time_max": max((float(row["capture_time"]) for row in rows), default=None),
        }

    manifest = {
        "seed_bank_version": 1,
        "definition": "MPC-solvable means captured by the configured MPC baseline; not theoretically solvable",
        "source": str(args.input),
        "shuffle_seed": args.shuffle_seed,
        "mpc_successes_available": len(successes),
        "splits": summaries,
        "test_difficulty_by_mpc_capture_time": difficulty,
        "env_config": payload.get("env_config"),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
