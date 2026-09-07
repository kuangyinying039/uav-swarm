"""Paired evaluation of heuristic and any number of trained policy checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from evaluate_checkpoint_sensitivity import METRICS, run_episode
from evaluate_discovery_speed import load_policy


def parse_policy(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use LABEL=CHECKPOINT.pt")
    label, raw_path = value.split("=", 1)
    if not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Use LABEL=CHECKPOINT.pt")
    return label.strip(), Path(raw_path.strip())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        model_ids = sorted({row["model_id"] for row in subset})
        result = {
            "method": method,
            "n_models": len(model_ids),
            "n_test_scenarios": len({int(row["seed"]) for row in subset}),
            "n_runs": len(subset),
        }
        for metric in METRICS:
            if method == "Heuristic":
                values = np.asarray([float(row[metric]) for row in subset], dtype=float)
                result[f"{metric}_interval_unit"] = "test_scenario"
            else:
                values = np.asarray(
                    [
                        np.mean(
                            [
                                float(row[metric])
                                for row in subset
                                if row["model_id"] == model_id
                            ]
                        )
                        for model_id in model_ids
                    ],
                    dtype=float,
                )
                result[f"{metric}_interval_unit"] = "training_seed_model"
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = (
                float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        output.append(result)
    return output


def paired(rows: list[dict]) -> list[dict]:
    output = []
    methods = sorted({row["method"] for row in rows if row["method"] != "Heuristic"})
    for method in methods:
        for seed in sorted({int(row["seed"]) for row in rows}):
            baseline = next(
                row for row in rows if row["method"] == "Heuristic" and int(row["seed"]) == seed
            )
            learned_rows = [
                row for row in rows if row["method"] == method and int(row["seed"]) == seed
            ]
            result = {"method": method, "seed": seed}
            for metric in METRICS:
                result[f"{metric}_delta_vs_heuristic"] = (
                    float(np.mean([float(row[metric]) for row in learned_rows]))
                    - float(baseline[metric])
                )
            output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", action="append", required=True, type=parse_policy, help="LABEL=CHECKPOINT.pt"
    )
    parser.add_argument("--seeds", default="101,113,127,139,151,163,179,191,211,227")
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="../outputs/policy_suite")
    args = parser.parse_args()

    first_payload = torch.load(args.policy[0][1], map_location="cpu", weights_only=False)
    base_env = dict(first_payload.get("env_config", {}))
    if args.search_steps is not None:
        base_env["search_steps"] = args.search_steps
    if args.completion_steps is not None:
        base_env["target_completion_steps"] = args.completion_steps
    policies = []
    for label, checkpoint in args.policy:
        _, payload, action_fn = load_policy(checkpoint, base_env)
        for key in ("n_uavs", "n_targets", "grid_size"):
            if payload.get("env_config", {}).get(key, base_env.get(key)) != base_env.get(key):
                raise ValueError(f"{label} has incompatible {key}: {checkpoint}")
        policies.append((label, checkpoint, action_fn))

    rows = []
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    for seed in seeds:
        episode_methods = [
            ("Heuristic", "heuristic", "", None),
            *[
                (label, checkpoint.stem, str(checkpoint), action_fn)
                for label, checkpoint, action_fn in policies
            ],
        ]
        for label, model_id, checkpoint, action_fn in episode_methods:
            row = run_episode(label, seed, base_env, action_fn)
            row.update({"model_id": model_id, "checkpoint": checkpoint})
            rows.append(row)
            print(
                f"{label:28s} seed={seed} discovery={row['discovery_rate']:.3f} "
                f"coverage={row['coverage']:.3f} collision_rate={row['collision_rate']:.4f}",
                flush=True,
            )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows)
    write_csv(out / "policy_suite_raw.csv", rows)
    write_csv(out / "policy_suite_summary.csv", summary)
    write_csv(out / "policy_suite_paired.csv", paired(rows))
    (out / "policy_suite_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = {
        "policies": [{"label": label, "checkpoint": str(path)} for label, path, _ in policies],
        "seeds": seeds,
        "environment": base_env,
        "paired_scenarios": True,
        "deterministic_action_selection": True,
    }
    (out / "policy_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
