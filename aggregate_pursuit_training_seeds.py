"""Aggregate held-out evaluations from independently trained pursuit policies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from artifact_paths import artifact_path, default_output


def aggregate_payloads(payloads, label, bootstrap_samples=10_000, seed=20260916):
    if len(payloads) < 2:
        raise ValueError("At least two independently trained policy evaluations are required")
    env_config = payloads[0].get("env_config")
    if any(payload.get("env_config") != env_config for payload in payloads[1:]):
        raise ValueError("All evaluations must use the identical environment configuration")
    seed_orders = []
    capture_rows = []
    summaries = []
    merged_rows = []
    for run, payload in enumerate(payloads):
        if len(payload.get("summary", {})) != 1:
            raise ValueError("Each input must contain exactly one learned-policy summary")
        summaries.append(next(iter(payload["summary"].values())))
        ordered = sorted(payload.get("rows", []), key=lambda row: int(row["seed"]))
        seeds = [int(row["seed"]) for row in ordered]
        if len(seeds) != len(set(seeds)):
            raise ValueError("An input contains duplicate evaluation seeds")
        seed_orders.append(seeds)
        capture_rows.append([float(row["captured"]) for row in ordered])
        merged_rows.extend({**row, "training_run": run} for row in ordered)
    if any(seeds != seed_orders[0] for seeds in seed_orders[1:]):
        raise ValueError("Training runs must be evaluated on the same ordered scenario seeds")

    capture = np.asarray(capture_rows, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.empty(int(bootstrap_samples), dtype=float)
    for index in range(len(boot)):
        model_indices = rng.integers(0, capture.shape[0], size=capture.shape[0])
        scenario_indices = rng.integers(0, capture.shape[1], size=capture.shape[1])
        boot[index] = capture[np.ix_(model_indices, scenario_indices)].mean()

    metrics = {
        "episodes": capture.shape[1],
        "training_runs": capture.shape[0],
        "capture_rate": float(capture.mean()),
        "capture_rate_training_std": float(capture.mean(axis=1).std(ddof=1)),
        "capture_ci_low": float(np.percentile(boot, 2.5)),
        "capture_ci_high": float(np.percentile(boot, 97.5)),
        "uncertainty_method": "hierarchical bootstrap over training runs and held-out scenarios",
    }
    numeric_keys = sorted(set.intersection(*(
        {key for key, value in summary.items() if isinstance(value, (int, float)) and value is not None}
        for summary in summaries
    )))
    for key in numeric_keys:
        if key in ("episodes", "capture_rate"):
            continue
        values = np.asarray([float(summary[key]) for summary in summaries], dtype=float)
        metrics[key] = float(values.mean())
        metrics[f"{key}_training_std"] = float(values.std(ddof=1))
    return {
        "env_config": env_config,
        "rows": merged_rows,
        "summary": {label: metrics},
        "evaluation_seeds": seed_orders[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=artifact_path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument(
        "--out", type=artifact_path,
        default=default_output("pursuit_multiseed_evaluation.json"),
    )
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    try:
        payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.input]
        result = aggregate_payloads(payloads, args.label, args.bootstrap_samples, args.seed)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
