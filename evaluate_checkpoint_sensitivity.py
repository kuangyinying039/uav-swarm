"""Evaluate fixed MAPPO/QMIX checkpoints and a heuristic under parameter sweeps.

No policy is retrained.  For every parameter-value-seed tuple, all methods run
from the same environment configuration and random seed.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import torch

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig, nominal_search_policy
from evaluate_discovery_speed import filtered_config, load_policy
from marl_trainers import seed_everything


SWEEPS = {
    "target_speed": [0.00, 0.25, 0.50, 0.75, 1.00],
    "obstacle_speed": [0.00, 0.25, 0.50, 0.75],
    "comm_radius": [4.0, 7.0, 10.0, 14.0],
    "sensor_radius": [1.0, 1.5, 2.0, 2.5, 3.0],
    "n_obstacles": [0, 5, 10, 20, 30],
    "outage_base_prob": [0.00, 0.02, 0.05, 0.10],
    "outage_jammed_prob": [0.10, 0.30, 0.45, 0.60],
    "outage_recovery_prob": [0.20, 0.35, 0.45, 0.60, 0.80],
}

METRICS = [
    "coverage",
    "discovery_rate",
    "tracking_rate",
    "completion_rate",
    "belief_decisiveness",
    "belief_brier_score",
    "discovery_auc",
    "completion_auc",
    "step_to_80pct_discovery",
    "step_to_80pct_completion",
    "censored_mean_discovery_step",
    "mfdt",
    "censored_mean_completion_step",
    "collisions",
    "collision_rate",
    "safety_intervention_rate",
    "lost_tracks",
    "total_reward",
]

COLORS = {"heuristic": "#555555", "mappo": "#1f77b4", "qmix": "#d62728"}


def threshold_step(curve: list[float], threshold: float, horizon: int) -> int:
    for step, value in enumerate(curve):
        if value >= threshold:
            return step
    return horizon + 1


def run_episode(method: str, seed: int, cfg_values: dict, action_fn) -> dict:
    values = dict(cfg_values)
    values["seed"] = seed
    cfg = filtered_config(WeakCommConfig, values)
    seed_everything(seed)
    env = WeakCommBeliefGraphEnv(cfg)
    obs = env.observe_search()
    discovery_steps = np.full(cfg.n_targets, -1, dtype=int)
    completion_steps = np.full(cfg.n_targets, -1, dtype=int)
    discovery_curve = [float(np.mean(env.found_targets))]
    completion_curve = [float(np.mean(env.completed_targets))]
    total_reward = 0.0
    collisions = 0
    safety_intervention_rates = []
    lost_tracks = 0
    for _ in range(cfg.search_steps):
        prev_found = env.found_targets.copy()
        prev_completed = env.completed_targets.copy()
        actions = nominal_search_policy(env) if action_fn is None else action_fn(env, obs)
        result = env.step_joint(np.asarray(actions, dtype=int))
        discovery_steps[env.found_targets & ~prev_found] = env.t
        completion_steps[env.completed_targets & ~prev_completed] = env.t
        total_reward += float(result["reward"])
        collisions += int(result.get("collisions", 0))
        safety_intervention_rates.append(float(result.get("safety_intervention_rate", 0.0)))
        lost_tracks += int(result.get("lost_tracks", 0))
        discovery_curve.append(float(np.mean(env.found_targets)))
        completion_curve.append(float(np.mean(env.completed_targets)))
        obs = result["obs"]
        if result["done"]:
            break
    discovery_curve.extend([discovery_curve[-1]] * (cfg.search_steps + 1 - len(discovery_curve)))
    completion_curve.extend([completion_curve[-1]] * (cfg.search_steps + 1 - len(completion_curve)))
    discovery_censored = np.where(discovery_steps >= 0, discovery_steps, cfg.search_steps + 1)
    completion_censored = np.where(completion_steps >= 0, completion_steps, cfg.search_steps + 1)
    return {
        "method": method,
        "seed": seed,
        "coverage": float(env.coverage()),
        "discovery_rate": float(np.mean(env.found_targets)),
        "tracking_rate": float(np.mean(env.tracked_targets | env.completed_targets)),
        "completion_rate": float(np.mean(env.completed_targets)),
        "belief_decisiveness": float(env.belief_decisiveness()),
        "belief_brier_score": float(env.belief_brier_score()),
        "discovery_auc": float(np.mean(discovery_curve)),
        "completion_auc": float(np.mean(completion_curve)),
        "step_to_80pct_discovery": threshold_step(discovery_curve, 0.80, cfg.search_steps),
        "step_to_80pct_completion": threshold_step(completion_curve, 0.80, cfg.search_steps),
        "censored_mean_discovery_step": float(np.mean(discovery_censored)),
        # Mean first-detection time over all targets. Targets that remain
        # undiscovered are right-censored at horizon + 1 so a low discovery
        # rate cannot look artificially fast by averaging successes only.
        "mfdt": float(np.mean(discovery_censored)),
        "censored_mean_completion_step": float(np.mean(completion_censored)),
        "collisions": collisions,
        "collision_rate": float(collisions / max(env.t, 1)),
        "safety_intervention_rate": float(np.mean(safety_intervention_rates))
        if safety_intervention_rates
        else 0.0,
        "lost_tracks": lost_tracks,
        "total_reward": total_reward,
        "episode_steps": int(env.t),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault((row["parameter"], row["value"], row["method"]), []).append(row)
    output = []
    for (parameter, value, method), subset in groups.items():
        result = {"parameter": parameter, "value": value, "method": method, "n_seeds": len(subset)}
        for metric in METRICS:
            values = np.asarray([float(row[metric]) for row in subset], dtype=float)
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            result[f"{metric}_mean"] = float(values.mean())
            result[f"{metric}_std"] = std
            result[f"{metric}_ci95"] = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
        output.append(result)
    return sorted(output, key=lambda row: (row["parameter"], float(row["value"]), row["method"]))


def paired_differences(rows: list[dict]) -> list[dict]:
    lookup = {(r["parameter"], r["value"], r["seed"], r["method"]): r for r in rows}
    output = []
    methods = sorted({r["method"] for r in rows if r["method"] != "heuristic"})
    for parameter, value, seed in sorted({(r["parameter"], r["value"], r["seed"]) for r in rows}):
        baseline = lookup[(parameter, value, seed, "heuristic")]
        for method in methods:
            learned = lookup[(parameter, value, seed, method)]
            row = {"parameter": parameter, "value": value, "method": method, "seed": seed}
            for metric in METRICS:
                row[f"{metric}_delta_vs_heuristic"] = float(learned[metric]) - float(baseline[metric])
            output.append(row)
    return output


def svg_for_parameter(path: Path, parameter: str, rows: list[dict]) -> None:
    panels = [
        ("completion_rate", "Completion rate", (0.0, 1.0)),
        ("tracking_rate", "Tracking rate", (0.0, 1.0)),
        ("discovery_rate", "Discovery rate", (0.0, 1.0)),
        ("collision_rate", "Collision rate", None),
    ]
    width, height = 1320, 860
    pw, ph = 520, 270
    origins = [(90, 105), (730, 105), (90, 515), (730, 515)]
    subset = [row for row in rows if row["parameter"] == parameter]
    methods = [m for m in ("heuristic", "mappo", "qmix") if any(r["method"] == m for r in subset)]
    x_values = sorted({float(row["value"]) for row in subset})
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="#fcfcfa"/>']
    body.append(f'<text x="660" y="35" text-anchor="middle" font-family="Times New Roman" font-size="30" font-weight="bold">Fixed-policy sensitivity: {html.escape(parameter)}</text>')
    for index, method in enumerate(methods):
        x = 130 + index * 190
        color = COLORS[method]
        body.append(f'<line x1="{x}" y1="68" x2="{x+30}" y2="68" stroke="{color}" stroke-width="4"/><text x="{x+40}" y="73" font-family="Times New Roman" font-size="19">{method}</text>')
    for panel_index, (metric, label, fixed) in enumerate(panels):
        x0, y0 = origins[panel_index]
        metric_rows = [r for r in subset if math.isfinite(float(r[f"{metric}_mean"]))]
        if fixed:
            y_min, y_max = fixed
        else:
            vals = [float(r[f"{metric}_mean"]) + float(r[f"{metric}_ci95"]) for r in metric_rows]
            y_min, y_max = 0.0, max(vals + [0.01]) * 1.10
        body.append(f'<text x="{x0+pw/2}" y="{y0-18}" text-anchor="middle" font-family="Times New Roman" font-size="22" font-weight="bold">{label}</text>')
        for tick in range(6):
            y = y0 + tick * ph / 5
            value = y_max - tick * (y_max - y_min) / 5
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+pw}" y2="{y:.1f}" stroke="#e4e4df"/><text x="{x0-10}" y="{y+5:.1f}" text-anchor="end" font-family="Times New Roman" font-size="16">{value:.2f}</text>')
        for xi, value in enumerate(x_values):
            x = x0 + (xi / max(len(x_values) - 1, 1)) * pw
            body.append(f'<text x="{x:.1f}" y="{y0+ph+23}" text-anchor="middle" font-family="Times New Roman" font-size="16">{value:g}</text>')
        for method in methods:
            points = []
            for xi, value in enumerate(x_values):
                row = next(r for r in subset if r["method"] == method and float(r["value"]) == value)
                mean, ci = float(row[f"{metric}_mean"]), float(row[f"{metric}_ci95"])
                x = x0 + (xi / max(len(x_values) - 1, 1)) * pw
                y = y0 + (y_max - mean) / max(y_max - y_min, 1e-9) * ph
                y_hi = y0 + (y_max - min(y_max, mean + ci)) / max(y_max - y_min, 1e-9) * ph
                y_lo = y0 + (y_max - max(y_min, mean - ci)) / max(y_max - y_min, 1e-9) * ph
                points.append(f"{x:.1f},{y:.1f}")
                body.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="{COLORS[method]}"/><circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{COLORS[method]}"/>')
            body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[method]}" stroke-width="2.5"/>')
        body.append(f'<text x="{x0+pw/2}" y="{y0+ph+43}" text-anchor="middle" font-family="Times New Roman" font-size="17">{html.escape(parameter)}</text>')
    body.append('</svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mappo", default=None, help="Fixed MAPPO .pt checkpoint")
    parser.add_argument("--qmix", default=None, help="Fixed QMIX .pt checkpoint")
    parser.add_argument("--seeds", default="101,113,127,139,151,163,179,191,211,227")
    parser.add_argument("--sweeps", default=",".join(SWEEPS), help="Comma-separated sweep names")
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--out-dir", default="../outputs/checkpoint_sensitivity")
    parser.add_argument("--quick", action="store_true", help="One seed and first two values per sweep")
    args = parser.parse_args()
    checkpoint_paths = [Path(value) for value in (args.mappo, args.qmix) if value]
    if not checkpoint_paths:
        raise ValueError("Provide --mappo and/or --qmix checkpoint.")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoint_paths]
    base_env = dict(payloads[0].get("env_config", {}))
    if args.search_steps is not None:
        base_env["search_steps"] = args.search_steps
    if args.completion_steps is not None:
        base_env["target_completion_steps"] = args.completion_steps
    sweep_names = [name.strip() for name in args.sweeps.split(",") if name.strip()]
    unknown = sorted(set(sweep_names) - set(SWEEPS))
    if unknown:
        raise ValueError(f"Unknown sweeps: {', '.join(unknown)}. Choices: {', '.join(SWEEPS)}")
    methods = [("heuristic", None)]
    for path in checkpoint_paths:
        algorithm, payload, action_fn = load_policy(path, base_env)
        if any(name == algorithm for name, _ in methods):
            raise ValueError(f"Only one checkpoint per algorithm is supported: {algorithm}")
        for key in ("n_uavs", "n_targets", "grid_size"):
            if key in payload.get("env_config", {}) and key in base_env and payload["env_config"][key] != base_env[key]:
                raise ValueError(f"Checkpoint mismatch for {key}: {path}")
        methods.append((algorithm, action_fn))
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if args.quick:
        seeds = seeds[:1]
    raw_rows = []
    for parameter in sweep_names:
        values = SWEEPS[parameter][:2] if args.quick else SWEEPS[parameter]
        for value in values:
            cfg_values = dict(base_env)
            cfg_values[parameter] = value
            for seed in seeds:
                for method, action_fn in methods:
                    row = run_episode(method, seed, cfg_values, action_fn)
                    row.update({"parameter": parameter, "value": value})
                    raw_rows.append(row)
                    print(f"{parameter}={value:g} {method:9s} seed={seed} completion={row['completion_rate']:.3f} discovery={row['discovery_rate']:.3f}", flush=True)
    summary_rows = aggregate(raw_rows)
    paired_rows = paired_differences(raw_rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "checkpoint_sensitivity_raw.csv", raw_rows)
    write_csv(out / "checkpoint_sensitivity_summary.csv", summary_rows)
    write_csv(out / "checkpoint_sensitivity_paired.csv", paired_rows)
    (out / "checkpoint_sensitivity_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    for parameter in sweep_names:
        svg_for_parameter(out / f"sensitivity_{parameter}.svg", parameter, summary_rows)
    manifest = {
        "checkpoints": [str(path) for path in checkpoint_paths],
        "methods": [name for name, _ in methods],
        "seeds": seeds,
        "sweeps": {name: SWEEPS[name][:2] if args.quick else SWEEPS[name] for name in sweep_names},
        "base_environment": base_env,
        "policy_weights_frozen": True,
    }
    (out / "checkpoint_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out.resolve()), "runs": len(raw_rows)}, indent=2))


if __name__ == "__main__":
    main()
