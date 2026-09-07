"""Benchmark Chang-inspired DWA/ORCA obstacle-avoidance variants.

This script isolates the obstacle-avoidance layer from the broader Wang/Zhao
search story. It compares the same heuristic task policy under three safety
filters:

- no_avoidance: action mask only handles map boundaries and turn constraints.
- dwa_only: DWA-style obstacle filtering, without inter-UAV ORCA-like spacing.
- dwa_orca: DWA-style obstacle filtering plus inter-UAV ORCA-like spacing.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig, nominal_search_policy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "chang_dwa_orca_benchmark"

CASES = [
    ("no_avoidance", "No avoidance", "none"),
    ("dwa_only", "DWA only", "dwa"),
    ("dwa_orca", "DWA + ORCA-like", "dwa_orca"),
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def make_config(args: argparse.Namespace, seed: int, avoidance_mode: str) -> WeakCommConfig:
    cfg = WeakCommConfig(seed=seed, avoidance_mode=avoidance_mode)
    updates = {
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_obstacles": args.n_obstacles,
        "grid_size": args.grid_size,
        "search_steps": args.search_steps,
        "uav_speed": args.uav_speed,
        "decision_dt": args.decision_dt,
        "target_speed": args.target_speed,
        "obstacle_speed": args.obstacle_speed,
        "safe_radius": args.safe_radius,
        "obstacle_radius": args.obstacle_radius,
        "tpm_prior_base": args.tpm_prior_base,
        "tpm_target_prior_enabled": True if args.enable_target_prior else None,
        "tpm_target_prior_strength": args.tpm_target_prior_strength,
        "tpm_target_prior_sigma": args.tpm_target_prior_sigma,
        "max_turn_steps": args.max_turn_steps,
        "allow_hover": True if args.allow_hover else None,
        "normalize_diagonal_speed": False if args.no_normalize_diagonal_speed else None,
        "avoidance_prediction_horizon": args.avoidance_prediction_horizon,
        "obstacle_prediction_buffer": args.obstacle_prediction_buffer,
        "pair_prediction_buffer": args.pair_prediction_buffer,
    }
    for key, value in updates.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def path_lengths(history: list[dict]) -> tuple[float, float]:
    n_agents = len(history[0]["positions"])
    per_agent = []
    for agent_idx in range(n_agents):
        pts = np.asarray([row["positions"][agent_idx] for row in history], dtype=float)
        per_agent.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    return float(np.mean(per_agent)), float(np.sum(per_agent))


def clearance_stats(history: list[dict]) -> tuple[float, float]:
    obstacle_clearances = []
    pair_clearances = []
    for row in history:
        positions = np.asarray(row["positions"], dtype=float)
        obstacles = np.asarray(row["obstacles"], dtype=float)
        if len(obstacles):
            d_po = np.linalg.norm(positions[:, None, :] - obstacles[None, :, :], axis=2)
            obstacle_clearances.append(float(np.min(d_po)))
        for i in range(len(positions)):
            for j in range(i):
                pair_clearances.append(float(np.linalg.norm(positions[i] - positions[j])))
    min_obstacle = min(obstacle_clearances) if obstacle_clearances else float("inf")
    min_pair = min(pair_clearances) if pair_clearances else float("inf")
    return min_obstacle, min_pair


def run_case(name: str, label: str, cfg: WeakCommConfig, seed: int) -> dict:
    env = WeakCommBeliefGraphEnv(cfg)
    history = []
    communication_history = []
    for _ in range(env.cfg.search_steps):
        actions = nominal_search_policy(env)
        result = env.step_joint(actions)
        step = env.t
        history.append(
            {
                "seed": seed,
                "case": name,
                "step": step,
                "coverage": result["coverage"],
                "reward": result["reward"],
                "collisions": result["collisions"],
                "uav_collisions": result.get("uav_collisions", 0),
                "obstacle_conflicts": result.get("obstacle_conflicts", 0),
                "new_crashes": result.get("new_crashes", 0),
                "active_uav_exposure": result.get("active_uav_exposure", cfg.n_uavs),
                "step_path_length": result.get("step_path_length", 0.0),
                "target_discovery_rate": result["target_discovery_rate"],
                "target_tracking_rate": result["target_tracking_rate"],
                "target_completion_rate": result["target_completion_rate"],
                "lost_tracks": result["lost_tracks"],
                "spectrum_reward": result["spectrum_reward"],
                "spectrum_success_rate": result["spectrum_success_rate"],
                "spectrum_packet_loss_rate": result["spectrum_packet_loss_rate"],
                "spectrum_latency_ms": result["spectrum_latency_ms"],
                "spectrum_throughput_kbps": result["spectrum_throughput_kbps"],
                "positions": env.positions.copy().tolist(),
                "targets": env.dynamic_targets.copy().tolist(),
                "obstacles": env.obstacles.copy().tolist(),
                "actions": actions.tolist(),
            }
        )
        for slot_rows in result["spectrum_slot_logs"]:
            for row in slot_rows:
                communication_history.append(
                    {
                        "seed": seed,
                        "case": name,
                        "step": step,
                        **row,
                    }
                )
        if result["done"]:
            break
    avg_path, total_path = path_lengths(history)
    min_obstacle_clearance, min_pair_clearance = clearance_stats(history)
    total_collisions = int(sum(row["collisions"] for row in history))
    active_exposure = int(sum(row["active_uav_exposure"] for row in history))
    exposed_distance = float(sum(row["step_path_length"] for row in history))
    return {
        "name": name,
        "label": label,
        "seed": seed,
        "config": cfg.__dict__,
        "summary": {
            "steps": env.t,
            "coverage": history[-1]["coverage"],
            "target_discovery_rate": history[-1]["target_discovery_rate"],
            "target_tracking_rate": history[-1]["target_tracking_rate"],
            "target_completion_rate": history[-1]["target_completion_rate"],
            "collisions": total_collisions,
            "uav_collisions": int(sum(row["uav_collisions"] for row in history)),
            "obstacle_conflicts": int(sum(row["obstacle_conflicts"] for row in history)),
            "new_crashes": int(sum(row["new_crashes"] for row in history)),
            "collision_rate": float(total_collisions / max(env.t, 1)),
            "collision_per_1000_active_steps": float(1000.0 * total_collisions / max(active_exposure, 1)),
            "collision_per_1000_distance": float(1000.0 * total_collisions / max(exposed_distance, 1e-6)),
            "crash_rate": float(np.mean(env.crashed_uavs)),
            "collision_free_completion": float(history[-1]["target_completion_rate"] if total_collisions == 0 else 0.0),
            "lost_tracks": int(sum(row["lost_tracks"] for row in history)),
            "avg_uav_path_length": avg_path,
            "total_uav_path_length": total_path,
            "min_obstacle_clearance": min_obstacle_clearance,
            "min_pair_clearance": min_pair_clearance,
            "total_reward": float(sum(row["reward"] for row in history)),
            "spectrum_reward": float(np.mean([row["spectrum_reward"] for row in history])),
            "spectrum_success_rate": float(np.mean([row["spectrum_success_rate"] for row in history])),
            "spectrum_packet_loss_rate": float(np.mean([row["spectrum_packet_loss_rate"] for row in history])),
            "spectrum_latency_ms": float(np.mean([row["spectrum_latency_ms"] for row in history])),
            "spectrum_throughput_kbps": float(np.mean([row["spectrum_throughput_kbps"] for row in history])),
        },
        "history": history,
        "communication_history": communication_history,
    }


def aggregate(rows: list[dict]) -> list[dict]:
    metrics = [
        "steps",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "new_crashes",
        "collision_rate",
        "collision_per_1000_active_steps",
        "collision_per_1000_distance",
        "crash_rate",
        "collision_free_completion",
        "lost_tracks",
        "avg_uav_path_length",
        "total_uav_path_length",
        "min_obstacle_clearance",
        "min_pair_clearance",
        "total_reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "spectrum_packet_loss_rate",
        "spectrum_latency_ms",
        "spectrum_throughput_kbps",
    ]
    out = []
    for name, label, _mode in CASES:
        case_rows = [row for row in rows if row["case"] == name]
        if not case_rows:
            continue
        item = {"case": name, "label": label, "n_seeds": len(case_rows)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in case_rows], dtype=float)
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=0))
        out.append(item)
    return out


def bar_svg(path: Path, rows: list[dict]) -> None:
    width, height = 980, 520
    left, right, top, bottom = 80, 70, 70, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    metrics = [
        ("collision_per_1000_active_steps_mean", "Collisions / 1k active steps", "#d62728", "rate"),
        ("min_obstacle_clearance_mean", "Obstacle clearance", "#1f77b4", "distance"),
        ("min_pair_clearance_mean", "Pair clearance", "#2ca02c", "distance"),
        ("target_completion_rate_mean", "Completion", "#111111", "rate"),
    ]
    max_value = max(float(row[key]) for row in rows for key, *_ in metrics)
    max_value = max(max_value, 1.0)
    group_w = plot_w / len(rows)
    bar_w = min(32, group_w / 6)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{left}" y="32" font-size="28" font-family="Times New Roman" font-weight="bold">Chang DWA-ORCA Benchmark</text>')
    body.append(f'<text x="{left}" y="55" font-size="17" font-family="Times New Roman" fill="#555">Collision risk is exposure-normalized; higher clearances and completion are better.</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = height - bottom - tick * plot_h / 5
        value = tick * max_value / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="{left-14}" y="{y+4:.1f}" font-size="16" font-family="Times New Roman" text-anchor="end">{value:.1f}</text>')
    for i, row in enumerate(rows):
        center = left + group_w * (i + 0.5)
        start = center - (len(metrics) * bar_w + (len(metrics) - 1) * 8) / 2
        for j, (key, _label, color, _unit) in enumerate(metrics):
            value = float(row[key])
            h = value / max_value * plot_h
            x = start + j * (bar_w + 8)
            y = height - bottom - h
            body.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            body.append(f'<text x="{x + bar_w/2:.1f}" y="{y-6:.1f}" font-size="15" font-family="Times New Roman" text-anchor="middle">{value:.2f}</text>')
        label = row["label"].replace(" + ", "+")
        body.append(f'<text x="{center:.1f}" y="{height-bottom+28}" font-size="17" font-family="Times New Roman" text-anchor="middle">{label}</text>')
    lx = left
    ly = height - 28
    for idx, (_key, label, color, unit) in enumerate(metrics):
        x = lx + idx * 220
        body.append(f'<rect x="{x}" y="{ly-13}" width="15" height="15" fill="{color}"/>')
        body.append(f'<text x="{x+22}" y="{ly}" font-size="16" font-family="Times New Roman">{label} ({unit})</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,23,41,59,83")
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--decision-dt", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--obstacle-speed", type=float, default=None)
    parser.add_argument("--safe-radius", type=float, default=None)
    parser.add_argument("--obstacle-radius", type=float, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    parser.add_argument("--max-turn-steps", type=int, default=None)
    parser.add_argument("--allow-hover", action="store_true")
    parser.add_argument("--no-normalize-diagonal-speed", action="store_true")
    parser.add_argument("--avoidance-prediction-horizon", type=int, default=None)
    parser.add_argument("--obstacle-prediction-buffer", type=float, default=None)
    parser.add_argument("--pair-prediction-buffer", type=float, default=None)
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    results = []
    flat_rows = []
    communication_rows = []
    for seed in seeds:
        for name, label, mode in CASES:
            result = run_case(name, label, make_config(args, seed, mode), seed)
            results.append(result)
            flat_rows.append({"seed": seed, "case": name, **result["summary"]})
            communication_rows.extend(result["communication_history"])

    summary = aggregate(flat_rows)
    (out_dir / "chang_dwa_orca_benchmark.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "chang_dwa_orca_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        out_dir / "chang_dwa_orca_by_seed.csv",
        flat_rows,
        [
            "seed",
            "case",
            "steps",
            "coverage",
            "target_discovery_rate",
            "target_tracking_rate",
            "target_completion_rate",
            "collisions",
            "uav_collisions",
            "obstacle_conflicts",
            "new_crashes",
            "collision_rate",
            "collision_per_1000_active_steps",
            "collision_per_1000_distance",
            "crash_rate",
            "collision_free_completion",
            "lost_tracks",
            "avg_uav_path_length",
            "total_uav_path_length",
            "min_obstacle_clearance",
            "min_pair_clearance",
            "total_reward",
            "spectrum_reward",
            "spectrum_success_rate",
            "spectrum_packet_loss_rate",
            "spectrum_latency_ms",
            "spectrum_throughput_kbps",
        ],
    )
    write_csv(
        out_dir / "communication_step_log.csv",
        communication_rows,
        [
            "seed",
            "case",
            "step",
            "slot",
            "uav_id",
            "success",
            "packet_loss",
            "latency_ms",
            "throughput_kbps",
        ],
    )
    fields = ["case", "label", "n_seeds"]
    for metric in [
        "steps",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "new_crashes",
        "collision_rate",
        "collision_per_1000_active_steps",
        "collision_per_1000_distance",
        "crash_rate",
        "collision_free_completion",
        "lost_tracks",
        "avg_uav_path_length",
        "total_uav_path_length",
        "min_obstacle_clearance",
        "min_pair_clearance",
        "total_reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "spectrum_packet_loss_rate",
        "spectrum_latency_ms",
        "spectrum_throughput_kbps",
    ]:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(out_dir / "chang_dwa_orca_summary.csv", summary, fields)
    bar_svg(out_dir / "chang_dwa_orca_summary.svg", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
