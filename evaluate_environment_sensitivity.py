"""Run parameter sensitivity experiments on the weak-communication environment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evaluate_ablation_cases import run_case
from cooperative_search_env import WeakCommConfig


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "sensitivity"


SWEEPS = {
    "target_speed": [0.0, 0.2, 0.42, 0.65],
    "obstacle_speed": [0.0, 0.15, 0.28, 0.45],
    "comm_radius": [4.0, 7.0, 10.0, 14.0],
    "n_obstacles": [8, 16, 24, 35],
}


METRICS = [
    "coverage",
    "target_discovery_rate",
    "target_tracking_rate",
    "target_completion_rate",
    "collisions",
    "lost_tracks",
    "avg_graph_density",
    "total_reward",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def aggregate(parameter: str, value, seeds: list[int], base: WeakCommConfig) -> dict:
    summaries = []
    for seed in seeds:
        cfg = replace(base, seed=seed)
        setattr(cfg, parameter, value)
        result = run_case(f"{parameter}_{value}_seed_{seed}", cfg)
        summaries.append(result["summary"])
    row = {"parameter": parameter, "value": value, "seeds": ",".join(map(str, seeds))}
    for metric in METRICS:
        values = np.asarray([summary[metric] for summary in summaries], dtype=float)
        row[f"{metric}_mean"] = float(values.mean())
        row[f"{metric}_std"] = float(values.std(ddof=0))
    return row


def sensitivity_svg(path: Path, rows: list[dict], metric: str) -> None:
    width, height = 980, 520
    left, top, right, bottom = 85, 60, 240, 75
    colors = {
        "target_speed": "#1f77b4",
        "obstacle_speed": "#d62728",
        "comm_radius": "#2ca02c",
        "n_obstacles": "#9467bd",
    }
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["parameter"], []).append(row)
    max_x = max(float(row["value"]) for row in rows)
    min_x = min(float(row["value"]) for row in rows)
    vals = [row[f"{metric}_mean"] + row[f"{metric}_std"] for row in rows]
    min_y = min([0.0] + vals)
    max_y = max(vals + [1.0])
    if abs(max_x - min_x) < 1e-9:
        max_x = min_x + 1.0
    if abs(max_y - min_y) < 1e-9:
        max_y = min_y + 1.0
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{left}" y="26" font-size="22" font-family="Times New Roman">Sensitivity: {metric}</text>')
    body.append('<text x="85" y="46" font-size="16" font-family="Times New Roman" fill="#555">Solid line: mean across seeds; marker label: swept parameter.</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = top + tick * (height - top - bottom) / 5
        v = max_y - tick * (max_y - min_y) / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="20" y="{y+4:.1f}" font-size="15" font-family="Times New Roman">{v:.2f}</text>')
    for idx, (parameter, items) in enumerate(grouped.items()):
        items = sorted(items, key=lambda item: float(item["value"]))
        pts = []
        for row in items:
            x = left + (float(row["value"]) - min_x) / (max_x - min_x) * (width - left - right)
            y = height - bottom - (row[f"{metric}_mean"] - min_y) / (max_y - min_y) * (height - top - bottom)
            pts.append(f"{x:.1f},{y:.1f}")
            body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[parameter]}"/>')
        body.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{colors[parameter]}" stroke-width="2.5"/>')
        lx = width - right + 25
        ly = top + 24 + idx * 28
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{colors[parameter]}" stroke-width="3"/>')
        body.append(f'<text x="{lx+38}" y="{ly+4}" font-size="16" font-family="Times New Roman">{parameter}</text>')
    body.append(f'<text x="{width-right+25}" y="{top}" font-size="17" font-family="Times New Roman" font-weight="bold">Legend</text>')
    body.append(f'<text x="{left + (width-left-right)/2 - 55:.1f}" y="{height-18}" font-size="16" font-family="Times New Roman">Parameter value</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="11,23,41")
    parser.add_argument("--quick", action="store_true", help="Use two values and one seed per parameter for a fast smoke test.")
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    if args.quick:
        seeds = seeds[:1]
    base = WeakCommConfig()
    if args.search_steps is not None:
        base.search_steps = args.search_steps
    if args.n_uavs is not None:
        base.n_uavs = args.n_uavs
    if args.n_targets is not None:
        base.n_targets = args.n_targets
    if args.completion_steps is not None:
        base.target_completion_steps = args.completion_steps
    if args.tpm_prior_base is not None:
        base.tpm_prior_base = args.tpm_prior_base
    if args.enable_target_prior:
        base.tpm_target_prior_enabled = True
    if args.tpm_target_prior_strength is not None:
        base.tpm_target_prior_strength = args.tpm_target_prior_strength
    if args.tpm_target_prior_sigma is not None:
        base.tpm_target_prior_sigma = args.tpm_target_prior_sigma
    rows = []
    for parameter, values in SWEEPS.items():
        selected = values[:2] if args.quick else values
        for value in selected:
            rows.append(aggregate(parameter, value, seeds, base))
    fields = ["parameter", "value", "seeds"]
    for metric in METRICS:
        fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(OUT / "sensitivity_summary.csv", rows, fields)
    (OUT / "sensitivity_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    for metric in METRICS:
        sensitivity_svg(OUT / f"{metric}_sensitivity.svg", rows, metric)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
