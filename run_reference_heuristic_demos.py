"""Run paper-inspired heuristic core-mechanism demos.

These demos validate the Wang/Zhao/Chang-inspired environment mechanics with
hand-coded policies. They are intentionally separate from MARL training scripts
(`train_cooperative_marl.py`, `train_multi_seed_cooperative_marl.py`) so a successful demo run is not confused
with a full learning-based paper reproduction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tpm_search_base_env import run_wang_heuristic_demo
from cooperative_search_env import run_weak_comm_demo
from graph_tracking_reference_env import run_zhao_heuristic_demo


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def polyline(points: list[list[float]], scale: float, color: str) -> str:
    pts = " ".join(f"{x * scale:.1f},{y * scale:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>'


def write_wang_svg(path: Path, result: dict) -> None:
    scale = 18
    size = 25 * scale
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    trajectories = [[] for _ in result["history"][0]["positions"]]
    for row in result["history"]:
        for i, pos in enumerate(row["positions"]):
            trajectories[i].append(pos)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    for i in range(26):
        x = i * scale
        body.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{size}" stroke="#ddd" stroke-width="1"/>')
        body.append(f'<line x1="0" y1="{x}" x2="{size}" y2="{x}" stroke="#ddd" stroke-width="1"/>')
    target_traj = []
    if result["history"] and "targets" in result["history"][0]:
        for target_idx in range(len(result["history"][0]["targets"])):
            target_traj.append([row["targets"][target_idx] for row in result["history"]])
    for traj in target_traj:
        body.append(polyline(traj, scale, "#111"))
    for x, y in result.get("final_dynamic_targets", result["targets"]):
        body.append(f'<circle cx="{(x + 0.5) * scale:.1f}" cy="{(y + 0.5) * scale:.1f}" r="4" fill="#111"/>')
    for i, traj in enumerate(trajectories):
        centered = [[p[0] + 0.5, p[1] + 0.5] for p in traj]
        body.append(polyline(centered, scale, colors[i % len(colors)]))
        sx, sy = centered[0]
        ex, ey = centered[-1]
        body.append(f'<circle cx="{sx * scale:.1f}" cy="{sy * scale:.1f}" r="5" fill="{colors[i % len(colors)]}"/>')
        body.append(f'<rect x="{ex * scale - 4:.1f}" y="{ey * scale - 4:.1f}" width="8" height="8" fill="{colors[i % len(colors)]}"/>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_weak_comm_svg(path: Path, result: dict) -> None:
    scale = 18
    size = 25 * scale
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    trajectories = [[] for _ in result["history"][0]["positions"]]
    for row in result["history"]:
        for i, pos in enumerate(row["positions"]):
            trajectories[i].append(pos)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    for i in range(26):
        x = i * scale
        body.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{size}" stroke="#ddd" stroke-width="1"/>')
        body.append(f'<line x1="0" y1="{x}" x2="{size}" y2="{x}" stroke="#ddd" stroke-width="1"/>')
    for x, y in result["obstacles"]:
        body.append(f'<circle cx="{(x + 0.5) * scale:.1f}" cy="{(y + 0.5) * scale:.1f}" r="10" fill="#777" opacity="0.35"/>')
    for x, y in result["targets"]:
        body.append(f'<circle cx="{(x + 0.5) * scale:.1f}" cy="{(y + 0.5) * scale:.1f}" r="4" fill="#111"/>')
    for i, traj in enumerate(trajectories):
        centered = [[p[0] + 0.5, p[1] + 0.5] for p in traj]
        body.append(polyline(centered, scale, colors[i % len(colors)]))
        sx, sy = centered[0]
        ex, ey = centered[-1]
        body.append(f'<circle cx="{sx * scale:.1f}" cy="{sy * scale:.1f}" r="5" fill="{colors[i % len(colors)]}"/>')
        body.append(f'<rect x="{ex * scale - 4:.1f}" y="{ey * scale - 4:.1f}" width="8" height="8" fill="{colors[i % len(colors)]}"/>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_zhao_svg(path: Path, result: dict, world_size: float) -> None:
    scale = 640 / world_size
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]
    n_uavs = len(result["history"][0]["uav_pos"])
    trajectories = [[] for _ in range(n_uavs)]
    target_traj = []
    for row in result["history"]:
        for i, pos in enumerate(row["uav_pos"]):
            trajectories[i].append(pos)
        target_traj.append(row["target_pos"][0])
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8" stroke="#333"/>')
    for x, y in result["obstacles"]:
        body.append(f'<circle cx="{x * scale:.1f}" cy="{y * scale:.1f}" r="{20 * scale:.1f}" fill="#888" opacity="0.35"/>')
    body.append(polyline(target_traj, scale, "#111"))
    for x, y in target_traj[-1:]:
        body.append(f'<path d="M {x*scale-6:.1f} {y*scale:.1f} L {x*scale:.1f} {y*scale-6:.1f} L {x*scale+6:.1f} {y*scale:.1f} L {x*scale:.1f} {y*scale+6:.1f} Z" fill="#111"/>')
    for i, traj in enumerate(trajectories):
        body.append(polyline(traj, scale, colors[i % len(colors)]))
        sx, sy = traj[0]
        ex, ey = traj[-1]
        body.append(f'<circle cx="{sx * scale:.1f}" cy="{sy * scale:.1f}" r="5" fill="{colors[i % len(colors)]}"/>')
        body.append(f'<rect x="{ex * scale - 4:.1f}" y="{ey * scale - 4:.1f}" width="8" height="8" fill="{colors[i % len(colors)]}"/>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    wang = run_wang_heuristic_demo()
    zhao_basic = run_zhao_heuristic_demo()
    zhao_large = run_zhao_heuristic_demo(seed=13, n_uavs=5, n_targets=2)
    weak_comm = run_weak_comm_demo()

    summary = {
        "wang2025_heuristic": wang,
        "zhao2025_basic_heuristic": zhao_basic,
        "zhao2025_large_heuristic": zhao_large,
        "integrated_weak_comm_heuristic": weak_comm,
    }
    (OUT / "paper_heuristic_demos_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        OUT / "wang2025_heuristic_coverage.csv",
        wang["history"],
        ["step", "coverage", "reward", "positions", "actions"],
    )
    write_csv(
        OUT / "zhao2025_basic_heuristic.csv",
        zhao_basic["history"],
        ["step", "reward", "min_target_distance", "success", "uav_pos", "target_pos"],
    )
    write_csv(
        OUT / "zhao2025_large_heuristic.csv",
        zhao_large["history"],
        ["step", "reward", "min_target_distance", "success", "uav_pos", "target_pos"],
    )
    write_csv(
        OUT / "integrated_weak_comm_heuristic_coverage.csv",
        weak_comm["history"],
        [
            "step",
            "coverage",
            "reward",
            "target_discovery_rate",
            "target_tracking_rate",
            "collisions",
            "graph_density",
            "positions",
            "targets",
            "obstacles",
            "actions",
        ],
    )
    write_wang_svg(OUT / "wang2025_heuristic_trajectories.svg", wang)
    write_weak_comm_svg(OUT / "integrated_weak_comm_heuristic_trajectories.svg", weak_comm)
    write_zhao_svg(OUT / "zhao2025_basic_heuristic_trajectories.svg", zhao_basic, 400.0)
    write_zhao_svg(OUT / "zhao2025_large_heuristic_trajectories.svg", zhao_large, 1000.0)

    compact = {
        "wang2025_heuristic": {"steps": wang["steps"], "coverage": wang["coverage"]},
        "zhao2025_basic_heuristic": {
            "steps": zhao_basic["steps"],
            "success": zhao_basic["success"],
            "final_min_target_distance": zhao_basic["final_min_target_distance"],
        },
        "zhao2025_large_heuristic": {
            "steps": zhao_large["steps"],
            "success": zhao_large["success"],
            "final_min_target_distance": zhao_large["final_min_target_distance"],
        },
        "integrated_weak_comm_heuristic": {
            "steps": weak_comm["steps"],
            "coverage": weak_comm["coverage"],
            "target_discovery_rate": weak_comm["history"][-1]["target_discovery_rate"],
            "target_tracking_rate": weak_comm["history"][-1]["target_tracking_rate"],
            "collisions": sum(row["collisions"] for row in weak_comm["history"]),
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
