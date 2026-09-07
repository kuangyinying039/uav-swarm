"""Run comparison and ablation experiments for the integrated environment."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig, nominal_search_policy


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "ablations"

CASE_LABELS = {
    "outage_dwa_orca": "Outage + DWA-ORCA",
    "full_comm_dwa_orca": "Full comm + DWA-ORCA",
    "outage_no_avoidance": "Outage + no avoidance",
    "outage_dwa_only": "Outage + DWA only",
}

METRIC_LABELS = {
    "coverage": "Coverage rate",
    "target_discovery_rate": "Target discovery rate",
    "target_tracking_rate": "Target tracking rate",
    "target_completion_rate": "Target completion rate",
    "graph_density": "Communication graph density",
    "avg_graph_density": "Average communication graph density",
    "outage_rate": "Outage rate",
    "disabled_rate": "Disabled UAV rate",
}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def run_case(name: str, cfg: WeakCommConfig) -> dict:
    env = WeakCommBeliefGraphEnv(cfg)
    history = []
    for _ in range(env.cfg.search_steps):
        actions = nominal_search_policy(env)
        result = env.step_joint(actions)
        history.append(
            {
                "step": env.t,
                "coverage": result["coverage"],
                "reward": result["reward"],
                "target_discovery_rate": result["target_discovery_rate"],
                "target_tracking_rate": result["target_tracking_rate"],
                "target_completion_rate": result["target_completion_rate"],
                "task_phase": result["task_phase"],
                "lost_tracks": result["lost_tracks"],
                "escort_score": result["escort_score"],
                "collisions": result["collisions"],
                "new_crashes": result.get("new_crashes", 0),
                "uav_collisions": result.get("uav_collisions", 0),
                "obstacle_conflicts": result.get("obstacle_conflicts", 0),
                "pair_conflicts": result.get("pair_conflicts", 0),
                "graph_density": result["graph_density"],
                "outage_rate": result.get("outage_rate", 0.0),
                "disabled_rate": result.get("disabled_rate", 0.0),
                "active_uav_count": result.get("active_uav_count", cfg.n_uavs),
                "active_uav_exposure": result.get("active_uav_exposure", cfg.n_uavs),
                "step_path_length": result.get("step_path_length", 0.0),
                "mean_interference": result.get("mean_interference", 0.0),
                "positions": env.positions.copy().tolist(),
                "targets": env.dynamic_targets.copy().tolist(),
                "completed_targets": env.completed_targets.astype(int).tolist(),
                "obstacles": env.obstacles.copy().tolist(),
                "actions": actions.tolist(),
                "crashed_uavs": result.get("crashed_uavs", []),
            }
        )
        if result["done"]:
            break
    final = history[-1]
    total_collisions = int(sum(row["collisions"] for row in history))
    active_exposure = int(sum(row["active_uav_exposure"] for row in history))
    exposed_distance = float(sum(row["step_path_length"] for row in history))
    path_lengths = []
    n_agents = len(history[0]["positions"])
    for agent_idx in range(n_agents):
        pts = np.asarray([row["positions"][agent_idx] for row in history], dtype=float)
        path_lengths.append(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()))
    return {
        "name": name,
        "config": cfg.__dict__,
        "summary": {
            "steps": env.t,
            "coverage": final["coverage"],
            "target_discovery_rate": final["target_discovery_rate"],
            "target_tracking_rate": final["target_tracking_rate"],
            "target_completion_rate": final["target_completion_rate"],
            "avg_uav_path_length": float(np.mean(path_lengths)),
            "total_uav_path_length": float(np.sum(path_lengths)),
            "collisions": total_collisions,
            "new_crashes": int(sum(row["new_crashes"] for row in history)),
            "final_crashed_count": int(sum(history[-1].get("crashed_uavs", []))),
            "uav_collisions": int(sum(row["uav_collisions"] for row in history)),
            "obstacle_conflicts": int(sum(row["obstacle_conflicts"] for row in history)),
            "pair_conflicts": int(sum(row["pair_conflicts"] for row in history)),
            "collision_rate": float(sum(row["collisions"] for row in history) / max(env.t, 1)),
            "collision_per_1000_active_steps": float(1000.0 * total_collisions / max(active_exposure, 1)),
            "collision_per_1000_distance": float(1000.0 * total_collisions / max(exposed_distance, 1e-6)),
            "crash_rate": float(sum(history[-1].get("crashed_uavs", [])) / max(cfg.n_uavs, 1)),
            "collision_free_completion": float(final["target_completion_rate"] if total_collisions == 0 else 0.0),
            "active_uav_exposure": active_exposure,
            "lost_tracks": int(sum(row["lost_tracks"] for row in history)),
            "avg_escort_score": float(np.mean([row["escort_score"] for row in history])),
            "avg_graph_density": float(np.mean([row["graph_density"] for row in history])),
            "avg_outage_rate": float(np.mean([row["outage_rate"] for row in history])),
            "avg_disabled_rate": float(np.mean([row["disabled_rate"] for row in history])),
            "avg_active_uav_count": float(np.mean([row["active_uav_count"] for row in history])),
            "avg_interference": float(np.mean([row["mean_interference"] for row in history])),
            "total_reward": float(sum(row["reward"] for row in history)),
        },
        "history": history,
    }


def polyline(points: list[tuple[float, float]], color: str, width: int = 2, dash: str | None = None, opacity: float = 1.0) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'


def marker_svg(x: float, y: float, color: str, marker: str) -> str:
    if marker == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.2"/>'
    if marker == "square":
        return f'<rect x="{x-5.5:.1f}" y="{y-5.5:.1f}" width="11" height="11" fill="{color}" stroke="#fff" stroke-width="1.2"/>'
    if marker == "triangle":
        return f'<path d="M {x:.1f} {y-7:.1f} L {x+7:.1f} {y+7:.1f} L {x-7:.1f} {y+7:.1f} Z" fill="{color}" stroke="#fff" stroke-width="1.2"/>'
    if marker == "diamond":
        return f'<path d="M {x:.1f} {y-7:.1f} L {x+7:.1f} {y:.1f} L {x:.1f} {y+7:.1f} L {x-7:.1f} {y:.1f} Z" fill="{color}" stroke="#fff" stroke-width="1.2"/>'
    if marker == "cross":
        return f'<path d="M {x-7:.1f} {y-7:.1f} L {x+7:.1f} {y+7:.1f} M {x+7:.1f} {y-7:.1f} L {x-7:.1f} {y+7:.1f}" stroke="{color}" stroke-width="2.4"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" stroke="#fff" stroke-width="1.2"/>'


def line_chart(path: Path, cases: list[dict], metric: str, title: str) -> None:
    width, height = 1180, 700
    left, top, right, bottom = 95, 92, 70, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    dashes = [None, "7 4", "2 4", "10 3 2 3", "1 3", "12 5", "4 2 1 2"]
    markers = ["circle", "square", "triangle", "diamond", "cross", "circle", "square"]
    max_step = max(row["step"] for case in cases for row in case["history"])
    values = [row[metric] for case in cases for row in case["history"]]
    min_v, max_v = min(values), max(values)
    if abs(max_v - min_v) < 1e-9:
        max_v = min_v + 1.0
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{width/2:.1f}" y="35" font-size="34" font-family="Times New Roman" font-weight="bold" text-anchor="middle">{title}</text>')
    body.append(f'<text x="{width/2:.1f}" y="61" font-size="19" font-family="Times New Roman" fill="#555" text-anchor="middle">X axis: simulation step; Y axis: {METRIC_LABELS.get(metric, metric)}</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = top + tick * plot_h / 5
        v = max_v - tick * (max_v - min_v) / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="{left-34}" y="{y+5:.1f}" font-size="18" font-family="Times New Roman" text-anchor="end">{v:.2f}</text>')
    for tick in range(6):
        x = left + tick * plot_w / 5
        step = tick * max_step / 5
        body.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+28}" font-size="18" font-family="Times New Roman" text-anchor="middle">{step:.0f}</text>')
    for idx, case in enumerate(cases):
        pts = []
        for row in case["history"]:
            x = left + row["step"] / max_step * plot_w
            y = height - bottom - (row[metric] - min_v) / (max_v - min_v) * plot_h
            pts.append((x, y))
        color = colors[idx % len(colors)]
        dash = dashes[idx % len(dashes)]
        marker = markers[idx % len(markers)]
        body.append(polyline(pts, color, width=3, dash=dash, opacity=0.92))
        if pts:
            body.append(marker_svg(pts[-1][0], pts[-1][1], color, marker))
    legend_x = left + plot_w - 315
    legend_y = top + plot_h - 245
    legend_w, legend_h = 292, 218
    body.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" rx="6" ry="6" fill="#fffaf0" stroke="#d9d1c4" opacity="0.94"/>')
    body.append(f'<text x="{legend_x+16}" y="{legend_y+28}" font-size="21" font-family="Times New Roman" font-weight="bold">Legend</text>')
    for idx, case in enumerate(cases):
        color = colors[idx % len(colors)]
        dash = dashes[idx % len(dashes)]
        marker = markers[idx % len(markers)]
        lx = legend_x + 18
        ly = legend_y + 55 + idx * 22
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+36}" y2="{ly}" stroke="{color}" stroke-width="3"{dash_attr}/>')
        body.append(marker_svg(lx + 18, ly, color, marker))
        body.append(f'<text x="{lx+50}" y="{ly+5}" font-size="19" font-family="Times New Roman" fill="#222">{CASE_LABELS.get(case["name"], case["name"])}</text>')
    body.append(f'<text x="{left + plot_w/2:.1f}" y="{height-22}" font-size="20" font-family="Times New Roman" text-anchor="middle">Step</text>')
    body.append(f'<text x="26" y="{top + plot_h/2:.1f}" font-size="20" font-family="Times New Roman" text-anchor="middle" transform="rotate(-90 26,{top + plot_h/2:.1f})">{METRIC_LABELS.get(metric, metric)}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def bar_chart(
    path: Path,
    cases: list[dict],
    title: str = "Ablation Final Metrics",
    subtitle: str = "Left axis: normalized rates; right axis: collisions per 1,000 active-UAV steps.",
) -> None:
    width, height = 1220, 650
    rate_metrics = [
        ("coverage", "Coverage rate", "#1f77b4"),
        ("target_discovery_rate", "Target discovery rate", "#2ca02c"),
        ("target_completion_rate", "Task completion rate", "#111111"),
    ]
    collision_metric = ("collision_per_1000_active_steps", "Collisions / 1k active steps", "#d62728")
    left, right, top, bottom = 78, 92, 82, 128
    plot_w = width - left - right
    base_y = height - bottom
    max_h = height - top - bottom
    collision_values = [float(case["summary"][collision_metric[0]]) for case in cases]
    collision_max = max(collision_values + [1.0])
    collision_axis_max = max(5.0, np.ceil(collision_max / 5.0) * 5.0)
    group_w = plot_w / len(cases)
    bar_w = min(24, group_w / 7)
    gap = bar_w * 0.34
    cluster_w = 4 * bar_w + 3 * gap

    def case_label(name: str) -> tuple[str, str]:
        label = CASE_LABELS.get(name, name).replace(" communication", " comm")
        parts = label.split(" ", 1)
        if len(parts) == 1:
            return parts[0], ""
        return parts[0], parts[1]

    def fmt_count(value: float) -> str:
        return f"{value:.0f}" if abs(value - round(value)) < 1e-9 else f"{value:.1f}"

    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{left}" y="34" font-size="31" font-family="Times New Roman" font-weight="bold">{title}</text>')
    body.append(f'<text x="{left}" y="58" font-size="19" font-family="Times New Roman" fill="#555">{subtitle}</text>')
    body.append(f'<line x1="{left}" y1="{base_y}" x2="{width-right}" y2="{base_y}" stroke="#333" stroke-width="1.2"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{base_y}" stroke="#333" stroke-width="1.2"/>')
    body.append(f'<line x1="{width-right}" y1="{top}" x2="{width-right}" y2="{base_y}" stroke="#333" stroke-width="1.2"/>')
    for tick in range(6):
        y = base_y - tick * max_h / 5
        rate = tick / 5
        collisions = tick * collision_axis_max / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="{left-16}" y="{y+5:.1f}" font-size="19" font-family="Times New Roman" text-anchor="end">{rate:.1f}</text>')
        body.append(f'<text x="{width-right+16}" y="{y+5:.1f}" font-size="19" font-family="Times New Roman" fill="{collision_metric[2]}">{collisions:.0f}</text>')
    for i, case in enumerate(cases):
        center = left + group_w * (i + 0.5)
        x0 = center - cluster_w / 2
        for j, (metric, _label, color) in enumerate(rate_metrics):
            v = float(case["summary"][metric])
            h = max(0.0, min(1.0, v)) * max_h
            x = x0 + j * (bar_w + gap)
            body.append(f'<rect x="{x:.1f}" y="{base_y-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
            body.append(f'<text x="{x + bar_w/2:.1f}" y="{base_y-h-7:.1f}" font-size="16" font-family="Times New Roman" text-anchor="middle">{v:.2f}</text>')
        collisions = float(case["summary"][collision_metric[0]])
        h = min(collisions / collision_axis_max, 1.0) * max_h
        x = x0 + 3 * (bar_w + gap)
        body.append(f'<rect x="{x:.1f}" y="{base_y-h:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{collision_metric[2]}"/>')
        body.append(f'<text x="{x + bar_w/2:.1f}" y="{base_y-h-7:.1f}" font-size="16" font-family="Times New Roman" fill="{collision_metric[2]}" text-anchor="middle">{fmt_count(collisions)}</text>')
        label_a, label_b = case_label(case["name"])
        body.append(f'<text x="{center:.1f}" y="{base_y+27}" font-size="18" font-family="Times New Roman" text-anchor="middle">{label_a}</text>')
        if label_b:
            body.append(f'<text x="{center:.1f}" y="{base_y+47}" font-size="18" font-family="Times New Roman" text-anchor="middle">{label_b}</text>')
    body.append(f'<text x="24" y="{top + max_h/2:.1f}" font-size="20" font-family="Times New Roman" text-anchor="middle" transform="rotate(-90 24,{top + max_h/2:.1f})">Normalized rate</text>')
    body.append(f'<text x="{width-24}" y="{top + max_h/2:.1f}" font-size="20" font-family="Times New Roman" fill="{collision_metric[2]}" text-anchor="middle" transform="rotate(90 {width-24},{top + max_h/2:.1f})">Collisions / 1k active steps</text>')
    legend_items = rate_metrics + [collision_metric]
    legend_y = height - 36
    legend_x = left + 20
    legend_gap = 260
    for j, (_metric, label, color) in enumerate(legend_items):
        x = legend_x + j * legend_gap
        body.append(f'<rect x="{x}" y="{legend_y-13}" width="16" height="16" fill="{color}"/>')
        body.append(f'<text x="{x+24}" y="{legend_y}" font-size="19" font-family="Times New Roman">{label}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def trajectory_svg(path: Path, case: dict) -> None:
    cfg = case.get("config", {})
    world_size = int(cfg.get("grid_size", 25))
    plot_px = 450
    scale = plot_px / max(world_size, 1)
    grid_size = plot_px
    legend_w = 250
    width = grid_size + legend_w
    size = grid_size
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e"]
    n_agents = len(case["history"][0]["positions"])
    agent_traj = [[] for _ in range(n_agents)]
    for row in case["history"]:
        for i, pos in enumerate(row["positions"]):
            agent_traj[i].append([pos[0] + 0.5, pos[1] + 0.5])
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{size}" viewBox="0 0 {width} {size}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="14" y="24" font-size="20" font-family="Times New Roman">{CASE_LABELS.get(case["name"], case["name"])}: UAV search and tracking</text>')
    body.append(f'<text x="14" y="42" font-size="15" font-family="Times New Roman" fill="#555">grid={world_size}x{world_size}, UAVs={cfg.get("n_uavs", n_agents)}, uav_speed={cfg.get("uav_speed", 1.0)}, decision_dt={cfg.get("decision_dt", 1.0)}</text>')
    for i in range(world_size + 1):
        x = i * scale
        stroke = "#e3edf8" if world_size <= 35 or i % max(1, world_size // 25) == 0 else "#f3f6fa"
        body.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{size}" stroke="{stroke}" stroke-width="1"/>')
        body.append(f'<line x1="0" y1="{x:.1f}" x2="{size}" y2="{x:.1f}" stroke="{stroke}" stroke-width="1"/>')
    for x, y in case["history"][-1]["obstacles"]:
        body.append(f'<circle cx="{(x + 0.5) * scale:.1f}" cy="{(y + 0.5) * scale:.1f}" r="6" fill="#9a9a9a" opacity="0.22"/>')
    final_targets = case["history"][-1]["targets"]
    completed = case["history"][-1].get("completed_targets", [0] * len(final_targets))
    for idx, (x, y) in enumerate(final_targets):
        cx = (x + 0.5) * scale
        cy = (y + 0.5) * scale
        if completed[idx]:
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#2ca02c" stroke="#111" stroke-width="1.2"/>')
        else:
            body.append(f'<path d="M {cx-5:.1f} {cy-5:.1f} L {cx+5:.1f} {cy+5:.1f} M {cx+5:.1f} {cy-5:.1f} L {cx-5:.1f} {cy+5:.1f}" stroke="#111" stroke-width="1.8"/>')
    for i, traj in enumerate(agent_traj):
        sampled = traj[::2] if len(traj) > 80 else traj
        body.append(polyline([(p[0] * scale, p[1] * scale) for p in sampled], colors[i % len(colors)], 2))
        sx, sy = traj[0]
        ex, ey = traj[-1]
        color = colors[i % len(colors)]
        body.append(f'<circle cx="{sx*scale:.1f}" cy="{sy*scale:.1f}" r="6" fill="{color}" stroke="#111"/>')
        body.append(f'<text x="{sx*scale+7:.1f}" y="{sy*scale-7:.1f}" font-size="15" font-family="Times New Roman" font-weight="bold" fill="{color}">U{i+1}</text>')
        body.append(f'<rect x="{ex*scale-5:.1f}" y="{ey*scale-5:.1f}" width="10" height="10" fill="{color}" stroke="#111"/>')
        body.append(f'<text x="{ex*scale+7:.1f}" y="{ey*scale+4:.1f}" font-size="15" font-family="Times New Roman" font-weight="bold" fill="{color}">U{i+1}</text>')
        if np.linalg.norm(np.asarray(traj[-1]) - np.asarray(traj[0])) < 0.1:
            body.append(f'<circle cx="{ex*scale:.1f}" cy="{ey*scale:.1f}" r="10" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 3"/>')
    lx = grid_size + 18
    body.append(f'<rect x="{grid_size}" y="0" width="{legend_w}" height="{size}" fill="#fffaf0" stroke="#ddd"/>')
    body.append(f'<text x="{lx}" y="28" font-size="19" font-family="Times New Roman" font-weight="bold">Legend</text>')
    body.append(f'<circle cx="{lx+8}" cy="55" r="6" fill="#d62728" stroke="#111"/>')
    body.append(f'<text x="{lx+24}" y="60" font-size="16" font-family="Times New Roman">UAV start</text>')
    body.append(f'<rect x="{lx+2}" y="77" width="12" height="12" fill="#d62728" stroke="#111"/>')
    body.append(f'<text x="{lx+24}" y="88" font-size="16" font-family="Times New Roman">UAV final position</text>')
    body.append(f'<line x1="{lx+2}" y1="112" x2="{lx+32}" y2="112" stroke="#1f77b4" stroke-width="3"/>')
    body.append(f'<text x="{lx+42}" y="116" font-size="16" font-family="Times New Roman">UAV path</text>')
    body.append(f'<path d="M {lx+2} 136 L {lx+14} 148 M {lx+14} 136 L {lx+2} 148" stroke="#111" stroke-width="2"/>')
    body.append(f'<text x="{lx+24}" y="148" font-size="16" font-family="Times New Roman">Active target</text>')
    body.append(f'<circle cx="{lx+8}" cy="174" r="6" fill="#2ca02c" stroke="#111"/>')
    body.append(f'<text x="{lx+24}" y="178" font-size="16" font-family="Times New Roman">Completed target</text>')
    body.append(f'<circle cx="{lx+8}" cy="204" r="6" fill="#9a9a9a" opacity="0.22"/>')
    body.append(f'<text x="{lx+24}" y="208" font-size="16" font-family="Times New Roman">Dynamic obstacle</text>')
    body.append(f'<text x="{lx}" y="242" font-size="16" font-family="Times New Roman">Only target states are shown.</text>')
    body.append(f'<text x="{lx}" y="262" font-size="16" font-family="Times New Roman">Target full paths are hidden.</text>')
    for i in range(n_agents):
        y = 286 + i * 18
        color = colors[i % len(colors)]
        body.append(f'<line x1="{lx}" y1="{y}" x2="{lx+22}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{lx+30}" y="{y+4}" font-size="15" font-family="Times New Roman">U{i+1} path</text>')
    body.append(f'<text x="{lx}" y="{300 + n_agents * 18}" font-size="16" font-family="Times New Roman">Summary</text>')
    s = case["summary"]
    summary_y = 324 + n_agents * 18
    body.append(f'<text x="{lx}" y="{summary_y}" font-size="15" font-family="Times New Roman">coverage={s["coverage"]:.3f}</text>')
    body.append(f'<text x="{lx}" y="{summary_y+18}" font-size="15" font-family="Times New Roman">discovery={s["target_discovery_rate"]:.3f}</text>')
    body.append(f'<text x="{lx}" y="{summary_y+36}" font-size="15" font-family="Times New Roman">tracking={s["target_tracking_rate"]:.3f}</text>')
    body.append(f'<text x="{lx}" y="{summary_y+54}" font-size="15" font-family="Times New Roman">completed={s.get("target_completion_rate", 0):.3f}</text>')
    body.append(f'<text x="{lx}" y="{summary_y+72}" font-size="15" font-family="Times New Roman">collisions={s["collisions"]}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def motion_svg(path: Path, case: dict) -> None:
    cfg = case.get("config", {})
    world_size = int(cfg.get("grid_size", 25))
    plot_px = 450
    scale = plot_px / max(world_size, 1)
    grid_size = plot_px
    legend_w = 250
    width = grid_size + legend_w
    size = grid_size
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{size}" viewBox="0 0 {width} {size}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="14" y="24" font-size="20" font-family="Times New Roman">{CASE_LABELS.get(case["name"], case["name"])}: target/obstacle motion</text>')
    for i in range(world_size + 1):
        x = i * scale
        stroke = "#e3edf8" if world_size <= 35 or i % max(1, world_size // 25) == 0 else "#f3f6fa"
        body.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{size}" stroke="{stroke}" stroke-width="1"/>')
        body.append(f'<line x1="0" y1="{x:.1f}" x2="{size}" y2="{x:.1f}" stroke="{stroke}" stroke-width="1"/>')
    target_count = len(case["history"][0]["targets"])
    obstacle_count = len(case["history"][0]["obstacles"])
    for idx in range(target_count):
        pts = []
        for row in case["history"][::4]:
            x, y = row["targets"][idx]
            pts.append(((x + 0.5) * scale, (y + 0.5) * scale))
        if len(pts) > 1:
            body.append(polyline(pts, "#111111", width=1, dash="3 4", opacity=0.35))
        x, y = case["history"][-1]["targets"][idx]
        completed = case["history"][-1].get("completed_targets", [0] * target_count)[idx]
        color = "#2ca02c" if completed else "#111111"
        body.append(f'<circle cx="{(x+0.5)*scale:.1f}" cy="{(y+0.5)*scale:.1f}" r="4" fill="{color}" stroke="#111"/>')
    for idx in range(obstacle_count):
        pts = []
        for row in case["history"][::6]:
            x, y = row["obstacles"][idx]
            pts.append(((x + 0.5) * scale, (y + 0.5) * scale))
        if len(pts) > 1:
            body.append(polyline(pts, "#8a8a8a", width=1, dash="2 5", opacity=0.22))
        x, y = case["history"][-1]["obstacles"][idx]
        body.append(f'<circle cx="{(x+0.5)*scale:.1f}" cy="{(y+0.5)*scale:.1f}" r="5" fill="#9a9a9a" opacity="0.25"/>')
    lx = grid_size + 18
    body.append(f'<rect x="{grid_size}" y="0" width="{legend_w}" height="{size}" fill="#fffaf0" stroke="#ddd"/>')
    body.append(f'<text x="{lx}" y="28" font-size="19" font-family="Times New Roman" font-weight="bold">Legend</text>')
    body.append(f'<line x1="{lx+2}" y1="60" x2="{lx+42}" y2="60" stroke="#111" stroke-width="1" stroke-dasharray="3 4" opacity="0.55"/>')
    body.append(f'<text x="{lx+52}" y="64" font-size="16" font-family="Times New Roman">Target motion trail</text>')
    body.append(f'<circle cx="{lx+10}" cy="88" r="5" fill="#2ca02c" stroke="#111"/>')
    body.append(f'<text x="{lx+25}" y="92" font-size="16" font-family="Times New Roman">Completed target</text>')
    body.append(f'<circle cx="{lx+10}" cy="116" r="5" fill="#111" stroke="#111"/>')
    body.append(f'<text x="{lx+25}" y="120" font-size="16" font-family="Times New Roman">Active target</text>')
    body.append(f'<line x1="{lx+2}" y1="146" x2="{lx+42}" y2="146" stroke="#8a8a8a" stroke-width="1" stroke-dasharray="2 5" opacity="0.35"/>')
    body.append(f'<text x="{lx+52}" y="150" font-size="16" font-family="Times New Roman">Obstacle motion trail</text>')
    body.append(f'<circle cx="{lx+10}" cy="176" r="6" fill="#9a9a9a" opacity="0.25"/>')
    body.append(f'<text x="{lx+25}" y="180" font-size="16" font-family="Times New Roman">Obstacle final position</text>')
    body.append(f'<text x="{lx}" y="220" font-size="16" font-family="Times New Roman">Trails are downsampled</text>')
    body.append(f'<text x="{lx}" y="240" font-size="16" font-family="Times New Roman">to avoid clutter.</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_analysis_report(path: Path, results: list[dict]) -> None:
    by_name = {r["name"]: r for r in results}
    real = by_name["outage_dwa_orca"]["summary"]
    full_comm = by_name["full_comm_dwa_orca"]["summary"]
    no_avoid = by_name["outage_no_avoidance"]["summary"]
    dwa_only = by_name["outage_dwa_only"]["summary"]
    lines = [
        "# 消融实验输出解释",
        "",
        "## 实验组说明",
        "",
        "- `outage_dwa_orca`：真实干扰断联环境 + DWA-ORCA，是当前主设定。",
        "- `full_comm_dwa_orca`：理想全通信 + DWA-ORCA，用于观察通信中断造成的协作损失。",
        "- `outage_no_avoidance`：真实干扰断联环境 + 无避障。",
        "- `outage_dwa_only`：真实干扰断联环境 + 仅DWA。",
        "",
        "判断是否发生断联，优先看 `avg_outage_rate`、`avg_disabled_rate`、`avg_active_uav_count` 和 `avg_graph_density`。",
        "",
        "## 关键对比",
        "",
        f"- 真实断联组 avg_outage_rate={real['avg_outage_rate']:.4f}，avg_disabled_rate={real['avg_disabled_rate']:.4f}，avg_active_uav_count={real['avg_active_uav_count']:.2f}。",
        f"- 全通信组 coverage={full_comm['coverage']:.4f}，真实断联组 coverage={real['coverage']:.4f}。",
        f"- 全通信组 completion={full_comm['target_completion_rate']:.4f}，真实断联组 completion={real['target_completion_rate']:.4f}。",
        f"- 每千有效 UAV 步碰撞：无避障={no_avoid['collision_per_1000_active_steps']:.3f}，仅DWA={dwa_only['collision_per_1000_active_steps']:.3f}，DWA-ORCA={real['collision_per_1000_active_steps']:.3f}。",
        "",
        "## 论文中建议报告",
        "",
        "1. 通信对比：`full_comm_dwa_orca` vs `outage_dwa_orca`。",
        "2. 避障对比使用相同无风险项名义策略，报告每千有效 UAV 步碰撞、障碍碰撞、UAV 碰撞、失效率和无碰撞完成率。",
        "3. 断联是否发生：报告 `avg_outage_rate`、`avg_disabled_rate`、`avg_active_uav_count`。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return
    by_name = {r["name"]: r for r in results}
    best_coverage = max(results, key=lambda r: r["summary"]["coverage"])
    best_discovery = max(results, key=lambda r: r["summary"]["target_discovery_rate"])
    best_tracking = max(results, key=lambda r: r["summary"]["target_tracking_rate"])
    safest = min(results, key=lambda r: r["summary"]["collisions"])
    full = by_name["full_model"]["summary"]
    no_avoid = by_name["no_avoidance"]["summary"]
    no_comm = by_name["no_comm"]["summary"]
    full_comm = by_name["full_comm"]["summary"]
    lines = [
        "# 消融实验输出解释",
        "",
        "## 图中元素说明",
        "",
        "- 轨迹图中彩色线表示不同 UAV 的运动轨迹。",
        "- 彩色圆点表示 UAV 起点，彩色方块表示 UAV 最终位置。",
        "- 黑色细线表示动态目标轨迹，黑色菱形表示目标最终位置。",
        "- 灰色半透明圆表示动态障碍物位置。",
        "- 曲线图横轴为仿真步数，纵轴为对应指标。",
        "- 柱状图展示各实验组最终归一化指标，碰撞数请查看 `ablation_summary.csv`。",
        "",
        "## 当前结果概览",
        "",
        f"- 覆盖率最高的是 `{best_coverage['name']}`，coverage={best_coverage['summary']['coverage']:.4f}。",
        f"- 目标发现率最高的是 `{best_discovery['name']}`，discovery={best_discovery['summary']['target_discovery_rate']:.4f}。",
        f"- 目标跟踪率最高的是 `{best_tracking['name']}`，tracking={best_tracking['summary']['target_tracking_rate']:.4f}。",
        f"- 碰撞数最低的是 `{safest['name']}`，collisions={safest['summary']['collisions']}。",
        "",
        "## 关键现象解释",
        "",
        f"- 完整模型 coverage={full['coverage']:.4f}，discovery={full['target_discovery_rate']:.4f}，tracking={full['target_tracking_rate']:.4f}，collisions={full['collisions']}。说明局部信念图和弱通信机制能够提高动态搜索覆盖与发现能力，但持续跟踪仍然偏弱。",
        f"- `no_avoidance` 的碰撞数为 {no_avoid['collisions']}，高于完整模型的 {full['collisions']}。这说明 DWA/ORCA 避障层确实降低了风险，但当前避障仍不够强，需要进一步优化动态障碍预测和安全距离控制。",
        f"- `full_comm` 和 `no_comm` 的覆盖率分别为 {full_comm['coverage']:.4f} 和 {no_comm['coverage']:.4f}，差距很小。这说明当前启发式策略还没有充分利用通信信息，后续需要用 MAPPO/QMIX 学习何时交换、信任和使用邻居信念。",
        f"- `target_tracking_rate` 整体较低，说明策略更偏向覆盖搜索和偶然发现目标，还没有形成发现目标后的持续跟踪/围捕行为。",
        "",
        "## 对 MAPPO/QMIX 训练输出的解释",
        "",
        "- 你当前 30 episode 的 MAPPO 结果 coverage=0.6304，discovery=0.8，tracking=0.0，说明 MAPPO 已经比启发式在覆盖率上有提升潜力，但尚未学会稳定跟踪。",
        "- 你当前 30 episode 的 QMIX 结果 coverage=0.3072，discovery=0.5333，tracking=0.0，说明当前 QMIX 在短训练下探索不足，且离散动作价值分解还没有稳定学到协同搜索策略。",
        "- 两个算法 tracking 都为 0，主要原因是奖励中“发现目标”和“降低不确定性”更容易获得，而持续跟踪奖励稀疏、目标动态移动、动作受避障和转向约束影响。",
        "",
        "## 后续改进方向",
        "",
        "1. 增强跟踪奖励：发现目标后增加保持距离、持续可见、协同包围和目标丢失惩罚。",
        "2. 加入任务阶段切换：搜索阶段重覆盖率，发现后切换到跟踪阶段，重目标保持和多 UAV 分工。",
        "3. 改进通信策略：从固定机会式融合升级为学习型消息门控，只传高不确定区域、目标状态和障碍风险。",
        "4. 强化动态图网络：将当前手工图特征替换为可训练 GAT/GraphSAGE 编码器。",
        "5. 优化避障层：加入更长时间窗的动态障碍预测，区分真实碰撞、危险接近和被迫悬停。",
        "6. 扩大训练规模：30 episode 只适合连通性测试，建议至少 500-2000 episode，并保存多随机种子的均值和方差。",
        "7. 做参数敏感性实验：目标速度、障碍密度、丢包率、干扰强度、通信半径分别扫描。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds for mean/std ablations, e.g. 11,23,41,59,83.")
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--obstacle-speed", type=float, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--decision-dt", type=float, default=None)
    parser.add_argument("--max-turn-steps", type=int, default=None, help="Maximum heading change per step in 45-degree units.")
    parser.add_argument("--allow-hover", action="store_true", help="Allow UAVs to choose the hover action even when movement is feasible.")
    parser.add_argument("--no-normalize-diagonal-speed", action="store_true", help="Keep diagonal grid moves longer than straight moves.")
    parser.add_argument("--comm-radius", type=float, default=None)
    parser.add_argument("--outage-base-prob", type=float, default=None)
    parser.add_argument("--outage-jammed-prob", type=float, default=None)
    parser.add_argument("--failure-base-prob", type=float, default=None)
    parser.add_argument("--failure-jammed-prob", type=float, default=None)
    parser.add_argument("--jammer-effect-radius", type=float, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    parser.add_argument("--avoidance-prediction-horizon", type=int, default=None)
    parser.add_argument("--obstacle-prediction-buffer", type=float, default=None)
    parser.add_argument("--pair-prediction-buffer", type=float, default=None)
    return parser.parse_args()


def parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    return [args.seed]


def make_base_config(args: argparse.Namespace) -> WeakCommConfig:
    cfg = WeakCommConfig(seed=args.seed)
    updates = {
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_obstacles": args.n_obstacles,
        "grid_size": args.grid_size,
        "search_steps": args.search_steps,
        "target_completion_steps": args.completion_steps,
        "target_speed": args.target_speed,
        "obstacle_speed": args.obstacle_speed,
        "uav_speed": args.uav_speed,
        "decision_dt": args.decision_dt,
        "max_turn_steps": args.max_turn_steps,
        "allow_hover": True if args.allow_hover else None,
        "normalize_diagonal_speed": False if args.no_normalize_diagonal_speed else None,
        "comm_radius": args.comm_radius,
        "outage_base_prob": args.outage_base_prob,
        "outage_jammed_prob": args.outage_jammed_prob,
        "failure_base_prob": args.failure_base_prob,
        "failure_jammed_prob": args.failure_jammed_prob,
        "jammer_effect_radius": args.jammer_effect_radius,
        "tpm_prior_base": args.tpm_prior_base,
        "tpm_target_prior_enabled": True if args.enable_target_prior else None,
        "tpm_target_prior_strength": args.tpm_target_prior_strength,
        "tpm_target_prior_sigma": args.tpm_target_prior_sigma,
        "avoidance_prediction_horizon": args.avoidance_prediction_horizon,
        "obstacle_prediction_buffer": args.obstacle_prediction_buffer,
        "pair_prediction_buffer": args.pair_prediction_buffer,
    }
    for key, value in updates.items():
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def case_configs(base: WeakCommConfig) -> list[tuple[str, WeakCommConfig]]:
    return [
        ("outage_dwa_orca", replace(base, comm_mode="outage", avoidance_mode="dwa_orca")),
        ("full_comm_dwa_orca", replace(base, comm_mode="full", avoidance_mode="dwa_orca")),
        ("outage_no_avoidance", replace(base, comm_mode="outage", avoidance_mode="none")),
        ("outage_dwa_only", replace(base, comm_mode="outage", avoidance_mode="dwa")),
    ]


def aggregate_seed_rows(rows: list[dict]) -> list[dict]:
    metric_names = [
        "steps",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "avg_uav_path_length",
        "total_uav_path_length",
        "collisions",
        "new_crashes",
        "final_crashed_count",
        "uav_collisions",
        "obstacle_conflicts",
        "pair_conflicts",
        "collision_rate",
        "collision_per_1000_active_steps",
        "collision_per_1000_distance",
        "crash_rate",
        "collision_free_completion",
        "active_uav_exposure",
        "lost_tracks",
        "avg_escort_score",
        "avg_graph_density",
        "avg_outage_rate",
        "avg_disabled_rate",
        "avg_active_uav_count",
        "avg_interference",
        "total_reward",
    ]
    cases = list(CASE_LABELS)
    out = []
    for case in cases:
        case_rows = [row for row in rows if row["case"] == case]
        if not case_rows:
            continue
        agg = {"case": case, "n_seeds": len(case_rows)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in case_rows], dtype=float)
            agg[f"{metric}_mean"] = float(values.mean())
            agg[f"{metric}_std"] = float(values.std(ddof=0))
        out.append(agg)
    return out


def mean_rows_as_cases(rows: list[dict]) -> list[dict]:
    # Keep this adapter aligned with every metric consumed by ``bar_chart``.
    # The aggregated rows store means with a ``_mean`` suffix, whereas
    # ``bar_chart`` deliberately consumes the same unsuffixed schema as a
    # single-seed result.
    metrics = [
        "coverage",
        "target_discovery_rate",
        "target_completion_rate",
        "collision_per_1000_active_steps",
    ]
    cases = []
    for row in rows:
        cases.append(
            {
                "name": row["case"],
                "summary": {metric: row[f"{metric}_mean"] for metric in metrics},
            }
        )
    return cases


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args)
    base = make_base_config(args)
    print(
        json.dumps(
            {
                "effective_config": {
                    "n_uavs": base.n_uavs,
                    "n_targets": base.n_targets,
                    "grid_size": base.grid_size,
                    "search_steps": base.search_steps,
                    "n_obstacles": base.n_obstacles,
                    "allow_hover": base.allow_hover,
                    "avoidance_prediction_horizon": base.avoidance_prediction_horizon,
                    "obstacle_prediction_buffer": base.obstacle_prediction_buffer,
                    "pair_prediction_buffer": base.pair_prediction_buffer,
                },
                "seeds": seeds,
            },
            indent=2,
        )
    )
    if len(seeds) > 1:
        base = replace(base, seed=seeds[0])
    cases = case_configs(base)
    results = [run_case(name, cfg) for name, cfg in cases]
    (OUT / "ablation_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_rows = [{"case": r["name"], **r["summary"]} for r in results]
    write_csv(
        OUT / "ablation_summary.csv",
        summary_rows,
        [
            "case",
            "steps",
            "coverage",
            "target_discovery_rate",
            "target_tracking_rate",
            "target_completion_rate",
            "avg_uav_path_length",
            "total_uav_path_length",
            "collisions",
            "new_crashes",
            "final_crashed_count",
            "uav_collisions",
            "obstacle_conflicts",
            "pair_conflicts",
            "collision_rate",
            "collision_per_1000_active_steps",
            "collision_per_1000_distance",
            "crash_rate",
            "collision_free_completion",
            "active_uav_exposure",
            "lost_tracks",
            "avg_escort_score",
            "avg_graph_density",
            "avg_outage_rate",
            "avg_disabled_rate",
            "avg_active_uav_count",
            "avg_interference",
            "total_reward",
        ],
    )
    for result in results:
        write_csv(
            OUT / f"{result['name']}_timeseries.csv",
            result["history"],
            [
                "step",
                "coverage",
                "reward",
                "target_discovery_rate",
                "target_tracking_rate",
                "target_completion_rate",
                "task_phase",
                "lost_tracks",
                "escort_score",
                "collisions",
                "new_crashes",
                "uav_collisions",
                "obstacle_conflicts",
                "pair_conflicts",
                "graph_density",
                "outage_rate",
                "disabled_rate",
                "active_uav_count",
                "mean_interference",
                "positions",
                "targets",
                "obstacles",
                "actions",
                "crashed_uavs",
            ],
        )
        trajectory_svg(OUT / f"{result['name']}_trajectory.svg", result)
        motion_svg(OUT / f"{result['name']}_target_obstacle_motion.svg", result)
    line_chart(OUT / "coverage_curves.svg", results, "coverage", "Coverage Curves")
    line_chart(OUT / "tracking_curves.svg", results, "target_tracking_rate", "Tracking Curves")
    line_chart(OUT / "completion_curves.svg", results, "target_completion_rate", "Completion Curves")
    line_chart(OUT / "discovery_curves.svg", results, "target_discovery_rate", "Discovery Curves")
    line_chart(OUT / "graph_density_curves.svg", results, "graph_density", "Graph Density Curves")
    line_chart(OUT / "outage_rate_curves.svg", results, "outage_rate", "Outage Rate Curves")
    line_chart(OUT / "disabled_rate_curves.svg", results, "disabled_rate", "Disabled UAV Rate Curves")
    bar_chart(OUT / "final_metrics.svg", results)
    write_analysis_report(OUT / "experiment_analysis.zh-CN.md", results)
    if len(seeds) > 1:
        bar_chart(
            OUT / f"final_metrics_seed_{base.seed}.svg",
            results,
            title=f"Ablation Final Metrics (Seed {base.seed})",
            subtitle="Single-seed result; multi-seed mean is saved as final_metrics.svg.",
        )
        all_rows = [{"seed": base.seed, "case": r["name"], **r["summary"]} for r in results]
        all_results = results.copy()
        for seed in seeds:
            if seed == base.seed:
                continue
            seed_base = replace(base, seed=seed)
            for name, cfg in case_configs(seed_base):
                case_result = run_case(name, cfg)
                all_results.append(case_result)
                all_rows.append({"seed": seed, "case": name, **case_result["summary"]})
        mean_std_rows = aggregate_seed_rows(all_rows)
        (OUT / "ablation_summary_by_seed.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        (OUT / "ablation_summary_mean_std.json").write_text(json.dumps(mean_std_rows, indent=2), encoding="utf-8")
        write_csv(
            OUT / "ablation_summary_by_seed.csv",
            all_rows,
            [
                "seed",
                "case",
                "steps",
                "coverage",
                "target_discovery_rate",
                "target_tracking_rate",
                "target_completion_rate",
                "avg_uav_path_length",
                "total_uav_path_length",
                "collisions",
                "new_crashes",
                "final_crashed_count",
                "uav_collisions",
                "obstacle_conflicts",
                "pair_conflicts",
                "collision_rate",
                "collision_per_1000_active_steps",
                "collision_per_1000_distance",
                "crash_rate",
                "collision_free_completion",
                "active_uav_exposure",
                "lost_tracks",
                "avg_escort_score",
                "avg_graph_density",
                "avg_outage_rate",
                "avg_disabled_rate",
                "avg_active_uav_count",
                "avg_interference",
                "total_reward",
            ],
        )
        mean_std_fields = ["case", "n_seeds"]
        for key in [
            "steps",
            "coverage",
            "target_discovery_rate",
            "target_tracking_rate",
            "target_completion_rate",
            "avg_uav_path_length",
            "total_uav_path_length",
            "collisions",
            "new_crashes",
            "final_crashed_count",
            "uav_collisions",
            "obstacle_conflicts",
            "pair_conflicts",
            "collision_rate",
            "collision_per_1000_active_steps",
            "collision_per_1000_distance",
            "crash_rate",
            "collision_free_completion",
            "active_uav_exposure",
            "lost_tracks",
            "avg_escort_score",
            "avg_graph_density",
            "avg_outage_rate",
            "avg_disabled_rate",
            "avg_active_uav_count",
            "avg_interference",
            "total_reward",
        ]:
            mean_std_fields.extend([f"{key}_mean", f"{key}_std"])
        write_csv(OUT / "ablation_summary_mean_std.csv", mean_std_rows, mean_std_fields)
        mean_cases = mean_rows_as_cases(mean_std_rows)
        bar_chart(
            OUT / "final_metrics_mean.svg",
            mean_cases,
            title=f"Ablation Final Metrics ({len(seeds)}-Seed Mean)",
            subtitle="Bars show mean over random seeds; standard deviations are in ablation_summary_mean_std.csv.",
        )
        bar_chart(
            OUT / "final_metrics.svg",
            mean_cases,
            title=f"Ablation Final Metrics ({len(seeds)}-Seed Mean)",
            subtitle="Bars show mean over random seeds; standard deviations are in ablation_summary_mean_std.csv.",
        )
        print(json.dumps(mean_std_rows, indent=2))
    else:
        print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
