"""Train MAPPO or QMIX on cooperative search and pursuit environments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from marl_trainers import (
    IPPOTrainer,
    MAPPOTrainer,
    QMIXTrainer,
    TrainConfig,
    seed_everything,
)
from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
from pursuit_visualization import write_pursuit_animation_html


def moving_average(values: list[float], window: int | None = None) -> list[float]:
    if not values:
        return []
    if window is None:
        window = max(5, len(values) // 25)
    window = max(1, min(window, len(values)))
    left = window // 2
    smoothed = []
    for idx in range(len(values)):
        lo = max(0, idx - left)
        hi = min(len(values), lo + window)
        lo = max(0, hi - window)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def metric_points(
    rows: list[dict],
    metric: str,
    max_ep: int,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    min_v: float,
    max_v: float,
    *,
    values: list[float] | None = None,
) -> str:
    values = values if values is not None else [float(row[metric]) for row in rows]
    pts = []
    for row, value in zip(rows, values):
        x = left + row["episode"] / max_ep * plot_width
        y = top + (max_v - value) / (max_v - min_v) * plot_height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def write_training_csv(path: Path, history: list[dict]) -> None:
    fields = [
        "episode",
        "reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "pair_conflicts",
        "collision_rate",
        "outage_rate",
        "disabled_rate",
        "safety_intervention_rate",
        "new_cells",
        "revisit_gain",
        "stagnant_agent_rate",
        "new_crashes",
        "capture_success",
        "capture_hold_count",
        "capture_close_uavs",
        "capture_angular_span_deg",
        "direct_target_visible",
        "first_onboard_contact_time",
        "first_detection_time",
        "capture_time",
        "target_loss_count",
        "target_reacquisition_count",
        "mean_reacquisition_time",
        "continuous_visibility_ratio",
        "track_position_rmse",
        "track_velocity_rmse",
        "mean_track_nees",
        "track_nees_consistency_rate",
        "mean_track_age",
        "valid_track_rate",
        "track_messages",
        "track_communication_bytes",
        "assignment_search_rate",
        "assignment_switch_rate",
        "collision_exposure_rate",
        "mpc_feasible_rate",
        "path_length",
        "control_smoothness_cost",
        "weight_target",
        "weight_exploration",
        "weight_revisit",
        "weight_connectivity",
        "weight_safety",
        "policy_loss",
        "value_loss",
        "entropy",
        "learning_rate",
        "entropy_coef",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "ppo_early_stop",
        "steps",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k) for k in fields})


def tail_average(history: list[dict], window: int = 50) -> dict:
    rows = history[-min(window, len(history)) :]
    metrics = [
        "reward",
        "spectrum_reward",
        "spectrum_success_rate",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
        "collisions",
        "uav_collisions",
        "obstacle_conflicts",
        "pair_conflicts",
        "collision_rate",
        "outage_rate",
        "disabled_rate",
        "capture_success",
        "first_onboard_contact_time",
        "first_detection_time",
        "capture_time",
        "target_loss_count",
        "target_reacquisition_count",
        "mean_reacquisition_time",
        "continuous_visibility_ratio",
        "track_position_rmse",
        "track_velocity_rmse",
        "mean_track_nees",
        "track_nees_consistency_rate",
        "mean_track_age",
        "valid_track_rate",
        "track_messages",
        "track_communication_bytes",
        "assignment_search_rate",
        "assignment_switch_rate",
        "collision_exposure_rate",
        "mpc_feasible_rate",
        "path_length",
        "control_smoothness_cost",
        "steps",
    ]
    summary = {
        "episodes_averaged": len(rows),
        "from_episode": rows[0]["episode"],
        "to_episode": rows[-1]["episode"],
    }
    for metric in metrics:
        values = [float(row.get(metric, 0.0)) for row in rows]
        summary[metric] = sum(values) / len(values)
    return summary


def write_training_svg(
    path: Path,
    algo: str,
    history: list[dict],
    *,
    show_raw: bool = False,
    smooth_window: int | None = None,
    y_min: float | None = None,
    y_max: float | None = None,
    metrics: list[tuple[str, str, str]] | None = None,
    figure_title: str | None = None,
) -> None:
    width, height = 980, 520
    left, top, right, bottom = 80, 55, 55, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    font = "Times New Roman"
    if metrics is None:
        metrics = [
            ("coverage", "#1f77b4", "Coverage"),
            ("target_discovery_rate", "#2ca02c", "Discovery"),
            ("target_tracking_rate", "#9467bd", "Tracking"),
            ("target_completion_rate", "#111111", "Completed"),
        ]
    max_ep = max(row["episode"] for row in history)
    trends = {
        metric: moving_average([float(row[metric]) for row in history], smooth_window)
        for metric, _color, _label in metrics
    }
    trend_values_all = [value for values in trends.values() for value in values]
    data_min, data_max = min(trend_values_all), max(trend_values_all)
    padding = max(0.025, 0.08 * max(data_max - data_min, 0.1))
    min_v = max(0.0, data_min - padding) if y_min is None else float(y_min)
    max_v = min(1.0, data_max + padding) if y_max is None else float(y_max)
    if max_v <= min_v:
        raise ValueError("y_max must be greater than y_min")
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    title = figure_title or f"{algo.upper()} Training Curves"
    body.append(f'<text x="{width/2:.1f}" y="27" font-size="30" font-family="{font}" font-weight="bold" text-anchor="middle">{title}</text>')
    subtitle = f'Moving-average trends only (window={smooth_window or max(5, len(history)//25)}); y-axis is data-adaptive.'
    body.append(f'<text x="{width/2:.1f}" y="48" font-size="19" font-family="{font}" fill="#555" text-anchor="middle">{subtitle}</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = top + tick * plot_height / 5
        v = max_v - tick * (max_v - min_v) / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="18" y="{y+5:.1f}" font-size="18" font-family="{font}">{v:.2f}</text>')
    for tick in range(6):
        x = left + tick * plot_width / 5
        episode = tick * max_ep / 5
        body.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+25}" font-size="18" font-family="{font}" text-anchor="middle">{episode:.0f}</text>')
    for metric, color, _label in metrics:
        raw_values = [float(row[metric]) for row in history]
        trend_values = trends[metric]
        trend_pts = metric_points(history, metric, max_ep, left, top, plot_width, plot_height, min_v, max_v, values=trend_values)
        if show_raw:
            raw_pts = metric_points(history, metric, max_ep, left, top, plot_width, plot_height, min_v, max_v, values=raw_values)
            body.append(f'<polyline points="{raw_pts}" fill="none" stroke="{color}" stroke-width="1.2" opacity="0.14"/>')
        body.append(f'<polyline points="{trend_pts}" fill="none" stroke="{color}" stroke-width="3.4" stroke-linejoin="round" stroke-linecap="round"/>')
    final = tail_average(history, 50)
    stat_metrics = [(metric, label) for metric, _color, label in metrics]
    stat_start = 56 + len(metrics) * 29 + 12
    panel_w = 218
    panel_h = stat_start + len(stat_metrics) * 22 + 14
    panel_x = width - right - panel_w - 18
    panel_y = height - bottom - panel_h - 34
    body.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#fbfbf8" opacity="0.92"/>')
    body.append(f'<text x="{panel_x+18}" y="{panel_y+25}" font-size="20" font-family="{font}" font-weight="bold">Legend</text>')
    for idx, (_metric, color, label) in enumerate(metrics):
        ly = panel_y + 56 + idx * 29
        lx = panel_x + 18
        body.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+30}" y2="{ly}" stroke="{color}" stroke-width="3.4" stroke-linecap="round"/>')
        body.append(f'<circle cx="{lx+15}" cy="{ly}" r="4.2" fill="{color}"/>')
        body.append(f'<text x="{lx+46}" y="{ly+5}" font-size="19" font-family="{font}">{label}</text>')
    stat_x = panel_x + 18
    stat_y = panel_y + stat_start
    for idx, (metric, label) in enumerate(stat_metrics):
        body.append(
            f'<text x="{stat_x}" y="{stat_y + idx * 22}" font-size="19" '
            f'font-family="{font}">{label}={final.get(metric, 0):.3f}</text>'
        )
    body.append(f'<text x="{left + plot_width/2:.1f}" y="{height-16}" font-size="19" font-family="{font}" text-anchor="middle">Episode</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def svg_polyline(points: list[tuple[float, float]], color: str, width: float = 2.0, opacity: float = 1.0, dash: str | None = None) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'


def write_episode_scene_3d_svg(path: Path, algo: str, trace: dict) -> None:
    """Render a pursuit trace as an isometric 3-D scene with capture metrics."""
    frames = trace.get("frames", [])
    if not frames:
        return
    final = frames[-1]
    grid_size = int(final.get("grid_size", 25))
    width, height, panel_x = 940, 620, 690
    origin_x, origin_y = 42.0, 535.0
    xy_scale, depth_scale, z_scale = 18.0, 7.0, 20.0
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]

    def project(x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        return origin_x + float(x) * xy_scale + float(y) * depth_scale, origin_y - float(y) * depth_scale - float(z) * z_scale

    def polygon(points, fill, stroke="#555", opacity=1.0, width_=1.0):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width_}" opacity="{opacity}"/>'

    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="20" y="28" font-size="22" font-family="Times New Roman">{algo.upper()} episode {trace.get("episode")} 3-D capture scene</text>')
    body.append('<text x="20" y="49" font-size="15" font-family="Times New Roman" fill="#555">Quadrotor and evader trajectories, building prisms, and terminal capture outcome.</text>')

    # Ground plane and axes.
    ground = [project(0, 0), project(grid_size, 0), project(grid_size, grid_size), project(0, grid_size)]
    body.append(polygon(ground, "#f5f7f5", "#a7aaa7", 1.0, 1.2))
    for value in range(0, grid_size + 1, 5):
        body.append(svg_polyline([project(value, 0), project(value, grid_size)], "#dfe4df", 0.8))
        body.append(svg_polyline([project(0, value), project(grid_size, value)], "#dfe4df", 0.8))
    body.append(svg_polyline([project(0, 0), project(grid_size + 1, 0)], "#333", 1.5))
    body.append(svg_polyline([project(0, 0), project(0, grid_size + 1)], "#333", 1.5))
    body.append(svg_polyline([project(0, 0), project(0, 0, 12)], "#333", 1.5))
    for label, point in (("x", project(grid_size + 1.4, 0)), ("y", project(0, grid_size + 1.4)), ("z", project(0, 0, 12.8))):
        body.append(f'<text x="{point[0]:.1f}" y="{point[1]:.1f}" font-size="17" font-family="Times New Roman" font-style="italic">{label}</text>')

    # Static building prisms, drawn before trajectories.
    buildings = final.get("buildings", [])
    heights = final.get("building_heights", [])
    for idx, rect in enumerate(buildings):
        if len(rect) < 4:
            continue
        x0, y0, x1, y1 = map(float, rect[:4])
        h = float(heights[idx]) if idx < len(heights) else 0.0
        b0, b1, b2, b3 = project(x0, y0), project(x1, y0), project(x1, y1), project(x0, y1)
        t0, t1, t2, t3 = project(x0, y0, h), project(x1, y0, h), project(x1, y1, h), project(x0, y1, h)
        body.append(polygon([b0, b1, t1, t0], "#aeb8c2", "#66727d", 0.75))
        body.append(polygon([b1, b2, t2, t1], "#8f9ba7", "#66727d", 0.75))
        body.append(polygon([t0, t1, t2, t3], "#cbd2d8", "#66727d", 0.88))

    n_agents = len(final.get("positions", []))
    for agent_idx in range(n_agents):
        points = []
        for frame in frames:
            positions, altitudes = frame.get("positions", []), frame.get("altitudes", [])
            if agent_idx < len(positions) and agent_idx < len(altitudes):
                points.append(project(positions[agent_idx][0], positions[agent_idx][1], altitudes[agent_idx]))
        if not points:
            continue
        color = colors[agent_idx % len(colors)]
        body.append(svg_polyline(points, color, 2.5, 0.92))
        sx, sy = points[0]
        ex, ey = points[-1]
        body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5.5" fill="white" stroke="{color}" stroke-width="2"/>')
        body.append(f'<rect x="{ex-5:.1f}" y="{ey-5:.1f}" width="10" height="10" fill="{color}" stroke="#222"/>')
        body.append(f'<text x="{ex+7:.1f}" y="{ey-7:.1f}" font-size="14" font-family="Times New Roman" font-weight="bold" fill="{color}">U{agent_idx+1}</text>')

    target_points = []
    for frame in frames:
        targets, target_altitudes = frame.get("targets", []), frame.get("target_altitudes", [])
        if targets and target_altitudes:
            target_points.append(project(targets[0][0], targets[0][1], target_altitudes[0]))
    if target_points:
        body.append(svg_polyline(target_points, "#111", 1.8, 0.72, "5 4"))
        tx, ty = target_points[-1]
        target_fill = "#2ca02c" if final.get("capture_success", 0.0) else "#ffbf00"
        body.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="7" fill="{target_fill}" stroke="#111" stroke-width="1.5"/>')
        body.append(f'<text x="{tx+9:.1f}" y="{ty-8:.1f}" font-size="14" font-family="Times New Roman" font-weight="bold">Evader</text>')

    # Capture-task legend and outcome panel.
    body.append(f'<rect x="{panel_x}" y="0" width="{width-panel_x}" height="{height}" fill="#fffaf0" stroke="#ddd"/>')
    lx = panel_x + 18
    body.append(f'<text x="{lx}" y="30" font-size="20" font-family="Times New Roman" font-weight="bold">Legend</text>')
    legend_rows = [
        ('<circle cx="{x}" cy="{y}" r="5.5" fill="white" stroke="#d62728" stroke-width="2"/>', "UAV start"),
        ('<rect x="{xm}" y="{ym}" width="11" height="11" fill="#d62728" stroke="#222"/>', "UAV final"),
        ('<line x1="{xl}" y1="{y}" x2="{xr}" y2="{y}" stroke="#1f77b4" stroke-width="3"/>', "UAV 3-D path"),
        ('<circle cx="{x}" cy="{y}" r="6" fill="#ffbf00" stroke="#111"/>', "Evader final"),
        ('<line x1="{xl}" y1="{y}" x2="{xr}" y2="{y}" stroke="#111" stroke-width="2" stroke-dasharray="5 4"/>', "Evader path"),
        ('<rect x="{xm}" y="{ym}" width="12" height="12" fill="#aeb8c2" stroke="#66727d"/>', "Building prism"),
    ]
    for idx, (symbol, label) in enumerate(legend_rows):
        y = 61 + idx * 31
        body.append(symbol.format(x=lx+8, y=y, xm=lx+2, ym=y-6, xl=lx+1, xr=lx+35))
        body.append(f'<text x="{lx+45}" y="{y+5}" font-size="16" font-family="Times New Roman">{label}</text>')

    captured = bool(final.get("capture_success", 0.0))
    status = "CAPTURED" if captured else "NOT CAPTURED"
    status_color = "#16843b" if captured else "#b32323"
    total_collisions = sum(int(frame.get("collisions", 0)) for frame in frames)
    total_losses = sum(int(frame.get("target_lost", 0)) for frame in frames)
    total_interventions = sum(int(frame.get("continuous_safety_interventions", 0)) for frame in frames)
    visible_steps = sum(bool(frame.get("direct_visibility", [])) and any(any(row) if isinstance(row, list) else bool(row) for row in frame.get("direct_visibility", [])) for frame in frames)
    mean_feasible = sum(float(frame.get("controller_feasible_rate", 0.0)) for frame in frames) / max(len(frames), 1)
    miss = final.get("minimum_capture_gap", float("nan"))
    body.append(f'<text x="{lx}" y="270" font-size="19" font-family="Times New Roman" font-weight="bold">Capture outcome</text>')
    body.append(f'<text x="{lx}" y="300" font-size="22" font-family="Times New Roman" font-weight="bold" fill="{status_color}">{status}</text>')
    metric_lines = [
        f'steps = {final.get("step", 0)}',
        f'capture time = {final.get("step", 0) if captured else "censored"}',
        f'final capture gap = {float(miss):.3f}',
        f'visibility ratio = {visible_steps/max(len(frames), 1):.3f}',
        f'target losses = {total_losses}',
        f'collisions = {total_collisions}',
        f'safety interventions = {total_interventions}',
        f'controller feasible = {mean_feasible:.3f}',
        f'sensor = {final.get("sensor_mode", "lidar")}',
        f'control = {final.get("execution_mode", "velocity_yaw_rate")}',
    ]
    for idx, label in enumerate(metric_lines):
        body.append(f'<text x="{lx}" y="{333 + idx*25}" font-size="15" font-family="Times New Roman">{label}</text>')
    body.append('</svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def write_episode_scene_svg(path: Path, algo: str, trace: dict, grid_size: int | None = None) -> None:
    frames = trace.get("frames", [])
    if not frames:
        return
    if frames[0].get("altitudes") and frames[0].get("target_altitudes"):
        write_episode_scene_3d_svg(path, algo, trace)
        return
    if grid_size is None:
        grid_size = int(frames[0].get("grid_size", 25))
    scale = 22
    plot = grid_size * scale
    legend = 280
    width, height = plot + legend, plot
    colors = ["#d62728", "#2ca02c", "#1f77b4", "#9467bd", "#ff7f0e", "#17becf"]
    n_agents = len(frames[0].get("positions", []))
    final = frames[-1]
    agent_paths = [[] for _ in range(n_agents)]
    for frame in frames:
        for idx, pos in enumerate(frame.get("positions", [])):
            if idx < n_agents:
                agent_paths[idx].append(((float(pos[0]) + 0.5) * scale, (float(pos[1]) + 0.5) * scale))

    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="14" y="24" font-size="21" font-family="Times New Roman">{algo.upper()} episode {trace.get("episode")} final scene</text>')
    body.append(f'<text x="14" y="43" font-size="15" font-family="Times New Roman" fill="#555">UAV paths, final targets, final obstacles, and terminal task metrics.</text>')
    for i in range(grid_size + 1):
        x = i * scale
        body.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{plot}" stroke="#ececec" stroke-width="1"/>')
        body.append(f'<line x1="0" y1="{x}" x2="{plot}" y2="{x}" stroke="#ececec" stroke-width="1"/>')

    # Downsample dynamic obstacle trails lightly so the figure stays readable.
    obstacle_count = len(frames[0].get("obstacles", []))
    for obs_idx in range(obstacle_count):
        pts = []
        for frame in frames[:: max(1, len(frames) // 40)]:
            obstacles = frame.get("obstacles", [])
            if obs_idx < len(obstacles):
                x, y = obstacles[obs_idx]
                pts.append(((float(x) + 0.5) * scale, (float(y) + 0.5) * scale))
        if len(pts) > 1:
            body.append(svg_polyline(pts, "#8a8a8a", 1.0, 0.18, "2 5"))
    for x, y in final.get("obstacles", []):
        body.append(f'<circle cx="{(float(x)+0.5)*scale:.1f}" cy="{(float(y)+0.5)*scale:.1f}" r="7" fill="#8f8f8f" opacity="0.26"/>')

    target_count = len(final.get("targets", []))
    completed = final.get("completed_targets", [0] * target_count)
    found = final.get("found_targets", [0] * target_count)
    tracked = final.get("tracked_targets", [0] * target_count)
    for target_idx in range(target_count):
        pts = []
        for frame in frames[:: max(1, len(frames) // 45)]:
            targets = frame.get("targets", [])
            if target_idx < len(targets):
                x, y = targets[target_idx]
                pts.append(((float(x) + 0.5) * scale, (float(y) + 0.5) * scale))
        if len(pts) > 1:
            body.append(svg_polyline(pts, "#111111", 1.0, 0.26, "3 4"))
    for idx, target in enumerate(final.get("targets", [])):
        x, y = target
        cx, cy = (float(x) + 0.5) * scale, (float(y) + 0.5) * scale
        if idx < len(completed) and completed[idx]:
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#2ca02c" stroke="#111" stroke-width="1.1"/>')
        elif idx < len(tracked) and tracked[idx]:
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#9467bd" stroke="#111" stroke-width="1.1"/>')
        elif idx < len(found) and found[idx]:
            body.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" fill="#ffbf00" stroke="#111" stroke-width="1.1"/>')
        else:
            body.append(f'<path d="M {cx-5:.1f} {cy-5:.1f} L {cx+5:.1f} {cy+5:.1f} M {cx+5:.1f} {cy-5:.1f} L {cx-5:.1f} {cy+5:.1f}" stroke="#111" stroke-width="1.8"/>')

    for idx, path_pts in enumerate(agent_paths):
        color = colors[idx % len(colors)]
        body.append(svg_polyline(path_pts, color, 2.4, 0.92))
        sx, sy = path_pts[0]
        ex, ey = path_pts[-1]
        body.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{color}" stroke="#111" stroke-width="1"/>')
        body.append(f'<text x="{sx+7:.1f}" y="{sy-7:.1f}" font-size="15" font-family="Times New Roman" font-weight="bold" fill="{color}">U{idx+1}</text>')
        body.append(f'<rect x="{ex-5:.1f}" y="{ey-5:.1f}" width="10" height="10" fill="{color}" stroke="#111" stroke-width="1"/>')
        body.append(f'<text x="{ex+7:.1f}" y="{ey+4:.1f}" font-size="15" font-family="Times New Roman" font-weight="bold" fill="{color}">U{idx+1}</text>')

    lx = plot + 18
    body.append(f'<rect x="{plot}" y="0" width="{legend}" height="{height}" fill="#fffaf0" stroke="#ddd"/>')
    body.append(f'<text x="{lx}" y="28" font-size="19" font-family="Times New Roman" font-weight="bold">Legend</text>')
    body.append(f'<circle cx="{lx+8}" cy="56" r="6" fill="#d62728" stroke="#111"/><text x="{lx+24}" y="60" font-size="16" font-family="Times New Roman">UAV start</text>')
    body.append(f'<rect x="{lx+2}" y="78" width="12" height="12" fill="#d62728" stroke="#111"/><text x="{lx+24}" y="89" font-size="16" font-family="Times New Roman">UAV final</text>')
    body.append(f'<line x1="{lx+2}" y1="116" x2="{lx+35}" y2="116" stroke="#1f77b4" stroke-width="3"/><text x="{lx+45}" y="120" font-size="16" font-family="Times New Roman">UAV path</text>')
    body.append(f'<path d="M {lx+3} 140 L {lx+14} 151 M {lx+14} 140 L {lx+3} 151" stroke="#111" stroke-width="2"/><text x="{lx+24}" y="151" font-size="16" font-family="Times New Roman">Undiscovered target</text>')
    body.append(f'<circle cx="{lx+8}" cy="178" r="6" fill="#ffbf00" stroke="#111"/><text x="{lx+24}" y="182" font-size="16" font-family="Times New Roman">Found target</text>')
    body.append(f'<circle cx="{lx+8}" cy="206" r="6" fill="#9467bd" stroke="#111"/><text x="{lx+24}" y="210" font-size="16" font-family="Times New Roman">Tracked target</text>')
    body.append(f'<circle cx="{lx+8}" cy="234" r="6" fill="#2ca02c" stroke="#111"/><text x="{lx+24}" y="238" font-size="16" font-family="Times New Roman">Completed target</text>')
    body.append(f'<circle cx="{lx+8}" cy="262" r="7" fill="#8f8f8f" opacity="0.26"/><text x="{lx+24}" y="266" font-size="16" font-family="Times New Roman">Obstacle final</text>')
    body.append(f'<line x1="{lx+2}" y1="292" x2="{lx+35}" y2="292" stroke="#111" stroke-width="1" stroke-dasharray="3 4" opacity="0.45"/><text x="{lx+45}" y="296" font-size="16" font-family="Times New Roman">Target trail</text>')
    body.append(f'<line x1="{lx+2}" y1="320" x2="{lx+35}" y2="320" stroke="#8a8a8a" stroke-width="1" stroke-dasharray="2 5" opacity="0.35"/><text x="{lx+45}" y="324" font-size="16" font-family="Times New Roman">Obstacle trail</text>')
    body.append(f'<text x="{lx}" y="360" font-size="17" font-family="Times New Roman" font-weight="bold">Final metrics</text>')
    body.append(f'<text x="{lx}" y="382" font-size="15" font-family="Times New Roman">steps={final.get("step", 0)}</text>')
    body.append(f'<text x="{lx}" y="400" font-size="15" font-family="Times New Roman">coverage={final.get("coverage", 0):.3f}</text>')
    body.append(f'<text x="{lx}" y="418" font-size="15" font-family="Times New Roman">collisions={sum(f.get("collisions", 0) for f in frames)}</text>')
    body.append(f'<text x="{lx}" y="436" font-size="15" font-family="Times New Roman">lost_tracks={sum(f.get("lost_tracks", 0) for f in frames)}</text>')
    body.append(f'<text x="{lx}" y="454" font-size="15" font-family="Times New Roman">graph_density={final.get("graph_density", 0):.3f}</text>')
    body.append("</svg>")
    path.write_text("\n".join(body), encoding="utf-8")


def write_training_scene_svgs(out_dir: Path, algo: str, traces: list[dict]) -> None:
    if not traces:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, trace in enumerate(traces, start=1):
        path = out_dir / f"{algo}_last_scene_{idx}_episode_{trace.get('episode')}.svg"
        write_episode_scene_svg(path, algo, trace)
        if trace.get("frames") and trace["frames"][0].get("buildings") is not None:
            write_pursuit_animation_html(
                path.with_suffix(".html"),
                trace,
                f"{algo.upper()} episode {trace.get('episode')} pursuit simulation",
            )
        manifest.append({"episode": trace.get("episode"), "file": str(path.name), "steps": len(trace.get("frames", []))})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def make_env(seed: int, env_overrides: dict, task_mode: str = "search_multi_target"):
    def factory(episode: int = 0):
        # Train on a distribution of layouts and stochastic trajectories rather
        # than replaying the exact same seeded scenario every episode.
        mode_classes = {
            "search_multi_target": (WeakCommConfig, WeakCommBeliefGraphEnv),
        }
        config_cls, env_cls = mode_classes[task_mode]
        cfg = config_cls(seed=seed * 100_000 + int(episode))
        # Keep centralized-critic inputs invariant across randomized layouts.
        if hasattr(cfg, "building_state_capacity"):
            cfg.building_state_capacity = max(
                int(cfg.building_state_capacity), int(cfg.building_count)
            )
        for key, value in env_overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        if hasattr(cfg, "building_state_capacity"):
            cfg.building_state_capacity = max(
                int(cfg.building_state_capacity), int(cfg.building_count)
            )
        return env_cls(cfg)

    return factory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["mappo", "ippo", "qmix"], default="mappo")
    parser.add_argument(
        "--task-mode",
        choices=[
            "search_multi_target",
        ],
        default="search_multi_target",
    )
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume", default=None, help="Resume a MAPPO or QMIX run from a .pt checkpoint.")
    parser.add_argument("--target-kl", type=float, default=0.03, help="Stop a PPO epoch early after excessive policy drift; <=0 disables it.")
    parser.add_argument("--no-gat", action="store_true", help="Use MLP policies instead of the trainable GAT encoder.")
    parser.add_argument("--no-hetero-entities", action="store_true", help="Keep UAV-only GAT but disable target/obstacle entity nodes.")
    parser.add_argument("--search-weights", action="store_true", help="Legacy ablation: output adaptive J1..J5 weights instead of directly sampling 9 discrete actions.")
    assignment_group = parser.add_mutually_exclusive_group()
    assignment_group.add_argument(
        "--assignment-head",
        dest="assignment_head",
        action="store_true",
        help="Optional legacy/multi-target head that jointly learns target/search assignment and motion.",
    )
    assignment_group.add_argument(
        "--no-assignment-head",
        dest="assignment_head",
        action="store_false",
        help="Disable target/search assignment (default; required by the single-target 3-v-1 paper model).",
    )
    parser.set_defaults(assignment_head=None)
    parser.add_argument("--direct-actions", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--gat-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-end-factor", type=float, default=0.1)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.002)
    parser.add_argument("--disable-adaptive-entropy", action="store_true")
    parser.add_argument("--uncertainty-entropy-gain", type=float, default=1.0)
    parser.add_argument(
        "--continuous-log-std-max",
        type=float,
        default=-0.5,
        help="Upper bound on continuous-policy log standard deviation; -0.5 gives std <= 0.607.",
    )
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--update-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32, help="PPO minibatch size in environment timesteps.")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=["team_potential", "legacy_shaped"], default=None)
    parser.add_argument("--discovery-potential-weight", type=float, default=None)
    parser.add_argument("--discovery-tail-exponent", type=float, default=None)
    parser.add_argument("--exploration-potential-weight", type=float, default=None)
    parser.add_argument("--safety-cost-weight", type=float, default=None)
    parser.add_argument("--normalize-returns", action="store_true")
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--n-uavs", type=int, default=None)
    parser.add_argument("--n-targets", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--search-steps", type=int, default=None)
    parser.add_argument("--completion-steps", type=int, default=None)
    parser.add_argument("--tracking-grace-steps", type=int, default=None)
    parser.add_argument("--tracking-streak-decay", type=int, default=None)
    parser.add_argument("--target-prediction-weight", type=float, default=None)
    parser.add_argument("--discovery-reward", type=float, default=None)
    parser.add_argument("--tracking-streak-reward", type=float, default=None)
    parser.add_argument("--tracking-streak-loss-penalty", type=float, default=None)
    parser.add_argument("--persistent-tracking-reward", type=float, default=None)
    parser.add_argument("--tracking-stage-threshold", type=float, default=None)
    parser.add_argument("--completion-reward", type=float, default=None)
    parser.add_argument("--target-lost-penalty", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--obstacle-speed", type=float, default=None)
    parser.add_argument("--uav-speed", type=float, default=None)
    parser.add_argument("--decision-dt", type=float, default=None)
    parser.add_argument("--max-turn-steps", type=int, default=None, help="Maximum heading change per step in 45-degree units.")
    parser.add_argument("--allow-hover", action="store_true", help="Allow UAVs to choose the hover action even when movement is feasible.")
    parser.add_argument("--no-normalize-diagonal-speed", action="store_true", help="Keep diagonal grid moves longer than straight moves.")
    parser.add_argument("--comm-radius", type=float, default=None)
    parser.add_argument("--comm-mode", choices=["full", "outage", "none"], default=None)
    parser.add_argument("--avoidance-mode", choices=["none", "dwa", "dwa_orca"], default=None)
    parser.add_argument("--outage-base-prob", type=float, default=None)
    parser.add_argument("--outage-jammed-prob", type=float, default=None)
    parser.add_argument("--outage-recovery-prob", type=float, default=None)
    parser.add_argument("--failure-base-prob", type=float, default=None)
    parser.add_argument("--failure-jammed-prob", type=float, default=None)
    parser.add_argument("--failure-recovery-prob", type=float, default=None)
    parser.add_argument("--jammer-effect-radius", type=float, default=None)
    parser.add_argument("--tpm-prior-base", type=float, default=None)
    parser.add_argument("--enable-target-prior", action="store_true", help="Add Gaussian TPM prior bumps around initial target locations.")
    parser.add_argument("--tpm-target-prior-strength", type=float, default=None)
    parser.add_argument("--tpm-target-prior-sigma", type=float, default=None)
    parser.add_argument("--disable-dynamic-dt", action="store_true", help="Use the fixed decision_dt from the config.")
    parser.add_argument("--min-decision-dt", type=float, default=None)
    parser.add_argument("--max-decision-dt", type=float, default=None)
    parser.add_argument("--avoidance-prediction-horizon", type=int, default=None)
    parser.add_argument("--obstacle-prediction-buffer", type=float, default=None)
    parser.add_argument("--pair-prediction-buffer", type=float, default=None)
    parser.add_argument("--hetero-target-nodes", type=int, default=None)
    parser.add_argument("--hetero-frontier-nodes", type=int, default=None)
    parser.add_argument("--hetero-obstacle-nodes", type=int, default=None)
    parser.add_argument("--entity-observation-radius", type=float, default=None)
    parser.add_argument("--revisit-aging-rate", type=float, default=None)
    parser.add_argument("--revisit-diffusion", type=float, default=None)
    parser.add_argument("--revisit-gain-reward", type=float, default=None)
    parser.add_argument("--disable-revisit-map", action="store_true", help="Ablation: remove the third-layer revisit map and its policy objective.")
    parser.add_argument("--disable-revisit-reward", action="store_true", help="Ablation: retain the revisit map but remove its direct reward.")
    parser.add_argument("--disable-freshness-fusion", action="store_true", help="Ablation: ignore belief age and link confidence during map fusion.")
    parser.add_argument("--belief-fusion-mode", choices=["soft", "hard"], default=None)
    parser.add_argument("--belief-fusion-temperature", type=float, default=None)
    parser.add_argument("--belief-freshness-tau", type=float, default=None)
    parser.add_argument("--static-targets", action="store_true")
    parser.add_argument("--static-obstacles", action="store_true")
    parser.add_argument("--evader-policy", choices=["random", "repulsive", "occlusion"], default=None)
    parser.add_argument("--field-of-view-deg", type=float, default=None)
    parser.add_argument("--building-count", type=int, default=None)
    parser.add_argument("--capture-radius", type=float, default=None)
    parser.add_argument("--capture-required-uavs", type=int, default=None)
    parser.add_argument("--capture-hold-steps", type=int, default=None)
    parser.add_argument("--capture-angular-span-deg", type=float, default=None)
    parser.add_argument("--max-tracks-per-message", type=int, default=None)
    parser.add_argument("--communication-cost-per-kb", type=float, default=None)
    parser.add_argument("--track-freshness-tau", type=float, default=None)
    parser.add_argument(
        "--disable-track-freshness-fusion",
        action="store_true",
        help="Ablation: use ordinary covariance intersection for track messages.",
    )
    parser.add_argument("--max-horizontal-velocity", type=float, default=None)
    parser.add_argument("--max-vertical-velocity", type=float, default=None)
    args = parser.parse_args()

    config_classes = {
        "search_multi_target": WeakCommConfig,
    }
    base_env_cfg = config_classes[args.task_mode]()
    assignment_head_enabled = False if args.assignment_head is None else args.assignment_head
    if args.algo == "qmix" and assignment_head_enabled:
        parser.error("The assignment head is a MAPPO/IPPO policy head; use --no-assignment-head with QMIX.")
    cfg = TrainConfig(
        episodes=args.episodes,
        max_steps=0,
        use_gat=not args.no_gat,
        use_hetero_entities=not args.no_gat and not args.no_hetero_entities,
        use_search_weights=args.search_weights and not args.direct_actions,
        use_assignment_head=assignment_head_enabled,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        lr_end_factor=args.lr_end_factor,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        adaptive_entropy=not args.disable_adaptive_entropy,
        uncertainty_entropy_gain=args.uncertainty_entropy_gain,
        continuous_log_std_max=args.continuous_log_std_max,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        update_epochs=args.update_epochs,
        batch_size=args.batch_size,
        reward_scale=args.reward_scale,
        normalize_returns=args.normalize_returns,
        normalize_advantages=not args.no_normalize_advantages,
        device=args.device,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        target_kl=args.target_kl,
    )
    env_overrides = {
        "n_uavs": args.n_uavs,
        "n_targets": args.n_targets,
        "n_obstacles": args.n_obstacles,
        "grid_size": args.grid_size,
        "search_steps": args.search_steps,
        "target_completion_steps": args.completion_steps,
        "reward_mode": args.reward_mode,
        "discovery_potential_weight": args.discovery_potential_weight,
        "discovery_tail_exponent": args.discovery_tail_exponent,
        "exploration_potential_weight": args.exploration_potential_weight,
        "safety_cost_weight": args.safety_cost_weight,
        "tracking_grace_steps": args.tracking_grace_steps,
        "tracking_streak_decay": args.tracking_streak_decay,
        "target_prediction_weight": args.target_prediction_weight,
        "discovery_reward": args.discovery_reward,
        "tracking_streak_reward": args.tracking_streak_reward,
        "tracking_streak_loss_penalty": args.tracking_streak_loss_penalty,
        "persistent_tracking_reward": args.persistent_tracking_reward,
        "tracking_stage_threshold": args.tracking_stage_threshold,
        "completion_reward": args.completion_reward,
        "target_lost_penalty": args.target_lost_penalty,
        "target_speed": args.target_speed,
        "obstacle_speed": args.obstacle_speed,
        "uav_speed": args.uav_speed,
        "decision_dt": args.decision_dt,
        "max_turn_steps": args.max_turn_steps,
        "allow_hover": True if args.allow_hover else None,
        "normalize_diagonal_speed": False if args.no_normalize_diagonal_speed else None,
        "comm_radius": args.comm_radius,
        "comm_mode": args.comm_mode,
        "avoidance_mode": args.avoidance_mode,
        "outage_base_prob": args.outage_base_prob,
        "outage_jammed_prob": args.outage_jammed_prob,
        "outage_recovery_prob": args.outage_recovery_prob,
        "failure_base_prob": args.failure_base_prob,
        "failure_jammed_prob": args.failure_jammed_prob,
        "failure_recovery_prob": args.failure_recovery_prob,
        "jammer_effect_radius": args.jammer_effect_radius,
        "tpm_prior_base": args.tpm_prior_base,
        "tpm_target_prior_enabled": True if args.enable_target_prior else None,
        "tpm_target_prior_strength": args.tpm_target_prior_strength,
        "tpm_target_prior_sigma": args.tpm_target_prior_sigma,
        "dynamic_dt_enabled": False if args.disable_dynamic_dt else None,
        "min_decision_dt": args.min_decision_dt,
        "max_decision_dt": args.max_decision_dt,
        "avoidance_prediction_horizon": args.avoidance_prediction_horizon,
        "obstacle_prediction_buffer": args.obstacle_prediction_buffer,
        "pair_prediction_buffer": args.pair_prediction_buffer,
        "hetero_target_nodes": args.hetero_target_nodes,
        "hetero_frontier_nodes": args.hetero_frontier_nodes,
        "hetero_obstacle_nodes": args.hetero_obstacle_nodes,
        "entity_observation_radius": args.entity_observation_radius,
        "revisit_aging_rate": args.revisit_aging_rate,
        "revisit_diffusion": args.revisit_diffusion,
        "revisit_gain_reward": 0.0 if args.disable_revisit_reward else args.revisit_gain_reward,
        "revisit_map_enabled": False if args.disable_revisit_map else None,
        "freshness_fusion_enabled": False if args.disable_freshness_fusion else None,
        "belief_fusion_mode": args.belief_fusion_mode,
        "belief_fusion_temperature": args.belief_fusion_temperature,
        "belief_freshness_tau": args.belief_freshness_tau,
        "dynamic_targets_enabled": False if args.static_targets else None,
        "dynamic_obstacles_enabled": False if args.static_obstacles else None,
        "evader_policy": args.evader_policy,
        "field_of_view_deg": args.field_of_view_deg,
        "building_count": args.building_count,
        "capture_radius": args.capture_radius,
        "capture_required_uavs": args.capture_required_uavs,
        "capture_hold_steps": args.capture_hold_steps,
        "capture_angular_span_deg": args.capture_angular_span_deg,
        "max_tracks_per_message": args.max_tracks_per_message,
        "communication_cost_per_kb": args.communication_cost_per_kb,
        "track_freshness_tau": args.track_freshness_tau,
        "track_freshness_fusion_enabled": (
            False if args.disable_track_freshness_fusion else None
        ),
        "max_horizontal_velocity": args.max_horizontal_velocity,
        "max_vertical_velocity": args.max_vertical_velocity,
    }
    config_cls = config_classes[args.task_mode]
    effective_env_cfg = config_cls(seed=args.seed)
    for key, value in env_overrides.items():
        if value is not None:
            setattr(effective_env_cfg, key, value)
    print(
        json.dumps(
            {
                "effective_env_config": {
                    "n_uavs": effective_env_cfg.n_uavs,
                    "n_targets": effective_env_cfg.n_targets,
                    "grid_size": effective_env_cfg.grid_size,
                    "search_steps": effective_env_cfg.search_steps,
                    "n_obstacles": effective_env_cfg.n_obstacles,
                    "allow_hover": effective_env_cfg.allow_hover,
                    "avoidance_prediction_horizon": effective_env_cfg.avoidance_prediction_horizon,
                    "obstacle_prediction_buffer": effective_env_cfg.obstacle_prediction_buffer,
                    "pair_prediction_buffer": effective_env_cfg.pair_prediction_buffer,
                    "tracking_grace_steps": effective_env_cfg.tracking_grace_steps,
                    "tracking_streak_decay": effective_env_cfg.tracking_streak_decay,
                    "target_prediction_weight": effective_env_cfg.target_prediction_weight,
                    "discovery_reward": effective_env_cfg.discovery_reward,
                    "tracking_streak_reward": effective_env_cfg.tracking_streak_reward,
                    "completion_reward": effective_env_cfg.completion_reward,
                    "target_lost_penalty": effective_env_cfg.target_lost_penalty,
                    "task_mode": args.task_mode,
                    "evader_policy": getattr(effective_env_cfg, "evader_policy", None),
                    "field_of_view_deg": getattr(effective_env_cfg, "field_of_view_deg", None),
                    "pursuit_target_observable": getattr(
                        effective_env_cfg, "pursuit_target_observable", None
                    ),
                    "assignment_head": assignment_head_enabled,
                    "track_freshness_fusion_enabled": getattr(
                        effective_env_cfg, "track_freshness_fusion_enabled", None
                    ),
                    "pursuit_reward_mode": getattr(
                        effective_env_cfg, "pursuit_reward_mode", None
                    ),
                }
            },
            indent=2,
        )
    )
    trainer_cls = {
        "mappo": MAPPOTrainer,
        "ippo": IPPOTrainer,
        "qmix": QMIXTrainer,
    }[args.algo]
    seed_everything(args.seed)
    if args.out is None:
        out = Path(__file__).resolve().parents[1] / "outputs" / f"{args.algo}_training.json"
    else:
        out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_path = str(out.with_name(f"{out.stem}_latest.pt"))
    try:
        trainer = trainer_cls(
            make_env(args.seed, env_overrides, args.task_mode),
            cfg,
        )
    except RuntimeError as exc:
        print(str(exc))
        print("From workspace root: python .\\repro\\train_cooperative_marl.py --algo mappo --episodes 30")
        print("From repro directory: python .\\train_cooperative_marl.py --algo mappo --episodes 30")
        raise SystemExit(2) from exc
    if args.resume:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at episode {trainer.start_episode}.", flush=True)
    history = trainer.train()
    out.write_text(json.dumps({"algo": args.algo, "history": history}, indent=2), encoding="utf-8")
    trainer.save_checkpoint(out.with_suffix(".pt"), env_config=vars(effective_env_cfg), episode=args.episodes, history=history)
    write_training_csv(out.with_suffix(".csv"), history)
    write_training_svg(out.with_suffix(".svg"), args.algo, history)
    write_training_scene_svgs(out.with_name(f"{out.stem}_last3_scenes"), args.algo, getattr(trainer, "episode_traces", []))
    print(json.dumps(tail_average(history, 50), indent=2))


if __name__ == "__main__":
    main()
