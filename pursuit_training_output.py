"""Capture-focused reports; no legacy search plots or metrics."""
import csv
import html
import json
import math
from pathlib import Path

import numpy as np


FIELDS = (
    "reward_components", "closest_capture_gap", "disabled_uavs_final",
    "episode", "capture_success", "capture_time", "steps", "reward",
    "minimum_capture_gap", "collisions", "uav_collisions", "obstacle_conflicts",
    "pair_conflicts", "disabled_rate", "collision_exposure_rate", "controller_feasible_rate",
    "safety_intervention_rate", "path_length", "control_smoothness_cost",
    "first_onboard_contact_time", "target_loss_count", "target_reacquisition_count",
    "continuous_visibility_ratio", "track_position_rmse", "track_velocity_rmse",
    "valid_track_rate", "track_communication_bytes", "policy_loss", "value_loss",
    "entropy", "learning_rate", "actor_grad_norm", "critic_grad_norm", "entropy_coef",
    "approx_kl", "ppo_early_stop", "clip_fraction", "explained_variance", "demo_loss_weight",
    "environment_seconds", "ppo_update_seconds", "lidar_detection_ratio", "target_observation_ratio",
)


def clean_history_row(row):
    result = {k: row[k] for k in FIELDS if k in row}
    if "controller_feasible_rate" not in result and "mpc_feasible_rate" in row:
        result["controller_feasible_rate"] = row["mpc_feasible_rate"]
    if not result.get("capture_success"):
        result["capture_time"] = None
    for key, value in result.items():
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
    return result


def rolling(values, window=50):
    result = []
    for i in range(len(values)):
        group = [v for v in values[max(0, i-window+1):i+1] if v is not None and math.isfinite(v)]
        result.append(float(np.mean(group)) if group else None)
    return result


def write_learning_chart(path, history):
    panels = [
        ("Capture rate (rolling 50 episodes)", "capture_success", (0, 1)),
        ("Capture steps (successes only, rolling 50)", "capture_time", None),
        ("Episode length (includes timeouts, rolling 50)", "steps", None),
        ("Final capture gap (rolling 50)", "minimum_capture_gap", None),
        ("Collisions per episode (rolling 50)", "collisions", None),
        ("Controller feasible fraction (rolling 50)", "controller_feasible_rate", (0, 1)),
    ]
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="860" viewBox="0 0 1100 860">',
            '<rect width="1100" height="860" fill="#f4f7fb"/>',
            '<g font-family="Arial, sans-serif" fill="#183249">',
            '<text x="30" y="34" font-size="22">Cooperative capture | MAPPO + demonstrations</text>',
            '<text x="30" y="58" font-size="13">Stochastic training statistics. Compare policies using held-out deterministic evaluation.</text>']
    for index, (title, key, bounds) in enumerate(panels):
        x, y = 65 + (index % 2)*540, 115 + (index // 2)*250
        width, height = 450, 160
        values = rolling([r.get(key) for r in history])
        valid = [v for v in values if v is not None]
        low, high = bounds or ((min(0, min(valid)), max(valid)) if valid else (0, 1))
        if high <= low:
            high = low + 1
        body.append(f'<rect x="{x-40}" y="{y-35}" width="520" height="235" rx="12" fill="white"/>')
        body.append(f'<text x="{x-20}" y="{y-12}" font-size="14">{html.escape(title)}</text>')
        for fraction in (0, 0.5, 1):
            py = y + height*(1-fraction)
            body.append(f'<path d="M{x},{py}h{width}" stroke="#e1e7ef"/>')
            body.append(f'<text x="{x-5}" y="{py+4}" text-anchor="end" font-size="11">{low+(high-low)*fraction:.2g}</text>')
        # Break at missing success windows; never draw failed capture times as zero.
        segment = []
        for i, value in enumerate(values + [None]):
            if value is None:
                if segment:
                    body.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="#157f89" stroke-width="2.5"/>')
                    segment = []
            else:
                px = x + width*i/max(len(values)-1, 1)
                py = y + height*(1-(value-low)/(high-low))
                segment.append(f"{px:.1f},{py:.1f}")
                body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="1.4" fill="#157f89"/>')
        if not valid:
            body.append(f'<text x="{x+100}" y="{y+85}" font-size="14" fill="#667788">No successful captures / no data</text>')
        body.append(f'<text x="{x}" y="{y+height+22}" font-size="11">Episode {history[0]["episode"] if history else 0}</text>')
        body.append(f'<text x="{x+width}" y="{y+height+22}" text-anchor="end" font-size="11">{history[-1]["episode"] if history else 0}</text>')
    body.append('</g></svg>')
    Path(path).write_text("\n".join(body), encoding="utf-8")


def write_reward_capture_chart(path, history, window=50):
    """Dual-axis training plot; capture success is averaged, never accumulated."""
    history = sorted(history, key=lambda row: row["episode"])
    x0, y0, width, height = 100, 100, 800, 330
    rewards = [row.get("reward") for row in history]
    reward_mean = rolling(rewards, window)
    capture_mean = rolling([row.get("capture_success") for row in history], window)
    valid = [v for v in rewards if v is not None and math.isfinite(v)]
    lo, hi = (min(valid), max(valid)) if valid else (0, 1)
    padding = max((hi-lo)*.1, 1.)
    lo, hi = lo-padding, hi+padding
    first, last = (history[0]["episode"], history[-1]["episode"]) if history else (0, 1)
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="550" viewBox="0 0 1000 550">',
            '<rect width="1000" height="550" fill="white"/>', '<g font-family="Arial, sans-serif" fill="#183249">',
            '<text x="100" y="35" font-size="23">Cooperative capture: reward and capture rate</text>',
            f'<text x="100" y="60" font-size="13">Training statistics | trailing {window} episodes (available episodes at start) | separate y-axes</text>',
            '<text x="100" y="85" fill="#2563eb">Episode return (left)</text>',
            '<text x="900" y="85" text-anchor="end" fill="#d97706">Capture rate (right)</text>']
    for f in (0, .25, .5, .75, 1):
        y = y0+height*(1-f)
        body += [f'<path d="M{x0},{y}h{width}" stroke="#e2e8f0"/>',
                 f'<text x="90" y="{y+4}" text-anchor="end" fill="#2563eb">{lo+(hi-lo)*f:.1f}</text>',
                 f'<text x="910" y="{y+4}" fill="#d97706">{100*f:.0f}%</text>']
        x = x0+width*f
        body.append(f'<text x="{x}" y="455" text-anchor="middle">{first+(last-first)*f:.0f}</text>')
    def curve(values, low, high, color, stroke, opacity=1):
        points = []
        def flush():
            if points:
                body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="{stroke}" opacity="{opacity}"/>')
                if len(points) == 1:
                    px, py = points[0].split(',')
                    body.append(f'<circle cx="{px}" cy="{py}" r="3" fill="{color}"/>')
                points.clear()
        for row, value in zip(history, values):
            if value is None or not math.isfinite(value):
                flush()
                continue
            x = x0+width*(row["episode"]-first)/max(last-first, 1)
            y = y0+height*(1-(value-low)/(high-low))
            points.append(f'{x:.2f},{y:.2f}')
        flush()
    curve(rewards, lo, hi, '#2563eb', 1, .2)
    curve(reward_mean, lo, hi, '#2563eb', 2.5)
    curve(capture_mean, 0, 1, '#d97706', 2.5)
    body += ['<text x="500" y="480" text-anchor="middle">Training episode</text>',
             '<text x="100" y="515" fill="#2563eb">Blue: mean return; faint blue: raw episode return</text>',
             '<text x="600" y="515" fill="#d97706">Orange: successful episodes / window</text>', '</g></svg>']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def write_evaluation_chart(path, result):
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="880" height="420">',
            '<rect width="880" height="420" fill="#f4f7fb"/>',
            '<g font-family="Arial, sans-serif" fill="#183249">',
            '<text x="30" y="35" font-size="21">Held-out capture rate | identical environment and seeds</text>',
            '<text x="30" y="60" font-size="13">Bars: capture rate. Lines: 95% Wilson intervals. Time is measured in decision steps.</text>']
    for i, (method, metrics) in enumerate(result["summary"].items()):
        n, p = metrics["episodes"], metrics["capture_rate"]
        z = 1.96
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        y = 110+i*65
        body.extend([
            f'<text x="30" y="{y+20}" font-size="15">{html.escape(method.upper())}</text>',
            f'<rect x="130" y="{y}" width="{p*450:.1f}" height="30" fill="#157f89"/>',
            f'<path d="M{130+(center-half)*450:.1f},{y+15}H{130+(center+half)*450:.1f}" stroke="#17324a" stroke-width="3"/>',
            f'<text x="600" y="{y+20}" font-size="14">{p:.1%} ({round(p*n)}/{n}); mean steps {metrics["mean_censored_steps"]:.1f}</text>',
        ])
    body.append('</g></svg>')
    Path(path).write_text("\n".join(body), encoding="utf-8")


def write_pursuit_outputs(folder, history, traces):
    from train_cooperative_marl import write_episode_scene_3d_svg
    from pursuit_visualization import write_pursuit_animation_html

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    history = [clean_history_row(row) for row in history]
    (folder / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
    with (folder / "training.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(history)
    tail = history[-50:]
    success_times = [r["capture_time"] for r in tail if r.get("capture_time") is not None]
    summary = {"episodes_averaged": len(tail), "capture_rate": float(np.mean([r["capture_success"] for r in tail])) if tail else None,
               "mean_success_capture_steps": float(np.mean(success_times)) if success_times else None,
               "mean_censored_steps": float(np.mean([r["steps"] for r in tail])) if tail else None,
               "note": "Training samples, not held-out policy evaluation"}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_learning_chart(folder / "capture_learning.svg", history)
    write_reward_capture_chart(folder / "reward_capture.svg", history)
    links = ['<h1>协同围捕训练</h1><p>训练统计与独立评估分开；失败回合不计入成功捕获耗时。</p>',
             '<img src="reward_capture.svg" style="width:100%;max-width:1000px" alt="奖励和捕获率双轴曲线">',
             '<img src="capture_learning.svg" style="width:100%;max-width:1100px" alt="围捕学习曲线">']
    for trace in traces:
        trace = dict(trace, mode="3d")
        trace["frames"] = [{k: v for k, v in frame.items() if k not in
                            ("coverage", "new_cells", "revisit_gain", "stagnant_agents",
                             "found_targets", "tracked_targets", "completed_targets")}
                           for frame in trace["frames"]]
        stem = f"capture_episode_{trace['episode']}"
        write_episode_scene_3d_svg(folder / f"{stem}.svg", "MAPPO + demos", trace)
        write_pursuit_animation_html(folder / f"{stem}.html", trace, f"Cooperative capture | episode {trace['episode']}")
        links.append(f'<p><a href="{stem}.html">第 {trace["episode"]} 回合三维围捕回放</a></p><img src="{stem}.svg" style="max-width:940px;width:100%">')
    (folder / "index.html").write_text('<!doctype html><meta charset="utf-8"><title>协同围捕</title><body style="font-family:Arial;background:#f4f7fb;margin:24px">'+"\n".join(links)+'</body>', encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot reward and capture rate from saved history; no training")
    parser.add_argument("--history", type=Path, required=True, help="history.json or a trusted local training checkpoint .pt")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.history.suffix == ".pt":
        import torch
        payload = torch.load(args.history, map_location="cpu", weights_only=False)
    else:
        payload = json.loads(args.history.read_text(encoding="utf-8"))
    history = payload.get("history", []) if isinstance(payload, dict) else payload
    if not history:
        parser.error("No training history yet; use a later checkpoint or completed history.json")
    write_reward_capture_chart(args.out, history)
    print(args.out)
