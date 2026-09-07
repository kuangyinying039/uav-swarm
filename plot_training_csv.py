"""Redraw a saved MARL training CSV without rerunning training."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from train_cooperative_marl import metric_points, moving_average, write_training_svg


def write_reward_svg(
    path: Path,
    history: list[dict],
    *,
    title: str,
    smooth_window: int | None,
    show_raw: bool,
    y_min: float | None,
    y_max: float | None,
) -> None:
    width, height = 980, 520
    left, top, right, bottom = 90, 65, 45, 70
    plot_width, plot_height = width - left - right, height - top - bottom
    font = "Times New Roman"
    max_ep = max(int(row["episode"]) for row in history)
    raw = [float(row["reward"]) for row in history]
    window = smooth_window or max(5, len(raw) // 25)
    trend = moving_average(raw, window)
    data_min, data_max = min(trend), max(trend)
    padding = max(5.0, 0.10 * max(data_max - data_min, 1.0))
    min_v = data_min - padding if y_min is None else float(y_min)
    max_v = data_max + padding if y_max is None else float(y_max)
    if max_v <= min_v:
        raise ValueError("y_max must be greater than y_min")

    early_n = min(500, max(1, len(raw) // 10))
    late_n = early_n
    early_mean = sum(raw[:early_n]) / early_n
    late_mean = sum(raw[-late_n:]) / late_n
    delta = late_mean - early_mean
    delta_pct = 100.0 * delta / abs(early_mean) if abs(early_mean) > 1e-9 else 0.0

    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fbfbf8"/>')
    body.append(f'<text x="{width/2:.1f}" y="29" font-size="30" font-family="{font}" font-weight="bold" text-anchor="middle">{title} Reward Curve</text>')
    body.append(f'<text x="{width/2:.1f}" y="51" font-size="19" font-family="{font}" fill="#555" text-anchor="middle">Moving-average training reward (window={window})</text>')
    body.append(f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>')
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>')
    for tick in range(6):
        y = top + tick * plot_height / 5
        value = max_v - tick * (max_v - min_v) / 5
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e0d8"/>')
        body.append(f'<text x="{left-12}" y="{y+5:.1f}" font-size="18" font-family="{font}" text-anchor="end">{value:.0f}</text>')
    for tick in range(6):
        x = left + tick * plot_width / 5
        episode = tick * max_ep / 5
        body.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
        body.append(f'<text x="{x:.1f}" y="{height-bottom+25}" font-size="18" font-family="{font}" text-anchor="middle">{episode:.0f}</text>')
    if show_raw:
        raw_pts = metric_points(history, "reward", max_ep, left, top, plot_width, plot_height, min_v, max_v, values=raw)
        body.append(f'<polyline points="{raw_pts}" fill="none" stroke="#1f77b4" stroke-width="1.0" opacity="0.12"/>')
    trend_pts = metric_points(history, "reward", max_ep, left, top, plot_width, plot_height, min_v, max_v, values=trend)
    body.append(f'<polyline points="{trend_pts}" fill="none" stroke="#b22222" stroke-width="3.6" stroke-linejoin="round" stroke-linecap="round"/>')
    panel_x, panel_y = width - right - 275, top + 18
    body.append(f'<rect x="{panel_x}" y="{panel_y}" width="255" height="92" rx="8" fill="#fbfbf8" opacity="0.92" stroke="#ddd"/>')
    body.append(f'<text x="{panel_x+15}" y="{panel_y+25}" font-size="19" font-family="{font}">First {early_n} mean: {early_mean:.2f}</text>')
    body.append(f'<text x="{panel_x+15}" y="{panel_y+49}" font-size="19" font-family="{font}">Last {late_n} mean: {late_mean:.2f}</text>')
    body.append(f'<text x="{panel_x+15}" y="{panel_y+73}" font-size="19" font-family="{font}" font-weight="bold">Change: {delta:+.2f} ({delta_pct:+.1f}%)</text>')
    body.append(f'<text x="{left + plot_width/2:.1f}" y="{height-16}" font-size="19" font-family="{font}" text-anchor="middle">Episode</text>')
    body.append(f'<text x="23" y="{top + plot_height/2:.1f}" font-size="19" font-family="{font}" text-anchor="middle" transform="rotate(-90 23 {top + plot_height/2:.1f})">Episode reward</text>')
    body.append('</svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def read_history(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    required = {
        "episode",
        "coverage",
        "target_discovery_rate",
        "target_tracking_rate",
        "target_completion_rate",
    }
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")
    for row in rows:
        row["episode"] = int(float(row["episode"]))
        for key, value in list(row.items()):
            if key != "episode" and (value is None or value.strip() == ""):
                row[key] = "0"
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Training CSV produced by train_cooperative_marl.py")
    parser.add_argument("--out", default=None, help="Output SVG; defaults to <csv_stem>_trend.svg")
    parser.add_argument("--title", default="MAPPO", help="Title prefix")
    parser.add_argument("--window", type=int, default=None, help="Moving-average window; default is episodes/25")
    parser.add_argument("--y-min", type=float, default=None, help="Optional fixed lower y-axis limit")
    parser.add_argument("--y-max", type=float, default=None, help="Optional fixed upper y-axis limit")
    parser.add_argument("--show-raw", action="store_true", help="Also draw faint raw episode lines")
    parser.add_argument("--plot", choices=["metrics", "reward", "both"], default="metrics", help="Which curve(s) to generate")
    parser.add_argument(
        "--metrics",
        choices=["all", "coverage-discovery"],
        default="all",
        help="Metrics shown in the training-curve panel.",
    )
    parser.add_argument(
        "--figure-title",
        default=None,
        help="Exact title for the metrics figure.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_name(f"{csv_path.stem}_trend.svg")
    history = read_history(csv_path)
    outputs = []
    if args.plot in {"metrics", "both"}:
        metric_out = out_path if args.plot == "metrics" else out_path.with_name(f"{out_path.stem}_metrics{out_path.suffix}")
        selected_metrics = None
        if args.metrics == "coverage-discovery":
            selected_metrics = [
                ("coverage", "#1f77b4", "Coverage"),
                ("target_discovery_rate", "#2ca02c", "Discovery"),
            ]
        write_training_svg(
            metric_out,
            args.title,
            history,
            show_raw=args.show_raw,
            smooth_window=args.window,
            y_min=args.y_min,
            y_max=args.y_max,
            metrics=selected_metrics,
            figure_title=args.figure_title,
        )
        outputs.append(metric_out)
    if args.plot in {"reward", "both"}:
        reward_out = out_path if args.plot == "reward" else out_path.with_name(f"{out_path.stem}_reward{out_path.suffix}")
        write_reward_svg(reward_out, history, title=args.title, smooth_window=args.window, show_raw=args.show_raw, y_min=args.y_min, y_max=args.y_max)
        outputs.append(reward_out)
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
