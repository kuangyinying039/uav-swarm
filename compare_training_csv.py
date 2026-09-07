"""Compare MARL/heuristic training CSV files and draw a four-panel SVG dashboard."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#7f7f7f"]
METRICS = [
    ("reward", "Episode reward", None),
    ("target_discovery_rate", "Target discovery rate", (0.0, 1.0)),
    ("coverage", "Coverage", (0.0, 1.0)),
    ("collision_rate", "Collision rate", None),
]


def finite(value: str | None) -> float:
    try:
        result = float(value or "nan")
        return result if math.isfinite(result) else float("nan")
    except ValueError:
        return float("nan")


def moving_average(values: list[float], window: int) -> list[float]:
    output = []
    for index in range(len(values)):
        chunk = [v for v in values[max(0, index - window + 1) : index + 1] if math.isfinite(v)]
        output.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return output


def read_series(spec: str) -> dict:
    if "=" not in spec:
        raise ValueError(f"Series must be LABEL=CSV_PATH, got: {spec}")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and not path.exists():
        # Commands are commonly launched from either the project root or the
        # repro directory.  If CWD-relative lookup fails, also resolve paths
        # such as outputs/... from the project root next to repro/.
        project_relative = Path(__file__).resolve().parent.parent / path
        if project_relative.exists():
            path = project_relative
    if not path.exists():
        cwd_candidate = (Path.cwd() / path).resolve()
        project_candidate = (Path(__file__).resolve().parent.parent / path).resolve()
        raise FileNotFoundError(
            f"CSV not found: {raw_path}\n"
            f"Tried from current directory: {cwd_candidate}\n"
            f"Tried from project root: {project_candidate}"
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"CSV is empty: {path}")
    mean_file = "reward_mean" in rows[0]
    values = {"episode": [int(float(row["episode"])) for row in rows]}
    for metric, _, _ in METRICS:
        column = f"{metric}_mean" if mean_file else metric
        values[metric] = [finite(row.get(column)) for row in rows]
    return {"label": label.strip(), "path": path, "values": values}


def points(xs, ys, x0, y0, width, height, x_max, y_min, y_max) -> str:
    output = []
    for x, y in zip(xs, ys):
        if math.isfinite(y):
            px = x0 + x / max(x_max, 1) * width
            py = y0 + (y_max - y) / max(y_max - y_min, 1e-9) * height
            output.append(f"{px:.1f},{py:.1f}")
    return " ".join(output)


def write_dashboard(path: Path, series: list[dict], window: int, title: str) -> None:
    width, height = 1400, 940
    panel_w, panel_h = 620, 340
    origins = [(90, 110), (760, 110), (90, 520), (760, 520)]
    x_max = max(max(item["values"]["episode"]) for item in series)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    body.append('<rect width="100%" height="100%" fill="#fcfcfa"/>')
    body.append(f'<text x="700" y="42" text-anchor="middle" font-family="Times New Roman" font-size="32" font-weight="bold">{html.escape(title)}</text>')
    legend_x = 120
    for index, item in enumerate(series):
        color = COLORS[index % len(COLORS)]
        body.append(f'<line x1="{legend_x}" y1="72" x2="{legend_x+30}" y2="72" stroke="{color}" stroke-width="4"/>')
        body.append(f'<text x="{legend_x+38}" y="77" font-family="Times New Roman" font-size="19">{html.escape(item["label"])}</text>')
        legend_x += 55 + 9 * len(item["label"])
    for metric_index, (metric, label, fixed_range) in enumerate(METRICS):
        x0, y0 = origins[metric_index]
        all_values = []
        smoothed = []
        for item in series:
            trend = moving_average(item["values"][metric], window)
            smoothed.append(trend)
            all_values.extend(v for v in trend if math.isfinite(v))
        if fixed_range:
            y_min, y_max = fixed_range
        elif all_values:
            data_min, data_max = min(all_values), max(all_values)
            padding = max((data_max - data_min) * 0.08, 0.01)
            y_min, y_max = data_min - padding, data_max + padding
        else:
            y_min, y_max = 0.0, 1.0
        body.append(f'<text x="{x0+panel_w/2}" y="{y0-22}" text-anchor="middle" font-family="Times New Roman" font-size="23" font-weight="bold">{label}</text>')
        for tick in range(6):
            py = y0 + tick * panel_h / 5
            value = y_max - tick * (y_max - y_min) / 5
            body.append(f'<line x1="{x0}" y1="{py:.1f}" x2="{x0+panel_w}" y2="{py:.1f}" stroke="#e2e2de"/>')
            body.append(f'<text x="{x0-10}" y="{py+5:.1f}" text-anchor="end" font-family="Times New Roman" font-size="17">{value:.2f}</text>')
        for tick in range(6):
            px = x0 + tick * panel_w / 5
            episode = tick * x_max / 5
            body.append(f'<text x="{px:.1f}" y="{y0+panel_h+23}" text-anchor="middle" font-family="Times New Roman" font-size="17">{episode:.0f}</text>')
        body.append(f'<line x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}" stroke="#333"/>')
        body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}" stroke="#333"/>')
        for index, item in enumerate(series):
            curve = points(item["values"]["episode"], smoothed[index], x0, y0, panel_w, panel_h, x_max, y_min, y_max)
            body.append(f'<polyline points="{curve}" fill="none" stroke="{COLORS[index % len(COLORS)]}" stroke-width="2.5" stroke-linejoin="round" opacity="0.92"/>')
        body.append(f'<text x="{x0+panel_w/2}" y="{y0+panel_h+45}" text-anchor="middle" font-family="Times New Roman" font-size="18">Episode</text>')
    body.append(f'<text x="700" y="925" text-anchor="middle" font-family="Times New Roman" font-size="17" fill="#555">Moving average window = {window}; curves are training episodes, not deterministic evaluation.</text>')
    body.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def write_summary(path: Path, series: list[dict], tail: int) -> None:
    fields = ["series", "source", "tail_episodes"] + [f"{metric}_tail_mean" for metric, _, _ in METRICS]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in series:
            row = {"series": item["label"], "source": str(item["path"]), "tail_episodes": tail}
            for metric, _, _ in METRICS:
                values = [v for v in item["values"][metric][-tail:] if math.isfinite(v)]
                row[f"{metric}_tail_mean"] = sum(values) / len(values) if values else ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", action="append", required=True, help="Repeat LABEL=CSV_PATH for each curve")
    parser.add_argument("--out", default="training_comparison.svg")
    parser.add_argument("--title", default="MARL training comparison")
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--tail", type=int, default=100)
    args = parser.parse_args()
    loaded = [read_series(spec) for spec in args.series]
    output = Path(args.out)
    write_dashboard(output, loaded, max(1, args.window), args.title)
    summary = output.with_name(f"{output.stem}_summary.csv")
    write_summary(summary, loaded, max(1, args.tail))
    print(output.resolve())
    print(summary.resolve())


if __name__ == "__main__":
    main()
