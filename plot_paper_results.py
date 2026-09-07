"""Create publication-ready Times New Roman SVG figures from experiment CSV files."""

from __future__ import annotations

import argparse
import csv
import html
import math
from pathlib import Path


PALETTE = ["#2166AC", "#B2182B", "#4D4D4D", "#1B9E77", "#7570B3", "#D95F02", "#666666"]
MARKERS = ["circle", "square", "triangle", "diamond", "cross", "circle", "square"]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value) -> str:
    return html.escape(str(value))


def marker(x: float, y: float, color: str, kind: str) -> str:
    if kind == "square":
        return f'<rect x="{x-4:.1f}" y="{y-4:.1f}" width="8" height="8" fill="{color}"/>'
    if kind == "triangle":
        return f'<path d="M{x:.1f},{y-5:.1f} L{x+5:.1f},{y+4:.1f} L{x-5:.1f},{y+4:.1f}Z" fill="{color}"/>'
    if kind == "diamond":
        return f'<path d="M{x:.1f},{y-5:.1f} L{x+5:.1f},{y:.1f} L{x:.1f},{y+5:.1f} L{x-5:.1f},{y:.1f}Z" fill="{color}"/>'
    if kind == "cross":
        return f'<path d="M{x-4:.1f},{y-4:.1f}L{x+4:.1f},{y+4:.1f}M{x+4:.1f},{y-4:.1f}L{x-4:.1f},{y+4:.1f}" stroke="{color}" stroke-width="2"/>'
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'


def svg_start(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:"Times New Roman",Times,serif;fill:#111}'
        '.axis{stroke:#111;stroke-width:1}.grid{stroke:#D9D9D9;stroke-width:.8}'
        '.tick{font-size:15px}.label{font-size:18px}.title{font-size:22px;font-weight:bold}</style>',
    ]


def comparison_figure(rows: list[dict], output: Path, title: str) -> None:
    panels = [
        ("discovery_rate", "Target discovery rate", True),
        ("coverage", "Coverage rate", True),
        ("mfdt", "Mean first-detection time", False),
        ("collision_rate", "Collision rate", False),
    ]
    methods = [row["method"] for row in rows]
    width, height = 1320, 820
    body = svg_start(width, height)
    body.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    panel_w, panel_h = 520, 260
    origins = [(90, 105), (730, 105), (90, 500), (730, 500)]
    for panel_idx, (metric, label, unit_rate) in enumerate(panels):
        x0, y0 = origins[panel_idx]
        means = [float(row[f"{metric}_mean"]) for row in rows]
        cis = [float(row.get(f"{metric}_ci95", 0.0)) for row in rows]
        ymax = 1.0 if unit_rate else max([m + c for m, c in zip(means, cis)] + [1e-6]) * 1.15
        body.append(f'<text x="{x0+panel_w/2}" y="{y0-18}" text-anchor="middle" class="title">{esc(label)}</text>')
        for tick in range(6):
            y = y0 + tick * panel_h / 5
            value = ymax * (5 - tick) / 5
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{x0-10}" y="{y+5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
        body.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}" class="axis"/>')
        body.append(f'<line x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}" class="axis"/>')
        slot = panel_w / max(len(rows), 1)
        bar_w = min(62.0, slot * 0.56)
        for idx, (method, mean, ci) in enumerate(zip(methods, means, cis)):
            x = x0 + (idx + 0.5) * slot
            y = y0 + panel_h * (1 - mean / max(ymax, 1e-12))
            bar_h = y0 + panel_h - y
            color = PALETTE[idx % len(PALETTE)]
            body.append(f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity=".88"/>')
            y_hi = y0 + panel_h * (1 - min(ymax, mean + ci) / max(ymax, 1e-12))
            y_lo = y0 + panel_h * (1 - max(0.0, mean - ci) / max(ymax, 1e-12))
            body.append(f'<line x1="{x:.1f}" y1="{y_hi:.1f}" x2="{x:.1f}" y2="{y_lo:.1f}" stroke="#111"/>')
            body.append(f'<line x1="{x-6:.1f}" y1="{y_hi:.1f}" x2="{x+6:.1f}" y2="{y_hi:.1f}" stroke="#111"/>')
            body.append(f'<line x1="{x-6:.1f}" y1="{y_lo:.1f}" x2="{x+6:.1f}" y2="{y_lo:.1f}" stroke="#111"/>')
            body.append(f'<text x="{x:.1f}" y="{y0+panel_h+23}" text-anchor="middle" class="tick">{esc(method)}</text>')
    body.append("</svg>")
    output.write_text("\n".join(body), encoding="utf-8")


def sensitivity_figure(rows: list[dict], parameter: str, output: Path, title: str) -> None:
    subset = [row for row in rows if row["parameter"] == parameter]
    if not subset:
        raise ValueError(f"No rows for parameter {parameter!r}")
    panels = [
        ("discovery_rate", "Target discovery rate", True),
        ("coverage", "Coverage rate", True),
        ("mfdt", "Mean first-detection time", False),
        ("collision_rate", "Collision rate", False),
    ]
    methods = sorted({row["method"] for row in subset})
    values = sorted({float(row["value"]) for row in subset})
    width, height = 1320, 820
    body = svg_start(width, height)
    body.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    for idx, method in enumerate(methods):
        x = 360 + idx * 190
        color = PALETTE[idx % len(PALETTE)]
        body.append(f'<line x1="{x}" y1="66" x2="{x+28}" y2="66" stroke="{color}" stroke-width="2.5"/>')
        body.append(marker(x + 14, 66, color, MARKERS[idx % len(MARKERS)]))
        body.append(f'<text x="{x+36}" y="71" class="tick">{esc(method)}</text>')
    origins = [(90, 120), (730, 120), (90, 510), (730, 510)]
    panel_w, panel_h = 520, 235
    for panel_idx, (metric, label, unit_rate) in enumerate(panels):
        x0, y0 = origins[panel_idx]
        metric_rows = [row for row in subset if math.isfinite(float(row[f"{metric}_mean"]))]
        ymax = 1.0 if unit_rate else max(
            [float(row[f"{metric}_mean"]) + float(row.get(f"{metric}_ci95", 0.0)) for row in metric_rows] + [1e-6]
        ) * 1.12
        body.append(f'<text x="{x0+panel_w/2}" y="{y0-16}" text-anchor="middle" class="title">{esc(label)}</text>')
        for tick in range(6):
            y = y0 + tick * panel_h / 5
            value = ymax * (5 - tick) / 5
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{x0-10}" y="{y+5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
        for value_idx, value in enumerate(values):
            x = x0 + value_idx / max(len(values) - 1, 1) * panel_w
            body.append(f'<text x="{x:.1f}" y="{y0+panel_h+22}" text-anchor="middle" class="tick">{value:g}</text>')
        for method_idx, method in enumerate(methods):
            points = []
            color = PALETTE[method_idx % len(PALETTE)]
            for value_idx, value in enumerate(values):
                row = next(item for item in subset if item["method"] == method and float(item["value"]) == value)
                mean = float(row[f"{metric}_mean"])
                ci = float(row.get(f"{metric}_ci95", 0.0))
                x = x0 + value_idx / max(len(values) - 1, 1) * panel_w
                y = y0 + panel_h * (1 - mean / max(ymax, 1e-12))
                y1 = y0 + panel_h * (1 - min(ymax, mean + ci) / max(ymax, 1e-12))
                y2 = y0 + panel_h * (1 - max(0, mean - ci) / max(ymax, 1e-12))
                points.append(f"{x:.1f},{y:.1f}")
                body.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="{color}"/>')
                body.append(marker(x, y, color, MARKERS[method_idx % len(MARKERS)]))
            body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.3"/>')
        body.append(f'<text x="{x0+panel_w/2}" y="{y0+panel_h+45}" text-anchor="middle" class="label">{esc(parameter.replace("_", " ").title())}</text>')
    body.append("</svg>")
    output.write_text("\n".join(body), encoding="utf-8")


def training_figure(rows: list[dict], output: Path, title: str) -> None:
    panels = [
        ("reward", "Episode reward"),
        ("target_discovery_rate", "Target discovery rate"),
        ("coverage", "Coverage rate"),
        ("collision_rate", "Collision rate"),
    ]
    width, height = 1320, 820
    body = svg_start(width, height)
    body.append(f'<text x="{width/2}" y="34" text-anchor="middle" class="title">{esc(title)}</text>')
    origins = [(90, 100), (730, 100), (90, 490), (730, 490)]
    panel_w, panel_h = 520, 260
    episodes = [float(row["episode"]) for row in rows]
    for panel_idx, (metric, label) in enumerate(panels):
        x0, y0 = origins[panel_idx]
        means = [float(row[f"{metric}_mean"]) for row in rows]
        stds = [float(row.get(f"{metric}_std", 0.0)) for row in rows]
        ymin = min([m - s for m, s in zip(means, stds)] + [0.0])
        ymax = max([m + s for m, s in zip(means, stds)] + [1e-6])
        if label.endswith("rate"):
            ymin, ymax = 0.0, max(1.0, ymax)
        pad = max((ymax - ymin) * 0.05, 1e-8)
        ymin, ymax = ymin - pad, ymax + pad
        body.append(f'<text x="{x0+panel_w/2}" y="{y0-16}" text-anchor="middle" class="title">{esc(label)}</text>')
        for tick in range(6):
            y = y0 + tick * panel_h / 5
            value = ymax - tick * (ymax - ymin) / 5
            body.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+panel_w}" y2="{y:.1f}" class="grid"/>')
            body.append(f'<text x="{x0-10}" y="{y+5:.1f}" text-anchor="end" class="tick">{value:.2f}</text>')
        upper, lower, line = [], [], []
        for episode, mean, std in zip(episodes, means, stds):
            x = x0 + (episode - episodes[0]) / max(episodes[-1] - episodes[0], 1) * panel_w
            y = y0 + (ymax - mean) / (ymax - ymin) * panel_h
            yu = y0 + (ymax - min(ymax, mean + std)) / (ymax - ymin) * panel_h
            yl = y0 + (ymax - max(ymin, mean - std)) / (ymax - ymin) * panel_h
            line.append(f"{x:.1f},{y:.1f}")
            upper.append(f"{x:.1f},{yu:.1f}")
            lower.append(f"{x:.1f},{yl:.1f}")
        band = " ".join(upper + list(reversed(lower)))
        body.append(f'<polygon points="{band}" fill="{PALETTE[0]}" opacity=".16"/>')
        body.append(f'<polyline points="{" ".join(line)}" fill="none" stroke="{PALETTE[0]}" stroke-width="2.2"/>')
        body.append(f'<text x="{x0+panel_w/2}" y="{y0+panel_h+38}" text-anchor="middle" class="label">Training episode</text>')
    body.append("</svg>")
    output.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["comparison", "ablation", "sensitivity", "training"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameter", default=None, help="Required for sensitivity mode.")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    rows = read_csv(Path(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode in {"comparison", "ablation"}:
        comparison_figure(rows, output, args.title or "Method Comparison")
    elif args.mode == "sensitivity":
        if not args.parameter:
            raise ValueError("--parameter is required for sensitivity mode")
        sensitivity_figure(rows, args.parameter, output, args.title or "Sensitivity Analysis")
    else:
        training_figure(rows, output, args.title or "Training Convergence")
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
