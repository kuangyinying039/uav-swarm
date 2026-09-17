"""Combine held-out pursuit evaluation JSON files into CSV and a paper-ready SVG."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path

from artifact_paths import artifact_path, default_output


FIELDS = (
    "label", "episodes", "capture_rate", "capture_ci_low", "capture_ci_high",
    "mean_success_steps", "mean_censored_steps", "mean_team_visibility_ratio",
    "mean_uav_visibility_ratio", "mean_building_occlusion_ratio",
    "mean_n_uavs_seeing_target", "mean_return", "mean_safety_interventions",
    "mean_safety_correction_rate", "mean_safety_correction_magnitude",
    "p90_safety_interventions", "safety_intervention_free_episode_rate",
    "mean_emergency_stop_rate", "mean_policy_compute_ms",
)


def _metric(metrics, *names):
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _wilson(success_rate, n):
    if not n:
        return None, None
    z = 1.96
    center = (success_rate + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(success_rate*(1-success_rate)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return max(0.0, center-half), min(1.0, center+half)


def load_row(spec):
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=PATH, got {spec!r}")
    label, raw_path = spec.split("=", 1)
    path = artifact_path(raw_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    if len(summary) == 1:
        metrics = next(iter(summary.values()))
    elif label.lower() in summary:
        metrics = summary[label.lower()]
    else:
        raise ValueError(
            f"{path} contains {list(summary)}; use one of those method names as LABEL"
        )
    n = int(metrics["episodes"])
    rate = float(metrics["capture_rate"])
    low, high = (
        metrics.get("capture_ci_low"), metrics.get("capture_ci_high")
    )
    if low is None or high is None:
        low, high = _wilson(rate, n)
    row = {
        "label": label, "episodes": n, "capture_rate": rate,
        "capture_ci_low": low, "capture_ci_high": high,
        "mean_success_steps": _metric(metrics, "mean_success_steps", "mean_capture_time_success_only"),
        "mean_censored_steps": _metric(metrics, "mean_censored_steps", "mean_censored_time"),
    }
    for field in FIELDS:
        if field.startswith("mean_") and field not in row:
            row[field] = metrics.get(field)
    return row


def write_svg(path, rows):
    panel_specs = (
        ("Capture rate", "capture_rate", 0.0, 1.0, True),
        ("Team visibility ratio", "mean_team_visibility_ratio", 0.0, 1.0, False),
        ("Successful capture steps", "mean_success_steps", 0.0, None, False),
    )
    width, left, panel_width, row_height = 1380, 180, 330, 48
    height = 105 + len(rows)*row_height + 70
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect width="{width}" height="{height}" fill="white"/>',
            '<g font-family="Arial, sans-serif" fill="#183249">',
            '<text x="24" y="32" font-size="22">Held-out pursuit evaluation</text>',
            '<text x="24" y="55" font-size="12">Identical environment and test seeds; capture whiskers are 95% Wilson intervals</text>']
    for panel, (title, key, low, high, interval) in enumerate(panel_specs):
        x0 = left + panel*395
        values = [row.get(key) for row in rows if row.get(key) is not None]
        upper = high if high is not None else max(values + [1.0]) * 1.08
        body.append(f'<text x="{x0}" y="82" font-size="14">{html.escape(title)}</text>')
        for fraction in (0, .5, 1):
            x = x0 + fraction*panel_width
            body.append(f'<path d="M{x},{90}V{height-45}" stroke="#e2e8f0"/>')
            body.append(f'<text x="{x}" y="{height-25}" text-anchor="middle" font-size="11">{low+(upper-low)*fraction:.2g}</text>')
        for index, row in enumerate(rows):
            y = 105 + index*row_height
            value = row.get(key)
            if panel == 0:
                body.append(f'<text x="{left-12}" y="{y+19}" text-anchor="end" font-size="13">{html.escape(row["label"])}</text>')
            if value is None:
                body.append(f'<text x="{x0+8}" y="{y+19}" font-size="12" fill="#64748b">N/A</text>')
                continue
            bar = panel_width * (float(value)-low) / max(upper-low, 1e-12)
            body.append(f'<rect x="{x0}" y="{y}" width="{max(bar, 0):.1f}" height="25" fill="#157f89"/>')
            if interval:
                a = x0 + panel_width*(row["capture_ci_low"]-low)/(upper-low)
                b = x0 + panel_width*(row["capture_ci_high"]-low)/(upper-low)
                body.append(f'<path d="M{a:.1f},{y+12.5}H{b:.1f}" stroke="#172b4d" stroke-width="3"/>')
            suffix = "%" if high == 1.0 else ""
            shown = f"{100*value:.1f}{suffix}" if suffix else f"{value:.1f}"
            body.append(f'<text x="{x0+bar+6:.1f}" y="{y+18}" font-size="12">{shown}</text>')
    body.append('</g></svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out-dir", type=artifact_path, default=default_output("radar5_paper_comparison"))
    args = parser.parse_args()
    try:
        rows = [load_row(spec) for spec in args.input]
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_svg(args.out_dir / "comparison.svg", rows)
    print(json.dumps(rows, indent=2))
    print(args.out_dir / "comparison.svg")


if __name__ == "__main__":
    main()
