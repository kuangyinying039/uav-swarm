"""Compare the same pursuit methods across ordered benchmark difficulties."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

from artifact_paths import artifact_path, default_output
from compare_pursuit_evaluations import FIELDS, load_row


DIFFICULTY_ORDER = {"Nominal": 0, "Medium": 1, "Hard": 2, "Extreme": 3}


def load_spec(spec):
    if ":" not in spec:
        raise ValueError(f"Expected DIFFICULTY:LABEL=PATH, got {spec!r}")
    difficulty, row_spec = spec.split(":", 1)
    row = load_row(row_spec)
    row["difficulty"] = difficulty
    return row


def ordered_difficulties(rows):
    present = {row["difficulty"] for row in rows}
    return sorted(present, key=lambda name: DIFFICULTY_ORDER[name])


def validate_matrix(rows, required_methods=(), required_difficulties=None):
    """Reject incomplete or duplicate difficulty-by-method comparisons.

    By default, completeness is checked only over difficulties present in the
    inputs so a three-level paper figure and a four-level Extreme stress plot
    can share the same tool. Pass ``required_difficulties`` to force a full grid.
    """
    pairs = [(row["difficulty"], row["label"]) for row in rows]
    if len(pairs) != len(set(pairs)):
        duplicates = sorted({pair for pair in pairs if pairs.count(pair) > 1})
        raise ValueError(f"Duplicate difficulty/method inputs: {duplicates}")
    unknown = sorted({row["difficulty"] for row in rows} - set(DIFFICULTY_ORDER))
    if unknown:
        raise ValueError(
            f"Unknown difficulties {unknown}; use Nominal, Medium, Hard, or Extreme"
        )
    if required_difficulties is None:
        difficulties = ordered_difficulties(rows)
    else:
        difficulties = list(required_difficulties)
        bad = sorted(set(difficulties) - set(DIFFICULTY_ORDER))
        if bad:
            raise ValueError(f"Unknown required difficulties {bad}")
    missing = [
        f"{difficulty}:{method}"
        for difficulty in difficulties
        for method in required_methods
        if (difficulty, method) not in set(pairs)
    ]
    if missing:
        raise ValueError("Incomplete difficulty comparison; missing " + ", ".join(missing))


def _points(rows, method, key, x0, y0, width, height, low, high, difficulties):
    selected = sorted(
        (row for row in rows if row["label"] == method and row.get(key) is not None),
        key=lambda row: DIFFICULTY_ORDER.get(row["difficulty"], 999),
    )
    span = max(len(difficulties) - 1, 1)
    index_of = {name: index for index, name in enumerate(difficulties)}
    points = []
    for row in selected:
        index = index_of.get(row["difficulty"])
        if index is None:
            continue
        x = x0 + index * width / span
        y = y0 + height * (1.0 - (float(row[key]) - low) / (high - low))
        points.append((x, y))
    return points


def write_svg(path, rows):
    methods = list(dict.fromkeys(row["label"] for row in rows))
    difficulties = ordered_difficulties(rows)
    panels = (
        ("Capture rate", "capture_rate", 0.0, 1.0),
        ("Team visibility ratio", "mean_team_visibility_ratio", 0.0, 1.0),
        ("Mean censored steps", "mean_censored_steps", 0.0, 300.0),
    )
    colors = ("#157f89", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#475569")
    width, height = 1320, 520
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#183249">',
        '<text x="28" y="34" font-size="22">Pursuit performance across difficulty</text>',
        '<text x="28" y="56" font-size="12">Same capture condition and held-out seed range in each profile</text>',
    ]
    for panel_index, (title, key, low, high) in enumerate(panels):
        x0, y0, panel_width, panel_height = 75 + panel_index * 420, 105, 320, 300
        body.append(f'<text x="{x0}" y="86" font-size="14">{html.escape(title)}</text>')
        for fraction in (0.0, 0.5, 1.0):
            y = y0 + panel_height * (1.0 - fraction)
            body.append(f'<path d="M{x0},{y:.1f}H{x0+panel_width}" stroke="#e2e8f0"/>')
            body.append(
                f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" font-size="10">'
                f'{low+(high-low)*fraction:.2g}</text>'
            )
        span = max(len(difficulties) - 1, 1)
        for index, difficulty in enumerate(difficulties):
            x = x0 + index * panel_width / span
            body.append(
                f'<text x="{x:.1f}" y="430" text-anchor="middle" font-size="11">'
                f'{html.escape(difficulty)}</text>'
            )
        for method_index, method in enumerate(methods):
            color = colors[method_index % len(colors)]
            points = _points(
                rows, method, key, x0, y0, panel_width, panel_height, low, high, difficulties
            )
            if len(points) > 1:
                encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
                body.append(f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for x, y in points:
                body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
    legend_y = 470
    for index, method in enumerate(methods):
        x = 40 + (index % 6) * 205
        y = legend_y + (index // 6) * 22
        color = colors[index % len(colors)]
        body.append(f'<path d="M{x},{y}h24" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{x+30}" y="{y+4}" font-size="11">{html.escape(method)}</text>')
    body.append('</g></svg>')
    path.write_text("\n".join(body), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, metavar="DIFFICULTY:LABEL=PATH")
    parser.add_argument(
        "--require-methods", nargs="+", default=(), metavar="LABEL",
        help="Require every listed method at each difficulty present in --input",
    )
    parser.add_argument(
        "--require-difficulties", nargs="+", default=None, metavar="DIFFICULTY",
        help="Optionally force a full difficulty grid (e.g. Nominal Medium Hard Extreme)",
    )
    parser.add_argument("--out-dir", type=artifact_path, default=default_output("radar5_difficulty_comparison"))
    args = parser.parse_args()
    try:
        rows = [load_spec(spec) for spec in args.input]
        validate_matrix(rows, args.require_methods, args.require_difficulties)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ("difficulty", *FIELDS)
    with (args.out_dir / "difficulty_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "difficulty_comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    write_svg(args.out_dir / "difficulty_comparison.svg", rows)
    print(args.out_dir / "difficulty_comparison.svg")


if __name__ == "__main__":
    main()
