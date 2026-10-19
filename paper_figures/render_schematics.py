"""Paper schematics for the dual-capture Isaac experiments.

Numbers are read from configs/pursuit_v2/radar5_dual_*.json. Capture-rate
charts are intentionally not drawn here: the saved comparison used the
kinematic-or scorer and stays in paper_figures/legacy_or_scorer/.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "pursuit_v2"
OUT = Path(__file__).resolve().parent

LEVELS = ("nominal", "medium", "hard", "severe", "extreme")
TRAIN = {"hard"}


def load(level: str) -> dict:
    return json.loads((CONFIG_DIR / f"radar5_dual_{level}.json").read_text())


def svg_header(width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<g font-family="Helvetica, Arial, sans-serif" fill="#1a1a1a">\n'
    )


def difficulty() -> str:
    rows = []
    for level in LEVELS:
        cfg = load(level)
        role = "train + test" if level in TRAIN else "test only"
        rows.append(
            (
                level.capitalize(),
                role,
                f"{cfg['target_speed']:.2f} / {cfg['target_vertical_speed']:.2f}",
                f"{cfg['handoff_formation_min_distance']:.1f}–{cfg['handoff_formation_max_distance']:.1f}",
                f"{cfg['lidar_target_detection_range']:.0f}",
                f"{cfg['lidar_detection_probability']:.2f}",
                f"{cfg['building_min_size']:.2f}–{cfg['building_max_size']:.2f}",
                f"{cfg['building_min_height']:.1f}–{cfg['building_max_height']:.1f}",
            )
        )
    width, height = 980, 360
    parts = [
        svg_header(width, height),
        '<text x="36" y="32" font-size="18">Difficulty ladder and where each level is used</text>',
        '<text x="36" y="54" font-size="12" fill="#444">Map 25 m × 25 m, altitude 1–12 m, 3 pursuers, 1 occlusion evader, 5 buildings, 300 decisions. Training and checkpoint selection use Hard only. The other four levels are tests.</text>',
    ]
    headers = ["Level", "Role", "Evader speed h/v (m/s)", "Start range (m)", "Lidar (m)", "Detect p", "Footprint (m)", "Height (m)"]
    xs = [36, 130, 250, 430, 560, 650, 740, 860]
    y0 = 96
    parts.append(f'<line x1="36" y1="{y0 - 8}" x2="944" y2="{y0 - 8}" stroke="#222"/>')
    for x, header in zip(xs, headers):
        parts.append(f'<text x="{x}" y="{y0}" font-size="12" font-weight="500">{header}</text>')
    parts.append(f'<line x1="36" y1="{y0 + 10}" x2="944" y2="{y0 + 10}" stroke="#222"/>')
    for index, row in enumerate(rows):
        y = y0 + 42 + index * 36
        fill = "#f4f7fb" if row[0].lower() in TRAIN else "#fafafa"
        parts.append(f'<rect x="28" y="{y - 22}" width="916" height="32" fill="{fill}"/>')
        for x, cell in zip(xs, row):
            parts.append(f'<text x="{x}" y="{y}" font-size="13">{cell}</text>')
    parts.append('<text x="36" y="332" font-size="12" fill="#444">Shared contract: capture radius 1.0 m (twice the 0.5 m target diameter), 2 of 3 bodies inside for 1 decision, decision period 0.2 s. Pursuer limits 1.4 m/s horizontal, 0.8 m/s vertical.</text>')
    parts.append("</g></svg>\n")
    return "".join(parts)


def geometry() -> str:
    width, height = 980, 420
    parts = [
        svg_header(width, height),
        '<text x="36" y="32" font-size="18">Physical dual capture and the lidar surrogate</text>',
        '<text x="36" y="54" font-size="12" fill="#444">Success requires two Hummingbird bodies inside the 1 m sphere. A kinematic proxy inside that sphere does not end the episode and does not pay the capture bonus.</text>',
        '<text x="150" y="86" text-anchor="middle" font-size="13">Top view, capture plane</text>',
        '<rect x="70" y="100" width="280" height="280" fill="none" stroke="#222"/>',
        '<text x="78" y="118" font-size="11" fill="#444">25 m</text>',
        '<circle cx="210" cy="240" r="56" fill="none" stroke="#1F4E79" stroke-width="1.5"/>',
        '<circle cx="210" cy="240" r="6" fill="#1a1a1a"/>',
        '<text x="222" y="236" font-size="12">evader</text>',
        '<text x="222" y="292" font-size="12" fill="#1F4E79">1.0 m sphere</text>',
        '<circle cx="176" cy="214" r="7" fill="#1F4E79"/>',
        '<circle cx="246" cy="228" r="7" fill="#1F4E79"/>',
        '<circle cx="168" cy="292" r="7" fill="none" stroke="#1F4E79" stroke-width="1.5"/>',
        '<text x="78" y="368" font-size="12">Filled: bodies inside. Open: the third pursuer still outside.</text>',
        '<text x="620" y="86" text-anchor="middle" font-size="13">Side view, lidar vertical field</text>',
        '<line x1="470" y1="300" x2="860" y2="300" stroke="#222"/>',
        '<text x="848" y="318" font-size="11" fill="#444">horizon</text>',
        '<polygon points="560,300 860,188 860,332" fill="#e8eef5" stroke="#1F4E79"/>',
        '<text x="700" y="250" font-size="12">+52°</text>',
        '<text x="700" y="328" font-size="12">−7°</text>',
        '<text x="470" y="150" font-size="13">Horizontal field 360°. Range 0.2–12 m</text>',
        '<text x="470" y="172" font-size="13">on Nominal–Hard, 11 m on Severe, 9 m on Extreme.</text>',
        '<text x="470" y="194" font-size="13">Scan 10 Hz. Detection probability 0.90, 0.85, then 0.80.</text>',
        '<text x="470" y="360" font-size="12" fill="#444">Livox MID-360 field of view, with a shorter range than the hardware specification.</text>',
        "</g></svg>\n",
    ]
    return "".join(parts)


def protocol() -> str:
    width, height = 980, 280
    parts = [
        svg_header(width, height),
        '<text x="36" y="32" font-size="18">What is trained, and what is only tested</text>',
        '<text x="36" y="54" font-size="12" fill="#444">Training and best_capture.pt use Hard only, scored by physical capture. Medium is the easier held-out test. Severe and Extreme are harder held-out tests.</text>',
    ]
    boxes = [
        (36, "Medium", "test"),
        (250, "Hard", "train"),
        (464, "Severe", "test"),
        (678, "Extreme", "test"),
    ]
    for x, label, role in boxes:
        fill = "#1F4E79" if role == "train" else "#ffffff"
        text = "#ffffff" if role == "train" else "#1a1a1a"
        parts.append(f'<rect x="{x}" y="88" width="180" height="64" fill="{fill}" stroke="#1F4E79"/>')
        parts.append(f'<text x="{x + 90}" y="116" text-anchor="middle" font-size="16" fill="{text}">{label}</text>')
        parts.append(f'<text x="{x + 90}" y="138" text-anchor="middle" font-size="12" fill="{text}">{role}</text>')
    parts.append('<text x="36" y="190" font-size="13">Hard keeps the 12 m lidar while raising speed, occlusion, and start distance. Nominal is easier. Severe shortens lidar to 11 m. Extreme stays a held-out test.</text>')
    parts.append('<text x="36" y="220" font-size="13">Reward: time cost, visibility shaping, a potential on the physical gap of the second-closest body, +100 on the first physical capture, −50 on timeout.</text>')
    parts.append('<text x="36" y="250" font-size="13">Kept methods: APF, FRPN, MPC, A6 (HGAT-MATD3), MAPPO+. Ablation: MAPPO with the earlier DAgger retention. A0–A5 are not retrained.</text>')
    parts.append("</g></svg>\n")
    return "".join(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fig_difficulty_ladder.svg").write_text(difficulty(), encoding="utf-8")
    (OUT / "fig_physical_capture.svg").write_text(geometry(), encoding="utf-8")
    (OUT / "fig_train_test_split.svg").write_text(protocol(), encoding="utf-8")


if __name__ == "__main__":
    main()
