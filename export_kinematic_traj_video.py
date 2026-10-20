#!/usr/bin/env python3
"""Render a top-down kinematic pursuit video from a recorded trajectory NPZ.

Matches the style of ``pursuit_visualization`` HTML (top view): gray building
blocks, UAV/target markers, short trails. Encodes MP4 via ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


UAV_COLORS = (
    "#1649b7",
    "#ec8b19",
    "#169c62",
    "#8a3fc5",
)
TARGET_COLOR = "#d9272e"
BUILDING_FACE = "#2c3038"
BUILDING_EDGE = "#15181e"


def _load_traj(path: Path):
    payload = np.load(path, allow_pickle=True)
    pursuer = np.asarray(payload["pursuer"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    buildings = np.asarray(payload["buildings"], dtype=np.float64)
    heights = np.asarray(payload["building_heights"], dtype=np.float64)
    meta_raw = payload["meta"]
    if getattr(meta_raw, "ndim", 0) == 0:
        meta_raw = meta_raw.item()
    if isinstance(meta_raw, (bytes, np.bytes_)):
        meta_raw = meta_raw.decode("utf-8")
    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw or {})
    return pursuer, target, buildings, heights, meta


def _render_frames(
    *,
    pursuer: np.ndarray,
    target: np.ndarray,
    buildings: np.ndarray,
    out_dir: Path,
    grid_size: float,
    trail_len: int,
    width: int,
    height: int,
    dpi: int,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    n_steps = int(pursuer.shape[0])
    n_uavs = int(pursuer.shape[1])
    margin = 1.2
    fig_w = width / dpi
    fig_h = height / dpi

    for step in range(n_steps):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        ax.set_xlim(-margin, grid_size + margin)
        ax.set_ylim(-margin, grid_size + margin)
        ax.set_aspect("equal")
        ax.set_facecolor("#fbfcfe")
        fig.patch.set_facecolor("#fbfcfe")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#c8ced8")
            spine.set_linewidth(1.0)

        # Map pad
        ax.add_patch(
            Rectangle(
                (0, 0),
                grid_size,
                grid_size,
                facecolor="#eef2f7",
                edgecolor="#b8c0cc",
                linewidth=1.2,
                zorder=0,
            )
        )

        for rect in buildings:
            x0, y0, x1, y1 = map(float, rect)
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    facecolor=BUILDING_FACE,
                    edgecolor=BUILDING_EDGE,
                    alpha=0.55,
                    linewidth=1.4,
                    zorder=1,
                )
            )

        t0 = max(0, step - trail_len + 1)
        for u in range(n_uavs):
            trail = pursuer[t0 : step + 1, u, :2]
            color = UAV_COLORS[u % len(UAV_COLORS)]
            if len(trail) >= 2:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    color=color,
                    linewidth=2.0,
                    alpha=0.75,
                    zorder=2,
                    solid_capstyle="round",
                )
            xy = pursuer[step, u, :2]
            ax.plot(xy[0], xy[1], marker="^", markersize=11, color=color, zorder=4)

        t_trail = target[t0 : step + 1, :2]
        if len(t_trail) >= 2:
            ax.plot(
                t_trail[:, 0],
                t_trail[:, 1],
                color=TARGET_COLOR,
                linewidth=2.4,
                alpha=0.85,
                zorder=2,
                solid_capstyle="round",
            )
        tx, ty = target[step, :2]
        ax.add_patch(
            Circle(
                (tx, ty),
                0.55,
                facecolor=TARGET_COLOR,
                edgecolor="#8d1117",
                linewidth=1.2,
                zorder=5,
            )
        )

        ax.set_title(
            f"Kinematic top view  |  step {step}/{n_steps - 1}",
            fontsize=11,
            color="#1b2430",
            pad=8,
        )
        out_path = out_dir / f"frame_{step:05d}.png"
        # Fixed canvas size (even dims) so libx264 accepts every frame.
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)

    return n_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--trail-len", type=int, default=60)
    parser.add_argument("--grid-size", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    pursuer, target, buildings, _heights, meta = _load_traj(args.traj)
    grid_size = float(args.grid_size if args.grid_size is not None else meta.get("grid_size", 25.0))
    if "grid_size" not in meta:
        # Infer from building extents / positions if meta omitted it.
        grid_size = float(
            max(
                grid_size,
                float(np.max(buildings[:, [2, 3]])) + 1.0 if len(buildings) else 25.0,
                float(np.max(pursuer[:, :, :2])) + 1.0,
                float(np.max(target[:, :2])) + 1.0,
            )
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kinematic_vid_") as tmp:
        tmp_dir = Path(tmp)
        n = _render_frames(
            pursuer=pursuer,
            target=target,
            buildings=buildings,
            out_dir=tmp_dir,
            grid_size=grid_size,
            trail_len=max(2, int(args.trail_len)),
            width=int(args.width),
            height=int(args.height),
            dpi=int(args.dpi),
        )
        pattern = str(tmp_dir / "frame_%05d.png")
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            pattern,
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.out),
        ]
        print(f"[kinematic-video] frames={n} -> {args.out}", flush=True)
        subprocess.run(cmd, check=True)
    print(args.out)


if __name__ == "__main__":
    main()
