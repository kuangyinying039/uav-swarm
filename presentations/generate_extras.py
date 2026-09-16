#!/usr/bin/env python3
"""Extra schematics: framework with source selection, R aging flow."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("/tmp/ppt-work/assets")
OUT.mkdir(parents=True, exist_ok=True)
BLUE = "#2F5597"
BLUE2 = "#5B8CC9"
ORANGE = "#ED7D31"
GREEN = "#2E8B57"
DARK = "#1F2A44"


def save(fig, name):
    p = OUT / name
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.06)
    plt.close(fig)
    print("wrote", p)


def box(ax, xy, w, h, text, fc=BLUE, tc="white", fs=8.5):
    x, y = xy
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=fc, edgecolor=fc, linewidth=1.1, zorder=3,
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", zorder=4)


def arrow(ax, p1, p2, color=DARK):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=2)


def fig_framework():
    fig, ax = plt.subplots(figsize=(13.4, 4.35))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 4.35)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (0.12, 1.82), 13.15, 2.38,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#EAF0F8", edgecolor=BLUE, lw=1.5, linestyle="--",
    ))
    ax.add_patch(FancyBboxPatch(
        (0.12, 0.12), 9.55, 1.50,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor="#FFF3E8", edgecolor=ORANGE, lw=1.5, linestyle="--",
    ))
    ax.text(0.32, 3.95, "Decentralized Execution", color=BLUE, fontsize=12, fontweight="bold")
    ax.text(0.32, 1.38, "Centralized Training", color=ORANGE, fontsize=12, fontweight="bold")

    box(ax, (0.28, 2.82), 1.45, 0.82, "Local\nObservation", BLUE, fs=8)
    box(ax, (0.28, 2.02), 1.45, 0.68, "Link Mask", BLUE2, fs=8)
    box(ax, (2.00, 2.28), 1.55, 1.18, "P/E/R\nMemory", BLUE, fs=9)
    box(ax, (3.82, 2.28), 1.85, 1.18, "Source\nSelection", BLUE, fs=9)
    box(ax, (5.94, 2.28), 2.05, 1.18, "Heterogeneous Graph\nUAV / Task / Obstacle", BLUE, fs=7.8)
    box(ax, (8.26, 2.28), 1.45, 1.18, "Shared\nActor", BLUE, fs=9)
    box(ax, (9.98, 2.28), 1.50, 1.18, "Safety\nProjection", "#7A8DA8", fs=8.5)
    box(ax, (11.75, 2.38), 1.32, 0.98, "Executed\nAction", GREEN, fs=8)

    y = 2.85
    arrow(ax, (1.73, y), (2.00, y))
    arrow(ax, (3.55, y), (3.82, y))
    arrow(ax, (5.67, y), (5.94, y))
    arrow(ax, (7.99, y), (8.26, y))
    arrow(ax, (9.71, y), (9.98, y))
    arrow(ax, (11.48, y), (11.75, y))

    box(ax, (0.32, 0.32), 1.90, 0.82, "Global State", ORANGE, fs=9)
    box(ax, (2.55, 0.32), 2.15, 0.82, "Centralized Critic", ORANGE, fs=9)
    box(ax, (5.05, 0.32), 2.15, 0.82, "MAPPO\nOptimization", ORANGE, fs=9)
    box(ax, (7.55, 0.32), 1.55, 0.82, "Param.\nUpdate", ORANGE, fs=8.5)
    arrow(ax, (2.22, 0.73), (2.55, 0.73), ORANGE)
    arrow(ax, (4.70, 0.73), (5.05, 0.73), ORANGE)
    arrow(ax, (7.20, 0.73), (7.55, 0.73), ORANGE)
    ax.annotate("", xy=(8.95, 2.28), xytext=(8.35, 1.14),
                arrowprops=dict(arrowstyle="-|>", color=BLUE2, lw=2.0))
    ax.text(9.05, 1.50, "update", color=BLUE, fontsize=8, style="italic")
    save(fig, "framework.png")


def fig_aging():
    fig, ax = plt.subplots(figsize=(7.2, 1.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    labels = ["Unsensed", "Aging", "Neighbor\ndiffusion", "Sensed cell\nreset to 0"]
    xs = [0.15, 2.55, 4.95, 7.45]
    for i, (x, t) in enumerate(zip(xs, labels)):
        fc = "#7A8DA8" if i == 3 else BLUE
        box(ax, (x, 0.18), 2.15, 0.88, t, fc, fs=9)
        if i < 3:
            arrow(ax, (x + 2.15, 0.62), (xs[i + 1], 0.62), BLUE)
    save(fig, "aging_flow.png")


if __name__ == "__main__":
    fig_framework()
    fig_aging()
    from PIL import Image
    for name in ["framework.png", "aging_flow.png"]:
        im = Image.open(OUT / name)
        print(name, im.size, round(im.width / im.height, 3))
