#!/usr/bin/env python3
"""Render a small set of clean Times/STIX equation PNGs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

OUT = Path("/workspace/presentations/eq")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "text.color": "#1F2A44",
})

EQS = {
    "motion": r"$p_{i,t+1}=\Pi_{\Omega}(p_{i,t}+v\Delta t_t\, d(a_{i,t}))$",
    "link": r"$A_{i,j,t}=1(\Vert p_{i,t}-p_{j,t}\Vert \leq R_c)\,z_{i,t}z_{j,t}u_{i,t}u_{j,t}$",
    "dt": r"$D_T=|\mathcal{T}_{found}|/N_{tgt}$",
    "ct": r"$C_T=\frac{1}{|\Omega_g|}\sum_{x\in\Omega_g}[\mathbf{1}(P_t(x)\leq p_{\mathrm{low}})+\mathbf{1}(P_t(x)\geq p_{\mathrm{high}})]$",
    "plim": r"$p_{low}=0.05,\quad p_{high}=0.95$",
    "t80": r"$T_{80}^{step}\ /\ T_{80}^{time}$",
    "jstar": r"$j^{\star}=\arg\max_j\ w_{i,j,t}\,|L_{j,t}(x)|$",
    "w": r"$w_{i,j,t}=e^{-\beta_d\Vert p_i-p_j\Vert}/(1+g_{j,t})$",
    "att": r"$\alpha_{r,i,k}=\mathrm{softmax}(Q_iK_k^{\top}/\sqrt{d}+\log\chi_{r,i,k}-\kappa_r d_{ik})$",
    "reward": r"$r_t=\lambda_D\Delta D_t+\lambda_C\Delta C_t-\lambda_{\mathrm{safe}}c_t$",
    "clip": r"$L^{\mathrm{CLIP}}=\mathbb{E}[\min(\rho\hat A,\ \mathrm{clip}(\rho,1-\varepsilon,1+\varepsilon)\hat A)]$",
}


def render(name: str, tex: str, fontsize: float) -> Path:
    fig = plt.figure(figsize=(15.0, 1.15))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    ax.text(0.5, 0.5, tex, ha="center", va="center", fontsize=fontsize)
    tmp = OUT / f"_{name}.png"
    fig.savefig(tmp, dpi=240, transparent=True)
    plt.close(fig)
    im = Image.open(tmp).convert("RGBA")
    arr = np.array(im)
    ys, xs = np.where(arr[:, :, 3] > 15)
    pad = 14
    crop = im.crop((
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + pad, im.width),
        min(int(ys.max()) + pad, im.height),
    ))
    path = OUT / f"{name}.png"
    crop.save(path)
    tmp.unlink(missing_ok=True)
    return path


def main():
    ar = {}
    for name, tex in EQS.items():
        fs = 18 if name in {"ct", "link", "att", "clip"} else 20
        p = render(name, tex, fs)
        im = Image.open(p)
        ar[f"{name}.png"] = im.width / im.height
        print(name, im.size)
    (OUT / "ar.json").write_text(json.dumps(ar, indent=2))


if __name__ == "__main__":
    main()
