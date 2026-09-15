#!/usr/bin/env python3
"""Render MathType-style equation PNGs (Times/STIX math) for the CAC 2026 deck."""
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
    "dec": r"$\mathcal{G}=\langle \mathcal{N},\mathcal{S},\{\mathcal{A}_i\},P,R,\{\mathcal{O}_i\},\gamma\rangle$",
    "ni": r"$\mathcal{N}_i(t)=\{ j\neq i\ |\ A_{i,j,t}=1\}$",
    "motion": r"$p_{i,t+1}=\Pi_{\Omega}(p_{i,t}+v\,\Delta t_t\, d(a_{i,t}))$",
    "link": r"$A_{i,j,t}=1(\Vert p_{i,t}-p_{j,t}\Vert \leq R_c)\ z_{i,t}\, z_{j,t}\, u_{i,t}\, u_{j,t}$",
    "dt": r"$D_t=\vert\mathcal{T}_{found}\vert / N_{tgt}$",
    "ct": r"$C_t=\mathrm{mean}_x\,\mathrm{abs}(2P_t(x)-1)$",
    "t80": r"$T_{80}=\min\{ t\ |\ D_t\geq 0.8\}$",
    "L": r"$L_{i,t}(x)=L_{i,t-1}(x)+\ell_{i,t}(x)-L_0$",
    "P": r"$P_{i,t}(x)=(1+\exp(-L_{i,t}(x)))^{-1}$",
    "E": r"$E_{i,t}(x)=\exp(-k_q\,\mathrm{abs}(L_{i,t}(x)))$",
    "Rfield": r"$R_{i,t}(x)=(1-\alpha)R_{i,t-1}(x)+\alpha+\beta\,\Delta R_{i,t}(x)$",
    "front": r"$q(x)=E(x)+0.15\,P(x)+0.65\,R(x)$",
    "w": r"$w_{i,j,t}=e^{-\beta_d \|p_i-p_j\|}/(1+g_{j,t})$",
    "jstar": r"$j^{\star}=\arg\max_{j}\ w_{i,j,t}\,\mathrm{abs}(L_{j,t}(x))$",
    "att": r"$\alpha_{r,i,k}=\mathrm{softmax}(Q_i K_k^{\top}/\sqrt{d}+\log\chi_{r,i,k}-\kappa_r d_{ik})$",
    "reward": r"$r_t=\lambda_D \Delta D_t+\lambda_C \Delta C_t-\lambda_{safe} c_t$",
    "rho": r"$\rho_t(\theta)=\pi_{\theta}(a_t|o_t)/\pi_{\theta_{old}}(a_t|o_t)$",
    "delta": r"$\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$",
    "clip": r"$L^{CLIP}(\theta)=\mathbb{E}_t[\min(\rho_t \hat{A}_t,\ \mathrm{clip}(\rho_t,1-\varepsilon,1+\varepsilon)\hat{A}_t)]$",
    "gae": r"$\hat{A}_t=\sum_{l=0}^{H}(\gamma\lambda)^{l}\delta_{t+l}$",
}


def render(name: str, tex: str, fontsize: float = 22) -> Path:
    fig = plt.figure(figsize=(14.5, 1.05))
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
    pad = 12
    box = (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + pad, im.width),
        min(int(ys.max()) + pad, im.height),
    )
    crop = im.crop(box)
    path = OUT / f"{name}.png"
    crop.save(path)
    tmp.unlink(missing_ok=True)
    return path


def main():
    ar = {}
    for name, tex in EQS.items():
        fs = 18 if name in {"link", "att", "clip"} else 21
        p = render(name, tex, fs)
        im = Image.open(p)
        ar[f"{name}.png"] = im.width / im.height
        print(name, im.size)
    (OUT / "ar.json").write_text(json.dumps(ar, indent=2))
    print("wrote", len(ar), "equations")


if __name__ == "__main__":
    main()
