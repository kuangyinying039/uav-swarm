#!/usr/bin/env python
"""FRPN on physx_closed with kinematic_evader forced on (debug comparator)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def main():
    cfg = QuadrotorPursuitConfig(
        **json.loads(Path("configs/pursuit_v2/radar5_dual_hard_physx_closed.json").read_text())
    )
    env = IsaacPursuitEnv(
        cfg, seed=2000000, headless=True, native_lidar=False, physx_closed_loop=True
    )
    env._kinematic_evader = True
    print(f"kinematic_evader={env._kinematic_evader}", flush=True)
    policy = BASELINES_3D["frpn"]()
    n = 12
    caps = 0
    gaps = []
    for i in range(n):
        seed = 2000000 + i
        env.reset(seed=seed)
        policy.reset(env.bridge.env)
        captured = False
        min_gap = 1e9
        for _ in range(cfg.search_steps):
            result = env.step_joint(
                np.asarray(policy.actions(env.bridge.env), dtype=np.float32)
            )
            min_gap = min(min_gap, float(env.minimum_capture_gap(0)))
            if result.get("terminated") or result.get("truncated"):
                captured = bool(result.get("capture_success"))
                break
        caps += int(captured)
        gaps.append(min_gap)
        print(
            f"[frpn] ep={i + 1}/{n} cap={int(captured)} min_gap={min_gap:.3f}",
            flush=True,
        )
    print(
        f"[frpn] DONE rate={caps / n:.3f} caps={caps}/{n} "
        f"mean_min_gap={np.mean(gaps):.3f}",
        flush=True,
    )
    env.close()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
