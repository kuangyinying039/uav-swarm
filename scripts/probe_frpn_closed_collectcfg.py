#!/usr/bin/env python
import json
from pathlib import Path

import numpy as np

from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def main():
    cfg = QuadrotorPursuitConfig(
        **json.loads(
            Path("configs/pursuit_v2/radar5_dual_hard_physx_closed_collect.json").read_text()
        )
    )
    env = IsaacPursuitEnv(
        cfg, seed=2000000, headless=True, native_lidar=False, physx_closed_loop=True
    )
    policy = BASELINES_3D["frpn"]()
    n = 12
    caps = 0
    gaps = []
    try:
        for i in range(n):
            seed = 2000000 + i
            env.reset(seed=seed)
            captured = False
            steps = 0
            min_gap = 1e9
            for t in range(cfg.search_steps):
                action = policy.actions(env.bridge.env)
                result = env.step_joint(np.asarray(action, dtype=np.float32))
                steps = t + 1
                gap = float(env.bridge.env.minimum_capture_gap())
                min_gap = min(min_gap, gap)
                if result.get("terminated") or result.get("truncated"):
                    captured = bool(result.get("capture_success"))
                    break
            caps += int(captured)
            gaps.append(min_gap)
            print(
                f"[probe] ep={i + 1}/{n} seed={seed} cap={int(captured)} "
                f"steps={steps} min_gap={min_gap:.3f}",
                flush=True,
            )
        print(
            f"[probe] DONE rate={caps / n:.3f} caps={caps}/{n} "
            f"mean_min_gap={np.mean(gaps):.3f}",
            flush=True,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
