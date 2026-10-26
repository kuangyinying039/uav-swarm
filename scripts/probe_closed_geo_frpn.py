#!/usr/bin/env python
"""Pure closed Lee + geometric LiDAR FRPN probe (isolates sensing from PhysX lidar)."""
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
        cfg,
        seed=2000000,
        headless=True,
        native_lidar=False,
        physx_closed_loop=True,
    )
    print(
        f"[probe] kinematic_evader={env._kinematic_evader} "
        f"native_lidar={env._native_lidar_enabled}",
        flush=True,
    )
    n = 8
    for name in ("frpn", "mpc"):
        policy = BASELINES_3D[name]()
        caps = 0
        gaps = []
        for i in range(n):
            seed = 2000000 + i
            env.reset(seed=seed)
            be = env.bridge.env
            tracks0 = sum(1 for a in range(cfg.n_uavs) if be.track_memory[a][0].initialized)
            vis0 = int(np.count_nonzero(be.direct_visibility_mask()))
            if i == 0:
                print(
                    f"[reset] policy={name} tracks={tracks0} vis={vis0} "
                    f"tgt={np.round([*be.dynamic_targets[0], be.target_altitudes[0]], 2)} "
                    f"p0={np.round(be.quadrotor_states[0, :3], 2)}",
                    flush=True,
                )
            policy.reset(be)
            captured = False
            min_gap = 1e9
            max_e = 0.0
            max_tracks = tracks0
            for t in range(cfg.search_steps):
                be = env.bridge.env
                n_tracks = sum(
                    1 for a in range(cfg.n_uavs) if be.track_memory[a][0].initialized
                )
                max_tracks = max(max_tracks, n_tracks)
                action = policy.actions(be)
                result = env.step_joint(np.asarray(action, dtype=np.float32))
                gap = float(env.minimum_capture_gap(0))
                min_gap = min(min_gap, gap)
                ev = env._evader_state_batch()[0]
                if np.isfinite(ev).all():
                    max_e = max(max_e, float(np.linalg.norm(ev[3:6])))
                if t in (0, 49) and i == 0:
                    print(
                        f"[{name}-dbg] t={t} tracks={n_tracks} gap={gap:.3f} "
                        f"act={float(np.linalg.norm(action[:, :3])):.3f} "
                        f"p0z={be.quadrotor_states[0, 2]:.2f} "
                        f"ez={float(ev[2]) if np.isfinite(ev).all() else float('nan'):.2f}",
                        flush=True,
                    )
                if result.get("terminated") or result.get("truncated"):
                    captured = bool(result.get("capture_success"))
                    break
            caps += int(captured)
            gaps.append(min_gap)
            print(
                f"[{name}] ep={i+1}/{n} seed={seed} cap={int(captured)} "
                f"min_gap={min_gap:.3f} max_espd={max_e:.3f} max_tracks={max_tracks}",
                flush=True,
            )
        print(
            f"[{name}] DONE rate={caps/n:.3f} caps={caps}/{n} "
            f"mean_min_gap={np.mean(gaps):.3f}",
            flush=True,
        )
    env.close()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
