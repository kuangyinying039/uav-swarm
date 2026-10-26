#!/usr/bin/env python
"""Probe pure physx_closed: Lee evader stability + native LiDAR readability + FRPN."""
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
        native_lidar=True,
        physx_closed_loop=True,
    )
    assert env._physx_closed_loop and not env._kinematic_evader
    assert env._native_lidar_enabled
    print(
        f"[probe] kinematic_evader={env._kinematic_evader} "
        f"native_lidar={env._native_lidar_enabled} "
        f"sensor_paths={sum(1 for p in env.sensor_paths if p)}",
        flush=True,
    )

    # Phase A: constant-velocity Lee track on evader only (pursuers hover).
    env.reset(seed=2000000)
    max_espd = 0.0
    lidar_ok = 0
    lidar_hits = 0
    for t in range(100):
        actions = np.zeros((cfg.n_uavs, 4), dtype=np.float32)
        # Force a constant world-x command on the PhysX evader via desired policy path
        # by stepping with hover pursuers; evader uses occlusion policy internally.
        result = env.step_joint(actions)
        ev = env._evader_state_batch()[0]
        if np.isfinite(ev).all():
            max_espd = max(max_espd, float(np.linalg.norm(ev[3:6])))
        scan = env._last_native_scan
        if scan is not None:
            if scan.get("sensor_available"):
                lidar_ok += 1
            if scan.get("visible") is not None and np.any(scan["visible"]):
                lidar_hits += 1
        if t in (0, 9, 49, 99):
            n_pts = int(scan.get("n_points", -1)) if scan else -1
            n_match = int(scan.get("n_matched", -1)) if scan else -1
            sample = (scan.get("sample_prims") or [])[:4] if scan else []
            print(
                f"[lee] t={t} espd={float(np.linalg.norm(ev[3:6])) if np.isfinite(ev).all() else float('nan'):.3f} "
                f"epos={np.round(ev[:3], 2) if np.isfinite(ev).all() else None} "
                f"lidar_avail={bool(scan and scan.get('sensor_available'))} "
                f"hits={int(np.count_nonzero(scan['visible'])) if scan and scan.get('visible') is not None else 0} "
                f"n_pts={n_pts} n_match={n_match} prims={sample}",
                flush=True,
            )
    print(f"[lee] DONE max_espd={max_espd:.3f} lidar_ok={lidar_ok}/100 lidar_hits={lidar_hits}/100", flush=True)

    # Phase B: short FRPN capture probe
    policy = BASELINES_3D["frpn"]()
    n = 8
    caps = 0
    gaps = []
    espds = []
    for i in range(n):
        seed = 2000000 + i
        env.reset(seed=seed)
        captured = False
        min_gap = 1e9
        max_e = 0.0
        tracks_on = 0
        vis_on = 0
        for t in range(cfg.search_steps):
            be = env.bridge.env
            n_tracks = sum(1 for a in range(cfg.n_uavs) if be.track_memory[a][0].initialized)
            vis = int(np.count_nonzero(be.direct_visibility_mask()))
            tracks_on = max(tracks_on, n_tracks)
            vis_on = max(vis_on, vis)
            action = policy.actions(be)
            result = env.step_joint(np.asarray(action, dtype=np.float32))
            gap = float(env.minimum_capture_gap(0))
            min_gap = min(min_gap, gap)
            ev = env._evader_state_batch()[0]
            if np.isfinite(ev).all():
                max_e = max(max_e, float(np.linalg.norm(ev[3:6])))
            if t in (0, 24, 99) and i == 0:
                mode = (env.bridge.last_sensor_diagnostics or {}).get("sensor_mode")
                print(
                    f"[frpn-dbg] t={t} tracks={n_tracks} vis={vis} mode={mode} "
                    f"gap={gap:.3f} pursuer0={np.round(be.quadrotor_states[0, :3], 2)}",
                    flush=True,
                )
            if result.get("terminated") or result.get("truncated"):
                captured = bool(result.get("capture_success"))
                break
        caps += int(captured)
        gaps.append(min_gap)
        espds.append(max_e)
        print(
            f"[frpn] ep={i + 1}/{n} seed={seed} cap={int(captured)} "
            f"min_gap={min_gap:.3f} max_espd={max_e:.3f} "
            f"max_tracks={tracks_on} max_vis={vis_on}",
            flush=True,
        )
    print(
        f"[frpn] DONE rate={caps / n:.3f} caps={caps}/{n} "
        f"mean_min_gap={np.mean(gaps):.3f} mean_max_espd={np.mean(espds):.3f}",
        flush=True,
    )
    env.close()
    print("PROBE_DONE", flush=True)


if __name__ == "__main__":
    main()
