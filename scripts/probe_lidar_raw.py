#!/usr/bin/env python
"""Dump raw PhysX LiDAR buffers for one closed-loop reset (diagnose empty clouds)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def main():
    cfg = QuadrotorPursuitConfig(
        **json.loads(Path("configs/pursuit_v2/radar5_dual_hard_physx_closed.json").read_text())
    )
    env = IsaacPursuitEnv(
        cfg, seed=2000000, headless=True, native_lidar=True, physx_closed_loop=True
    )
    env.reset(seed=2000000)
    iface = env.lidar_interface
    print(f"[raw] sensor_paths={env.sensor_paths}", flush=True)
    print(f"[raw] target_paths={getattr(env, '_target_paths', None)}", flush=True)
    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        for path in list(env.sensor_paths) + list(getattr(env, "_target_paths", [])):
            if not path:
                continue
            prim = stage.GetPrimAtPath(path)
            print(
                f"[raw] prim {path} valid={bool(prim and prim.IsValid())} "
                f"type={prim.GetTypeName() if prim else None}",
                flush=True,
            )
        ground = stage.GetPrimAtPath("/World/envs/env_0/radar5_ground")
        print(
            f"[raw] ground valid={bool(ground and ground.IsValid())} "
            f"collision={bool(ground and ground.HasAPI(UsdPhysics.CollisionAPI))}",
            flush=True,
        )
    except Exception as exc:
        print(f"[raw] stage inspect failed: {exc}", flush=True)

    for step in range(10):
        env.step_joint(np.zeros((cfg.n_uavs, 4), dtype=np.float32))
        for path in env.sensor_paths:
            if not path:
                continue
            try:
                alive = iface.is_lidar_sensor(path) if hasattr(iface, "is_lidar_sensor") else "?"
            except Exception as exc:
                alive = f"err:{exc}"
            try:
                data = iface.get_point_cloud_data(path)
                arr = np.asarray(data, dtype=float)
            except Exception as exc:
                arr = None
                print(f"[raw] t={step} {path} read_fail={exc}", flush=True)
                continue
            try:
                prims = list(iface.get_prim_data(path))
            except Exception as exc:
                prims = [f"err:{exc}"]
            depth = None
            if hasattr(iface, "get_linear_depth_data"):
                try:
                    depth = np.asarray(iface.get_linear_depth_data(path), dtype=float)
                except Exception:
                    depth = None
            print(
                f"[raw] t={step} path={path} alive={alive} "
                f"pts_shape={None if arr is None else arr.shape} "
                f"pts_finite={0 if arr is None else int(np.isfinite(arr).sum())} "
                f"n_prims={len(prims)} depth_shape={None if depth is None else depth.shape} "
                f"sample_prims={prims[:5]}",
                flush=True,
            )
    env.close()
    print("RAW_DONE", flush=True)


if __name__ == "__main__":
    main()
