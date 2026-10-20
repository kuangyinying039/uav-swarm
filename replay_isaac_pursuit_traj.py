"""Isaac Viewport pose-replay of a recorded pursuit trajectory.

Loads ``record_pursuit_trajectory.py`` output and teleports Hummingbirds + target
each step. Does NOT run Lee velocity / live PhysX control, so flash-quit and
gap=25 corruption from online control are avoided.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from isaac_cli_flags import add_isaac_display_flags, resolve_headless
from isaac_pursuit_backend import IsaacPursuitEnv
from isaac_pursuit_viewport_overlay import PursuitViewportOverlay
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "pursuit_traj.npz",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_medium.json"),
    )
    parser.add_argument("--speed", type=float, default=1.4, help=">1 slows playback")
    parser.add_argument("--trail-len", type=int, default=120)
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--camera-view",
        choices=("side", "side_left", "oblique", "high", "top"),
        default="side",
        help="Chase-camera preset: side / top (overhead) / high / oblique",
    )
    parser.add_argument(
        "--visual-scale",
        type=float,
        default=4.0,
        help="GUI-only Hummingbird mesh scale (physics unchanged; 1=native)",
    )
    parser.add_argument(
        "--body-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw wireframe spheres on UAV/target (default off when meshes are enlarged)",
    )
    add_isaac_display_flags(parser, default_headless=True)
    return parser


def main():
    args = build_parser().parse_args()
    if not args.traj.is_file():
        raise SystemExit(f"trajectory not found: {args.traj}")
    payload = np.load(args.traj, allow_pickle=False)
    pursuer = np.asarray(payload["pursuer"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    buildings = np.asarray(payload["buildings"], dtype=np.float64)
    building_heights = np.asarray(payload["building_heights"], dtype=np.float64)
    meta = json.loads(str(payload["meta"]))
    seed = int(meta["seed"])
    steps = int(pursuer.shape[0])

    headless = resolve_headless(args)
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    cfg = replace(cfg, seed=seed, evader_policy=str(meta.get("evader_policy", cfg.evader_policy)))
    print(
        f"[replay] traj={args.traj} seed={seed} steps={steps} "
        f"cap={meta.get('captured')} follow={args.follow_camera} "
        f"camera={args.camera_view} visual_scale={args.visual_scale} "
        f"body_markers={args.body_markers} visualize={not headless}",
        flush=True,
    )
    env = IsaacPursuitEnv(
        cfg,
        seed=seed,
        headless=headless,
        visual_model_scale=float(args.visual_scale) if not headless else 1.0,
    )
    overlay = PursuitViewportOverlay(
        env,
        trail_len=args.trail_len,
        enabled=not headless,
        # Target capture wireframe on; no UAV spheres; keep thin trails.
        show_radar_rings=False,
        show_target_wireframe=True,
        show_body_markers=bool(args.body_markers) and not headless,
        follow_camera=bool(args.follow_camera) and not headless,
        camera_view=str(args.camera_view),
    )
    try:
        env.reset(seed=seed)
        # Force buildings to the recorded layout (same seed should match, but pin it).
        bridge = env.bridge.env
        bridge.buildings = buildings.copy()
        bridge.building_heights = building_heights.copy()
        env._set_building_transforms(0)
        overlay.reset()

        dt = float(cfg.decision_dt) * float(args.speed)
        for step in range(steps):
            started = time.perf_counter()
            env.apply_replay_frame(pursuer[step], target[step])
            # Synthetic result for HUD / trails.
            active = ~np.asarray(bridge.disabled_uavs, dtype=bool)
            dists = np.linalg.norm(pursuer[step, active, :3] - target[step][None, :], axis=1)
            required = int(cfg.capture_required_uavs)
            if len(dists) >= required:
                gap = float(max(np.partition(dists, required - 1)[required - 1] - cfg.capture_radius, 0.0))
            else:
                gap = float(cfg.grid_size)
            captured = bool(step == steps - 1 and meta.get("captured"))
            result = {
                "minimum_capture_gap": gap,
                "capture_success": captured,
                "terminated": captured,
                "truncated": False,
                "reward": 0.0,
                "reward_components": {},
            }
            hud = overlay.update(result)
            if step % 10 == 0 or step == steps - 1:
                print(f"[replay] step={step}/{steps - 1} {hud}", flush=True)
            elapsed = time.perf_counter() - started
            if elapsed < dt:
                time.sleep(dt - elapsed)
        print(
            f"[replay] done seed={seed} steps={steps} capture={bool(meta.get('captured'))}",
            flush=True,
        )
        if not headless:
            # Hold the final frame so the user can inspect / screenshot.
            time.sleep(12.0)
    finally:
        env.close()


if __name__ == "__main__":
    main()
