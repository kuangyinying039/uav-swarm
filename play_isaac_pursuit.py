"""Short Isaac Viewport playback for dual-capture pursuit.

Opens the Kit window when ``--visualize`` is set (requires DISPLAY). Prefer this
entrypoint over full training for demos and screen capture. Suite training stays
headless.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from artifact_paths import artifact_path
from evaluate_isaac_pursuit import load_policy
from isaac_cli_flags import add_isaac_display_flags, resolve_headless
from isaac_pursuit_backend import IsaacPursuitEnv
from isaac_pursuit_viewport_overlay import PursuitViewportOverlay
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=("apf", "frpn", "mpc", "mappo", "matd3"),
        default="mpc",
    )
    parser.add_argument("--checkpoint", type=artifact_path, default=None)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_medium.json"),
    )
    parser.add_argument("--seed", type=int, default=7_000_000)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 = full horizon")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback slowdown (>1 slower)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--native-lidar", action="store_true")
    parser.add_argument("--trail-len", type=int, default=40)
    add_isaac_display_flags(parser, default_headless=True)
    # Default this script toward visualization when flag set; without --visualize stays headless.
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.episodes < 1 or args.speed <= 0:
        parser.error("--episodes must be >=1 and --speed > 0")
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        parser.error("--checkpoint is required for learned methods")
    if args.method in ("apf", "frpn", "mpc") and args.checkpoint is not None:
        parser.error("baseline methods do not accept --checkpoint")

    headless = resolve_headless(args)
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    if cfg.capture_required_uavs != 2 or cfg.execution_reward_version != 5:
        raise ValueError(f"{args.env_config} is not dual-capture reward-v5")

    print(
        f"[play] method={args.method} seed={args.seed} headless={headless} "
        f"visualize={not headless}",
        flush=True,
    )
    env = IsaacPursuitEnv(
        cfg,
        seed=args.seed,
        headless=headless,
        native_lidar=args.native_lidar,
    )
    overlay = PursuitViewportOverlay(
        env,
        trail_len=args.trail_len,
        enabled=not headless,
    )
    if not headless:
        try:
            from omni.isaac.core.utils.viewports import set_camera_view

            mid = float(cfg.grid_size) * 0.5
            set_camera_view(
                eye=[mid + 18.0, mid - 18.0, 16.0],
                target=[mid, mid, float(cfg.initial_altitude)],
            )
        except Exception as exc:
            print(f"[play] camera view skipped: {exc}", flush=True)
    try:
        action_fn, policy = load_policy(args.method, args.checkpoint, env, args.device)
        for episode in range(args.episodes):
            seed = int(args.seed) + episode
            observation = env.reset(seed=seed)
            overlay.reset()
            if hasattr(policy, "reset"):
                policy.reset(env)
            limit = min(env.cfg.search_steps, args.max_steps) if args.max_steps else env.cfg.search_steps
            result = None
            for step in range(limit):
                started = time.perf_counter()
                action = np.asarray(action_fn(observation), dtype=np.float32)
                result = env.step_joint(action)
                observation = result["obs"]
                hud = overlay.update(result)
                if step % 10 == 0 or result["terminated"] or result["truncated"]:
                    print(
                        f"[play] ep={episode + 1} step={step + 1} {hud} "
                        f"reward={float(result['reward']):.2f}",
                        flush=True,
                    )
                # Pace wall-clock toward decision_dt * speed when visualizing.
                if not headless:
                    target = float(env.cfg.decision_dt) * float(args.speed)
                    elapsed = time.perf_counter() - started
                    if elapsed < target:
                        time.sleep(target - elapsed)
                if result["terminated"] or result["truncated"]:
                    break
            captured = bool(result and result["capture_success"])
            print(
                f"[play] episode={episode + 1} done capture={captured} steps={step + 1}",
                flush=True,
            )
        if hasattr(policy, "close"):
            policy.close()
    finally:
        overlay.reset()
        env.close()


if __name__ == "__main__":
    main()
