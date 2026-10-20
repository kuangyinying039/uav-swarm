"""Short Isaac Viewport playback for dual-capture pursuit.

Opens the Kit window when ``--visualize`` is set (requires DISPLAY). Prefer this
entrypoint over full training for demos and screen capture. Suite training stays
headless.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
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
    parser.add_argument("--trail-len", type=int, default=80)
    parser.add_argument(
        "--evader-policy",
        choices=("occlusion", "repulsive"),
        default=None,
        help="Override env-config evader (repulsive flees instead of camping walls)",
    )
    parser.add_argument(
        "--show-radar-rings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw lidar detection spheres on each UAV (same style as capture sphere)",
    )
    parser.add_argument(
        "--follow-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-frame pursuer/target each step (default on; use --no-follow-camera for free look)",
    )
    parser.add_argument(
        "--visual-scale",
        type=float,
        default=4.0,
        help="GUI-only Hummingbird mesh scale (physics unchanged)",
    )
    parser.add_argument(
        "--body-markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw wireframe spheres on UAV/target (default off)",
    )
    add_isaac_display_flags(parser, default_headless=True)
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
    if args.evader_policy is not None:
        cfg = replace(cfg, evader_policy=str(args.evader_policy))

    print(
        f"[play] method={args.method} seed={args.seed} headless={headless} "
        f"visualize={not headless} follow_camera={bool(args.follow_camera)} "
        f"evader={cfg.evader_policy}",
        flush=True,
    )
    env = IsaacPursuitEnv(
        cfg,
        seed=args.seed,
        headless=headless,
        native_lidar=args.native_lidar,
        visual_model_scale=float(args.visual_scale) if not headless else 1.0,
    )
    overlay = PursuitViewportOverlay(
        env,
        trail_len=args.trail_len,
        enabled=not headless,
        show_radar_rings=bool(args.show_radar_rings) and not headless,
        show_target_wireframe=True,
        show_body_markers=bool(args.body_markers) and not headless,
        follow_camera=bool(args.follow_camera) and not headless,
    )
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
