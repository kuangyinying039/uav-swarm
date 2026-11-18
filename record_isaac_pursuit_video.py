"""Record an Isaac PhysX MAPPO/MPC capture episode.

Primary path (same as Oct20 coop videos):
  live PhysX rollout → pose npz → pose-replay RGB export → MP4.

Also tries live Replicator RGB during the rollout; if that yields too few
frames (common headless LdrColor issue), export still builds the MP4 from npz.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from artifact_paths import artifact_path
from evaluate_isaac_pursuit import load_policy
from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("apf", "frpn", "mpc", "nmpc", "mappo", "matd3"), default="mappo")
    parser.add_argument("--checkpoint", type=artifact_path, default=None)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_hard_physx_closed.json"),
    )
    parser.add_argument("--seed", type=int, default=7_000_110)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--physx-closed-loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pure PhysX 3+1 closed loop (default on for formal MAPPO replay).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "mappo_capture.mp4",
    )
    parser.add_argument(
        "--traj-out",
        type=Path,
        default=None,
        help="Pose npz path (default: same stem as --out with .npz).",
    )
    parser.add_argument(
        "--skip-live-rgb",
        action="store_true",
        help="Do not attach live Replicator RGB (still writes traj for export).",
    )
    return parser


def _attach_rgb(env, width: int, height: int):
    import omni.isaac.core.utils.prims as prim_utils
    import omni.replicator.core as rep
    from omni_drones.sensors.camera import orientation_from_view

    mid = float(env.cfg.grid_size) * 0.5
    alt = float(env.cfg.initial_altitude)
    eye = (mid + 22.0, mid - 22.0, 18.0)
    target = (mid, mid, alt)
    cam_path = "/World/RecordCamera"
    if prim_utils.is_prim_path_valid(cam_path):
        prim_utils.delete_prim(cam_path)
    prim_utils.create_prim(
        cam_path,
        prim_type="Camera",
        translation=list(eye),
        orientation=orientation_from_view(eye, target),
    )
    render_product = rep.create.render_product(cam_path, (width, height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([render_product])
    for _ in range(2):
        try:
            env.sim_env.sim_context.render()
        except Exception:
            break
    return rgb


def _save_frame(annotator, path: Path) -> bool:
    data = annotator.get_data()
    if data is None:
        return False
    arr = np.asarray(data)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return False
    frame = arr[:, :, :3]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except ImportError:
        import imageio

        imageio.imwrite(path, frame)
        return True
    Image.fromarray(frame).save(path)
    return True


def _physical_or_bridge(env):
    physical = np.asarray(env._physical_state(0), dtype=np.float64)
    if np.isfinite(physical).all():
        return physical.reshape(env.cfg.n_uavs, 13).copy(), True
    bridge = env.bridge.env
    return np.asarray(bridge.quadrotor_states, dtype=np.float64).copy(), False


def main():
    args = build_parser().parse_args()
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        raise SystemExit("--checkpoint required for mappo/matd3")
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    traj_out = args.traj_out or out.with_suffix(".npz")
    frames_dir = out.parent / f"{out.stem}_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[record] method={args.method} seed={args.seed} out={out} "
        f"res={args.width}x{args.height} physx_closed={args.physx_closed_loop}",
        flush=True,
    )
    env = IsaacPursuitEnv(
        cfg,
        seed=args.seed,
        headless=True,
        native_lidar=False,
        physx_closed_loop=bool(args.physx_closed_loop),
    )
    saved = 0
    result = None
    frames_pursuer = []
    frames_target = []
    buildings = None
    building_heights = None
    captured = False
    steps = 0
    try:
        observation = env.reset(seed=args.seed)
        env.step_joint(np.zeros((cfg.n_uavs, 4), dtype=np.float32))
        rgb = None
        if not args.skip_live_rgb:
            rgb = _attach_rgb(env, args.width, args.height)
            for _ in range(3):
                try:
                    env.sim_env.sim_context.render()
                except Exception:
                    pass
                time.sleep(0.05)

        action_fn, policy = load_policy(args.method, args.checkpoint, env, args.device)
        observation = env.reset(seed=args.seed)
        if hasattr(policy, "reset"):
            policy.reset(env)
        bridge = env.bridge.env
        buildings = np.asarray(bridge.buildings, dtype=np.float64).copy()
        building_heights = np.asarray(bridge.building_heights, dtype=np.float64).copy()
        limit = min(int(cfg.search_steps), int(args.max_steps))
        for step in range(limit):
            action = np.asarray(action_fn(observation), dtype=np.float32)
            result = env.step_joint(action)
            observation = result["obs"]
            steps = step + 1
            pursuer, _ok = _physical_or_bridge(env)
            target = np.asarray(env._target_position(0), dtype=np.float64).reshape(3)
            frames_pursuer.append(pursuer)
            frames_target.append(target.copy())
            if rgb is not None:
                try:
                    env.sim_env.sim_context.render()
                except Exception:
                    pass
                frame_path = frames_dir / f"frame_{saved:04d}.png"
                if _save_frame(rgb, frame_path):
                    saved += 1
            if step % 10 == 0 or result["terminated"] or result["truncated"]:
                print(
                    f"[record] step={step + 1}/{limit} frames={saved} "
                    f"gap={float(result.get('minimum_capture_gap', -1)):.2f} "
                    f"capture={bool(result.get('capture_success'))}",
                    flush=True,
                )
            if result["terminated"] or result["truncated"]:
                captured = bool(result.get("capture_success"))
                break
        print(
            f"[record] done capture={captured} frames={saved} steps={steps}",
            flush=True,
        )
        if hasattr(policy, "close"):
            policy.close()
    finally:
        # Persist poses before Kit teardown (shutdown can SIGSEGV).
        if frames_pursuer:
            meta = {
                "method": args.method,
                "evader_policy": str(cfg.evader_policy),
                "env_config": str(args.env_config),
                "backend": "isaac_physx_closed" if args.physx_closed_loop else "isaac_physx",
                "seed": int(args.seed),
                "captured": bool(captured),
                "steps": int(steps),
                "note": "Live PhysX rollout poses for pose-replay MP4 export.",
            }
            traj_out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                traj_out,
                pursuer=np.stack(frames_pursuer, axis=0),
                target=np.stack(frames_target, axis=0),
                buildings=buildings if buildings is not None else np.zeros((0, 4)),
                building_heights=building_heights
                if building_heights is not None
                else np.zeros((0,)),
                meta=json.dumps(meta),
            )
            traj_out.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[record] wrote traj {traj_out} cap={captured} steps={steps}", flush=True)
        try:
            env.close()
        except Exception as exc:
            print(f"[record] env.close warning: {exc}", flush=True)

    if saved >= 2:
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ]
        print(f"[record] ffmpeg {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[record] wrote {out} ({out.stat().st_size} bytes)", flush=True)
        return

    if not captured or not traj_out.exists():
        raise SystemExit(
            f"no MP4: live RGB frames={saved}, capture={captured}, traj={traj_out.exists()}"
        )
    print(
        f"[record] live RGB unavailable (frames={saved}); traj ready for export: {traj_out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
