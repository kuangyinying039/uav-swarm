"""Record an Isaac PhysX MAPPO/MPC episode to MP4 via headless viewport RGB.

Does not need a local DISPLAY: uses Omniverse Replicator on the Kit perspective
camera, then ffmpeg. Output is real PhysX geometry (buildings, drones), not the
Radar5 HTML schematic player.
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
    parser.add_argument("--method", choices=("apf", "frpn", "mpc", "mappo", "matd3"), default="mappo")
    parser.add_argument("--checkpoint", type=artifact_path, default=None)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("configs/pursuit_v2/radar5_dual_hard.json"),
    )
    parser.add_argument("--seed", type=int, default=7_000_011)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_physx_replay" / "mappo_ep.mp4",
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
    # Warm the annotator pipeline.
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
    # Drop alpha if present; ensure uint8 RGB.
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


def main():
    args = build_parser().parse_args()
    if args.method in ("mappo", "matd3") and args.checkpoint is None:
        raise SystemExit("--checkpoint required for mappo/matd3")
    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / f"{out.stem}_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[record] method={args.method} seed={args.seed} out={out} "
        f"res={args.width}x{args.height}",
        flush=True,
    )
    env = IsaacPursuitEnv(cfg, seed=args.seed, headless=True, native_lidar=False)
    try:
        # One warm step so the USD stage / viewport exist before Replicator attach.
        observation = env.reset(seed=args.seed)
        env.step_joint(np.zeros((cfg.n_uavs, 4), dtype=np.float32))
        rgb = _attach_rgb(env, args.width, args.height)
        # Flush a few renders so the annotator starts delivering.
        for _ in range(3):
            env.sim_env.sim_context.render() if hasattr(env, "sim_env") else None
            time.sleep(0.05)

        action_fn, policy = load_policy(args.method, args.checkpoint, env, args.device)
        observation = env.reset(seed=args.seed)
        if hasattr(policy, "reset"):
            policy.reset(env)
        limit = min(int(cfg.search_steps), int(args.max_steps))
        saved = 0
        result = None
        for step in range(limit):
            action = np.asarray(action_fn(observation), dtype=np.float32)
            result = env.step_joint(action)
            observation = result["obs"]
            # Extra render for RGB path (PhysX already advanced in step_joint).
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
                break
        print(
            f"[record] done capture={bool(result and result.get('capture_success'))} "
            f"frames={saved}",
            flush=True,
        )
        if hasattr(policy, "close"):
            policy.close()
    finally:
        env.close()

    if saved < 2:
        raise SystemExit(f"too few frames saved ({saved}); RGB annotator may be unavailable")
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


if __name__ == "__main__":
    main()
