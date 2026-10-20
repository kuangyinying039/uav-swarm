"""Export a recorded pursuit trajectory to MP4 / GIF via Isaac pose-replay + Replicator RGB.

Uses the same pose-only path as ``replay_isaac_pursuit_traj.py`` (no live Lee control).
The RecordCamera tracks the same chase-camera presets as the interactive viewport
(``side`` / ``side_left`` / ``oblique`` / ``high``).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

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
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--gif-fps", type=int, default=6)
    parser.add_argument("--gif-scale", type=int, default=640, help="GIF width in pixels")
    parser.add_argument(
        "--camera-view",
        choices=("side", "side_left", "oblique", "high", "top"),
        default="side",
        help="Same chase-camera presets as interactive replay (default: side)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "pursuit_replay.mp4",
    )
    parser.add_argument("--no-gif", action="store_true")
    return parser


def _attach_rgb(env, width: int, height: int, view: str = "side"):
    import omni.isaac.core.utils.prims as prim_utils
    import omni.replicator.core as rep
    from omni_drones.sensors.camera import orientation_from_view

    pair = env.camera_eye_target(view=view)
    if pair is None:
        g = float(env.cfg.grid_size)
        mid = g * 0.5
        alt = float(env.cfg.initial_altitude)
        frac = IsaacPursuitEnv.CAMERA_PRESET_FRAC.get(
            view, IsaacPursuitEnv.CAMERA_PRESET_FRAC["side"]
        )
        focus = np.array([mid, mid, alt], dtype=float)
        eye = focus + np.asarray((frac[0] * g, frac[1] * g, frac[2] * g), dtype=float)
    else:
        eye, focus = pair
    cam_path = "/World/RecordCamera"
    if prim_utils.is_prim_path_valid(cam_path):
        prim_utils.delete_prim(cam_path)
    prim_utils.create_prim(
        cam_path,
        prim_type="Camera",
        translation=list(map(float, eye)),
        orientation=orientation_from_view(eye, focus),
    )
    render_product = rep.create.render_product(cam_path, (width, height))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach([render_product])
    for _ in range(3):
        try:
            env.sim_env.sim_context.render()
        except Exception:
            break
        time.sleep(0.05)
    return rgb, cam_path


def _aim_record_camera(cam_path: str, eye, target_xyz) -> None:
    """Move the Replicator RecordCamera (viewport set_camera_view alone does not)."""
    try:
        import omni.usd
        from omni_drones.sensors.camera import orientation_from_view
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(cam_path)
        if not prim.IsValid():
            return
        eye_list = [float(x) for x in eye]
        target_list = [float(x) for x in target_xyz]
        orient = orientation_from_view(eye_list, target_list)
        xform = UsdGeom.Xformable(prim)
        # Prefer updating existing ops so the render product stays attached.
        translate_attr = prim.GetAttribute("xformOp:translate")
        orient_attr = prim.GetAttribute("xformOp:orient")
        if translate_attr and orient_attr:
            translate_attr.Set(Gf.Vec3d(*eye_list))
            orient_attr.Set(Gf.Quatd(orient[0], orient[1], orient[2], orient[3]))
            return
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*eye_list))
        xform.AddOrientOp().Set(Gf.Quatd(orient[0], orient[1], orient[2], orient[3]))
    except Exception:
        pass


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

        Image.fromarray(frame).save(path)
    except ImportError:
        import imageio

        imageio.imwrite(path, frame)
    return True


def main():
    args = build_parser().parse_args()
    payload = np.load(args.traj, allow_pickle=False)
    pursuer = np.asarray(payload["pursuer"], dtype=np.float64)
    target = np.asarray(payload["target"], dtype=np.float64)
    buildings = np.asarray(payload["buildings"], dtype=np.float64)
    building_heights = np.asarray(payload["building_heights"], dtype=np.float64)
    meta = json.loads(str(payload["meta"]))
    seed = int(meta["seed"])
    steps = int(pursuer.shape[0])

    cfg = QuadrotorPursuitConfig(**json.loads(args.env_config.read_text(encoding="utf-8")))
    cfg = replace(cfg, seed=seed, evader_policy=str(meta.get("evader_policy", cfg.evader_policy)))
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = out.parent / f"{out.stem}_frames"
    if frames_dir.exists():
        for old in frames_dir.glob("*.png"):
            old.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[export] method={meta.get('method')} seed={seed} steps={steps} "
        f"cap={meta.get('captured')} camera={args.camera_view} out={out}",
        flush=True,
    )
    env = IsaacPursuitEnv(cfg, seed=seed, headless=True)
    overlay = PursuitViewportOverlay(
        env,
        trail_len=120,
        enabled=True,
        show_radar_rings=False,
        follow_camera=False,  # RecordCamera is aimed explicitly below.
        camera_view=str(args.camera_view),
    )
    saved = 0
    try:
        env.reset(seed=seed)
        bridge = env.bridge.env
        bridge.buildings = buildings.copy()
        bridge.building_heights = building_heights.copy()
        env._set_building_transforms(0)
        overlay.reset()
        # Warm one pose so the stage exists, then attach RGB.
        env.apply_replay_frame(pursuer[0], target[0])
        rgb, cam_path = _attach_rgb(env, args.width, args.height, view=args.camera_view)

        for step in range(steps):
            env.apply_replay_frame(pursuer[step], target[step])
            pair = env.camera_eye_target(view=args.camera_view)
            if pair is not None:
                eye, focus = pair
                _aim_record_camera(cam_path, eye, focus)
                try:
                    env.follow_camera(view=args.camera_view)
                except Exception:
                    pass
            result = {
                "minimum_capture_gap": 0.0,
                "capture_success": bool(step == steps - 1 and meta.get("captured")),
                "terminated": False,
                "truncated": False,
            }
            # Approximate gap for HUD trails.
            active = ~np.asarray(bridge.disabled_uavs, dtype=bool)
            dists = np.linalg.norm(pursuer[step, active, :3] - target[step][None, :], axis=1)
            required = int(cfg.capture_required_uavs)
            if len(dists) >= required:
                result["minimum_capture_gap"] = float(
                    max(np.partition(dists, required - 1)[required - 1] - cfg.capture_radius, 0.0)
                )
            overlay.update(result)
            try:
                env.sim_env.sim_context.render()
            except Exception:
                pass
            frame_path = frames_dir / f"frame_{saved:04d}.png"
            if _save_frame(rgb, frame_path):
                saved += 1
            if step % 20 == 0 or step == steps - 1:
                print(f"[export] step={step}/{steps - 1} frames={saved}", flush=True)
    finally:
        env.close()

    if saved < 2:
        raise SystemExit(f"too few frames ({saved})")

    mp4_cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out),
    ]
    print(f"[export] {' '.join(mp4_cmd)}", flush=True)
    subprocess.run(mp4_cmd, check=True)
    print(f"[export] wrote {out} ({out.stat().st_size} bytes)", flush=True)

    if not args.no_gif:
        gif = out.with_suffix(".gif")
        gif_cmd = [
            "ffmpeg", "-y", "-framerate", str(args.gif_fps),
            "-i", str(frames_dir / "frame_%04d.png"),
            "-vf", f"scale={int(args.gif_scale)}:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-loop", "0", str(gif),
        ]
        print(f"[export] {' '.join(gif_cmd)}", flush=True)
        subprocess.run(gif_cmd, check=True)
        print(f"[export] wrote {gif} ({gif.stat().st_size} bytes)", flush=True)


if __name__ == "__main__":
    main()
