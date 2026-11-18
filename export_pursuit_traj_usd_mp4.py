"""Headless USD pose-replay → Replicator RGB → MP4 (no PhysX Direct-GPU teleports).

Uses the same portable-root + ``rep.orchestrator.step`` path that already works
in ``omnidrones/.../replay_trajectory.py``. Buildings and drone poses come from
a live PhysX-closed capture ``.npz``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traj",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "mappo_physx_closed_capture_7000110.npz",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path.home() / "runs" / "isaac_dual_logs" / "mappo_physx_closed_capture_7000110.mp4",
    )
    parser.add_argument(
        "--portable-root",
        type=Path,
        default=Path("/srv/data/projects/kyy/omnidrones/.kit_portable"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.traj.is_file():
        raise SystemExit(f"missing traj: {args.traj}")

    portable = args.portable_root
    portable.mkdir(parents=True, exist_ok=True)
    # Must be set before SimulationApp imports Kit extensions.
    sys.argv.extend(["--portable-root", str(portable)])

    import isaacsim  # noqa: F401  — required before omni.isaac.kit on Isaac 4.1
    from omni.isaac.kit import SimulationApp

    exp = os.path.join(os.environ["EXP_PATH"], "omni.isaac.sim.python.kit")
    app = SimulationApp({"headless": True, "anti_aliasing": 1}, experience=exp)

    try:
        from PIL import Image
        from pxr import Gf, Sdf, UsdGeom, UsdShade
        import omni.replicator.core as rep
        import omni.usd

        payload = np.load(args.traj, allow_pickle=False)
        pursuer = np.asarray(payload["pursuer"], dtype=np.float64)
        target = np.asarray(payload["target"], dtype=np.float64)
        buildings = np.asarray(payload["buildings"], dtype=np.float64)
        building_heights = np.asarray(payload["building_heights"], dtype=np.float64)
        meta = json.loads(str(payload["meta"]))
        steps = int(pursuer.shape[0])
        n_uavs = int(pursuer.shape[1])
        grid = float(max(buildings[:, 2].max() if len(buildings) else 25.0, 25.0))
        # buildings are x0,y0,x1,y1 — infer grid from extents
        if len(buildings):
            grid = float(max(buildings[:, 2].max(), buildings[:, 3].max(), 25.0))

        out = args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        frames_dir = out.parent / f"{out.stem}_usd_frames"
        if frames_dir.exists():
            for old in frames_dir.glob("*.png"):
                old.unlink()
        frames_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"[usd-export] seed={meta.get('seed')} steps={steps} cap={meta.get('captured')} "
            f"out={out}",
            flush=True,
        )

        stage = omni.usd.get_context().get_stage()

        def bind_material(prim, name, color, opacity=1.0):
            material = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
            shader = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(opacity))
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)

        ground = UsdGeom.Cube.Define(stage, "/World/replay_ground")
        ground.AddScaleOp().Set(Gf.Vec3f(grid / 2, grid / 2, 0.05))
        ground.AddTranslateOp().Set(Gf.Vec3d(grid / 2, grid / 2, -0.05))
        ground.CreateDisplayColorAttr([(0.12, 0.14, 0.16)])
        bind_material(ground.GetPrim(), "Ground", (0.12, 0.14, 0.16))

        building_color = (0.18, 0.42, 0.48)
        for index, (rect, height) in enumerate(zip(buildings, building_heights)):
            x0, y0, x1, y1 = map(float, rect)
            h = float(height)
            cube = UsdGeom.Cube.Define(stage, f"/World/building_{index}")
            cube.AddScaleOp().Set(Gf.Vec3f((x1 - x0) / 2, (y1 - y0) / 2, h / 2))
            cube.AddTranslateOp().Set(Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, h / 2))
            cube.CreateDisplayColorAttr([building_color])
            bind_material(cube.GetPrim(), f"Building_{index}", building_color, 0.96)

        # Translate-then-scale order for cubes.
        for path in ["/World/replay_ground"] + [f"/World/building_{i}" for i in range(len(buildings))]:
            prim = UsdGeom.Cube.Get(stage, path)
            ops = prim.GetOrderedXformOps()
            translate = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate)
            scale = next(op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeScale)
            prim.SetXformOpOrder([translate, scale])

        pursuer_ops = []
        for index in range(n_uavs):
            sphere = UsdGeom.Sphere.Define(stage, f"/World/pursuer_{index}")
            sphere.CreateRadiusAttr(0.70)
            sphere.CreateDisplayColorAttr([(0.10, 0.45, 1.0)])
            bind_material(sphere.GetPrim(), f"Pursuer_{index}", (0.10, 0.45, 1.0))
            pursuer_ops.append(sphere.AddTranslateOp())

        evader = UsdGeom.Sphere.Define(stage, "/World/evader")
        evader.CreateRadiusAttr(0.80)
        evader.CreateDisplayColorAttr([(1.0, 0.20, 0.08)])
        bind_material(evader.GetPrim(), "Evader", (1.0, 0.20, 0.08))
        evader_op = evader.AddTranslateOp()

        # Capture-radius disc under the evader (visual only).
        capture_r = 2.0
        try:
            cfg_path = Path(str(meta.get("env_config", "")))
            if cfg_path.is_file():
                capture_r = float(json.loads(cfg_path.read_text(encoding="utf-8")).get("capture_radius", 2.0))
        except Exception:
            pass
        ring = UsdGeom.Cylinder.Define(stage, "/World/capture_ring")
        ring.CreateRadiusAttr(float(capture_r))
        ring.CreateHeightAttr(0.08)
        ring.CreateDisplayColorAttr([(1.0, 0.85, 0.15)])
        bind_material(ring.GetPrim(), "CaptureRing", (1.0, 0.85, 0.15), 0.45)
        ring_op = ring.AddTranslateOp()

        # Frame the actual chase volume, not empty map center.
        focus0 = np.mean(np.vstack([pursuer[:, :, :3].reshape(-1, 3), target]), axis=0)
        cam = rep.create.camera(
            position=(
                float(focus0[0] + 14.0),
                float(focus0[1] - 12.0),
                float(focus0[2] + 9.0),
            ),
            look_at=(float(focus0[0]), float(focus0[1]), float(focus0[2])),
            focal_length=24.0,
        )
        rep.create.light(light_type="distant", rotation=(-42, 28, 8), intensity=4500)
        rep.create.light(
            light_type="sphere",
            position=(grid * 0.3, grid * 0.2, 16.0),
            intensity=14000,
            scale=6.0,
        )
        render_product = rep.create.render_product(cam, (args.width, args.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([render_product])

        saved = 0
        for step in range(steps):
            for index in range(n_uavs):
                xyz = pursuer[step, index, :3]
                pursuer_ops[index].Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
            tgt = target[step]
            evader_op.Set(Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(tgt[2])))
            ring_op.Set(Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(max(tgt[2] - 0.85, 0.05))))

            rep.orchestrator.step(rt_subframes=2 if step else 4)
            data = rgb.get_data()
            if data is None:
                continue
            arr = np.asarray(data)
            if arr.ndim != 3 or arr.shape[2] < 3:
                continue
            frame = arr[:, :, :3]
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            Image.fromarray(frame).save(frames_dir / f"frame_{saved:04d}.png")
            saved += 1
            if step % 5 == 0 or step == steps - 1:
                print(f"[usd-export] step={step}/{steps - 1} frames={saved}", flush=True)

        if saved < 2:
            raise SystemExit(f"too few frames ({saved})")

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
            "-movflags",
            "+faststart",
            str(out),
        ]
        print(f"[usd-export] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        print(f"[usd-export] wrote {out} ({out.stat().st_size} bytes) frames={saved}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
