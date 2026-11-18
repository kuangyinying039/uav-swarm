"""录一局 PEG 场景的三维飞行，直到 PEG 终局或时间用尽。

场景是原来的圆柱和运动障碍。捕获是任意一架追捕机与逃逸机几何接触。
无显示器时用仿真相机写 MP4，不抓桌面。

    source ./activate.sh
    python -m parallel.examples.radar5_preview --method mpc --seed 1000 --output runtime/episode-inspect/episode.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch

from parallel.env.radar5_contract import physics_substeps
from parallel.examples.radar5_guidance import PursuerGuidance, evader_command
from parallel.training.mappo import _radar5_settings


def draw_buildings(env) -> None:
    """把这一局的五栋棱柱画成方块。不参与物理碰撞。"""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    base = env.unwrapped
    stage = base.stage
    offset = base.offsets[0].detach().cpu().tolist()
    material = UsdShade.Material.Define(stage, "/World/DemoMaterials/Buildings")
    shader = UsdShade.Shader.Define(stage, "/World/DemoMaterials/Buildings/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.55, 0.58, 0.62))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.22, 0.24, 0.28))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    for index, building in enumerate(base.rules.buildings[0].detach().cpu().tolist()):
        x0, y0, x1, y1, height = building
        cube = UsdGeom.Cube.Define(stage, f"/World/envs/env_0/Buildings/B{index}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(offset[0] + (x0 + x1) / 2, offset[1] + (y0 + y1) / 2, offset[2] + height / 2))
        cube.AddScaleOp().Set(Gf.Vec3d(x1 - x0, y1 - y0, height))
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


OUTCOME_NAMES = (
    "ongoing",
    "captured",
    "static_collision",
    "dynamic_collision",
    "out_of_bounds",
    "physical_failure",
    "success",
    "timeout",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--method", choices=("apf", "frpn", "orca", "cbf", "mpc"), default="apf")
    parser.add_argument("--methods", nargs="+", choices=("apf", "frpn", "orca", "cbf", "mpc"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--formation", choices=("stratified", "one_side"), default="one_side")
    parser.add_argument("--perception", choices=("truth", "mid360"), default="mid360")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    methods = args.methods or [args.method]
    if len(methods) == 1 and args.output.suffix == ".mp4" and args.output.exists():
        parser.error(f"output already exists: {args.output}")

    from parallel.env.config_loader import configure_isaac_app, load_visualization
    from parallel.env.dependencies import select_source
    from parallel.examples.demo_render import DemoCamera, VideoRecorder, create_demo_app

    settings = _radar5_settings()
    visuals = load_visualization()
    visuals["enabled"] = True
    visuals["static_obstacles"]["height"] = 12.0
    visuals["dynamic_obstacles"]["height"] = 12.0
    visuals["camera"]["resolution"] = [960, 540]
    visuals["camera"]["follow_evader"] = False
    visuals["camera"]["zoom"] = 1.0
    visuals["camera"]["view"] = "oblique"
    visuals["recording"]["enabled"] = True
    visuals["recording"]["fps"] = 10
    visuals = load_visualization(visuals)
    source = select_source(None)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(source / "examples/demo.yaml")
    configure_isaac_app(cfg, settings["simulation"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    app = None
    env = None
    camera = None
    writer = None
    try:
        print("[preview] starting Isaac", flush=True)
        torch.set_num_threads(4)
        cache = Path(__file__).resolve().parents[2] / "runtime/isaac-kit"
        cache.mkdir(parents=True, exist_ok=True)
        sys.argv.extend(["--portable-root", str(cache)])
        app = create_demo_app(cfg, visuals)
        from omni.isaac.core.utils.extensions import enable_extension

        enable_extension("omni.isaac.sensor")
        from parallel.env.od_env import ParallelEnv
        from parallel.strategies.config import load_config

        env = ParallelEnv(
            cfg,
            [args.seed],
            environment_config=settings,
            pursuer_config=load_config()[0],
            visualization_config=visuals,
            perception=None if args.perception == "truth" else args.perception,
            formation=None if args.formation == "stratified" else args.formation,
        )
        env.visualize = False
        draw_buildings(env)
        print("[preview] camera", flush=True)
        camera = DemoCamera(env, visuals)
        app.reset_render_settings()
        import omni.replicator.core as rep

        rep.settings.set_render_rtx_realtime(antialiasing="FXAA")
        camera.warmup()
        dt = float(settings["simulation"]["dt"])
        stride = max(1, int(round(0.1 / dt)))
        substeps = physics_substeps(dt)
        max_steps = int(settings["task"]["max_episode_steps"])
        reports = []
        for method in methods:
            video = args.output if len(methods) == 1 and args.output.suffix == ".mp4" else args.output / f"{method}.mp4"
            if video.exists():
                raise FileExistsError(video)
            env.reset(seeds=[args.seed])
            guidance = PursuerGuidance(method, env.n, env.device)
            guidance.reset()
            print(f"[preview] {method} seed={args.seed}", flush=True)
            writer = VideoRecorder(video, fps=visuals["recording"]["fps"], resolution=tuple(visuals["camera"]["resolution"]))
            writer.append(camera.frame(), 0.0)
            steps = 0
            limit = int(round(args.seconds / dt))
            outcome = "preview_complete"
            min_height = float("inf")
            max_height = float("-inf")
            while steps < min(limit, max_steps):
                if not app.is_running():
                    outcome = "window_closed"
                    break
                state = env.state()
                command = torch.zeros(env.n, state.shape[1], 4, device=env.device, dtype=state.dtype)
                command[:, 0] = evader_command(env)
                command[:, 1:] = guidance.command(env)
                finished = False
                for _ in range(substeps):
                    if steps >= min(limit, max_steps):
                        break
                    _, _, terminated, truncated, info = env.step(command)
                    steps += 1
                    height = info["state"][..., 2]
                    min_height = min(min_height, float(height.min()))
                    max_height = max(max_height, float(height.max()))
                    if steps % stride == 0:
                        env.visualizer.update(
                            env.rules.static,
                            env.rules.bg,
                            env.rules.radii[:, env.rules.layout.dynamic_objects],
                        )
                        writer.append(camera.frame(), steps * dt)
                    if bool(terminated[0] or truncated[0]):
                        outcome = OUTCOME_NAMES[int(info["outcome"][0])]
                        finished = True
                        break
                if finished:
                    break
            state = env.state()[0]
            evader = state[0, :3]
            pursuers = state[1:, :3]
            distance = torch.linalg.vector_norm(pursuers - evader, dim=-1)
            report = {
                "method": method,
                "seed": args.seed,
                "formation": args.formation,
                "perception": args.perception,
                "outcome": outcome,
                "physics_steps": steps,
                "simulation_seconds": steps * dt,
                "height_min": min_height,
                "height_max": max_height,
                "evader_xyz": [float(v) for v in evader.tolist()],
                "pursuer_distance": [float(v) for v in distance.tolist()],
                "video": str(video),
            }
            video.with_suffix(".json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
            writer.close()
            writer = None
            reports.append(report)
        if len(reports) > 1:
            (args.output / "preview.json").write_text(json.dumps(reports, indent=2))
    except BaseException:
        print("[preview] FAILED", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        if writer is not None:
            writer.close(success=False)
        if camera is not None:
            camera.close()
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == "__main__":
    main()
