"""Roll out one Isaac MPC episode and write the Radar5 HTML replay."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac_pursuit_backend import IsaacPursuitEnv
from pursuit_baselines_3d import CooperativeGuidanceMPC3D
from pursuit_visualization import write_pursuit_animation_html
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def main():
    out = Path.home() / "runs" / "isaac_mpc_episode_viz"
    out.mkdir(parents=True, exist_ok=True)
    cfg = QuadrotorPursuitConfig(
        **json.loads(Path("configs/pursuit_v2/radar5_dual_medium.json").read_text())
    )
    env = IsaacPursuitEnv(cfg, seed=7_000_000, headless=True, native_lidar=False)
    try:
        observation = env.reset(seed=7_000_000)
        expert = CooperativeGuidanceMPC3D()
        expert.reset(env)
        frames = []
        result = None
        for step in range(cfg.search_steps):
            action = np.asarray(expert.actions(env), dtype=np.float32)
            result = env.step_joint(action)
            observation = result["obs"]
            physical = np.asarray(env._physical_state(), dtype=float)
            frame = {
                "step": step + 1,
                "reward": float(result["reward"]),
                "positions": physical[:, :2].tolist(),
                "altitudes": physical[:, 2].tolist(),
                "target": env.bridge.env.dynamic_targets[0].tolist(),
                "target_altitudes": [float(env.bridge.env.target_altitudes[0])],
                "buildings": [list(map(float, rect)) for rect in env.bridge.env.buildings],
                "building_heights": [float(h) for h in env.bridge.env.building_heights],
                "minimum_capture_gap": float(result["minimum_capture_gap"]),
                "controller_feasible_rate": float(result["controller_feasible_rate"]),
                "capture_success": bool(result["capture_success"]),
                "execution_mode": result.get("execution_mode"),
                "sensor_mode": result.get("sensor_mode"),
                "direct_visibility": env.bridge.env.direct_visibility_mask().tolist(),
                "lidar_detection_ratio": float(result.get("lidar_detection_ratio", 0.0)),
                "lidar_vertical_fov_deg": [
                    float(cfg.lidar_vertical_min_deg),
                    float(cfg.lidar_vertical_max_deg),
                ],
                "continuous_safety_interventions": int(
                    result.get("continuous_safety_interventions", 0)
                ),
                "mode": "3d",
            }
            frames.append(frame)
            if step < 5 or (step + 1) % 25 == 0 or result["terminated"] or result["truncated"]:
                print(
                    f"[mpc-isaac] step={step + 1} gap={frame['minimum_capture_gap']:.3f} "
                    f"mean_z={physical[:, 2].mean():.3f} feasible={frame['controller_feasible_rate']:.2f} "
                    f"capture={frame['capture_success']}",
                    flush=True,
                )
            if result["terminated"] or result["truncated"]:
                break
        trace = {
            "episode": 1,
            "mode": "3d",
            "captured": bool(result and result["capture_success"]),
            "frames": frames,
        }
        write_pursuit_animation_html(
            out / "capture_episode_1.html",
            trace,
            "Isaac PhysX MPC | episode 1",
        )
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "steps": len(frames),
                    "captured": trace["captured"],
                    "final_gap": frames[-1]["minimum_capture_gap"],
                    "mean_z_last": float(np.mean(frames[-1]["altitudes"])),
                    "mean_feasible": float(
                        np.mean([frame["controller_feasible_rate"] for frame in frames])
                    ),
                    "html": str(out / "capture_episode_1.html"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(json.dumps(json.loads((out / "summary.json").read_text()), indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
