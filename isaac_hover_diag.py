"""Short hover diagnostic for the Isaac PhysX pursuit backend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def main():
    cfg = QuadrotorPursuitConfig(
        **json.loads(Path("configs/pursuit_v2/radar5_dual_medium.json").read_text())
    )
    cfg.search_steps = 40
    env = IsaacPursuitEnv(cfg, seed=11, headless=True, native_lidar=False)
    try:
        env.reset(seed=11)
        rows = []
        for step in range(30):
            result = env.step_joint(np.zeros((3, 4), dtype=np.float32))
            physical = np.asarray(env._physical_state(), dtype=float)
            rows.append(
                {
                    "step": step + 1,
                    "z": physical[:, 2].tolist(),
                    "mean_z": float(physical[:, 2].mean()),
                    "min_z": float(physical[:, 2].min()),
                    "feasible": float(result["controller_feasible_rate"]),
                    "gap": float(result["minimum_capture_gap"]),
                }
            )
            print(
                f"[hover] step={step + 1} mean_z={rows[-1]['mean_z']:.3f} "
                f"min_z={rows[-1]['min_z']:.3f} feasible={rows[-1]['feasible']:.2f}",
                flush=True,
            )
        out = Path.home() / "runs" / "isaac_hover_diag.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
        first_z = rows[0]["mean_z"]
        last_z = rows[-1]["mean_z"]
        ok = rows[-1]["min_z"] > 1.0 and abs(last_z - first_z) < 1.5
        print(
            json.dumps(
                {
                    "ok": ok,
                    "first_mean_z": first_z,
                    "last_mean_z": last_z,
                    "last_min_z": rows[-1]["min_z"],
                    "mean_feasible": float(np.mean([r["feasible"] for r in rows])),
                    "out": str(out),
                },
                indent=2,
            ),
            flush=True,
        )
        if not ok:
            raise SystemExit(1)
    finally:
        env.close()


if __name__ == "__main__":
    main()
