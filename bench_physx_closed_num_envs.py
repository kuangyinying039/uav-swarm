"""Find max num_envs for physx_closed + native PhysX LiDAR.

Each candidate runs in a **fresh subprocess** (Isaac SimulationApp is once-per-process).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _gpu_mem_mb():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        used, free, total = [float(x) for x in out.split(",")]
        return {"used_mb": used, "free_mb": free, "total_mb": total}
    except Exception as exc:
        return {"error": str(exc)}


def _run_one_trial(num_envs: int, steps: int, cfg_path: Path, result_path: Path) -> dict:
    """Worker entry: construct env, step, write JSON, hard-exit."""
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from isaac_pursuit_backend import IsaacPursuitEnv
    from quadrotor_pursuit_env import QuadrotorPursuitConfig

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    data["search_steps"] = max(steps + 2, 8)
    data["building_count"] = min(int(data.get("building_count", 4)), 3)
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)

    row = {
        "num_envs": num_envs,
        "ok": False,
        "native_lidar": None,
        "sensors_created": None,
        "steps": steps,
        "gpu_before": _gpu_mem_mb(),
        "error": None,
    }
    env = None
    t0 = time.time()
    try:
        env = IsaacPursuitEnv(
            cfg,
            seed=11,
            headless=True,
            num_envs=num_envs,
            physx_closed_loop=True,
            native_lidar=True,
        )
        row["gpu_after_init"] = _gpu_mem_mb()
        row["native_lidar"] = bool(env._native_lidar_enabled)
        row["sensors_created"] = sum(1 for p in env.sensor_paths if p is not None)
        row["physics_drones"] = int(env._n_physics_drones)
        assert int(env.sim_env.drone.n) == 4, env.sim_env.drone.n
        obs = env.reset(seeds=[11_000 + i for i in range(num_envs)])
        assert isinstance(obs, list) and len(obs) == num_envs

        actions = np.zeros((num_envs, cfg.n_uavs, 4), dtype=np.float32)
        actions[..., 0] = 0.1
        t_steps = time.time()
        for step in range(steps):
            results = env.step_joint(actions)
            assert len(results) == num_envs
            if step == 0:
                row["sensor_modes_step0"] = [r.get("sensor_mode") for r in results]
                row["sensor_diag0"] = getattr(
                    env.bridges[0], "last_sensor_diagnostics", None
                )
        elapsed = time.time() - t_steps
        row["mean_step_s"] = elapsed / max(steps, 1)
        row["decisions_per_s"] = (num_envs * steps) / max(elapsed, 1e-6)
        row["gpu_after_steps"] = _gpu_mem_mb()
        row["init_s"] = time.time() - t0
        row["ok"] = True
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-2500:]
    finally:
        # Write BEFORE close: SimulationApp.close() often hard-exits the process.
        try:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
        except Exception as write_exc:
            print(f"[bench] failed to write {result_path}: {write_exc}", flush=True)
        if env is not None:
            try:
                env.close()
            except Exception as close_exc:
                row["close_error"] = f"{type(close_exc).__name__}: {close_exc}"
                try:
                    result_path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                except Exception:
                    pass
        os._exit(0 if row["ok"] else 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="1,2,4,6,8")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=ROOT / "configs/pursuit_v2/radar5_dual_hard_physx_closed.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/srv/data/projects/kyy/runs/isaac_dual_physx_closed/num_envs_bench.json"),
    )
    parser.add_argument("--trial", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--trial-out", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.trial > 0:
        assert args.trial_out is not None
        _run_one_trial(args.trial, args.steps, args.env_config, args.trial_out)
        return

    candidates = [int(x) for x in str(args.candidates).split(",") if x.strip()]
    results = []
    best = 0
    work = args.out.parent / "_bench_trials"
    work.mkdir(parents=True, exist_ok=True)

    for n in candidates:
        trial_out = work / f"n{n}.json"
        if trial_out.exists():
            trial_out.unlink()
        print(f"\n===== TRY num_envs={n} gpu={_gpu_mem_mb()} =====", flush=True)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--trial",
            str(n),
            "--trial-out",
            str(trial_out),
            "--steps",
            str(args.steps),
            "--env-config",
            str(args.env_config),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/srv/data/projects/kyy/omnidrones:{ROOT}:{env.get('PYTHONPATH', '')}"
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
        env["PYTHONUNBUFFERED"] = "1"
        # Drop DISPLAY so Kit stays headless even if a desktop session exists.
        env.pop("DISPLAY", None)
        proc = subprocess.run(cmd, env=env, cwd=str(ROOT))
        if trial_out.is_file():
            row = json.loads(trial_out.read_text(encoding="utf-8"))
        else:
            row = {
                "num_envs": n,
                "ok": False,
                "error": f"subprocess exit={proc.returncode}, no result file",
            }
        results.append(row)
        slim = {k: v for k, v in row.items() if k != "traceback"}
        print(json.dumps(slim, indent=2), flush=True)
        if row.get("ok"):
            best = n
        else:
            print(f"STOP at num_envs={n}", flush=True)
            break
        # Let GPU memory settle between Kit tears-downs.
        time.sleep(5)

    payload = {
        "best_num_envs": best,
        "candidates": candidates,
        "steps_per_trial": args.steps,
        "results": results,
        "gpu_final": _gpu_mem_mb(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\n[bench] best_num_envs={best} wrote {args.out}", flush=True)
    return 0 if best > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
