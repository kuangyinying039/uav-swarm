"""Headless smoke: IsaacPursuitEnv num_envs=4, a few vector steps."""
from __future__ import annotations

import json
import os
import sys
import traceback

import numpy as np

ROOT = "/srv/data/projects/kyy/xieton/repro"
sys.path.insert(0, ROOT)


def main():
    from isaac_pursuit_backend import IsaacPursuitEnv
    from quadrotor_pursuit_env import QuadrotorPursuitConfig

    cfg_path = os.path.join(ROOT, "configs/pursuit_v2/radar5_dual_medium.json")
    data = json.loads(open(cfg_path, encoding="utf-8").read())
    # Short horizon for smoke; keep dual-capture contract.
    data["search_steps"] = 8
    data["building_count"] = min(int(data.get("building_count", 4)), 4)
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)

    num_envs = int(os.environ.get("SMOKE_NUM_ENVS", "4"))
    print(f"[smoke-vec] creating IsaacPursuitEnv num_envs={num_envs}", flush=True)
    env = IsaacPursuitEnv(cfg, seed=11, headless=True, num_envs=num_envs)
    code = 1
    try:
        seeds = [11_000 + i for i in range(num_envs)]
        obs = env.reset(seeds=seeds)
        assert isinstance(obs, list) and len(obs) == num_envs, type(obs)
        print(f"[smoke-vec] reset ok obs_keys={list(obs[0].keys())[:6]}", flush=True)

        actions = np.zeros((num_envs, cfg.n_uavs, 4), dtype=np.float32)
        actions[..., 0] = 0.1
        for step in range(3):
            results = env.step_joint(actions)
            assert isinstance(results, list) and len(results) == num_envs
            rewards = [float(r["reward"]) for r in results]
            dones = [bool(r.get("_episode_done")) for r in results]
            print(
                f"[smoke-vec] step={step} rewards={rewards} dones={dones} "
                f"capture={[bool(r.get('capture_success')) for r in results]}",
                flush=True,
            )
            assert all(np.isfinite(rewards))
        print("[smoke-vec] SUCCESS", flush=True)
        code = 0
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[smoke-vec] close: {type(exc).__name__}: {exc}", flush=True)
        os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
