"""Stability smoke for Isaac multi-env: random actions, auto-reset, corrupt recovery."""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

import numpy as np

ROOT = "/srv/data/projects/kyy/xieton/repro"
sys.path.insert(0, ROOT)


def main():
    from isaac_pursuit_backend import IsaacPursuitEnv
    from quadrotor_pursuit_env import QuadrotorPursuitConfig

    cfg_path = os.path.join(ROOT, "configs/pursuit_v2/radar5_dual_medium.json")
    data = json.loads(open(cfg_path, encoding="utf-8").read())
    # Short episodes so auto-reset fires many times during the smoke.
    data["search_steps"] = int(os.environ.get("SMOKE_SEARCH_STEPS", "12"))
    data["building_count"] = min(int(data.get("building_count", 4)), 4)
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)

    num_envs = int(os.environ.get("SMOKE_NUM_ENVS", "4"))
    vector_steps = int(os.environ.get("SMOKE_VECTOR_STEPS", "48"))
    print(
        f"[stable-smoke] num_envs={num_envs} search_steps={cfg.search_steps} "
        f"vector_steps={vector_steps}",
        flush=True,
    )
    env = IsaacPursuitEnv(cfg, seed=11, headless=True, num_envs=num_envs)
    code = 1
    rng = np.random.default_rng(11)
    try:
        obs = env.reset(seeds=[11_000 + i for i in range(num_envs)])
        assert len(obs) == num_envs
        completed = 0
        t0 = time.time()
        for step in range(vector_steps):
            actions = rng.uniform(-1.0, 1.0, size=(num_envs, cfg.n_uavs, 4)).astype(np.float32)
            results = env.step_joint(actions)
            assert len(results) == num_envs
            rewards = [float(r["reward"]) for r in results]
            assert all(np.isfinite(rewards)), rewards
            for result in results:
                if result.get("_episode_done"):
                    completed += 1
                    assert "final_obs" in (result.get("info") or {}), result.get("info")
                    assert result["obs"] is not None
            if step % 8 == 0 or step + 1 == vector_steps:
                print(
                    f"[stable-smoke] step={step + 1}/{vector_steps} "
                    f"completed_eps={completed} physics_fails={env._physics_fail_count} "
                    f"rewards_mean={float(np.mean(rewards)):.3f}",
                    flush=True,
                )
        elapsed = time.time() - t0
        sps = (vector_steps * num_envs) / max(elapsed, 1e-6)
        print(
            f"[stable-smoke] SUCCESS completed_eps={completed} "
            f"physics_fails={env._physics_fail_count} "
            f"elapsed_s={elapsed:.1f} env_steps_per_s={sps:.2f}",
            flush=True,
        )
        # Expect multiple auto-resets with short search_steps.
        assert completed >= num_envs, f"expected auto-resets, got completed={completed}"
        code = 0
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[stable-smoke] close: {type(exc).__name__}: {exc}", flush=True)
        os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
