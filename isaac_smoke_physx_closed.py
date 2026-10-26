"""Headless smoke: PhysX-closed 3+1 loop — evader pose from PhysX, no teleport."""
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

    cfg_path = os.path.join(ROOT, "configs/pursuit_v2/radar5_dual_hard_physx_closed.json")
    data = json.loads(open(cfg_path, encoding="utf-8").read())
    data["search_steps"] = 12
    data["building_count"] = min(int(data.get("building_count", 4)), 3)
    cfg = QuadrotorPursuitConfig(**data)
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)

    num_envs = int(os.environ.get("SMOKE_NUM_ENVS", "2"))
    print(f"[smoke-closed] creating physx_closed num_envs={num_envs}", flush=True)
    env = IsaacPursuitEnv(
        cfg, seed=11, headless=True, num_envs=num_envs, physx_closed_loop=True
    )
    code = 1
    try:
        assert env._physx_closed_loop and env._n_physics_drones == 4
        assert int(env.sim_env.drone.n) == 4
        obs = env.reset(seeds=[11_000 + i for i in range(num_envs)])
        assert isinstance(obs, list) and len(obs) == num_envs

        prev_targets = [env._target_position(i).copy() for i in range(num_envs)]
        actions = np.zeros((num_envs, cfg.n_uavs, 4), dtype=np.float32)
        actions[..., 0] = 0.15
        moved = 0
        for step in range(5):
            results = env.step_joint(actions)
            assert isinstance(results, list) and len(results) == num_envs
            for i in range(num_envs):
                tgt = env._target_position(i)
                assert np.isfinite(tgt).all(), tgt
                delta = float(np.linalg.norm(tgt - prev_targets[i]))
                # Evader should integrate under PhysX (non-zero motion over steps).
                if delta > 1e-4:
                    moved += 1
                prev_targets[i] = tgt.copy()
                # Bridge target must match PhysX authority.
                bridge_tgt = np.array(
                    [
                        *env.bridges[i].env.dynamic_targets[0],
                        env.bridges[i].env.target_altitudes[0],
                    ],
                    dtype=float,
                )
                assert np.allclose(bridge_tgt, tgt, atol=0.05), (bridge_tgt, tgt)
            print(
                f"[smoke-closed] step={step} "
                f"capture={[bool(r.get('capture_success')) for r in results]} "
                f"trunc={[bool(r.get('truncated')) for r in results]}",
                flush=True,
            )
        assert moved > 0, "evader never moved under PhysX"
        print("[smoke-closed] SUCCESS", flush=True)
        code = 0
    finally:
        try:
            env.close()
        except Exception as exc:
            print(f"[smoke-closed] close: {type(exc).__name__}: {exc}", flush=True)
        os._exit(code)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        os._exit(1)
