"""Vectorized pursuit environments for higher sample throughput.

Why this exists
---------------
Serial Isaac / kinematic rollouts waste wall-clock: one decision step advances
only one scene. OmniDrones already batches PhysX across ``num_envs``; Radar5
glue historically forced ``num_envs=1``.

API
---
Both vector wrappers expose::

    obs_list = vec.reset(seeds)
    results = vec.step_joint(actions)   # actions: (num_envs, n_uavs, 4)

``SyncVectorPursuitEnv``
    Same-process sequential steps. Useful for API tests / debugging. **Does not**
    increase env throughput (still O(N) Python time).

``SubprocVectorPursuitEnv``
    One process per scene. Real kinematic speedup on multi-core hosts
    (typically ~0.6N–0.9N× until CPU-bound).

Isaac Phase 1 (see ``isaac_pursuit_backend.IsaacPursuitEnv`` ``num_envs``)
    Batches PhysX on GPU; N Radar5 bridges still advance in Python (or later
    fused). Expected PhysX decision throughput ~5–15× at N=64–128.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import multiprocessing as mp
from typing import Callable, Sequence

import numpy as np

from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def _obs_from_env(env) -> dict:
    if hasattr(env, "observe_search"):
        return env.observe_search()
    return env.reset()


def _worker(pipe, cfg_dict: dict, seed: int):
    cfg = QuadrotorPursuitConfig(**cfg_dict)
    cfg.seed = int(seed)
    env = QuadrotorPursuitEnv(cfg)
    try:
        while True:
            cmd, payload = pipe.recv()
            if cmd == "reset":
                if payload is not None:
                    env.cfg.seed = int(payload)
                    env.rng = np.random.default_rng(int(payload))
                env.reset()
                pipe.send(_obs_from_env(env))
            elif cmd == "step":
                result = env.step_joint(payload)
                done = bool(result.get("terminated") or result.get("truncated"))
                out = dict(result)
                out["_episode_done"] = done
                if done:
                    final_obs = result["obs"]
                    env.reset()
                    out["info"] = {
                        "final_obs": final_obs,
                        "final_terminated": bool(result.get("terminated", False)),
                        "final_truncated": bool(result.get("truncated", False)),
                        "auto_reset": True,
                        "capture_success": bool(result.get("capture_success", False)),
                    }
                    out["obs"] = _obs_from_env(env)
                    out["terminated"] = False
                    out["truncated"] = False
                else:
                    out["info"] = {"auto_reset": False}
                pipe.send(out)
            elif cmd == "close":
                if hasattr(env, "close"):
                    env.close()
                pipe.send(True)
                break
            else:
                pipe.send(RuntimeError(f"unknown command {cmd}"))
    except KeyboardInterrupt:
        pass
    finally:
        pipe.close()


class SyncVectorPursuitEnv:
    """Lockstep vector env in one process (correctness / API parity only)."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], QuadrotorPursuitEnv]],
        *,
        auto_reset: bool = True,
    ):
        if not env_fns:
            raise ValueError("env_fns must be non-empty")
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.auto_reset = bool(auto_reset)
        self.cfg = self.envs[0].cfg
        self.n_uavs = int(self.cfg.n_uavs)

    @classmethod
    def from_config(cls, cfg: QuadrotorPursuitConfig, num_envs: int, *, seed: int = 0):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")

        def make(i: int):
            local = replace(cfg, seed=int(seed) + i)
            return lambda: QuadrotorPursuitEnv(local)

        return cls([make(i) for i in range(num_envs)])

    def reset(self, seeds: Sequence[int] | None = None) -> list[dict]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        observations = []
        for index, env in enumerate(self.envs):
            if seeds is not None:
                env.cfg.seed = int(seeds[index])
                env.rng = np.random.default_rng(int(seeds[index]))
            env.reset()
            observations.append(_obs_from_env(env))
        return observations

    def step_joint(self, actions: np.ndarray) -> list[dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions leading dim {actions.shape[0]} != num_envs {self.num_envs}")
        results = []
        for index, env in enumerate(self.envs):
            result = dict(env.step_joint(actions[index]))
            done = bool(result.get("terminated") or result.get("truncated"))
            result["_episode_done"] = done
            if done and self.auto_reset:
                result["info"] = {
                    "final_obs": result["obs"],
                    "final_terminated": bool(result.get("terminated", False)),
                    "final_truncated": bool(result.get("truncated", False)),
                    "auto_reset": True,
                    "capture_success": bool(result.get("capture_success", False)),
                }
                env.reset()
                result["obs"] = _obs_from_env(env)
                result["terminated"] = False
                result["truncated"] = False
            else:
                result["info"] = {"auto_reset": False}
            results.append(result)
        return results

    def close(self) -> None:
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


class SubprocVectorPursuitEnv:
    """One OS process per kinematic scene — real multi-core sample throughput."""

    def __init__(self, cfg: QuadrotorPursuitConfig, num_envs: int, *, seed: int = 0):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        # spawn avoids fork+CUDA issues if the parent already touched torch CUDA
        ctx = mp.get_context("spawn")
        self.num_envs = int(num_envs)
        self.cfg = cfg
        self.n_uavs = int(cfg.n_uavs)
        self._parents = []
        self._procs = []
        cfg_dict = asdict(cfg)
        for index in range(self.num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(child, cfg_dict, int(seed) + index),
                daemon=True,
            )
            proc.start()
            child.close()
            self._parents.append(parent)
            self._procs.append(proc)

    def reset(self, seeds: Sequence[int] | None = None) -> list[dict]:
        if seeds is not None and len(seeds) != self.num_envs:
            raise ValueError(f"expected {self.num_envs} seeds, got {len(seeds)}")
        for index, pipe in enumerate(self._parents):
            pipe.send(("reset", None if seeds is None else int(seeds[index])))
        return [pipe.recv() for pipe in self._parents]

    def step_joint(self, actions: np.ndarray) -> list[dict]:
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"actions leading dim {actions.shape[0]} != num_envs {self.num_envs}")
        for pipe, action in zip(self._parents, actions):
            pipe.send(("step", action))
        results = []
        for pipe in self._parents:
            payload = pipe.recv()
            if isinstance(payload, Exception):
                raise payload
            results.append(payload)
        return results

    def close(self) -> None:
        for pipe in self._parents:
            try:
                pipe.send(("close", None))
                pipe.recv()
            except (EOFError, BrokenPipeError, OSError):
                pass
            pipe.close()
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
