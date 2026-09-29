"""Shared helpers for multi-difficulty pursuit training mixes."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


def load_pursuit_config(path: Path | str) -> QuadrotorPursuitConfig:
    cfg = QuadrotorPursuitConfig(**json.loads(Path(path).read_text(encoding="utf-8")))
    cfg.building_state_capacity = max(cfg.building_state_capacity, cfg.building_count)
    return cfg


def pursuit_contract_key(cfg: QuadrotorPursuitConfig) -> tuple:
    """Identity for mix compatibility without constructing a full episode."""
    return (
        int(cfg.n_uavs),
        int(cfg.n_targets),
        int(cfg.building_state_capacity),
        int(cfg.policy_observation_version),
        int(cfg.scenario_version),
        str(cfg.task_mode),
        str(cfg.action_mode),
        int(cfg.execution_reward_version),
        str(cfg.capture_mode),
    )


def assert_compatible_pursuit_mix(configs: list[QuadrotorPursuitConfig]) -> tuple:
    """Require identical observation/action contracts across a training mix."""
    if not configs:
        raise ValueError("At least one environment config is required")
    keys = [pursuit_contract_key(cfg) for cfg in configs]
    if len(set(keys)) != 1:
        raise ValueError(f"Mixed environment configs change tensor contracts: {keys}")
    return keys[0]


def make_pursuit_env_factory(
    configs: list[QuadrotorPursuitConfig],
    *,
    seed_fn,
    primary_index: int = -1,
):
    """Build ``factory(episode=None|int)`` for probe/validation vs mixed rollouts.

    ``factory()`` / ``factory(None)`` always returns the primary config (default:
    last entry, typically Hard). ``factory(episode)`` cycles through the mix.
    """
    if not configs:
        raise ValueError("configs must be non-empty")
    primary = configs[primary_index]
    assert_compatible_pursuit_mix(configs)

    def factory(episode=None):
        if episode is None:
            cfg = primary
            seed = seed_fn(0)
        else:
            cfg = configs[int(episode) % len(configs)]
            seed = seed_fn(int(episode))
        return QuadrotorPursuitEnv(replace(cfg, seed=seed))

    return factory, primary


def mix_manifest(paths: list[Path | str], primary: QuadrotorPursuitConfig) -> dict:
    return {
        "mix_env_configs": [str(Path(path)) for path in paths],
        "primary_env_config": asdict(primary),
        "capture_required_uavs": int(primary.capture_required_uavs),
    }
