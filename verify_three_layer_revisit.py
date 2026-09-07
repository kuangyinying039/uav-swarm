"""Small deterministic checks for local three-layer maps and weight actions."""

from __future__ import annotations

import numpy as np

from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig


def build_env(**overrides) -> WeakCommBeliefGraphEnv:
    cfg = WeakCommConfig(
        n_uavs=3,
        n_targets=2,
        n_obstacles=0,
        grid_size=12,
        search_steps=20,
        outage_base_prob=0.0,
        outage_jammed_prob=0.0,
        failure_base_prob=0.0,
        failure_jammed_prob=0.0,
        seed=17,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    return WeakCommBeliefGraphEnv(cfg)


def verify_shapes_and_weight_actions() -> None:
    env = build_env()
    obs = env.reset()
    assert obs["graph_features"].shape == (3, 32)
    assert obs["agent_observations"].shape == (3, 41)
    assert obs["hetero_graph"]["target_nodes"].shape[-1] == 13
    weights = np.tile(np.array([0.25, 0.25, 0.20, 0.10, 0.20]), (3, 1))
    actions = env.actions_from_search_weights(weights)
    assert actions.shape == (3,) and np.all((0 <= actions) & (actions < 9))
    result = env.step_search(actions)
    assert "revisit_gain" in result


def verify_causal_fusion() -> None:
    env = build_env(comm_mode="none")
    env.reset()
    env.local_revisit_map.fill(1.0)
    env.local_revisit_map[0, 7, 7] = 0.0
    env._outage_graph_fusion()
    assert env.local_revisit_map[0, 7, 7] == 0.0
    assert env.local_revisit_map[1, 7, 7] == 1.0
    env.cfg.comm_mode = "full"
    env._outage_graph_fusion()
    assert np.all(env.local_revisit_map[:, 7, 7] == 0.0)


def verify_revisit_gain_and_ablation() -> None:
    env = build_env(n_uavs=1, n_targets=1, revisit_aging_rate=0.0, revisit_diffusion=0.0)
    env.reset()
    env.positions[0] = np.array([4, 4])
    env.visit_counts[4, 4] = 1
    env.local_revisit_map[0, 4, 4] = 1.0
    assert env._update_local_revisit_maps(np.array([1.0])) > 0.0

    disabled = build_env(n_uavs=1, n_targets=1, revisit_map_enabled=False)
    disabled.reset()
    assert not np.any(disabled.local_revisit_map)
    benefits, _ = disabled.search_benefit_matrix(0)
    assert not np.any(benefits[:, 2])


if __name__ == "__main__":
    verify_shapes_and_weight_actions()
    verify_causal_fusion()
    verify_revisit_gain_and_ablation()
    print("three-layer revisit and adaptive-weight checks passed")
