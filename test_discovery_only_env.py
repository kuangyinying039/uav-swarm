import unittest

import numpy as np

try:
    from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
except ImportError:
    from .cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig


class DiscoveryOnlyRewardTests(unittest.TestCase):
    def make_env(self, completion_steps: int = 0) -> WeakCommBeliefGraphEnv:
        cfg = WeakCommConfig(
            n_uavs=2,
            n_targets=3,
            grid_size=12,
            n_obstacles=0,
            n_jammers=0,
            comm_mode="full",
            dynamic_targets_enabled=False,
            dynamic_obstacles_enabled=False,
            failure_base_prob=0.0,
            target_completion_steps=completion_steps,
            search_steps=20,
        )
        env = WeakCommBeliefGraphEnv(cfg)
        env.reset()
        return env

    def test_zero_steps_does_not_complete_unseen_targets(self):
        env = self.make_env(0)
        env.dynamic_targets[:] = np.array([[10.0, 10.0], [10.0, 9.0], [9.0, 10.0]])
        metrics = env._update_tracking(env.found_targets.copy(), env.tracked_targets.copy())
        self.assertEqual(metrics["discovery_rate"], 0.0)
        self.assertEqual(metrics["completion_rate"], 0.0)
        self.assertFalse(np.any(env.completed_targets))

    def test_discovery_only_terminates_when_every_target_is_found(self):
        env = self.make_env(0)
        env.dynamic_targets[:] = env.positions[0]
        result = env.step_joint(np.array([8, 8]))
        self.assertTrue(result["terminated"])
        self.assertEqual(result["target_discovery_rate"], 1.0)
        self.assertEqual(result["target_tracking_rate"], 0.0)
        self.assertEqual(result["reward_components"]["tracking"], 0.0)
        self.assertEqual(result["reward_components"]["discovery_terminal"], 0.0)
        self.assertEqual(result["reward_components"]["team_discovery_potential"], 1.0)
        self.assertEqual(env.causal_target_indices(0).size, 0)
        np.testing.assert_array_equal(env.target_first_discovery_step, np.ones(3, dtype=int))

    def test_positive_completion_steps_keep_tracking_semantics(self):
        env = self.make_env(2)
        env.dynamic_targets[:] = env.positions[0]
        first = env.step_joint(np.array([8, 8]))
        self.assertFalse(first["terminated"])
        second = env.step_joint(np.array([8, 8]))
        self.assertTrue(second["terminated"])


if __name__ == "__main__":
    unittest.main()
