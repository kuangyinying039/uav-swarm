"""Tests for kinematic vectorized pursuit envs."""
import unittest

import numpy as np

from pursuit_vector_env import SubprocVectorPursuitEnv, SyncVectorPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig


def _tiny_cfg(seed=0):
    return QuadrotorPursuitConfig(
        seed=seed,
        search_steps=8,
        building_count=0,
        capture_required_uavs=2,
        execution_reward_version=5,
        n_uavs=3,
        n_targets=1,
        grid_size=25,
        target_diameter=0.5,
    )


class SyncVectorTests(unittest.TestCase):
    def test_step_shape_and_autoreset(self):
        vec = SyncVectorPursuitEnv.from_config(_tiny_cfg(), num_envs=3, seed=11)
        try:
            obs = vec.reset(seeds=[11, 12, 13])
            self.assertEqual(len(obs), 3)
            actions = np.zeros((3, 3, 4), dtype=np.float32)
            for _ in range(3):
                results = vec.step_joint(actions)
                self.assertEqual(len(results), 3)
                for result in results:
                    self.assertIn("obs", result)
                    self.assertIn("reward", result)
        finally:
            vec.close()


class SubprocVectorTests(unittest.TestCase):
    def test_parallel_step(self):
        vec = SubprocVectorPursuitEnv(_tiny_cfg(), num_envs=2, seed=21)
        try:
            obs = vec.reset(seeds=[21, 22])
            self.assertEqual(len(obs), 2)
            actions = np.zeros((2, 3, 4), dtype=np.float32)
            results = vec.step_joint(actions)
            self.assertEqual(len(results), 2)
            self.assertTrue(all(np.isfinite(float(r["reward"])) for r in results))
        finally:
            vec.close()


if __name__ == "__main__":
    unittest.main()
