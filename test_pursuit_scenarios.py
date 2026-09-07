import unittest
import numpy as np
from quadrotor_pursuit_env import QuadrotorPursuitEnv, QuadrotorPursuitConfig
from pursuit_scenarios import velocity_step
from pursuit_baselines_3d import CooperativeGuidanceMPC3D


class ScenarioTests(unittest.TestCase):
    def test_limits_and_capture_contract(self):
        cfg = QuadrotorPursuitConfig()
        self.assertAlmostEqual(cfg.target_speed/cfg.max_horizontal_velocity, 1.3)
        self.assertGreater(cfg.target_vertical_speed, cfg.max_vertical_velocity)
        self.assertEqual((cfg.capture_radius, cfg.target_diameter, cfg.capture_required_uavs,
                          cfg.capture_hold_steps), (1., .5, 1, 1))
        self.assertEqual(QuadrotorPursuitConfig(target_diameter=.7).capture_radius, 1.4)
        velocity = np.zeros(3)
        for _ in range(100):
            new = velocity_step(velocity, [10., 10., 10.], .2, cfg)
            self.assertLessEqual(np.linalg.norm(new[:2]-velocity[:2]), cfg.evader_horizontal_acceleration*.2+1e-10)
            self.assertLessEqual(abs(new[2]-velocity[2]), cfg.evader_vertical_acceleration*.2+1e-10)
            self.assertLessEqual(np.linalg.norm(new[:2]), cfg.target_speed+1e-10)
            self.assertLessEqual(abs(new[2]), cfg.target_vertical_speed+1e-10)
            velocity = new
        self.assertGreater(np.linalg.norm(velocity[:2]), cfg.max_horizontal_velocity)

    def test_layouts_repeatable_and_collision_free(self):
        layouts = set()
        for seed in range(12):
            env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=seed))
            layouts.add(env.initial_layout)
            twin = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=seed))
            np.testing.assert_array_equal(env.quadrotor_states, twin.quadrotor_states)
            self.assertTrue(all(8 <= distance <= 16 for distance in env.initial_distances))
            self.assertEqual(env._count_uav_collisions()[0], 0)
            self.assertFalse(env._capture_geometry()[0])
        self.assertEqual(layouts, {'same_side', 'crossing', 'dispersed'})

    def test_mpc_agent_does_not_read_other_local_target_estimates(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(building_count=0))
        before = CooperativeGuidanceMPC3D().actions(env)[0]
        for agent in (1, 2):
            env.track_memory[agent][0].mean[:] += 100
        after = CooperativeGuidanceMPC3D().actions(env)[0]
        np.testing.assert_allclose(before, after)

    def test_actor_gets_map_and_only_communicated_peer_context(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig())
        env.last_graph[:] = np.eye(env.cfg.n_uavs)
        before = env.agent_observation_vectors()[0]
        env.quadrotor_states[1, :3] += 3
        after = env.agent_observation_vectors()[0]
        np.testing.assert_allclose(before, after)
        self.assertEqual(len(after), env.obs_dim())

    def test_actor_features_do_not_read_target_truth(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig())
        before = env.agent_observation_vectors().copy()
        env.dynamic_targets[:] += 10
        env.target_altitudes[:] += 2
        env.target_velocity[:] *= -1
        np.testing.assert_allclose(before, env.agent_observation_vectors())


if __name__ == '__main__':
    unittest.main()
