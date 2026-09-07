import unittest

import numpy as np

from pursuit_kinematics import quaternion_from_yaw
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_baselines_3d import (
    ArtificialPotentialField3D,
    CooperativeGuidanceMPC3D,
    FastResponseProportionalNavigation3D,
)


class QuadrotorPursuitTests(unittest.TestCase):
    def initial_state(self, altitude=5.0):
        state = np.zeros(13, dtype=float)
        state[:3] = [4.0, 5.0, altitude]
        state[6:10] = quaternion_from_yaw(0.0)
        return state

    def test_randomized_building_count_keeps_critic_state_dimension_fixed(self):
        dimensions = []
        for building_count in (0, 2, 5):
            cfg = QuadrotorPursuitConfig(
                seed=31,
                building_count=building_count,
                building_state_capacity=5,
            )
            env = QuadrotorPursuitEnv(cfg)
            state = env.observe_search()["state_vector"]
            self.assertEqual(len(state), env.state_dim())
            dimensions.append(env.state_dim())
        self.assertEqual(len(set(dimensions)), 1)

    def test_pursuit_defaults_do_not_randomly_disable_uavs(self):
        cfg = QuadrotorPursuitConfig()
        self.assertEqual(cfg.failure_base_prob, 0.0)
        self.assertEqual(cfg.failure_jammed_prob, 0.0)
        self.assertEqual(cfg.failure_recovery_prob, 0.0)

    def test_timeout_penalty_dominates_nonterminal_shaping(self):
        cfg = QuadrotorPursuitConfig(
            seed=73,
            search_steps=1,
            building_count=0,


        )
        env = QuadrotorPursuitEnv(cfg)
        result = env.step_joint(np.zeros((cfg.n_uavs, 4), dtype=float))
        components = result["reward_components"]
        self.assertFalse(result["capture_success"])
        self.assertEqual(components["timeout"], -cfg.timeout_penalty)
        self.assertLessEqual(
            components["information_gain"],
            cfg.information_gain_reward * cfg.information_gain_clip,
        )
        self.assertNotIn("containment", components)
        self.assertNotIn("valid_track", components)

    def test_environment_executes_velocity_reference(self):
        cfg = QuadrotorPursuitConfig(
            seed=18,
            building_count=0,
            n_obstacles=0,
            dynamic_obstacles_enabled=False,



            failure_base_prob=0.0,
        )
        env = QuadrotorPursuitEnv(cfg)
        before = env.quadrotor_states.copy()
        actions = np.tile(np.array([0.0, 0.0, 0.2, 0.0]), (cfg.n_uavs, 1))
        result = env.step_joint(actions)
        controls = np.asarray(result["desired_velocities"])
        states = np.asarray(result["quadrotor_states"])
        self.assertEqual(env.continuous_action_dim(), 4)
        self.assertEqual(controls.shape, (3, 3))
        self.assertEqual(states.shape, (3, 13))
        self.assertTrue(np.all(controls[:, 0] >= 0.0))
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.any(np.abs(states - before) > 1e-8))
        self.assertEqual(env.observe_search()["agent_observations"].shape, (3, env.obs_dim()))
        self.assertIn("controller_feasible_rate", result)

    def test_three_dimensional_baselines_produce_bounded_references(self):
        cfg = QuadrotorPursuitConfig(seed=9, building_count=2, n_obstacles=0)
        env = QuadrotorPursuitEnv(cfg)
        for policy in (ArtificialPotentialField3D(), FastResponseProportionalNavigation3D(), CooperativeGuidanceMPC3D()):
            actions = policy.actions(env)
            self.assertEqual(actions.shape, (3, 4))
            self.assertTrue(np.all(np.isfinite(actions)))
            self.assertTrue(np.all(np.abs(actions) <= 1.0 + 1e-12))

    def test_baselines_do_not_read_target_truth(self):
        cfg = QuadrotorPursuitConfig(seed=10, building_count=0, n_obstacles=0)
        env = QuadrotorPursuitEnv(cfg)
        for policy in (ArtificialPotentialField3D(), FastResponseProportionalNavigation3D(), CooperativeGuidanceMPC3D()):
            before = policy.actions(env)
            env.dynamic_targets[0] += np.array([7.0, -5.0])
            env.target_altitudes[0] += 3.0
            env.target_velocity[0] *= -1.0
            env.target_vertical_velocity[0] *= -1.0
            after = policy.actions(env)
            np.testing.assert_allclose(after, before, atol=1e-12)

    def test_randomized_start_distance_and_observability(self):
        for seed in (23, 37, 51, 71, 89):
            env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=seed, lidar_detection_probability=1))
            target = np.array([*env.dynamic_targets[0], env.target_altitudes[0]])
            distance = np.linalg.norm(env.quadrotor_states[:, :3] - target, axis=1)
            self.assertTrue(np.all(distance >= env.cfg.initial_distance_min - 1e-9))
            self.assertTrue(np.all(distance <= env.cfg.initial_distance_max + 1e-9))
            self.assertTrue(bool(np.any(env.direct_visibility_mask()[:, 0])))

    def test_final_collision_logic_uses_three_dimensional_distance(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=12, building_count=0, n_obstacles=0))
        env.positions[1] = env.positions[0]
        env.altitudes[1] = env.altitudes[0] + 2.0
        collisions, crashed = env._count_uav_collisions()
        self.assertEqual(collisions, 0)
        self.assertEqual(crashed, set())
        env.altitudes[1] = env.altitudes[0] + 0.4
        collisions, crashed = env._count_uav_collisions()
        self.assertEqual(collisions, 1)
        self.assertEqual(crashed, {0, 1})

    def test_distance_capture_boundary_height_and_disabled(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=15, building_count=0))
        env.quadrotor_states[:, :3] = [[5, 5, 5], [15, 15, 5], [20, 20, 5]]
        env.dynamic_targets[0] = [5, 5]
        for height, expected in [(5.999, True), (6.0, True), (6.001, False)]:
            env.target_altitudes[0] = height
            self.assertEqual(env._capture_geometry()[0], expected)
            self.assertAlmostEqual(env.minimum_capture_gap(), max(height-6, 0))
        env.target_altitudes[0] = 5.5
        env.disabled_uavs[0] = True
        self.assertFalse(env._capture_geometry()[0])
        env.disabled_uavs[0] = False
        env.quadrotor_states[0, 6:10] = [0, 1, 0, 0]
        self.assertTrue(env._capture_geometry()[0])

    def test_repeated_velocity_reference_causes_horizontal_translation(self):
        cfg = QuadrotorPursuitConfig(
            seed=4, building_count=0, n_obstacles=0, dynamic_obstacles_enabled=False,

        )
        env = QuadrotorPursuitEnv(cfg)
        initial = env.quadrotor_states[:, :3].copy()
        action = np.tile(np.array([0.8, 0.0, 0.0, 0.0]), (cfg.n_uavs, 1))
        for _ in range(8):
            env.step_joint(action)
        displacement = np.linalg.norm(env.quadrotor_states[:, :3] - initial, axis=1)
        self.assertGreater(float(np.mean(displacement)), 0.05)


if __name__ == "__main__":
    unittest.main()
