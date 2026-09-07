import unittest
from unittest.mock import patch
import numpy as np

from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_flight_controller import integrate_velocity_reference
from pursuit_lidar import lidar_visibility
from pursuit_kinematics import quaternion_from_yaw


class VelocityLidarTests(unittest.TestCase):
    def test_default_contract_has_no_mode_switches(self):
        cfg = QuadrotorPursuitConfig()
        self.assertFalse(hasattr(cfg, 'execution_mode'))
        self.assertFalse(hasattr(cfg, 'sensor_mode'))
        self.assertFalse(hasattr(cfg, 'nmpc_samples'))
        self.assertEqual(cfg.lidar_model, 'mid360')
        self.assertEqual((cfg.lidar_vertical_min_deg, cfg.lidar_vertical_max_deg), (-7, 52))
        self.assertTrue(cfg.pursuit_target_observable)

    def test_response_lag_acceleration_limit_and_reset(self):
        cfg = self.config()
        state = np.zeros(13)
        state[6] = 1
        after = integrate_velocity_reference(state, [cfg.max_horizontal_velocity, 0, 0], 1., .2, cfg)
        self.assertGreater(after[3], 0)
        self.assertLess(after[3], cfg.max_horizontal_velocity)
        self.assertLessEqual(np.linalg.norm(after[3:5]), cfg.max_horizontal_acceleration*.2 + 1e-9)
        self.assertGreater(after[12], 0)
        self.assertLess(after[12], 1.)
        env = QuadrotorPursuitEnv(cfg)
        env.step_joint(np.ones((cfg.n_uavs, 4))*.2)
        observation = env.reset()
        np.testing.assert_allclose(env.quadrotor_states[:, 3:6], 0)
        np.testing.assert_allclose(env.quadrotor_states[:, 10:13], 0)
        self.assertEqual(env.t, 0)
        self.assertEqual(env.target_belief_mean.shape, (cfg.n_uavs, 1, 3))
        self.assertTrue(np.isfinite(observation['agent_observations']).all())
        env.step_joint(np.zeros((cfg.n_uavs, 4)))

    def config(self, **kwargs):
        return QuadrotorPursuitConfig(pursuit_target_observable=False, building_count=0, **kwargs)

    def test_fast_control_does_not_call_nmpc_or_output_torque(self):
        env = QuadrotorPursuitEnv(self.config())
        self.assertFalse(hasattr(env, 'nmpc_warm_starts'))
        result = env.step_joint(np.zeros((3, 4)))
        self.assertNotIn('quadrotor_controls', result)
        self.assertNotIn('nmpc_feasible_rate', result)
        self.assertIn('controller_feasible_rate', result)
        self.assertEqual(np.asarray(result['body_angular_rates']).shape, (3, 3))
        np.testing.assert_array_equal(result['obs']['agent_observations'], env.observe_search()['agent_observations'])

    def test_velocity_and_rate_limits_and_hover(self):
        cfg = self.config()
        state = np.zeros(13)
        state[:3] = [5, 5, 5]
        state[6] = 1
        np.testing.assert_allclose(integrate_velocity_reference(state, np.zeros(3), 0, .2, cfg), state)
        initial = state.copy()
        for _ in range(10):
            state = integrate_velocity_reference(state, [100, 100, 100], 100, .2, cfg)
            self.assertLessEqual(np.linalg.norm(state[3:5]), cfg.max_horizontal_velocity + 1e-10)
            self.assertLessEqual(abs(state[5]), cfg.max_vertical_velocity + 1e-10)
            self.assertLessEqual(max(abs(state[10:13])), cfg.max_body_rate + 1e-10)
            self.assertAlmostEqual(np.linalg.norm(state[6:10]), 1.0)
        self.assertGreater(np.linalg.norm(state[:3]-initial[:3]), 0.1)

    def test_lidar_range_elevation_rear_and_occlusion(self):
        env = QuadrotorPursuitEnv(self.config(lidar_detection_probability=1))
        env.quadrotor_states[0, :3] = [10, 10, 5]
        env.quadrotor_states[0, 6:10] = [1, 0, 0, 0]
        for point, expected in (([11, 10, 5], True), ([9, 10, 5], True),
                                ([10.1, 10, 5], False), ([23, 10, 5], False), ([10, 10, 8], False)):
            env.dynamic_targets[0], env.target_altitudes[0] = point[:2], point[2]
            self.assertEqual(bool(lidar_visibility(env)[0, 0]), expected)
        env.dynamic_targets[0], env.target_altitudes[0] = [11, 10], 5
        env.buildings, env.building_heights = [(10.4, 9, 10.6, 11)], [10]
        self.assertFalse(lidar_visibility(env)[0, 0])

    def test_mount_and_attitude_rotate_lidar_fov(self):
        env = QuadrotorPursuitEnv(self.config(lidar_horizontal_fov_deg=90))
        env.quadrotor_states[0, :3] = [10, 10, 5]
        env.quadrotor_states[0, 6:10] = [1, 0, 0, 0]
        env.dynamic_targets[0], env.target_altitudes[0] = [9, 10], 5
        self.assertFalse(lidar_visibility(env)[0, 0])
        env.cfg.lidar_mount_rpy_deg = (0, 0, 180)
        self.assertTrue(lidar_visibility(env)[0, 0])
        env.cfg.lidar_mount_rpy_deg = (0, 0, 0)
        env.quadrotor_states[0, 6:10] = quaternion_from_yaw(np.pi)
        self.assertTrue(lidar_visibility(env)[0, 0])

    def test_complete_dropout_does_not_inject_truth_measurements(self):
        env = QuadrotorPursuitEnv(self.config(lidar_detection_probability=0, handoff_initial_track=False))
        for _ in range(3):
            result = env.step_joint(np.zeros((3, 4)))
            self.assertFalse(env.direct_visibility_mask().any())
            self.assertEqual(result['lidar_detection_ratio'], 0)
            self.assertFalse(any(track.initialized for tracks in env.track_memory for track in tracks))

    def test_scan_rate_and_repeated_observation_do_not_resample(self):
        env = QuadrotorPursuitEnv(self.config(lidar_detection_probability=1, lidar_scan_hz=1,
                                           dynamic_targets_enabled=False, lidar_vertical_min_deg=-89,
                                           lidar_vertical_max_deg=89))
        env.observe_search()
        stamps = env.last_direct_detection_step.copy()
        rng_state = env._lidar_rng.bit_generator.state
        env.observe_search()
        self.assertEqual(rng_state, env._lidar_rng.bit_generator.state)
        for _ in range(4):
            env.step_joint(np.zeros((3, 4)))
        np.testing.assert_array_equal(env.last_direct_detection_step, stamps)
        env.step_joint(np.zeros((3, 4)))
        self.assertTrue(np.any(env.last_direct_detection_step > stamps))

    def test_game_observation_is_noisy_and_independent_of_lidar(self):
        cfg = QuadrotorPursuitConfig(building_count=0, lidar_detection_probability=0)
        env = QuadrotorPursuitEnv(cfg)
        for _ in range(3):
            result = env.step_joint(np.zeros((cfg.n_uavs, 4)))
            self.assertEqual(result['direct_target_visible'], 1.)
            self.assertEqual(result['target_observation_ratio'], 1.)
            self.assertNotIn('lidar_detection_ratio', result)
            self.assertEqual(result['sensor_mode'], 'noisy_game_observation')
        env._game_measurement_steps.clear()
        measurement, variance = env._target_measurement(0, 0)
        truth = np.r_[env.dynamic_targets[0], env.target_altitudes[0]]
        self.assertGreater(np.linalg.norm(measurement-truth), 0)
        self.assertEqual(variance, cfg.game_position_noise_std**2)
        self.assertIsNone(env._target_measurement(0, 0)[0])

    def test_game_evader_moves_away_from_nearby_pursuers(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(building_count=0))
        env.dynamic_targets[0], env.target_altitudes[0] = [10, 10], 5
        env.positions[:] = [[8, 9], [8, 10], [8, 11]]
        env.altitudes[:] = 5
        self.assertGreater(env._game_escape_direction(0)[0], 0)


    def test_batched_3d_fusion_matches_scalar_grid(self):
        from target_tracker import covariance_intersection, _stable_inverse_spd, _symmetrize_positive
        rng = np.random.default_rng(56)
        for _ in range(8):
            a, b = rng.normal(size=(6, 6)), rng.normal(size=(6, 6))
            p1, p2 = a@a.T+np.eye(6)*0.2, b@b.T+np.eye(6)*0.2
            m1, m2 = rng.normal(size=6), rng.normal(size=6)
            i1, i2 = _stable_inverse_spd(_symmetrize_positive(p1)), _stable_inverse_spd(_symmetrize_positive(p2))
            best = None
            for weight in np.linspace(0, 1, 101):
                info = weight*i1+(1-weight)*i2
                sign, objective = np.linalg.slogdet(info)
                if sign > 0 and (best is None or objective > best[0]):
                    covariance = _stable_inverse_spd(info)
                    mean = covariance@(weight*i1@m1+(1-weight)*i2@m2)
                    best = (objective, mean, _symmetrize_positive(covariance), weight)
            mean, covariance, weight = covariance_intersection(m1, p1, m2, p2)
            self.assertEqual(weight, best[3])
            np.testing.assert_allclose(mean, best[1], atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(covariance, best[2], atol=1e-11, rtol=1e-11)


if __name__ == '__main__':
    unittest.main()
