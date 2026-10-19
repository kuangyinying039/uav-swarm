from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import numpy as np

OMNIDRONES_ROOT = Path(__file__).resolve().parents[2] / "omnidrones"
if str(OMNIDRONES_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIDRONES_ROOT))

from isaac_pursuit_backend import IsaacPursuitEnv
from quadrotor_pursuit_env import QuadrotorPursuitConfig
from experiments.radar5_frozen_transfer_v1.quadrotor_pursuit_env import (
    QuadrotorPursuitConfig as ExperimentPursuitConfig,
)


class IsaacCaptureContractTests(unittest.TestCase):
    def test_requires_two_active_uavs_and_uses_second_closest_gap(self):
        env = object.__new__(IsaacPursuitEnv)
        env.cfg = QuadrotorPursuitConfig(building_count=0, capture_required_uavs=2)
        env.bridge = SimpleNamespace(
            env=SimpleNamespace(disabled_uavs=np.zeros(3, dtype=bool))
        )
        env.bridges = [env.bridge]
        physical = np.zeros((3, 13), dtype=float)
        physical[:, :3] = [[0.5, 0.0, 0.0], [1.2, 0.0, 0.0], [3.0, 0.0, 0.0]]
        env._physical_state = lambda env_index=0: physical
        env._target_position = lambda env_index=0: np.zeros(3)

        self.assertFalse(env._has_physical_capture(physical, np.zeros(3)))
        self.assertAlmostEqual(env.minimum_capture_gap(), 0.2)

        physical[1, 0] = 0.8
        self.assertTrue(env._has_physical_capture(physical, np.zeros(3)))
        self.assertEqual(env.minimum_capture_gap(), 0.0)

        env.bridge.env.disabled_uavs[1] = True
        self.assertFalse(env._has_physical_capture(physical, np.zeros(3)))
        self.assertAlmostEqual(env.minimum_capture_gap(), 2.0)

    def test_kinematic_capture_does_not_count_or_end_the_episode(self):
        env = object.__new__(IsaacPursuitEnv)
        env.cfg = QuadrotorPursuitConfig(building_count=0, capture_required_uavs=2, capture_reward=100.0)
        env.bridges = [SimpleNamespace(env=SimpleNamespace(disabled_uavs=np.zeros(3, dtype=bool)))]
        env._capture = np.array([False])
        env._steps = np.array([10])
        env._safety_reasons = [None]
        env._episode_return = np.zeros(1)
        env._physical_potential = np.zeros(1)
        env.num_envs = 1
        physical = np.zeros((3, 13), dtype=float)
        physical[:, :3] = [[6.0, 0.0, 0.0], [6.0, 1.0, 0.0], [8.0, 0.0, 0.0]]
        env._physical_state = lambda env_index=0: physical
        env._target_position = lambda env_index=0: np.zeros(3)
        env._native_lidar_enabled = False
        env._controller_feasible_rate = lambda *args, **kwargs: 1.0
        result = env._finalize_env_result(0, {
            "capture_success": True,
            "terminated": True,
            "truncated": False,
            "reward": 100.0,
            "reward_components": {"capture": 100.0},
        }, [])
        self.assertFalse(result["capture_success"])
        self.assertFalse(result["terminated"])
        self.assertEqual(result["reward_components"]["capture"], 0.0)
        self.assertAlmostEqual(
            result["reward"],
            result["reward_components"]["physical_capture_progress"],
        )
        self.assertGreater(result["minimum_capture_gap"], 1.0)

    def test_physical_gap_shaping_rewards_closing_and_zeros_at_capture(self):
        env = object.__new__(IsaacPursuitEnv)
        env.cfg = QuadrotorPursuitConfig(building_count=0, capture_required_uavs=2, capture_reward=100.0)
        env.num_envs = 1
        env._physical_potential = np.array([env._gap_potential(4.0)])
        far = env._apply_physical_gap_shaping(0, 4.0, False)
        self.assertLess(far, 0.0)
        closer = env._apply_physical_gap_shaping(0, 1.0, False)
        self.assertGreater(closer, 0.0)
        terminal = env._apply_physical_gap_shaping(0, 0.0, True)
        self.assertLess(terminal, 0.0)
        self.assertEqual(env._physical_potential[0], 0.0)

    def test_default_isaac_training_config_is_dual_hard(self):
        from train_isaac_pursuit import build_parser

        args = build_parser().parse_args(["mappo", "--out", "/tmp/isaac-capture-test"])
        self.assertEqual(args.env_config.name, "radar5_dual_hard.json")

    def test_task_timeout_is_terminal_but_external_cutoff_is_truncated(self):
        self.assertEqual(
            IsaacPursuitEnv._episode_end_flags(False, True, False, True),
            (True, False),
        )
        self.assertEqual(
            IsaacPursuitEnv._episode_end_flags(False, False, False, True),
            (False, True),
        )

    def test_both_config_definitions_accept_all_four_dual_levels(self):
        from pathlib import Path
        import json

        config_dir = Path(__file__).parent / "configs/pursuit_v2"
        levels = ("nominal", "medium", "hard", "severe", "extreme")
        for level in levels:
            values = json.loads((config_dir / f"radar5_dual_{level}.json").read_text())
            canonical = QuadrotorPursuitConfig(**values)
            mirror = ExperimentPursuitConfig(**values)
            self.assertEqual(canonical.capture_required_uavs, 2)
            self.assertEqual(mirror.capture_required_uavs, 2)
            self.assertEqual(canonical.execution_reward_version, 5)
            self.assertEqual(mirror.execution_reward_version, 5)
            self.assertGreater(canonical.capture_team_progress_weight, 0.0)
            self.assertGreater(mirror.capture_team_progress_weight, 0.0)

    def test_soft_reset_skips_second_formation_reset(self):
        env = object.__new__(IsaacPursuitEnv)
        env.cfg = QuadrotorPursuitConfig(building_count=0, n_uavs=3, capture_required_uavs=2)
        env.lidar = SimpleNamespace(reset=lambda: None)
        env._physics_initialized = False
        env._last_native_scan = None
        env._steps = 7
        env._capture = True
        env._needs_reset = True
        calls = {"formation_reset": 0, "bridge_reset": 0, "teleport": 0, "write_target": 0}

        class _Sim:
            def reset(self):
                calls["formation_reset"] += 1
                return {"td": True}

        class _Bridge:
            def __init__(self):
                self.env = SimpleNamespace(
                    cfg=env.cfg,
                    rng=None,
                    quadrotor_states=np.zeros((3, 13), dtype=float),
                )

            def reset(self):
                calls["bridge_reset"] += 1
                self.env.quadrotor_states[:, 6] = 1.0
                return {}

            def observe(self):
                return {"obs": True}

            def clear_native_sensor_scan(self):
                return None

        env.sim_env = _Sim()
        env.bridge = _Bridge()
        env._set_building_transforms = lambda: None
        env._teleport_pursuers_from_bridge = lambda: calls.__setitem__("teleport", calls["teleport"] + 1)
        env._write_target_pose = lambda: calls.__setitem__("write_target", calls["write_target"] + 1)
        env._sync_bridge = lambda: None
        env._read_native_lidar = lambda: (_ for _ in ()).throw(RuntimeError("no lidar"))

        first = IsaacPursuitEnv.reset(env, seed=11)
        second = IsaacPursuitEnv.reset(env, seed=12)
        self.assertEqual(first, {"obs": True})
        self.assertEqual(second, {"obs": True})
        self.assertEqual(calls["formation_reset"], 1)
        self.assertEqual(calls["bridge_reset"], 2)
        self.assertEqual(calls["teleport"], 2)
        self.assertTrue(env._physics_initialized)
        self.assertEqual(env._steps, 0)
        self.assertFalse(env._capture)
        self.assertFalse(env._needs_reset)

    def test_experiment_and_repro_rewards_share_dual_team_shaping(self):
        from pursuit_rewards import shaped_rewards as repro_shaped
        from experiments.radar5_frozen_transfer_v1.pursuit_rewards import (
            shaped_rewards as experiment_shaped,
        )

        cfg = QuadrotorPursuitConfig(building_count=0, capture_required_uavs=2)
        previous = (np.array([1.0, 1.1, 1.2]), 0.0, 0.0)
        current = (np.array([0.8, 0.9, 1.0]), 0.0, 0.0)
        active = np.ones(3, dtype=bool)
        repro_rewards, _ = repro_shaped(previous, current, active, active, cfg)
        experiment_rewards, _ = experiment_shaped(previous, current, active, active, cfg)
        self.assertIn("capture_team_progress", repro_rewards)
        self.assertIn("capture_team_progress", experiment_rewards)
        self.assertNotIn("nearest_approach", repro_rewards)
        self.assertNotIn("nearest_approach", experiment_rewards)
        self.assertAlmostEqual(
            repro_rewards["capture_team_progress"],
            experiment_rewards["capture_team_progress"],
        )

    def test_sync_pursuers_keeps_omnidrones_wxyz_identity(self):
        from experiments.radar5_frozen_transfer_v1.isaac_pursuit_observation import (
            IsaacPursuitObservationBridge,
        )
        from experiments.radar5_frozen_transfer_v1.quadrotor_pursuit_env import (
            QuadrotorPursuitConfig as ExperimentConfig,
        )

        bridge = IsaacPursuitObservationBridge(
            ExperimentConfig(building_count=0, n_obstacles=0, search_steps=3)
        )
        states = np.zeros((3, 13), dtype=float)
        states[:, 2] = 2.0
        states[:, 6] = 1.0  # identity wxyz
        bridge.sync_pursuers(states)
        self.assertTrue(np.allclose(bridge.env.quadrotor_states[:, 6:10], states[:, 6:10]))
        self.assertTrue(np.allclose(bridge.env.altitudes, 2.0))

    def test_controller_feasible_rate_requires_upright_airborne_uavs(self):
        from experiments.radar5_frozen_transfer_v1.pursuit_kinematics import (
            quaternion_from_yaw,
        )

        env = object.__new__(IsaacPursuitEnv)
        env.cfg = QuadrotorPursuitConfig(building_count=0, capture_required_uavs=2)
        env.bridge = SimpleNamespace(
            env=SimpleNamespace(disabled_uavs=np.zeros(3, dtype=bool))
        )
        physical = np.zeros((3, 13), dtype=float)
        physical[:, 2] = [2.0, 2.0, -1.0]
        for index in range(3):
            physical[index, 6:10] = quaternion_from_yaw(0.0)
        # Flip the second UAV upside-down: 180 deg about x => wxyz (0,1,0,0)
        physical[1, 6:10] = [0.0, 1.0, 0.0, 0.0]
        self.assertAlmostEqual(env._controller_feasible_rate(physical), 1.0 / 3.0)