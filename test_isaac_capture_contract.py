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
        physical = np.zeros((3, 13), dtype=float)
        physical[:, :3] = [[0.5, 0.0, 0.0], [1.2, 0.0, 0.0], [3.0, 0.0, 0.0]]
        env._physical_state = lambda: physical
        env._target_position = lambda: np.zeros(3)

        self.assertFalse(env._has_physical_capture(physical, np.zeros(3)))
        self.assertAlmostEqual(env.minimum_capture_gap(), 0.2)

        physical[1, 0] = 0.8
        self.assertTrue(env._has_physical_capture(physical, np.zeros(3)))
        self.assertEqual(env.minimum_capture_gap(), 0.0)

        env.bridge.env.disabled_uavs[1] = True
        self.assertFalse(env._has_physical_capture(physical, np.zeros(3)))
        self.assertAlmostEqual(env.minimum_capture_gap(), 2.0)

    def test_default_isaac_training_config_is_dual_medium(self):
        from train_isaac_pursuit import build_parser

        args = build_parser().parse_args(["mappo", "--out", "/tmp/isaac-capture-test"])
        self.assertEqual(args.env_config.name, "radar5_dual_medium.json")

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
        levels = ("nominal", "medium", "hard", "extreme")
        for level in levels:
            values = json.loads((config_dir / f"radar5_dual_{level}.json").read_text())
            canonical = QuadrotorPursuitConfig(**values)
            mirror = ExperimentPursuitConfig(**values)
            self.assertEqual(canonical.capture_required_uavs, 2)
            self.assertEqual(mirror.capture_required_uavs, 2)
            self.assertEqual(canonical.execution_reward_version, 5)
            self.assertEqual(mirror.execution_reward_version, 5)