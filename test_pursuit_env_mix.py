"""Tests for multi-difficulty pursuit environment mixes."""

import unittest
from pathlib import Path

from pursuit_env_mix import (
    assert_compatible_pursuit_mix,
    load_pursuit_config,
    make_pursuit_env_factory,
    pursuit_contract_key,
)
from quadrotor_pursuit_env import QuadrotorPursuitConfig


class PursuitEnvMixTests(unittest.TestCase):
    def test_dual_mix_keeps_contract_and_cycles_primary(self):
        root = Path(__file__).parent / "configs/pursuit_v2"
        paths = [
            root / "radar5_dual_nominal.json",
            root / "radar5_dual_medium.json",
            root / "radar5_dual_hard.json",
        ]
        configs = [load_pursuit_config(path) for path in paths]
        key = assert_compatible_pursuit_mix(configs)
        self.assertEqual(key, pursuit_contract_key(configs[0]))
        factory, primary = make_pursuit_env_factory(
            configs, seed_fn=lambda episode: 1000 + episode, primary_index=-1
        )
        self.assertEqual(primary.capture_required_uavs, 2)
        self.assertAlmostEqual(primary.target_speed, configs[-1].target_speed)
        self.assertTrue(callable(factory))
        chosen = [configs[episode % 3].target_speed for episode in range(3)]
        self.assertEqual(
            chosen,
            [configs[0].target_speed, configs[1].target_speed, configs[2].target_speed],
        )

    def test_incompatible_mix_is_rejected(self):
        with self.assertRaises(ValueError):
            assert_compatible_pursuit_mix(
                [
                    QuadrotorPursuitConfig(n_uavs=3, building_count=0),
                    QuadrotorPursuitConfig(n_uavs=2, building_count=0),
                ]
            )


if __name__ == "__main__":
    unittest.main()
