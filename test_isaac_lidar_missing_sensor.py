from pathlib import Path
import sys

import numpy as np
import unittest

OMNIDRONES_ROOT = Path(__file__).resolve().parents[2] / "omnidrones"
if str(OMNIDRONES_ROOT) not in sys.path:
    sys.path.insert(0, str(OMNIDRONES_ROOT))

from experiments.radar5_frozen_transfer_v1.isaac_pursuit_observation import (
    IsaacPursuitObservationBridge,
)
from experiments.radar5_frozen_transfer_v1.isaac_sensor_bridge import IsaacLidarAdapter
from experiments.radar5_frozen_transfer_v1.quadrotor_pursuit_env import (
    QuadrotorPursuitConfig,
)


class MissingLidarInterface:
    def __init__(self):
        self.calls = 0

    def get_point_cloud_data(self, path):
        self.calls += 1
        raise RuntimeError("Lidar Sensor does not exist")

    def get_prim_data(self, path):
        return []


class IsaacLidarMissingSensorTests(unittest.TestCase):
    def test_missing_sensor_soft_fails_without_permanent_blacklist(self):
        """Transient 'does not exist' during warmup must not quarantine forever."""
        adapter = IsaacLidarAdapter(1, n_targets=1, decision_dt=0.2)
        interface = MissingLidarInterface()
        scan = adapter.read_native_scan(
            sim_time=0.2,
            lidar_interface=interface,
            sensor_paths=["/World/robot_0/radar5_lidar"],
            sensor_origins=np.zeros((1, 3), dtype=float),
            sensor_rotations=np.eye(3, dtype=float)[None, :, :],
            target_prim_tokens=("radar5_target",),
        )
        self.assertEqual(scan["visible"].shape, (1, 1))
        self.assertFalse(np.any(scan["visible"]))
        self.assertEqual(scan["measurements"], {})
        self.assertFalse(scan["sensor_available"])

        next_scan = adapter.read_native_scan(
            sim_time=0.4,
            lidar_interface=interface,
            sensor_paths=["/World/robot_0/radar5_lidar"],
            sensor_origins=np.zeros((1, 3), dtype=float),
            sensor_rotations=np.eye(3, dtype=float)[None, :, :],
            target_prim_tokens=("radar5_target",),
        )
        self.assertFalse(next_scan["sensor_available"])
        # Soft-fail retries so sensors can come online after physics warmup.
        self.assertEqual(interface.calls, 2)


    def test_reset_allows_sensor_time_to_restart_for_next_episode(self):
        class WorkingLidarInterface:
            def get_point_cloud_data(self, path):
                return []

            def get_prim_data(self, path):
                return []

        adapter = IsaacLidarAdapter(1, n_targets=1, decision_dt=0.2)
        kwargs = {
            "lidar_interface": WorkingLidarInterface(),
            "sensor_paths": ["/World/robot_0/radar5_lidar"],
            "sensor_origins": np.zeros((1, 3), dtype=float),
            "sensor_rotations": np.eye(3, dtype=float)[None, :, :],
            "target_prim_tokens": ("radar5_target",),
        }
        adapter.read_native_scan(sim_time=0.4, **kwargs)
        adapter.reset()
        scan = adapter.read_native_scan(sim_time=0.2, **kwargs)
        self.assertTrue(scan["sensor_available"])
        self.assertEqual(scan["scan_sequence"], 1)


    def test_clearing_native_scan_restores_geometric_lidar_fallback(self):
        adapter = IsaacLidarAdapter(3, n_targets=1, decision_dt=0.2)
        bridge = IsaacPursuitObservationBridge(
            QuadrotorPursuitConfig(building_count=0, n_obstacles=0),
            sensor_adapter=adapter,
        )
        bridge.set_native_sensor_scan({
            "scan_time": 0.2,
            "measurements": {},
            "variances": {},
            "visible": np.zeros((3, 1), dtype=bool),
        })
        bridge.clear_native_sensor_scan()
        self.assertIsNone(bridge.env._external_sensor_scan)
        self.assertEqual(bridge.last_sensor_diagnostics["sensor_mode"], "geometric_lidar_fallback")

    def test_repro_quadrotor_env_supports_external_sensor_hooks(self):
        from quadrotor_pursuit_env import (
            QuadrotorPursuitConfig as ReproConfig,
            QuadrotorPursuitEnv as ReproEnv,
        )

        env = ReproEnv(ReproConfig(building_count=0, n_obstacles=0, search_steps=3))
        env.set_external_sensor_scan(
            0.2,
            {(0, 0): np.array([1.0, 2.0, 3.0])},
            np.array([[True], [False], [False]], dtype=bool),
            {(0, 0): 0.01},
        )
        measurement, variance = env._target_measurement(0, 0)
        self.assertTrue(np.allclose(measurement, [1.0, 2.0, 3.0]))
        self.assertAlmostEqual(variance, 0.01)
        env.clear_external_sensor_scan()
        self.assertIsNone(env._external_sensor_scan)
