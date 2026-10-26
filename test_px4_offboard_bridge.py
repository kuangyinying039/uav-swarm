"""Offline tests for PX4 offboard mapping (no ROS required)."""

from __future__ import annotations

import unittest

import numpy as np

from px4_offboard_bridge import (
    FleetOffboardBridge,
    OffboardLimits,
    Px4OffboardBridge,
    enu_velocity_to_ned,
    ned_velocity_to_enu,
    policy_action_to_world_velocity,
)


class Px4BridgeTests(unittest.TestCase):
    def test_enu_ned_roundtrip(self):
        ned = enu_velocity_to_ned(1.0, 2.0, 3.0)
        self.assertEqual(ned, (2.0, 1.0, -3.0))
        self.assertEqual(ned_velocity_to_enu(*ned), (1.0, 2.0, 3.0))

    def test_action_scaling_and_horizontal_cap(self):
        limits = OffboardLimits(max_horizontal_velocity=1.4, max_vertical_velocity=0.8, max_yaw_rate=1.4)
        cmd = policy_action_to_world_velocity(np.array([1.0, 1.0, 1.0, 1.0]), limits)
        self.assertAlmostEqual(float(np.linalg.norm(cmd[:2])), 1.4, places=5)
        self.assertAlmostEqual(cmd[2], 0.8)
        self.assertAlmostEqual(cmd[3], 1.4)

    def test_publisher_receives_ned_velocity(self):
        published = []

        class FakePub:
            def publish(self, msg):
                published.append(msg)

        class FakePT:
            IGNORE_PX = 1
            IGNORE_PY = 2
            IGNORE_PZ = 4
            IGNORE_AFX = 8
            IGNORE_AFY = 16
            IGNORE_AFZ = 32
            IGNORE_YAW = 64
            FRAME_LOCAL_NED = 1

            def __call__(self):
                return type("Msg", (), {"header": type("H", (), {"stamp": None, "frame_id": ""})(), "velocity": type("V", (), {"x": 0, "y": 0, "z": 0})(), "yaw_rate": 0.0, "type_mask": 0, "coordinate_frame": 0})()

        bridge = Px4OffboardBridge("/uav0", publisher=FakePub())
        bridge._PositionTarget = FakePT()
        bridge._publisher = FakePub()
        bridge.send_policy_action(np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(len(published), 1)
        # ENU +x east → NED +y east
        self.assertAlmostEqual(published[0].velocity.y, 1.4, places=5)
        self.assertAlmostEqual(published[0].velocity.x, 0.0, places=5)

    def test_fleet_capture_uses_shared_geometry(self):
        fleet = FleetOffboardBridge(
            ["/uav0", "/uav1", "/uav2", "/uav3"],
            capture_radius=1.0,
            capture_required_uavs=2,
            capture_mode="single_distance",
        )
        for i, v in enumerate(fleet.vehicles[:3]):
            v._pose.xyz = np.array([0.4 if i < 2 else 5.0, 0.0, 1.0])
        fleet.vehicles[3]._pose.xyz = np.array([0.0, 0.0, 1.0])
        result = fleet.evaluate_capture_now()
        self.assertTrue(result["success"])
        self.assertEqual(result["close_count"], 2)


if __name__ == "__main__":
    unittest.main()
