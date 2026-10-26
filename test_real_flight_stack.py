"""Offline tests for onboard obs, safety filter, and 5 Hz inference path."""

from __future__ import annotations

import time
import unittest

import numpy as np

from px4_offboard_bridge import FleetOffboardBridge, OffboardLimits, VehiclePose
from real_flight_node import DecisionClock, RealFlightController, make_expert_fn
from real_flight_obs import (
    FleetEstimates,
    OnboardObsBuilder,
    TargetTrackEstimate,
    broadcast_track,
    pose_to_quadrotor_state,
)
from real_flight_safety import SafetyFilter, SafetyLimits
from residual_policy import ResidualPolicy


class RealFlightObsTests(unittest.TestCase):
    def test_pose_pack_matches_yaw(self):
        pose = VehiclePose(xyz=np.array([1.0, 2.0, 3.0]), yaw=0.5, xyz_vel=np.array([0.1, 0.0, 0.0]))
        state = pose_to_quadrotor_state(pose)
        self.assertEqual(state.shape, (13,))
        np.testing.assert_allclose(state[:3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(state[3]), 0.1)

    def test_observe_matches_training_layout(self):
        builder = OnboardObsBuilder(seed=1)
        poses = [
            VehiclePose(xyz=np.array([2.0, 0.0, 2.0]), yaw=0.0, stamp_s=time.time()),
            VehiclePose(xyz=np.array([-2.0, 0.0, 2.0]), yaw=np.pi, stamp_s=time.time()),
            VehiclePose(xyz=np.array([0.0, 2.0, 2.0]), yaw=1.0, stamp_s=time.time()),
        ]
        tracks = broadcast_track(np.array([0.0, 0.0, 2.0]), n_uavs=3)
        obs = builder.observe(FleetEstimates(vehicles=poses, tracks=tracks, tick=3))
        self.assertIn("agent_observations", obs)
        self.assertIn("hetero_graph", obs)
        self.assertEqual(obs["agent_observations"].shape[0], 3)
        self.assertEqual(obs["agent_observations"].shape[1], builder.env.obs_dim())
        self.assertTrue(obs["hetero_graph"]["target_mask"].any())
        # Injected positions visible to graph.
        np.testing.assert_allclose(obs["hetero_graph"]["uav_xyz"][0], [2.0, 0.0, 2.0], atol=1e-5)


class SafetyFilterTests(unittest.TestCase):
    def test_stale_odom_holds(self):
        filt = SafetyFilter(SafetyLimits(max_odom_age_s=0.2, min_peer_distance_m=0.5))
        poses = [
            VehiclePose(xyz=np.array([0.0, 0.0, 2.0]), yaw=0.0, stamp_s=time.time() - 1.0),
            VehiclePose(xyz=np.array([3.0, 0.0, 2.0]), yaw=0.0, stamp_s=time.time()),
            VehiclePose(xyz=np.array([0.0, 3.0, 2.0]), yaw=0.0, stamp_s=time.time()),
        ]
        actions = np.ones((3, 4))
        out, report = filt.apply(actions, poses, now_s=time.time())
        self.assertTrue(report.hold)
        np.testing.assert_allclose(out, 0.0)

    def test_peer_close_holds(self):
        filt = SafetyFilter(SafetyLimits(max_odom_age_s=5.0, min_peer_distance_m=1.5))
        now = time.time()
        poses = [
            VehiclePose(xyz=np.array([0.0, 0.0, 2.0]), yaw=0.0, stamp_s=now),
            VehiclePose(xyz=np.array([0.5, 0.0, 2.0]), yaw=0.0, stamp_s=now),
            VehiclePose(xyz=np.array([4.0, 0.0, 2.0]), yaw=0.0, stamp_s=now),
        ]
        out, report = filt.apply(np.ones((3, 4)), poses, now_s=now)
        self.assertTrue(report.hold)
        self.assertLess(report.peer_min_distance_m, 1.5)
        np.testing.assert_allclose(out, 0.0)


class DecisionClockTests(unittest.TestCase):
    def test_five_hz_spacing(self):
        clock = DecisionClock(hz=5.0)
        stamps = [clock.sleep_to_next() for _ in range(6)]
        gaps = np.diff(stamps)
        self.assertTrue(np.all(gaps > 0.15))
        self.assertTrue(np.all(gaps < 0.25))
        self.assertEqual(clock.tick, 6)


class InferenceLoopTests(unittest.TestCase):
    def test_dry_run_residual_tick(self):
        bridge = FleetOffboardBridge(
            ["/uav0", "/uav1", "/uav2", "/uav3"],
            limits=OffboardLimits(decision_dt=0.2),
        )
        now = time.time()
        for i, v in enumerate(bridge.vehicles):
            angle = 2.0 * np.pi * i / 4.0
            v._pose = VehiclePose(
                xyz=np.array([3.0 * np.cos(angle), 3.0 * np.sin(angle), 2.0]),
                yaw=angle,
                stamp_s=now,
            )
        builder = OnboardObsBuilder(seed=2)
        expert = make_expert_fn(builder.env, teacher="frpn")
        rl = lambda obs, env_index=0: np.zeros((3, 4), dtype=np.float32)
        policy = ResidualPolicy(expert, rl, residual_gain=0.3)
        ctrl = RealFlightController(
            bridge,
            obs_builder=builder,
            policy_fn=policy,
            safety=SafetyFilter(SafetyLimits(min_peer_distance_m=0.5, max_odom_age_s=5.0)),
            clock=DecisionClock(hz=5.0),
        )
        ctrl.set_shared_track(np.array([0.0, 0.0, 2.0]))
        record = ctrl.step_once(now_s=now)
        self.assertEqual(len(record["actions"]), 4)
        self.assertFalse(record["hold"])
        self.assertIn("capture", record)
        ctrl.close()


if __name__ == "__main__":
    unittest.main()
