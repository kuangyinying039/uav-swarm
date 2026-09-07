import unittest

import numpy as np

try:
    from target_tracker import (
        TrackMessage,
        TrackState,
        TrackState3D,
        covariance_intersection,
        freshness_aware_covariance_intersection,
        freshness_score,
        fuse_track_with_message,
    )
except ImportError:
    from .target_tracker import (
        TrackMessage,
        TrackState,
        TrackState3D,
        covariance_intersection,
        freshness_aware_covariance_intersection,
        freshness_score,
        fuse_track_with_message,
    )


class TargetTrackerTests(unittest.TestCase):
    def test_3d_constant_velocity_filter_tracks_xyz_and_velocity(self):
        track = TrackState3D.uninitialized(0)
        for timestamp in range(14):
            if timestamp:
                track.predict(0.5, 0.01)
            truth = np.array([timestamp, -0.5 * timestamp, 0.25 * timestamp])
            track.update_xyz(truth, timestamp, 0.01)
        self.assertEqual(track.mean.shape, (6,))
        self.assertEqual(track.covariance.shape, (6, 6))
        np.testing.assert_allclose(track.mean[:3], [13.0, -6.5, 3.25], atol=0.15)
        np.testing.assert_allclose(track.mean[3:], [2.0, -1.0, 0.5], atol=0.2)
        self.assertTrue(np.all(np.linalg.eigvalsh(track.covariance) > 0.0))

    def test_freshness_aware_ci_supports_six_dimensional_tracks(self):
        first = TrackState3D.uninitialized(0)
        second = TrackState3D.uninitialized(0)
        first.initialize(np.array([0.0, 0.0, 3.0]), 10, 1.0, 1.0)
        second.initialize(np.array([8.0, 0.0, 7.0]), 0, 1.0, 1.0)
        fuse_track_with_message(
            first,
            TrackMessage.from_track(1, second),
            now=10,
            freshness_tau=2.0,
            link_confidence=0.2,
        )
        self.assertEqual(first.mean.shape, (6,))
        self.assertLess(first.mean[0], 2.0)
        self.assertLess(first.mean[2], 4.0)

    def test_track_starts_unknown_and_initializes_from_measurement(self):
        track = TrackState.uninitialized(3)
        self.assertFalse(track.initialized)
        track.update_xy(np.array([4.0, 7.0]), 2, 0.04)
        self.assertTrue(track.initialized)
        np.testing.assert_allclose(track.mean[:2], [4.0, 7.0])
        np.testing.assert_allclose(track.mean[2:], [0.0, 0.0])
        self.assertEqual(track.timestamp, 2)

    def test_prediction_increases_uncertainty_and_measurement_reduces_it(self):
        track = TrackState.uninitialized(0)
        track.update_xy(np.array([1.0, 2.0]), 0, 0.01)
        initial_trace = track.position_trace()
        track.predict(1.0, 0.2)
        predicted_trace = track.position_trace()
        gain = track.update_xy(np.array([1.8, 2.1]), 1, 0.01)
        self.assertGreater(predicted_trace, initial_trace)
        self.assertLess(track.position_trace(), predicted_trace)
        self.assertGreater(gain, 0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(track.covariance) > 0.0))

    def test_constant_velocity_filter_estimates_velocity(self):
        track = TrackState.uninitialized(0)
        for timestamp in range(12):
            if timestamp:
                track.predict(1.0, 0.01)
            track.update_xy(np.array([2.0 * timestamp, -timestamp]), timestamp, 0.01)
        np.testing.assert_allclose(track.mean[2:], [2.0, -1.0], atol=0.15)

    def test_covariance_intersection_is_symmetric_positive_definite(self):
        mean, covariance, weight = covariance_intersection(
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.diag([2.0, 1.0, 1.0, 1.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.diag([1.0, 2.0, 1.0, 1.0]),
        )
        self.assertEqual(mean.shape, (4,))
        self.assertTrue(0.0 <= weight <= 1.0)
        np.testing.assert_allclose(covariance, covariance.T)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0.0))

    def test_message_initializes_unknown_receiver_without_truth_access(self):
        source = TrackState.uninitialized(0)
        source.update_xy(np.array([5.0, 6.0]), 4, 0.04)
        receiver = TrackState.uninitialized(0)
        fuse_track_with_message(receiver, TrackMessage.from_track(1, source))
        self.assertTrue(receiver.initialized)
        np.testing.assert_allclose(receiver.mean, source.mean)
        self.assertEqual(receiver.timestamp, 4)

    def test_freshness_score_decreases_with_age_and_link_degradation(self):
        fresh = freshness_score(age=0, tau=10.0, link_confidence=1.0)
        stale = freshness_score(age=20, tau=10.0, link_confidence=1.0)
        weak = freshness_score(age=0, tau=10.0, link_confidence=0.2)
        self.assertGreater(fresh, stale)
        self.assertGreater(fresh, weak)
        self.assertGreaterEqual(stale, 0.05)

    def test_freshness_aware_ci_prefers_the_newer_estimate(self):
        local_mean = np.array([0.0, 0.0, 0.0, 0.0])
        peer_mean = np.array([10.0, 0.0, 0.0, 0.0])
        covariance = np.eye(4)
        toward_fresh_peer, _, _ = freshness_aware_covariance_intersection(
            local_mean,
            covariance,
            peer_mean,
            covariance,
            first_age=20,
            second_age=0,
            freshness_tau=5.0,
        )
        away_from_stale_peer, _, _ = freshness_aware_covariance_intersection(
            local_mean,
            covariance,
            peer_mean,
            covariance,
            first_age=0,
            second_age=20,
            freshness_tau=5.0,
        )
        self.assertGreater(toward_fresh_peer[0], 8.0)
        self.assertLess(away_from_stale_peer[0], 2.0)

    def test_message_fusion_uses_age_and_link_confidence(self):
        local = TrackState.uninitialized(0)
        local.initialize(np.array([0.0, 0.0]), 10, 1.0, 1.0)
        source = TrackState.uninitialized(0)
        source.initialize(np.array([10.0, 0.0]), 0, 1.0, 1.0)
        fuse_track_with_message(
            local,
            TrackMessage.from_track(1, source),
            now=10,
            freshness_tau=2.0,
            link_confidence=0.1,
        )
        self.assertLess(local.mean[0], 2.0)


if __name__ == "__main__":
    unittest.main()
