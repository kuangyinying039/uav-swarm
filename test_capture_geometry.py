"""Unit tests for the shared capture_geometry API."""

from __future__ import annotations

import unittest

import numpy as np

from capture_geometry import (
    angular_span_from_cfg,
    evaluate_capture,
    evaluate_capture_batch,
    horizontal_angular_span_deg,
)


class CaptureGeometryTests(unittest.TestCase):
    def test_single_uav_inside_radius_is_not_success_when_required_two(self):
        pursuers = np.array([[0.3, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=float)
        result = evaluate_capture(
            pursuers, np.zeros(3), radius=1.0, required_uavs=2, angular_span_deg=0.0
        )
        self.assertFalse(result.success)
        self.assertEqual(result.close_count, 1)
        self.assertAlmostEqual(result.min_distance, 0.3)
        self.assertGreater(result.kth_gap, 0.0)

    def test_two_uavs_inside_radius_distance_only_succeeds(self):
        # Same bearing — angular span ~0; distance-only must still succeed.
        pursuers = np.array([[0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
        result = evaluate_capture(
            pursuers, np.zeros(3), radius=1.0, required_uavs=2, angular_span_deg=0.0
        )
        self.assertTrue(result.success)
        self.assertEqual(result.close_count, 2)
        self.assertEqual(result.kth_gap, 0.0)

    def test_two_uavs_same_side_fails_when_angular_span_required(self):
        pursuers = np.array([[0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
        result = evaluate_capture(
            pursuers, np.zeros(3), radius=1.0, required_uavs=2, angular_span_deg=100.0
        )
        self.assertFalse(result.success)
        self.assertEqual(result.close_count, 2)
        self.assertLess(result.angular_span_deg, 100.0)

    def test_opposite_bearings_satisfy_angular_span(self):
        pursuers = np.array([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float)
        result = evaluate_capture(
            pursuers, np.zeros(3), radius=1.0, required_uavs=2, angular_span_deg=100.0
        )
        self.assertTrue(result.success)
        self.assertGreaterEqual(result.angular_span_deg, 100.0)

    def test_disabled_uav_cannot_count_toward_capture(self):
        pursuers = np.array([[0.4, 0.0, 0.0], [0.5, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=float)
        active = np.array([True, False, False])
        result = evaluate_capture(
            pursuers,
            np.zeros(3),
            radius=1.0,
            required_uavs=2,
            angular_span_deg=0.0,
            active_mask=active,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.close_count, 1)

    def test_batch_matches_scalar(self):
        batch_pos = np.array(
            [
                [[0.4, 0.0, 0.0], [0.7, 0.0, 0.0], [4.0, 0.0, 0.0]],
                [[0.4, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            ],
            dtype=float,
        )
        batch_tgt = np.zeros((2, 3), dtype=float)
        out = evaluate_capture_batch(
            batch_pos, batch_tgt, radius=1.0, required_uavs=2, angular_span_deg=0.0
        )
        self.assertTrue(bool(out["success"][0]))
        self.assertFalse(bool(out["success"][1]))
        a = evaluate_capture(batch_pos[0], batch_tgt[0], radius=1.0, required_uavs=2)
        b = evaluate_capture(batch_pos[1], batch_tgt[1], radius=1.0, required_uavs=2)
        self.assertEqual(bool(out["success"][0]), a.success)
        self.assertEqual(bool(out["success"][1]), b.success)
        np.testing.assert_allclose(out["kth_gap"][0], a.kth_gap)
        np.testing.assert_allclose(out["kth_gap"][1], b.kth_gap)

    def test_horizontal_span_helper_opposite(self):
        span = horizontal_angular_span_deg(
            np.array([[1.0, 0.0], [-1.0, 0.0]]), np.zeros(2)
        )
        self.assertAlmostEqual(span, 180.0, places=5)

    def test_cfg_single_distance_disables_angular_gate(self):
        cfg = type("C", (), {"capture_mode": "single_distance", "capture_angular_span_deg": 100.0})()
        self.assertEqual(angular_span_from_cfg(cfg), 0.0)
        cfg2 = type("C", (), {"capture_mode": "angular_enclose", "capture_angular_span_deg": 100.0})()
        self.assertEqual(angular_span_from_cfg(cfg2), 100.0)


if __name__ == "__main__":
    unittest.main()
