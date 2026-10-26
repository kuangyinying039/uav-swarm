"""Unit tests for Viewport overlay geometry helpers (no Isaac launch)."""
from __future__ import annotations

import unittest

from isaac_cli_flags import resolve_headless
from isaac_pursuit_viewport_overlay import (
    closed_polyline_segments,
    horizontal_circle_points,
    open_polyline_segments,
    sphere_wireframe_segments,
)


class FakeArgs:
    def __init__(self, headless=True, visualize=False):
        self.headless = headless
        self.visualize = visualize


class ViewportOverlayHelpersTest(unittest.TestCase):
    def test_visualize_forces_gui(self):
        self.assertFalse(resolve_headless(FakeArgs(headless=True, visualize=True)))
        self.assertTrue(resolve_headless(FakeArgs(headless=True, visualize=False)))
        self.assertFalse(resolve_headless(FakeArgs(headless=False, visualize=False)))

    def test_horizontal_circle_closed(self):
        points = horizontal_circle_points((1.0, 2.0, 3.0), 4.0, segments=12)
        self.assertEqual(len(points), 12)
        starts, ends = closed_polyline_segments(points)
        self.assertEqual(len(starts), 12)
        self.assertEqual(ends[-1], points[0])
        # All z coordinates stay at the center height.
        self.assertTrue(all(abs(p[2] - 3.0) < 1e-9 for p in points))

    def test_sphere_wireframe_has_three_rings(self):
        starts, ends = sphere_wireframe_segments((0, 0, 1), 1.0, segments=24)
        self.assertEqual(len(starts), 24 * 3)
        self.assertEqual(len(starts), len(ends))

    def test_open_polyline(self):
        starts, ends = open_polyline_segments([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        self.assertEqual(starts, [(0, 0, 0), (1, 0, 0)])
        self.assertEqual(ends, [(1, 0, 0), (2, 0, 0)])
        self.assertEqual(open_polyline_segments([(0, 0, 0)]), ([], []))


if __name__ == "__main__":
    unittest.main()
