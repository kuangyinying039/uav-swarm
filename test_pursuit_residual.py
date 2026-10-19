"""Unit tests for residual / hybrid action composition."""
import unittest

import numpy as np

from pursuit_residual import ResidualComposeConfig, compose_actions, residualize_demo_actions


class PursuitResidualTests(unittest.TestCase):
    def test_blend_preserves_base_at_zero_weight(self):
        base = np.ones((3, 4), dtype=np.float32)
        rl = -np.ones((3, 4), dtype=np.float32)
        out = compose_actions(base, rl, ResidualComposeConfig(mode="blend", weight=0.0))
        np.testing.assert_allclose(out, base)

    def test_blend_interpolates(self):
        base = np.zeros((3, 4), dtype=np.float32)
        rl = np.ones((3, 4), dtype=np.float32)
        out = compose_actions(base, rl, ResidualComposeConfig(mode="blend", weight=0.25))
        np.testing.assert_allclose(out, 0.25 * np.ones((3, 4)))

    def test_residual_adds_scaled_delta(self):
        base = np.full((3, 4), 0.2, dtype=np.float32)
        delta = np.full((3, 4), 0.5, dtype=np.float32)
        out = compose_actions(base, delta, ResidualComposeConfig(mode="residual", weight=0.4))
        np.testing.assert_allclose(out, np.full((3, 4), 0.4))

    def test_residualize_roundtrip(self):
        base = np.full((3, 4), 0.5, dtype=np.float32)
        demo = np.full((3, 4), 0.7, dtype=np.float32)
        delta = residualize_demo_actions(demo, base, scale=0.5)
        rebuilt = compose_actions(base, delta, ResidualComposeConfig(mode="residual", weight=0.5))
        np.testing.assert_allclose(rebuilt, demo, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
