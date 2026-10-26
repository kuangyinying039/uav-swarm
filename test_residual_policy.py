import unittest

import numpy as np

from residual_policy import ResidualPolicy


class ResidualPolicyTests(unittest.TestCase):
    def test_pure_expert_when_gain_zero(self):
        expert = lambda obs, env_index=0: np.array([0.5, -0.5, 0.0, 0.1], dtype=np.float32)
        rl = lambda obs, env_index=0: np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        policy = ResidualPolicy(expert, rl, residual_gain=0.0)
        out = policy({})
        np.testing.assert_allclose(out, [0.5, -0.5, 0.0, 0.1])

    def test_blend_and_clip(self):
        expert = lambda obs, env_index=0: np.array([0.8, 0.0, 0.0, 0.0], dtype=np.float32)
        rl = lambda obs, env_index=0: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        policy = ResidualPolicy(expert, rl, residual_gain=0.5)
        out = policy({})
        self.assertAlmostEqual(float(out[0]), 1.0)  # 0.8+0.5 clipped


if __name__ == "__main__":
    unittest.main()
