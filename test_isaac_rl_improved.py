"""Smoke tests for A6 / mappo_plus recipe wiring (no Isaac launch)."""
from __future__ import annotations

import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pursuit_residual import ResidualComposeConfig, compose_actions
from quadrotor_pursuit_env import QuadrotorPursuitConfig
from train_isaac_pursuit import (
    _apply_o2o_stable,
    _resolve_residual,
    _variant_config,
)


class ImprovedRecipeTests(unittest.TestCase):
    def test_a6_enables_o2o_and_high_utd(self):
        args = Namespace(
            episodes=1000,
            hidden_dim=128,
            batch_size=256,
            device="cpu",
            prior_fraction=None,
            budget_checkpoints="250,500",
            actor_init=Path("/tmp/a.pt"),
            prior=Path("/tmp/p.pt"),
        )
        cfg = QuadrotorPursuitConfig()
        matd3 = _variant_config("A6", args, cfg)
        self.assertGreaterEqual(matd3.utd, 2)
        self.assertGreater(matd3.td3_bc_alpha, 0.0)
        self.assertTrue(matd3.bc_on_batch)
        self.assertGreater(matd3.actor_anchor_weight, 0.0)
        self.assertEqual(matd3.critic_pretrain_updates, 5000)

    def test_a6_default_is_pure_actor(self):
        args = Namespace(residual_base=None, residual_weight=0.40, episodes=1000)
        self.assertIsNone(_resolve_residual(args, "A6"))
        cfg = QuadrotorPursuitConfig()
        matd3 = _variant_config(
            "A6",
            Namespace(
                episodes=1000,
                hidden_dim=128,
                batch_size=256,
                device="cpu",
                prior_fraction=None,
                budget_checkpoints="250,500",
                actor_init=Path("/tmp/a.pt"),
                prior=Path("/tmp/p.pt"),
            ),
            cfg,
        )
        self.assertTrue(matd3.success_replay_only)

    def test_blend_is_convex_combination(self):
        base = np.ones((3, 4), dtype=np.float32)
        rl = -np.ones((3, 4), dtype=np.float32)
        out = compose_actions(base, rl, ResidualComposeConfig(mode="blend", weight=0.25))
        np.testing.assert_allclose(out, 0.5 * np.ones_like(base))

    def test_o2o_helper_sets_stable_bc_floor(self):
        values = {"demo_bc_final_weight": 0.0, "utd": 1}
        _apply_o2o_stable(values, decay=50_000)
        self.assertGreaterEqual(values["demo_bc_final_weight"], 0.5)
        self.assertGreaterEqual(values["utd"], 2)


if __name__ == "__main__":
    unittest.main()
