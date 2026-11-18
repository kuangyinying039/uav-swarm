"""Unit tests for Evader Prediction Network helpers."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from pursuit_epn import (
    EPNConfig,
    EPNRuntime,
    attach_epn_runtime,
    build_epn_frame,
    epn_input_dim,
    future_relative_labels,
    make_epn,
    update_epn_from_rollout,
)
from pursuit_graph_encoder import TARGET_DIM
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv


class PursuitEPNTests(unittest.TestCase):
    def test_input_dim_and_missing_fill(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=3, building_count=0))
        # Force an uninitialized track so the missing-fill path is exercised.
        env.track_memory[0][0].initialized = False
        frame = build_epn_frame(env, 0)
        self.assertEqual(frame.shape, (epn_input_dim(env.cfg.n_uavs),))
        self.assertTrue(np.all(frame[1:7] == -1.0))

    def test_predict_shape_and_graph_injection(self):
        env = QuadrotorPursuitEnv(QuadrotorPursuitConfig(seed=5, building_count=0))
        cfg = EPNConfig(n_uavs=env.cfg.n_uavs)
        epn = make_epn(cfg, device="cpu")
        runtime = EPNRuntime(epn, cfg, device="cpu")
        attach_epn_runtime(env, runtime)
        obs = env.observe_search()
        self.assertEqual(obs["hetero_graph"]["target_nodes"].shape[-1], TARGET_DIM)
        self.assertEqual(runtime.last_prediction.shape, (env.cfg.n_uavs, cfg.future_steps, 3))
        # Cold history still produces finite tanh outputs.
        self.assertTrue(np.isfinite(runtime.last_prediction).all())

    def test_mse_decreases_on_constant_motion(self):
        cfg = EPNConfig(n_uavs=2, history_steps=4, future_steps=3, epochs=5, lr=1e-2)
        epn = make_epn(cfg, device="cpu")
        optim = torch.optim.Adam(epn.parameters(), lr=cfg.lr)
        t_steps = 12
        agents = np.zeros((t_steps, 2, 3), dtype=np.float64)
        agents[:, 0, 0] = np.linspace(0.0, 2.0, t_steps)
        agents[:, 1, 0] = np.linspace(0.5, 2.5, t_steps)
        evaders = np.zeros((t_steps, 3), dtype=np.float64)
        evaders[:, 0] = np.linspace(1.0, 4.0, t_steps)
        env_cfg = QuadrotorPursuitConfig(seed=0, building_count=0)
        hist = np.zeros((t_steps, 2, cfg.history_steps, epn_input_dim(2)), dtype=np.float32)
        for t in range(t_steps):
            hist[t, :, :, 0] = t / max(t_steps - 1, 1)
            hist[t, :, :, 1] = (evaders[t, 0] - agents[t, :, 0])[:, None] / env_cfg.grid_size
        first = update_epn_from_rollout(
            epn, optim, hist, agents, evaders, env_cfg, cfg, device="cpu"
        )
        second = update_epn_from_rollout(
            epn, optim, hist, agents, evaders, env_cfg, cfg, device="cpu"
        )
        self.assertTrue(np.isfinite(first) and np.isfinite(second))
        self.assertLessEqual(second, first + 1e-5)

    def test_future_labels_shape(self):
        agents = np.zeros((6, 3, 3))
        evaders = np.linspace(0, 1, 6).reshape(6, 1) * np.array([[1.0, 0.0, 0.0]])
        labels = future_relative_labels(
            agents, evaders, QuadrotorPursuitConfig(seed=0), future_steps=5
        )
        self.assertEqual(labels.shape, (6, 3, 5, 3))
        self.assertTrue(np.all(labels <= 1.0) and np.all(labels >= -1.0))


if __name__ == "__main__":
    unittest.main()
