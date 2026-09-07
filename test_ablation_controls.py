import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - the training runtime requires torch.
    torch = None

try:
    from cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
    from marl_trainers import FlatMLP, IPPOTrainer, TrainConfig, episode_task_metrics
except ImportError:
    from .cooperative_search_env import WeakCommBeliefGraphEnv, WeakCommConfig
    from .marl_trainers import FlatMLP, IPPOTrainer, TrainConfig, episode_task_metrics


class RevisitMapAblationTests(unittest.TestCase):
    def make_env(self, enabled: bool) -> WeakCommBeliefGraphEnv:
        cfg = WeakCommConfig(
            n_uavs=2,
            n_targets=3,
            grid_size=12,
            n_obstacles=0,
            n_jammers=0,
            comm_mode="full",
            dynamic_targets_enabled=False,
            dynamic_obstacles_enabled=False,
            failure_base_prob=0.0,
            revisit_map_enabled=enabled,
            target_completion_steps=0,
            search_steps=10,
        )
        env = WeakCommBeliefGraphEnv(cfg)
        env.reset()
        return env

    def test_disabled_revisit_features_are_neutral(self):
        env = self.make_env(False)
        features = env.graph_features()

        # Keep the ordinary probability/uncertainty frontier active.
        self.assertTrue(np.all(np.isfinite(features[:, 23:27])))
        self.assertTrue(np.all(features[:, 26] > 0.0))

        # Neutralize every channel derived exclusively from revisit memory.
        np.testing.assert_allclose(features[:, 25], 0.0)
        np.testing.assert_allclose(features[:, 29], 0.0)
        np.testing.assert_allclose(features[:, 30:32], 0.0)

    def test_discovery_only_logging_uses_discovery_completion_and_success(self):
        env = self.make_env(False)
        env.found_targets[:] = np.array([True, True, False])
        metrics = episode_task_metrics(env)
        self.assertAlmostEqual(metrics["target_discovery_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["target_tracking_rate"], 0.0)
        self.assertAlmostEqual(metrics["target_completion_rate"], 2.0 / 3.0)
        self.assertEqual(metrics["task_success"], 0.0)

        env.found_targets[:] = True
        metrics = episode_task_metrics(env)
        self.assertEqual(metrics["target_completion_rate"], 1.0)
        self.assertEqual(metrics["task_success"], 1.0)


class CellWiseBeliefFusionTests(unittest.TestCase):
    def make_env(self, mode: str = "soft") -> WeakCommBeliefGraphEnv:
        cfg = WeakCommConfig(
            n_uavs=2,
            n_targets=1,
            grid_size=8,
            n_obstacles=0,
            n_jammers=0,
            comm_mode="outage",
            dynamic_targets_enabled=False,
            dynamic_obstacles_enabled=False,
            failure_base_prob=0.0,
            belief_fusion_mode=mode,
            belief_fusion_temperature=0.5,
            belief_freshness_tau=2.0,
        )
        env = WeakCommBeliefGraphEnv(cfg)
        env.reset()
        return env

    def test_scan_resets_only_observed_cell_age(self):
        env = self.make_env()
        env.local_belief_age_map[:] = 7.0
        env.positions[0] = np.array([3.0, 3.0])
        env._scan_and_update(0)
        self.assertEqual(env.local_belief_age_map[0, 3, 3], 0.0)
        self.assertEqual(env.local_belief_age_map[0, 0, 0], 7.0)
        self.assertEqual(env.local_belief_age_map[1, 3, 3], 7.0)

    def test_soft_fusion_prefers_fresh_evidence_over_stale_extreme(self):
        env = self.make_env("soft")
        beliefs = np.array([[[1.0]], [[6.0]]])
        ages = np.array([[[0.0]], [[20.0]]])
        fused, fused_age = env._fuse_belief_sources(
            beliefs, ages, np.ones(2)
        )
        self.assertLess(float(fused[0, 0]), 2.0)
        self.assertLess(float(fused_age[0, 0]), 1.0)

    def test_hard_mode_reproduces_winner_take_all(self):
        env = self.make_env("hard")
        beliefs = np.array([[[1.0]], [[6.0]]])
        ages = np.array([[[0.0]], [[20.0]]])
        fused, fused_age = env._fuse_belief_sources(
            beliefs, ages, np.ones(2)
        )
        self.assertEqual(float(fused[0, 0]), 6.0)
        self.assertEqual(float(fused_age[0, 0]), 20.0)


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class NoGATAblationTests(unittest.TestCase):
    def test_plain_mlp_ignores_all_graph_inputs(self):
        torch.manual_seed(7)
        ablated = FlatMLP(in_dim=41, out_dim=9, hidden_dim=32)
        obs = torch.randn(3, 41)
        identity = torch.eye(3, dtype=torch.bool)
        complete = torch.ones(3, 3, dtype=torch.bool)
        with torch.no_grad():
            plain_output = ablated(obs)
            graph_output = ablated(
                obs,
                complete,
                {"target_nodes": torch.randn(3, 4, 13)},
            )
        torch.testing.assert_close(plain_output, graph_output)

    def test_ippo_forces_local_actor_and_local_critic(self):
        env_cfg = WeakCommConfig(
            n_uavs=2,
            n_targets=2,
            grid_size=10,
            n_obstacles=0,
            n_jammers=0,
            search_steps=2,
        )
        trainer = IPPOTrainer(
            lambda: WeakCommBeliefGraphEnv(env_cfg),
            TrainConfig(episodes=1, use_gat=True, use_hetero_entities=True),
        )
        self.assertFalse(trainer.cfg.use_gat)
        self.assertFalse(trainer.cfg.use_hetero_entities)
        self.assertEqual(trainer.algorithm_name, "ippo")
        self.assertEqual(trainer.critic.net[0].in_features, trainer.obs_dim)


if __name__ == "__main__":
    unittest.main()
