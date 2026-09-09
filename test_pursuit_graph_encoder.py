import unittest

import numpy as np
import torch

from marl_trainers import GraphActor, GraphAttentionEncoder, TrainConfig
from pursuit_graph_encoder import (
    BUILDING_DIM, PEER_DIM, SELF_DIM, TARGET_DIM,
    PursuitGraphActor, PursuitGraphAttentionBlock, pursuit_graph_from_flat_observation,
    pursuit_graph_observation,
)
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from train_pursuit_with_demos import PursuitDemoTrainer


class PursuitGraphTests(unittest.TestCase):
    def environment(self, buildings=5):
        return QuadrotorPursuitEnv(
            QuadrotorPursuitConfig(seed=17, building_count=buildings,
                                   building_state_capacity=max(buildings, 5))
        )

    def test_target_node_uses_track_not_ground_truth(self):
        env = self.environment(0)
        track = env.track_memory[0][0]
        track.initialized = True
        track.mean[:6] = np.array([8.0, 9.0, 4.0, 0.4, -0.2, 0.1])
        before = pursuit_graph_observation(env)["target_nodes"][0].copy()
        env.dynamic_targets[0] += np.array([7.0, -5.0])
        env.target_altitudes[0] += 3.0
        after = pursuit_graph_observation(env)["target_nodes"][0]
        np.testing.assert_array_equal(before, after)

    def test_all_buildings_are_individual_entities(self):
        env = self.environment(5)
        graph = env.observe_search()["hetero_graph"]
        self.assertEqual(graph["building_nodes"].shape[-1], BUILDING_DIM)
        np.testing.assert_array_equal(
            graph["building_mask"].sum(axis=1), np.full(env.cfg.n_uavs, 5)
        )

    def test_building_height_changes_building_entity(self):
        env = self.environment(1)
        before = pursuit_graph_observation(env)["building_nodes"].copy()
        env.building_heights[0] += 2.0
        after = pursuit_graph_observation(env)["building_nodes"]
        self.assertFalse(np.array_equal(before[:, 0], after[:, 0]))
        self.assertTrue(np.all(after[:, 0, 5] > before[:, 0, 5]))

    def test_peer_distance_is_three_dimensional(self):
        xyz = torch.tensor([[5.0, 5.0, 2.0], [5.0, 5.0, 10.0]])
        distance = PursuitGraphAttentionBlock.pairwise_distances(xyz)
        self.assertAlmostEqual(float(distance[0, 1]), 8.0)

    def test_pursuit_actor_consumes_typed_graph(self):
        env = self.environment(2)
        obs = env.observe_search()
        graph = {
            key: torch.as_tensor(value, dtype=torch.bool if key.endswith("_mask") else torch.float32)
            for key, value in obs["hetero_graph"].items()
        }
        actor = PursuitGraphActor(
            env.obs_dim(), 4, hidden_dim=16, heads=1, layers=1,
            spatial_scale=env.cfg.grid_size,
        )
        output = actor(
            torch.as_tensor(obs["agent_observations"]),
            torch.as_tensor(obs["comm_adjacency"]), graph,
        )
        self.assertEqual(tuple(output.shape), (env.cfg.n_uavs, 4))
        self.assertEqual(graph["self_nodes"].shape[-1], SELF_DIM)
        self.assertEqual(graph["target_nodes"].shape[-1], TARGET_DIM)
        self.assertEqual(graph["peer_nodes"].shape[-1], PEER_DIM)

    def test_legacy_flat_demo_can_be_upgraded_without_truth(self):
        env = self.environment(2)
        observation = env.agent_observation_vectors()
        before_truth = env.dynamic_targets.copy()
        graph = pursuit_graph_from_flat_observation(observation, env.cfg)
        env.dynamic_targets += 4.0
        repeated = pursuit_graph_from_flat_observation(observation, env.cfg)
        np.testing.assert_array_equal(graph["target_nodes"], repeated["target_nodes"])
        self.assertFalse(np.array_equal(before_truth, env.dynamic_targets))
        self.assertEqual(int(graph["building_mask"].sum(axis=1)[0]), 2)

    def test_search_encoder_and_flat_pursuit_path_remain_separate(self):
        search_actor = GraphActor(43, 9, hidden_dim=16, heads=1, layers=1)
        self.assertIsInstance(search_actor.encoder, GraphAttentionEncoder)
        env = self.environment(0)
        trainer = PursuitDemoTrainer(
            lambda: QuadrotorPursuitEnv(env.cfg),
            TrainConfig(episodes=1, gamma=env.cfg.reward_gamma, hidden_dim=16,
                        use_gat=False, use_hetero_entities=False, device="cpu"),
        )
        self.assertEqual(trainer.action(env.observe_search()).shape, (env.cfg.n_uavs, 4))


if __name__ == "__main__":
    unittest.main()
