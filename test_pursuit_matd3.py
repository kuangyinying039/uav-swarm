import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from marl_trainers import TrainConfig, seed_everything
from pursuit.algorithms.matd3 import Matd3Config, update_matd3
from pursuit.data.replay_buffer import JointReplayBuffer, mix_batches
from pursuit.data.transition_dataset import (
    TRANSITION_DATASET_V2,
    collect_teacher_transitions,
    make_transition,
    teacher_step,
    validate_transition_dataset,
)
from pursuit.eval import evaluate_policy
from pursuit.trainers.matd3_trainer import Matd3Trainer, mean_actor_state_from_checkpoint
from pursuit_baselines_3d import CooperativeGuidanceMPC3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from train_pursuit_with_demos import PursuitDemoTrainer


class PursuitMatd3Tests(unittest.TestCase):
    def setUp(self):
        seed_everything(4)
        torch.set_num_threads(1)
        self.env_cfg = QuadrotorPursuitConfig(seed=4, building_count=0, search_steps=3)

    def trainer(self, **overrides):
        cfg = Matd3Config(
            gamma=self.env_cfg.reward_gamma, hidden_dim=16, critic_hidden=16, critic_layers=2,
            gat_heads=1, gat_layers=1, batch_size=4, replay_size=32, warmup_steps=2,
            episodes=1, device="cpu", utd=1, policy_delay=2, **overrides,
        )
        return Matd3Trainer(lambda episode=0: QuadrotorPursuitEnv(self.env_cfg), cfg, seed=4)

    def test_mpc_plan_matches_actions(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        planned = teacher.plan(env)
        teacher.reset(env)
        np.testing.assert_allclose(planned["actions"], teacher.actions(env))
        self.assertEqual(planned["teacher_role"].shape, (env.cfg.n_uavs,))
        self.assertTrue(set(planned["teacher_role"]).issubset({0, 1}))

    def test_transition_keeps_proposed_action_and_failures(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        obs = env.observe_search()
        action, labels = teacher_step(teacher, env)
        result = env.step_joint(action)
        row = make_transition(obs, action, result, labels, np.ones(env.cfg.n_uavs, dtype=bool), ~env.disabled_uavs)
        np.testing.assert_allclose(row["actions"], action)
        self.assertIn("reward", row)
        self.assertIn("next_state", row)
        self.assertEqual(row["executed_velocity"].shape, (env.cfg.n_uavs, 3))
        dataset = collect_teacher_transitions(self.env_cfg, 4, 1, "mpc")
        self.assertEqual(dataset["format"], TRANSITION_DATASET_V2)
        self.assertEqual(len(dataset["episodes"]), 1)
        validate_transition_dataset(dataset, self.env_cfg)
        self.assertTrue(dataset["report"][0]["kept"])

    def test_replay_and_prior_mix(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        obs = env.observe_search()
        action, labels = teacher_step(teacher, env)
        result = env.step_joint(action)
        row = make_transition(obs, action, result, labels, ~np.zeros(env.cfg.n_uavs, dtype=bool), ~env.disabled_uavs)
        prior = JointReplayBuffer(8, seed=1)
        online = JointReplayBuffer(8, seed=2)
        for _ in range(4):
            prior.add(row)
            online.add(row)
        batch = mix_batches(prior, online, 6, 0.5, "cpu")
        self.assertEqual(tuple(batch["actions"].shape), (6, env.cfg.n_uavs, 4))
        self.assertEqual(tuple(batch["state"].shape[0:1]), (6,))

    def test_matd3_update_and_bc_actor_load(self):
        mappo = PursuitDemoTrainer(
            lambda: QuadrotorPursuitEnv(self.env_cfg),
            TrainConfig(episodes=1, gamma=self.env_cfg.reward_gamma, hidden_dim=16, batch_size=3,
                        update_epochs=1, use_gat=True, use_hetero_entities=True, gat_heads=1,
                        gat_layers=1, device="cpu"),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bc.pt"
            mappo.save_checkpoint(path, episode=0)
            payload = torch.load(path, map_location="cpu", weights_only=False)
        trainer = self.trainer()
        trainer.load_actor_weights(mean_actor_state_from_checkpoint(payload))
        env = QuadrotorPursuitEnv(self.env_cfg)
        obs = env.observe_search()
        action = trainer.deterministic_action(obs)
        self.assertEqual(action.shape, (env.cfg.n_uavs, 4))
        self.assertTrue(np.all(np.abs(action) <= 1.0 + 1e-6))
        result = env.step_joint(action)
        row = make_transition(
            obs, action, result, {}, np.ones(env.cfg.n_uavs, dtype=bool), ~env.disabled_uavs
        )
        for _ in range(8):
            trainer.online_replay.add(row)
        batch = trainer.online_replay.sample(4, trainer.device)
        stats = update_matd3(trainer, batch, update_actor=True)
        self.assertTrue(np.isfinite(stats["critic_loss"]))
        self.assertTrue(stats["updated_actor"])

    def test_short_train_and_eval(self):
        trainer = self.trainer(episodes=1, warmup_steps=1, batch_size=2, replay_size=16)
        history = trainer.train(np.random.default_rng(4))
        self.assertEqual(len(history), 1)
        self.assertIn("capture_success", history[0])
        result = evaluate_policy(
            lambda obs, env: trainer.deterministic_action(obs),
            self.env_cfg, [4], method_name="matd3",
        )
        self.assertEqual(result["summary"]["matd3"]["episodes"], 1)
        self.assertIn("near_miss_0_5m_failures", result["summary"]["matd3"])
        self.assertIn("max_uavs_in_capture", result["rows"][0])


if __name__ == "__main__":
    unittest.main()
