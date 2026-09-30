import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from train_pursuit_matd3 import main as matd3_cli_main


class PursuitMatd3Tests(unittest.TestCase):
    def setUp(self):
        seed_everything(4)
        torch.set_num_threads(1)
        self.env_cfg = QuadrotorPursuitConfig(seed=4, building_count=0, search_steps=3)

    def trainer(self, **overrides):
        values = dict(
            gamma=self.env_cfg.reward_gamma, hidden_dim=16, critic_hidden=16, critic_layers=2,
            gat_heads=1, gat_layers=1, batch_size=4, replay_size=32, warmup_steps=2,
            episodes=1, device="cpu", utd=1, policy_delay=2,
        )
        values.update(overrides)
        cfg = Matd3Config(**values)
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

    def test_mpc_dual_capture_assigns_multiple_interceptors(self):
        cfg = QuadrotorPursuitConfig(
            seed=4, building_count=0, capture_required_uavs=2, search_steps=3
        )
        env = QuadrotorPursuitEnv.__new__(QuadrotorPursuitEnv)
        env.cfg = cfg
        env.current_decision_dt = cfg.decision_dt
        env.disabled_uavs = np.zeros(cfg.n_uavs, dtype=bool)
        env.quadrotor_states = np.zeros((cfg.n_uavs, 13), dtype=float)
        env.quadrotor_states[:, 6] = 1.0
        env.quadrotor_states[:, :3] = [[4.0, 5.0, 5.0], [6.0, 5.0, 5.0], [12.0, 12.0, 5.0]]
        env.dynamic_targets = np.array([[5.0, 5.0]])
        env.target_altitudes = np.array([5.0])
        env.target_velocity = np.zeros((1, 2), dtype=float)
        env.target_vertical_velocity = np.zeros(1, dtype=float)
        # Minimal track memory: initialized local estimates at the true target.
        class _Track:
            def __init__(self):
                self.initialized = True
                self.mean = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0], dtype=float)

        env.track_memory = [[_Track()] for _ in range(cfg.n_uavs)]
        env.last_graph = np.ones((cfg.n_uavs, cfg.n_uavs), dtype=float)
        env.buildings = []
        env.building_heights = []
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        planned = teacher.plan(env)
        self.assertEqual(planned["capture_required_uavs"], 2)
        self.assertGreaterEqual(int(np.count_nonzero(planned["teacher_role"] == 0)), 2)
        # Intercept goals stay inside the capture sphere around the prediction.
        for agent in np.flatnonzero(planned["teacher_role"] == 0):
            offset = np.linalg.norm(
                planned["teacher_goal"][agent] - planned["predicted_target"][agent]
            )
            self.assertLessEqual(offset, cfg.capture_radius + 1e-6)

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
        self.assertEqual(tuple(batch["next_active"].shape), (6, env.cfg.n_uavs))
        self.assertEqual(int(batch["is_prior"].sum()), 3)

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

    def test_demo_bc_regularizer_is_explicit_and_reports_drift(self):
        trainer = self.trainer(demo_bc_weight=1.0, demo_bc_final_weight=0.25,
                               demo_bc_decay_steps=10, prior_fraction=0.5)
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        obs = env.observe_search()
        action, labels = teacher_step(teacher, env)
        result = env.step_joint(action)
        row = make_transition(obs, action, result, labels, np.ones(env.cfg.n_uavs, dtype=bool),
                              ~env.disabled_uavs)
        for _ in range(8):
            trainer.prior_replay.add(row)
            trainer.online_replay.add(row)
        trainer.env_steps = 5
        stats = update_matd3(trainer, trainer._sample_batch(), update_actor=True)
        self.assertIsNotNone(stats["demo_bc_loss"])
        self.assertAlmostEqual(stats["demo_bc_weight"], 0.625)
        self.assertAlmostEqual(stats["prior_batch_fraction"], 0.5)

    def test_critic_pretraining_preserves_actor_and_anneals_exploration(self):
        trainer = self.trainer(
            critic_pretrain_updates=3,
            exploration_std=0.12,
            exploration_final_std=0.02,
            exploration_decay_steps=10,
            warmup_steps=2,
        )
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = CooperativeGuidanceMPC3D()
        teacher.reset(env)
        obs = env.observe_search()
        action, labels = teacher_step(teacher, env)
        result = env.step_joint(action)
        row = make_transition(
            obs, action, result, labels, np.ones(env.cfg.n_uavs, dtype=bool),
            ~env.disabled_uavs,
        )
        for _ in range(4):
            trainer.prior_replay.add(row)
        actor_before = {
            key: value.detach().clone() for key, value in trainer.actor.state_dict().items()
        }
        critic_before = {
            key: value.detach().clone() for key, value in trainer.critic.state_dict().items()
        }
        summary = trainer.pretrain_critic()
        self.assertEqual(summary["completed_updates"], 3)
        self.assertEqual(trainer.total_updates, 3)
        self.assertEqual(trainer.pretrain_critic(), {})
        self.assertTrue(all(
            torch.equal(value, trainer.actor.state_dict()[key])
            for key, value in actor_before.items()
        ))
        self.assertTrue(any(
            not torch.equal(value, trainer.critic.state_dict()[key])
            for key, value in critic_before.items()
        ))
        trainer.env_steps = 2
        self.assertAlmostEqual(trainer.current_exploration_std(), 0.12)
        trainer.env_steps = 7
        self.assertAlmostEqual(trainer.current_exploration_std(), 0.07)
        trainer.env_steps = 12
        self.assertAlmostEqual(trainer.current_exploration_std(), 0.02)

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
        self.assertIn("mean_safety_correction_magnitude", result["summary"]["matd3"])
        self.assertIn("safety_intervention_free_episode_rate", result["summary"]["matd3"])
        self.assertIn("max_uavs_in_capture", result["rows"][0])

    def test_flat_matd3_actor_ablation(self):
        trainer = self.trainer(use_gat=False)
        self.assertEqual(trainer.algorithm_name, "matd3")
        env = QuadrotorPursuitEnv(self.env_cfg)
        action = trainer.deterministic_action(env.observe_search())
        self.assertEqual(action.shape, (env.cfg.n_uavs, 4))

    def test_pre_rl_validation_does_not_claim_best_capture(self):
        trainer = self.trainer()
        with tempfile.TemporaryDirectory() as folder:
            trainer.output_directory = Path(folder)
            trainer.validation_seeds = [4]

            def fake_evaluate(action_fn, env_cfg, seeds, method_name="matd3"):
                del action_fn, env_cfg, seeds
                rate = 0.94 if trainer.env_steps == 0 else 0.50
                return {
                    "summary": {
                        method_name: {
                            "capture_rate": rate,
                            "mean_censored_steps": 60.0,
                            "mean_return": 1.0,
                        }
                    }
                }

            trainer.validate(0, fake_evaluate)
            self.assertFalse((trainer.output_directory / "best_capture.pt").exists())
            self.assertTrue(trainer.validation_records[0]["pre_rl"])
            self.assertFalse(trainer.validation_records[0]["considered_for_best"])
            self.assertEqual(trainer.best_capture_score[0], -1.0)

            trainer.env_steps = 1000
            trainer.validate(100, fake_evaluate)
            self.assertTrue((trainer.output_directory / "best_capture.pt").exists())
            self.assertTrue(trainer.validation_records[1]["considered_for_best"])
            self.assertAlmostEqual(trainer.best_capture_score[0], 0.50)

    def test_evaluate_demo_bc_checkpoint_does_not_require_prior_dataset(self):
        trainer = self.trainer(
            demo_bc_weight=1.0,
            demo_bc_final_weight=0.1,
            demo_bc_decay_steps=10,
            prior_fraction=0.5,
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "best_capture.pt"
            output = root / "evaluation"
            trainer.save_checkpoint(checkpoint, episode=0)
            argv = [
                "train_pursuit_matd3.py", "evaluate",
                "--checkpoint", str(checkpoint),
                "--eval-episodes", "1",
                "--eval-seed", "7000000",
                "--device", "cpu",
                "--out", str(output),
            ]
            with patch("sys.argv", argv):
                matd3_cli_main()
            self.assertTrue((output / "evaluation.json").is_file())


if __name__ == "__main__":
    unittest.main()
