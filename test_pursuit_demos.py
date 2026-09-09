import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from marl_trainers import TrainConfig, seed_everything
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from pursuit_baselines_3d import BASELINES_3D
from train_pursuit_with_demos import (
    PursuitDemoTrainer, collect_demos, pretrain, snapshot, validate_demos, evaluate,
)
from pursuit_training_output import write_pursuit_outputs


class PursuitDemoTests(unittest.TestCase):
    def setUp(self):
        seed_everything(4)
        torch.set_num_threads(1)
        self.env_cfg = QuadrotorPursuitConfig(seed=4, building_count=0, search_steps=3)

    def trainer(self, gat=False):
        return PursuitDemoTrainer(lambda: QuadrotorPursuitEnv(self.env_cfg),
                                  TrainConfig(episodes=1, gamma=self.env_cfg.reward_gamma, hidden_dim=16, batch_size=3, update_epochs=1,
                                              use_gat=gat, use_hetero_entities=gat, gat_heads=1,
                                              gat_layers=1, device="cpu"))

    def test_exploration_has_gradient_and_changes(self):
        trainer = self.trainer()
        actor = trainer.actor
        obs = torch.zeros((3, trainer.obs_dim))
        _, std = actor(obs).chunk(2, -1)
        self.assertTrue(torch.allclose(std.exp(), torch.full_like(std, 0.4)))
        before = actor.std_parameter.detach().clone()
        std.sum().backward()
        self.assertTrue(torch.all(actor.std_parameter.grad.abs() > 0))
        trainer.actor_optim.step()
        self.assertFalse(torch.equal(before, actor.std_parameter))

    def test_warm_start_exploration_can_be_set_conservatively(self):
        trainer = self.trainer()
        trainer.actor.set_std(0.08)
        obs = torch.zeros((3, trainer.obs_dim))
        _, log_std = trainer.actor(obs).chunk(2, -1)
        torch.testing.assert_close(log_std.exp(), torch.full_like(log_std, 0.08))

    def test_demo_weight_decays_to_nonzero_floor(self):
        trainer = self.trainer()
        trainer.cfg.episodes = 10
        trainer.demo_episodes = [[{"unused": True}]]
        trainer.demo_weight = 0.1
        trainer.demo_weight_floor = 0.02
        self.assertAlmostEqual(trainer.demo_loss_weight(0), 0.1)
        self.assertAlmostEqual(trainer.demo_loss_weight(9), 0.02)

    def test_auxiliary_batch_separates_original_and_correction_states(self):
        trainer = self.trainer()
        trainer.cfg.batch_size = 8
        trainer.demo_episodes = [[{"source": "original"}]]
        trainer.correction_episodes = [[{"source": "correction"}]]
        trainer.correction_fraction = 0.75
        with patch("train_pursuit_with_demos.cloning_loss", side_effect=[torch.tensor(2.0), torch.tensor(4.0)]) as loss:
            value = trainer.pursuit_demo_loss(0)
        self.assertEqual(loss.call_args_list[0].args[1], trainer.demo_episodes)
        self.assertEqual(loss.call_args_list[0].args[2], 2)
        self.assertEqual(loss.call_args_list[1].args[1], trainer.correction_episodes)
        self.assertEqual(loss.call_args_list[1].args[2], 6)
        self.assertAlmostEqual(float(value), 0.1 * 3.5)

    def test_frozen_reference_policy_is_checkpointed(self):
        trainer = self.trainer()
        trainer.capture_reference_policy()
        reference = {key: value.clone() for key, value in trainer.reference_actor.state_dict().items()}
        self.assertTrue(all(not parameter.requires_grad for parameter in trainer.reference_actor.parameters()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.pt"
            trainer.save_checkpoint(path, episode=0)
            restored = self.trainer()
            restored.load_checkpoint(path)
            for key, value in reference.items():
                torch.testing.assert_close(restored.reference_actor.state_dict()[key], value)

    def test_validation_rollback_restores_actor_and_reduces_learning_rate(self):
        trainer = self.trainer()
        best = {key: value.detach().cpu().clone() for key, value in trainer.actor.state_dict().items()}
        trainer.best_actor_state = best
        trainer.best_capture_score = (0.5, -3.0, -10.0)
        trainer.safeguard_patience = 1
        trainer.actor_base_lr = 5e-5
        trainer.min_actor_lr = 1e-5
        with torch.no_grad():
            next(trainer.actor.parameters()).add_(1.0)
        validation = {
            "summary": {"mappo": {"capture_rate": 0.0, "mean_censored_steps": 3.0,
                                      "mean_return": -60.0}}
        }
        with tempfile.TemporaryDirectory() as folder:
            trainer.output_directory = Path(folder)
            trainer.validation_seeds = [91]
            with patch("train_pursuit_with_demos.evaluate", return_value=validation):
                trainer.validate_capture(50)
        for key, value in best.items():
            torch.testing.assert_close(trainer.actor.state_dict()[key].cpu(), value)
        self.assertEqual(trainer.actor_base_lr, 2.5e-5)
        self.assertEqual(trainer.validation_records[-1]["actor_lr_before_rollback"], 5e-5)
        self.assertEqual(trainer.validation_records[-1]["actor_lr_after_rollback"], 2.5e-5)

    def test_policy_likelihood_stable_before_ppo_update(self):
        from train_pursuit_with_demos import observation_tensors
        trainer = self.trainer(True)
        trainer.actor.train()
        inputs = observation_tensors(QuadrotorPursuitEnv(self.env_cfg).observe_search(),
                                     trainer.device, trainer.cfg)
        first = trainer.actor(*inputs)
        for _ in range(5):
            torch.testing.assert_close(trainer.actor(*inputs), first, rtol=0, atol=0)
        self.assertEqual(trainer.cfg.dropout, 0.0)

    def test_deadline_is_terminal_failure(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        for _ in range(self.env_cfg.search_steps):
            result = env.step_joint(np.zeros((self.env_cfg.n_uavs, 4)))
        self.assertFalse(result["capture_success"])
        self.assertTrue(result["terminated"])
        self.assertFalse(result["truncated"])
        self.assertEqual(result["reward_components"]["timeout"], -self.env_cfg.timeout_penalty)

    def test_flat_features_use_local_target_motion_and_deadline(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        before = env.graph_features().copy()
        track = env.track_memory[0][0]
        track.initialized = True
        track.mean[2] += 1.0
        track.mean[3:6] += np.array([0.3, -0.2, 0.1])
        after = env.graph_features()
        self.assertFalse(np.array_equal(before[0, [7, 8, 9, 15]], after[0, [7, 8, 9, 15]]))
        self.assertEqual(after[0, 21], 1.0)
        env.t = self.env_cfg.search_steps
        self.assertEqual(env.graph_features()[0, 21], 0.0)

    def test_cloning_reduces_error_with_graph_and_flat_actor(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        obs = env.observe_search()
        teacher = BASELINES_3D["mpc"]()
        teacher.reset(env)
        action = teacher.actions(env)
        rows = [[snapshot(obs, action, ~env.disabled_uavs)]]
        for gat in (False, True):
            trainer = self.trainer(gat)
            before = np.mean((trainer.action(obs) - action) ** 2)
            pretrain(trainer, rows, 400, 2, 4)
            after = np.mean((trainer.action(obs) - action) ** 2)
            self.assertLess(after, before * 0.6)

    def test_online_update_checkpoint_and_evaluation(self):
        trainer = self.trainer(True)
        env = QuadrotorPursuitEnv(self.env_cfg)
        teacher = BASELINES_3D["mpc"]()
        teacher.reset(env)
        trainer.demo_episodes = [[snapshot(env.observe_search(), teacher.actions(env), ~env.disabled_uavs)]]
        trainer.reference_kl_weight = 0.05
        trainer.capture_reference_policy()
        reference_before = {key: value.clone() for key, value in trainer.reference_actor.state_dict().items()}
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertAlmostEqual(history[0]["reward"], sum(history[0]["reward_components"].values()))
        self.assertLessEqual(history[0]["closest_capture_gap"], history[0]["minimum_capture_gap"])
        self.assertTrue(np.isfinite(history[0]["policy_loss"]))
        self.assertLessEqual(history[0]['exact_policy_kl'], 1.5*trainer.cfg.target_kl)
        self.assertLess(history[0]['pre_update_logprob_error'], 2e-3)
        self.assertGreater(history[0]['ppo_accepted_steps'], 0)
        self.assertGreaterEqual(history[0]['fixed_reference_kl'], 0.0)
        for key, value in reference_before.items():
            torch.testing.assert_close(trainer.reference_actor.state_dict()[key], value)
        obs = QuadrotorPursuitEnv(self.env_cfg).observe_search()
        before = trainer.action(obs)
        with tempfile.TemporaryDirectory() as folder:
            write_pursuit_outputs(folder, history, trainer.episode_traces)
            self.assertTrue((Path(folder) / "index.html").exists())
            columns = (Path(folder) / "training.csv").read_text(encoding="utf-8-sig").splitlines()[0]
            for forbidden in ("coverage", "discovery", "spectrum", "assignment_search", "weight_target"):
                self.assertNotIn(forbidden, columns)
                self.assertNotIn(forbidden, history[0])
            self.assertIsNone(history[0]["capture_time"])
            path = Path(folder) / "model.pt"
            trainer.save_checkpoint(path, episode=1, history=history)
            restored = self.trainer(True)
            payload = restored.load_checkpoint(path)
            self.assertEqual(payload["env_config"], asdict(self.env_cfg))
            np.testing.assert_array_equal(before, restored.action(obs))
            result = evaluate(restored, self.env_cfg, [9], ["mappo", "mpc"])
            for row in result["rows"]:
                self.assertAlmostEqual(row["return"], sum(row["reward_components"].values()))

    def test_environment_mismatch_rejected(self):
        dataset = {"env_config": asdict(self.env_cfg), "episodes": [[1]]}
        validate_demos(dataset, self.env_cfg)
        dataset["env_config"]["target_diameter"] = 100
        with self.assertRaises(ValueError):
            validate_demos(dataset, self.env_cfg)

    def test_legacy_execution_contract_rejected(self):
        from train_pursuit_with_demos import load_environment_config
        recorded = asdict(self.env_cfg)
        recorded.pop('execution_reward_version')
        with self.assertRaisesRegex(ValueError, 'Safety execution'):
            load_environment_config(recorded)

    def test_initial_action_mean_is_not_directionally_saturated(self):
        env = QuadrotorPursuitEnv(self.env_cfg)
        for gat in (False, True):
            trainer = self.trainer(gat)
            self.assertLess(np.max(np.abs(trainer.action(env.observe_search()))), .1)

    def test_old_control_dataset_rejected_even_with_same_action_dimension(self):
        recorded = asdict(self.env_cfg)
        recorded.update(task_mode='pursuit_quadrotor_nmpc_3d', nmpc_horizon=5)
        with self.assertRaisesRegex(ValueError, 'Recollect'):
            validate_demos({'env_config': recorded, 'episodes': [[1]]}, self.env_cfg)

    def test_failed_trajectories_not_accepted_as_demonstrations(self):
        # Real three-step rollout cannot normally reach the target; force the
        # success flag false to make this assertion independent of dynamics.
        original = QuadrotorPursuitEnv.step_joint

        def failed_step(env, actions):
            result = original(env, actions)
            result["capture_success"] = False
            return result

        with patch.object(QuadrotorPursuitEnv, "step_joint", failed_step):
            with self.assertRaisesRegex(RuntimeError, "Only 0/1"):
                collect_demos(self.env_cfg, 4, 1, 1, "mpc")


if __name__ == "__main__":
    unittest.main()
