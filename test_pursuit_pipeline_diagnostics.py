import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from analyze_pursuit_reward import AUDIT_COMPONENTS, run_reward_audit, write_reward_audit
from marl_trainers import TrainConfig, seed_everything
from pursuit_baselines_3d import BASELINES_3D
from quadrotor_pursuit_env import QuadrotorPursuitConfig, QuadrotorPursuitEnv
from train_pursuit_with_demos import (
    PursuitDemoTrainer, aggregate_corrections, evaluate, imitation_score,
    load_correction_replay, pretrain_mixed, resolve_auxiliary_path, snapshot,
    validate_correction_checkpoint_lineage,
)


class PursuitPipelineDiagnosticTests(unittest.TestCase):
    def setUp(self):
        seed_everything(12)
        torch.set_num_threads(1)
        self.cfg = QuadrotorPursuitConfig(seed=12, building_count=0, search_steps=2)

    def trainer(self):
        return PursuitDemoTrainer(
            lambda: QuadrotorPursuitEnv(self.cfg),
            TrainConfig(episodes=1, gamma=self.cfg.reward_gamma, hidden_dim=16,
                        batch_size=4, update_epochs=1, use_gat=False,
                        use_hetero_entities=False, device="cpu"),
        )

    def demo_episode(self):
        env = QuadrotorPursuitEnv(self.cfg)
        teacher = BASELINES_3D["mpc"]()
        teacher.reset(env)
        return [snapshot(env.observe_search(), teacher.actions(env), ~env.disabled_uavs)]

    def test_auxiliary_replay_is_never_implicit(self):
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / "pretrained.pt"
            checkpoint.touch()
            beside_checkpoint = Path(folder) / "corrections.pt"
            beside_checkpoint.touch()
            self.assertIsNone(resolve_auxiliary_path(None, False))
            self.assertEqual(resolve_auxiliary_path(beside_checkpoint, False), beside_checkpoint)
            self.assertIsNone(resolve_auxiliary_path(None, True))
            with self.assertRaisesRegex(ValueError, "cannot be used together"):
                resolve_auxiliary_path(beside_checkpoint, True)

    def test_explicit_corrections_load_and_lineage_is_checked(self):
        original = {
            "env_config": asdict(self.cfg), "teacher": "mpc",
            "episodes": [[1]], "report": [{"seed": 1}],
        }
        corrections = {
            "env_config": asdict(self.cfg), "teacher": "mpc",
            "episodes": [[1], [2]], "report": [{"seed": 1}],
            "correction_report": [{"seed": 2}], "dagger_round": 1,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "corrections_round_01.pt"
            torch.save(corrections, path)
            loaded, episodes = load_correction_replay(path, original, self.cfg)
        self.assertEqual(episodes, [[2]])
        validate_correction_checkpoint_lineage(
            {"initialization_type": "dagger", "dagger_round": 1,
             "correction_seeds": [2]}, loaded,
        )
        with self.assertRaisesRegex(ValueError, "expects DAgger round"):
            validate_correction_checkpoint_lineage(
                {"initialization_type": "dagger", "dagger_round": 2,
                 "correction_seeds": [2]}, loaded,
            )

    def test_explicit_dagger_sampling_fraction(self):
        trainer = self.trainer()
        episode = self.demo_episode()
        _, sampled = pretrain_mixed(
            trainer, [episode], [episode], updates=2, batch_size=4,
            correction_fraction=0.25, seed=12,
        )
        self.assertEqual(sampled["original_samples"], 6)
        self.assertEqual(sampled["correction_samples"], 2)
        self.assertAlmostEqual(sampled["actual_correction_fraction"], 0.25)

    def test_each_dagger_round_has_matching_files_and_validation_selection(self):
        trainer = self.trainer()
        trainer.demo_metadata = {
            "bc_updates": 100, "initialization_type": "bc", "correction_seeds": [],
        }
        dataset = {
            "env_config": asdict(self.cfg), "teacher": "mpc",
            "episodes": [self.demo_episode()], "report": [{"seed": 10}],
        }
        validation = [
            {"summary": {"mappo": {
                "capture_rate": 0.6, "mean_censored_steps": 1.5,
                "mean_success_steps": 1.0, "mean_return": -5.0,
                "mean_closest_capture_gap": 1.0, "mean_final_capture_gap": 2.0,
                "mean_safety_correction_rate": 0.1, "mean_emergency_stop_rate": 0.0,
            }}},
            {"summary": {"mappo": {
                "capture_rate": 0.4, "mean_censored_steps": 1.0,
                "mean_success_steps": 1.0, "mean_return": -4.0,
                "mean_closest_capture_gap": 0.8, "mean_final_capture_gap": 1.5,
                "mean_safety_correction_rate": 0.1, "mean_emergency_stop_rate": 0.0,
            }}},
        ]
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            with patch("train_pursuit_with_demos.pretrain", return_value=[]), patch(
                    "train_pursuit_with_demos.evaluate", side_effect=validation):
                aggregate_corrections(
                    trainer, self.cfg, dataset, 2, 1, 1, 2, 1000, out,
                    validation_seeds=[900], validation_records=[],
                    best_score=None, best_stage="bc",
                )
            for round_number in (1, 2):
                checkpoint = torch.load(
                    out / f"corrected_round_{round_number:02d}.pt",
                    map_location="cpu", weights_only=False,
                )
                corrections = torch.load(
                    out / f"corrections_round_{round_number:02d}.pt",
                    map_location="cpu", weights_only=False,
                )
                self.assertEqual(checkpoint["demo_metadata"]["dagger_round"], round_number)
                self.assertEqual(corrections["dagger_round"], round_number)
                self.assertEqual(
                    checkpoint["demo_metadata"]["correction_dataset"],
                    f"corrections_round_{round_number:02d}.pt",
                )
            best = torch.load(out / "best_imitation.pt", map_location="cpu", weights_only=False)
            self.assertEqual(best["demo_metadata"]["dagger_round"], 1)
            self.assertTrue((out / "dagger_validation.json").exists())

    def test_imitation_selection_uses_declared_validation_order(self):
        self.assertGreater(
            imitation_score({"capture_rate": 0.5, "mean_censored_steps": 2, "mean_return": -20}),
            imitation_score({"capture_rate": 0.4, "mean_censored_steps": 1, "mean_return": 20}),
        )
        self.assertGreater(
            imitation_score({"capture_rate": 0.5, "mean_censored_steps": 1, "mean_return": -20}),
            imitation_score({"capture_rate": 0.5, "mean_censored_steps": 2, "mean_return": 20}),
        )

    def test_evaluation_and_reward_audit_track_closest_gap_without_mutation(self):
        trainer = self.trainer()
        before = asdict(self.cfg)
        evaluated = evaluate(trainer, self.cfg, [123], ["mappo"])
        row = evaluated["rows"][0]
        self.assertLessEqual(row["closest_capture_gap"], row["final_capture_gap"])
        self.assertEqual(
            evaluated["summary"]["mappo"]["mean_closest_capture_gap"],
            row["closest_capture_gap"],
        )
        audited = run_reward_audit(self.cfg, [124], baseline="apf")
        self.assertEqual(asdict(self.cfg), before)
        self.assertEqual(set(audited["rows"][0]["reward_components"]), set(AUDIT_COMPONENTS))
        self.assertIn("successful", audited["summary"])
        self.assertIn("failed", audited["summary"])
        with tempfile.TemporaryDirectory() as folder:
            write_reward_audit(audited, Path(folder))
            self.assertTrue((Path(folder) / "reward_audit.json").exists())
            self.assertTrue((Path(folder) / "reward_audit.csv").exists())


if __name__ == "__main__":
    unittest.main()
