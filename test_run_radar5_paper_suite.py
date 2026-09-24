"""Unit tests for the Radar5 paper suite command builder."""
from __future__ import annotations

import unittest
from argparse import Namespace

from run_radar5_paper_suite import ABLATION_VARIANTS, build_manifest


def _args(**overrides):
    values = dict(
        stage="all",
        train_level="medium",
        prefix=None,
        seeds="11,12",
        episodes=3000,
        include_ablation_eval=False,
        dry_run=True,
        manifest_out=None,
    )
    values.update(overrides)
    return Namespace(**values)


class Radar5PaperSuiteTests(unittest.TestCase):
    def test_default_train_level_is_medium_and_recollects_assets(self):
        manifest = build_manifest(_args(stage="assets"))
        self.assertEqual(manifest["train_level"], "medium")
        ids = [job["id"] for job in manifest["jobs"]]
        self.assertEqual(
            ids,
            [
                "collect_transitions",
                "collect_success_demos",
                "dagger_pretrain",
                "collect_no_visibility_transitions",
            ],
        )
        transition_cmd = manifest["jobs"][0]["command"]
        self.assertIn("radar5_benchmark_medium.json", " ".join(transition_cmd))

    def test_main_methods_cover_paper_table(self):
        manifest = build_manifest(_args(stage="train_main", seeds="11"))
        ids = [job["id"] for job in manifest["jobs"]]
        self.assertEqual(ids, ["train_hgat_matd3_opt_seed11", "train_mappo_seed11"])
        mappo = " ".join(manifest["jobs"][1]["command"])
        self.assertNotIn("--env-config", mappo)
        self.assertIn("--checkpoint", mappo)
        matd3 = " ".join(manifest["jobs"][0]["command"])
        self.assertIn("--critic-pretrain-updates", matd3)
        self.assertIn("--demo-bc-weight", matd3)

    def test_ablations_include_a0_to_a4_with_matching_priors(self):
        manifest = build_manifest(_args(stage="train_ablation", seeds="11"))
        ids = [job["id"] for job in manifest["jobs"]]
        self.assertTrue(ids[0].startswith("ablation_A0"))
        self.assertTrue(any(job["id"].startswith("ablation_A4") for job in manifest["jobs"]))
        a4 = next(job for job in manifest["jobs"] if job["id"].startswith("ablation_A4"))
        joined = " ".join(a4["command"])
        self.assertIn("radar5_ablation_no_visibility_medium.json", joined)
        self.assertIn("no_visibility.pt", joined)
        self.assertEqual(len(ABLATION_VARIANTS), 5)

    def test_ablation_eval_is_train_domain_only(self):
        manifest = build_manifest(_args(stage="evaluate_ablation", seeds="11"))
        ids = [job["id"] for job in manifest["jobs"]]
        self.assertTrue(all("_medium" in job_id for job_id in ids))
        self.assertFalse(any("mappo" in job_id for job_id in ids))
        self.assertTrue(any("ablation_a0" in job_id for job_id in ids))

    def test_extreme_stage_only_targets_extreme(self):
        manifest = build_manifest(_args(stage="evaluate_extreme", seeds="11"))
        ids = [job["id"] for job in manifest["jobs"]]
        self.assertTrue(ids)
        self.assertTrue(all(job_id.endswith("_extreme") for job_id in ids))
        self.assertTrue(any("hgat_matd3_opt" in job_id for job_id in ids))
        self.assertTrue(any("mappo" in job_id for job_id in ids))

    def test_difficulty_aggregate_requires_five_main_methods(self):
        manifest = build_manifest(_args(stage="aggregate", seeds="11,12"))
        compare = next(job for job in manifest["jobs"] if job["id"] == "difficulty_five_methods")
        joined = " ".join(compare["command"])
        for method in ("APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3_OPT"):
            self.assertIn(method, joined)
        self.assertIn("--require-methods", joined)
        self.assertNotIn("Extreme", joined)


if __name__ == "__main__":
    unittest.main()
