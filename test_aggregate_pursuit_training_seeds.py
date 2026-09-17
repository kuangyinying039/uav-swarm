import unittest

from aggregate_pursuit_training_seeds import aggregate_payloads


def payload(captures):
    rows = [
        {"seed": 7000000 + index, "captured": bool(value)}
        for index, value in enumerate(captures)
    ]
    return {
        "env_config": {"scenario_version": 3},
        "rows": rows,
        "summary": {
            "matd3": {
                "episodes": len(rows),
                "capture_rate": sum(captures) / len(captures),
                "mean_censored_steps": 100.0,
            }
        },
    }


class AggregateTrainingSeedTests(unittest.TestCase):
    def test_hierarchical_aggregate(self):
        result = aggregate_payloads(
            [payload([1, 1, 0, 0]), payload([1, 0, 0, 0])],
            "HGAT_MATD3", bootstrap_samples=200, seed=1,
        )
        metrics = result["summary"]["HGAT_MATD3"]
        self.assertEqual(metrics["episodes"], 4)
        self.assertEqual(metrics["training_runs"], 2)
        self.assertAlmostEqual(metrics["capture_rate"], 0.375)
        self.assertLessEqual(metrics["capture_ci_low"], metrics["capture_rate"])
        self.assertGreaterEqual(metrics["capture_ci_high"], metrics["capture_rate"])

    def test_requires_paired_scenarios(self):
        second = payload([1, 0, 0])
        with self.assertRaisesRegex(ValueError, "same ordered scenario seeds"):
            aggregate_payloads([payload([1, 0]), second], "MATD3", bootstrap_samples=100)


if __name__ == "__main__":
    unittest.main()
