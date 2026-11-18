"""Unit tests for discrete Adaptive Environment Generator."""

from __future__ import annotations

import unittest

from pursuit_aeg import AEGConfig, DiscreteAEG


class DiscreteAEGTests(unittest.TestCase):
    def test_cold_start_balances_under_explored_levels(self):
        aeg = DiscreteAEG(
            AEGConfig(
                levels=("medium", "hard", "severe"),
                window=20,
                min_samples_per_level=3,
                cold_start_level="hard",
            )
        )
        aeg.set_seed(0)
        seen = {aeg.next_level() for _ in range(9)}
        self.assertEqual(seen, {"medium", "hard", "severe"})

    def test_mid_band_preferred_after_enough_samples(self):
        aeg = DiscreteAEG(
            AEGConfig(
                levels=("medium", "hard", "severe"),
                r_min=0.5,
                r_max=0.9,
                window=50,
                min_samples_per_level=3,
                cold_start_level="hard",
            )
        )
        aeg.set_seed(1)
        # medium in band, hard too hard, severe too easy
        for _ in range(6):
            aeg.observe("medium", True)
            aeg.observe("medium", False)  # ~0.5
            aeg.observe("hard", False)
            aeg.observe("hard", False)  # ~0.0
            aeg.observe("severe", True)  # ~1.0
        picks = [aeg.next_level() for _ in range(20)]
        self.assertGreaterEqual(picks.count("medium"), 15)

    def test_stats_keys(self):
        aeg = DiscreteAEG()
        aeg.observe("hard", True)
        stats = aeg.stats()
        self.assertIn("capture_rates", stats)
        self.assertIn("selections", stats)
        self.assertEqual(stats["levels"], ["medium", "hard", "severe"])


if __name__ == "__main__":
    unittest.main()
