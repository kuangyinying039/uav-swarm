"""Unit tests for sample-efficiency budget parsing and decay scaling."""
from __future__ import annotations

import unittest

from train_isaac_pursuit import _parse_budget_checkpoints, _scaled_decay_steps


class BudgetProtocolTests(unittest.TestCase):
    def test_parse_budgets_filters_and_sorts(self):
        self.assertEqual(_parse_budget_checkpoints("1000,250,500,250,2000", 1000), (250, 500, 1000))
        self.assertEqual(_parse_budget_checkpoints("", 1000), ())

    def test_scaled_decay_fits_short_budget(self):
        self.assertEqual(_scaled_decay_steps(1000), 100_000)
        self.assertEqual(_scaled_decay_steps(3000), 150_000)
        self.assertGreaterEqual(_scaled_decay_steps(100), 20_000)


if __name__ == "__main__":
    unittest.main()
