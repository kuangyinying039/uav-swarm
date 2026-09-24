import unittest

from compare_pursuit_difficulty import validate_matrix


class DifficultyComparisonTests(unittest.TestCase):
    def test_complete_five_method_matrix(self):
        methods = ("APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3")
        rows = [
            {"difficulty": difficulty, "label": method}
            for difficulty in ("Nominal", "Medium", "Hard")
            for method in methods
        ]
        validate_matrix(rows, methods)

    def test_extreme_matrix_is_optional(self):
        methods = ("APF", "MPC")
        rows = [
            {"difficulty": difficulty, "label": method}
            for difficulty in ("Hard", "Extreme")
            for method in methods
        ]
        validate_matrix(rows, methods)
        with self.assertRaisesRegex(ValueError, "Extreme:MPC"):
            validate_matrix(
                [{"difficulty": "Hard", "label": "MPC"}],
                ("MPC",),
                required_difficulties=("Hard", "Extreme"),
            )

    def test_missing_method_is_rejected(self):
        rows = [
            {"difficulty": "Nominal", "label": "MPC"},
            {"difficulty": "Medium", "label": "MPC"},
        ]
        with self.assertRaisesRegex(ValueError, "Medium:APF|Nominal:APF"):
            validate_matrix(rows, ("MPC", "APF"))

    def test_duplicate_pair_is_rejected(self):
        rows = [
            {"difficulty": "Nominal", "label": "MPC"},
            {"difficulty": "Nominal", "label": "MPC"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_matrix(rows)


if __name__ == "__main__":
    unittest.main()
