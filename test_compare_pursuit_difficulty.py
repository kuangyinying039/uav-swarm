import unittest

from compare_pursuit_difficulty import validate_matrix


class DifficultyComparisonTests(unittest.TestCase):
    def test_complete_five_method_matrix(self):
        methods = ("APF", "FRPN", "MPC", "MAPPO", "HGAT_MATD3")
        rows = [
            {"difficulty": difficulty, "label": method}
            for difficulty in ("Nominal", "Medium", "Hard", "Extreme")
            for method in methods
        ]
        validate_matrix(rows, methods)

    def test_missing_method_is_rejected(self):
        rows = [
            {"difficulty": "Nominal", "label": "MPC"},
            {"difficulty": "Medium", "label": "MPC"},
            {"difficulty": "Hard", "label": "MPC"},
            {"difficulty": "Extreme", "label": "APF"},
        ]
        with self.assertRaisesRegex(ValueError, "Extreme:MPC"):
            validate_matrix(rows, ("MPC",))

    def test_three_difficulty_matrix_still_validates(self):
        rows = [
            {"difficulty": difficulty, "label": "MPC"}
            for difficulty in ("Nominal", "Medium", "Hard")
        ]
        validate_matrix(rows, ("MPC",))

    def test_duplicate_pair_is_rejected(self):
        rows = [
            {"difficulty": "Nominal", "label": "MPC"},
            {"difficulty": "Nominal", "label": "MPC"},
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_matrix(rows)


if __name__ == "__main__":
    unittest.main()
