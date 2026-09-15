import unittest

from opt_causal_prediction_check import _tie_aware_win_rate
from opt_causal_prediction_v2 import _prediction_gate


class TestCausalStatistics(unittest.TestCase):
    def test_tie_aware_win_rate_assigns_half_credit_to_ties(self) -> None:
        result = _tie_aware_win_rate([0.0, 1.0, 2.0, 1.0], [0.0, 2.0, 1.0, 1.0], 70000)

        self.assertEqual(result["comparison_count"], 4)
        self.assertEqual(result["strict_wins"], 1)
        self.assertEqual(result["ties"], 2)
        self.assertEqual(result["losses"], 1)
        self.assertEqual(result["canonical_tie_aware_win_rate"], 0.5)
        self.assertEqual(len(result["canonical_tie_aware_win_rate_ci95"]), 2)

    def test_tie_aware_win_rate_rejects_unpaired_inputs(self) -> None:
        with self.assertRaises(ValueError):
            _tie_aware_win_rate([], [], 70001)
        with self.assertRaises(ValueError):
            _tie_aware_win_rate([0.0], [], 70002)

    def test_v2_gate_uses_tie_aware_random_control(self) -> None:
        passing = {
            "canonical": {"mean_relative_l2": 0.0},
            "canonical_beats_raw": {"canonical_win_rate": 1.0},
            "canonical_beats_random_tie_aware": {"canonical_tie_aware_win_rate": 0.8},
        }
        failing = {
            **passing,
            "canonical_beats_random_tie_aware": {"canonical_tie_aware_win_rate": 0.7999},
        }

        self.assertTrue(_prediction_gate(passing))
        self.assertFalse(_prediction_gate(failing))


if __name__ == "__main__":
    unittest.main()
