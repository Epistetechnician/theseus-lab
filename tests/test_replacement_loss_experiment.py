"""Tests for the Theseus structural replacement protocol."""

import unittest

import numpy as np

from replacement_loss_experiment import (
    CONTROL_RELATIVE_L2_MIN,
    TREATMENT_RELATIVE_L2_MAX,
    _behavior_summary,
)


class TestReplacementLossProtocol(unittest.TestCase):
    def test_behavior_summary_reports_exact_equivalence(self) -> None:
        baseline = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0])]
        self.assertEqual(_behavior_summary(baseline, baseline)["top1_agreement"], 1.0)
        self.assertEqual(_behavior_summary(baseline, baseline)["mean_relative_l2"], 0.0)

    def test_locked_thresholds_are_separated(self) -> None:
        self.assertLess(TREATMENT_RELATIVE_L2_MAX, CONTROL_RELATIVE_L2_MIN)

    def test_behavior_summary_rejects_empty_pairs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            _behavior_summary([], [])


if __name__ == "__main__":
    unittest.main()
