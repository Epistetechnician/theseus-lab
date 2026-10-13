"""Hermetic synthetic checks for replacement-loss-v4.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

import unittest

import numpy as np

import replacement_loss_v4 as v4
import replacement_loss_v4_fixtures as fixtures


class ReplacementLossV4Tests(unittest.TestCase):
    def test_adaptation_arms_share_only_predeclared_ordinary_records(self) -> None:
        pair = fixtures.generate_adaptation_fixtures(0)
        ordinary = pair["ordinary_only"]["records"]
        augmented = pair["compositional_augmented"]["records"]
        self.assertEqual(len(ordinary), 128)
        self.assertEqual(len(augmented), 128)
        self.assertEqual(sum(row["stratum"] == "ordinary" for row in ordinary), 128)
        self.assertEqual(sum(row["stratum"] == "ordinary" for row in augmented), 64)
        self.assertEqual(sum(row["stratum"] == "compositional" for row in augmented), 64)
        self.assertEqual(
            {row["example_id"] for row in ordinary} & {row["example_id"] for row in augmented},
            {row["example_id"] for row in ordinary[:64]},
        )

    def test_challenge_partition_is_fresh_from_both_adaptation_arms(self) -> None:
        arms = fixtures.generate_adaptation_fixtures(1)
        challenge = fixtures.generate_challenge_fixture(1, b"v4-test-entropy-0000000000000000")
        fixtures.validate_fresh_disjointness(arms, challenge)
        self.assertEqual(challenge["count"], 60)
        self.assertEqual(challenge["stratum_counts"], {"adversarial": 20, "compositional": 20, "ordinary": 20})

    def test_activation_metrics_separate_scale_and_orthogonal_error(self) -> None:
        original = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        scaled = 1.1 * original
        metrics = v4.activation_rows(scaled, original)
        np.testing.assert_allclose(metrics["normalized_error"], [0.1, 0.1])
        np.testing.assert_allclose(metrics["signed_parallel_scale_error"], [0.1, 0.1])
        np.testing.assert_allclose(metrics["orthogonal_residual_fraction"], [0.0, 0.0], atol=1e-12)

    def test_seed_gate_requires_improvement_and_behavior_guard(self) -> None:
        strata = np.asarray([name for name in ("ordinary", "compositional", "adversarial") for _ in range(20)])
        original = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (60, 1))
        baseline = original.copy()
        augmented = original.copy()
        baseline[strata == "compositional", 1] = 0.30
        augmented[strata == "compositional", 1] = 0.05
        logits = np.ones((60, 4), dtype=np.float64)
        ce = np.zeros(60, dtype=np.float64)
        result = v4.evaluate_seed(
            strata,
            original,
            logits,
            ce,
            {"ordinary_only": baseline, "compositional_augmented": augmented},
            {"ordinary_only": logits, "compositional_augmented": logits},
            {"ordinary_only": ce, "compositional_augmented": ce},
        )
        self.assertTrue(v4.seed_pass(result))
        augmented[strata == "compositional", 1] = 0.25
        failed = v4.evaluate_seed(
            strata,
            original,
            logits,
            ce,
            {"ordinary_only": baseline, "compositional_augmented": augmented},
            {"ordinary_only": logits, "compositional_augmented": logits},
            {"ordinary_only": ce, "compositional_augmented": ce},
        )
        self.assertFalse(v4.seed_pass(failed))


if __name__ == "__main__":
    unittest.main()
