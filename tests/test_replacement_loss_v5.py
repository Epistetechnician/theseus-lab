"""Hermetic synthetic checks for replacement-loss-v5.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

import unittest

import numpy as np

import replacement_loss_v5 as v5
import replacement_loss_v5_fixtures as fixtures


class ReplacementLossV5Tests(unittest.TestCase):
    def test_adaptation_arms_have_equal_rows_and_declared_shared_support(self) -> None:
        arms = fixtures._generate_adaptation_fixtures_for_seed(0, fixtures.DEVELOPMENT_TEST_SEEDS[0])
        self.assertEqual(set(arms), set(v5.ARMS))
        for arm in arms.values():
            self.assertEqual(arm["count"], 128)
        ordinary_ids = {row["example_id"] for row in arms["ordinary_only"]["records"][:64]}
        comp_ids = {row["example_id"] for row in arms["compositional_augmented"]["records"][:64]}
        balanced = arms["balanced_augmented"]["records"]
        self.assertEqual(ordinary_ids, comp_ids)
        self.assertEqual({row["stratum"] for row in balanced[:64]}, {"ordinary"})
        self.assertEqual(sum(row["stratum"] == "compositional" for row in balanced), 32)
        self.assertEqual(sum(row["stratum"] == "adversarial" for row in balanced), 32)
        self.assertEqual(len({row["example_id"] for row in balanced}), 128)

    def test_fresh_challenge_is_disjoint_from_adaptation_union(self) -> None:
        arms = fixtures._generate_adaptation_fixtures_for_seed(1, fixtures.DEVELOPMENT_TEST_SEEDS[1])
        challenge = fixtures.generate_challenge_fixture(1, fixtures.TEST_CHALLENGE_ENTROPY)
        fixtures.validate_partition_disjointness([fixtures.union_adaptation_fixture(arms), challenge])
        self.assertEqual(challenge["count"], 60)
        self.assertEqual(challenge["stratum_counts"], {"adversarial": 20, "compositional": 20, "ordinary": 20})

    def test_balanced_seed_gate_requires_behavior_and_both_support_effects(self) -> None:
        strata = np.asarray([name for name in v5.STRATA for _ in range(20)])
        original = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (60, 1))
        activations = {arm: original.copy() for arm in v5.ARMS}
        activations["ordinary_only"][strata == "compositional", 1] = 0.30
        activations["compositional_augmented"][strata == "compositional", 1] = 0.05
        activations["compositional_augmented"][strata == "adversarial", 1] = 0.30
        activations["balanced_augmented"][strata == "compositional", 1] = 0.05
        activations["balanced_augmented"][strata == "adversarial", 1] = 0.05
        logits = {arm: np.ones((60, 4), dtype=np.float64) for arm in v5.ARMS}
        logits["compositional_augmented"][strata == "adversarial", 1] += 0.16
        logits["balanced_augmented"][strata == "adversarial", 1] += 0.08
        ce = {arm: np.zeros(60, dtype=np.float64) for arm in v5.ARMS}
        result = v5.evaluate_seed(strata, original, np.ones((60, 4)), np.zeros(60), activations, logits, ce)
        self.assertTrue(v5.seed_pass(result))
        logits["balanced_augmented"][strata == "adversarial", 1] += 0.04
        failed = v5.evaluate_seed(strata, original, np.ones((60, 4)), np.zeros(60), activations, logits, ce)
        self.assertFalse(v5.seed_pass(failed))


if __name__ == "__main__":
    unittest.main()
