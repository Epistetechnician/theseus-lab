"""Tests for the v9 execution-continuation read-time compatibility bridge.

State slice: theseus-lab-sandbox-v1.
"""

import unittest

import replacement_loss_v9_continuation as continuation
import replacement_loss_v9_runner as runner


class CandidateFreezeAliasTests(unittest.TestCase):
    def test_alias_reads_existing_true_predicate_without_mutating_receipt(self):
        frozen = {
            "predictions_functionally_bound_before_challenge_entropy": True,
            "prediction_rule_count": 81,
            "candidate_placement_count": 81,
        }
        original_digest = runner._canonical_digest(frozen)
        view = continuation._CandidateFreezeAlias(frozen)

        self.assertTrue(view.get("prediction_rules_functionally_bound_before_challenge_entropy"))
        self.assertTrue(view.get("predictions_functionally_bound_before_challenge_entropy"))
        self.assertEqual(view.get("missing_predicate", "absent"), "absent")
        self.assertNotIn("prediction_rules_functionally_bound_before_challenge_entropy", view)
        self.assertEqual(runner._canonical_digest(dict(view)), original_digest)

    def test_false_original_predicate_is_not_promoted_to_true(self):
        view = continuation._CandidateFreezeAlias(
            {"predictions_functionally_bound_before_challenge_entropy": False}
        )
        self.assertFalse(view.get("prediction_rules_functionally_bound_before_challenge_entropy"))


if __name__ == "__main__":
    unittest.main()
