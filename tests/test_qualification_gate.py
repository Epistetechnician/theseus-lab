import unittest
from math import nan

from qualification_gate import (
    STATE_SLICE,
    evaluate_alignment_evidence,
    evaluate_mechanistic_evidence,
    evaluate_production_evidence,
    qualification_summary,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64
DIGEST_D = "d" * 64
DIGEST_E = "e" * 64
DIGEST_F = "f" * 64


def alignment_payload() -> dict[str, object]:
    return {
        "state_slice": STATE_SLICE,
        "target_digest": DIGEST_A,
        "assessment_corpus_digest": DIGEST_B,
        "selection_probe_digest": DIGEST_C,
        "assessment_probe_digest": DIGEST_D,
        "thresholds_digest": DIGEST_E,
        "assessment_receipt_digest": DIGEST_F,
        "thresholds_locked_before_assessment": True,
        "real_assessment_data_consumed": True,
        "aggregate_only": True,
        "heldout_observation_count": 64,
        "pass_rate": 0.95,
        "minimum_pass_rate": 0.9,
    }


def production_payload() -> dict[str, object]:
    return {
        "state_slice": STATE_SLICE,
        "weights_loaded": True,
        "provider_called": True,
        "provider_name": "named-provider",
        "provider_model": "named-model",
        "checkpoint_digest": DIGEST_A,
        "provider_input_digest": DIGEST_B,
        "provider_output_digest": DIGEST_C,
        "provider_receipt_digest": DIGEST_D,
        "assessment_corpus_digest": DIGEST_E,
        "latency_slo_verified": True,
        "repeatability_verified": True,
        "observability_verified": True,
        "failure_recovery_verified": True,
        "rollback_verified": True,
        "credentials_persisted": False,
        "production_authorization": True,
    }


class TestQualificationGate(unittest.TestCase):
    def test_mechanistic_gate_requires_all_identity_predicates(self) -> None:
        payload = {
            "state_slice": STATE_SLICE,
            "weights_loaded": True,
            "heldout_observation_count": 576,
            "checkpoint_digest": DIGEST_A,
            "probe_set_digest": DIGEST_B,
            "thresholds_digest": DIGEST_C,
            "raw_digest_diff": True,
            "canonical_digest_match": True,
            "blind_mapping_match": True,
            "behavior_match": True,
            "aligned_intervention_match": True,
            "control_canonical_digest_diff": True,
        }
        result = evaluate_mechanistic_evidence(payload)
        self.assertTrue(result.passed)
        self.assertEqual(result.claim_ceiling, "LOCAL_VALIDATED_EQUIVALENCE_CLASS_OBSERVATION_ONLY")

        payload["blind_mapping_match"] = False
        failed = evaluate_mechanistic_evidence(payload)
        self.assertFalse(failed.passed)
        self.assertIn("blind_mapping_match_failed", failed.blockers)

    def test_alignment_gate_rejects_reused_probes_and_unlocked_thresholds(self) -> None:
        payload = alignment_payload()
        result = evaluate_alignment_evidence(payload)
        self.assertTrue(result.passed)

        payload["assessment_probe_digest"] = payload["selection_probe_digest"]
        payload["thresholds_locked_before_assessment"] = False
        failed = evaluate_alignment_evidence(payload)
        self.assertFalse(failed.passed)
        self.assertIn("selection_and_assessment_probes_not_disjoint", failed.blockers)
        self.assertIn("thresholds_not_locked_before_assessment", failed.blockers)

    def test_alignment_gate_rejects_local_synthetic_only_evidence(self) -> None:
        payload = alignment_payload()
        payload["real_assessment_data_consumed"] = False
        failed = evaluate_alignment_evidence(payload)
        self.assertFalse(failed.passed)
        self.assertIn("real_assessment_data_not_consumed", failed.blockers)

    def test_malformed_attestation_values_fail_closed(self) -> None:
        payload = alignment_payload()
        payload["heldout_observation_count"] = "64"
        payload["pass_rate"] = nan
        payload["minimum_pass_rate"] = True
        failed = evaluate_alignment_evidence(payload)
        self.assertFalse(failed.passed)
        self.assertIn("heldout_assessment_observations_missing", failed.blockers)
        self.assertIn("pass_rate_invalid", failed.blockers)
        self.assertIn("minimum_pass_rate_invalid", failed.blockers)

    def test_production_gate_requires_alignment_and_operations_evidence(self) -> None:
        alignment = evaluate_alignment_evidence(alignment_payload())
        result = evaluate_production_evidence(production_payload(), alignment)
        self.assertTrue(result.passed)

        payload = production_payload()
        payload["provider_called"] = False
        payload["rollback_verified"] = False
        failed = evaluate_production_evidence(payload, alignment)
        self.assertFalse(failed.passed)
        self.assertIn("provider_not_called", failed.blockers)
        self.assertIn("rollback_unverified", failed.blockers)

        blocked_alignment = evaluate_alignment_evidence({})
        blocked = evaluate_production_evidence(production_payload(), blocked_alignment)
        self.assertFalse(blocked.passed)
        self.assertIn("alignment_validation_missing", blocked.blockers)

        malformed = evaluate_production_evidence(production_payload(), None)
        self.assertFalse(malformed.passed)
        self.assertIn("alignment_validation_missing", malformed.blockers)

    def test_summary_preserves_separate_claim_ceilings(self) -> None:
        mechanistic = evaluate_mechanistic_evidence({})
        alignment = evaluate_alignment_evidence({})
        production = evaluate_production_evidence({}, alignment)
        summary = qualification_summary(mechanistic, alignment, production)

        self.assertFalse(summary["all_gates_passed"])
        self.assertEqual(
            summary["mechanistic_equivalence_observation"]["claim_ceiling"],
            "LOCAL_VALIDATED_EQUIVALENCE_CLASS_OBSERVATION_ONLY",
        )
        self.assertEqual(
            summary["alignment_validation"]["claim_ceiling"],
            "HELD_OUT_ALIGNMENT_ASSESSMENT_VALIDATION_ONLY",
        )
        self.assertEqual(
            summary["production_readiness"]["claim_ceiling"],
            "PRODUCTION_READINESS_VALIDATION_ONLY",
        )


if __name__ == "__main__":
    unittest.main()
