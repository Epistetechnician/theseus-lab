from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import replacement_loss_v7_fixtures as v7_fixtures
import replacement_loss_v8_fixtures as v8_fixtures
import replacement_loss_v8s_fixtures as fixtures
import replacement_loss_v8s_runner as runner


class ReplacementLossV8ScaleTests(unittest.TestCase):
    def test_v8s_partitions_are_namespaced_and_disjoint_from_v7_and_v8(self) -> None:
        visible = fixtures.fixture_payload(
            fixtures.generate_visible_fixture(), fixture_kind="visible_selection"
        )
        adaptation = fixtures.fixture_payload(
            fixtures.generate_adaptation_fixture(0), fixture_kind="adaptation"
        )
        challenge = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=b"v8-fixture-test-entropy"),
            fixture_kind="challenge",
        )
        prior = [v7_fixtures.fixture_payload(
            v7_fixtures.generate_visible_fixture(), fixture_kind="visible_selection"
        ), v8_fixtures.fixture_payload(
            v8_fixtures.generate_visible_fixture(), fixture_kind="visible_selection"
        )]

        for earlier in prior:
            fixtures.validate_partition_disjointness(visible, adaptation, challenge, earlier)

        self.assertEqual(visible["protocol_id"], fixtures.PROTOCOL_ID)
        self.assertEqual(visible["stratum_counts"], {"visible": fixtures.VISIBLE_COUNT})
        self.assertEqual(adaptation["stratum_counts"], fixtures.ADAPTATION_PROMPTS)
        self.assertEqual(challenge["stratum_counts"], {key: 20 for key in fixtures.STRATA})
        self.assertTrue(all(row["input_text"].startswith("V8S ") for row in visible["records"]))

    def test_v8s_treatment_wrapper_uses_pinned_independent_fit_case(self) -> None:
        expected = {"fit_elapsed_wall_seconds": 1.0, "treatment_fit": {"method": "ridge"}}

        with patch.object(runner, "_ORIGINAL_FIT_CASE", return_value=expected) as fit:
            result = runner._fit_case_v8s(
                object(), object(), object(), 0, 0, [{"example_id": "fixture"}], Path("/external")
            )

        fit.assert_called_once()
        self.assertEqual(
            result["v8s_treatment_method"],
            "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        )
        self.assertTrue(result["v8s_adaptation_only"])
        self.assertTrue(result["v8s_no_target_parameter_copy"])
        self.assertGreaterEqual(result["fit_elapsed_wall_seconds"], 1.0)

    def test_checkpoint_exclusion_covers_v7_fresh_targets(self) -> None:
        self.assertIn("babylm/opt-125m-strict-small-2023", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("erwanf/gpt2-mini", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("pgryko/babyllama-10m", runner.PRIOR_CHECKPOINT_IDS)

    def test_scale_profile_raises_parameter_cap_and_restores_it(self) -> None:
        import replacement_loss_v3_runner as v3

        before = v3.MAX_PARAMETER_COUNT
        with runner._activate_v8s():
            self.assertEqual(v3.MAX_PARAMETER_COUNT, 9_000_000_000)
        self.assertEqual(v3.MAX_PARAMETER_COUNT, before)

    def test_generated_prior_payloads_need_no_custody(self) -> None:
        payloads, digests = runner._generated_prior_payloads([])
        self.assertGreater(len(payloads), 8)
        self.assertEqual(len(payloads), len(digests))

    def test_pinned_sources_verify_without_custody(self) -> None:
        self.assertEqual(runner._verify_pinned_v7()["packet_sha256"], runner.V7_PACKET_SHA256)


if __name__ == "__main__":
    unittest.main()
