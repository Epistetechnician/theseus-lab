from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import replacement_loss_v7_fixtures as v7_fixtures
import replacement_loss_v8_fixtures as v8_fixtures
import replacement_loss_v8s4_fixtures as fixtures
import replacement_loss_v8s4_runner as runner


class ReplacementLossV8ScaleR2Tests(unittest.TestCase):
    def test_v8s4_partitions_are_namespaced_and_disjoint_from_v7_and_v8(self) -> None:
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
        self.assertTrue(all(row["input_text"].startswith("V8S4 ") for row in visible["records"]))

    def test_v8s4_treatment_wrapper_uses_pinned_independent_fit_case(self) -> None:
        expected = {"fit_elapsed_wall_seconds": 1.0, "treatment_fit": {"method": "ridge"}}

        with patch.object(runner, "_ORIGINAL_FIT_CASE", return_value=expected) as fit:
            result = runner._fit_case_v8s4(
                object(), object(), object(), 0, 0, [{"example_id": "fixture"}], Path("/external")
            )

        fit.assert_called_once()
        self.assertEqual(
            result["v8s4_treatment_method"],
            "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        )
        self.assertTrue(result["v8s4_adaptation_only"])
        self.assertTrue(result["v8s4_no_target_parameter_copy"])
        self.assertGreaterEqual(result["fit_elapsed_wall_seconds"], 1.0)

    def test_checkpoint_exclusion_covers_v7_fresh_targets(self) -> None:
        self.assertIn("babylm/opt-125m-strict-small-2023", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("erwanf/gpt2-mini", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("pgryko/babyllama-10m", runner.PRIOR_CHECKPOINT_IDS)

    def test_scale_profile_raises_parameter_cap_and_restores_it(self) -> None:
        import replacement_loss_v3_runner as v3

        before = v3.MAX_PARAMETER_COUNT
        with runner._activate_v8s4():
            self.assertEqual(v3.MAX_PARAMETER_COUNT, 9_000_000_000)
        self.assertEqual(v3.MAX_PARAMETER_COUNT, before)

    def test_generated_prior_payloads_need_no_custody(self) -> None:
        payloads, digests = runner._generated_prior_payloads([])
        self.assertGreater(len(payloads), 8)
        self.assertEqual(len(payloads), len(digests))

    def test_pinned_sources_verify_without_custody(self) -> None:
        self.assertEqual(runner._verify_pinned_v7()["packet_sha256"], runner.V7_PACKET_SHA256)

    def test_arm_profile_keeps_prompts_and_direction_and_restores_everything(self) -> None:
        import numpy as np
        import replacement_loss_v3_runner as v3

        base_prompts = dict(fixtures.ADAPTATION_PROMPTS)
        original_direction = v3._direction
        original_fit = v3.fit_independent_linear_replacement
        record = {"example_id": "a" * 24, "stratum": "ordinary"}
        activation = np.ones(16, dtype=np.float32)
        entropy = b"x" * 16
        reference = original_direction(record, activation, entropy)
        with runner._arm_profile("R2"):
            self.assertEqual(fixtures.ADAPTATION_PROMPTS, base_prompts)
            np.testing.assert_allclose(v3._direction(record, activation, entropy), reference, rtol=1e-6)
            self.assertIsNot(v3.fit_independent_linear_replacement, original_fit)
        self.assertIs(v3._direction, original_direction)
        self.assertIs(v3.fit_independent_linear_replacement, original_fit)
        self.assertEqual(fixtures.ADAPTATION_PROMPTS, base_prompts)

    def test_relative_ridge_is_alpha_times_mean_gram_eigenvalue(self) -> None:
        import torch
        import replacement_loss_v3_runner as v3
        from unittest.mock import patch

        generator = torch.Generator().manual_seed(7)
        x = torch.randn(40, 8, generator=generator, dtype=torch.float32) * 3.0 + 1.0
        y = torch.randn(40, 4, generator=generator, dtype=torch.float32)
        seen: dict[str, float] = {}

        def fake(in_features, out_features, xx, yy, *, use_bias, seed, ridge=1e-6):
            seen["ridge"] = ridge
            return "module", {"ridge": ridge}

        with patch.object(runner, "_ORIGINAL_FIT", fake):
            with runner._arm_profile("R4"):
                for use_bias, data in ((True, x - x.mean(dim=0)), (False, x)):
                    v3.fit_independent_linear_replacement(8, 4, x, y, use_bias=use_bias, seed=1)
                    expected = 1e-4 * float((data.double() ** 2).sum()) / 8
                    self.assertAlmostEqual(seen["ridge"], expected, places=9)

    def test_arm_alphas_are_the_preregistered_ladder(self) -> None:
        self.assertEqual(sorted(runner.ARMS), ["R0", "R2", "R4", "R6"])
        self.assertEqual(
            {name: arm["ridge_alpha"] for name, arm in runner.ARMS.items()},
            {"R6": 1e-6, "R4": 1e-4, "R2": 1e-2, "R0": 1.0},
        )
        self.assertEqual({arm["adaptation_multiplier"] for arm in runner.ARMS.values()}, {1})
        self.assertEqual({arm["intervention_scale"] for arm in runner.ARMS.values()}, {1})

    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with runner._arm_profile("C4"):
                pass

    def _inner(self, peak_gib: float, **overrides):
        diag = {"module_effect_max_abs_error_vs_treatment": 0.0, "logit_effect_max_abs_error_vs_treatment": 0.0}
        inner = {
            "stage_peak_rss_bytes": int(peak_gib * 1024**3),
            "resource_blockers": ["process_peak_rss_exceeded_24_GiB"] if peak_gib > 24 else [],
            "cases": [{"case_status": "PASS", "mapping_aligned_permutation_diagnostic": dict(diag)} for _ in range(9)],
            "cluster_bootstrap": {
                "treatment_direction_accuracy_95_percent_lower": 0.95,
                "treatment_direction_accuracy_mean": 0.99,
                "paired_treatment_minus_control_95_percent_lower": {"no_op": 0.9, "random": 0.2},
            },
        }
        inner.update(overrides)
        return inner

    def test_reclassify_passes_at_27_gib_under_the_30_gib_limit(self) -> None:
        out = runner._reclassify_outcome(self._inner(27.1))
        self.assertEqual(out["outcome_status"], "PASS")
        self.assertEqual(out["resource_blockers"], [])
        self.assertTrue(out["gates_pass_ignoring_resources"])

    def test_reclassify_is_incomplete_above_30_gib(self) -> None:
        out = runner._reclassify_outcome(self._inner(31.0))
        self.assertEqual(out["outcome_status"], "INCOMPLETE")
        self.assertIn("process_peak_rss_exceeded_30_GiB", out["resource_blockers"])
        self.assertTrue(out["gates_pass_ignoring_resources"])

    def test_reclassify_fails_on_a_failed_case_and_never_hides_it(self) -> None:
        inner = self._inner(20.0)
        inner["cases"][3]["case_status"] = "FAIL"
        out = runner._reclassify_outcome(inner)
        self.assertEqual(out["outcome_status"], "FAIL")
        self.assertFalse(out["gates_pass_ignoring_resources"])

    def test_reclassify_fails_when_bootstrap_or_equivalence_gate_misses(self) -> None:
        low = self._inner(20.0)
        low["cluster_bootstrap"]["treatment_direction_accuracy_95_percent_lower"] = 0.6
        self.assertEqual(runner._reclassify_outcome(low)["outcome_status"], "FAIL")
        eq = self._inner(20.0)
        eq["cases"][0]["mapping_aligned_permutation_diagnostic"]["logit_effect_max_abs_error_vs_treatment"] = 1e-3
        self.assertEqual(runner._reclassify_outcome(eq)["outcome_status"], "FAIL")
        ctl = self._inner(20.0)
        ctl["cluster_bootstrap"]["paired_treatment_minus_control_95_percent_lower"]["random"] = -0.1
        self.assertEqual(runner._reclassify_outcome(ctl)["outcome_status"], "FAIL")


if __name__ == "__main__":
    unittest.main()
