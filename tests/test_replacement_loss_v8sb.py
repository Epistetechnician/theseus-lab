from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import replacement_loss_v7_fixtures as v7_fixtures
import replacement_loss_v8_fixtures as v8_fixtures
import replacement_loss_v8sb_fixtures as fixtures
import replacement_loss_v8sb_runner as runner


class ReplacementLossV8ScaleBandBTests(unittest.TestCase):
    def test_v8sb_partitions_are_namespaced_and_disjoint_from_v7_and_v8(self) -> None:
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
        self.assertTrue(all(row["input_text"].startswith("V8SB ") for row in visible["records"]))

    def test_v8sb_treatment_wrapper_uses_pinned_independent_fit_case(self) -> None:
        expected = {"fit_elapsed_wall_seconds": 1.0, "treatment_fit": {"method": "ridge"}}

        with patch.object(runner, "_ORIGINAL_FIT_CASE", return_value=expected) as fit:
            result = runner._fit_case_v8sb(
                object(), object(), object(), 0, 0, [{"example_id": "fixture"}], Path("/external")
            )

        fit.assert_called_once()
        self.assertEqual(
            result["v8sb_treatment_method"],
            "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        )
        self.assertTrue(result["v8sb_adaptation_only"])
        self.assertTrue(result["v8sb_no_target_parameter_copy"])
        self.assertGreaterEqual(result["fit_elapsed_wall_seconds"], 1.0)

    def test_checkpoint_exclusion_covers_v7_fresh_targets(self) -> None:
        self.assertIn("babylm/opt-125m-strict-small-2023", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("erwanf/gpt2-mini", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("pgryko/babyllama-10m", runner.PRIOR_CHECKPOINT_IDS)

    def test_scale_profile_raises_parameter_cap_and_restores_it(self) -> None:
        import replacement_loss_v3_runner as v3

        before = v3.MAX_PARAMETER_COUNT
        with runner._activate_v8sb():
            self.assertEqual(v3.MAX_PARAMETER_COUNT, 9_000_000_000)
        self.assertEqual(v3.MAX_PARAMETER_COUNT, before)

    def test_generated_prior_payloads_need_no_custody(self) -> None:
        payloads, digests = runner._generated_prior_payloads([])
        self.assertGreater(len(payloads), 8)
        self.assertEqual(len(payloads), len(digests))

    def test_pinned_sources_verify_without_custody(self) -> None:
        self.assertEqual(runner._verify_pinned_v7()["packet_sha256"], runner.V7_PACKET_SHA256)

    LINES = [
        " ".join(f"word{i}_{j}" for j in range(70)) for i in range(400)
    ] + ["= Heading =", "short line only", ""]

    def test_eligible_paragraphs_filters_headings_short_lines_and_duplicates(self) -> None:
        lines = self.LINES + [self.LINES[0], "  " + self.LINES[1] + "  "]
        eligible = fixtures.eligible_paragraphs(lines)
        self.assertEqual(len(eligible), 400)
        self.assertTrue(all(len(text.split()) >= fixtures.NATURAL_MIN_WORDS for text in eligible))
        self.assertFalse(any(text.startswith("=") for text in eligible))

    def test_natural_selection_is_deterministic_truncated_and_disjoint_across_seeds(self) -> None:
        paragraphs = fixtures.eligible_paragraphs(self.LINES)
        picks = [fixtures.select_natural_prompts(paragraphs, index, 40) for index in range(3)]
        self.assertEqual(picks[0], fixtures.select_natural_prompts(paragraphs, 0, 40))
        for prompts in picks:
            self.assertEqual(len(prompts), 40)
            self.assertTrue(all(len(p.split()) == fixtures.NATURAL_WORDS for p in prompts))
        flat = [p for prompts in picks for p in prompts]
        self.assertEqual(len(set(flat)), 120)
        with self.assertRaises(ValueError):
            fixtures.select_natural_prompts(paragraphs, 0, 10_000)

    def test_natural_records_form_a_valid_disjoint_fixture(self) -> None:
        records = fixtures.natural_records(1, 64, lines=self.LINES)
        self.assertEqual({row.stratum for row in records}, {"natural"})
        self.assertEqual(len({row.example_id for row in records}), 64)
        payload = fixtures.fixture_payload(records, fixture_kind="adaptation")
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible_selection")
        fixtures.validate_partition_disjointness(visible, payload)

    def test_wikitext_file_must_match_the_pinned_hash(self) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"not the pinned file")
        try:
            with patch.dict(os.environ, {fixtures.WIKITEXT_ENV: handle.name}):
                with self.assertRaises(ValueError):
                    fixtures.natural_records(0, 8)
        finally:
            os.unlink(handle.name)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(fixtures.WIKITEXT_ENV, None)
            with self.assertRaises(RuntimeError):
                fixtures.natural_records(0, 8)

    def test_arm_profile_switches_source_and_restores_everything(self) -> None:
        base_prompts = dict(fixtures.ADAPTATION_PROMPTS)
        self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.NATURAL_PROMPTS), ("template", 128))
        with runner._arm_profile("N256"):
            self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.NATURAL_PROMPTS), ("natural", 256))
            self.assertEqual(fixtures.ADAPTATION_PROMPTS, base_prompts)
        with self.assertRaises(ValueError):
            with runner._arm_profile("T256"):
                pass
        self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.NATURAL_PROMPTS), ("template", 128))
        self.assertEqual(fixtures.ADAPTATION_PROMPTS, base_prompts)

    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with runner._arm_profile("C4"):
                pass

    def test_bandb_arms_are_the_preregistered_natural_text_pair(self) -> None:
        self.assertEqual(sorted(runner.ARMS), ["N256", "N512"])
        self.assertEqual({n: runner.ARMS[n]["adaptation_prompts"] for n in runner.ARMS}, {"N256": 256, "N512": 512})
        self.assertTrue(all(arm["mode"] == "natural" for arm in runner.ARMS.values()))

    def test_packet_chain_extends_r5_and_has_the_expected_file_count(self) -> None:
        files = runner.V8SB_PACKET_FILES
        self.assertEqual(len(files), len(set(files)))
        self.assertEqual(len(files), 55)
        for name in ("docs/replacement-loss-v8s5-freeze.json", "replacement_loss_v8s5_runner.py", "tests/test_replacement_loss_v8s5.py", "docs/replacement-loss-v8-scale-r5.md", "docs/replacement-loss-v8-scale-r4.md"):
            self.assertIn(name, files)

    def test_device_shim_patches_apply_to_the_pinned_v3_source_and_restore(self) -> None:
        import os
        import replacement_loss_v3_runner as v3

        for name in runner._SHIM_SOURCE_PATCHES:
            self.assertTrue(callable(runner._patched_v3_function(name)))
        before = (v3._load_local_model, v3._fit_case, v3._run_projection, v3._set_projection, runner._ORIGINAL_FIT_CASE)
        with patch.dict(os.environ, {runner.DEVICE_ENV: "cpu"}):
            with runner._device_shim():
                self.assertIs(v3._run_projection, before[2])
        self.assertEqual(before, (v3._load_local_model, v3._fit_case, v3._run_projection, v3._set_projection, runner._ORIGINAL_FIT_CASE))
        import torch
        if torch.cuda.is_available():
            with runner._device_shim():
                self.assertIsNot(v3._run_projection, before[2])
            self.assertEqual(before, (v3._load_local_model, v3._fit_case, v3._run_projection, v3._set_projection, runner._ORIGINAL_FIT_CASE))

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

    def test_reclassify_passes_at_90_gib_under_the_100_gib_limit(self) -> None:
        out = runner._reclassify_outcome(self._inner(90.1))
        self.assertEqual(out["outcome_status"], "PASS")
        self.assertEqual(out["resource_blockers"], [])
        self.assertTrue(out["gates_pass_ignoring_resources"])

    def test_reclassify_is_incomplete_above_100_gib(self) -> None:
        out = runner._reclassify_outcome(self._inner(101.0))
        self.assertEqual(out["outcome_status"], "INCOMPLETE")
        self.assertIn("process_peak_rss_exceeded_100_GiB", out["resource_blockers"])
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
