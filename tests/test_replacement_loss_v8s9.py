from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import replacement_loss_v7_fixtures as v7_fixtures
import replacement_loss_v8_fixtures as v8_fixtures
import replacement_loss_v8s9_fixtures as fixtures
import replacement_loss_v8s9_runner as runner


class ReplacementLossV8ScaleR6Tests(unittest.TestCase):
    def test_v8s9_partitions_are_namespaced_and_disjoint_from_v7_and_v8(self) -> None:
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
        self.assertTrue(all(row["input_text"].startswith("V8S9 ") for row in visible["records"]))

    def test_v8s9_treatment_wrapper_uses_pinned_independent_fit_case(self) -> None:
        expected = {"fit_elapsed_wall_seconds": 1.0, "treatment_fit": {"method": "ridge"}}

        with patch.object(runner, "_ORIGINAL_FIT_CASE", return_value=expected) as fit:
            result = runner._fit_case_v8s9(
                object(), object(), object(), 0, 0, [{"example_id": "fixture"}], Path("/external")
            )

        fit.assert_called_once()
        self.assertEqual(
            result["v8s9_treatment_method"],
            "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        )
        self.assertTrue(result["v8s9_adaptation_only"])
        self.assertTrue(result["v8s9_no_target_parameter_copy"])
        self.assertGreaterEqual(result["fit_elapsed_wall_seconds"], 1.0)

    def test_checkpoint_exclusion_covers_v7_fresh_targets(self) -> None:
        self.assertIn("babylm/opt-125m-strict-small-2023", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("erwanf/gpt2-mini", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("pgryko/babyllama-10m", runner.PRIOR_CHECKPOINT_IDS)

    def test_scale_profile_raises_parameter_cap_and_restores_it(self) -> None:
        import replacement_loss_v3_runner as v3

        before = v3.MAX_PARAMETER_COUNT
        with runner._activate_v8s9():
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

    def test_news_file_must_match_the_pinned_hash(self) -> None:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"not the pinned file")
        try:
            with patch.dict(os.environ, {fixtures.NEWS_ENV: handle.name}):
                with self.assertRaises(ValueError):
                    fixtures.natural_records(0, 8)
        finally:
            os.unlink(handle.name)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(fixtures.NEWS_ENV, None)
            with self.assertRaises(RuntimeError):
                fixtures.natural_records(0, 8)


    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            with runner._arm_profile("N128"):
                pass



    def test_natural_challenge_is_strata_balanced_disjoint_from_every_adaptation_arm(self) -> None:
        
        lines = [" ".join(f"w{i}_{j}" for j in range(70)) for i in range(3 * 700)]
        paragraphs = fixtures.eligible_paragraphs(lines)
        for index in range(3):
            challenge = fixtures.select_natural_challenge_prompts(paragraphs, index, b"e" * 32)
            adaptation = set(fixtures.select_natural_prompts(paragraphs, index, fixtures.MAX_NATURAL_ADAPTATION))
            self.assertEqual(len(challenge), 60)
            self.assertEqual([sum(s == k for s, _ in challenge) for k in fixtures.STRATA], [20, 20, 20])
            self.assertFalse({p for _, p in challenge} & adaptation)
            self.assertEqual(len({p for _, p in challenge}), 60)
        a = fixtures.select_natural_challenge_prompts(paragraphs, 0, b"a" * 32)
        b = fixtures.select_natural_challenge_prompts(paragraphs, 0, b"b" * 32)
        self.assertNotEqual(a, b)
        self.assertEqual(a, fixtures.select_natural_challenge_prompts(paragraphs, 0, b"a" * 32))

    def test_natural_challenge_records_form_a_valid_disjoint_fixture(self) -> None:
        lines = [" ".join(f"w{i}_{j}" for j in range(70)) for i in range(3 * 700)]
        records = fixtures.natural_challenge_records(2, b"k" * 32, lines=lines)
        payload = fixtures.fixture_payload(records, fixture_kind="challenge")
        self.assertEqual(payload["stratum_counts"], {k: 20 for k in fixtures.STRATA})
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible_selection")
        fixtures.validate_partition_disjointness(visible, payload)

    def test_repeated_challenge_draws_across_families_are_mutually_disjoint(self) -> None:
        lines = [" ".join(f"w{i}_{j}" for j in range(70)) for i in range(3 * 1500)]
        fixtures.USED_CHALLENGE_PROMPTS.clear()
        try:
            payloads = [
                fixtures.fixture_payload(fixtures.natural_challenge_records(seed, bytes([family]) * 32, lines=lines), fixture_kind="challenge")
                for family in range(3)
                for seed in range(3)
            ]
            fixtures.validate_partition_disjointness(*payloads)
        finally:
            fixtures.USED_CHALLENGE_PROMPTS.clear()

    def test_challenge_mode_selects_the_challenge_source(self) -> None:
        template = fixtures.generate_challenge_fixture(0, secret_seed=b"t" * 32)
        self.assertTrue(all(row.input_text.startswith("V8S9 ") for row in template))
        with patch.object(fixtures, "CHALLENGE_MODE", "natural"), patch.object(
            fixtures, "natural_challenge_records", return_value=["sentinel"]
        ) as natural:
            self.assertEqual(fixtures.generate_challenge_fixture(0, secret_seed=b"t" * 32), ["sentinel"])
        natural.assert_called_once()


    def test_arm_profile_switches_sources_and_restores_everything(self) -> None:
        base_prompts = dict(fixtures.ADAPTATION_PROMPTS)
        self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.CHALLENGE_MODE, fixtures.NATURAL_PROMPTS), ("template", "template", 128))
        with runner._arm_profile("NN256"):
            self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.CHALLENGE_MODE, fixtures.NATURAL_PROMPTS), ("natural", "natural", 256))
        self.assertEqual((fixtures.ADAPTATION_MODE, fixtures.CHALLENGE_MODE, fixtures.NATURAL_PROMPTS), ("template", "template", 128))
        self.assertEqual(fixtures.ADAPTATION_PROMPTS, base_prompts)

    def test_r9_arms_are_exactly_n256_and_n512_natural_natural(self) -> None:
        self.assertEqual(sorted(runner.ARMS), ["NN256", "NN512"])
        self.assertEqual({n: runner.ARMS[n]["adaptation_prompts"] for n in runner.ARMS}, {"NN256": 256, "NN512": 512})
        self.assertTrue(all(runner.ARMS[n]["challenge"] == "natural" and runner.ARMS[n]["mode"] == "natural" for n in runner.ARMS))

    def test_corpus_pin_constants_and_filter_mirror_wikitext_paragraph_filter(self) -> None:
        import replacement_loss_v8s5_fixtures as r5

        self.assertEqual(fixtures.NEWS_REVISION, "96df5e686bee6baa90b8bee7c28b81fa3fa6223d")
        self.assertEqual(fixtures.NEWS_SHA256, "65a5ccce932b08f050114ce6e3c39355d563fa24f194052bb2f27d1c6c499c91")
        self.assertEqual((fixtures.NATURAL_WORDS, fixtures.NATURAL_MIN_WORDS), (r5.NATURAL_WORDS, r5.NATURAL_MIN_WORDS))
        lines = self.LINES + ["= Heading = " + "w " * 80, " ".join(["x"] * 59)]
        self.assertEqual(fixtures.eligible_paragraphs(lines), r5.eligible_paragraphs(lines))
        self.assertEqual(fixtures.SEEDS, r5.SEEDS)

    def test_filter_also_deduplicates_on_the_truncated_prompt(self) -> None:
        head = " ".join(f"h{j}" for j in range(45))
        lines = [head + " tail one " + " ".join(["a"] * 20), head + " tail two " + " ".join(["b"] * 20)]
        self.assertEqual(len(fixtures.eligible_paragraphs(lines)), 1)

    def test_real_corpus_shape_adaptation_arms_are_cross_seed_disjoint(self) -> None:
        shared = " ".join(f"s{j}" for j in range(50))
        lines = self.LINES + [shared + f" x{i}" for i in range(3)]
        paragraphs = fixtures.eligible_paragraphs(lines)
        picks = [p for i in range(3) for p in fixtures.select_natural_prompts(paragraphs, i, 100)]
        self.assertEqual(len(picks), len(set(picks)))

    def test_r9_adaptation_differs_from_wikitext_namespace(self) -> None:
        import replacement_loss_v8s5_fixtures as r5

        ours = fixtures.natural_records(0, 16, lines=self.LINES)
        theirs = r5.natural_records(0, 16, lines=self.LINES)
        self.assertNotEqual([r.example_id for r in ours], [r.example_id for r in theirs])

    def test_pinned_chain_covers_r5_to_r7_and_band_b_and_matches_the_freeze(self) -> None:
        import json

        root = Path(runner.__file__).resolve().parent
        chain = set(runner.V8S9_PACKET_FILES)
        self.assertEqual(len(runner.V8S9_PACKET_FILES), len(chain))
        for name in ("s5", "s6", "s7", "sb"):
            freeze = root / f"docs/replacement-loss-v8{name}-freeze.json"
            self.assertIn(f"docs/replacement-loss-v8{name}-freeze.json", chain)
            for row in json.loads(freeze.read_text())["packet_files"]:
                self.assertIn(row["path"], chain)
        self.assertEqual(len(chain), 105)
        for path in runner.V8S9_OWN_FILES:
            self.assertIn(path, chain)
        # Everything tracked for r5-r7 and Band B on main is in the chain (r8 has no files on main).
        import subprocess

        tracked = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
        import re

        pattern = re.compile(r"scale-r[567]|scale-bandb|v8s[567b]-freeze|r7-logit|bandb-open|^scripts/r7_|^scripts/v8sb|^tests/test_r7_|v8s[67b]_|v8s[67b]\.py")
        expected = {f for f in tracked if pattern.search(f)}
        self.assertEqual(expected - chain, set())
        self.assertFalse([f for f in tracked if re.search(r"r8|v8s8", f)])
        for path in chain:
            self.assertTrue((root / path).is_file(), path)
        freeze_path = root / "docs/replacement-loss-v8s9-freeze.json"
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            self.assertEqual([row["path"] for row in frozen["packet_files"]], list(runner.V8S9_PACKET_FILES))
            for row in frozen["packet_files"]:
                self.assertEqual(runner._sha(root / row["path"]), row["sha256"])

    def test_split_gate_report_separates_per_case_from_equivalence_diagnostic(self) -> None:
        inner = self._inner(20.0)
        for i, row in enumerate(inner["cases"]):
            row.update(family="f", seed_index=i)
        inner["cases"][4]["mapping_aligned_permutation_diagnostic"]["module_effect_max_abs_error_vs_treatment"] = 6e-4
        split = runner._split_gate_report(inner)
        self.assertTrue(split["per_case_gates"]["per_case_and_bootstrap_pass"])
        self.assertEqual(split["per_case_gates"]["cases_passing"], 9)
        self.assertFalse(split["equivalence_diagnostic_1e-4"]["all_within_bound"])
        self.assertEqual(runner._reclassify_outcome(inner)["outcome_status"], "FAIL")

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
