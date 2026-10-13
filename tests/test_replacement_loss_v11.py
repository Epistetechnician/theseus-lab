from __future__ import annotations

import unittest

import replacement_loss_v11_fixtures as fixtures
import replacement_loss_v11_runner as runner


class ReplacementLossV11Tests(unittest.TestCase):
    def test_protocol_namespaces_and_seed_count(self) -> None:
        self.assertEqual(fixtures.PROTOCOL_ID, runner.PROTOCOL_ID)
        self.assertEqual(len(fixtures.SEEDS), 3)
        self.assertEqual(len(set(fixtures.SEEDS)), 3)
        self.assertEqual(runner.TARGET_MODEL_ID, "grushaprasad/opt125m-rootp-surprisal-curriculum")

    def test_visible_and_adaptation_partitions_are_disjoint(self) -> None:
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptations = [fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation_{index}") for index in range(3)]
        fixtures.validate_partition_disjointness(visible, *adaptations)
        self.assertTrue(all(row["input_text"].startswith("V11 reused-target ") for row in visible["records"]))

    def test_challenge_fixture_is_entropy_bound_and_disjoint(self) -> None:
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptation = fixtures.fixture_payload(fixtures.generate_adaptation_fixture(0), fixture_kind="adaptation")
        first, entropy = fixtures.generate_challenge_fixture(0, entropy=bytes(range(32)))
        second, repeated = fixtures.generate_challenge_fixture(0, entropy=bytes(range(32)))
        challenge = fixtures.fixture_payload(first, fixture_kind="challenge")
        self.assertEqual(entropy, repeated)
        self.assertEqual(challenge["example_set_sha256"], fixtures.fixture_payload(second, fixture_kind="challenge")["example_set_sha256"])
        fixtures.validate_partition_disjointness(visible, adaptation, challenge)

    def test_short_entropy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fixtures.generate_challenge_fixture(0, entropy=b"short")

    def test_secondary_endpoint_targets_only_v9_missed_cells(self) -> None:
        report = {"results": []}
        for seed in range(3):
            for layer in (0, 7):
                for method in ("activation_ridge", "causal_fit"):
                    per_stratum = {}
                    for stratum in fixtures.STRATA:
                        per_stratum[stratum] = {"next_logits_causal_effect": {
                            "resolvable_count": 20,
                            "direction_accuracy": 1.0,
                            "median_cosine": 0.95 if method == "causal_fit" else 0.8,
                            "median_normalized_error": 0.1,
                        }}
                    report["results"].append({"seed_index": seed, "layer": layer, "method": method, "per_stratum": per_stratum})
        summary = runner._failure_cell_summary(report)
        self.assertEqual(len(summary["cells"]), 9)
        self.assertEqual(summary["causal_fit_cells_passing"], 9)
        self.assertEqual(summary["activation_ridge_cells_passing"], 0)

    def test_failure_cell_summary_is_not_a_primary_gate(self) -> None:
        report = {"results": []}
        for seed in range(3):
            for layer in (0, 7):
                for method in ("activation_ridge", "causal_fit"):
                    per_stratum = {stratum: {"next_logits_causal_effect": {
                        "resolvable_count": 20,
                        "direction_accuracy": 0.0,
                        "median_cosine": 0.0,
                        "median_normalized_error": 2.0,
                    }} for stratum in fixtures.STRATA}
                    report["results"].append({"seed_index": seed, "layer": layer, "method": method, "per_stratum": per_stratum})
        summary = runner._failure_cell_summary(report)
        self.assertIn("not an additional PASS gate", summary["role"])
        self.assertEqual(summary["causal_fit_cells_passing"], 0)


if __name__ == "__main__":
    unittest.main()
