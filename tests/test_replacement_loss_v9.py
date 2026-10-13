"""Focused contracts for replacement-loss-v9.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM

import replacement_loss_v3_fixtures as v3_fixtures
import replacement_loss_v9_fixtures as fixtures
import replacement_loss_v9_replicator as replicator
import replacement_loss_v9_runner as runner


class ReplacementLossV9FixtureTests(unittest.TestCase):
    def test_visible_and_adaptation_partitions_match_frozen_counts(self) -> None:
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptation = [
            fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation-{index}")
            for index in range(len(fixtures.SEEDS))
        ]

        fixtures.validate_partition_disjointness(visible, *adaptation)

        self.assertEqual(visible["count"], fixtures.VISIBLE_COUNT)
        self.assertEqual(visible["stratum_counts"], {"visible": fixtures.VISIBLE_COUNT})
        for payload in adaptation:
            self.assertEqual(payload["count"], sum(fixtures.ADAPTATION_COUNTS.values()))
            self.assertEqual(payload["stratum_counts"], fixtures.ADAPTATION_COUNTS)
            self.assertTrue(all(row["input_text"].startswith("V9 ") for row in payload["records"]))

    def test_challenge_partitions_are_seeded_by_post_freeze_entropy(self) -> None:
        first = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=b"v9-test-entropy-0001"),
            fixture_kind="challenge",
        )
        repeat = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=b"v9-test-entropy-0001"),
            fixture_kind="challenge",
        )
        changed = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=b"v9-test-entropy-0002"),
            fixture_kind="challenge",
        )

        self.assertEqual(first["example_set_sha256"], repeat["example_set_sha256"])
        self.assertNotEqual(first["example_set_sha256"], changed["example_set_sha256"])
        self.assertEqual(first["stratum_counts"], {key: 20 for key in fixtures.STRATA})

    def test_v9_fixtures_are_disjoint_from_pinned_v3_rules(self) -> None:
        v9_payload = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        v3_payload = v3_fixtures.fixture_payload(v3_fixtures.generate_visible_fixture(), fixture_kind="visible")
        fixtures.validate_partition_disjointness(v9_payload, v3_payload)
        self.assertNotEqual(v9_payload["example_set_sha256"], v3_payload["example_set_sha256"])


class ReplacementLossV9RunnerTests(unittest.TestCase):
    class WhitespaceTokenizer:
        pad_token_id = 0

        def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
            ids = [2 + sum(token.encode("utf-8")) % 60 for token in text.split()]
            return {"input_ids": ids}

    def _exercise_toy_family(self, family: str, model: torch.nn.Module) -> None:
        model.eval()
        tokenizer = self.WhitespaceTokenizer()
        visible = [
            {"example_id": f"visible-{i}", "input_text": "visible signal map", "continuation": "answer", "stratum": "visible"}
            for i in range(3)
        ]
        adaptation = [
            {
                "example_id": f"adapt-{i}",
                "input_text": " ".join(f"word{i}_{j}" for j in range(72)),
                "continuation": "answer",
                "stratum": "ordinary",
            }
            for i in range(8)
        ]
        layers, variances = runner._select_layers(model, tokenizer, family, visible)
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers, sorted(layers))
        self.assertTrue(all(np.isfinite(variances)))

        x, y, _teacher_logits = runner._capture_pairs(model, tokenizer, family, layers[0], adaptation)
        self.assertGreaterEqual(len(x), runner.MIN_ADAPTATION_ROWS)
        target = runner.v3._projection(model, family, layers[0])
        in_features, out_features, use_bias = runner.v3._module_dimensions(target)
        candidate, _fit = runner.v3.fit_independent_linear_replacement(
            in_features, out_features, x, y, use_bias=use_bias, seed=19, ridge=runner.RIDGE
        )
        runner.v3._set_projection(model, family, layers[0], candidate)

        heldout = [{
            "example_id": "heldout-0",
            "input_text": "one two three four five six seven eight",
            "continuation": "answer relation",
            "stratum": "ordinary",
        }]
        token_rows = runner._tokenized(heldout, tokenizer, continuation=True)
        group, inputs = next(runner.v3._batches(token_rows, tokenizer.pad_token_id, batch_size=1))
        delta = torch.full((1, in_features), 1e-3, dtype=torch.float32)
        predicted = runner._predict_stage(model, tokenizer, family, layers[0], group, inputs, delta)
        self.assertEqual(predicted["module_effect"].shape, (1, out_features))
        self.assertEqual(predicted["logit_effect"].shape, (1, model.config.vocab_size))

    def test_opt_relu_projection_capture_and_prediction(self) -> None:
        config = OPTConfig(
            vocab_size=64, hidden_size=32, word_embed_proj_dim=32, ffn_dim=64,
            num_hidden_layers=3, num_attention_heads=4, max_position_embeddings=128,
            activation_function="relu", pad_token_id=0, eos_token_id=1,
        )
        self._exercise_toy_family("opt_relu", OPTForCausalLM(config))

    def test_gpt2_gelu_projection_capture_and_prediction(self) -> None:
        config = GPT2Config(
            vocab_size=64, n_embd=32, n_inner=64, n_layer=3, n_head=4, n_positions=128,
            activation_function="gelu", pad_token_id=0, eos_token_id=1,
        )
        self._exercise_toy_family("gpt2_gelu", GPT2LMHeadModel(config))

    def test_llama_silu_projection_capture_and_prediction(self) -> None:
        config = LlamaConfig(
            vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
            num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
            hidden_act="silu", pad_token_id=0, eos_token_id=1, bos_token_id=1,
        )
        self._exercise_toy_family("llama_silu_gated", LlamaForCausalLM(config))

    def test_centered_linear_cka_is_one_for_identical_matrices(self) -> None:
        values = torch.tensor([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [5.0, 3.0]])
        self.assertAlmostEqual(float(runner._linear_cka(values, values)), 1.0, places=6)

    def test_cka_fit_is_deterministic_for_frozen_seed(self) -> None:
        generator = torch.Generator().manual_seed(41)
        x = torch.randn((128, 4), generator=generator)
        y = torch.randn((128, 3), generator=generator)
        first, receipt_first = runner._fit_cka(4, 3, x, y, use_bias=True, seed=90210)
        second, receipt_second = runner._fit_cka(4, 3, x, y, use_bias=True, seed=90210)

        self.assertEqual(runner._module_digest(first), runner._module_digest(second))
        self.assertEqual(receipt_first["fitted_replacement_sha256"], receipt_second["fitted_replacement_sha256"])

    def test_effect_error_uses_the_frozen_outcome_scale(self) -> None:
        predicted = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        observed = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        result = runner._summary_effect(predicted, observed, np.asarray([2.0, 4.0]))

        self.assertEqual(result["resolvable_count"], 2)
        self.assertEqual(result["direction_accuracy"], 1.0)
        self.assertEqual(result["median_cosine"], 1.0)
        self.assertAlmostEqual(result["median_normalized_error"], 0.375)

    def test_tokenization_rejects_long_prompts_and_truncates_only_continuations(self) -> None:
        class Tokenizer:
            pad_token_id = 0

            def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
                if add_special_tokens:
                    return {"input_ids": [1] * (129 if text == "long" else 120)}
                return {"input_ids": [2] * 20}

        long_record = {"input_text": "long", "continuation": "answer"}
        with self.assertRaises(ValueError):
            runner._tokenized([long_record], Tokenizer(), continuation=True)

        short_record = {"input_text": "short", "continuation": "answer"}
        tokenized = runner._tokenized([short_record], Tokenizer(), continuation=True)
        self.assertEqual(len(tokenized[0]["ids"]), 128)
        self.assertEqual(len(tokenized[0]["continuation_ids"]), 8)

    def test_wrong_module_control_exists_for_three_layer_toy_models(self) -> None:
        self.assertEqual(runner._adjacent_layer(1, [0, 1, 2], 3), 0)

    def test_candidate_seeds_bind_family_method_seed_and_stage(self) -> None:
        base = runner._candidate_seed("gpt2_gelu", 202609271, "ridge", 0)
        self.assertEqual(base, runner._candidate_seed("gpt2_gelu", 202609271, "ridge", 0))
        self.assertNotEqual(base, runner._candidate_seed("opt_relu", 202609271, "ridge", 0))
        self.assertNotEqual(base, runner._candidate_seed("gpt2_gelu", 202609271, "cka", 0))
        self.assertNotEqual(base, runner._candidate_seed("gpt2_gelu", 202609271, "ridge", 1))

    def test_prior_checkpoint_exclusions_include_all_v8_targets(self) -> None:
        self.assertIn(
            "kanishka/opt-babylm2-rewritten-clean-spacy_no-num-adj-earlystop-strictest-bpe_seed-211_1e-3",
            runner.PRIOR_CHECKPOINT_IDS,
        )
        self.assertIn("ai-forever/rugpt3small_based_on_gpt2", runner.PRIOR_CHECKPOINT_IDS)
        self.assertIn("amd/amd-llama-135m", runner.PRIOR_CHECKPOINT_IDS)

    def test_weight_digest_binds_supported_weight_file_bytes(self) -> None:
        rows = [
            {"path": "model.safetensors", "sha256": "a" * 64, "bytes": 3},
            {"path": "config.json", "sha256": "b" * 64, "bytes": 2},
        ]
        digest = runner._weights_digest(rows)
        self.assertEqual(digest, runner._canonical_digest([rows[0]]))
        with self.assertRaises(ValueError):
            runner._weights_digest([rows[1]])

    def test_bootstrap_uses_nine_trajectory_clusters(self) -> None:
        values = [0.1] * 9
        first = runner._paired_bootstrap(values, seed=77, draws=200)
        repeat = runner._paired_bootstrap(values, seed=77, draws=200)
        self.assertEqual(first, repeat)
        self.assertEqual(first["trajectory_count"], 9)
        with self.assertRaises(ValueError):
            runner._paired_bootstrap(values[:8], seed=77)

    def test_replicator_reveals_challenge_bytes_only_after_its_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            primary = Path(directory) / "primary"
            replica = Path(directory) / "replica"
            primary.mkdir()
            replica.mkdir()
            (primary / "candidate-freeze.json").write_text("primary freeze", encoding="utf-8")
            (replica / "candidate-freeze.json").write_text("replica freeze", encoding="utf-8")
            (primary / "challenges" / "toy").mkdir(parents=True)
            (primary / "challenges" / "toy" / "fixture.bin").write_bytes(b"sealed")
            (primary / "fixtures").mkdir()
            (primary / "fixtures" / "challenge-seed-0.json").write_text("{}", encoding="utf-8")
            primary_freeze_sha = replicator._sha_file(primary / "candidate-freeze.json")
            manifest = {
                "candidate_freeze_sha256": primary_freeze_sha,
                "cases": [{"fixture_sha256": "fixture", "interventions_file_sha256": "directions"}],
            }
            manifest["challenge_manifest_sha256"] = replicator._canonical_digest(manifest)
            (primary / "challenge-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with patch.object(runner, "verify_packet", return_value={"packet_sha256": "test-packet"}), patch.object(
                runner, "_verify_candidate_freeze", return_value={}
            ):
                result = replicator._reveal_challenges_after_candidate_freeze(primary, replica)

            self.assertTrue(result["revealed_after_replica_candidate_freeze"])
            replica_manifest = json.loads((replica / "challenge-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                replica_manifest["candidate_freeze_sha256"],
                replicator._sha_file(replica / "candidate-freeze.json"),
            )
            self.assertEqual((replica / "challenges" / "toy" / "fixture.bin").read_bytes(), b"sealed")


if __name__ == "__main__":
    unittest.main()
