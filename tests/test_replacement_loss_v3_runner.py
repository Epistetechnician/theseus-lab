"""Hermetic tests for replacement-loss-v3 capture and custody mechanics.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM, OPTConfig, OPTForCausalLM
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from replacement_loss_v3_fixtures import (
    STRATA,
    fixture_payload,
    generate_adaptation_fixture,
    generate_challenge_fixture,
    generate_visible_fixture,
    validate_partition_disjointness,
)
from replacement_loss_v3_runner import (
    Checkpoint,
    _capture_visible_layer,
    _case_evaluation,
    _config_blockers,
    _direction,
    _fit_case,
    _make_variant_modules,
    _module_dimensions,
    _projection,
    _run_projection,
    _token_rows,
    _batches,
    _sha256,
    _write_json,
)
import replacement_loss_v3_runner as runner


class _FixtureTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [3 + ord(character) % 61 for character in text[:12]]
        return {"input_ids": ([1] + ids if add_special_tokens else ids)}


def _tiny_models() -> list[tuple[str, torch.nn.Module]]:
    return [
        (
            "opt_relu",
            OPTForCausalLM(
                OPTConfig(
                    vocab_size=64,
                    hidden_size=16,
                    ffn_dim=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    word_embed_proj_dim=16,
                    max_position_embeddings=64,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
        ),
        (
            "gpt2_gelu",
            GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=64,
                    n_embd=16,
                    n_layer=2,
                    n_head=4,
                    n_positions=64,
                    n_ctx=64,
                    bos_token_id=1,
                    eos_token_id=2,
                )
            ),
        ),
        (
            "llama_silu_gated",
            LlamaForCausalLM(
                LlamaConfig(
                    vocab_size=64,
                    hidden_size=16,
                    intermediate_size=32,
                    num_hidden_layers=2,
                    num_attention_heads=4,
                    num_key_value_heads=4,
                    max_position_embeddings=64,
                    bos_token_id=1,
                    eos_token_id=2,
                    pad_token_id=0,
                )
            ),
        ),
    ]


class TestReplacementLossV3Fixtures(unittest.TestCase):
    def test_visible_adaptation_and_heldout_partitions_are_disjoint(self) -> None:
        fixtures = [fixture_payload(generate_visible_fixture(), fixture_kind="visible")]
        fixtures.extend(
            fixture_payload(generate_adaptation_fixture(index), fixture_kind=f"adaptation_{index}")
            for index in range(3)
        )
        fixtures.append(
            fixture_payload(generate_challenge_fixture(0, secret_seed=b"frozen challenge seed" * 2), fixture_kind="challenge")
        )
        validate_partition_disjointness(*fixtures)
        challenge_counts = fixtures[-1]["stratum_counts"]
        self.assertEqual(challenge_counts, {name: 20 for name in STRATA})

    def test_intervention_norm_and_adversarial_sparsity_are_locked(self) -> None:
        activation = np.asarray([1.0, -2.0, 0.5, 4.0], dtype=np.float32)
        target = fixture_payload(
            generate_challenge_fixture(1, secret_seed=b"a" * 32), fixture_kind="challenge"
        )["records"]
        adversarial = next(row for row in target if row["stratum"] == "adversarial")
        delta = _direction(adversarial, activation, b"b" * 32)
        expected_norm = 0.01 * float(np.sqrt(np.mean(np.square(activation, dtype=np.float64))))
        self.assertAlmostEqual(float(np.linalg.norm(delta)), expected_norm, places=7)
        self.assertEqual(int(np.count_nonzero(delta)), 1)


class TestReplacementLossV3Runner(unittest.TestCase):
    def test_checkpoint_manifest_requires_postfreeze_timezone_aware_selection(self) -> None:
        freeze = {
            "packet_sha256": "d" * 64,
            "created_at_utc": "2026-09-24T11:00:00+00:00",
        }
        payload = {
            "protocol_id": "theseus-replacement-loss-v3",
            "state_slice": "theseus-lab-sandbox-v1",
            "packet_sha256": freeze["packet_sha256"],
            "selected_at_utc": "2026-09-24T11:01:00+00:00",
            "checkpoints": [
                {
                    "family": family,
                    "model_id": f"fixture/{family}",
                    "revision": "a" * 40,
                    "local_path": f"checkpoints/{family}",
                    "selected_after_protocol_freeze": True,
                    "not_used_for_protocol_design": True,
                    "base_model": True,
                }
                for family in runner.FAMILIES
            ],
        }
        with tempfile.TemporaryDirectory(prefix="theseus-v3-manifest-") as temp_dir:
            root = Path(temp_dir)
            manifest = root / "checkpoints.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with mock.patch.object(runner, "verify_packet", return_value=freeze):
                self.assertEqual(len(runner._load_checkpoints(manifest, root)), 3)
                payload["selected_at_utc"] = "2026-09-24T11:01:00"
                manifest.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(ValueError):
                    runner._load_checkpoints(manifest, root)

    def test_all_family_projection_adapters_and_pre_outcome_capture(self) -> None:
        tokenizer = _FixtureTokenizer()
        for family, model in _tiny_models():
            with self.subTest(family=family):
                self.assertEqual(_config_blockers(model.config, family), [])
                self.assertIsNotNone(_projection(model, family, 0))
                records = [
                    {"input_text": "A short visible prompt", "continuation": " next", "example_id": "x", "stratum": "ordinary"}
                ]
                rows = _token_rows(tokenizer, records, with_continuation=True)
                positions = [rows[0]["prompt_len"] - 1]
                inputs = {
                    "input_ids": torch.tensor([rows[0]["ids"]], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(rows[0]["ids"])), dtype=torch.long),
                }
                logits, captured, outputs = _run_projection(
                    model,
                    family,
                    0,
                    inputs,
                    positions=positions,
                    abort_after_input=True,
                )
                self.assertIsNone(logits)
                self.assertIsNone(outputs)
                self.assertEqual(tuple(captured.shape), (1, _module_dimensions(_projection(model, family, 0))[0]))

    def test_visible_selection_and_adaptation_fit_write_only_external_artifacts(self) -> None:
        tokenizer = _FixtureTokenizer()
        visible = [row.__dict__ for row in generate_visible_fixture()]
        adaptation = [row.__dict__ for row in generate_adaptation_fixture(0)]
        with tempfile.TemporaryDirectory(prefix="theseus-v3-test-") as temp_dir:
            root = Path(temp_dir).resolve()
            for family, model in _tiny_models():
                with self.subTest(family=family):
                    model.eval()
                    selected, variances = _capture_visible_layer(model, tokenizer, family, visible)
                    self.assertIn(selected, (0, 1))
                    self.assertEqual(len(variances), 2)
                    checkpoint = Checkpoint(family, f"toy-{family}", "a" * 40, Path("/external/toy"), True)
                    case = _fit_case(model, tokenizer, checkpoint, selected, 0, adaptation, root)
                    self.assertGreaterEqual(case["adaptation_row_count"], 512)
                    self.assertLess(case["mapping_aligned_equivalence_max_abs_error"], 1e-6)
                    for item in case["modules"]:
                        self.assertTrue((root / item["file"]).is_file())

    def test_evaluator_seals_treatment_predictions_before_original_outcomes(self) -> None:
        family = "opt_relu"
        model = _tiny_models()[0][1].eval()
        tokenizer = _FixtureTokenizer()
        checkpoint = Checkpoint(family, "toy-opt", "a" * 40, Path("/external/toy"), True)
        adaptation = [row.__dict__ for row in generate_adaptation_fixture(0)]
        challenge = fixture_payload(
            generate_challenge_fixture(0, secret_seed=b"c" * 32), fixture_kind="challenge"
        )
        with tempfile.TemporaryDirectory(prefix="theseus-v3-eval-test-") as temp_dir:
            root = Path(temp_dir).resolve()
            case = _fit_case(model, tokenizer, checkpoint, 0, 0, adaptation, root)
            case["checkpoint_tree_sha256"] = "f" * 64
            challenge_dir = root / "challenges" / family / "seed-0"
            prompt_file = challenge_dir / "prompts.json"
            intervention_file = challenge_dir / "interventions.npz"
            _write_json(prompt_file, challenge)
            encoded = _token_rows(tokenizer, challenge["records"], with_continuation=True)
            activations = []
            for group, inputs in _batches(encoded, tokenizer.pad_token_id):
                positions = [row["prompt_len"] - 1 for row in group]
                _logits, values, _outputs = _run_projection(
                    model, family, 0, inputs, positions=positions, abort_after_input=True
                )
                activations.extend(values.numpy())
            rng = np.random.default_rng(109)
            deltas = []
            for activation in activations:
                direction = rng.normal(size=activation.shape[0])
                direction /= np.linalg.norm(direction)
                scale = 0.01 * np.sqrt(np.mean(np.square(activation, dtype=np.float64)))
                deltas.append((direction * scale).astype(np.float32))
            np.savez_compressed(intervention_file, deltas=np.stack(deltas))
            challenge_manifest_row = {
                "fixture_sha256": challenge["example_set_sha256"],
                "prompt_file_sha256": _sha256(prompt_file.read_bytes()),
                "interventions_file_sha256": _sha256(intervention_file.read_bytes()),
            }
            _write_json(root / "candidate-freeze.json", {"candidate_files_sha256": "e" * 64})
            with mock.patch(
                "replacement_loss_v3_runner.verify_packet",
                return_value={"packet_sha256": "d" * 64},
            ):
                report = _case_evaluation(
                    model, tokenizer, checkpoint, case, root, challenge_manifest_row
                )
            self.assertTrue(report["prediction_outcome_order_verified"])
            self.assertEqual(report["challenge_counts"], {name: 20 for name in STRATA})
            receipt = root / "predictions" / family / "seed-0.receipt.json"
            self.assertFalse(json.loads(receipt.read_text())["original_outcomes_opened"])
            self.assertTrue((root / "outcomes" / family / "seed-0.npz").is_file())

    def test_stage_pipeline_runs_all_nine_toy_cases_with_external_custody(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theseus-v3-pipeline-") as temp_dir:
            root = Path(temp_dir).resolve()
            tokenizer_dir = root / "tokenizer"
            tokenizer_dir.mkdir()
            vocabulary = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            vocabulary.update({f"token_{index}": index + 4 for index in range(60)})
            backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            backend.post_processor = TemplateProcessing(
                single="<bos> $A",
                pair="<bos> $A $B",
                special_tokens=[("<bos>", 1)],
            )
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token="<pad>",
                bos_token="<bos>",
                eos_token="<eos>",
                unk_token="<unk>",
                model_max_length=64,
            )
            tokenizer.save_pretrained(tokenizer_dir)
            checkpoints = []
            model_specs = (
                ("opt_relu", OPTForCausalLM(OPTConfig(vocab_size=64, hidden_size=16, ffn_dim=32, num_hidden_layers=2, num_attention_heads=4, word_embed_proj_dim=16, max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))),
                ("gpt2_gelu", GPT2LMHeadModel(GPT2Config(vocab_size=64, n_embd=16, n_layer=2, n_head=4, n_positions=64, n_ctx=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))),
                ("llama_silu_gated", LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64, bos_token_id=1, eos_token_id=2, pad_token_id=0))),
            )
            for family, model in model_specs:
                checkpoint_path = root / "checkpoints" / family
                checkpoint_path.mkdir(parents=True)
                model.save_pretrained(checkpoint_path, safe_serialization=True)
                for asset in tokenizer_dir.iterdir():
                    (checkpoint_path / asset.name).write_bytes(asset.read_bytes())
                checkpoints.append(
                    Checkpoint(family, f"toy-{family}", "a" * 40, checkpoint_path, True)
                )
            manifest_path = root / "checkpoint-manifest.json"
            manifest_path.write_text(json.dumps({"protocol_id": "toy"}))
            fake_freeze = {"packet_sha256": "d" * 64}
            with (
                mock.patch.object(runner, "verify_packet", return_value=fake_freeze),
                mock.patch.object(runner, "_load_checkpoints", return_value=checkpoints),
            ):
                prepared = runner.prepare_candidates(root, manifest_path)
                self.assertEqual(prepared["case_count"], 9)
                runner.freeze_candidates(root)
                challenge_summary = runner.make_challenges(root, manifest_path)
                self.assertEqual(challenge_summary["case_count"], 9)
                report_summary = runner.evaluate_packet(root, manifest_path)
                self.assertEqual(report_summary["case_count"], 9)
                self.assertEqual(report_summary["packet_sha256"], fake_freeze["packet_sha256"])
                report = json.loads((root / "replacement-loss-v3-report.json").read_text())
                self.assertEqual(len(report["cases"]), 9)
                self.assertIn(report["assurance_level"], ("INTERNAL_PASS", "INCONCLUSIVE"))
                for case in report["cases"]:
                    self.assertLessEqual(
                        case["mapping_aligned_permutation_diagnostic"]["module_effect_max_abs_error_vs_treatment"],
                        runner.EQUIVALENCE_MAX_ABS_ERROR,
                    )
                    self.assertLessEqual(
                        case["mapping_aligned_permutation_diagnostic"]["logit_effect_max_abs_error_vs_treatment"],
                        runner.EQUIVALENCE_MAX_ABS_ERROR,
                    )
                with tempfile.TemporaryDirectory(prefix="theseus-v3-replica-") as replica_dir:
                    replication_events = []
                    original_copy = runner._copy_custody_file

                    def record_replication_copy(source, destination, relative):
                        if Path(relative).parts[0] == "challenges":
                            replication_events.append(fit_spy.call_count)
                        return original_copy(source, destination, relative)

                    with (
                        mock.patch.object(runner, "_fit_case", wraps=runner._fit_case) as fit_spy,
                        mock.patch.object(
                            runner, "_copy_custody_file", side_effect=record_replication_copy
                        ),
                    ):
                        replica_summary = runner.replicate_packet(root, replica_dir, manifest_path)
                    self.assertTrue(replica_summary["candidate_reconstruction_match"])
                    self.assertTrue(replica_summary["prediction_and_outcome_array_digests_match"])
                    self.assertFalse(replica_summary["distinct_clean_runtime"])
                    self.assertEqual(replica_summary["assurance_level"], "INCONCLUSIVE")
                    self.assertEqual(fit_spy.call_count, 9)
                    self.assertEqual(len(replication_events), 18)
                    self.assertTrue(all(count == 9 for count in replication_events))

    def test_aligned_permutation_is_function_equivalent(self) -> None:
        module = torch.nn.Linear(11, 5, bias=True)
        _raw, _permutation, error = _make_variant_modules(module, 77)
        self.assertLessEqual(error, 1e-6)


if __name__ == "__main__":
    unittest.main()
