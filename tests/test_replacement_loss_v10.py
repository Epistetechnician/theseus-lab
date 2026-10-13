"""Focused, hermetic qualification for replacement-loss-v10.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v10-opt-causal-fit-v1.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import OPTConfig, OPTForCausalLM, PreTrainedTokenizerFast

import replacement_loss_v10_fixtures as fixtures
import replacement_loss_v10_runner as runner
import replacement_loss_v3_runner as v3


class ReplacementLossV10FixtureTests(unittest.TestCase):
    def test_visible_and_adaptation_fixtures_are_deterministic_and_disjoint(self) -> None:
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptations = [
            fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation_{index}")
            for index in range(len(fixtures.SEEDS))
        ]
        repeat = fixtures.fixture_payload(fixtures.generate_adaptation_fixture(0), fixture_kind="adaptation_0")
        self.assertEqual(adaptations[0]["example_set_sha256"], repeat["example_set_sha256"])
        self.assertEqual(visible["count"], fixtures.VISIBLE_COUNT)
        self.assertEqual(adaptations[0]["stratum_counts"], {name: fixtures.ADAPTATION_PER_STRATUM for name in fixtures.STRATA})
        fixtures.validate_partition_disjointness(visible, *adaptations)

    def test_challenges_require_postfreeze_entropy_and_are_fresh(self) -> None:
        entropy_a = hashlib.sha256(b"v10 fixed challenge entropy a").digest()
        entropy_b = hashlib.sha256(b"v10 fixed challenge entropy b").digest()
        left = fixtures.fixture_payload(fixtures.generate_challenge_fixture(0, entropy=entropy_a)[0], fixture_kind="challenge")
        repeat = fixtures.fixture_payload(fixtures.generate_challenge_fixture(0, entropy=entropy_a)[0], fixture_kind="challenge")
        other = fixtures.fixture_payload(fixtures.generate_challenge_fixture(0, entropy=entropy_b)[0], fixture_kind="challenge")
        self.assertEqual(left["example_set_sha256"], repeat["example_set_sha256"])
        self.assertNotEqual(left["example_set_sha256"], other["example_set_sha256"])
        self.assertEqual(left["stratum_counts"], {name: fixtures.CHALLENGES_PER_STRATUM for name in fixtures.STRATA})
        with self.assertRaises(ValueError):
            fixtures.generate_challenge_fixture(0, entropy=b"too short")

    def test_sparse_direction_is_deterministic_and_matches_scale(self) -> None:
        activation = torch.tensor([1.0, -2.0, 0.5, 4.0])
        left = runner._direction(activation, key=b"frozen adaptation direction")
        repeat = runner._direction(activation, key=b"frozen adaptation direction")
        self.assertTrue(torch.equal(left, repeat))
        self.assertEqual(int(torch.count_nonzero(left)), 1)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(left)), 0.01 * float(activation.square().mean().sqrt()), places=7)

    def test_adversarial_challenge_uses_adaptation_selected_coordinate(self) -> None:
        delta = runner._challenge_direction(
            hashlib.sha256(b"challenge entropy").digest(),
            0,
            1,
            "heldout-0",
            7,
            2.0,
            stratum="adversarial",
            stress_coordinate=5,
        )
        self.assertEqual(torch.nonzero(delta).flatten().tolist(), [5])
        self.assertAlmostEqual(abs(float(delta[5])), 0.02, places=8)


class ReplacementLossV10RunnerTests(unittest.TestCase):
    @staticmethod
    def _toy_opt() -> OPTForCausalLM:
        config = OPTConfig(
            vocab_size=24,
            hidden_size=8,
            word_embed_proj_dim=8,
            ffn_dim=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            activation_function="relu",
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        return OPTForCausalLM(config).eval()

    def test_causal_fitter_builds_a_new_module_and_updates_it(self) -> None:
        torch.manual_seed(33)
        model = self._toy_opt()
        original = v3._projection(model, runner.FAMILY, 0)
        original_digest = runner.v3_core._module_digest(original)
        records = []
        for index in range(runner.TRAIN_BATCH_SIZE):
            ids = [1, 3 + index, 5 + index]
            inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
            delta = torch.zeros((1, original.in_features), dtype=torch.float32)
            delta[0, index % original.in_features] = 0.01
            base_logits, _x, base_y = v3._run_projection(model, runner.FAMILY, 0, inputs, positions=[2])
            changed_logits, _x2, changed_y = v3._run_projection(model, runner.FAMILY, 0, inputs, positions=[2], deltas=delta)
            assert base_logits is not None and base_y is not None and changed_logits is not None and changed_y is not None
            effect = runner._next_logits(changed_logits, [2])[0] - runner._next_logits(base_logits, [2])[0]
            records.append({
                "input_ids": ids,
                "position": 2,
                "delta": delta[0].numpy(),
                "target_y": base_y[0].numpy(),
                "teacher_effect": effect.numpy(),
                "stratum": fixtures.STRATA[index % len(fixtures.STRATA)],
                "example_id": f"toy-{index:02d}",
            })
        candidate, receipt = runner._fit_causal_candidate(
            model,
            0,
            original.in_features,
            original.out_features,
            original.bias is not None,
            records,
            0,
            seed=84,
            updates=2,
        )
        self.assertNotEqual(runner.v3_core._module_digest(candidate), original_digest)
        self.assertNotEqual(receipt["initialization_sha256"], original_digest)
        self.assertEqual(receipt["updates"], 2)
        self.assertTrue(np.isfinite(receipt["final_update_normalized_causal_mse"]))
        self.assertIs(v3._projection(model, runner.FAMILY, 0), original)

    def test_activation_rescue_restores_heldout_logits_after_replacement(self) -> None:
        model = self._toy_opt()
        model.eval()
        ids = [1, 4, 8, 9]
        inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        row = {"input_text": "unused", "continuation": "unused", "example_id": "rescue-0", "stratum": "ordinary"}
        group = [{**row, "ids": ids, "prompt_len": 3, "continuation_ids": [9]}]
        original_logits, _x, original_outputs = v3._run_projection(model, runner.FAMILY, 0, inputs)
        assert original_logits is not None and original_outputs is not None
        expected = runner._next_logits(original_logits, [2]).numpy()
        target = v3._projection(model, runner.FAMILY, 0)
        in_features, out_features, use_bias = v3._module_dimensions(target)
        candidate = runner._new_linear(in_features, out_features, use_bias, 110).eval()
        rescued_logits, rescued_ce = runner._activation_rescue_forward(model, 0, group, inputs, candidate, original_outputs)
        self.assertTrue(np.allclose(rescued_logits, expected, rtol=0.0, atol=1e-7))
        self.assertTrue(np.isfinite(rescued_ce).all())
        self.assertIs(v3._projection(model, runner.FAMILY, 0), target)

    def test_candidate_seed_and_metric_gates_are_frozen_deterministically(self) -> None:
        first = runner._candidate_seed(1, 7, "causal_fit")
        self.assertEqual(first, runner._candidate_seed(1, 7, "causal_fit"))
        self.assertNotEqual(first, runner._candidate_seed(1, 7, "activation_ridge"))
        self.assertNotEqual(first, runner._candidate_seed(2, 7, "causal_fit"))
        self.assertTrue(runner._passes_effect({"resolvable_count": 20, "direction_accuracy": 0.8, "median_cosine": 0.9, "median_normalized_error": 0.2}))
        self.assertFalse(runner._passes_effect({"resolvable_count": 19, "direction_accuracy": 1.0, "median_cosine": 1.0, "median_normalized_error": 0.0}))
        self.assertTrue(runner._passes_behavior({"continuation_cross_entropy_delta_nats_per_token": 0.05, "next_token_normalized_logit_error_max": 0.05}))

    def test_prior_checkpoints_are_excluded_from_selection(self) -> None:
        self.assertIn("facebook/opt-350m", runner.KNOWN_PRIOR_CHECKPOINT_IDS)
        self.assertIn("grushaprasad/opt125m-rootp-surprisal-curriculum", runner.KNOWN_PRIOR_CHECKPOINT_IDS)
        self.assertIn("distilbert/distilgpt2", runner.KNOWN_PRIOR_CHECKPOINT_IDS)
        self.assertIn("stentorlabs/stentor-30m", runner.KNOWN_PRIOR_CHECKPOINT_IDS)

    def test_tiny_checkpoint_runs_frozen_prepare_predict_score_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory(prefix="theseus-v10-e2e-") as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            source_repo = Path(runner.__file__).resolve().parent
            for relative in runner.PACKET_FILES:
                destination = repo / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_repo / relative, destination)
            subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "v10 test"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "v10-test@example.invalid"], check=True, capture_output=True, text=True)
            subprocess.run(["git", "-C", str(repo), "commit", "--allow-empty", "-m", "v10 test base"], check=True, capture_output=True, text=True)
            packet = runner.freeze_packet(repo)

            custody = root / "external-custody"
            packet_archive = runner.archive_packet(custody, repo_root=repo)
            checkpoint_path = custody / "checkpoint" / "files"
            checkpoint_path.mkdir(parents=True)
            torch.manual_seed(801)
            model = OPTForCausalLM(OPTConfig(
                vocab_size=4,
                hidden_size=8,
                word_embed_proj_dim=8,
                ffn_dim=16,
                num_hidden_layers=3,
                num_attention_heads=2,
                max_position_embeddings=128,
                activation_function="relu",
                bos_token_id=2,
                eos_token_id=3,
                pad_token_id=0,
                dropout=0.0,
                attention_dropout=0.0,
            )).eval()
            model.save_pretrained(checkpoint_path, safe_serialization=True)
            backend = Tokenizer(WordLevel({"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}, unk_token="<unk>"))
            backend.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                pad_token="<pad>",
                unk_token="<unk>",
                bos_token="<bos>",
                eos_token="<eos>",
            )
            tokenizer.save_pretrained(checkpoint_path)
            model_card = "---\nlicense: synthetic-test-only\npipeline_tag: text-generation\n---\nToy base OPT fixture for runner qualification.\n"
            (checkpoint_path / "README.md").write_text(model_card, encoding="utf-8")
            weight_rows = [
                {"path": item.name, "bytes": item.stat().st_size, "sha256": runner._sha_file(item)}
                for item in checkpoint_path.iterdir()
                if item.is_file() and item.suffix in {".bin", ".safetensors"}
            ]
            selected_at = datetime.now(timezone.utc).isoformat()
            selection = {
                "protocol_id": runner.PROTOCOL_ID,
                "state_slice": runner.STATE_SLICE,
                "protocol_slice": runner.PROTOCOL_SLICE,
                "packet_sha256": packet["packet_sha256"],
                "packet_archive_sha256": packet_archive["archive_sha256"],
                "selected_after_packet_freeze": True,
                "selected_at_utc": selected_at,
                "model_id": "fixture/v10-toy-opt",
                "revision": "a" * 40,
                "resolved_revision": "a" * 40,
                "freshness_screen": {
                    "prior_identity_inventory_sha256": runner.PRIOR_IDENTITY_INVENTORY_SHA256,
                    "v9_checkpoint_manifest_sha256": runner.V9_CHECKPOINT_MANIFEST_SHA256,
                    "known_prior_identity_count": len(runner.KNOWN_PRIOR_CHECKPOINT_IDS),
                    "known_prior_weight_file_digest_count": len(runner.KNOWN_PRIOR_WEIGHT_FILE_DIGESTS),
                    "identity_match_count": 0,
                    "weight_file_digest_match_count": 0,
                },
                "license_id": "synthetic-test-only",
                "license_source_url": "https://example.invalid/test-license",
                "license_checked_at_utc": selected_at,
                "model_visibility": "public_ungated",
                "model_card_url": "https://huggingface.co/fixture/v10-toy-opt/blob/" + "a" * 40 + "/README.md",
                "model_card_sha256": runner._sha_file(checkpoint_path / "README.md"),
                "model_card_metadata": {"license": "synthetic-test-only", "pipeline_tag": "text-generation"},
                "model_card_metadata_sha256": runner._canonical_digest({"license": "synthetic-test-only", "pipeline_tag": "text-generation"}),
                "config_sha256": runner._sha_file(checkpoint_path / "config.json"),
                "config_architecture": {"model_type": "opt", "activation_function": "relu", "num_hidden_layers": 3},
                "local_path": "checkpoint/files",
                "weight_files": weight_rows,
                "total_weight_bytes": sum(row["bytes"] for row in weight_rows),
                "weights_sha256": runner._canonical_digest(weight_rows),
            }
            selection["selection_sha256"] = runner._canonical_digest(selection)
            runner._write_json(custody / "checkpoint-selection.json", selection, immutable=True)

            with patch.object(runner, "_root", return_value=repo):
                result = runner.run_local(custody)

            challenge = runner._read_json(custody / "challenge-manifest.json")
            candidates = runner._read_json(custody / "candidate-freeze.json")
            predictions = runner._read_json(custody / "prediction-manifest.json")
            self.assertGreater(datetime.fromisoformat(challenge["created_at_utc"]), datetime.fromisoformat(candidates["created_at_utc"]))
            self.assertFalse(predictions["original_model_outcomes_opened"])
            self.assertEqual(len(candidates["candidates"]), len(runner.METHODS) * 3 * len(runner.DEPTH_BANDS))
            self.assertIn(result["score"]["scientific_outcome"], {"PASS", "FAIL"})
            self.assertEqual(runner._read_json(custody / "outcome-report.json")["reviewer_status"], "NOT_SUBMITTED")


if __name__ == "__main__":
    unittest.main()
