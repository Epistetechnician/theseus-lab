"""Focused tests for replacement-loss-v6 fitting and custody rules.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

import replacement_loss_v3_fixtures as v3_fixtures
import replacement_loss_v3_runner as v3
import replacement_loss_v6 as v6
import replacement_loss_v6_fixtures as fixtures
import replacement_loss_v6_runner as runner


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        ids = [1 + (sum(word.encode("utf-8")) % 30) for word in text.split()]
        return {"input_ids": ([31] + ids) if add_special_tokens else ids}


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.projection = nn.Linear(4, 4)
        self.output = nn.Linear(4, 32)

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        hidden = torch.tanh(self.projection(hidden))
        return SimpleNamespace(logits=self.output(hidden))


class TestReplacementLossV6(unittest.TestCase):
    def test_fixture_rules_are_balanced_disjoint_and_challenge_entropy_is_replayable(self) -> None:
        visible = fixtures.fixture_payload(
            fixtures.generate_visible_fixture(), fixture_kind="visible_selection"
        )
        adaptations = [
            fixtures.fixture_payload(
                fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation-{index}"
            )
            for index in range(3)
        ]
        for payload in adaptations:
            fixtures.assert_stratum_counts(payload, fixtures.ADAPTATION_PROMPTS)
        entropy = b"v6 deterministic test entropy".ljust(32, b"!")
        first = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=entropy), fixture_kind="challenge"
        )
        second = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=entropy), fixture_kind="challenge"
        )
        self.assertEqual(first["example_set_sha256"], second["example_set_sha256"])
        self.assertEqual(first["count"], 60)
        fixtures.validate_partition_disjointness(visible, *adaptations, first)

    def test_prior_fixture_overlap_uses_union_without_rejecting_historical_arm_overlap(self) -> None:
        old_records = [
            v3_fixtures.generate_adaptation_fixture(0)[index]
            for index in range(3)
        ]
        first_old = v3_fixtures.fixture_payload(old_records[:2], fixture_kind="old-arm-a")
        second_old = v3_fixtures.fixture_payload(old_records[1:], fixture_kind="old-arm-b")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arm-a.json").write_text(json.dumps(first_old), encoding="utf-8")
            (root / "arm-b.json").write_text(json.dumps(second_old), encoding="utf-8")
            current = fixtures.fixture_payload(
                fixtures.generate_visible_fixture(), fixture_kind="current-visible"
            )
            receipt = runner._validate_all_disjoint([root], [current])
            self.assertTrue(receipt["all_partitions_disjoint"])
            self.assertEqual(receipt["prior_fixture_count"], 2)

            overlapping = fixtures.fixture_payload(
                [old_records[0]], fixture_kind="overlap"
            )
            with self.assertRaisesRegex(ValueError, "overlap prior"):
                runner._validate_all_disjoint([root], [overlapping])

    def test_behavior_aware_fitter_is_repeatable_and_restores_base_model_state(self) -> None:
        model = _TinyCausalModel()
        model.train()
        tokenizer = _TinyTokenizer()
        records = [
            {"example_id": f"row-{index}", "input_text": f"alpha token{index}", "continuation": " next"}
            for index in range(6)
        ]
        original_projection = v3._projection

        def projection(target: nn.Module, family: str, layer: int) -> nn.Module:
            if target is model and family == "opt_relu" and layer == 0:
                return model.projection
            return original_projection(target, family, layer)

        requires_grad_before = [parameter.requires_grad for parameter in model.parameters()]
        with patch.object(v3, "_projection", side_effect=projection):
            first, first_receipt = v6.train_behavior_aware_replacement(
                model, tokenizer, "opt_relu", 0, records, seed=77, epochs=2
            )
            second, second_receipt = v6.train_behavior_aware_replacement(
                model, tokenizer, "opt_relu", 0, records, seed=77, epochs=2
            )
        self.assertEqual(first_receipt["candidate_state_sha256"], second_receipt["candidate_state_sha256"])
        self.assertEqual(first_receipt["optimizer_update_count"], 4)
        self.assertTrue(torch.isfinite(torch.tensor(first_receipt["mean_training_loss"])))
        self.assertEqual([parameter.requires_grad for parameter in model.parameters()], requires_grad_before)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(first.weight, second.weight))

    def test_v6_fit_maps_evaluator_control_to_the_output_only_ridge_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            custody = Path(temporary)
            dimensions = {"input_features": 4, "output_features": 3, "use_bias": True}
            modules = []
            for name in ("treatment", "wrong_module", "raw_index_permutation", "mapping_aligned_permutation"):
                path = Path("candidates") / "opt_relu" / "seed-0" / f"{name}.safetensors"
                digest = v3._save_linear(custody / path, nn.Linear(4, 3, bias=True))
                modules.append({"name": name, "file": path.as_posix(), "state_sha256": digest})
            original_wrong = custody / modules[1]["file"]

            def fit_stub(*_args: object, **_kwargs: object) -> dict[str, object]:
                return {
                    **dimensions,
                    "modules": modules.copy(),
                    "fit_elapsed_wall_seconds": 0.1,
                    "treatment_fit": {"method": "output_only_ridge"},
                    "wrong_module_fit": {"method": "historical_wrong_module"},
                }

            candidate = nn.Linear(4, 3, bias=True)
            receipt = {"candidate_state_sha256": v6._digest_state(candidate)}
            with (
                patch.object(runner, "_ORIGINAL_FIT_CASE", side_effect=fit_stub),
                patch.object(v6, "train_behavior_aware_replacement", return_value=(candidate, receipt)),
            ):
                result = runner._fit_case_v6(
                    nn.Identity(),
                    _TinyTokenizer(),
                    SimpleNamespace(family="opt_relu"),
                    0,
                    0,
                    [],
                    custody,
                )
            rows = {row["name"]: row for row in result["modules"]}
            self.assertEqual(result["v6_control_aliases"]["wrong_module"], "output_only_ridge")
            self.assertEqual(rows["wrong_module"]["file"], "candidates/opt_relu/seed-0/output-only-ridge.safetensors")
            self.assertTrue((custody / rows["wrong_module"]["file"]).is_file())
            self.assertFalse(original_wrong.exists())
            self.assertEqual(rows["treatment"]["state_sha256"], receipt["candidate_state_sha256"])

    def test_packet_freeze_archives_exact_bytes_and_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source_root = Path(runner.__file__).resolve().parent
            packet_root = temporary_root / "packet-source"
            custody_root = temporary_root / "custody"
            packet_root.mkdir()
            all_paths = set(runner.V6_PACKET_FILES) | set(runner.V3_PACKET_FILES)
            for relative in all_paths:
                destination = packet_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, destination)
            with (
                patch.object(runner, "_root", return_value=packet_root),
                patch.object(v3, "_repo_root", return_value=packet_root),
            ):
                result = runner.freeze_packet(custody_root)
                self.assertTrue(runner.verify_packet(custody_root)["verified"])
                self.assertEqual(result["packet_file_count"], len(all_paths))
                with self.assertRaises(FileExistsError):
                    runner.freeze_packet(custody_root)

                source_file = packet_root / "replacement_loss_v6.py"
                source_file.write_text(source_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "frozen v6 packet bytes changed"):
                    runner.verify_packet(custody_root)


if __name__ == "__main__":
    unittest.main()
