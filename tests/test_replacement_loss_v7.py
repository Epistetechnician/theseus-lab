"""Regression and packet tests for replacement-loss-v7.

State slice: theseus-lab-sandbox-v1. Tests use tiny synthetic models only.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import replacement_loss_v3_fixtures as v3_fixtures
import replacement_loss_v3_runner as v3
import replacement_loss_v6_fixtures as v6_fixtures
import replacement_loss_v7 as v7
import replacement_loss_v7_fixtures as fixtures
import replacement_loss_v7_runner as runner


class _TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        ids = [1 + (sum(word.encode("utf-8")) % 30) for word in text.split()]
        return {"input_ids": ([31] + ids) if add_special_tokens else ids}


class _FlattenedLinear(nn.Linear):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs.reshape(-1, inputs.shape[-1]), self.weight, self.bias)


class _TinyCausalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.projection = _FlattenedLinear(4, 4)
        self.output = nn.Linear(4, 32)

    def forward(self, input_ids: torch.Tensor, **_kwargs: object) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        projected = torch.tanh(self.projection(hidden))
        projected = projected.reshape_as(hidden)
        return SimpleNamespace(logits=self.output(projected))


class TestReplacementLossV7(unittest.TestCase):
    def test_fresh_namespaced_fixtures_are_balanced_and_disjoint_from_v3_v6(self) -> None:
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptations = [
            fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adapt-{index}")
            for index in range(3)
        ]
        for payload in adaptations:
            fixtures.assert_stratum_counts(payload, fixtures.ADAPTATION_PROMPTS)
        challenge = fixtures.fixture_payload(
            fixtures.generate_challenge_fixture(0, secret_seed=b"v7 test entropy".ljust(32, b"!")),
            fixture_kind="challenge",
        )
        fixtures.validate_partition_disjointness(visible, *adaptations, challenge)
        old_v3 = v3_fixtures.fixture_payload(v3_fixtures.generate_adaptation_fixture(0), fixture_kind="old-v3")
        old_v6 = v6_fixtures.fixture_payload(v6_fixtures.generate_adaptation_fixture(0), fixture_kind="old-v6")
        old_ids = {row["example_id"] for payload in (old_v3, old_v6) for row in payload["records"]}
        old_prompts = {row["input_text"] for payload in (old_v3, old_v6) for row in payload["records"]}
        for payload in (visible, *adaptations, challenge):
            self.assertFalse(old_ids.intersection(row["example_id"] for row in payload["records"]))
            self.assertFalse(old_prompts.intersection(row["input_text"] for row in payload["records"]))
        self.assertEqual(challenge["count"], 60)

    def test_flattened_hook_output_is_normalized_before_activation_loss(self) -> None:
        model = _TinyCausalModel()
        model.train()
        tokenizer = _TinyTokenizer()
        records = [
            {"example_id": f"row-{index}", "input_text": f"alpha token{index}", "continuation": " next"}
            for index in range(6)
        ]
        original_projection = v3._projection

        def projection(target: nn.Module, family: str, layer: int) -> nn.Module:
            if target is model and family == "gpt2_gelu" and layer == 0:
                return model.projection
            return original_projection(target, family, layer)

        with patch.object(v3, "_projection", side_effect=projection):
            first, first_receipt = v7.train_behavior_aware_replacement(
                model, tokenizer, "gpt2_gelu", 0, records, seed=77, epochs=2
            )
            second, second_receipt = v7.train_behavior_aware_replacement(
                model, tokenizer, "gpt2_gelu", 0, records, seed=77, epochs=2
            )
        self.assertEqual(first_receipt["candidate_state_sha256"], second_receipt["candidate_state_sha256"])
        self.assertEqual(first_receipt["optimizer_update_count"], 4)
        self.assertEqual(first_receipt["weight_decay"], 0.0)
        self.assertEqual(
            first_receipt["projection_output_normalization"],
            "v3_activation_bth_for_flat_or_batched_hook_outputs",
        )
        self.assertTrue(torch.isfinite(torch.tensor(first_receipt["mean_training_loss"])))
        self.assertTrue(torch.equal(first.weight, second.weight))
        self.assertTrue(model.training)

    def test_freeze_archives_exact_v7_packet_and_rejects_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            packet_root = temporary_root / "packet-source"
            custody_root = temporary_root / "custody"
            packet_root.mkdir()
            source_root = Path(runner.__file__).resolve().parent
            for relative in set(runner.V7_PACKET_FILES):
                target = packet_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, target)
            fake_v6 = {
                "packet_sha256": runner.V6_PACKET_SHA256,
                "packet_files": [],
            }
            fake_v3 = {"packet_sha256": runner.V3_PACKET_SHA256, "packet_files": []}
            with (
                patch.object(runner, "_root", return_value=packet_root),
                patch.object(runner, "_verify_pinned_v6_source_and_archive", return_value=(fake_v6, [])),
                patch.object(runner, "_verify_pinned_v3_source", return_value=(fake_v3, [])),
            ):
                frozen = runner.freeze_packet(custody_root)
                with (
                    patch.object(runner, "_verify_pinned_v6_source_and_archive", return_value=(fake_v6, [])),
                    patch.object(runner, "_verify_pinned_v3_source", return_value=(fake_v3, [])),
                ):
                    verified = runner.verify_packet(custody_root)
                    self.assertTrue(verified["verified"])
                    self.assertEqual(frozen["packet_sha256"], verified["packet_sha256"])
                    with self.assertRaises(FileExistsError):
                        runner.freeze_packet(custody_root)
                    changed = packet_root / "replacement_loss_v7.py"
                    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, "frozen v7 source changed"):
                        runner.verify_packet(custody_root)


if __name__ == "__main__":
    unittest.main()
