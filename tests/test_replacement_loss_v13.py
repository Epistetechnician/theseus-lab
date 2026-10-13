"""Focused qualification for the frozen V13 candidate and identity rules."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

import replacement_loss_v13_fixtures as fixtures
import replacement_loss_v13_runner as runner


class V13QualificationTests(unittest.TestCase):
    def test_nonlinear_replacement_is_low_parameter_and_shape_preserving(self):
        candidate = runner._new_nonlinear(3072, 768, seed=1234)
        source_parameter_count = 3072 * 768 + 768
        candidate_parameter_count = sum(parameter.numel() for parameter in candidate.parameters())
        output = candidate(torch.zeros(2, 3072, dtype=torch.float32))

        self.assertEqual(tuple(output.shape), (2, 768))
        self.assertEqual(candidate_parameter_count, 1_967_360)
        self.assertLess(candidate_parameter_count, source_parameter_count)
        self.assertEqual(candidate.in_features, 3072)
        self.assertEqual(candidate.out_features, 768)
        self.assertIsNotNone(candidate.bias)

    def test_nonlinear_initialization_is_seed_deterministic(self):
        first = runner._new_nonlinear(3072, 768, seed=202609401)
        second = runner._new_nonlinear(3072, 768, seed=202609401)

        self.assertEqual(
            runner.v3_core._module_digest(first),
            runner.v3_core._module_digest(second),
        )

    def test_nonlinear_candidate_round_trips_through_safetensors(self):
        candidate = runner._new_nonlinear(3072, 768, seed=17)
        previous_custody = runner.engine._CUSTODY
        with TemporaryDirectory(prefix="theseus-v13-candidate-") as directory:
            custody = Path(directory)
            runner.engine._CUSTODY = custody
            try:
                saved = runner._save_candidate(custody / "candidate.safetensors", candidate)
                restored = runner._load_candidate(custody, saved)
            finally:
                runner.engine._CUSTODY = previous_custody

        self.assertEqual(
            runner.v3_core._module_digest(candidate),
            runner.v3_core._module_digest(restored),
        )

    def test_causal_fit_updates_only_the_fresh_replacement(self):
        class ToyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc2 = torch.nn.Linear(3072, 768)

        class ToyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([ToyLayer()])

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.decoder = ToyDecoder()
                self.head = torch.nn.Linear(768, 13)

            def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
                del attention_mask, use_cache, return_dict
                values = input_ids.float().unsqueeze(-1).expand(-1, -1, 3072) / 10.0
                projected = self.model.decoder.layers[0].fc2(values)
                return SimpleNamespace(logits=self.head(projected))

        rng = np.random.default_rng(912)
        rows = []
        for index in range(8):
            delta = np.zeros(3072, dtype=np.float32)
            delta[index] = 0.01
            rows.append({
                "input_ids": [index + 1, index + 2, index + 3],
                "position": 2,
                "delta": delta,
                "target_y": rng.normal(size=768).astype(np.float32),
                "teacher_effect": rng.normal(size=13).astype(np.float32),
            })
        model = ToyModel()
        source_digest = runner.v3_core._module_digest(model.model.decoder.layers[0].fc2)
        candidate, receipt = runner._fit_nonlinear_candidate(
            model,
            0,
            3072,
            768,
            True,
            rows,
            0,
            seed=64,
            updates=2,
        )

        self.assertEqual(receipt["updates"], 2)
        self.assertEqual(receipt["parameter_count"], 1_967_360)
        self.assertNotEqual(receipt["initialization_sha256"], receipt["fitted_candidate_sha256"])
        self.assertEqual(runner.v3_core._module_digest(model.model.decoder.layers[0].fc2), source_digest)
        self.assertEqual(candidate.training, False)

    def test_fixture_seed_contract_is_fixed_and_unique(self):
        self.assertEqual(len(fixtures.SEEDS), 6)
        self.assertEqual(len(set(fixtures.SEEDS)), 6)
        self.assertEqual(fixtures.SEEDS, tuple(range(202609401, 202609407)))
        self.assertEqual(fixtures.STRATA, ("ordinary", "compositional", "adversarial"))

    def test_stable_candidate_identity_excludes_runtime_location_fields(self):
        selection = {
            "model_id": "example/known-opt",
            "revision": "a" * 40,
            "source_artifact_tree_sha256": "tree",
            "weights_sha256": "weights",
            "selection_sha256": "selection",
        }
        candidate_row = {
            "seed_index": 0,
            "seed": fixtures.SEEDS[0],
            "layer": 0,
            "method": "causal_fit",
            "file_sha256": "file",
            "state_sha256": "state",
            "architecture": "Linear-GELU-Linear",
            "fit_receipt": {"updates": runner.CAUSAL_UPDATES, "seed": 12},
            "adaptation_sha256": "adaptation",
        }
        common = {
            "packet_sha256": "packet",
            "visible_fixture_sha256": "visible",
            "selected_layers": [0, 7],
            "adaptation_fixture_sha256": ["adapt-0"],
            "candidates": [candidate_row],
        }
        primary = {**common, "candidate_set_sha256": "path-bound-a", "runtime": {"python_executable": "/one/python"}}
        replica = {**common, "candidate_set_sha256": "path-bound-b", "runtime": {"python_executable": "/two/python"}}

        self.assertEqual(
            runner.stable_candidate_identity(primary, selection),
            runner.stable_candidate_identity(replica, selection),
        )


if __name__ == "__main__":
    unittest.main()
