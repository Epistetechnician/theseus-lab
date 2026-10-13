"""Focused qualification for the V14 paired activation ablation."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

import replacement_loss_v14_fixtures as fixtures
import replacement_loss_v14_runner as runner


class V14QualificationTests(unittest.TestCase):
    def test_pair_has_same_shape_parameter_count_and_boundary(self):
        affine = runner._new_stack(3072, 768, seed=71, gelu=False)
        gelu = runner._new_stack(3072, 768, seed=71, gelu=True)

        self.assertEqual(affine.in_features, gelu.in_features)
        self.assertEqual(affine.out_features, gelu.out_features)
        self.assertEqual(affine.architecture, "Linear-Identity-Linear")
        self.assertEqual(gelu.architecture, "Linear-GELU-Linear")
        self.assertEqual(sum(item.numel() for item in affine.parameters()), 1_967_360)
        self.assertEqual(sum(item.numel() for item in gelu.parameters()), 1_967_360)
        self.assertEqual(tuple(affine(torch.zeros(2, 3072)).shape), (2, 768))
        self.assertEqual(tuple(gelu(torch.zeros(2, 3072)).shape), (2, 768))

    def test_initial_parameter_tensors_are_identical(self):
        affine = runner._new_stack(3072, 768, seed=912, gelu=False)
        gelu = runner._new_stack(3072, 768, seed=912, gelu=True)

        self.assertEqual(
            runner.v3_core._module_digest(affine),
            runner.v3_core._module_digest(gelu),
        )
        for name, tensor in affine.state_dict().items():
            self.assertTrue(torch.equal(tensor, gelu.state_dict()[name]), name)

    def test_paired_training_uses_same_batches_but_only_gelu_changes_map(self):
        class ToyLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc2 = torch.nn.Linear(3072, 768)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.decoder = torch.nn.Module()
                self.model.decoder.layers = torch.nn.ModuleList([ToyLayer()])
                self.head = torch.nn.Linear(768, 13)

            def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
                del attention_mask, use_cache, return_dict
                hidden = input_ids.float().unsqueeze(-1).expand(-1, -1, 3072) / 10.0
                return SimpleNamespace(logits=self.head(self.model.decoder.layers[0].fc2(hidden)))

        rng = np.random.default_rng(1414)
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
                "stratum": "ordinary",
                "example_id": f"row-{index}",
            })

        torch.set_num_threads(1)
        affine, affine_receipt = runner._train_stack_candidate(
            ToyModel(), 0, 3072, 768, True, rows, 0, seed=77, gelu=False, updates=2,
        )
        gelu, gelu_receipt = runner._train_stack_candidate(
            ToyModel(), 0, 3072, 768, True, rows, 0, seed=77, gelu=True, updates=2,
        )

        self.assertEqual(affine_receipt["initialization_sha256"], gelu_receipt["initialization_sha256"])
        self.assertEqual(affine_receipt["batch_order_sha256"], gelu_receipt["batch_order_sha256"])
        self.assertEqual(affine_receipt["updates"], 2)
        self.assertEqual(gelu_receipt["updates"], 2)
        self.assertNotEqual(affine_receipt["fitted_candidate_sha256"], gelu_receipt["fitted_candidate_sha256"])

    def test_paired_candidate_round_trip_preserves_architecture_and_state(self):
        module = runner._new_stack(3072, 768, seed=18, gelu=True)
        engine = runner.configure_engine()
        previous_custody = engine._CUSTODY
        with TemporaryDirectory(prefix="theseus-v14-candidate-") as directory:
            custody = Path(directory)
            engine._CUSTODY = custody
            try:
                row = runner._save_candidate(custody / "candidate.safetensors", module)
                restored = runner._load_candidate(custody, row)
            finally:
                engine._CUSTODY = previous_custody

        self.assertEqual(row["architecture"], "Linear-GELU-Linear")
        self.assertEqual(
            runner.v3_core._module_digest(module),
            runner.v3_core._module_digest(restored),
        )

    def test_seed_mapping_pairs_both_methods_and_separates_cells(self):
        left = runner._paired_candidate_seed(0, 0, "activation_ridge")
        right = runner._paired_candidate_seed(0, 0, "causal_fit")

        self.assertEqual(left, right)
        self.assertNotEqual(left, runner._paired_candidate_seed(0, 7, "causal_fit"))
        self.assertNotEqual(left, runner._paired_candidate_seed(1, 0, "causal_fit"))

    def test_v14_fixture_namespaces_are_fresh_deterministic_and_disjoint(self):
        freeze_path = runner._REPO_ROOT / runner.FREEZE_FILE
        if not freeze_path.is_file():
            self.assertEqual(fixtures.VISIBLE_SEED, 202610100)
            self.assertEqual(fixtures.SEEDS, (202610101, 202610102, 202610103, 202610104, 202610105, 202610106))
            self.assertEqual(fixtures.VISIBLE_COUNT, 32)
            self.assertEqual(fixtures.ADAPTATION_PER_STRATUM, 32)
            self.assertEqual(fixtures.CHALLENGES_PER_STRATUM, 20)
            self.skipTest("V14 fixture generation is sequenced after the immutable packet freeze")
        runner.verify_packet()
        visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        adaptation_rows = fixtures.generate_adaptation_fixture(0)
        adaptation = fixtures.fixture_payload(adaptation_rows, fixture_kind="adaptation")
        challenge_rows, entropy = fixtures.generate_challenge_fixture(0, entropy=b"v" * 32)
        challenge = fixtures.fixture_payload(challenge_rows, fixture_kind="challenge")

        fixtures.validate_partition_disjointness(visible, adaptation, challenge)
        self.assertEqual(len(fixtures.SEEDS), 6)
        self.assertEqual(fixtures.SEEDS[0], 202610101)
        self.assertEqual(entropy, b"v" * 32)
        self.assertEqual(adaptation["protocol_id"], runner.PROTOCOL_ID)
        self.assertNotEqual(adaptation["example_set_sha256"], challenge["example_set_sha256"])


if __name__ == "__main__":
    unittest.main()
