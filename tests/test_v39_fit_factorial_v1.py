"""Hermetic contracts for the V39 first-MLP fit-factorial pilot.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v1,
global-experiment-assurance-lifecycle-v1.
"""
import unittest

import numpy as np
import torch
from torch import nn

import replacement_loss_v36_core as core
import theseus_v39_fit_factorial_v1 as pilot


class FactorialContracts(unittest.TestCase):
    def test_capacity_is_the_only_architecture_change(self):
        small = pilot.CapacityReplacement(8, "gpt2_gelu", 11, 2)
        large = pilot.CapacityReplacement(8, "gpt2_gelu", 11, 4)
        self.assertEqual(small.up.out_features, 16)
        self.assertEqual(large.up.out_features, 32)
        small_parameters = 2 * 8 * 16 + 16 + 8
        large_parameters = 2 * 8 * 32 + 32 + 8
        self.assertEqual(sum(p.numel() for p in small.parameters()), small_parameters)
        self.assertEqual(sum(p.numel() for p in large.parameters()), large_parameters)
        self.assertGreater(large_parameters / small_parameters, 1.9)
        self.assertLess(large_parameters / small_parameters, 2.0)
        values = torch.randn(5, 8)
        self.assertEqual(small(values).shape, values.shape)
        self.assertEqual(large(values).shape, values.shape)

    def test_baseline_cell_matches_frozen_core_fit(self):
        torch.use_deterministic_algorithms(True)
        teacher = nn.Linear(8, 8)
        teacher.eval().requires_grad_(False)
        x = torch.randn(12, 8)
        donor = torch.randn(12, 8)
        direct, direct_trace = pilot.parent.ORIGINAL_FIT(
            teacher, x, donor, "gpt2_gelu", 173, "jvp", updates=3
        )
        candidate, candidate_trace = pilot.fit_candidate(
            teacher, x, donor, 173, expansion=2, updates=3, budget=lambda: None
        )
        for name, value in direct.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]))
        self.assertTrue(np.array_equal(direct_trace, candidate_trace))

    def test_training_direction_reconstruction_advances_mask_rng_first(self):
        count, width, seed = 9, 8, 203
        generator = torch.Generator().manual_seed(seed)
        torch.randint(width, (count,), generator=generator)
        expected = (torch.randint(2, (4, count, width), generator=generator).float() * 2 - 1) / np.sqrt(width)
        actual = pilot.probe_directions(count, width, seed)
        self.assertTrue(torch.equal(expected, actual))

    def test_longer_budget_preserves_paired_short_run_prefix(self):
        teacher = nn.Linear(8, 8)
        teacher.eval().requires_grad_(False)
        x = torch.randn(12, 8)
        donor = torch.randn(12, 8)
        _, short_trace = pilot.fit_candidate(
            teacher, x, donor, 271, expansion=2, updates=2, budget=lambda: None
        )
        _, long_trace = pilot.fit_candidate(
            teacher, x, donor, 271, expansion=2, updates=3, budget=lambda: None
        )
        self.assertTrue(np.array_equal(short_trace, long_trace[:2]))

    def test_factor_decisions_require_size_and_seed_direction(self):
        cells = {}
        for seed in pilot.SEEDS:
            cells[f"gpt2_gelu-{seed}"] = {
                "cap2_updates1024": {"training_probe": {"nrmse": 1.0}},
                "cap2_updates4096": {"training_probe": {"nrmse": 0.7}},
                "cap4_updates1024": {"training_probe": {"nrmse": 0.75}},
                "cap4_updates4096": {"training_probe": {"nrmse": 0.5}},
            }
        decision = pilot.factorial_summary(cells)
        self.assertTrue(decision["optimization_budget_main_effect"]["supported"])
        self.assertTrue(decision["capacity_main_effect"]["supported"])
        cells["gpt2_gelu-37102"]["cap2_updates4096"]["training_probe"]["nrmse"] = 1.2
        cells["gpt2_gelu-37102"]["cap4_updates4096"]["training_probe"]["nrmse"] = 1.0
        decision = pilot.factorial_summary(cells)
        self.assertFalse(decision["optimization_budget_main_effect"]["supported"])

    def test_disjoint_hashes_are_added_without_retaining_fixture_text(self):
        seen_text, seen_tokens = set(), set()
        row = {"text_sha256": "a" * 64, "ids": [1, 2, 3]}
        pilot.add_encoded_hashes([row], seen_text, seen_tokens)
        self.assertEqual(seen_text, {"a" * 64})
        self.assertEqual(seen_tokens, {core.digest([1, 2, 3])})


if __name__ == "__main__":
    unittest.main()
