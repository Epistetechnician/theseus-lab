"""V2 wrapper and frozen factorial contracts.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v2,
global-experiment-assurance-lifecycle-v1.
"""
import unittest

import numpy as np
import torch
from torch import nn

import theseus_v39_fit_factorial_v2 as pilot


class FactorialV2Contracts(unittest.TestCase):
    def test_v2_uses_distinct_protocol_and_artifact_root(self):
        self.assertEqual(pilot.PROTOCOL, "theseus-v39-first-mlp-fit-factorial-v2")
        self.assertIn("v2", pilot.BASE.name)
        self.assertIs(pilot.engine.verify_parent, pilot.verify_parent)

    def test_capacity_changes_hidden_width_only(self):
        small = pilot.CapacityReplacement(8, "gpt2_gelu", 11, 2)
        large = pilot.CapacityReplacement(8, "gpt2_gelu", 11, 4)
        self.assertEqual((small.up.out_features, large.up.out_features), (16, 32))
        self.assertEqual(sum(p.numel() for p in small.parameters()), 280)
        self.assertEqual(sum(p.numel() for p in large.parameters()), 552)
        inputs = torch.randn(5, 8)
        self.assertEqual(small(inputs).shape, inputs.shape)
        self.assertEqual(large(inputs).shape, inputs.shape)

    def test_baseline_matches_original_fitter(self):
        teacher = nn.Linear(8, 8).eval().requires_grad_(False)
        x, donor = torch.randn(12, 8), torch.randn(12, 8)
        reference, reference_trace = pilot.engine.parent.ORIGINAL_FIT(
            teacher, x, donor, "gpt2_gelu", 173, "jvp", updates=3
        )
        candidate, candidate_trace = pilot.fit_candidate(
            teacher, x, donor, 173, expansion=2, updates=3, budget=lambda: None
        )
        for name, value in reference.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]))
        self.assertTrue(np.array_equal(reference_trace, candidate_trace))

    def test_update_budget_reuses_the_same_training_prefix(self):
        teacher = nn.Linear(8, 8).eval().requires_grad_(False)
        x, donor = torch.randn(12, 8), torch.randn(12, 8)
        _, short = pilot.fit_candidate(teacher, x, donor, 271, 2, 2, lambda: None)
        _, long = pilot.fit_candidate(teacher, x, donor, 271, 2, 3, lambda: None)
        self.assertTrue(np.array_equal(short, long[:2]))

    def test_paired_factorial_reduces_to_frozen_decisions(self):
        cells = {}
        for seed in pilot.SEEDS:
            cells[f"gpt2_gelu-{seed}"] = {
                "cap2_updates1024": {"training_probe": {"nrmse": 1.0}},
                "cap2_updates4096": {"training_probe": {"nrmse": 0.7}},
                "cap4_updates1024": {"training_probe": {"nrmse": 0.75}},
                "cap4_updates4096": {"training_probe": {"nrmse": 0.5}},
            }
        result = pilot.factorial_summary(cells)
        self.assertTrue(result["optimization_budget_main_effect"]["supported"])
        self.assertTrue(result["capacity_main_effect"]["supported"])
        cells["gpt2_gelu-37102"]["cap2_updates4096"]["training_probe"]["nrmse"] = 1.2
        cells["gpt2_gelu-37102"]["cap4_updates4096"]["training_probe"]["nrmse"] = 1.0
        result = pilot.factorial_summary(cells)
        self.assertFalse(result["optimization_budget_main_effect"]["supported"])


if __name__ == "__main__":
    unittest.main()
