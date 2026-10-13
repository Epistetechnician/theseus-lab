"""Focused tests for replacement-loss-v3's learned causal predictor core.

State slice: theseus-lab-sandbox-v1.
"""

import unittest

import numpy as np
import torch
from torch import nn

from replacement_loss_v3 import (
    bootstrap_mean_difference_lower_bound,
    causal_effect_metrics,
    fit_independent_linear_replacement,
    linear_intervention_prediction,
    prediction_commitment_digest,
)


class TestReplacementLossV3(unittest.TestCase):
    def test_fit_uses_adaptation_pairs_and_recovers_linear_map(self) -> None:
        torch.manual_seed(812)
        teacher = nn.Linear(5, 3, bias=True, dtype=torch.float64)
        adaptation_x = torch.randn(80, 5, dtype=torch.float64)
        adaptation_y = teacher(adaptation_x).detach()
        target_before = teacher.weight.detach().clone()

        replacement, receipt = fit_independent_linear_replacement(
            5,
            3,
            adaptation_x,
            adaptation_y,
            use_bias=True,
            seed=42,
            ridge=1e-10,
        )

        heldout_x = torch.randn(12, 5, dtype=torch.float64)
        np.testing.assert_allclose(replacement(heldout_x).detach().numpy(), teacher(heldout_x).detach().numpy(), atol=1e-8)
        self.assertTrue(torch.equal(teacher.weight, target_before))
        self.assertIsNot(replacement.weight, teacher.weight)
        self.assertNotEqual(receipt["initialization_digest"], receipt["fitted_replacement_digest"])
        self.assertEqual(receipt["method"], "primal_ridge_least_squares")

    def test_fit_uses_dual_solve_when_feature_width_exceeds_rows(self) -> None:
        torch.manual_seed(314)
        teacher = nn.Linear(12, 4, bias=False, dtype=torch.float64)
        adaptation_x = torch.randn(7, 12, dtype=torch.float64)
        adaptation_y = teacher(adaptation_x).detach()
        replacement, receipt = fit_independent_linear_replacement(
            12,
            4,
            adaptation_x,
            adaptation_y,
            use_bias=False,
            seed=31,
            ridge=1e-12,
        )
        torch.testing.assert_close(replacement(adaptation_x), adaptation_y, atol=1e-9, rtol=1e-9)
        self.assertEqual(receipt["method"], "dual_ridge_least_squares")

    def test_prediction_is_paired_module_effect(self) -> None:
        replacement = nn.Linear(3, 2, bias=True, dtype=torch.float64)
        base = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
        delta = torch.tensor([0.1, 0.0, -0.2], dtype=torch.float64)
        predicted = linear_intervention_prediction(replacement, base, delta)
        expected = replacement.weight @ delta
        torch.testing.assert_close(predicted, expected)

    def test_biasless_projection_fit_does_not_introduce_an_intercept(self) -> None:
        torch.manual_seed(721)
        teacher = nn.Linear(4, 2, bias=False, dtype=torch.float64)
        adaptation_x = torch.randn(20, 4, dtype=torch.float64) + 3.0
        adaptation_y = teacher(adaptation_x).detach()
        replacement, _ = fit_independent_linear_replacement(
            4,
            2,
            adaptation_x,
            adaptation_y,
            use_bias=False,
            seed=5,
            ridge=1e-12,
        )
        heldout_x = torch.randn(7, 4, dtype=torch.float64) + 2.0
        torch.testing.assert_close(replacement(heldout_x), teacher(heldout_x), atol=1e-9, rtol=1e-9)

    def test_effect_metric_keeps_unresolvable_rows_explicit(self) -> None:
        unresolved = causal_effect_metrics(
            np.array([1.0, 0.0]),
            np.array([0.01, 0.0]),
            resolution_floor=0.02,
            scale=1.0,
        )
        self.assertFalse(unresolved["resolvable"])

        scored = causal_effect_metrics(
            np.array([1.0, 0.0]),
            np.array([0.8, 0.2]),
            resolution_floor=0.01,
            scale=1.0,
        )
        self.assertTrue(scored["resolvable"])
        self.assertTrue(scored["direction_correct"])
        self.assertGreater(float(scored["cosine_similarity"]), 0.9)

    def test_commitment_and_cluster_bootstrap_are_deterministic(self) -> None:
        prediction = {"rows": [{"challenge_id": "c1", "effect": [1.0, 2.0]}]}
        self.assertEqual(prediction_commitment_digest(prediction), prediction_commitment_digest(prediction))
        treatment = np.array([0.8, 0.9, 1.0, 0.7])
        control = np.array([0.1, 0.2, 0.3, 0.2])
        first = bootstrap_mean_difference_lower_bound(treatment, control, seed=9, draws=1000)
        second = bootstrap_mean_difference_lower_bound(treatment, control, seed=9, draws=1000)
        self.assertEqual(first, second)
        self.assertGreater(first, 0.0)

    def test_fit_rejects_unpaired_or_wrong_shape_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "paired"):
            fit_independent_linear_replacement(
                4,
                2,
                torch.randn(8, 4),
                torch.randn(7, 2),
                use_bias=True,
                seed=3,
            )
        with self.assertRaisesRegex(ValueError, "width"):
            fit_independent_linear_replacement(
                4,
                2,
                torch.randn(8, 3),
                torch.randn(8, 2),
                use_bias=True,
                seed=3,
            )


if __name__ == "__main__":
    unittest.main()
