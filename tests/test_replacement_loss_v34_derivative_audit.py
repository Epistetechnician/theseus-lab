"""Prefreeze engineering tests for theseus-v34-adaptation-jacobian-audit-v1.

Also touches theseus-lab-sandbox-v1 and global-experiment-assurance-lifecycle-v1.
"""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import replacement_loss_v34_derivative_audit as d


class Engineering(unittest.TestCase):
    def test_analytic_block_jacobians_match_autograd_all_adapters(self):
        for family in d.v.b.FAMILIES:
            torch.manual_seed(41)
            block = d.v.b.Block(family, 13).double()
            x = torch.linspace(-.7, .9, 16, dtype=torch.float64)
            expected = torch.autograd.functional.jacobian(block, x)
            actual = d.block_jacobian(block, x[None])[0]
            self.assertTrue(torch.allclose(actual, expected, atol=1e-12, rtol=1e-10), family)

    def test_prefix_jacobians_match_autograd_all_adapters(self):
        for family in d.v.b.FAMILIES:
            torch.manual_seed(42)
            model = d.v.b.Model(family).double()
            x = torch.linspace(-.4, .5, 8, dtype=torch.float64)
            actual = d.prefix_jacobians(model, x[None])
            for site in range(2):
                expected = torch.autograd.functional.jacobian(lambda a: model(a)[1][site], x)
                self.assertTrue(torch.allclose(actual[site][0], expected, atol=1e-12, rtol=1e-10))

    def test_relu_zero_subgradient_and_trainable_jacobian_loss(self):
        block = d.v.b.Block('relu', 12)
        with torch.no_grad():
            block.up.weight.zero_()
            block.up.bias.zero_()
        self.assertTrue(torch.equal(d.block_jacobian(block, torch.ones(3,16)), torch.zeros(3,16,16)))
        torch.manual_seed(43)
        block = d.v.b.Block('relu', 12)
        d.block_jacobian(block, torch.ones(3,16)).square().mean().backward()
        self.assertTrue(torch.isfinite(block.up.weight.grad).all())
        self.assertTrue(torch.isfinite(block.down.weight.grad).all())
        self.assertGreater(float(block.up.weight.grad.abs().sum()), 0.)

    def test_identity_decomposition_and_shared_suffix_control(self):
        torch.manual_seed(44)
        guide = d.v.b.Model('relu').eval().requires_grad_(False)
        x, donor = d.v.b.draw(45,64), d.v.b.draw(46,64)
        masks = d.v.b.masks(np.random.default_rng(47),64,1)
        arrays, metrics = d.analyze_site(guide, guide, x, donor, masks, 0)
        self.assertLess(metrics['decomposition_max_absolute_residual'], 1e-6)
        self.assertEqual(metrics['prefix_jacobian_normalized_rmse'], 0.)
        self.assertTrue(np.all(arrays['suffix_error'] == 0))
        candidate = copy.deepcopy(guide)
        with torch.no_grad():
            candidate.blocks[0].down.weight.mul_(1.2)
        arrays, metrics = d.analyze_site(guide, candidate, x, donor, masks, 0)
        self.assertTrue(np.all(arrays['suffix_error'] == 0))
        self.assertLess(metrics['decomposition_max_absolute_residual'], 1e-6)
        self.assertGreater(metrics['component_rms']['delta_error'], 0.)

    def test_changed_suffix_is_measured_separately(self):
        torch.manual_seed(48)
        guide = d.v.b.Model('relu').eval().requires_grad_(False)
        candidate = copy.deepcopy(guide)
        with torch.no_grad():
            candidate.head.weight.mul_(2)
        x, donor = d.v.b.draw(49,64), d.v.b.draw(50,64)
        masks = d.v.b.masks(np.random.default_rng(51),64,1)
        _, metrics = d.analyze_site(guide, candidate, x, donor, masks, 1)
        self.assertGreater(metrics['component_rms']['suffix_error'], 0.)
        self.assertLess(metrics['decomposition_max_absolute_residual'], 1e-6)
        self.assertEqual(metrics['mean_base_boundary_disagreement'], 0.)

    def test_spearman_ties_and_zero_variance(self):
        self.assertTrue(np.array_equal(d.ranks(np.array([1.,1.,3.])), [.5,.5,2.]))
        self.assertAlmostEqual(d.spearman(np.array([1,1,3]),np.array([2,2,4])),1.)
        self.assertIsNone(d.spearman(np.ones(4),np.arange(4)))

    @staticmethod
    def decision_rows():
        return [{'case':str(i),'arm':'relu_global','stage':'block_1','site':0,
                 'metrics':{'decomposition_max_absolute_residual':0.,'resolved_fraction':1.,
                            'weak_error_excess':.2,'secant_error_correlation':.4,'prefix_jacobian_normalized_rmse':.2}}
                for i in range(3)]

    def test_decision_requires_two_guides_and_valid_decomposition(self):
        rows=self.decision_rows()
        self.assertEqual(len(d.decide(rows)['supporting_guides']),3)
        rows[0]['metrics']['weak_error_excess']=0.
        self.assertTrue(d.decide(rows)['outcome_status'].startswith('DIAGNOSTIC_SUPPORTS'))
        rows[1]['metrics']['weak_error_excess']=0.
        self.assertEqual(d.decide(rows)['outcome_status'],'DIAGNOSTIC_DOES_NOT_SUPPORT_CONTINUATION')
        rows=self.decision_rows()
        rows[0]['metrics']['decomposition_max_absolute_residual']=1.
        self.assertFalse(d.decide(rows)['decomposition_control_passed'])

    def test_invalid_role_and_terminal_root_preserved(self):
        with self.assertRaises(ValueError):
            d.run('other')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'primary').mkdir()
            marker=root/'primary/marker'
            marker.write_text('preserve')
            with patch.object(d,'BASE',root),patch.object(d,'verify',return_value='packet'),patch.object(d.v.b,'runtime',return_value={}):
                with self.assertRaises(FileExistsError):
                    d.run('primary')
            self.assertEqual(marker.read_text(),'preserve')


if __name__ == '__main__':
    unittest.main()
