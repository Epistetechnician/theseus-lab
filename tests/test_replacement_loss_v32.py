"""Constructed pre-freeze engineering fixtures only; no scientific task samples.

State slice: theseus-replacement-loss-v32-causal-ladder-v1.
"""
import json
import unittest

import numpy as np
import torch

import replacement_loss_v32_runner as v


class Contracts(unittest.TestCase):
    def test_internal_interchange_routes_through_suffix(self):
        for family in v.FAMILIES:
            model = v.Model(family)
            x = torch.ones(3, 8)
            donor = -x
            none = [torch.zeros(3, 16, dtype=torch.bool) for _ in range(2)]
            all_rows = [torch.ones(3, 16, dtype=torch.bool) for _ in range(2)]
            out, _, unchanged = v.effects(model, x, donor, none)
            self.assertTrue(all(torch.equal(effect, torch.zeros_like(out)) for effect in unchanged))
            _, _, swaps = v.effects(model, x, donor, all_rows)
            expected = model(donor)[0] - out
            self.assertTrue(all(torch.allclose(effect, expected, atol=1e-7) for effect in swaps))

    def test_zero_block_removes_all_input_dependence(self):
        for site in range(2):
            model = v.Model('gelu')
            model.blocks[site] = v.Zero()
            result = model(torch.stack([torch.ones(8), -torch.ones(8)]))[0]
            self.assertTrue(torch.equal(result[0], result[1]))

    def test_cka_invariant_scale_and_gradients(self):
        # Explicit non-scientific probe matrix.
        a = torch.tensor([[1.,2.,3.], [2.,1.,0.], [-1.,3.,2.], [0.,1.,-2.]], requires_grad=True)
        loss = 1 - v.cka(a, a.detach() * 3)
        self.assertLess(abs(float(loss.detach())), 1e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(a.grad).all())

    def test_hash_and_json_roundtrip(self):
        a = {'nested': {'3201': {'passed': True}}, 'array': [1.,2.]}
        self.assertEqual(v.canonical(a), v.canonical(json.loads(v.canonical(a))))
        left = {'a': np.ones((2, 4), dtype='float32'), 'b': np.zeros(2, dtype='int64')}
        self.assertEqual(v.array_hash(left), v.array_hash(dict(reversed(list(left.items())))))
        right = {**left, 'a': left['a'].astype('float64')}
        self.assertNotEqual(v.array_hash(left), v.array_hash(right))

    def test_effect_metric_controls_and_zero_resolution(self):
        reference = np.ones((5,4), dtype='float32')
        self.assertTrue(v.causal_metrics(reference, reference)['passed'])
        self.assertFalse(v.causal_metrics(np.zeros_like(reference), reference)['passed'])
        self.assertFalse(v.causal_metrics(reference, np.zeros_like(reference))['passed'])

    def test_shape_validation(self):
        with self.assertRaises(ValueError):
            v.patch(torch.ones(2,3), torch.ones(2,3), torch.ones(2,3))

    def test_scoring_rescue_is_separate_from_prediction(self):
        torch.manual_seed(71)
        model = v.Model('gelu').eval()
        x = torch.stack([torch.ones(8), -torch.ones(8)])
        full = torch.ones(2,16,dtype=torch.bool)
        queries = {'engineering': {'x': x, 'donor': -x, 'counterfactual': -x, 'masks': [full, full]}}
        predictions = v.predicted(model, queries)
        cells = v.score(predictions, predictions, queries, 2)
        self.assertTrue(cells['engineering']['passed'])
        rescues = {k: np.zeros_like(a) for k,a in predictions.items() if '/rescue' in k}
        failed = v.score(predictions, predictions, queries, 2, rescues)
        self.assertFalse(failed['engineering']['passed'])

    def test_source_replacement_has_distinct_shape_and_lineage(self):
        for family in v.FAMILIES:
            guide = v.Model(family)
            candidate = v.Block('gelu',96)
            self.assertNotEqual(v.array_hash(v.state_arrays(guide.blocks[0])), v.array_hash(v.state_arrays(candidate)))
            self.assertEqual(candidate(torch.ones(2,16)).shape, (2,16))

    def test_all_methods_complete_constructed_micro_ladder(self):
        # Four explicit engineering vectors, not the study's random corpus.
        x = torch.tensor([[1.,0.,-1.,0.,1.,0.,-1.,0.], [0.,1.,0.,-1.,0.,1.,0.,-1.],
                          [-1.,0.,1.,0.,-1.,0.,1.,0.], [0.,-1.,0.,1.,0.,-1.,0.,1.]])
        donor = -x
        mask = torch.zeros(4,16,dtype=torch.bool)
        mask[:,0] = True
        guide = v.Model('silu_gated').eval().requires_grad_(False)
        original = v.array_hash(v.state_arrays(guide))
        old = v.STAGE_STEPS, v.FINAL_STEPS
        v.STAGE_STEPS = v.FINAL_STEPS = 2
        try:
            for method in v.METHODS:
                snapshots = v.fit(guide, method, 77, x, donor, [mask,mask])
                self.assertEqual(tuple(snapshots), v.STAGES)
                self.assertTrue(all(torch.isfinite(m(x)[0]).all() for m in snapshots.values()))
                self.assertTrue(torch.equal(snapshots['block_1'].head.weight, guide.head.weight))
                self.assertTrue(torch.equal(snapshots['block_2'].stem.weight, guide.stem.weight))
                self.assertIsNot(snapshots['block_1'].blocks[0], snapshots['block_2'].blocks[0])
            self.assertEqual(original, v.array_hash(v.state_arrays(guide)))
        finally:
            v.STAGE_STEPS, v.FINAL_STEPS = old


if __name__ == '__main__':
    unittest.main()
