"""Engineering fixtures only, frozen before V33 scientific data generation.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v33-relu-geometry-v1,
global-experiment-assurance-lifecycle-v1.
"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import replacement_loss_v33_runner as v


class Engineering(unittest.TestCase):
    def test_equal_parameter_count_and_initialization(self):
        torch.manual_seed(7)
        relu = v.b.Block('relu', 96)
        torch.manual_seed(7)
        gelu = v.b.Block('gelu', 96)
        self.assertEqual(v.b.array_hash(v.b.state_arrays(relu)), v.b.array_hash(v.b.state_arrays(gelu)))
        self.assertEqual(sum(p.numel() for p in relu.parameters()), sum(p.numel() for p in gelu.parameters()))

    def test_global_loss_exact_baseline(self):
        actual, target = torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[2., 3.], [4., 5.]])
        self.assertTrue(torch.equal(v.effect_loss(actual, target, 'global'), v.b.normalized_mse(actual, target)))

    def test_row_weighting_and_finite_zero_effect_gradient(self):
        target = torch.tensor([[.001, 0.], [1., 0.], [0., 0.]])
        actual = (target + torch.tensor([[.001, 0.], [.001, 0.], [.001, 0.]])).requires_grad_(True)
        loss = v.effect_loss(actual, target, 'row')
        self.assertAlmostEqual(float(loss.detach()), (1 + .000001 + 1) / 3, places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(actual.grad).all())
        with self.assertRaises(ValueError):
            v.effect_loss(actual, target, 'unknown')

    def test_micro_ladder_preserves_guide_and_frozen_prefix(self):
        x = torch.tensor([[1., 0., -1., 0., 1., 0., -1., 0.], [0., 1., 0., -1., 0., 1., 0., -1.],
                          [-1., 0., 1., 0., -1., 0., 1., 0.], [0., -1., 0., 1., 0., -1., 0., 1.]])
        masks = [torch.zeros(4, 16, dtype=torch.bool) for _ in range(2)]
        for mask in masks:
            mask[:, 0] = True
        guide = v.b.Model('relu').eval().requires_grad_(False)
        before = v.b.array_hash(v.b.state_arrays(guide))
        for method in ('causal', 'pointwise', 'output', 'cka'):
            result = v.fit(guide, method, 'relu', 'row', 79, x, -x, masks, stage_steps=2, final_steps=2)
            self.assertEqual(tuple(result), v.b.STAGES)
            self.assertTrue(all(torch.isfinite(model(x)[0]).all() for model in result.values()))
            self.assertTrue(torch.equal(result['block_1'].stem.weight, guide.stem.weight))
            self.assertTrue(torch.equal(result['block_2'].head.weight, guide.head.weight))
            self.assertIsNot(result['block_1'].blocks[0], result['block_2'].blocks[0])
        self.assertEqual(before, v.b.array_hash(v.b.state_arrays(guide)))

    def test_affine_map_known_transform_and_inverse(self):
        rng = np.random.default_rng(811)
        a, av = rng.normal(size=(128, 16)), rng.normal(size=(64, 16))
        matrix, _ = np.linalg.qr(rng.normal(size=(16, 16)))
        bias = np.linspace(-.2, .2, 16)
        mapping, metrics = v.fit_map(a, a @ matrix + bias, av, av @ matrix + bias)
        self.assertTrue(metrics['qualified'])
        self.assertLess(metrics['validation_normalized_rmse'], 1e-5)
        mapped = av @ mapping['matrix'] + mapping['bias']
        self.assertLess(v.nrmse((mapped - mapping['bias']) @ mapping['inverse'], av), 1e-12)

    def test_singular_map_unavailable_without_fallback(self):
        rng = np.random.default_rng(812)
        a = rng.normal(size=(64, 16))
        mapping, metrics = v.fit_map(a, np.zeros_like(a), a, np.zeros_like(a))
        self.assertFalse(metrics['qualified'])
        self.assertNotIn('inverse', mapping)
        self.assertIsNone(metrics['condition_number'])

    def test_bad_map_shape_rejected(self):
        a = np.ones((4, 15))
        with self.assertRaises(ValueError):
            v.fit_map(a, a, a, a)

    def test_identity_aligned_prediction_and_unavailable(self):
        torch.manual_seed(81)
        model = v.b.Model('gelu').eval()
        x = torch.tensor([[1., 0., -1., 0., 1., 0., -1., 0.], [0., 1., 0., -1., 0., 1., 0., -1.]])
        mask = torch.ones(2, 16, dtype=torch.bool)
        queries = {'engineering': {'x': x, 'donor': -x, 'counterfactual': -x, 'masks': [mask, mask]}}
        maps = {str(i): {'matrix': np.eye(16), 'bias': np.zeros(16), 'inverse': np.eye(16)} for i in range(2)}
        qualifications = {str(i): {'qualified': True} for i in range(2)}
        aligned = v.aligned_predictions(model, queries, maps, qualifications, 2)
        native = v.b.predicted(model, queries)
        for key in native:
            self.assertTrue(np.array_equal(native[key], aligned[key]))
        qualifications['1']['qualified'] = False
        self.assertIsNone(v.aligned_predictions(model, queries, maps, qualifications, 2))

    def test_freshness_exact_rows_and_duplicate_rejected(self):
        a = np.arange(16, dtype='float32').reshape(2, 8)
        seen = set()
        v.ensure_fresh([a], seen)
        with self.assertRaises(ValueError):
            v.ensure_fresh([a], seen)
        with self.assertRaises(ValueError):
            v.ensure_fresh([a, a], set())

    @staticmethod
    def selection_cases():
        effect = {'median_error': .01, 'p90_error': .02, 'median_cosine': 1.}
        cell = {'output_normalized_rmse': .01, 'activation_errors': {'0': .01},
                'learned_rescue_errors': {'0': .01}, 'rescue_prediction_errors': {'0': .01},
                'input_effect': effect, 'internal_effects': {'0': effect}}
        return {str(i): {'arms': {a: {'passed': True, 'native': {'stage': {'stratum': copy.deepcopy(cell)}}}
                                for a in v.VARIANTS}} for i in range(3)}

    def test_selection_all_seeds_and_declared_tie(self):
        cases = self.selection_cases()
        self.assertEqual(v.select_development(cases)['selected_variant'], 'gelu_global')
        cases['2']['arms']['gelu_global']['passed'] = False
        self.assertEqual(v.select_development(cases)['selected_variant'], 'relu_global')
        for arm in v.VARIANTS:
            cases['0']['arms'][arm]['passed'] = False
        self.assertIsNone(v.select_development(cases)['selected_variant'])

    def test_confirmation_geometry_policy(self):
        self.assertEqual(v.arm_configs('confirmation', 'relu', 'relu_row')['selected_causal'], ('causal', 'relu', 'row'))
        self.assertEqual(v.arm_configs('confirmation', 'silu_gated', 'relu_row')['selected_causal'], ('causal', 'gelu', 'row'))
        self.assertEqual(v.arm_configs('confirmation', 'relu', 'relu_row')['baseline_causal_gelu_global'], ('causal', 'gelu', 'global'))

    def test_terminal_root_reuse_rejected_without_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'development/primary').mkdir(parents=True)
            marker = root / 'development/primary/marker'
            marker.write_text('preserve')
            with patch.object(v, 'BASE', root), patch.object(v, 'verify_freeze', return_value='packet'), patch.object(v.b, 'runtime', return_value={}):
                with self.assertRaises(FileExistsError):
                    v.run('development', 'primary')
            self.assertEqual(marker.read_text(), 'preserve')
            self.assertEqual(len(list(marker.parent.iterdir())), 1)

    def test_bad_frozen_packet_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            freeze = Path(folder) / 'freeze.json'
            freeze.write_text(json.dumps({'packet_files': [], 'packet_sha256': 'wrong'}))
            with patch.object(v, 'FREEZE', freeze), patch.object(v, 'material', return_value=[{'path': 'changed'}]), patch.object(v.b, 'verify_freeze', return_value='base'):
                with self.assertRaises(ValueError):
                    v.verify_freeze()

    def test_changed_artifact_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'asset').write_text('original')
            manifest = v.artifact_manifest(root)
            v.b.write_json(root / 'result.json', {'packet_sha256': 'packet', 'artifact_manifest': manifest})
            with patch.object(v, 'verify_freeze', return_value='packet'):
                v.verify_receipt(root)
                (root / 'asset').write_text('changed')
                with self.assertRaises(ValueError):
                    v.verify_receipt(root)

    def test_no_qualified_development_cannot_freeze_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'development/replica').mkdir(parents=True)
            v.b.write_json(root / 'development/replica/reconciliation.json', {'status': 'REPRODUCED_LOCAL', 'checks': {'a': True}})
            no_candidate = {'selection': {'selected_variant': None}, 'outcome_status': 'NO_QUALIFIED_DEVELOPMENT_CANDIDATE'}
            with patch.object(v, 'BASE', root), patch.object(v, 'CONFIRM', root / 'confirmation-freeze.json'), patch.object(v, 'verify_freeze', return_value='packet'), patch.object(v, 'verify_receipt', return_value=no_candidate):
                with self.assertRaises(ValueError):
                    v.freeze_confirmation()
            self.assertFalse((root / 'confirmation-freeze.json').exists())

    def test_crash_receipt_and_retry_preserve_partial_custody(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            freeze = root / 'source-freeze.json'
            freeze.write_text('{}')
            with patch.object(v, 'BASE', root / 'custody'), patch.object(v, 'FREEZE', freeze), patch.object(v, 'MEMBERS', ()), patch.object(v, 'REPO', root), patch.object(v, 'verify_freeze', return_value='packet'), patch.object(v.b, 'runtime', return_value={}), patch.object(v, 'exclude_rows', return_value=set()), patch.object(v, 'train_guide', side_effect=RuntimeError('constructed training crash')):
                # Give the source archive and lock contract their explicit fixtures.
                (root / 'docs').mkdir()
                (root / 'requirements-v33.lock.txt').write_text('engineering')
                (root / 'requirements-v32.lock.txt').write_text('engineering')
                with self.assertRaisesRegex(RuntimeError, 'constructed training crash'):
                    v.run('development', 'primary')
                custody = root / 'custody/development/primary'
                error = json.loads((custody / 'error.json').read_text())
                self.assertEqual(error['outcome_status'], 'INCONCLUSIVE')
                before = v.artifact_manifest(custody)
                with self.assertRaises(FileExistsError):
                    v.run('development', 'primary')
                self.assertEqual(before, v.artifact_manifest(custody))
                self.assertFalse((custody / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
