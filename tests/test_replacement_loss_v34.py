"""Engineering fixtures only, frozen before V34 scientific data generation.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v34-gradient-balance-v1,
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

import replacement_loss_v34_runner as v
import replacement_loss_v34_closeout as audit


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
            result, diagnostics, traces = v.fit(guide, method, 'relu', 'row', 79, x, -x, masks, stage_steps=2, final_steps=2)
            self.assertEqual(tuple(diagnostics), v.b.STAGES)
            self.assertEqual(tuple(traces), v.b.STAGES)
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

    def test_all_declared_partition_streams_are_unique(self):
        partitions = ('train', 'visible', 'adapt', 'donor', 'align_fit', 'align_validate', 'guide_optimizer', 'training_masks') + tuple('map/' + s + '/' + f for s in v.b.STRATA for f in ('x', 'donor', 'coordinates'))
        ids = [v.stream_seed(phase, family, seed, part)
               for phase, seeds in [('development', v.DEV_SEEDS), ('confirmation', v.CONFIRM_SEEDS)]
               for family in (('relu',) if phase == 'development' else v.b.FAMILIES)
               for seed in seeds for part in partitions]
        self.assertEqual(len(ids), len(set(ids)))
        arrays = [v.b.draw(v.stream_seed('development', 'relu', seed, part), 32).numpy()
                  for seed in v.DEV_SEEDS for part in partitions[:6]]
        v.ensure_fresh(arrays, set())
        # The former adjacent-offset rule really duplicated a stream.
        self.assertTrue(torch.equal(v.b.draw(33101 + 11, 32), v.b.draw(33102 + 10, 32)))

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
        self.assertEqual(v.select_development(cases)['selected_variant'], 'relu_global')
        cases['2']['arms']['relu_global']['passed'] = False
        self.assertEqual(v.select_development(cases)['selected_variant'], 'relu_row')
        for arm in v.VARIANTS:
            cases['0']['arms'][arm]['passed'] = False
        self.assertIsNone(v.select_development(cases)['selected_variant'])

    def test_confirmation_geometry_policy(self):
        self.assertEqual(v.arm_configs('confirmation', 'relu', 'relu_row')['selected_causal'], ('causal', 'relu', 'row'))
        self.assertEqual(v.arm_configs('confirmation', 'silu_gated', 'relu_row')['selected_causal'], ('causal', 'gelu', 'row'))
        self.assertEqual(v.arm_configs('confirmation', 'relu', 'relu_row')['baseline_causal_global'], ('causal', 'relu', 'global'))

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
            with patch.object(v, 'FREEZE', freeze), patch.object(v, 'material', return_value=[{'path': 'changed'}]), patch.object(v.parent, 'verify_freeze', return_value='base'):
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
                (root / 'requirements-v34.lock.txt').write_text('engineering')
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


    def test_gradient_norm_is_detached_without_grad_accumulation(self):
        parameter = torch.nn.Parameter(torch.tensor([3., 4.]))
        self.assertAlmostEqual(v.gradient_norm(parameter.square().sum(), (parameter,)), 10.)
        self.assertIsNone(parameter.grad)
        parameter.grad = torch.tensor([1., 2.])
        before = parameter.grad.clone()
        v.gradient_norm(parameter.square().sum(), (parameter,))
        self.assertTrue(torch.equal(before, parameter.grad))

    def test_coefficient_cap_and_degenerate_norms(self):
        for anchor, effect in ((1., 100.), (4., 1.), (0., 1.), (1., 0.), (0., 0.)):
            scale = v.coefficient(anchor, effect, True)
            self.assertTrue(0 <= scale <= 1)
            self.assertLessEqual(scale * effect, v.BALANCE_RATIO * anchor + 1e-12)
            self.assertEqual(v.coefficient(anchor, effect, False), 1.)
        for norms in ((float('nan'), 1.), (1., float('inf')), (-1., 1.)):
            with self.assertRaises(ValueError):
                v.coefficient(*norms, True)

    def test_global_gradient_readouts_preserve_parent_training_exactly(self):
        torch.manual_seed(113)
        guide = v.b.Model('relu').eval().requires_grad_(False)
        x, donor = v.b.draw(121, 32), v.b.draw(122, 32)
        masks = v.b.masks(np.random.default_rng(123), 32, 1)
        expected = v.parent.fit(guide, 'causal', 'relu', 'global', 124, x, donor, masks, stage_steps=2, final_steps=2)
        actual, diagnostics, traces = v.fit(guide, 'causal', 'relu', 'global', 124, x, donor, masks, stage_steps=2, final_steps=2)
        for stage in v.b.STAGES:
            self.assertEqual(v.b.array_hash(v.b.state_arrays(expected[stage])), v.b.array_hash(v.b.state_arrays(actual[stage])))
            self.assertEqual(traces[stage].shape, (2, 5))
            self.assertEqual(diagnostics[stage]['norm_readouts'], 2)
            self.assertTrue(np.all(traces[stage][:, 2] == 1))

    def test_balanced_micro_ladder_enforces_each_update_cap(self):
        torch.manual_seed(131)
        guide = v.b.Model('relu').eval().requires_grad_(False)
        x, donor = v.b.draw(132, 32), v.b.draw(133, 32)
        masks = v.b.masks(np.random.default_rng(134), 32, 1)
        stages, diagnostics, traces = v.fit(guide, 'causal', 'relu', 'balanced_row', 135, x, donor, masks, stage_steps=3, final_steps=3)
        for stage in v.b.STAGES:
            self.assertTrue(np.isfinite(traces[stage]).all())
            self.assertTrue(np.all(traces[stage][:, 1] * traces[stage][:, 2] <= .5 * traces[stage][:, 0] + 1e-8))
            self.assertTrue(diagnostics[stage]['balanced'])
            self.assertTrue(all(torch.isfinite(p).all() for p in stages[stage].parameters()))

    def test_map_causal_qualification_identity_and_bad_transport(self):
        torch.manual_seed(141)
        model = v.b.Model('gelu').eval().requires_grad_(False)
        x = v.b.draw(142, 64)
        mask = torch.ones(64, 16, dtype=torch.bool)
        queries = {'engineering': {'x': x, 'donor': -x, 'counterfactual': -x, 'masks': [mask, mask]}}
        identity = {'matrix': np.eye(16), 'bias': np.zeros(16), 'inverse': np.eye(16)}
        for site in range(2):
            good = v.qualify_map_site(model, model, identity, queries, site)['engineering']
            self.assertTrue(good['passed'])
            bad = {'matrix': np.eye(16) * 3, 'bias': np.ones(16), 'inverse': np.eye(16) / 3}
            failure = v.qualify_map_site(model, model, bad, queries, site)['engineering']
            self.assertFalse(failure['passed'])
            self.assertGreater(failure['activation_normalized_rmse'], .15)

    def test_representation_qualification_cannot_override_causal_failure(self):
        torch.manual_seed(151)
        model = v.b.Model('gelu').eval().requires_grad_(False)
        x, validation = v.b.draw(152, 64), v.b.draw(153, 64)
        identity = {'matrix': np.eye(16), 'bias': np.zeros(16), 'inverse': np.eye(16)}
        representation = {'qualified': True, 'validation_normalized_rmse': 0., 'roundtrip_normalized_rmse': 0., 'condition_number': 1.}
        with patch.object(v, 'fit_map', return_value=(identity, representation)), patch.object(v, 'qualify_map_site', return_value={'constructed': {'passed': False}}):
            maps, metrics = v.calibrate(model, model, x, validation, 1, {})
        self.assertTrue(metrics['0']['representation_qualified'])
        self.assertFalse(metrics['0']['qualified'])
        self.assertIsNone(v.aligned_predictions(model, {}, maps, metrics, 1))

    def test_map_validation_streams_are_fresh_and_separate(self):
        seen = set()
        for seed in v.DEV_SEEDS:
            queries = v.map_queries('development', 'relu', seed, rows=16)
            self.assertEqual(tuple(queries), v.b.STRATA)
            v.ensure_fresh([q[k].numpy() for q in queries.values() for k in ('x', 'donor', 'counterfactual')], seen)
        self.assertEqual(len(seen), 3 * 4 * 3 * 16)

    def test_balanced_confirmation_policy(self):
        for family in v.b.FAMILIES:
            configs = v.arm_configs('confirmation', family, 'relu_balanced_row')
            self.assertEqual(tuple(configs), v.CONFIRM_ARMS)
            self.assertEqual(configs['selected_causal'][2], 'balanced_row')
            self.assertEqual(configs['selected_causal'][1], 'relu' if family == 'relu' else 'gelu')
        with self.assertRaises(ValueError):
            v.arm_configs('confirmation', 'relu', 'unknown')

    def test_independent_effect_score_resolution_and_direction(self):
        reference = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype='float32')
        self.assertTrue(audit.relative_effect(reference, reference)['passed'])
        self.assertFalse(audit.relative_effect(-reference, reference)['passed'])
        empty = audit.relative_effect(np.zeros_like(reference), np.zeros_like(reference))
        self.assertEqual(empty['resolved_fraction'], 0.)
        self.assertIsNone(empty['p90_error'])
        self.assertFalse(empty['passed'])

    def test_independent_native_score_and_corruption_detection(self):
        torch.manual_seed(161)
        model = v.b.Model('gelu').eval().requires_grad_(False)
        x = v.b.draw(162, 64)
        mask = torch.ones(64, 16, dtype=torch.bool)
        queries = {'engineering': {'x': x, 'donor': -x, 'counterfactual': -x, 'masks': [mask, mask]}}
        oracle = v.b.predicted(model, queries)
        rescues = {k: a for k, a in oracle.items() if '/rescue' in k}
        expected = v.b.score(oracle, oracle, queries, 2, rescues)['engineering']
        actual = audit.native_cell(oracle, oracle, rescues, v.b.query_arrays(queries), 'engineering', 2)
        self.assertTrue(audit.close(actual, expected))
        self.assertTrue(actual['passed'])
        altered = {k: a.copy() for k, a in oracle.items()}
        altered['engineering/effect0'] *= -1
        failure = audit.native_cell(altered, oracle, rescues, v.b.query_arrays(queries), 'engineering', 2)
        self.assertFalse(failure['passed'])
        self.assertGreater(failure['internal_effects']['0']['p90_error'], .5)

    def test_source_members_are_unique(self):
        self.assertEqual(len(v.MEMBERS), len(set(v.MEMBERS)))
        self.assertIn('replacement_loss_v34_closeout.py', v.MEMBERS)


if __name__ == '__main__':
    unittest.main()

