"""Read-only V34 custody and numeric audit, separate from runner scoring.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v34-gradient-balance-v1,
global-experiment-assurance-lifecycle-v1. Writes only outside terminal run roots.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import replacement_loss_v34_runner as v


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def relative_effect(prediction, reference):
    resolved = np.linalg.norm(reference, axis=1) >= 1e-4
    if not resolved.any():
        return {'resolved_fraction': 0., 'median_error': None, 'p90_error': None,
                'median_cosine': None, 'passed': False}
    prediction, reference = prediction[resolved], reference[resolved]
    error = np.linalg.norm(prediction - reference, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-6)
    cosine = np.sum(prediction * reference, axis=1) / np.maximum(np.linalg.norm(prediction, axis=1) * np.linalg.norm(reference, axis=1), 1e-12)
    metrics = {'resolved_fraction': float(np.mean(resolved)), 'median_error': float(np.median(error)),
               'p90_error': float(np.quantile(error, .9)), 'median_cosine': float(np.median(cosine))}
    metrics['passed'] = (metrics['resolved_fraction'] >= .9 and metrics['median_error'] <= .2
                         and metrics['p90_error'] <= .5 and metrics['median_cosine'] >= .9)
    return metrics


def close(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (float, np.floating)):
        return b is not None and np.isclose(a, b, rtol=1e-6, atol=1e-7)
    return a == b


def native_cell(prediction, oracle, rescues, inputs, stratum, active):
    prefix = stratum + '/'
    output, reference = prediction[prefix + 'output'], oracle[prefix + 'output']
    ground = v.b.truth(torch.from_numpy(inputs[prefix + 'x'])).numpy()
    guide_accuracy = float(np.mean((reference >= 0) == (ground >= 0)))
    candidate_accuracy = float(np.mean((output >= 0) == (ground >= 0)))
    activation, rescue, prediction_error = {}, {}, {}
    for site in range(active):
        key = str(site)
        activation[key] = v.nrmse(prediction[prefix + 'activation' + key], oracle[prefix + 'activation' + key])
        rescue[key] = v.nrmse(rescues[prefix + 'rescue' + key], reference)
        prediction_error[key] = v.nrmse(prediction[prefix + 'rescue' + key], rescues[prefix + 'rescue' + key])
    effects = {str(site): relative_effect(prediction[prefix + 'effect' + str(site)], oracle[prefix + 'effect' + str(site)]) for site in range(active)}
    input_effect = relative_effect(prediction[prefix + 'input_effect'], oracle[prefix + 'input_effect'])
    result = {'output_normalized_rmse': v.nrmse(output, reference),
              'guide_sign_agreement': float(np.mean((output >= 0) == (reference >= 0))),
              'guide_truth_accuracy': guide_accuracy, 'candidate_truth_accuracy': candidate_accuracy,
              'truth_accuracy_drop': guide_accuracy - candidate_accuracy,
              'input_effect': input_effect, 'internal_effects': effects,
              'activation_errors': activation, 'learned_rescue_errors': rescue,
              'rescue_prediction_errors': prediction_error}
    result['passed'] = bool(result['output_normalized_rmse'] <= .1 and result['guide_sign_agreement'] >= .95
                             and result['truth_accuracy_drop'] <= .03 and input_effect['passed']
                             and all(e['passed'] for e in effects.values())
                             and max(activation.values()) <= .15 and max(rescue.values()) <= .1
                             and max(prediction_error.values()) <= .1)
    return result


def audit_run(root, result, phase):
    checked_files, checked_cells, checked_traces = 0, 0, 0
    seen = v.exclude_rows(phase)
    for member in result['artifact_manifest']:
        path = root / member['path']
        if not path.is_relative_to(root) or path.stat().st_size != member['bytes'] or digest(path) != member['sha256']:
            raise ValueError('independent manifest verification failed')
        checked_files += 1
    freeze = json.loads(v.FREEZE.read_text())
    for member in freeze['packet_files']:
        if digest(root / 'frozen-packet' / member['path']) != member['sha256']:
            raise ValueError('frozen source archive differs')
    if digest(root / 'frozen-packet' / v.FREEZE.relative_to(v.REPO)) != digest(v.FREEZE):
        raise ValueError('archived freeze receipt differs')
    for case in result['preparation']:
        for filename in ('preparation-data.npz', 'map-validation-inputs.npz'):
            arrays = load_arrays(root / case / filename)
            inputs = [value for key, value in arrays.items() if key in ('train', 'visible', 'adapt', 'donor', 'align_fit', 'align_validate') or key.rsplit('/', 1)[-1] in ('x', 'donor', 'counterfactual')]
            v.ensure_fresh(inputs, seen)
    if not result['challenge_generated']:
        return {'manifest_files': checked_files, 'native_cells': 0, 'gradient_readouts': 0,
                'final_artifact_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}
    inputs = load_arrays(root / 'challenge.npz')
    v.ensure_fresh([a for key, a in inputs.items() if key.rsplit('/', 1)[-1] in ('x', 'donor', 'counterfactual')], seen)
    for case, item in result['cases'].items():
        oracle = load_arrays(root / case / 'oracle-outcomes.npz')
        rescues = load_arrays(root / case / 'rescue-outcomes.npz')
        guide = v.b.Model(case.rsplit('-', 1)[0])
        guide.load_state_dict({key: torch.from_numpy(value) for key, value in load_arrays(root / case / 'guide.npz').items()})
        guide.eval().requires_grad_(False)
        for arm, report in item['arms'].items():
            training = result['preparation'][case]['training_diagnostics'][arm]
            traces = load_arrays(root / case / (arm + '-training-diagnostics.npz'))
            for stage, diagnostic in training.items():
                trace = traces[stage]
                if trace.shape != (diagnostic['norm_readouts'], 5) or not np.isfinite(trace).all():
                    raise ValueError('gradient trace contract')
                if v.b.array_hash({'readouts': trace}) != diagnostic['trace_sha256']:
                    raise ValueError('gradient content hash differs')
                if len(trace):
                    anchor, effect, scale = trace[:, 0], trace[:, 1], trace[:, 2]
                    expected = np.minimum(1., .5 * anchor / np.maximum(effect, 1e-12)) if diagnostic['balanced'] else np.ones(len(trace))
                    if not np.array_equal(scale, expected) or (diagnostic['balanced'] and np.any(scale * effect > .5 * anchor + 1e-8)):
                        raise ValueError('independent gradient cap failed')
                checked_traces += len(trace)
            for stage, cells in report['native'].items():
                active = 1 if stage == 'block_1' else 2
                prediction = load_arrays(root / case / (arm + '-' + stage + '-native-predictions.npz'))
                actual = {key.split('/native/', 1)[1]: value for key, value in rescues.items() if key.startswith(arm + '/' + stage + '/native/')}
                with torch.no_grad():
                    for stratum in v.b.STRATA:
                        for site in range(active):
                            prefix = stratum + '/'
                            restored = guide.downstream(torch.from_numpy(prediction[prefix + 'activation' + str(site)]), site).numpy()
                            if not np.array_equal(restored, actual[prefix + 'rescue' + str(site)]):
                                raise ValueError('stored restoration differs from actual guide suffix')
                        recalculated = native_cell(prediction, oracle, actual, inputs, stratum, active)
                        if not close(recalculated, cells[stratum]):
                            raise ValueError('independent native numeric score differs')
                        checked_cells += 1
    return {'manifest_files': checked_files, 'native_cells': checked_cells, 'gradient_readouts': checked_traces,
            'frozen_archive_files': len(freeze['packet_files']) + 1 + int(phase == 'confirmation'),
            'final_artifact_bytes': sum(p.stat().st_size for p in root.rglob('*') if p.is_file())}


def validate(phase):
    packet = v.verify_freeze()
    roots = {role: v.BASE / phase / role for role in ('primary', 'replica')}
    results = {role: v.verify_receipt(root) for role, root in roots.items()}
    primary, replica = results['primary'], results['replica']
    reconciliation_path = roots['replica'] / 'reconciliation.json'
    reconciliation = json.loads(reconciliation_path.read_text())
    keys = ('protocol_id', 'phase', 'packet_sha256', 'confirmation_freeze_sha256', 'outcome_status',
            'preparation', 'guide_qualification', 'challenge_binding', 'prediction_binding',
            'oracle_binding', 'cases', 'selection', 'challenge_generated')
    checks = {key: primary[key] == replica[key] for key in keys}
    runtimes = {role: json.loads((root / 'runtime.json').read_text()) for role, root in roots.items()}
    checks['distinct_runtime'] = all(runtimes['primary'][key] != runtimes['replica'][key] for key in ('sys_prefix', 'site_packages'))
    checks['toolchain'] = {k: a for k, a in runtimes['primary'].items() if k not in ('sys_prefix', 'site_packages')} == {k: a for k, a in runtimes['replica'].items() if k not in ('sys_prefix', 'site_packages')}
    checks['source_packet_unchanged'] = primary['packet_sha256'] == packet
    if checks != reconciliation['checks'] or not all(checks.values()) or reconciliation['status'] != 'REPRODUCED_LOCAL':
        raise ValueError('reconciliation independently recomputed false')
    if any(reconciliation[role + '_result_sha256'] != digest(root / 'result.json') for role, root in roots.items()):
        raise ValueError('reconciliation result binding differs')
    audits = {role: audit_run(root, results[role], phase) for role, root in roots.items()}
    for root in roots.values():
        v.verify_receipt(root)
    return {'state_slices': ['theseus-lab-sandbox-v1', v.SLICE, 'global-experiment-assurance-lifecycle-v1'],
            'phase': phase, 'packet_sha256': packet, 'assurance_level': 'REPRODUCED_LOCAL',
            'scientific_outcome': primary['outcome_status'], 'reviewer_status': 'NOT_SUBMITTED',
            'reconciliation_checks': checks, 'audits': audits, 'monetary_cost_usd': 0,
            'result_digests': {role: digest(root / 'result.json') for role, root in roots.items()},
            'reconciliation_sha256': digest(reconciliation_path), 'same_host_and_owner': True,
            'claim_ceiling': v.CLAIM, 'nonclaims': v.b.NONCLAIMS}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('development', 'confirmation'))
    args = parser.parse_args()
    result = validate(args.phase)
    path = v.BASE / 'closeout' / (args.phase + '-validation.json')
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('existing closeout validation changed')
    else:
        v.b.write_json(path, result)
    print(json.dumps({'scientific_outcome': result['scientific_outcome'], 'assurance_level': result['assurance_level'],
                      'audits': result['audits'], 'validation_sha256': digest(path)}, sort_keys=True))
