"""Frozen adaptation-only secant, Jacobian, and boundary diagnosis.

State slices: theseus-lab-sandbox-v1,
theseus-v34-adaptation-jacobian-audit-v1,
global-experiment-assurance-lifecycle-v1. No training or challenge evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import replacement_loss_v34_runner as v

REPO = v.REPO
BASE = Path('/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-v34-derivative-audit')
SOURCE = v.BASE / 'development/primary'
SLICE = 'theseus-v34-adaptation-jacobian-audit-v1'
CLAIM = 'ADAPTATION_ONLY_ENDPOINT_AND_DERIVATIVE_DIAGNOSIS'
OWN = ('replacement_loss_v34_derivative_audit.py', 'tests/test_replacement_loss_v34_derivative_audit.py',
       'docs/replacement-loss-v34-derivative-audit.md', 'docs/replacement-loss-v34-derivative-inputs.json')
MEMBERS = (*v.MEMBERS, 'docs/replacement-loss-v34-freeze.json', *OWN)
FREEZE = REPO / 'docs/replacement-loss-v34-derivative-freeze.json'
INPUTS = REPO / OWN[-1]


def material():
    return [{'path': p, 'bytes': (REPO / p).stat().st_size, 'sha256': v.b.file_hash(REPO / p)} for p in MEMBERS]


def input_material():
    paths = [f'relu-{seed}/{name}.npz' for seed in v.DEV_SEEDS
             for name in ('preparation-data', 'guide', *(arm + '-' + stage for arm in v.VARIANTS for stage in v.b.STAGES))]
    return [{'path': p, 'bytes': (SOURCE / p).stat().st_size, 'sha256': v.b.file_hash(SOURCE / p)} for p in paths]


def freeze():
    if FREEZE.exists() or INPUTS.exists() or BASE.exists():
        raise FileExistsError('derivative audit already frozen or started')
    v.verify_receipt(SOURCE)
    v.verify_receipt(v.BASE / 'development/replica')
    validation = json.loads((v.BASE / 'closeout/development-validation.json').read_text())
    if validation['assurance_level'] != 'REPRODUCED_LOCAL' or not all(validation['reconciliation_checks'].values()):
        raise ValueError('parent closeout unqualified')
    v.b.write_json(INPUTS, {'parent_packet_sha256': v.verify_freeze(),
                           'parent_primary_result_sha256': v.b.file_hash(SOURCE / 'result.json'),
                           'parent_replica_result_sha256': v.b.file_hash(v.BASE / 'development/replica/result.json'),
                           'parent_validation_sha256': v.b.file_hash(v.BASE / 'closeout/development-validation.json'),
                           'files': input_material(), 'fields': ['adapt', 'donor', 'mask0', 'mask1', 'model state arrays']})
    members = material()
    v.b.write_json(FREEZE, {'state_slice': SLICE, 'claim_ceiling': CLAIM,
                           'freeze_state': 'FROZEN_BEFORE_ADAPTATION_DIAGNOSTIC',
                           'packet_files': members, 'packet_sha256': v.b.hash_value(members),
                           'revision': '202f459d6d89df9a038d78d58e0af825d287fcd8', 'monetary_cap_usd': 0})
    return verify()


def verify():
    v.verify_freeze()
    f = json.loads(FREEZE.read_text())
    m = material()
    i = json.loads(INPUTS.read_text())
    if (f['packet_files'] != m or f['packet_sha256'] != v.b.hash_value(m)
        or f['state_slice'] != SLICE or f['claim_ceiling'] != CLAIM
        or f['freeze_state'] != 'FROZEN_BEFORE_ADAPTATION_DIAGNOSTIC'
        or i['files'] != input_material() or i['parent_packet_sha256'] != v.verify_freeze()
        or i['parent_primary_result_sha256'] != v.b.file_hash(SOURCE / 'result.json')
        or i['parent_replica_result_sha256'] != v.b.file_hash(v.BASE / 'development/replica/result.json')
        or i['parent_validation_sha256'] != v.b.file_hash(v.BASE / 'closeout/development-validation.json')):
        raise ValueError('derivative audit source or input binding changed')
    return f['packet_sha256']


def block_jacobian(block, x):
    """Exact batch Jacobian for these adapters; ReLU zero subgradient at zero."""
    z = block.up(x)
    if block.family == 'relu':
        derivative = (z > 0).to(z.dtype)
        hidden = derivative[:, :, None] * block.up.weight[None]
    elif block.family == 'gelu':
        derivative = .5 * (1 + torch.erf(z / math.sqrt(2))) + z * torch.exp(-z.square() / 2) / math.sqrt(2 * math.pi)
        hidden = derivative[:, :, None] * block.up.weight[None]
    elif block.family == 'silu_gated':
        gate = block.gate(x)
        sigmoid = torch.sigmoid(gate)
        activation = gate * sigmoid
        derivative = sigmoid * (1 + gate * (1 - sigmoid))
        hidden = (activation[:, :, None] * block.up.weight[None]
                  + (z * derivative)[:, :, None] * block.gate.weight[None])
    else:
        raise ValueError('unsupported Jacobian adapter')
    return torch.matmul(block.down.weight[None], hidden)


def prefix_jacobians(model, x):
    h = model.stem(x)
    jacobian = model.stem.weight[None].expand(len(x), -1, -1)
    result = []
    for block in model.blocks:
        jacobian = torch.bmm(block_jacobian(block, h), jacobian)
        result.append(jacobian)
        h = block(h)
    return result


def ranks(values):
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype='float64')
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return result


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def summarize(arrays):
    resolved = arrays['reference_norm'] >= 1e-4
    ids = np.flatnonzero(resolved)
    if len(ids) < 20:
        raise ValueError('adaptation effect insufficiently resolved')
    order = ids[np.argsort(arrays['reference_norm'][ids], kind='stable')]
    weak, other = order[:math.ceil(.2 * len(order))], order[math.ceil(.2 * len(order)):]
    error = arrays['relative_error']
    weak_median, other_median = float(np.median(error[weak])), float(np.median(error[other]))
    return {'resolved_fraction': float(resolved.mean()), 'weak_rows': len(weak),
            'weak_median_relative_error': weak_median, 'other_median_relative_error': other_median,
            'weak_error_excess': weak_median - other_median,
            'effect_error_p90': float(np.quantile(error[ids], .9)),
            'secant_error_correlation': spearman(error[ids], arrays['secant_relative_error'][ids]),
            'boundary_error_correlation': spearman(error[ids], arrays['base_boundary_disagreement'][ids]),
            'prefix_jacobian_normalized_rmse': v.nrmse(arrays['candidate_prefix_jacobian'], arrays['guide_prefix_jacobian']),
            'base_activation_normalized_rmse': v.nrmse(arrays['candidate_base'], arrays['guide_base']),
            'donor_activation_normalized_rmse': v.nrmse(arrays['candidate_donor'], arrays['guide_donor']),
            'component_rms': {key: float(np.sqrt(np.mean(arrays[key]**2))) for key in ('delta_error', 'base_error', 'suffix_error')},
            'decomposition_max_absolute_residual': float(np.abs(arrays['decomposition_residual']).max()),
            'mean_base_boundary_disagreement': float(arrays['base_boundary_disagreement'].mean()),
            'mean_intervention_boundary_crossing': float(arrays['intervention_boundary_crossing'].mean()),
            'linearization_normalized_rmse': v.nrmse(arrays['guide_linear_effect'], arrays['reference_effect'])}


def analyze_site(guide, candidate, x, donor, masks, site):
    with torch.no_grad():
        _, gs, ge = v.b.effects(guide, x, donor, masks)
        _, cs, ce = v.b.effects(candidate, x, donor, masks)
        gd, cd = guide(donor)[1], candidate(donor)[1]
        gb, cb = gs[site], cs[site]
        delta_g, delta_c = masks[site] * (gd[site] - gb), masks[site] * (cd[site] - cb)
        reference = ge[site]
        shared_base = guide.downstream(gb + delta_c, site) - guide.downstream(gb, site)
        shared_candidate = guide.downstream(v.b.patch(cb, cd[site], masks[site]), site) - guide.downstream(cb, site)
        delta_error = shared_base - reference
        base_error = shared_candidate - shared_base
        suffix_error = ce[site] - shared_candidate
        residual = ce[site] - reference - (delta_error + base_error + suffix_error)
        gp, cp = prefix_jacobians(guide, x)[site], prefix_jacobians(candidate, x)[site]
        if site == 0:
            block = guide.blocks[1]
            before = block.up(gb)
            disagreement = ((before > 0) != (block.up(cb) > 0)).float().mean(1)
            crossing = ((before > 0) != (block.up(v.b.patch(gb, gd[site], masks[site])) > 0)).float().mean(1)
            suffix_jacobian = torch.matmul(guide.head.weight[None], block_jacobian(block, gb))
        else:
            disagreement, crossing = torch.zeros(len(x)), torch.zeros(len(x))
            suffix_jacobian = guide.head.weight[None].expand(len(x), -1, -1)
        linear_effect = torch.bmm(suffix_jacobian, delta_g[:, :, None])[:, :, 0]
        arrays = {'reference_effect': reference.numpy(), 'candidate_effect': ce[site].numpy(),
                  'reference_norm': reference.norm(dim=1).numpy(),
                  'relative_error': v.b.relative_rows(ce[site].numpy(), reference.numpy()),
                  'secant_relative_error': v.b.relative_rows(delta_c.numpy(), delta_g.numpy()),
                  'guide_base': gb.numpy(), 'candidate_base': cb.numpy(),
                  'guide_donor': gd[site].numpy(), 'candidate_donor': cd[site].numpy(),
                  'guide_prefix_jacobian': gp.numpy(), 'candidate_prefix_jacobian': cp.numpy(),
                  'delta_error': delta_error.numpy(), 'base_error': base_error.numpy(), 'suffix_error': suffix_error.numpy(),
                  'decomposition_residual': residual.numpy(), 'base_boundary_disagreement': disagreement.numpy(),
                  'intervention_boundary_crossing': crossing.numpy(), 'guide_linear_effect': linear_effect.numpy()}
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('nonfinite derivative diagnosis')
    return arrays, summarize(arrays)


def decide(rows):
    primary = [r for r in rows if r['arm'] == 'relu_global' and r['stage'] == 'block_1' and r['site'] == 0]
    supporting = []
    valid = len(primary) == 3 and len({r['case'] for r in primary}) == 3 and all(r['metrics']['decomposition_max_absolute_residual'] <= 1e-5 for r in rows)
    for row in primary:
        m = row['metrics']
        if (m['resolved_fraction'] >= .9 and m['weak_error_excess'] >= .10
            and ((m['secant_error_correlation'] is not None and m['secant_error_correlation'] >= .30)
                 or m['prefix_jacobian_normalized_rmse'] >= .15)):
            supporting.append(row['case'])
    return {'supporting_guides': supporting, 'decomposition_control_passed': valid,
            'outcome_status': 'DIAGNOSTIC_SUPPORTS_ENDPOINT_OR_DERIVATIVE_HYPOTHESIS' if valid and len(supporting) >= 2 else 'DIAGNOSTIC_DOES_NOT_SUPPORT_CONTINUATION'}


def load_model(case, arm=None, stage=None):
    model = v.b.Model('relu')
    name = 'guide' if arm is None else arm + '-' + stage
    if arm is not None:
        model.blocks[0] = v.b.Block('relu', 96)
        if stage != 'block_1':
            model.blocks[1] = v.b.Block('relu', 96)
    with np.load(SOURCE / case / (name + '.npz'), allow_pickle=False) as archive:
        model.load_state_dict({key: torch.from_numpy(archive[key].copy()) for key in archive.files})
    return model.eval().requires_grad_(False)


def run(role):
    if role not in ('primary', 'replica'):
        raise ValueError('diagnostic role contract')
    packet = verify()
    runtime = v.b.runtime()
    if role == 'replica':
        original = json.loads((BASE / 'primary/result.json').read_text())
        if v.artifact_manifest(BASE / 'primary') != original['artifact_manifest']:
            raise ValueError('primary diagnostic custody changed')
        if any(original['runtime'][key] == runtime[key] for key in ('sys_prefix', 'site_packages')):
            raise ValueError('diagnostic replica runtime not segregated')
    root = BASE / role
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        for member in (*MEMBERS, str(FREEZE.relative_to(REPO))):
            target = root / 'frozen-packet' / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / member, target)
        rows = []
        for seed in v.DEV_SEEDS:
            case = 'relu-' + str(seed)
            with np.load(SOURCE / case / 'preparation-data.npz', allow_pickle=False) as data:
                x, donor = [torch.from_numpy(data[key].copy()) for key in ('adapt', 'donor')]
                masks = [torch.from_numpy(data['mask' + str(i)].copy()) for i in range(2)]
            guide = load_model(case)
            for arm in v.VARIANTS:
                for stage in v.b.STAGES:
                    candidate = load_model(case, arm, stage)
                    if stage == 'block_1' and not all(torch.equal(p, candidate.state_dict()[key]) for key, p in guide.state_dict().items() if key.startswith(('blocks.1.', 'head.', 'stem.'))):
                        raise ValueError('shared suffix/prefix positive control changed')
                    for site in range(1 if stage == 'block_1' else 2):
                        arrays, metrics = analyze_site(guide, candidate, x, donor, masks, site)
                        if stage == 'block_1' and np.max(np.abs(arrays['suffix_error'])) > 1e-6:
                            raise ValueError('identical suffix produced nonzero decomposition term')
                        name = case + '-' + arm + '-' + stage + '-site' + str(site)
                        np.savez(root / (name + '.npz'), **arrays)
                        rows.append({'case': case, 'arm': arm, 'stage': stage, 'site': site,
                                     'array_sha256': v.b.array_hash(arrays), 'metrics': metrics})
                        resource = v.b.check_budget(start, root)
                        if resource['wall_seconds'] > 300:
                            raise RuntimeError('diagnostic 300-second cap exceeded')
            print(json.dumps({'case': case, 'event': 'adaptation_diagnosis_complete'}), flush=True)
        decision = decide(rows)
        if verify() != packet:
            raise ValueError('diagnostic changed during execution')
        result = {'state_slice': SLICE, 'packet_sha256': packet, 'claim_ceiling': CLAIM,
                  'assurance_level': 'INTERNAL_PASS', 'reviewer_status': 'NOT_SUBMITTED',
                  'rows': rows, 'decision': decision, 'runtime': runtime,
                  'resources': v.b.check_budget(start, root), 'input_sha256': v.b.hash_value(input_material()),
                  'artifact_manifest': v.artifact_manifest(root), 'challenge_evaluation': False, 'optimizer_updates': 0}
        v.b.write_json(root / 'result.json', result)
        if role == 'replica':
            original = json.loads((BASE / 'primary/result.json').read_text())
            checks = {key: original[key] == result[key] for key in ('state_slice', 'packet_sha256', 'claim_ceiling', 'rows', 'decision', 'input_sha256', 'challenge_evaluation', 'optimizer_updates', 'artifact_manifest')}
            checks['distinct_runtime'] = all(original['runtime'][key] != runtime[key] for key in ('sys_prefix', 'site_packages'))
            checks['toolchain'] = {k: a for k, a in runtime.items() if k not in ('sys_prefix', 'site_packages')} == {k: a for k, a in original['runtime'].items() if k not in ('sys_prefix', 'site_packages')}
            receipt = {'checks': checks, 'assurance_level': 'REPRODUCED_LOCAL' if all(checks.values()) else 'INCONCLUSIVE',
                       'same_owner_and_host': True, 'reviewer_status': 'NOT_SUBMITTED',
                       'primary_sha256': v.b.file_hash(BASE / 'primary/result.json'), 'replica_sha256': v.b.file_hash(root / 'result.json')}
            v.b.write_json(root / 'reconciliation.json', receipt)
            if not all(checks.values()):
                raise ValueError('diagnostic reproduction mismatch')
        print(json.dumps({'role': role, 'decision': decision, 'result_sha256': v.b.file_hash(root / 'result.json')}), flush=True)
        return result
    except Exception as error:
        v.b.write_json(root / 'error.json', {'state_slice': SLICE, 'packet_sha256': packet, 'outcome_status': 'INCONCLUSIVE',
                                           'error_type': type(error).__name__, 'error': str(error)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'run', 'verify'))
    parser.add_argument('--role', choices=('primary', 'replica'), default='primary')
    args = parser.parse_args()
    if args.action == 'run':
        run(args.role)
    else:
        print(json.dumps({'packet_sha256': freeze() if args.action == 'freeze' else verify()}))
