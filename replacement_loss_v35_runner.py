"""Frozen paired endpoint, secant, and Jacobian supervision development.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v35-secant-jacobian-v1,
global-experiment-assurance-lifecycle-v1. Raw artifacts stay in owner custody.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import resource
import secrets
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import replacement_loss_v32_runner as b
import replacement_loss_v34_runner as parent
import replacement_loss_v34_derivative_audit as diagnosis

REPO = Path(__file__).resolve().parent
BASE = Path('/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v35')
PROTOCOL = 'theseus-replacement-loss-v35-secant-jacobian'
SLICE = 'theseus-replacement-loss-v35-secant-jacobian-v1'
CLAIM = 'LOCAL_SYNTHETIC_ENDPOINT_SECANT_AND_JACOBIAN_SUPERVISION_ONLY'
VARIANTS = ('global', 'paired_endpoints', 'paired_secant', 'paired_jacobian')
CONFIRM_ARMS = ('selected_causal', 'baseline_causal_global', 'paired_endpoints', 'pointwise', 'output', 'cka')
BALANCE_RATIO = .5
DEV_SEEDS = (35101, 35102, 35103)
CONFIRM_SEEDS = (35201, 35202, 35203)
OWN_MEMBERS = ('replacement_loss_v35_runner.py', 'replacement_loss_v35_closeout.py', 'tests/test_replacement_loss_v35.py',
               'requirements-v35.lock.txt', 'docs/replacement-loss-v35-secant-jacobian.md',
               'docs/replacement-loss-v35-input-exclusion.json')
MEMBERS = (*diagnosis.MEMBERS, 'docs/replacement-loss-v34-derivative-freeze.json', *OWN_MEMBERS)
FREEZE = REPO / 'docs/replacement-loss-v35-freeze.json'
CONFIRM = REPO / 'docs/replacement-loss-v35-confirmation-freeze.json'
EXCLUSION = REPO / 'docs/replacement-loss-v35-input-exclusion.json'
DEADLINE = float('inf')


def material():
    return [{'path': p, 'bytes': (REPO / p).stat().st_size, 'sha256': b.file_hash(REPO / p)} for p in MEMBERS]


def verify_freeze():
    parent.verify_freeze()
    diagnosis.verify()
    f = json.loads(FREEZE.read_text())
    m = material()
    if (f['packet_files'] != m or f['packet_sha256'] != b.hash_value(m)
        or f['protocol_id'] != PROTOCOL or f['state_slice'] != SLICE
        or f['claim_ceiling'] != CLAIM or f['freeze_state'] != 'FROZEN_BEFORE_DEVELOPMENT_DATA'):
        raise ValueError('frozen V35 packet mismatch')
    return f['packet_sha256']


def verify_diagnostic():
    packet = diagnosis.verify()
    primary_root, replica_root = diagnosis.BASE / 'primary', diagnosis.BASE / 'replica'
    primary = json.loads((primary_root / 'result.json').read_text())
    replica = json.loads((replica_root / 'result.json').read_text())
    reconciliation = json.loads((replica_root / 'reconciliation.json').read_text())
    validation = json.loads((diagnosis.BASE / 'closeout/validation.json').read_text())
    if (primary['decision']['outcome_status'] != 'DIAGNOSTIC_SUPPORTS_ENDPOINT_OR_DERIVATIVE_HYPOTHESIS'
        or primary['decision'] != replica['decision']
        or reconciliation['assurance_level'] != 'REPRODUCED_LOCAL'
        or not all(reconciliation['checks'].values()) or validation['assurance_level'] != 'REPRODUCED_LOCAL'
        or validation['packet_sha256'] != packet or not all(validation['checks'].values())):
        raise ValueError('adaptation diagnostic did not qualify and reproduce')
    for role, root in [('primary', primary_root), ('replica', replica_root)]:
        result = primary if role == 'primary' else replica
        if parent.artifact_manifest(root) != result['artifact_manifest'] or validation['result_digests'][role] != b.file_hash(root / 'result.json'):
            raise ValueError('diagnostic custody or validation binding changed')
    return {'packet_sha256': packet, 'primary_sha256': b.file_hash(primary_root / 'result.json'),
            'replica_sha256': b.file_hash(replica_root / 'result.json'),
            'reconciliation_sha256': b.file_hash(replica_root / 'reconciliation.json'),
            'validation_sha256': b.file_hash(diagnosis.BASE / 'closeout/validation.json')}


def freeze_initial():
    if FREEZE.exists() or EXCLUSION.exists() or BASE.exists():
        raise FileExistsError('V35 freeze or custody already exists')
    parent_packet = parent.verify_freeze()
    parent.verify_receipt(parent.BASE / 'development/primary')
    parent.verify_receipt(parent.BASE / 'development/replica')
    diagnostic = verify_diagnostic()
    prior_root = parent.BASE / 'development/primary'
    paths = sorted(prior_root.glob('*/preparation-data.npz')) + sorted(prior_root.glob('*/map-validation-inputs.npz')) + [prior_root / 'challenge.npz']
    if len(paths) != 7 or not all(p.is_file() for p in paths):
        raise ValueError('V34 prior input inventory incomplete')
    entries = [{'path': str(p.relative_to(prior_root)), 'bytes': p.stat().st_size, 'sha256': b.file_hash(p)} for p in paths]
    b.write_json(EXCLUSION, {'parent_packet_sha256': parent_packet,
                            'parent_result_sha256': b.file_hash(prior_root / 'result.json'),
                            'parent_exclusion_sha256': b.file_hash(parent.EXCLUSION),
                            'diagnostic': diagnostic, 'files': entries})
    m = material()
    b.write_json(FREEZE, {'protocol_id': PROTOCOL, 'state_slice': SLICE,
                         'state_slices': ['theseus-lab-sandbox-v1', SLICE, 'global-experiment-assurance-lifecycle-v1'],
                         'revision': '202f459d6d89df9a038d78d58e0af825d287fcd8',
                         'freeze_state': 'FROZEN_BEFORE_DEVELOPMENT_DATA', 'claim_ceiling': CLAIM,
                         'nonclaims': b.NONCLAIMS, 'monetary_cap_usd': 0,
                         'packet_files': m, 'packet_sha256': b.hash_value(m)})
    return {'packet_sha256': verify_freeze(), 'freeze_sha256': b.file_hash(FREEZE)}

def data_rows(path):
    with np.load(path, allow_pickle=False) as data:
        return {row.tobytes() for key in data.files
                if key in ('train', 'visible', 'adapt', 'donor', 'align_fit', 'align_validate')
                or key.rsplit('/', 1)[-1] in ('x', 'donor', 'counterfactual')
                for row in data[key]}


def exclude_rows(phase):
    manifest = json.loads(EXCLUSION.read_text())
    if (manifest['parent_packet_sha256'] != parent.verify_freeze()
        or manifest['parent_result_sha256'] != b.file_hash(parent.BASE / 'development/primary/result.json')
        or manifest['parent_exclusion_sha256'] != b.file_hash(parent.EXCLUSION)
        or manifest['diagnostic'] != verify_diagnostic()):
        raise ValueError('V35 parent or diagnostic lineage changed')
    seen = parent.exclude_rows('development')
    custody = parent.BASE / 'development/primary'
    for entry in manifest['files']:
        path = custody / entry['path']
        if not path.is_relative_to(custody) or b.file_hash(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('V34 prior input exclusion changed')
        seen.update(data_rows(path))
    if phase == 'confirmation':
        verify_receipt(BASE / 'development/primary')
        prior_root = BASE / 'development/primary'
        paths = sorted(prior_root.glob('*/preparation-data.npz')) + sorted(prior_root.glob('*/map-validation-inputs.npz')) + [prior_root / 'challenge.npz']
        for path in paths:
            seen.update(data_rows(path))
    return seen

def stream_seed(phase, family, seed, partition):
    """Unique declared stream identity; never add small offsets to adjacent seeds."""
    import hashlib
    label = f'{PROTOCOL}/{phase}/{family}/{seed}/{partition}'
    return int(hashlib.sha256(label.encode()).hexdigest()[:12], 16)


def ensure_fresh(arrays, seen):
    rows = [row.tobytes() for a in arrays for row in a]
    current = set(rows)
    if len(current) != len(rows) or current & seen:
        raise ValueError('fixture row overlap')
    seen.update(current)


def budget(step=0):
    if step % 50 == 0:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss * 1024
        if time.monotonic() > DEADLINE or rss > b.MAX_RSS:
            raise RuntimeError('V35 frozen training budget exceeded')


def gradient_norm(loss, parameters):
    """Float64 Euclidean norm, detached; extra calls do not accumulate .grad."""
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    squares = [g.detach().double().square().sum() for g in gradients if g is not None]
    norm = float(torch.stack(squares).sum().sqrt()) if squares else 0.
    if not math.isfinite(norm):
        raise ValueError('nonfinite component gradient')
    return norm


def coefficient(anchor_norm, effect_norm, balanced):
    if not all(math.isfinite(v) and v >= 0 for v in (anchor_norm, effect_norm)):
        raise ValueError('gradient-norm contract')
    return min(1., BALANCE_RATIO * anchor_norm / max(effect_norm, 1e-12)) if balanced else 1.


def train_guide(seed, family, x, steps=b.GUIDE_STEPS):
    torch.manual_seed(seed)
    guide = b.Model(family)
    optimizer = torch.optim.AdamW(guide.parameters(), lr=.002, weight_decay=.0001)
    rng = np.random.default_rng(seed + 1)
    targets = b.truth(x)
    for step in range(steps):
        budget(step)
        ids = rng.integers(0, len(x), 256)
        optimizer.zero_grad(set_to_none=True)
        F.mse_loss(guide(x[ids])[0], targets[ids]).backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.)
        optimizer.step()
    return guide.eval().requires_grad_(False)


def auxiliary_secant(actual, target, scale):
    return ((actual - target).square() / scale).mean()


def fit(guide, method, geometry, weighting, seed, x, donor, masks,
        stage_steps=b.STAGE_STEPS, final_steps=b.FINAL_STEPS):
    if method in ('causal', 'pointwise', 'output', 'cka'):
        stages, diagnostics, traces = parent.fit(guide, method, geometry, weighting, seed, x, donor, masks, stage_steps, final_steps)
        for diagnostic in diagnostics.values():
            diagnostic['norm_component_role'] = 'anchor_vs_causal' if method == 'causal' else 'none'
        return stages, diagnostics, traces
    if method not in ('causal_pair', 'causal_secant', 'causal_jacobian') or weighting != 'global':
        raise ValueError('unknown V35 supervision method')
    candidate = copy.deepcopy(guide)
    with torch.no_grad():
        teacher, reference_states, reference_effects = b.effects(guide, x, donor, masks)
        reference_donor = guide(donor)[1]
        targets = b.truth(x)
        reference_jacobians = diagnosis.prefix_jacobians(guide, x)
        donor_jacobians = diagnosis.prefix_jacobians(guide, donor)
        secant_scales = [(d - a).square().mean(0).clamp_min(1e-6) for a, d in zip(reference_states, reference_donor)]
    snapshots, diagnostics, traces = {}, {}, {}
    for stage, name in enumerate(b.STAGES):
        if stage < 2:
            torch.manual_seed(seed + 100 + stage)
            candidate.blocks[stage] = b.Block(geometry, 96)
        active = min(stage + 1, 2)
        candidate.requires_grad_(False)
        if stage == 2:
            candidate.requires_grad_(True)
        else:
            for block in candidate.blocks[:active]:
                block.requires_grad_(True)
        parameters = tuple(p for p in candidate.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed + 1000 + stage)
        readouts = []
        steps = final_steps if stage == 2 else stage_steps
        for step in range(steps):
            budget(step)
            ids = rng.integers(0, len(x), 128)
            optimizer.zero_grad(set_to_none=True)
            output, states = candidate(x[ids])
            source_states = candidate(donor[ids])[1]
            base = b.normalized_mse(output, teacher[ids])
            base = base + sum(.5 * (b.normalized_mse(states[i], reference_states[i][ids])
                                    + b.normalized_mse(source_states[i], reference_donor[i][ids])) for i in range(active)) / active
            base = base + sum(b.normalized_mse(candidate.downstream(b.patch(states[i], source_states[i], masks[i][ids]), i) - output,
                                              reference_effects[i][ids]) for i in range(active)) / active
            if stage == 2:
                base = base + F.mse_loss(output, targets[ids])
            if method == 'causal_secant':
                auxiliary = sum(auxiliary_secant(source_states[i] - states[i], reference_donor[i][ids] - reference_states[i][ids],
                                                 secant_scales[i]) for i in range(active)) / active
            elif method == 'causal_jacobian':
                jacobians = diagnosis.prefix_jacobians(candidate, x[ids])
                donor_actual = diagnosis.prefix_jacobians(candidate, donor[ids])
                auxiliary = sum(.5 * (b.normalized_mse(jacobians[i], reference_jacobians[i][ids])
                                      + b.normalized_mse(donor_actual[i], donor_jacobians[i][ids])) for i in range(active)) / active
            else:
                auxiliary = base * 0
            ga, ge = gradient_norm(base, parameters), gradient_norm(auxiliary, parameters)
            balanced = method != 'causal_pair'
            scale = coefficient(ga, ge, balanced)
            if balanced and scale * ge > BALANCE_RATIO * ga + 1e-8:
                raise ValueError('V35 auxiliary gradient cap violated')
            readouts.append([ga, ge, scale, ge / max(ga, 1e-12), scale * ge / max(ga, 1e-12)])
            loss = base + scale * auxiliary
            if not torch.isfinite(loss):
                raise ValueError('nonfinite V35 training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
        candidate.eval().requires_grad_(False)
        snapshots[name] = copy.deepcopy(candidate)
        trace = np.asarray(readouts, dtype='float64').reshape(-1, 5)
        traces[name] = trace
        diagnostics[name] = {'updates': steps, 'norm_readouts': len(trace), 'balanced': method != 'causal_pair',
                             'norm_component_role': 'base_vs_auxiliary', 'supervision_method': method,
                             'coefficient_quantiles': [float(a) for a in np.quantile(trace[:, 2], [0, .5, 1])],
                             'raw_ratio_max': float(trace[:, 3].max()), 'weighted_ratio_max': float(trace[:, 4].max()),
                             'trace_sha256': b.array_hash({'readouts': trace})}
    return snapshots, diagnostics, traces

def nrmse(a, target):
    return float(np.sqrt(np.mean((a - target)**2) / max(float(np.mean(target**2)), 1e-8)))


def fit_map(a, g, av, gv):
    """Fit and qualify one invertible affine map using adaptation inputs only."""
    a, g, av, gv = [np.asarray(x, dtype=np.float64) for x in (a, g, av, gv)]
    if not all(x.ndim == 2 and x.shape[1] == 16 and np.isfinite(x).all() for x in (a, g, av, gv)) or a.shape != g.shape or av.shape != gv.shape:
        raise ValueError('map shape/finite contract')
    ac, gc = a - a.mean(0), g - g.mean(0)
    matrix = np.linalg.solve(ac.T @ ac / len(a) + 1e-6 * np.eye(16), ac.T @ gc / len(a))
    bias = g.mean(0) - a.mean(0) @ matrix
    condition = float(np.linalg.cond(matrix))
    inverse = None
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        pass
    finite = np.isfinite(matrix).all() and np.isfinite(bias).all() and inverse is not None and np.isfinite(inverse).all()
    validation = nrmse(av @ matrix + bias, gv)
    roundtrip = nrmse((av @ matrix) @ inverse, av) if finite else None
    metrics = {'validation_normalized_rmse': validation,
               'condition_number': condition if math.isfinite(condition) else None,
               'roundtrip_normalized_rmse': roundtrip,
               'qualified': bool(finite and condition <= 1e4 and validation <= .15 and roundtrip <= 1e-4)}
    # A failed inverse is explicitly absent, never replaced by a pseudoinverse.
    arrays = {'matrix': matrix, 'bias': bias}
    if inverse is not None:
        arrays['inverse'] = inverse
    if not all(np.isfinite(x).all() for x in arrays.values()) or not math.isfinite(validation):
        raise ValueError('nonfinite alignment fit')
    return arrays, metrics



def map_queries(phase, family, seed, rows=128):
    """Separate adaptation-only transport checks, generated before model fitting."""
    queries = {}
    for index, stratum in enumerate(b.STRATA):
        bound = 1.4 if stratum == 'adversarial' else 1.
        x = b.draw(stream_seed(phase, family, seed, 'map/' + stratum + '/x'), rows, bound)
        donor = b.draw(stream_seed(phase, family, seed, 'map/' + stratum + '/donor'), rows, bound)
        rng = np.random.default_rng(stream_seed(phase, family, seed, 'map/' + stratum + '/coordinates'))
        cf = x.clone()
        positions = rng.permuted(np.tile(np.arange(8), (rows, 1)), axis=1)[:, :2]
        for row, coords in enumerate(positions):
            cf[row, coords] = donor[row, coords]
        queries[stratum] = {'x': x, 'donor': donor, 'counterfactual': cf,
                            'masks': b.masks(rng, rows, (1, 2, 4, 8)[index])}
    return queries


def qualify_map_site(guide, model, mapping, queries, site):
    metrics = {}
    with torch.no_grad():
        for name, q in queries.items():
            reference, guide_states, guide_effects = b.effects(guide, q['x'], q['donor'], q['masks'])
            output, states = model(q['x'])
            donor_states = model(q['donor'])[1]
            mapped = states[site].numpy().astype('float64') @ mapping['matrix'] + mapping['bias']
            mapped_donor = donor_states[site].numpy().astype('float64') @ mapping['matrix'] + mapping['bias']
            patched = np.where(q['masks'][site].numpy(), mapped_donor, mapped)
            original = (patched - mapping['bias']) @ mapping['inverse']
            effect = (model.downstream(torch.from_numpy(original.astype('float32')), site) - output).numpy()
            restored = guide.downstream(torch.from_numpy(mapped.astype('float32')), site).numpy()
            prediction = model.downstream(states[site], site).numpy()
            causal = b.causal_metrics(effect, guide_effects[site].numpy())
            activation = nrmse(mapped, guide_states[site].numpy())
            rescue = nrmse(restored, reference.numpy())
            prediction_error = nrmse(prediction, restored)
            metrics[name] = {'effect': causal, 'activation_normalized_rmse': activation,
                             'learned_rescue_normalized_rmse': rescue, 'rescue_prediction_normalized_rmse': prediction_error,
                             'passed': bool(causal['passed'] and activation <= .15 and rescue <= .1 and prediction_error <= .1)}
    return metrics


def calibrate(guide, model, x, validation, active, queries):
    with torch.no_grad():
        gs, gv = guide(x)[1], guide(validation)[1]
        cs, cv = model(x)[1], model(validation)[1]
    maps, qualifications = {}, {}
    for site in range(active):
        mapping, metrics = fit_map(cs[site].numpy(), gs[site].numpy(), cv[site].numpy(), gv[site].numpy())
        representation = metrics['qualified']
        causal = qualify_map_site(guide, model, mapping, queries, site) if representation else None
        metrics['representation_qualified'] = representation
        metrics['causal_validation'] = causal
        metrics['causal_qualified'] = causal is not None and all(cell['passed'] for cell in causal.values())
        metrics['qualified'] = bool(representation and metrics['causal_qualified'])
        maps[str(site)], qualifications[str(site)] = mapping, metrics
    return maps, qualifications

def map_arrays(maps):
    return {site + '/' + name: a for site, mapping in maps.items() for name, a in mapping.items()}


def aligned_predictions(model, queries, maps, qualifications, active):
    if not all(qualifications[str(i)]['qualified'] for i in range(active)):
        return None
    arrays = b.predicted(model, queries)
    with torch.no_grad():
        for name, q in queries.items():
            out, states = model(q['x'])
            donor = model(q['donor'])[1]
            for site in range(active):
                mapping = maps[str(site)]
                a = states[site].numpy().astype('float64') @ mapping['matrix'] + mapping['bias']
                d = donor[site].numpy().astype('float64') @ mapping['matrix'] + mapping['bias']
                patched = np.where(q['masks'][site].numpy(), d, a)
                inverse = (patched - mapping['bias']) @ mapping['inverse']
                effect = model.downstream(torch.from_numpy(inverse.astype('float32')), site) - out
                arrays[name + '/activation' + str(site)] = a.astype('float32')
                arrays[name + '/effect' + str(site)] = effect.numpy()
                # Candidate suffix uses its original coordinates for restoration.
                arrays[name + '/rescue' + str(site)] = model.downstream(states[site], site).numpy()
    return arrays


def gate_risk(scores):
    ratios = []
    for cells in scores.values():
        for cell in cells.values():
            ratios += [cell['output_normalized_rmse'] / .1]
            for key, threshold in [('activation_errors', .15), ('learned_rescue_errors', .1), ('rescue_prediction_errors', .1)]:
                ratios.extend(v / threshold for v in cell[key].values())
            for effect in [cell['input_effect'], *cell['internal_effects'].values()]:
                if any(effect[key] is None for key in ('median_error', 'p90_error', 'median_cosine')) or effect['median_cosine'] <= 0:
                    return None
                ratios += [effect['median_error'] / .2, effect['p90_error'] / .5, .9 / effect['median_cosine']]
    return max(ratios)


def select_development(cases):
    eligible, risks = [], {}
    for arm in VARIANTS:
        if all(case['arms'][arm]['passed'] for case in cases.values()):
            values = [gate_risk(case['arms'][arm]['native']) for case in cases.values()]
            if all(v is not None for v in values):
                risks[arm] = max(values)
                eligible.append(arm)
    chosen = min(eligible, key=lambda arm: (risks[arm], VARIANTS.index(arm))) if eligible else None
    return {'eligible_variants': eligible, 'eligible_worst_gate_risks': risks, 'selected_variant': chosen}


def arm_configs(phase, family, selected=None):
    geometry = 'relu' if family == 'relu' else 'gelu'
    methods = {'global': 'causal', 'paired_endpoints': 'causal_pair', 'paired_secant': 'causal_secant', 'paired_jacobian': 'causal_jacobian'}
    if phase == 'development':
        return {arm: (method, 'relu', 'global') for arm, method in methods.items()}
    if selected not in VARIANTS:
        raise ValueError('unknown selected development arm')
    return {'selected_causal': (methods[selected], geometry, 'global'),
            'baseline_causal_global': ('causal', geometry, 'global'),
            'paired_endpoints': ('causal_pair', geometry, 'global'),
            'pointwise': ('pointwise', geometry, 'global'),
            'output': ('output', geometry, 'global'), 'cka': ('cka', geometry, 'global')}

def artifact_manifest(root):
    return [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': b.file_hash(p)}
            for p in sorted(root.rglob('*')) if p.is_file()
            and p.name not in ('result.json', 'reconciliation.json', 'error.json')]


def verify_receipt(root):
    result = json.loads((root / 'result.json').read_text())
    if result['packet_sha256'] != verify_freeze():
        raise ValueError('receipt packet mismatch')
    if artifact_manifest(root) != result['artifact_manifest']:
        raise ValueError('custody artifact receipt changed')
    return result


def freeze_confirmation():
    if CONFIRM.exists() or (BASE / 'confirmation').exists():
        raise FileExistsError('confirmation already frozen or started')
    packet = verify_freeze()
    primary = verify_receipt(BASE / 'development/primary')
    replica = verify_receipt(BASE / 'development/replica')
    reconciliation = json.loads((BASE / 'development/replica/reconciliation.json').read_text())
    selected = (primary['selection'] or {}).get('selected_variant')
    if (selected is None or primary['outcome_status'] != 'DEVELOPMENT_QUALIFIED'
        or replica['selection'] != primary['selection'] or reconciliation['status'] != 'REPRODUCED_LOCAL'
        or not all(reconciliation['checks'].values())):
        raise ValueError('development not eligible and reproduced')
    receipt = {'protocol_id': PROTOCOL, 'parent_packet_sha256': packet,
               'freeze_state': 'FROZEN_BEFORE_CONFIRMATION_DATA', 'selected_variant': selected,
               'development_primary_sha256': b.file_hash(BASE / 'development/primary/result.json'),
               'development_replica_sha256': b.file_hash(BASE / 'development/replica/result.json'),
               'development_reconciliation_sha256': b.file_hash(BASE / 'development/replica/reconciliation.json'),
               'confirmation_seeds': list(CONFIRM_SEEDS), 'families': list(b.FAMILIES),
               'arms': list(CONFIRM_ARMS), 'claim_ceiling': CLAIM}
    b.write_json(CONFIRM, receipt)
    return receipt


def verify_confirmation():
    f = json.loads(CONFIRM.read_text())
    primary = verify_receipt(BASE / 'development/primary')
    replica = verify_receipt(BASE / 'development/replica')
    reconciliation = json.loads((BASE / 'development/replica/reconciliation.json').read_text())
    if (f['protocol_id'] != PROTOCOL or f['parent_packet_sha256'] != verify_freeze()
        or f['freeze_state'] != 'FROZEN_BEFORE_CONFIRMATION_DATA' or f['claim_ceiling'] != CLAIM
        or f['selected_variant'] != primary['selection']['selected_variant']
        or f['selected_variant'] != replica['selection']['selected_variant']
        or f['selected_variant'] != select_development(primary['cases'])['selected_variant']
        or f['confirmation_seeds'] != list(CONFIRM_SEEDS) or f['families'] != list(b.FAMILIES)
        or f['arms'] != list(CONFIRM_ARMS) or reconciliation['status'] != 'REPRODUCED_LOCAL'
        or not all(reconciliation['checks'].values())
        or f['development_primary_sha256'] != b.file_hash(BASE / 'development/primary/result.json')
        or f['development_replica_sha256'] != b.file_hash(BASE / 'development/replica/result.json')
        or f['development_reconciliation_sha256'] != b.file_hash(BASE / 'development/replica/reconciliation.json')):
        raise ValueError('confirmation selection binding mismatch')
    return f


def reconcile(root, result, rt, phase):
    prior_root = BASE / phase / 'primary'
    prior = verify_receipt(prior_root)
    prior_rt = json.loads((prior_root / 'runtime.json').read_text())
    keys = ('protocol_id', 'phase', 'packet_sha256', 'confirmation_freeze_sha256', 'outcome_status',
            'preparation', 'guide_qualification', 'challenge_binding', 'prediction_binding',
            'oracle_binding', 'cases', 'selection', 'challenge_generated')
    checks = {k: b.canonical(prior[k]) == b.canonical(result[k]) for k in keys}
    checks['distinct_runtime'] = rt['sys_prefix'] != prior_rt['sys_prefix'] and rt['site_packages'] != prior_rt['site_packages']
    checks['toolchain'] = {k: v for k, v in rt.items() if k not in ('sys_prefix', 'site_packages')} == {k: v for k, v in prior_rt.items() if k not in ('sys_prefix', 'site_packages')}
    checks['source_packet_unchanged'] = verify_freeze() == result['packet_sha256']
    receipt = {'checks': checks, 'status': 'REPRODUCED_LOCAL' if all(checks.values()) else 'INCONCLUSIVE',
               'same_owner': True, 'independent_review': False,
               'primary_result_sha256': b.file_hash(prior_root / 'result.json'),
               'replica_result_sha256': b.file_hash(root / 'result.json')}
    b.write_json(root / 'reconciliation.json', receipt)
    if not all(checks.values()):
        raise ValueError('V35 reproduction mismatch')
    return receipt


def run(phase, role):
    global DEADLINE
    if phase not in ('development', 'confirmation') or role not in ('primary', 'replica'):
        raise ValueError('phase/role contract')
    packet = verify_freeze()
    rt = b.runtime()
    if (REPO / 'requirements-v35.lock.txt').read_text() != (REPO / 'requirements-v32.lock.txt').read_text():
        raise ValueError('V35 lock differs from runtime validator')
    confirmation = verify_confirmation() if phase == 'confirmation' else None
    if role == 'replica':
        verify_receipt(BASE / phase / 'primary')
        prior_rt = json.loads((BASE / phase / 'primary/runtime.json').read_text())
        if rt['sys_prefix'] == prior_rt['sys_prefix'] or rt['site_packages'] == prior_rt['site_packages']:
            raise ValueError('replica runtime not segregated')
    root = BASE / phase / role
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DEADLINE = start + b.MAX_SECONDS
    try:
        b.write_json(root / 'runtime.json', rt)
        for member in (*MEMBERS, str(FREEZE.relative_to(REPO)), *((str(CONFIRM.relative_to(REPO)),) if confirmation else ())):
            target = root / 'frozen-packet' / member
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / member, target)
        seen = exclude_rows(phase)
        prepared, preparation, qualifications = {}, {}, {}
        families = ('relu',) if phase == 'development' else b.FAMILIES
        seeds = DEV_SEEDS if phase == 'development' else CONFIRM_SEEDS
        selected = confirmation['selected_variant'] if confirmation else None
        for family in families:
            for seed in seeds:
                key, case_seed = family + '-' + str(seed), stream_seed(phase, family, seed, 'guide_optimizer')
                case_root = root / key
                case_root.mkdir()
                fixtures = {name: b.draw(stream_seed(phase, family, seed, name), rows) for name, rows in
                            [('train', 4096), ('visible', 1024), ('adapt', 2048),
                             ('donor', 2048), ('align_fit', 1024), ('align_validate', 1024)]}
                ensure_fresh([v.numpy() for v in fixtures.values()], seen)
                masks = b.masks(np.random.default_rng(stream_seed(phase, family, seed, 'training_masks')), 2048, 1)
                arrays = {name: x.numpy() for name, x in fixtures.items()}
                arrays.update({'mask0': masks[0].numpy(), 'mask1': masks[1].numpy()})
                np.savez(case_root / 'preparation-data.npz', **arrays)
                map_validation = map_queries(phase, family, seed)
                ensure_fresh([q[field].numpy() for q in map_validation.values() for field in ('x', 'donor', 'counterfactual')], seen)
                map_fixture = b.query_arrays(map_validation)
                np.savez(case_root / 'map-validation-inputs.npz', **map_fixture)
                guide = train_guide(case_seed, family, fixtures['train'])
                qualifications[key] = b.qualify(guide, fixtures['visible'])
                preparation[key] = {'fixtures': b.array_hash(arrays), 'map_validation_inputs': b.array_hash(map_fixture),
                                    'guide': b.save_state(case_root, 'guide', guide), 'candidates': {}, 'training_diagnostics': {}}
                prepared[key] = {'guide': guide, 'fixtures': fixtures, 'masks': masks, 'seed': case_seed,
                                 'family': family, 'map_queries': map_validation, 'candidates': {}, 'maps': {}, 'map_qualifications': {}}
                print(json.dumps({'phase': phase, 'case': key, 'event': 'guide_qualification', **qualifications[key]}), flush=True)
                b.check_budget(start, root)
        result = {'protocol_id': PROTOCOL, 'phase': phase, 'state_slice': SLICE,
                  'packet_sha256': packet, 'confirmation_freeze_sha256': b.file_hash(CONFIRM) if confirmation else None,
                  'guide_qualification': qualifications, 'preparation': preparation,
                  'challenge_generated': False, 'challenge_binding': None, 'prediction_binding': None,
                  'oracle_binding': {}, 'cases': {}, 'selection': None, 'reviewer_status': 'NOT_SUBMITTED',
                  'assurance_level': 'INTERNAL_PASS', 'claim_ceiling': CLAIM, 'nonclaims': b.NONCLAIMS}
        if not all(v['passed'] for v in qualifications.values()):
            result['outcome_status'] = 'NO_QUALIFIED_GUIDE'
        else:
            for key, item in prepared.items():
                fixtures = item['fixtures']
                for arm, (method, geometry, weighting) in arm_configs(phase, item['family'], selected).items():
                    stages, training, traces = fit(item['guide'], method, geometry, weighting, item['seed'], fixtures['adapt'], fixtures['donor'], item['masks'])
                    np.savez(root / key / (arm + '-training-diagnostics.npz'), **traces)
                    preparation[key]['training_diagnostics'][arm] = training
                    item['candidates'][arm], item['maps'][arm], item['map_qualifications'][arm] = stages, {}, {}
                    preparation[key]['candidates'][arm] = {}
                    for stage, model in stages.items():
                        active = 1 if stage == 'block_1' else 2
                        maps, metrics = calibrate(item['guide'], model, fixtures['align_fit'], fixtures['align_validate'], active, item['map_queries'])
                        item['maps'][arm][stage], item['map_qualifications'][arm][stage] = maps, metrics
                        mapping = map_arrays(maps)
                        np.savez(root / key / (arm + '-' + stage + '-maps.npz'), **mapping)
                        preparation[key]['candidates'][arm][stage] = {'state': b.save_state(root / key, arm + '-' + stage, model),
                                                                    'maps': b.array_hash(mapping), 'map_qualification': metrics}
                    print(json.dumps({'phase': phase, 'case': key, 'arm': arm, 'event': 'candidate_and_maps_frozen'}), flush=True)
                    b.check_budget(start, root)
            candidate_binding = {'packet_sha256': packet, 'confirmation_freeze_sha256': result['confirmation_freeze_sha256'],
                                 'preparation': preparation, 'guide_qualification': qualifications}
            b.write_json(root / 'candidate-freeze.json', candidate_binding)
            if role == 'replica':
                prior_binding = json.loads((BASE / phase / 'primary/candidate-freeze.json').read_text())
                if b.canonical(candidate_binding) != b.canonical(prior_binding):
                    raise ValueError('replica candidates/maps differ before challenge opening')
                entropy = json.loads((BASE / phase / 'primary/challenge-binding.json').read_text())['entropy']
            else:
                entropy = secrets.token_hex(32)
            queries = b.challenge(entropy)
            ensure_fresh([q[field].numpy() for q in queries.values() for field in ('x', 'donor', 'counterfactual')], seen)
            query_arrays = b.query_arrays(queries)
            np.savez(root / 'challenge.npz', **query_arrays)
            challenge_binding = {'entropy': entropy, 'fixture_sha256': b.array_hash(query_arrays), 'candidate_freeze_sha256': b.hash_value(candidate_binding)}
            b.write_json(root / 'challenge-binding.json', challenge_binding)
            predictions, prediction_digests = {}, {}
            for key, item in prepared.items():
                predictions[key], prediction_digests[key] = {'arms': {}, 'controls': {}}, {'arms': {}, 'controls': {}}
                for arm, stages in item['candidates'].items():
                    predictions[key]['arms'][arm], prediction_digests[key]['arms'][arm] = {}, {}
                    for stage, model in stages.items():
                        active = 1 if stage == 'block_1' else 2
                        native = b.predicted(model, queries)
                        aligned = aligned_predictions(model, queries, item['maps'][arm][stage], item['map_qualifications'][arm][stage], active)
                        predictions[key]['arms'][arm][stage] = {'native': native, 'aligned': aligned}
                        prediction_digests[key]['arms'][arm][stage] = {}
                        for mode, arrays in [('native', native), ('aligned', aligned)]:
                            prediction_digests[key]['arms'][arm][stage][mode] = b.array_hash(arrays) if arrays is not None else 'ALIGNMENT_UNAVAILABLE'
                            if arrays is not None:
                                np.savez(root / key / (arm + '-' + stage + '-' + mode + '-predictions.npz'), **arrays)
                    _, geometry, _ = arm_configs(phase, item['family'], selected)[arm]
                    torch.manual_seed(item['seed'] + 300)
                    random = copy.deepcopy(item['guide'])
                    random.blocks = nn.ModuleList([b.Block(geometry, 96), b.Block(geometry, 96)])
                    zero = copy.deepcopy(item['guide'])
                    zero.blocks = nn.ModuleList([b.Zero(), b.Zero()])
                    wrong = copy.deepcopy(stages['block_2'])
                    wrong.blocks = nn.ModuleList([copy.deepcopy(wrong.blocks[1]), copy.deepcopy(wrong.blocks[0])])
                    predictions[key]['controls'][arm], prediction_digests[key]['controls'][arm] = {}, {}
                    for name, control in [('random', random), ('zero_module', zero), ('wrong_module', wrong)]:
                        arrays = b.predicted(control, queries)
                        predictions[key]['controls'][arm][name] = arrays
                        prediction_digests[key]['controls'][arm][name] = b.array_hash(arrays)
                        np.savez(root / key / (arm + '-' + name + '-predictions.npz'), **arrays)
            prediction_binding = {'packet_sha256': packet, 'challenge_sha256': b.hash_value(challenge_binding), 'predictions': prediction_digests}
            b.write_json(root / 'prediction-freeze.json', prediction_binding)
            if role == 'replica':
                prior_prediction = json.loads((BASE / phase / 'primary/prediction-freeze.json').read_text())
                if b.canonical(prior_prediction) != b.canonical(prediction_binding):
                    raise ValueError('replica predictions differ before oracle evaluation')
            # The proposer is sealed; only now evaluate guide challenge outcomes.
            cases, oracle_binding = {}, {}
            for key, item in prepared.items():
                guide = item['guide']
                oracle = b.predicted(guide, queries)
                np.savez(root / key / 'oracle-outcomes.npz', **oracle)
                rescue_arrays = {}

                def score_with_rescue(arrays, label, active):
                    rescues = {}
                    with torch.no_grad():
                        for name in b.STRATA:
                            for site in range(active):
                                field = name + '/rescue' + str(site)
                                h = torch.from_numpy(arrays[name + '/activation' + str(site)])
                                rescues[field] = guide.downstream(h, site).numpy()
                                rescue_arrays[label + '/' + field] = rescues[field]
                    return b.score(arrays, oracle, queries, active, rescues)

                positive = b.score(oracle, oracle, queries, 2)
                arms = {}
                for arm, stages in predictions[key]['arms'].items():
                    native, aligned = {}, {}
                    for stage, modes in stages.items():
                        active = 1 if stage == 'block_1' else 2
                        native[stage] = score_with_rescue(modes['native'], arm + '/' + stage + '/native', active)
                        aligned[stage] = (score_with_rescue(modes['aligned'], arm + '/' + stage + '/aligned', active)
                                          if modes['aligned'] is not None else None)
                    controls = {name: score_with_rescue(arrays, arm + '/' + name, 2) for name, arrays in predictions[key]['controls'][arm].items()}
                    sensitivity = {}
                    for stratum in b.STRATA:
                        treatment = native['block_2'][stratum]['input_effect']['median_error']
                        negatives = [cells[stratum]['input_effect']['median_error'] for cells in controls.values()]
                        sensitivity[stratum] = bool(treatment is not None and all(v is not None and v >= treatment + .10 for v in negatives))
                    passed = all(c['passed'] for cells in native.values() for c in cells.values()) and all(sensitivity.values()) and all(c['passed'] for c in positive.values())
                    arms[arm] = {'native': native, 'aligned': aligned, 'controls': controls, 'control_sensitivity': sensitivity,
                                 'native_retention_passed': all(c['passed'] for cells in native.values() for c in cells.values()),
                                 'aligned_status': 'AVAILABLE' if all(v is not None for v in aligned.values()) else 'ALIGNMENT_UNAVAILABLE',
                                 'aligned_retention_passed': all(c['passed'] for cells in aligned.values() for c in cells.values()) if all(v is not None for v in aligned.values()) else None,
                                 'passed': bool(passed), 'worst_gate_risk': gate_risk(native)}
                np.savez(root / key / 'rescue-outcomes.npz', **rescue_arrays)
                oracle_binding[key] = {'guide': b.array_hash(oracle), 'rescues': b.array_hash(rescue_arrays)}
                cases[key] = {'arms': arms, 'positive_controls': positive}
                print(json.dumps({'phase': phase, 'case': key, 'event': 'evaluated', 'arm_passes': {a: v['passed'] for a, v in arms.items()}}), flush=True)
            result.update({'cases': cases, 'oracle_binding': oracle_binding, 'challenge_generated': True,
                           'challenge_binding': challenge_binding, 'prediction_binding': prediction_binding})
            if phase == 'development':
                result['selection'] = select_development(cases)
                result['outcome_status'] = 'DEVELOPMENT_QUALIFIED' if result['selection']['selected_variant'] else 'NO_QUALIFIED_DEVELOPMENT_CANDIDATE'
            else:
                result['selection'] = {'selected_variant': selected}
                result['outcome_status'] = 'PASS' if all(c['arms']['selected_causal']['passed'] for c in cases.values()) else 'FAIL'
        if verify_freeze() != packet:
            raise ValueError('packet changed during run')
        if confirmation:
            verify_confirmation()
        result['resources'] = b.check_budget(start, root)
        result['artifact_manifest'] = artifact_manifest(root)
        b.write_json(root / 'result.json', result)
        b.check_budget(start, root)
        if role == 'replica':
            reconcile(root, result, rt, phase)
        print(json.dumps({'phase': phase, 'role': role, 'outcome_status': result['outcome_status'],
                          'selection': result['selection'], 'result_sha256': b.file_hash(root / 'result.json')}), flush=True)
        return result
    except Exception as error:
        b.write_json(root / 'error.json', {'protocol_id': PROTOCOL, 'phase': phase, 'role': role,
                                         'packet_sha256': packet, 'outcome_status': 'INCONCLUSIVE',
                                         'error_type': type(error).__name__, 'error': str(error),
                                         'claim_ceiling': CLAIM, 'reviewer_status': 'NOT_SUBMITTED'})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'freeze-confirmation', 'development', 'confirmation', 'verify'))
    parser.add_argument('--role', choices=('primary', 'replica'), default='primary')
    args = parser.parse_args()
    if args.action == 'freeze':
        print(json.dumps(freeze_initial(), sort_keys=True))
    elif args.action == 'freeze-confirmation':
        print(json.dumps(freeze_confirmation(), sort_keys=True))
    elif args.action == 'verify':
        print(json.dumps({'packet_sha256': verify_freeze()}))
    else:
        run(args.action, args.role)


if __name__ == '__main__':
    main()
