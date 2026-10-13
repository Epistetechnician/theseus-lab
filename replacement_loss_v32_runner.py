"""Digest-bound local, trained nonlinear replacement ladder.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v32-causal-ladder-v1,
global-experiment-assurance-lifecycle-v1. Raw data stay in external custody.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
os.umask(0o077)
REPO = Path(__file__).resolve().parent
PROTOCOL = 'theseus-replacement-loss-v32-causal-ladder'
SLICE = 'theseus-replacement-loss-v32-causal-ladder-v1'
BASE = Path('/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v32')
FAMILIES = ('relu', 'gelu', 'silu_gated')
SEEDS = (3201, 3202, 3203)
METHODS = ('causal', 'pointwise', 'logit', 'cka')
STRATA = ('ordinary', 'input_composition', 'interchange_composition', 'adversarial')
STAGES = ('block_1', 'block_2', 'final_task')
GUIDE_STEPS, STAGE_STEPS, FINAL_STEPS = 2400, 1400, 600
TRAIN_ROWS, ADAPT_ROWS, VISIBLE_ROWS, CHALLENGE_ROWS = 4096, 2048, 1024, 256
MAX_SECONDS, MAX_RSS, MAX_BYTES = 1800, 4 * 1024**3, 1024**3
CLAIM = 'LOCAL_SYNTHETIC_TRAINED_NONLINEAR_LADDER_CAUSAL_RETENTION_ONLY'
NONCLAIMS = ['pretrained transformer replacement', 'all architectures', 'novelty priority', 'metaphysical identity', 'consciousness', 'alignment', 'independent science', 'production readiness']
MEMBERS = ('replacement_loss_v32_runner.py', 'tests/test_replacement_loss_v32.py', 'requirements-v32.lock.txt', 'docs/replacement-loss-v32-causal-ladder.md', 'theseus_run_audit_v1.py', 'docs/theseus-all-runs-audit-v1.json', 'docs/theseus-all-runs-audit-v1.md')
DEADLINE = float('inf')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def hash_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def array_hash(arrays):
    h = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(canonical({'name': name, 'dtype': a.dtype.str, 'shape': list(a.shape)}))
        h.update(a.tobytes())
    return h.hexdigest()


def state_arrays(module):
    return {k: v.detach().numpy().copy() for k, v in module.state_dict().items()}


def save_state(root, name, model):
    arrays = state_arrays(model)
    np.savez(root / (name + '.npz'), **arrays)
    return array_hash(arrays)


def packet_material():
    return [{'path': p, 'bytes': (REPO / p).stat().st_size, 'sha256': file_hash(REPO / p)} for p in MEMBERS]


def verify_freeze():
    freeze = json.loads((REPO / 'docs/replacement-loss-v32-freeze.json').read_text())
    material = packet_material()
    if (freeze['packet_files'] != material or freeze['packet_sha256'] != hash_value(material)
        or freeze['freeze_state'] != 'FROZEN_BEFORE_GUIDE_OR_TASK_DATA_GENERATION'
        or freeze['protocol_id'] != PROTOCOL or freeze['state_slice'] != SLICE or freeze['claim_ceiling'] != CLAIM):
        raise ValueError('frozen V32 packet mismatch')
    return freeze['packet_sha256']


def runtime():
    packages = sorted(f"{d.metadata['Name'].lower().replace('_', '-')}=={d.version}" for d in importlib.metadata.distributions())
    expected = sorted((REPO / 'requirements-v32.lock.txt').read_text().splitlines())
    cfg = Path(sys.prefix) / 'pyvenv.cfg'
    isolated = cfg.is_file() and 'include-system-site-packages = false' in cfg.read_text().lower()
    if packages != expected or platform.python_version() != '3.12.13' or not isolated:
        raise ValueError('locked runtime mismatch')
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise ValueError('one-thread deterministic runtime required')
    return {'python': platform.python_version(), 'packages': packages, 'machine': platform.machine(),
            'platform': platform.platform(), 'sys_prefix': sys.prefix,
            'site_packages': str(Path(torch.__file__).resolve().parent.parent), 'system_site_packages': False,
            'threads': 1, 'device': 'cpu', 'dtype': 'float32'}


def loop_budget(step):
    if step % 50 == 0:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss = rss if sys.platform == 'darwin' else rss * 1024
        if time.monotonic() > DEADLINE or rss > MAX_RSS:
            raise RuntimeError('frozen training resource cap exceeded')


def truth(x):
    """Four signed interacting continuous outputs; fully procedural data."""
    return torch.stack((x[:, 0] + x[:, 1] * x[:, 2], x[:, 3] - x[:, 4] * x[:, 5],
                        torch.sin(x[:, 0] + x[:, 3]), x[:, 6] * x[:, 7] + x[:, 1] * x[:, 4]), dim=1)


class Block(nn.Module):
    def __init__(self, family, hidden=64):
        super().__init__()
        self.family = family
        self.up = nn.Linear(16, hidden)
        self.down = nn.Linear(hidden, 16)
        self.gate = nn.Linear(16, hidden) if family == 'silu_gated' else None

    def forward(self, x):
        if self.family == 'relu':
            hidden = F.relu(self.up(x))
        elif self.family == 'gelu':
            hidden = F.gelu(self.up(x))
        elif self.family == 'silu_gated':
            hidden = F.silu(self.gate(x)) * self.up(x)
        else:
            raise ValueError('unknown family')
        return self.down(hidden)


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class Model(nn.Module):
    def __init__(self, family):
        super().__init__()
        self.stem = nn.Linear(8, 16)
        self.blocks = nn.ModuleList([Block(family), Block(family)])
        self.head = nn.Linear(16, 4)

    def forward(self, x):
        h = self.stem(x)
        states = []
        for block in self.blocks:
            h = block(h)
            states.append(h)
        return self.head(h), states

    def downstream(self, h, site):
        for block in self.blocks[site + 1:]:
            h = block(h)
        return self.head(h)


def normalized_mse(actual, target):
    return F.mse_loss(actual, target) / target.square().mean().clamp_min(1e-8)


def cka(a, b):
    # Row normalization followed by centered linear CKA, as in NoT Appendix B.
    a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
    a, b = a - a.mean(0), b - b.mean(0)
    cross = (a.T @ b).square().sum()
    scale = ((a.T @ a).square().sum() * (b.T @ b).square().sum()).sqrt().clamp_min(1e-12)
    return cross / scale


def patch(base, donor, mask):
    if base.shape != donor.shape or mask.shape != base.shape or mask.dtype != torch.bool:
        raise ValueError('interchange shapes/dtype mismatch')
    return torch.where(mask, donor, base)


def effects(model, x, donor, masks):
    output, states = model(x)
    _, donor_states = model(donor)
    interventions = [model.downstream(patch(states[i], donor_states[i], masks[i]), i) for i in range(2)]
    return output, states, [item - output for item in interventions]


def draw(seed, rows, bound=1.):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.uniform(-bound, bound, (rows, 8)).astype('float32'))


def masks(rng, rows, count):
    result = []
    for _ in range(2):
        order = np.argsort(rng.random((rows, 16)), axis=1)[:, :count]
        mask = np.zeros((rows, 16), dtype=bool)
        np.put_along_axis(mask, order, True, axis=1)
        result.append(torch.from_numpy(mask))
    return result


def train_guide(seed, family, x):
    torch.manual_seed(seed)
    guide = Model(family)
    opt = torch.optim.AdamW(guide.parameters(), lr=.002, weight_decay=.0001)
    rng = np.random.default_rng(seed + 1)
    targets = truth(x)
    for step in range(GUIDE_STEPS):
        loop_budget(step)
        ids = rng.integers(0, len(x), 256)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(guide(x[ids])[0], targets[ids])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(guide.parameters(), 1.)
        opt.step()
    guide.eval().requires_grad_(False)
    return guide


def accuracy(actual, target):
    return float(((actual >= 0) == (target >= 0)).float().mean())


def qualify(guide, visible):
    with torch.no_grad():
        target = truth(visible)
        reference = guide(visible)[0]
        acc = accuracy(reference, target)
        error = float(normalized_mse(reference, target).sqrt())
        drops = []
        for site in range(2):
            removed = copy.deepcopy(guide)
            removed.blocks[site] = Zero()
            drops.append(acc - accuracy(removed(visible)[0], target))
    return {'guide_sign_accuracy': acc, 'guide_normalized_rmse': error,
            'block_removal_accuracy_drops': drops,
            'passed': acc >= .95 and error <= .15 and min(drops) >= .20}


def fit(guide, method, seed, x, donor, training_masks):
    candidate = copy.deepcopy(guide)
    with torch.no_grad():
        teacher, teacher_states, teacher_effects = effects(guide, x, donor, training_masks)
        teacher_truth = truth(x)
    snapshots = {}
    # All methods share new module initialization and minibatch order.
    for stage in range(3):
        if stage < 2:
            torch.manual_seed(seed + 100 + stage)
            candidate.blocks[stage] = Block('gelu', 96)
        active = min(stage + 1, 2)
        candidate.requires_grad_(False)
        if stage == 2:
            candidate.requires_grad_(True)
        else:
            for block in candidate.blocks[:active]:
                block.requires_grad_(True)
        optimizer = torch.optim.AdamW((p for p in candidate.parameters() if p.requires_grad), lr=.001, weight_decay=.0001)
        rng = np.random.default_rng(seed + 1000 + stage)
        for step in range(FINAL_STEPS if stage == 2 else STAGE_STEPS):
            loop_budget(step)
            ids = rng.integers(0, len(x), 128)
            optimizer.zero_grad(set_to_none=True)
            output, states = candidate(x[ids])
            if method == 'cka' and stage < 2:
                loss = sum(1 - cka(states[i], teacher_states[i][ids]) for i in range(active)) / active
            elif method == 'cka':
                loss = F.mse_loss(output, teacher_truth[ids])
            else:
                loss = normalized_mse(output, teacher[ids])
                if method in ('pointwise', 'causal'):
                    loss = loss + sum(normalized_mse(states[i], teacher_states[i][ids]) for i in range(active)) / active
                if method == 'causal':
                    _, source_states = candidate(donor[ids])
                    causal = []
                    for site in range(active):
                        intervened = candidate.downstream(patch(states[site], source_states[site], training_masks[site][ids]), site)
                        causal.append(normalized_mse(intervened - output, teacher_effects[site][ids]))
                    loss = loss + sum(causal) / active
                if stage == 2:
                    loss = loss + F.mse_loss(output, teacher_truth[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), 1.)
            optimizer.step()
        candidate.eval().requires_grad_(False)
        snapshots[STAGES[stage]] = copy.deepcopy(candidate)
    return snapshots


def relative_rows(actual, reference):
    return np.linalg.norm(actual - reference, axis=1) / np.maximum(np.linalg.norm(reference, axis=1), 1e-6)


def causal_metrics(actual, reference):
    norms = np.linalg.norm(reference, axis=1)
    resolved = norms >= 1e-4
    if not resolved.any():
        return {'resolved_fraction': 0., 'median_error': None, 'p90_error': None, 'median_cosine': None, 'passed': False}
    a, r = actual[resolved], reference[resolved]
    errors = relative_rows(a, r)
    cosine = (a * r).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(r, axis=1), 1e-12)
    med, p90, cos = float(np.median(errors)), float(np.quantile(errors, .9)), float(np.median(cosine))
    return {'resolved_fraction': float(resolved.mean()), 'median_error': med, 'p90_error': p90, 'median_cosine': cos,
            'passed': resolved.mean() >= .9 and med <= .2 and p90 <= .5 and cos >= .9}


def challenge(entropy):
    if len(entropy) != 64 or len(bytes.fromhex(entropy)) != 32:
        raise ValueError('challenge entropy must be 32 bytes hex')
    result = {}
    for index, name in enumerate(STRATA):
        seed = int(hashlib.sha256((entropy + name).encode()).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        bound = 1.4 if name == 'adversarial' else 1.
        x = torch.from_numpy(rng.uniform(-bound, bound, (CHALLENGE_ROWS, 8)).astype('float32'))
        donor = torch.from_numpy(rng.uniform(-bound, bound, (CHALLENGE_ROWS, 8)).astype('float32'))
        cf = x.clone()
        coords = rng.permuted(np.tile(np.arange(8), (CHALLENGE_ROWS, 1)), axis=1)[:, :2]
        for row, positions in enumerate(coords):
            cf[row, positions] = donor[row, positions]
        counts = (1, 2, 4, 8)[index]
        result[name] = {'x': x, 'donor': donor, 'counterfactual': cf, 'masks': masks(rng, CHALLENGE_ROWS, counts)}
    return result


def query_arrays(queries):
    arrays = {}
    for name, values in queries.items():
        for field in ('x', 'donor', 'counterfactual'):
            arrays[name + '/' + field] = values[field].numpy()
        for site in range(2):
            arrays[name + '/mask' + str(site)] = values['masks'][site].numpy()
    return arrays


def predicted(model, queries):
    arrays = {}
    with torch.no_grad():
        for name, q in queries.items():
            out, states, eff = effects(model, q['x'], q['donor'], q['masks'])
            arrays[name + '/output'] = out.numpy()
            arrays[name + '/input_effect'] = (model(q['counterfactual'])[0] - out).numpy()
            for site in range(2):
                arrays[name + '/activation' + str(site)] = states[site].numpy()
                arrays[name + '/effect' + str(site)] = eff[site].numpy()
                # Predict restoration with the candidate suffix. The evaluator
                # uses the guide suffix only after all predictions are sealed.
                arrays[name + '/rescue' + str(site)] = model.downstream(states[site], site).numpy()
    return arrays


def score(predictions, oracle, queries, active, actual_rescues=None):
    cells = {}
    for name, q in queries.items():
        key = name + '/'
        output, reference = predictions[key + 'output'], oracle[key + 'output']
        rms = float(np.sqrt(np.mean((output - reference)**2) / max(np.mean(reference**2), 1e-8)))
        sign = float(((output >= 0) == (reference >= 0)).mean())
        ground = truth(q['x']).numpy()
        guide_truth = float(((reference >= 0) == (ground >= 0)).mean())
        candidate_truth = float(((output >= 0) == (ground >= 0)).mean())
        truth_drop = guide_truth - candidate_truth
        effects_by_site = {str(i): causal_metrics(predictions[key + 'effect' + str(i)], oracle[key + 'effect' + str(i)]) for i in range(active)}
        activation, rescue, rescue_prediction = {}, {}, {}
        for site in range(active):
            a, b = predictions[key + 'activation' + str(site)], oracle[key + 'activation' + str(site)]
            activation[str(site)] = float(np.sqrt(np.mean((a - b)**2) / max(np.mean(b**2), 1e-8)))
            rescued = (actual_rescues or predictions)[key + 'rescue' + str(site)]
            rescue[str(site)] = float(np.sqrt(np.mean((rescued - reference)**2) / max(np.mean(reference**2), 1e-8)))
            rescue_prediction[str(site)] = float(np.sqrt(np.mean((predictions[key + 'rescue' + str(site)] - rescued)**2) / max(np.mean(rescued**2), 1e-8)))
        input_causal = causal_metrics(predictions[key + 'input_effect'], oracle[key + 'input_effect'])
        passed = (rms <= .1 and sign >= .95 and truth_drop <= .03 and input_causal['passed']
                  and all(v['passed'] for v in effects_by_site.values())
                  and max(activation.values()) <= .15 and max(rescue.values()) <= .1 and max(rescue_prediction.values()) <= .1)
        cells[name] = {'output_normalized_rmse': rms, 'guide_sign_agreement': sign, 'guide_truth_accuracy': guide_truth,
                       'candidate_truth_accuracy': candidate_truth, 'truth_accuracy_drop': truth_drop,
                       'input_effect': input_causal, 'internal_effects': effects_by_site, 'activation_errors': activation,
                       'learned_rescue_errors': rescue, 'rescue_prediction_errors': rescue_prediction, 'passed': bool(passed)}
    return cells


def check_budget(start, root):
    wall = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss = rss if sys.platform == 'darwin' else rss * 1024
    size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    if wall > MAX_SECONDS or rss > MAX_RSS or size > MAX_BYTES:
        raise RuntimeError('frozen resource cap exceeded')
    return {'wall_seconds': wall, 'peak_rss_bytes': rss, 'artifact_bytes': size, 'monetary_cost_usd': 0, 'energy': 'NOT_MEASURED'}


def run(role):
    global DEADLINE
    packet = verify_freeze()
    rt = runtime()
    root = BASE / role
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    DEADLINE = start + MAX_SECONDS
    write_json(root / 'runtime.json', rt)
    shutil.copyfile(REPO / 'docs/replacement-loss-v32-freeze.json', root / 'freeze.json')
    archive = root / 'frozen-packet'
    for member in MEMBERS:
        target = archive / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / member, target)
    shutil.copyfile(REPO / 'docs/replacement-loss-v32-freeze.json', archive / 'docs/replacement-loss-v32-freeze.json')
    prepared, preparations, guide_qualification = {}, {}, {}
    try:
        for family_index, family in enumerate(FAMILIES):
            for seed in SEEDS:
                key = family + '-' + str(seed)
                case_seed = seed + family_index * 10000
                case_root = root / key
                case_root.mkdir()
                train, visible, adapt, donor = [draw(case_seed + offset, rows) for offset, rows in ((10, TRAIN_ROWS), (11, VISIBLE_ROWS), (12, ADAPT_ROWS), (13, ADAPT_ROWS))]
                # Exact row disjointness, not merely distinct seed labels.
                sets = [{row.tobytes() for row in t.numpy()} for t in (train, visible, adapt, donor)]
                if any(sets[i] & sets[j] for i in range(4) for j in range(i)):
                    raise ValueError('preparation fixture overlap')
                train_masks = masks(np.random.default_rng(case_seed + 14), ADAPT_ROWS, 1)
                fixture = {'train': train.numpy(), 'visible': visible.numpy(), 'adapt': adapt.numpy(), 'donor': donor.numpy(),
                           'mask0': train_masks[0].numpy(), 'mask1': train_masks[1].numpy()}
                np.savez(case_root / 'preparation-data.npz', **fixture)
                guide = train_guide(case_seed, family, train)
                qualification = qualify(guide, visible)
                guide_qualification[key] = qualification
                preparations[key] = {'fixtures': array_hash(fixture), 'guide': save_state(case_root, 'guide', guide), 'candidates': {}}
                print(json.dumps({'case': key, 'phase': 'guide_qualification', **qualification}), flush=True)
                check_budget(start, root)
                prepared[key] = {'guide': guide, 'adapt': adapt, 'donor': donor, 'masks': train_masks, 'seed': case_seed, 'candidates': {}}
        if not all(x['passed'] for x in guide_qualification.values()):
            result = {'protocol_id': PROTOCOL, 'packet_sha256': packet, 'outcome_status': 'NO_QUALIFIED_GUIDE',
                      'assurance_level': 'INTERNAL_PASS', 'guide_qualification': guide_qualification,
                      'preparation': preparations, 'challenge_generated': False, 'reviewer_status': 'NOT_SUBMITTED',
                      'claim_ceiling': CLAIM, 'nonclaims': NONCLAIMS, 'resources': check_budget(start, root)}
            write_json(root / 'result.json', result)
            if role == 'replica':
                primary = json.loads((BASE / 'primary/result.json').read_text())
                prior = json.loads((BASE / 'primary/runtime.json').read_text())
                checks = {key: canonical(primary[key]) == canonical(result[key]) for key in ('protocol_id','packet_sha256','outcome_status','guide_qualification','preparation','challenge_generated')}
                checks['distinct_runtime'] = rt['sys_prefix'] != prior['sys_prefix'] and rt['site_packages'] != prior['site_packages']
                checks['toolchain'] = {k:v for k,v in rt.items() if k not in ('sys_prefix','site_packages')} == {k:v for k,v in prior.items() if k not in ('sys_prefix','site_packages')}
                write_json(root / 'reconciliation.json', {'checks': checks, 'status': 'REPRODUCED_LOCAL' if all(checks.values()) else 'INCONCLUSIVE', 'same_owner': True, 'independent_review': False,
                                                         'primary_result_sha256': file_hash(BASE / 'primary/result.json'), 'replica_result_sha256': file_hash(root / 'result.json')})
            return result
        for key, item in prepared.items():
            for method in METHODS:
                candidates = fit(item['guide'], method, item['seed'], item['adapt'], item['donor'], item['masks'])
                item['candidates'][method] = candidates
                preparations[key]['candidates'][method] = {stage: save_state(root / key, method + '-' + stage, model) for stage, model in candidates.items()}
                print(json.dumps({'case': key, 'method': method, 'phase': 'candidate_frozen'}), flush=True)
                check_budget(start, root)
        preparation_binding = {'packet_sha256': packet, 'preparation': preparations, 'guide_qualification': guide_qualification}
        write_json(root / 'candidate-freeze.json', preparation_binding)
        if role == 'replica':
            primary_binding = json.loads((BASE / 'primary/candidate-freeze.json').read_text())
            if canonical(preparation_binding) != canonical(primary_binding):
                raise ValueError('replica preparation differs before challenge opening')
            primary_runtime = json.loads((BASE / 'primary/runtime.json').read_text())
            if rt['sys_prefix'] == primary_runtime['sys_prefix'] or rt['site_packages'] == primary_runtime['site_packages']:
                raise ValueError('replica runtime not segregated')
            entropy = json.loads((BASE / 'primary/challenge-binding.json').read_text())['entropy']
        else:
            entropy = secrets.token_hex(32)
        queries = challenge(entropy)
        # Verify challenge endpoints against all guide/adaptation/visible rows.
        challenge_rows = {row.tobytes() for q in queries.values() for field in ('x', 'donor', 'counterfactual') for row in q[field].numpy()}
        for key in prepared:
            with np.load(root / key / 'preparation-data.npz') as data:
                earlier = {row.tobytes() for field in ('train', 'visible', 'adapt', 'donor') for row in data[field]}
            if earlier & challenge_rows:
                raise ValueError('challenge preparation overlap')
        fixture_arrays = query_arrays(queries)
        np.savez(root / 'challenge.npz', **fixture_arrays)
        challenge_binding = {'entropy': entropy, 'fixture_sha256': array_hash(fixture_arrays), 'candidate_freeze_sha256': hash_value(preparation_binding)}
        if role == 'replica' and challenge_binding != json.loads((BASE / 'primary/challenge-binding.json').read_text()):
            raise ValueError('challenge reconstruction mismatch')
        write_json(root / 'challenge-binding.json', challenge_binding)
        predictions, predictions_binding = {}, {}
        for key, item in prepared.items():
            guide = item['guide']
            predictions[key], predictions_binding[key] = {}, {}
            for method, stages in item['candidates'].items():
                predictions[key][method], predictions_binding[key][method] = {}, {}
                for stage, model in stages.items():
                    arrays = predicted(model, queries)
                    predictions[key][method][stage] = arrays
                    predictions_binding[key][method][stage] = array_hash(arrays)
                    np.savez(root / key / (method + '-' + stage + '-predictions.npz'), **arrays)
            controls = {}
            torch.manual_seed(item['seed'] + 300)
            random = copy.deepcopy(guide)
            random.blocks = nn.ModuleList([Block('gelu', 96), Block('gelu', 96)])
            zero = copy.deepcopy(guide)
            zero.blocks = nn.ModuleList([Zero(), Zero()])
            wrong = copy.deepcopy(item['candidates']['causal']['block_2'])
            wrong.blocks = nn.ModuleList([copy.deepcopy(wrong.blocks[1]), copy.deepcopy(wrong.blocks[0])])
            for name, control in [('random', random), ('zero_module', zero), ('wrong_module', wrong)]:
                arrays = predicted(control, queries)
                controls[name] = arrays
                predictions_binding[key][name] = array_hash(arrays)
                np.savez(root / key / (name + '-predictions.npz'), **arrays)
            predictions[key]['controls'] = controls
        # All proposer predictions are on disk and bound before oracle scoring.
        prediction_binding = {'packet_sha256': packet, 'challenge_sha256': hash_value(challenge_binding), 'predictions': predictions_binding}
        write_json(root / 'prediction-freeze.json', prediction_binding)
        results, oracle_binding = {}, {}
        for key, item in prepared.items():
            guide = item['guide']
            oracle = predicted(guide, queries)
            np.savez(root / key / 'oracle-outcomes.npz', **oracle)
            rescue_outcomes = {}
            def actual_rescue(arrays, label):
                rescued = {}
                with torch.no_grad():
                    for name in STRATA:
                        for site in range(2):
                            field = name + '/rescue' + str(site)
                            h = torch.from_numpy(arrays[name + '/activation' + str(site)])
                            rescued[field] = guide.downstream(h, site).numpy()
                            rescue_outcomes[label + '/' + field] = rescued[field]
                return rescued
            scores = {}
            for method in METHODS:
                scores[method] = {stage: score(predictions[key][method][stage], oracle, queries, 1 if stage == 'block_1' else 2,
                                              actual_rescue(predictions[key][method][stage], method + '/' + stage)) for stage in STAGES}
            controls = {name: score(arrays, oracle, queries, 2, actual_rescue(arrays, name)) for name, arrays in predictions[key]['controls'].items()}
            np.savez(root / key / 'rescue-outcomes.npz', **rescue_outcomes)
            oracle_binding[key] = {'guide': array_hash(oracle), 'rescues': array_hash(rescue_outcomes)}
            positive = score(oracle, oracle, queries, 2)
            control_gates = {}
            for name in STRATA:
                treatment = scores['causal']['block_2'][name]['input_effect']['median_error']
                negatives = [controls[c][name]['input_effect']['median_error'] for c in ('random', 'zero_module', 'wrong_module')]
                control_gates[name] = bool(treatment is not None and all(v is not None and v >= treatment + .10 for v in negatives))
            retained = all(cell['passed'] for stage in scores['causal'].values() for cell in stage.values())
            passed = retained and all(control_gates.values()) and all(cell['passed'] for cell in positive.values())
            comparisons = {}
            for method in METHODS[1:]:
                pairs = [(scores['causal']['block_2'][name]['input_effect']['median_error'], scores[method]['block_2'][name]['input_effect']['median_error']) for name in STRATA]
                comparisons[method] = {'mean_paired_error_difference_treatment_minus_comparator': float(np.mean([a-b for a,b in pairs])),
                                       'comparator_retention': all(cell['passed'] for stage in scores[method].values() for cell in stage.values())}
            results[key] = {'scores': scores, 'controls': controls, 'unchanged_guide_positive_control': positive,
                            'control_sensitivity': control_gates, 'absolute_retention': retained, 'comparisons': comparisons, 'passed': bool(passed)}
            print(json.dumps({'case': key, 'phase': 'evaluation', 'passed': bool(passed), 'absolute_retention': retained}), flush=True)
        result = {'protocol_id': PROTOCOL, 'state_slices': ['theseus-lab-sandbox-v1', SLICE, 'global-experiment-assurance-lifecycle-v1'],
                  'packet_sha256': packet, 'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                  'assurance_level': 'INTERNAL_PASS', 'reviewer_status': 'NOT_SUBMITTED', 'claim_ceiling': CLAIM, 'nonclaims': NONCLAIMS,
                  'outcome_status': 'PASS' if all(x['passed'] for x in results.values()) else 'FAIL',
                  'guide_qualification': guide_qualification, 'preparation': preparations,
                  'challenge_sha256': hash_value(challenge_binding), 'prediction_sha256': hash_value(prediction_binding),
                  'oracle_array_digests': oracle_binding, 'cases': results, 'resources': check_budget(start, root)}
        verify_freeze()
        write_json(root / 'result.json', result)
        if role == 'replica':
            primary = json.loads((BASE / 'primary/result.json').read_text())
            current = json.loads((root / 'result.json').read_text())
            keys = ('protocol_id', 'packet_sha256', 'revision', 'outcome_status', 'guide_qualification', 'preparation', 'challenge_sha256', 'prediction_sha256', 'oracle_array_digests', 'cases')
            checks = {key: canonical(primary[key]) == canonical(current[key]) for key in keys}
            checks['distinct_runtime'] = rt['sys_prefix'] != primary_runtime['sys_prefix'] and rt['site_packages'] != primary_runtime['site_packages']
            checks['toolchain'] = {k:v for k,v in rt.items() if k not in ('sys_prefix','site_packages')} == {k:v for k,v in primary_runtime.items() if k not in ('sys_prefix','site_packages')}
            write_json(root / 'reconciliation.json', {'checks': checks, 'status': 'REPRODUCED_LOCAL' if all(checks.values()) else 'INCONCLUSIVE',
                                                     'primary_result_sha256': file_hash(BASE / 'primary/result.json'), 'replica_result_sha256': file_hash(root / 'result.json'),
                                                     'same_owner': True, 'independent_review': False})
        return result
    except Exception as error:
        write_json(root / 'failure-receipt.json', {'protocol_id': PROTOCOL, 'packet_sha256': packet, 'outcome_status': 'INCONCLUSIVE',
                                                 'error_type': type(error).__name__, 'error': str(error), 'reviewer_status': 'NOT_SUBMITTED',
                                                 'claim_ceiling': CLAIM, 'nonclaims': NONCLAIMS, 'monetary_cost_usd': 0})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'verify', 'run'))
    parser.add_argument('--role', choices=('primary', 'replica'), default='primary')
    args = parser.parse_args()
    if args.action == 'freeze':
        target = REPO / 'docs/replacement-loss-v32-freeze.json'
        if target.exists() or BASE.exists():
            raise FileExistsError('cannot refreeze or reuse V32 custody')
        material = packet_material()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO)
        write_json(target, {'protocol_id': PROTOCOL, 'packet_files': material, 'packet_sha256': hash_value(material),
                            'freeze_state': 'FROZEN_BEFORE_GUIDE_OR_TASK_DATA_GENERATION', 'guide_generated': False, 'adaptation_generated': False,
                            'challenge_generated': False, 'claim_ceiling': CLAIM, 'nonclaims': NONCLAIMS, 'state_slice': SLICE,
                            'revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
                            'dirty_tree_status_sha256_at_freeze': hashlib.sha256(status).hexdigest(), 'monetary_cap_usd': 0,
                            'max_seconds_per_runtime': MAX_SECONDS, 'max_rss_bytes': MAX_RSS, 'max_artifact_bytes_per_runtime': MAX_BYTES})
        print(json.dumps({'packet_sha256': hash_value(material), 'frozen': True}))
    elif args.action == 'verify':
        print(json.dumps({'packet_sha256': verify_freeze(), 'runtime': runtime()}))
    else:
        result = run(args.role)
        print(json.dumps({'role': args.role, 'outcome_status': result['outcome_status'], 'resources': result['resources']}))


if __name__ == '__main__':
    main()
