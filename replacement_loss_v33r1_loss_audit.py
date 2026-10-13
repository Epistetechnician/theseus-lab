"""Read-only adaptation gradient-scale audit; no training or heldout tuning.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v33r1-loss-scale-audit-v1,
global-experiment-assurance-lifecycle-v1.
"""
import copy
import json
import shutil
import time

import numpy as np
import torch

import replacement_loss_v33r1_runner as v

SLICE = 'theseus-replacement-loss-v33r1-loss-scale-audit-v1'
FREEZE = v.REPO / 'docs/replacement-loss-v33r1-loss-audit-freeze.json'
MEMBERS = ('replacement_loss_v33r1_loss_audit.py', 'docs/replacement-loss-v33r1-loss-audit.md')


def members():
    return [{'path': p, 'sha256': v.b.file_hash(v.REPO / p), 'bytes': (v.REPO / p).stat().st_size} for p in MEMBERS]


def run():
    start = time.monotonic()
    runtime = v.b.runtime()
    source = v.BASE / 'development/primary'
    replica = v.BASE / 'development/replica'
    receipt = v.verify_receipt(source)
    v.verify_receipt(replica)
    reconciliation = json.loads((replica / 'reconciliation.json').read_text())
    if reconciliation['status'] != 'REPRODUCED_LOCAL' or not all(reconciliation['checks'].values()):
        raise ValueError('parent development not reproduced')
    material = members()
    binding = {'state_slice': SLICE, 'parent_packet_sha256': v.verify_freeze(),
               'development_result_sha256': v.b.file_hash(source / 'result.json'),
               'files': material, 'packet_sha256': v.b.hash_value(material),
               'freeze_state': 'FROZEN_BEFORE_ADAPTATION_GRADIENT_READOUT'}
    if FREEZE.exists():
        if json.loads(FREEZE.read_text()) != binding:
            raise ValueError('loss audit freeze changed')
    else:
        v.b.write_json(FREEZE, binding)
    root = v.BASE / 'loss-scale-audit'
    root.mkdir(exist_ok=False)
    for member in (*MEMBERS, str(FREEZE.relative_to(v.REPO))):
        target = root / 'frozen-packet' / member
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v.REPO / member, target)
    rows = []
    for key in receipt['cases']:
        seed = int(key.rsplit('-', 1)[1])
        case_seed = v.stream_seed('development', 'relu', seed, 'guide_optimizer')
        guide = v.b.Model('relu')
        with np.load(source / key / 'guide.npz', allow_pickle=False) as state:
            guide.load_state_dict({name: torch.from_numpy(state[name].copy()) for name in state.files})
        guide.eval().requires_grad_(False)
        ids = np.random.default_rng(case_seed + 1000).integers(0, 2048, 128)
        with np.load(source / key / 'preparation-data.npz', allow_pickle=False) as data:
            x, donor = [torch.from_numpy(data[name][ids].copy()) for name in ('adapt', 'donor')]
            masks = [torch.from_numpy(data['mask' + str(i)][ids].copy()) for i in range(2)]
        with torch.no_grad():
            reference, reference_states, reference_effects = v.b.effects(guide, x, donor, masks)
        for geometry in ('gelu', 'relu'):
            candidate = copy.deepcopy(guide)
            torch.manual_seed(case_seed + 100)
            candidate.blocks[0] = v.b.Block(geometry, 96)
            output, states, effects = v.b.effects(candidate, x, donor, masks)
            losses = {'output': v.b.normalized_mse(output, reference),
                      'activation': v.b.normalized_mse(states[0], reference_states[0]),
                      'global_effect': v.effect_loss(effects[0], reference_effects[0], 'global'),
                      'row_effect': v.effect_loss(effects[0], reference_effects[0], 'row')}
            losses['anchor'] = losses['output'] + losses['activation']
            parameters = tuple(candidate.blocks[0].parameters())
            gradients = {name: np.concatenate([g.detach().numpy().reshape(-1).astype('float64')
                                               for g in torch.autograd.grad(loss, parameters, retain_graph=True)])
                         for name, loss in losses.items()}
            norms = {name: float(np.linalg.norm(g)) for name, g in gradients.items()}
            targets = reference_effects[0].square().sum(1).numpy()
            rows.append({'case': key, 'geometry': geometry,
                         'first_batch_input_sha256': v.b.array_hash({'x': x.numpy(), 'donor': donor.numpy(), 'mask': masks[0].numpy()}),
                         'initial_candidate_sha256': v.b.array_hash(v.b.state_arrays(candidate)),
                         'losses': {name: float(loss.detach()) for name, loss in losses.items()},
                         'gradient_norms': norms,
                         'global_effect_to_anchor_gradient_ratio': norms['global_effect'] / max(norms['anchor'], 1e-12),
                         'row_effect_to_anchor_gradient_ratio': norms['row_effect'] / max(norms['anchor'], 1e-12),
                         'row_to_global_effect_gradient_ratio': norms['row_effect'] / max(norms['global_effect'], 1e-12),
                         'target_norm_squared_quantiles': [float(a) for a in np.quantile(targets, [0, .1, .5, .9, 1])],
                         'row_floor_fraction': float((targets <= 1e-6).mean())})
    result = {'state_slice': SLICE, 'assurance_level': 'INTERNAL_PASS', 'reviewer_status': 'NOT_SUBMITTED',
              'reproducibility_status': 'NOT_RUN_SEPARATELY', 'claim_ceiling': 'INITIAL_ADAPTATION_GRADIENT_SCALE_ONLY',
              'binding': binding, 'rows': rows, 'monetary_cost_usd': 0, 'energy': 'NOT_MEASURED',
              'runtime': runtime, 'resources': v.b.check_budget(start, root),
              'limitation': 'Read-only first-batch diagnostic, not a training-causation result or confirmation evidence.'}
    if members() != material or v.b.file_hash(source / 'result.json') != binding['development_result_sha256']:
        raise ValueError('loss audit input changed')
    v.verify_receipt(source)
    v.b.write_json(root / 'result.json', result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    run()
