"""Whole-MLP adapters and derivative supervision for the V36 transfer study.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36,
global-experiment-assurance-lifecycle-v1. No model or prompt I/O at import.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn


FAMILIES = ('opt_relu', 'gpt2_gelu', 'llama_silu_gated')
METHODS = ('jvp', 'endpoints')
SEEDS = (36101, 36102, 36103)
STRATA = ('ordinary', 'input_composition', 'interchange_composition', 'adversarial')
PROTOCOL = 'theseus-pretrained-whole-block-transfer-v36'
SLICE = 'theseus-pretrained-whole-block-transfer-v36'
CLAIM = 'LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY'
UPDATES = 1024
BATCH = 64
PROBES = 4


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(canonical([PROTOCOL, *parts])).digest()[:8], 'big') % (2**31 - 1)


def array_digest(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        h.update(canonical([key, value.dtype.str, list(value.shape)]))
        h.update(value.tobytes())
    return h.hexdigest()


def state(module):
    return {k: v.detach().cpu().numpy().copy() for k, v in module.state_dict().items()}


def layers(model, family):
    if family == 'opt_relu':
        return model.model.decoder.layers
    if family == 'gpt2_gelu':
        return model.transformer.h
    if family == 'llama_silu_gated':
        return model.model.layers
    raise ValueError('unknown decoder family')


class OptMLP(nn.Module):
    """Read-only functional view of both OPT projections and activation."""
    def __init__(self, layer):
        super().__init__()
        self.fc1, self.fc2, self.activation = layer.fc1, layer.fc2, layer.activation_fn

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


def original_mlp(model, family, site):
    layer = layers(model, family)[site]
    return OptMLP(layer) if family == 'opt_relu' else layer.mlp


def endpoints(model, family, site):
    layer = layers(model, family)[site]
    return (layer.fc1, layer.fc2) if family == 'opt_relu' else (layer.mlp, layer.mlp)


@contextlib.contextmanager
def installed(model, family, replacements):
    """Replace the full MLP graph temporarily; restore exact original objects."""
    saved = []
    try:
        for site, module in sorted(replacements.items()):
            layer = layers(model, family)[site]
            if family == 'opt_relu':
                saved.append((layer, (layer.fc1, layer.fc2, layer.activation_fn)))
                layer.fc1, layer.fc2, layer.activation_fn = module, nn.Identity(), nn.Identity()
            else:
                saved.append((layer, layer.mlp))
                layer.mlp = module
        yield model
    finally:
        for layer, previous in reversed(saved):
            if family == 'opt_relu':
                layer.fc1, layer.fc2, layer.activation_fn = previous
            else:
                layer.mlp = previous


class Replacement(nn.Module):
    """Independently initialized full MLP; data-fitted output scale only."""
    def __init__(self, width, family, seed, mean=None, scale=None):
        super().__init__()
        if family not in FAMILIES or width < 1:
            raise ValueError('replacement dimensions/family')
        self.family, self.width = family, width
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.up = nn.Linear(width, 2 * width)
            self.down = nn.Linear(2 * width, width)
        self.register_buffer('mean', torch.zeros(width) if mean is None else mean.detach().clone())
        self.register_buffer('scale', torch.ones(width) if scale is None else scale.detach().clone())

    def forward(self, x):
        z = self.up(x)
        z = torch.relu(z) if self.family == 'opt_relu' else torch.nn.functional.gelu(z)
        return self.mean + self.scale * self.down(z)


class Zero(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def normalized_mse(actual, target):
    return (actual - target).square().mean() / target.square().mean().clamp_min(1e-8)


def norm(loss, parameters):
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    value = sum(float(g.detach().double().square().sum()) for g in grads if g is not None)
    if not math.isfinite(value):
        raise ValueError('nonfinite gradient')
    return math.sqrt(value)


def jvp(module, x, directions):
    """Directional module derivatives, preserving candidate parameter gradients."""
    return torch.stack([torch.func.jvp(module, (x,), (d,))[1] for d in directions], dim=1)


def fit(teacher, x, donor, family, seed, method, budget=lambda: None, updates=UPDATES):
    if method not in METHODS or x.shape != donor.shape or x.ndim != 2 or not len(x):
        raise ValueError('supervision contract')
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError('teacher must be frozen/eval')
    gen = torch.Generator().manual_seed(seed)
    count, width = x.shape
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[torch.arange(count), torch.randint(width, (count,), generator=gen)] = True
    patched = torch.where(mask, donor, x)
    directions = (torch.randint(2, (PROBES, count, width), generator=gen).float() * 2 - 1) / math.sqrt(width)
    targets = {}
    for key, value in (('x', x), ('donor', donor), ('patched', patched)):
        budget()
        with torch.no_grad():
            targets[key] = teacher(value).detach()
            targets[key + '_jvp'] = jvp(teacher, value, directions).detach() if key != 'patched' else None
    scale = torch.sqrt(targets['x'].var(dim=0, unbiased=False).clamp_min(1e-6))
    model = Replacement(width, family, seed, targets['x'].mean(dim=0), scale)
    if {id(p) for p in model.parameters()} & {id(p) for p in teacher.parameters()}:
        raise ValueError('replacement aliases teacher parameters')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    parameters, readouts = tuple(model.parameters()), []
    for step in range(updates):
        if step % 16 == 0:
            budget()
        index = torch.randperm(count, generator=gen)[:min(BATCH, count)]
        a, b, p = model(x[index]), model(donor[index]), model(patched[index])
        base = .5 * (normalized_mse(a, targets['x'][index]) + normalized_mse(b, targets['donor'][index]))
        base = base + normalized_mse(p - a, targets['patched'][index] - targets['x'][index])
        auxiliary = base * 0
        if method == 'jvp':
            auxiliary = .5 * (normalized_mse(jvp(model, x[index], directions[:, index]), targets['x_jvp'][index])
                              + normalized_mse(jvp(model, donor[index], directions[:, index]), targets['donor_jvp'][index]))
        nb, na = norm(base, parameters), norm(auxiliary, parameters)
        coefficient = min(1., .5 * nb / max(na, 1e-12)) if method == 'jvp' else 1.
        if coefficient * na > .5 * nb + 1e-8:
            raise ValueError('auxiliary cap failed')
        readouts.append((nb, na, coefficient))
        optimizer.zero_grad(set_to_none=True)
        (base + coefficient * auxiliary).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
    model.eval().requires_grad_(False)
    return model, np.asarray(readouts, dtype='float64')


def forward(model, family, ids, sites, intervention=None, capture_all=False):
    """Full decoder evaluation with one native interface intervention.

    Intervention tuple: (site, input/output, replacement-vector). Positions
    are the final prompt token; module inputs may be flattened by OPT.
    """
    if intervention is None:
        interventions = []
    elif isinstance(intervention[0], int):
        interventions = [intervention]
    else:
        interventions = intervention
    if len({(s, k) for s, k, _ in interventions}) != len(interventions):
        raise ValueError('duplicate interface intervention')
    if any(s not in sites or k not in ('input','output') for s,k,_ in interventions):
        raise ValueError('unknown intervention site or interface')
    handles, captured = [], {}
    for site in sites:
        start, end = endpoints(model, family, site)

        def pre(module, args, site=site):
            value = args[0]
            for target, kind, replacement in interventions:
                if (target, kind) == (site, 'input'):
                    value = value.clone()
                    value.reshape(-1, value.shape[-1])[-1] = replacement
            captured.setdefault(site, {})['x'] = value.reshape(-1, value.shape[-1]).detach().clone() if capture_all else value.reshape(-1, value.shape[-1])[-1].detach().clone()
            return (value, *args[1:])

        def post(module, args, output, site=site):
            for target, kind, replacement in interventions:
                if (target, kind) == (site, 'output'):
                    output = output.clone()
                    output.reshape(-1, output.shape[-1])[-1] = replacement
            captured.setdefault(site, {})['y'] = output.reshape(-1, output.shape[-1]).detach().clone() if capture_all else output.reshape(-1, output.shape[-1])[-1].detach().clone()
            return output

        handles.extend((start.register_forward_pre_hook(pre), end.register_forward_hook(post)))
    try:
        with torch.no_grad():
            tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
            output = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits[0]
            nll = torch.nn.functional.cross_entropy(output[:-1], tokens[0, 1:]).item()
            return {'logits': output[-1].cpu().numpy().copy(), 'nll': nll, 'sites': captured}
    finally:
        for handle in handles:
            handle.remove()


def intervention_vector(base, donor, count, seed, adversarial=False):
    width = base.numel()
    if count > width or count < 1:
        raise ValueError('intervention width')
    index = torch.randperm(width, generator=torch.Generator().manual_seed(seed))[:count]
    result = base.clone()
    result[index] = donor[index] if not adversarial else base[index] + 1.4 * (donor[index] - base[index])
    return result


def effect_metrics(actual, reference):
    norms = np.linalg.norm(reference, axis=1)
    mask = norms >= 1e-4
    if not mask.any():
        return {'resolved_fraction': 0., 'median_error': None, 'p90_error': None, 'median_cosine': None, 'passed': False}
    a, r = actual[mask].astype('float64'), reference[mask].astype('float64')
    relative = np.linalg.norm(a-r, axis=1) / np.maximum(np.linalg.norm(r, axis=1), 1e-6)
    cosine = np.sum(a*r, axis=1) / np.maximum(np.linalg.norm(a, axis=1)*np.linalg.norm(r, axis=1), 1e-12)
    m = {'resolved_fraction': float(mask.mean()), 'median_error': float(np.median(relative)), 'p90_error': float(np.quantile(relative, .9)), 'median_cosine': float(np.median(cosine))}
    m['passed'] = m['resolved_fraction'] >= .9 and m['median_error'] <= .2 and m['p90_error'] <= .5 and m['median_cosine'] >= .9
    return m


def nrmse(a, b):
    return float(np.sqrt(np.mean((a.astype('float64')-b)**2) / max(float(np.mean(b.astype('float64')**2)), 1e-8)))


def score(predicted, oracle, rescues, active):
    """Native interface scores; no internal hidden-unit alignment claim."""
    cells = {}
    for stratum in STRATA:
        prefix = stratum + '/'
        p, o = predicted[prefix+'logits'], oracle[prefix+'logits']
        m = {'output_nrmse': nrmse(p, o), 'top1_agreement': float(np.mean(p.argmax(1)==o.argmax(1))),
             'nll_increase': float(np.mean(predicted[prefix+'nll']-oracle[prefix+'nll'])),
             'effects': {}, 'activation': {}, 'rescue': {}, 'rescue_prediction': {}}
        m['joint_effect'] = effect_metrics(predicted[prefix+'joint_effect'], oracle[prefix+'joint_effect'])
        for site in active:
            s = str(site)
            m['effects'][s] = effect_metrics(predicted[prefix+'effect/'+s], oracle[prefix+'effect/'+s])
            m['activation'][s] = nrmse(predicted[prefix+'activation/'+s], oracle[prefix+'activation/'+s])
            m['rescue'][s] = nrmse(rescues[prefix+s], o)
            # Candidate's factual decoder output is its prebound prediction of
            # guide restoration; this shortcut is declared, never oracle tuned.
            m['rescue_prediction'][s] = nrmse(p, rescues[prefix+s])
        m['passed'] = bool(m['output_nrmse']<=.1 and m['top1_agreement']>=.95 and m['nll_increase']<=.03
                           and m['joint_effect']['passed'] and all(e['passed'] for e in m['effects'].values())
                           and max(m['activation'].values())<=.15 and max(m['rescue'].values())<=.1
                           and max(m['rescue_prediction'].values())<=.1)
        cells[stratum] = m
    return cells
