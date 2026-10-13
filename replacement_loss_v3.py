"""Core primitives for replacement-loss-v3.

State slice: theseus-lab-sandbox-v1.

This module fits a newly constructed affine replacement from adaptation-only
input/output pairs and scores sealed causal-effect predictions. It never reads
the removed module's parameters. Checkpoint loading, challenge custody, and
blind role separation are owned by the frozen external runner packet.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import torch
from torch import nn


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v3"


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _module_digest(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(bytes.fromhex(_tensor_digest(tensor)))
    return digest.hexdigest()


def fit_independent_linear_replacement(
    in_features: int,
    out_features: int,
    adaptation_inputs: torch.Tensor,
    adaptation_targets: torch.Tensor,
    *,
    use_bias: bool,
    seed: int,
    ridge: float = 1e-6,
) -> tuple[nn.Linear, dict[str, Any]]:
    """Fit a fresh linear module using only disjoint adaptation examples.

    Only dimensions and bias presence are accepted from the architecture
    adapter. No target module object or parameter is passed to this fitter. A
    deterministic random module is constructed, then fitted by a closed-form
    ridge solve in whichever of the primal or dual spaces is smaller. Inputs
    and targets must be captured at the selected subsystem boundary by the
    caller.
    """

    if adaptation_inputs.ndim != 2 or adaptation_targets.ndim != 2:
        raise ValueError("adaptation inputs and targets must be rank-2 matrices")
    if adaptation_inputs.shape[0] != adaptation_targets.shape[0]:
        raise ValueError("adaptation input and target rows must be paired")
    if adaptation_inputs.shape[0] < 2:
        raise ValueError("at least two adaptation rows are required")
    if adaptation_inputs.shape[1] != in_features:
        raise ValueError("adaptation input width does not match replacement")
    if adaptation_targets.shape[1] != out_features:
        raise ValueError("adaptation target width does not match replacement")
    if ridge <= 0:
        raise ValueError("ridge must be positive")

    device = adaptation_inputs.device
    dtype = adaptation_inputs.dtype
    if adaptation_targets.device != device:
        raise ValueError("adaptation inputs and targets must share a device")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        replacement = nn.Linear(
            in_features,
            out_features,
            bias=use_bias,
            device=device,
            dtype=dtype,
        )
    initialized_digest = _module_digest(replacement)

    x = adaptation_inputs.to(dtype=torch.float64)
    y = adaptation_targets.to(dtype=torch.float64)
    if use_bias:
        mean_x = x.mean(dim=0)
        mean_y = y.mean(dim=0)
        fit_x = x - mean_x
        fit_y = y - mean_y
    else:
        mean_x = torch.zeros(x.shape[1], dtype=x.dtype, device=x.device)
        mean_y = torch.zeros(y.shape[1], dtype=y.dtype, device=y.device)
        fit_x = x
        fit_y = y
    if fit_x.shape[1] <= fit_x.shape[0]:
        gram = fit_x.T @ fit_x
        gram.diagonal().add_(ridge)
        fitted_weight = torch.linalg.solve(gram, fit_x.T @ fit_y).T
        solver = "primal_ridge_least_squares"
    else:
        gram = fit_x @ fit_x.T
        gram.diagonal().add_(ridge)
        coefficients = torch.linalg.solve(gram, fit_y)
        fitted_weight = coefficients.T @ fit_x
        solver = "dual_ridge_least_squares"
    fitted_bias = mean_y - mean_x @ fitted_weight.T

    with torch.no_grad():
        replacement.weight.copy_(fitted_weight.to(device=device, dtype=dtype))
        if replacement.bias is not None:
            replacement.bias.copy_(fitted_bias.to(device=device, dtype=dtype))
    return replacement, {
        "method": solver,
        "seed": int(seed),
        "ridge": float(ridge),
        "adaptation_row_count": int(adaptation_inputs.shape[0]),
        "initialization_digest": initialized_digest,
        "fitted_replacement_digest": _module_digest(replacement),
    }


def linear_intervention_prediction(
    replacement: nn.Linear,
    base_input: torch.Tensor,
    intervention: torch.Tensor,
) -> torch.Tensor:
    """Predict the replacement's module-boundary effect for one intervention."""

    if base_input.shape != intervention.shape:
        raise ValueError("base input and intervention must have identical shapes")
    if base_input.shape[-1] != replacement.in_features:
        raise ValueError("intervention width does not match replacement")
    with torch.no_grad():
        before = replacement(base_input)
        after = replacement(base_input + intervention)
    return after - before


def causal_effect_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    resolution_floor: float,
    scale: float,
) -> dict[str, float | bool]:
    """Score a sealed predicted vector against an evaluator-only outcome."""

    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    if predicted.size == 0 or predicted.shape != observed.shape:
        raise ValueError("predicted and observed effects must be non-empty and paired")
    if resolution_floor < 0 or scale <= 0:
        raise ValueError("resolution floor must be non-negative and scale must be positive")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("effect vectors must be finite")
    predicted_norm = float(np.linalg.norm(predicted))
    observed_norm = float(np.linalg.norm(observed))
    if observed_norm <= resolution_floor:
        return {
            "resolvable": False,
            "direction_correct": False,
            "cosine_similarity": 0.0,
            "normalized_error": 0.0,
        }
    denominator = max(float(scale), resolution_floor)
    dot = float(np.dot(predicted, observed))
    cosine = dot / max(predicted_norm * observed_norm, 1e-30)
    return {
        "resolvable": True,
        "direction_correct": dot > 0.0,
        "cosine_similarity": float(np.clip(cosine, -1.0, 1.0)),
        "normalized_error": float(np.linalg.norm(predicted - observed) / denominator),
    }


def bootstrap_mean_difference_lower_bound(
    treatment: np.ndarray,
    control: np.ndarray,
    *,
    seed: int,
    draws: int = 10_000,
    quantile: float = 0.025,
) -> float:
    """Cluster-level paired bootstrap lower bound for treatment minus control."""

    treatment = np.asarray(treatment, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    if treatment.ndim != 1 or treatment.shape != control.shape or treatment.size < 2:
        raise ValueError("paired cluster scores must be equal one-dimensional arrays")
    if draws < 1 or not 0.0 < quantile < 1.0:
        raise ValueError("draws and quantile are outside their valid ranges")
    paired = treatment - control
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, paired.size, size=(draws, paired.size))
    resampled_means = paired[indices].mean(axis=1)
    return float(np.quantile(resampled_means, quantile))


def prediction_commitment_digest(predictions: dict[str, Any]) -> str:
    """Hash canonical prediction rows before evaluator outcomes are opened."""

    payload = json.dumps(predictions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
