"""Paired compositional-support diagnostic for replacement-loss-v4.

State slice: theseus-lab-sandbox-v1.

The sole treatment difference is the stratum of 64 equal-count adaptation
prompts. Candidate fitting is delegated to the digest-pinned v3 linear fitter.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v4-compositional-support"
ACTIVATION_ERROR_MAX = 0.20
MIN_PAIRED_MEDIAN_IMPROVEMENT = 0.10
MAX_NONCOMPOSITIONAL_REGRESSION = 0.05
BEHAVIOR_LOGIT_ERROR_MAX = 0.05
BEHAVIOR_CE_DELTA_MAX = 0.05


def activation_rows(candidate: np.ndarray, original: np.ndarray) -> dict[str, np.ndarray]:
    """Return per-row activation reconstruction metrics."""

    candidate = np.asarray(candidate, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape != original.shape or candidate.shape[0] == 0:
        raise ValueError("candidate and original activations must be non-empty paired matrices")
    original_norm = np.maximum(np.linalg.norm(original, axis=1), 1e-12)
    candidate_norm = np.maximum(np.linalg.norm(candidate, axis=1), 1e-30)
    error = candidate - original
    cosine = np.sum(candidate * original, axis=1) / np.maximum(candidate_norm * original_norm, 1e-30)
    scale = np.sum(candidate * original, axis=1) / np.maximum(np.sum(original * original, axis=1), 1e-30)
    orthogonal = candidate - scale[:, None] * original
    return {
        "normalized_error": np.linalg.norm(error, axis=1) / original_norm,
        "cosine": cosine,
        "norm_ratio": candidate_norm / original_norm,
        "signed_parallel_scale_error": scale - 1.0,
        "orthogonal_residual_fraction": np.linalg.norm(orthogonal, axis=1) / original_norm,
    }


def behavior_rows(candidate_logits: np.ndarray, original_logits: np.ndarray) -> np.ndarray:
    """Return row-wise normalized next-logit discrepancies."""

    candidate_logits = np.asarray(candidate_logits, dtype=np.float64)
    original_logits = np.asarray(original_logits, dtype=np.float64)
    if candidate_logits.ndim != 2 or candidate_logits.shape != original_logits.shape:
        raise ValueError("candidate and original next logits must be paired matrices")
    denom = np.maximum(np.linalg.norm(original_logits, axis=1), 1e-12)
    return np.linalg.norm(candidate_logits - original_logits, axis=1) / denom


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("summary requires non-empty finite values")
    return {
        "count": int(values.size),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
    }


def evaluate_seed(
    strata: np.ndarray,
    original_activation: np.ndarray,
    original_logits: np.ndarray,
    original_ce: np.ndarray,
    candidate_activation: dict[str, np.ndarray],
    candidate_logits: dict[str, np.ndarray],
    candidate_ce: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute locked per-stratum metrics and paired treatment effects."""

    strata = np.asarray(strata).astype(str)
    if strata.shape != (60,) or any(np.sum(strata == item) != 20 for item in ("ordinary", "compositional", "adversarial")):
        raise ValueError("challenge fixture must contain 20 rows per locked stratum")
    arms = ("ordinary_only", "compositional_augmented")
    for arm in arms:
        if arm not in candidate_activation or arm not in candidate_logits or arm not in candidate_ce:
            raise ValueError(f"missing candidate prediction arm {arm}")
    metrics: dict[str, Any] = {}
    row_errors: dict[str, np.ndarray] = {}
    for arm in arms:
        activation = activation_rows(candidate_activation[arm], original_activation)
        behavior = behavior_rows(candidate_logits[arm], original_logits)
        ce_delta = np.asarray(candidate_ce[arm], dtype=np.float64) - np.asarray(original_ce, dtype=np.float64)
        row_errors[arm] = activation["normalized_error"]
        metrics[arm] = {"per_stratum": {}}
        for stratum in ("ordinary", "compositional", "adversarial"):
            mask = strata == stratum
            metrics[arm]["per_stratum"][stratum] = {
                "activation_error": summarize(activation["normalized_error"][mask]),
                "activation_cosine": summarize(activation["cosine"][mask]),
                "candidate_to_original_activation_norm_ratio": summarize(activation["norm_ratio"][mask]),
                "orthogonal_residual_fraction": summarize(activation["orthogonal_residual_fraction"][mask]),
                "max_normalized_next_logit_error": float(np.max(behavior[mask])),
                "mean_cross_entropy_delta_nats_per_token": float(np.mean(ce_delta[mask])),
            }
        metrics[arm]["activation_gate_pass"] = all(
            metrics[arm]["per_stratum"][stratum]["activation_error"]["median"] <= ACTIVATION_ERROR_MAX
            and metrics[arm]["per_stratum"][stratum]["activation_cosine"]["median"] >= 0.90
            for stratum in ("ordinary", "compositional", "adversarial")
        )
        metrics[arm]["behavior_guard_pass"] = all(
            metrics[arm]["per_stratum"][stratum]["max_normalized_next_logit_error"] <= BEHAVIOR_LOGIT_ERROR_MAX
            and metrics[arm]["per_stratum"][stratum]["mean_cross_entropy_delta_nats_per_token"] <= BEHAVIOR_CE_DELTA_MAX
            for stratum in ("ordinary", "compositional", "adversarial")
        )
    comp = strata == "compositional"
    treatment_error = row_errors["compositional_augmented"][comp]
    baseline_error = row_errors["ordinary_only"][comp]
    improvement = baseline_error - treatment_error
    noncomp = ~comp
    regression = row_errors["compositional_augmented"][noncomp] - row_errors["ordinary_only"][noncomp]
    comparison = {
        "compositional_median_error_improvement": float(np.median(baseline_error) - np.median(treatment_error)),
        "compositional_mean_paired_error_improvement": float(np.mean(improvement)),
        "compositional_fraction_rows_improved": float(np.mean(improvement > 0.0)),
        "ordinary_and_adversarial_median_regression": float(np.median(regression)),
        "treatment_compositional_activation_gate_pass": bool(
            metrics["compositional_augmented"]["per_stratum"]["compositional"]["activation_error"]["median"] <= ACTIVATION_ERROR_MAX
            and metrics["compositional_augmented"]["per_stratum"]["compositional"]["activation_cosine"]["median"] >= 0.90
        ),
        "minimum_improvement_pass": bool(np.median(baseline_error) - np.median(treatment_error) >= MIN_PAIRED_MEDIAN_IMPROVEMENT),
        "noncompositional_noninferiority_pass": bool(np.median(regression) <= MAX_NONCOMPOSITIONAL_REGRESSION),
    }
    return {"arms": metrics, "paired_compositional_comparison": comparison}


def seed_pass(result: dict[str, Any]) -> bool:
    """Apply the predeclared all-seed adaptation-support intervention gate."""

    treatment = result["arms"]["compositional_augmented"]
    comparison = result["paired_compositional_comparison"]
    return bool(
        comparison["treatment_compositional_activation_gate_pass"]
        and comparison["minimum_improvement_pass"]
        and comparison["noncompositional_noninferiority_pass"]
        and treatment["behavior_guard_pass"]
    )

