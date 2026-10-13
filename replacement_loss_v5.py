"""Metric definitions and locked gates for replacement-loss-v5.

State slice: theseus-lab-sandbox-v1.
"""

from __future__ import annotations

from typing import Any

import numpy as np


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v5-balanced-adversarial-support"
ARMS = ("ordinary_only", "compositional_augmented", "balanced_augmented")
STRATA = ("ordinary", "compositional", "adversarial")
ACTIVATION_ERROR_MAX = 0.20
COMPOSITIONAL_MIN_IMPROVEMENT = 0.10
COMPOSITIONAL_MAX_REGRESSION_VS_COMPOSITION_ARM = 0.05
ADVERSARIAL_MIN_ACTIVATION_IMPROVEMENT = 0.03
ADVERSARIAL_MIN_MAX_LOGIT_IMPROVEMENT = 0.02
MAX_NONCOMPOSITIONAL_REGRESSION = 0.05
BEHAVIOR_LOGIT_ERROR_MAX = 0.05
BEHAVIOR_CE_DELTA_MAX = 0.05


def activation_rows(candidate: np.ndarray, original: np.ndarray) -> dict[str, np.ndarray]:
    """Return v4-compatible per-row activation reconstruction metrics."""

    candidate = np.asarray(candidate, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if candidate.ndim != 2 or candidate.shape != original.shape or candidate.shape[0] == 0:
        raise ValueError("candidate and original activations must be non-empty paired matrices")
    original_norm = np.maximum(np.linalg.norm(original, axis=1), 1e-12)
    candidate_norm = np.maximum(np.linalg.norm(candidate, axis=1), 1e-30)
    cosine = np.sum(candidate * original, axis=1) / np.maximum(candidate_norm * original_norm, 1e-30)
    scale = np.sum(candidate * original, axis=1) / np.maximum(np.sum(original * original, axis=1), 1e-30)
    orthogonal = candidate - scale[:, None] * original
    return {
        "normalized_error": np.linalg.norm(candidate - original, axis=1) / original_norm,
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
    denominator = np.maximum(np.linalg.norm(original_logits, axis=1), 1e-12)
    return np.linalg.norm(candidate_logits - original_logits, axis=1) / denominator


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
    """Compute locked v5 per-stratum metrics and paired intervention effects."""

    strata = np.asarray(strata).astype(str)
    if strata.shape != (60,) or any(np.sum(strata == item) != 20 for item in STRATA):
        raise ValueError("challenge fixture must contain 20 rows per locked stratum")
    for arm in ARMS:
        if arm not in candidate_activation or arm not in candidate_logits or arm not in candidate_ce:
            raise ValueError(f"missing candidate prediction arm {arm}")
    metrics: dict[str, Any] = {}
    act_errors: dict[str, np.ndarray] = {}
    logit_errors: dict[str, np.ndarray] = {}
    for arm in ARMS:
        activation = activation_rows(candidate_activation[arm], original_activation)
        behavior = behavior_rows(candidate_logits[arm], original_logits)
        ce_delta = np.asarray(candidate_ce[arm], dtype=np.float64) - np.asarray(original_ce, dtype=np.float64)
        if ce_delta.shape != (60,) or not np.isfinite(ce_delta).all():
            raise ValueError("candidate cross-entropy values must be 60 finite paired rows")
        act_errors[arm] = activation["normalized_error"]
        logit_errors[arm] = behavior
        metrics[arm] = {"per_stratum": {}}
        for stratum in STRATA:
            mask = strata == stratum
            metrics[arm]["per_stratum"][stratum] = {
                "activation_error": summarize(activation["normalized_error"][mask]),
                "activation_cosine": summarize(activation["cosine"][mask]),
                "candidate_to_original_activation_norm_ratio": summarize(activation["norm_ratio"][mask]),
                "orthogonal_residual_fraction": summarize(activation["orthogonal_residual_fraction"][mask]),
                "max_normalized_next_logit_error": float(np.max(behavior[mask])),
                "mean_cross_entropy_delta_nats_per_token": float(np.mean(ce_delta[mask])),
            }
        metrics[arm]["behavior_guard_pass"] = all(
            metrics[arm]["per_stratum"][stratum]["max_normalized_next_logit_error"] <= BEHAVIOR_LOGIT_ERROR_MAX
            and metrics[arm]["per_stratum"][stratum]["mean_cross_entropy_delta_nats_per_token"] <= BEHAVIOR_CE_DELTA_MAX
            for stratum in STRATA
        )

    comp_mask = strata == "compositional"
    adv_mask = strata == "adversarial"
    noncomp_mask = ~comp_mask
    balanced_comp = act_errors["balanced_augmented"][comp_mask]
    comparator_comp = act_errors["compositional_augmented"][comp_mask]
    control_comp = act_errors["ordinary_only"][comp_mask]
    balanced_adv_act = act_errors["balanced_augmented"][adv_mask]
    comparator_adv_act = act_errors["compositional_augmented"][adv_mask]
    balanced_adv_logits = logit_errors["balanced_augmented"][adv_mask]
    comparator_adv_logits = logit_errors["compositional_augmented"][adv_mask]
    comp_median_improvement = float(np.median(control_comp) - np.median(balanced_comp))
    adv_median_activation_improvement = float(np.median(comparator_adv_act) - np.median(balanced_adv_act))
    adv_max_logit_improvement = float(np.max(comparator_adv_logits) - np.max(balanced_adv_logits))
    comp_regression_vs_comparator = float(np.median(balanced_comp) - np.median(comparator_comp))
    noncomp_regression = float(np.median(act_errors["balanced_augmented"][noncomp_mask] - act_errors["ordinary_only"][noncomp_mask]))

    balanced_metrics = metrics["balanced_augmented"]["per_stratum"]
    comparison = {
        "compositional_median_error_improvement_vs_ordinary": comp_median_improvement,
        "compositional_fraction_rows_improved_vs_ordinary": float(np.mean(balanced_comp < control_comp)),
        "compositional_activation_error_regression_vs_compositional_arm": comp_regression_vs_comparator,
        "compositional_activation_gate_pass": bool(
            balanced_metrics["compositional"]["activation_error"]["median"] <= ACTIVATION_ERROR_MAX
            and balanced_metrics["compositional"]["activation_cosine"]["median"] >= 0.90
        ),
        "compositional_minimum_improvement_pass": comp_median_improvement >= COMPOSITIONAL_MIN_IMPROVEMENT,
        "compositional_preservation_pass": comp_regression_vs_comparator <= COMPOSITIONAL_MAX_REGRESSION_VS_COMPOSITION_ARM,
        "adversarial_median_activation_error_improvement_vs_compositional_arm": adv_median_activation_improvement,
        "adversarial_activation_gate_pass": bool(
            balanced_metrics["adversarial"]["activation_error"]["median"] <= ACTIVATION_ERROR_MAX
            and balanced_metrics["adversarial"]["activation_cosine"]["median"] >= 0.90
            and adv_median_activation_improvement >= ADVERSARIAL_MIN_ACTIVATION_IMPROVEMENT
        ),
        "adversarial_max_next_logit_error_improvement_vs_compositional_arm": adv_max_logit_improvement,
        "adversarial_behavior_effect_pass": adv_max_logit_improvement >= ADVERSARIAL_MIN_MAX_LOGIT_IMPROVEMENT,
        "balanced_behavior_guard_pass": metrics["balanced_augmented"]["behavior_guard_pass"],
        "ordinary_and_adversarial_noninferiority_pass": noncomp_regression <= MAX_NONCOMPOSITIONAL_REGRESSION,
        "ordinary_and_adversarial_median_activation_regression": noncomp_regression,
    }
    return {"arms": metrics, "paired_comparisons": comparison}


def seed_pass(result: dict[str, Any]) -> bool:
    """Apply every preregistered v5 gate to one seed."""

    comparison = result["paired_comparisons"]
    return bool(all(comparison[key] for key in (
        "compositional_activation_gate_pass",
        "compositional_minimum_improvement_pass",
        "compositional_preservation_pass",
        "adversarial_activation_gate_pass",
        "adversarial_behavior_effect_pass",
        "balanced_behavior_guard_pass",
        "ordinary_and_adversarial_noninferiority_pass",
    )))

