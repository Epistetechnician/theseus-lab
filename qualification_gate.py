"""Explicit evidence gates for Theseus qualification claims.

The gates validate the shape and consistency of attestations supplied by a
runner. They do not manufacture evidence: a local process must truthfully
report whether it loaded weights, consumed assessment data, or called a
provider. Passing the local causal gate only authorizes the narrow recorded
equivalence-class observation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any, Mapping


STATE_SLICE = "theseus-lab-sandbox-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


_GATE_RESULT_TOKEN = object()


@dataclass(frozen=True, init=False)
class GateResult:
    """Machine-readable result with an explicit evidence ceiling."""

    gate: str
    passed: bool
    claim_ceiling: str
    blockers: tuple[str, ...]

    def __init__(
        self,
        gate: str,
        passed: bool,
        claim_ceiling: str,
        blockers: tuple[str, ...],
        *,
        _token: object | None = None,
    ) -> None:
        if _token is not _GATE_RESULT_TOKEN:
            raise TypeError("gate results must be created by an evidence evaluator")
        if not isinstance(gate, str) or not gate:
            raise ValueError("gate result requires a gate name")
        if not isinstance(passed, bool):
            raise ValueError("gate result passed flag must be boolean")
        if not isinstance(claim_ceiling, str) or not claim_ceiling:
            raise ValueError("gate result requires a claim ceiling")
        if not isinstance(blockers, tuple) or not all(
            isinstance(blocker, str) and blocker for blocker in blockers
        ):
            raise ValueError("gate result blockers must be a tuple of non-empty strings")
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "claim_ceiling", claim_ceiling)
        object.__setattr__(self, "blockers", blockers)


def _gate_result(
    gate: str,
    passed: bool,
    claim_ceiling: str,
    blockers: tuple[str, ...],
) -> GateResult:
    """Construct a result only from this module's evaluator paths."""

    return GateResult(
        gate,
        passed,
        claim_ceiling,
        blockers,
        _token=_GATE_RESULT_TOKEN,
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(SHA256_RE.fullmatch(value))


def _required_digest(payload: Mapping[str, Any], key: str, blockers: list[str]) -> None:
    if not _is_sha256(payload.get(key)):
        blockers.append(f"{key}_missing_or_invalid")


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _unit_interval_number(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def evaluate_mechanistic_evidence(payload: Mapping[str, Any]) -> GateResult:
    """Validate the narrow raw/canonical/intervention identity criterion."""

    blockers: list[str] = []
    if payload.get("state_slice") != STATE_SLICE:
        blockers.append("state_slice_mismatch")
    if not bool(payload.get("weights_loaded")):
        blockers.append("weights_not_loaded")
    if not _positive_int(payload.get("heldout_observation_count", 0)):
        blockers.append("heldout_observations_missing")
    for key in (
        "checkpoint_digest",
        "probe_set_digest",
        "thresholds_digest",
    ):
        _required_digest(payload, key, blockers)
    for key in (
        "raw_digest_diff",
        "canonical_digest_match",
        "blind_mapping_match",
        "behavior_match",
        "aligned_intervention_match",
        "control_canonical_digest_diff",
    ):
        if not bool(payload.get(key)):
            blockers.append(f"{key}_failed")
    return _gate_result(
        gate="mechanistic_equivalence_observation",
        passed=not blockers,
        claim_ceiling="LOCAL_VALIDATED_EQUIVALENCE_CLASS_OBSERVATION_ONLY",
        blockers=tuple(blockers),
    )


def evaluate_alignment_evidence(payload: Mapping[str, Any]) -> GateResult:
    """Require a real, disjoint, threshold-locked held-out assessment."""

    blockers: list[str] = []
    if payload.get("state_slice") != STATE_SLICE:
        blockers.append("state_slice_mismatch")
    for key in (
        "target_digest",
        "assessment_corpus_digest",
        "selection_probe_digest",
        "assessment_probe_digest",
        "thresholds_digest",
        "assessment_receipt_digest",
    ):
        _required_digest(payload, key, blockers)
    if payload.get("selection_probe_digest") == payload.get("assessment_probe_digest"):
        blockers.append("selection_and_assessment_probes_not_disjoint")
    if not bool(payload.get("thresholds_locked_before_assessment")):
        blockers.append("thresholds_not_locked_before_assessment")
    if not bool(payload.get("real_assessment_data_consumed")):
        blockers.append("real_assessment_data_not_consumed")
    if not bool(payload.get("aggregate_only")):
        blockers.append("non_aggregate_assessment_output")
    if not _positive_int(payload.get("heldout_observation_count", 0)):
        blockers.append("heldout_assessment_observations_missing")
    pass_rate = payload.get("pass_rate")
    threshold = payload.get("minimum_pass_rate")
    if not _unit_interval_number(pass_rate):
        blockers.append("pass_rate_invalid")
    if not _unit_interval_number(threshold):
        blockers.append("minimum_pass_rate_invalid")
    if (
        _unit_interval_number(pass_rate)
        and _unit_interval_number(threshold)
        and float(pass_rate) < float(threshold)
    ):
        blockers.append("alignment_target_threshold_failed")
    return _gate_result(
        gate="alignment_validation",
        passed=not blockers,
        claim_ceiling="HELD_OUT_ALIGNMENT_ASSESSMENT_VALIDATION_ONLY",
        blockers=tuple(blockers),
    )


def evaluate_production_evidence(
    payload: Mapping[str, Any],
    alignment_result: GateResult,
) -> GateResult:
    """Require provider, operations, custody, and alignment evidence."""

    blockers: list[str] = []
    if payload.get("state_slice") != STATE_SLICE:
        blockers.append("state_slice_mismatch")
    if not bool(payload.get("weights_loaded")):
        blockers.append("weights_not_loaded")
    if not bool(payload.get("provider_called")):
        blockers.append("provider_not_called")
    if not isinstance(payload.get("provider_name"), str) or not payload["provider_name"]:
        blockers.append("provider_name_missing")
    if not isinstance(payload.get("provider_model"), str) or not payload["provider_model"]:
        blockers.append("provider_model_missing")
    for key in (
        "checkpoint_digest",
        "provider_input_digest",
        "provider_output_digest",
        "provider_receipt_digest",
        "assessment_corpus_digest",
    ):
        _required_digest(payload, key, blockers)
    if (
        not isinstance(alignment_result, GateResult)
        or alignment_result.gate != "alignment_validation"
        or not alignment_result.passed
    ):
        blockers.append("alignment_validation_missing")
    if not bool(payload.get("latency_slo_verified")):
        blockers.append("latency_slo_unverified")
    if not bool(payload.get("repeatability_verified")):
        blockers.append("repeatability_unverified")
    if not bool(payload.get("observability_verified")):
        blockers.append("observability_unverified")
    if not bool(payload.get("failure_recovery_verified")):
        blockers.append("failure_recovery_unverified")
    if not bool(payload.get("rollback_verified")):
        blockers.append("rollback_unverified")
    if bool(payload.get("credentials_persisted")):
        blockers.append("credentials_persisted")
    if not bool(payload.get("production_authorization")):
        blockers.append("production_authorization_missing")
    return _gate_result(
        gate="production_readiness",
        passed=not blockers,
        claim_ceiling="PRODUCTION_READINESS_VALIDATION_ONLY",
        blockers=tuple(blockers),
    )


def qualification_summary(
    mechanistic: GateResult,
    alignment: GateResult,
    production: GateResult,
) -> dict[str, object]:
    """Return an aggregate status without widening any individual ceiling."""

    return {
        "state_slice": STATE_SLICE,
        "mechanistic_equivalence_observation": {
            "passed": mechanistic.passed,
            "claim_ceiling": mechanistic.claim_ceiling,
            "blockers": list(mechanistic.blockers),
        },
        "alignment_validation": {
            "passed": alignment.passed,
            "claim_ceiling": alignment.claim_ceiling,
            "blockers": list(alignment.blockers),
        },
        "production_readiness": {
            "passed": production.passed,
            "claim_ceiling": production.claim_ceiling,
            "blockers": list(production.blockers),
        },
        "all_gates_passed": mechanistic.passed and alignment.passed and production.passed,
    }
