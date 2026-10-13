"""Digest-bound execution continuation for the frozen replacement-loss-v9 packet.

State slice: theseus-lab-sandbox-v1.

This module only bridges a misspelled read-time validation key after the
original primary stopped before challenge entropy existed. It does not alter
the frozen v9 runner, candidate files, thresholds, prediction rule, or target
manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import replacement_loss_v9_runner as runner

CONTINUATION_ID = "theseus-replacement-loss-v9-execution-continuation-v1"
STATE_SLICE = runner.STATE_SLICE
FREEZE_NAME = "v9-continuation-freeze.json"
PRIMARY_RECEIPT_NAME = "v9-continuation-primary.json"
PRIMARY_STOP_NAME = "v9-continuation-stop.json"
REPLICATION_RECEIPT_NAME = "v9-continuation-replication.json"
CONTINUATION_SOURCE_FILES = (
    "replacement_loss_v9_continuation.py",
    "tests/test_replacement_loss_v9_continuation.py",
    "docs/replacement-loss-v9-continuation.md",
)
BASE_RECEIPT_FILES = (
    "v9-frozen-packet-manifest.json",
    "v9-checkpoint-manifest.json",
    "prior-checkpoint-identity-inventory.json",
    "candidate-preparation.json",
    "candidate-freeze.json",
    "primary-run-stop.json",
    "v9-disjointness-preflight.json",
    "prior-identities.json",
    "checkpoint-source-evidence.json",
    "architecture-preflight.json",
    "selection-spec.json",
    "selection-source-binding.json",
)
EXPECTED_STOP_ERROR = "candidate prediction functions were not bound before challenge generation"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return runner._canonical_digest(value)


def _read_json(path: Path) -> dict[str, Any]:
    return runner._read_json(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    runner._write_json(path, value)


def _custody(path: str | Path) -> Path:
    return runner._custody_root(path)


def _source_rows() -> list[dict[str, Any]]:
    repo = runner._root()
    rows = []
    for relative in CONTINUATION_SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"continuation packet source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _assert_no_held_out_material(root: Path) -> None:
    forbidden = (
        "challenge-manifest.json",
        "challenge-secrets",
        "challenges",
        "prediction-manifest.json",
        "predictions",
        "outcomes",
        "outcome-manifest.json",
    )
    present = [name for name in forbidden if (root / name).exists()]
    if present:
        raise FileExistsError(f"held-out material already exists; continuation is not safe: {present}")
    if any((root / "fixtures").glob("challenge-seed-*.json")):
        raise FileExistsError("held-out prompt fixtures already exist; continuation is not safe")


def _assert_stopped_at_unrevealed_boundary(
    root: Path,
    *,
    require_no_held_out_material: bool = True,
) -> dict[str, Any]:
    packet = runner.verify_packet(root)
    runner._load_checkpoint_specs(root, packet["packet_sha256"])
    preparation = _read_json(root / "candidate-preparation.json")
    freeze = runner._verify_candidate_freeze(root, packet["packet_sha256"])
    stop = _read_json(root / "primary-run-stop.json")
    stop_material = {key: value for key, value in stop.items() if key != "stop_sha256"}
    if stop.get("stop_sha256") != _canonical_digest(stop_material):
        raise ValueError("original primary stop receipt digest is invalid")
    if (
        stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("stopped_phase") != "challenge_generation"
        or stop.get("error") != EXPECTED_STOP_ERROR
        or stop.get("checkpoint_manifest_sha256") != _sha_file(root / "v9-checkpoint-manifest.json")
    ):
        raise ValueError("primary stop does not match the narrowly scoped continuation case")
    preparation_material = {
        key: value
        for key, value in preparation.items()
        if key != "candidate_preparation_sha256"
    }
    if preparation.get("candidate_preparation_sha256") != _canonical_digest(
        preparation_material
    ):
        raise ValueError("candidate preparation's internal digest is invalid")
    if _sha_file(root / "candidate-preparation.json") != freeze.get(
        "candidate_preparation_sha256"
    ):
        raise ValueError("candidate preparation is not bound to the frozen candidates")
    if (
        freeze.get("predictions_functionally_bound_before_challenge_entropy") is not True
        or freeze.get("prediction_rule_count") != 81
        or freeze.get("candidate_placement_count") != 81
        or freeze.get("challenge_fixtures_generated") is not False
        or freeze.get("original_outcomes_opened") is not False
        or len(freeze.get("prediction_rules", [])) != 81
    ):
        raise ValueError("the frozen candidate receipt does not bind all prediction rules")
    if require_no_held_out_material:
        _assert_no_held_out_material(root)
    return {
        "packet": packet,
        "preparation": preparation,
        "candidate_freeze": freeze,
        "stop": stop,
    }


def _base_receipt_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in BASE_RECEIPT_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"continuation base receipt is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def freeze_continuation(custody_root: str | Path) -> dict[str, Any]:
    """Freeze the correction and its exact v9 pre-challenge base state."""

    root = _custody(custody_root)
    if (root / FREEZE_NAME).exists():
        raise FileExistsError("continuation packet freeze is immutable")
    base = _assert_stopped_at_unrevealed_boundary(root)
    runtime = runner.verify_runtime()
    payload = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": runner.ASSURANCE_OVERLAY,
        "base_packet_sha256": base["packet"]["packet_sha256"],
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "prior_identity_manifest_sha256": _canonical_digest(
            _read_json(root / "prior-identities.json")["known_prior_checkpoint_identities"]
        ),
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "prediction_rules_sha256": _canonical_digest(base["candidate_freeze"]["prediction_rules"]),
        "original_stop_receipt_sha256": _sha_file(root / "primary-run-stop.json"),
        "base_receipts": _base_receipt_rows(root),
        "continuation_sources": _source_rows(),
        "runtime": {
            "python": runtime["python"],
            "dependencies": runtime["dependencies"],
            "dependency_manifest_sha256": runtime["dependency_manifest_sha256"],
            "locked_environment_pass": runtime["locked_environment_pass"],
            "cpu_thread_count": runtime["cpu_thread_count"],
        },
        "repair_contract": {
            "read_time_alias_from": "predictions_functionally_bound_before_challenge_entropy",
            "read_time_alias_to": "prediction_rules_functionally_bound_before_challenge_entropy",
            "frozen_candidate_receipt_is_not_modified": True,
            "candidate_refit": False,
            "checkpoint_or_threshold_change": False,
            "challenge_entropy_exists_at_freeze": False,
            "held_out_outcomes_opened_at_freeze": False,
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["continuation_packet_sha256"] = _canonical_digest(payload)
    _write_json(root / FREEZE_NAME, payload)
    return {
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": payload["continuation_packet_sha256"],
        "base_packet_sha256": payload["base_packet_sha256"],
        "candidate_freeze_sha256": payload["candidate_freeze_sha256"],
        "prediction_rule_count": 81,
    }


def verify_continuation(custody_root: str | Path) -> dict[str, Any]:
    root = _custody(custody_root)
    base = _assert_stopped_at_unrevealed_boundary(root, require_no_held_out_material=False)
    payload = _read_json(root / FREEZE_NAME)
    material = {key: value for key, value in payload.items() if key != "continuation_packet_sha256"}
    if payload.get("continuation_packet_sha256") != _canonical_digest(material):
        raise ValueError("continuation packet digest is invalid")
    expected = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "base_packet_sha256": base["packet"]["packet_sha256"],
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "prior_identity_manifest_sha256": _canonical_digest(
            _read_json(root / "prior-identities.json")["known_prior_checkpoint_identities"]
        ),
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "prediction_rules_sha256": _canonical_digest(base["candidate_freeze"]["prediction_rules"]),
        "original_stop_receipt_sha256": _sha_file(root / "primary-run-stop.json"),
        "base_receipts": _base_receipt_rows(root),
        "continuation_sources": _source_rows(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"continuation packet binding changed: {key}")
    if (payload.get("repair_contract") or {}).get("candidate_refit") is not False:
        raise ValueError("continuation packet permits an unexpected candidate refit")
    runtime = runner.verify_runtime()
    if (
        payload["runtime"]["dependency_manifest_sha256"] != runtime["dependency_manifest_sha256"]
        or payload["runtime"]["locked_environment_pass"] is not True
    ):
        raise ValueError("continuation runtime differs from the frozen environment")
    return payload


class _CandidateFreezeAlias(dict):
    """Expose the generator's misspelled predicate without changing receipt bytes."""

    def get(self, key: str, default: Any = None) -> Any:
        if key == "prediction_rules_functionally_bound_before_challenge_entropy":
            return dict.get(self, "predictions_functionally_bound_before_challenge_entropy", default)
        return dict.get(self, key, default)


def _generate_challenges_with_frozen_alias(
    custody: Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    packet = runner.verify_packet(custody)
    runner._verify_candidate_freeze(custody, packet["packet_sha256"])
    original_reader = runner._read_json
    target_freeze = (custody / "candidate-freeze.json").resolve()

    def compatibility_reader(path: str | Path) -> dict[str, Any]:
        value = original_reader(Path(path))
        if Path(path).expanduser().resolve() == target_freeze:
            return _CandidateFreezeAlias(value)
        return value

    runner._read_json = compatibility_reader
    try:
        return runner.generate_challenges(custody, prior_custody_roots)
    finally:
        runner._read_json = original_reader


def continue_primary(
    custody_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    """Resume only at challenge generation, with the original candidates frozen."""

    root = _custody(custody_root)
    packet = verify_continuation(root)
    _assert_no_held_out_material(root)
    if (root / PRIMARY_RECEIPT_NAME).exists() or (root / PRIMARY_STOP_NAME).exists():
        raise FileExistsError("continuation execution is immutable")
    runner._ACTIVE_RUN_STARTED = time.monotonic()
    runner._ACTIVE_CASE_STARTED = None
    runner._ACTIVE_PHASE = "challenge_generation"
    started = time.monotonic()
    try:
        challenge = _generate_challenges_with_frozen_alias(root, prior_custody_roots)
        runner._ACTIVE_PHASE = "prediction_sealing"
        prediction = runner.seal_predictions(root)
        runner._ACTIVE_PHASE = "source_outcome_evaluation"
        outcome = runner.evaluate_outcomes(root)
        report_path = root / "outcome-manifest.json"
        report = _read_json(report_path)
        original_stop = _read_json(root / "primary-run-stop.json")
        continuation_elapsed = time.monotonic() - started
        combined_elapsed = float(original_stop["elapsed_wall_seconds"]) + continuation_elapsed
        original_runtime = original_stop["runtime"]
        current_runtime = report["runtime"]
        combined_peak_rss = max(
            int(original_runtime["peak_rss_bytes"]),
            int(current_runtime["peak_rss_bytes"]),
        )
        combined_cpu = float(original_runtime["process_cpu_seconds"]) + float(
            current_runtime["process_cpu_seconds"]
        )
        if combined_elapsed > runner.MAX_TOTAL_SECONDS or combined_peak_rss > runner.MAX_RSS_BYTES:
            report["outcome_status"] = "INCONCLUSIVE"
            report["assurance_level"] = "INCONCLUSIVE"
        report["reproducibility_status"] = "PENDING_CLEAN_LOCAL_REPRODUCTION"
        report["primary_protocol_wall_seconds"] = combined_elapsed
        report["primary_budget_pass"] = report["outcome_status"] != "INCONCLUSIVE"
        report["runtime"]["peak_rss_bytes"] = combined_peak_rss
        report["runtime"]["process_cpu_seconds"] = combined_cpu
        report["runtime"]["initial_stopped_attempt_cpu_seconds"] = original_runtime["process_cpu_seconds"]
        report["runtime"]["continuation_cpu_seconds"] = current_runtime["process_cpu_seconds"]
        report["continuation_id"] = CONTINUATION_ID
        report["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
        report["original_primary_stop_sha256"] = _sha_file(root / "primary-run-stop.json")
        report["candidate_refit_during_continuation"] = False
        report["claim_ceiling"] = (
            "LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS"
        )
        report.pop("report_sha256", None)
        report["report_sha256"] = _canonical_digest(report)
        _write_json(report_path, report)

        receipt = {
            "continuation_id": CONTINUATION_ID,
            "protocol_id": runner.PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "assurance_overlay": runner.ASSURANCE_OVERLAY,
            "continuation_packet_sha256": packet["continuation_packet_sha256"],
            "base_packet_sha256": packet["base_packet_sha256"],
            "checkpoint_manifest_sha256": packet["checkpoint_manifest_sha256"],
            "candidate_freeze_sha256": packet["candidate_freeze_sha256"],
            "original_stop_receipt_sha256": packet["original_stop_receipt_sha256"],
            "challenge_manifest_sha256": challenge["challenge_manifest_sha256"],
            "prediction_manifest_sha256": prediction["prediction_manifest_sha256"],
            "outcome_report_sha256": report["report_sha256"],
            "outcome_status": report["outcome_status"],
            "assurance_level": report["assurance_level"],
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
            "candidate_refit": False,
            "held_out_challenges_generated_after_continuation_freeze": True,
            "original_outcomes_opened_after_prediction_seal": True,
            "continuation_elapsed_wall_seconds": continuation_elapsed,
            "combined_primary_active_wall_seconds": combined_elapsed,
            "combined_primary_cpu_seconds": combined_cpu,
            "combined_peak_rss_bytes": combined_peak_rss,
            "monetary_cost": "NOT_MEASURED",
            "energy": "NOT_MEASURED",
            "claim_ceiling": report["claim_ceiling"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["continuation_receipt_sha256"] = _canonical_digest(receipt)
        _write_json(root / PRIMARY_RECEIPT_NAME, receipt)
        runner._write_results_document(root, report_path)
        return {
            "outcome_status": report["outcome_status"],
            "assurance_level": report["assurance_level"],
            "reproducibility_status": report["reproducibility_status"],
            "continuation_receipt_sha256": receipt["continuation_receipt_sha256"],
            "report_sha256": report["report_sha256"],
        }
    except Exception as error:
        stop = {
            "continuation_id": CONTINUATION_ID,
            "continuation_packet_sha256": packet["continuation_packet_sha256"],
            "base_packet_sha256": packet["base_packet_sha256"],
            "candidate_freeze_sha256": packet["candidate_freeze_sha256"],
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": runner._ACTIVE_PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_wall_seconds": time.monotonic() - started,
            "runtime": runner._runtime_receipt(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["continuation_stop_sha256"] = _canonical_digest(stop)
        _write_json(root / PRIMARY_STOP_NAME, stop)
        raise
    finally:
        runner._ACTIVE_RUN_STARTED = None
        runner._ACTIVE_CASE_STARTED = None
        runner._ACTIVE_PHASE = "idle"


def _copy_continuation_context(primary: Path, replica: Path) -> None:
    packet_dir = replica / "v9-continuation-packet"
    for relative in CONTINUATION_SOURCE_FILES:
        source = runner._root() / relative
        target = packet_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for relative in (
        FREEZE_NAME,
        PRIMARY_RECEIPT_NAME,
        "primary-run-stop.json",
        "prior-identities.json",
        "checkpoint-source-evidence.json",
        "architecture-preflight.json",
        "selection-spec.json",
        "selection-source-binding.json",
    ):
        source = primary / relative
        target = replica / relative
        if not source.is_file():
            raise FileNotFoundError(f"primary continuation evidence missing: {relative}")
        shutil.copy2(source, target)


def replicate(
    primary_root: str | Path,
    replica_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    """Run the frozen v9 clean-runtime replicator and bind the continuation."""

    primary = _custody(primary_root)
    replica = _custody(replica_root)
    packet = verify_continuation(primary)
    primary_receipt = _read_json(primary / PRIMARY_RECEIPT_NAME)
    if primary_receipt.get("continuation_packet_sha256") != packet["continuation_packet_sha256"]:
        raise ValueError("primary continuation receipt does not bind the continuation packet")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("replica custody must be empty before replication")
    import replacement_loss_v9_replicator as replicator

    base_result = replicator._run_replica_impl(primary, replica, prior_custody_roots)
    _copy_continuation_context(primary, replica)
    replication_path = primary / "replication-manifest.json"
    replication = _read_json(replication_path)
    base_replication_sha = replication["replication_sha256"]
    replication.pop("replication_sha256", None)
    replication["continuation_id"] = CONTINUATION_ID
    replication["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
    replication["continuation_primary_receipt_sha256"] = _sha_file(primary / PRIMARY_RECEIPT_NAME)
    replication["original_primary_stop_sha256"] = _sha_file(primary / "primary-run-stop.json")
    replication["base_replication_manifest_sha256"] = base_replication_sha
    replication["replication_sha256"] = _canonical_digest(replication)
    _write_json(primary / "replication-manifest.json", replication)
    _write_json(replica / "replication-manifest.json", replication)

    continuation_replication = {
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": packet["continuation_packet_sha256"],
        "base_v9_replication_manifest_sha256": base_replication_sha,
        "bound_replication_manifest_sha256": replication["replication_sha256"],
        "reproducibility_status": base_result["reproducibility_status"],
        "assurance_level": base_result["assurance_level"],
        "checks": base_result["checks"],
        "reviewer_status": "NOT_SUBMITTED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    continuation_replication["continuation_replication_sha256"] = _canonical_digest(
        continuation_replication
    )
    _write_json(primary / REPLICATION_RECEIPT_NAME, continuation_replication)
    _write_json(replica / REPLICATION_RECEIPT_NAME, continuation_replication)

    report_path = primary / "outcome-manifest.json"
    report = _read_json(report_path)
    report["reproducibility_status"] = base_result["reproducibility_status"]
    report["assurance_level"] = base_result["assurance_level"]
    report["replication_manifest_sha256"] = _sha_file(replication_path)
    report["continuation_replication_sha256"] = continuation_replication[
        "continuation_replication_sha256"
    ]
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_digest(report)
    _write_json(report_path, report)
    replica_report_path = replica / "outcome-manifest.json"
    replica_report = _read_json(replica_report_path)
    replica_report["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
    replica_report["replication_manifest_sha256"] = _sha_file(replica / "replication-manifest.json")
    replica_report.pop("report_sha256", None)
    replica_report["report_sha256"] = _canonical_digest(replica_report)
    _write_json(replica_report_path, replica_report)
    runner._write_results_document(primary, report_path)
    return {
        "reproducibility_status": base_result["reproducibility_status"],
        "assurance_level": base_result["assurance_level"],
        "checks": base_result["checks"],
        "replication_sha256": replication["replication_sha256"],
        "continuation_replication_sha256": continuation_replication[
            "continuation_replication_sha256"
        ],
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "continue-primary", "replicate"))
    parser.add_argument("--custody-root")
    parser.add_argument("--primary-root")
    parser.add_argument("--replica-root")
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    if args.command == "freeze":
        if not args.custody_root:
            parser.error("freeze requires --custody-root")
        result = freeze_continuation(args.custody_root)
    elif args.command == "verify":
        if not args.custody_root:
            parser.error("verify requires --custody-root")
        payload = verify_continuation(args.custody_root)
        result = {
            "continuation_id": payload["continuation_id"],
            "continuation_packet_sha256": payload["continuation_packet_sha256"],
        }
    elif args.command == "continue-primary":
        if not args.custody_root:
            parser.error("continue-primary requires --custody-root")
        result = continue_primary(args.custody_root, args.prior_custody_root)
    else:
        if not args.primary_root or not args.replica_root:
            parser.error("replicate requires --primary-root and --replica-root")
        result = replicate(args.primary_root, args.replica_root, args.prior_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
