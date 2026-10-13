"""Bind the completed v9 clean-runtime result after a context-copy stop.

State slice: theseus-lab-sandbox-v1.

The v9 replicator completed and wrote matching replication manifests before
the v3 adapter failed while copying audit context. This module verifies and
binds those existing results; it never runs the experiment or changes outcome
arrays and reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import replacement_loss_v9_continuation_v2 as continuation_v2
import replacement_loss_v9_continuation_v3 as continuation_v3
import replacement_loss_v9_runner as runner

CONTINUATION_ID = "theseus-replacement-loss-v9-replication-binding-v4"
STATE_SLICE = runner.STATE_SLICE
POSTPROCESS_STOP_NAME = "v9-continuation-v3-replication-stop.json"
FREEZE_NAME = "v9-continuation-v4-freeze.json"
RECEIPT_NAME = "v9-continuation-v4-replication-binding.json"
SOURCE_FILES = (
    "replacement_loss_v9_replication_binding_v4.py",
    "tests/test_replacement_loss_v9_replication_binding_v4.py",
    "docs/replacement-loss-v9-replication-binding-v4.md",
)


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
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"v4 binding source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _verified_report(path: Path) -> dict[str, Any]:
    report = _read_json(path)
    material = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != _canonical_digest(material):
        raise ValueError(f"outcome report digest is invalid: {path.name}")
    return report


def _pre_replication_report_identity(
    final_report: dict[str, Any],
    v3_receipt: dict[str, Any],
) -> dict[str, str]:
    """Reconstruct report identity before the replicator added its two fields."""

    original = {
        key: value
        for key, value in final_report.items()
        if key not in {"report_sha256", "replication_manifest_sha256", "reproducibility_status"}
    }
    original["reproducibility_status"] = v3_receipt.get("reproducibility_status")
    original["report_sha256"] = _canonical_digest(original)
    if original["report_sha256"] != v3_receipt.get("outcome_report_sha256"):
        raise ValueError("reconstructed pre-replication report differs from the v3 receipt digest")
    serialized = (json.dumps(original, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return {
        "canonical_report_sha256": original["report_sha256"],
        "report_file_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def record_postprocessing_stop(primary_root: str | Path, replica_root: str | Path) -> dict[str, Any]:
    """Record the observed v3 receipt-copy error without rerunning replication."""

    primary = _custody(primary_root)
    replica = _custody(replica_root)
    stop_path = primary / POSTPROCESS_STOP_NAME
    if stop_path.exists():
        raise FileExistsError("v3 postprocessing stop receipt is immutable")
    manifest_path = primary / "replication-manifest.json"
    replica_manifest_path = replica / "replication-manifest.json"
    if not manifest_path.is_file() or not replica_manifest_path.is_file():
        raise FileNotFoundError("completed base replication manifests are required")
    manifest = _read_json(manifest_path)
    stop = {
        "continuation_id": continuation_v3.CONTINUATION_ID,
        "state_slice": STATE_SLICE,
        "stop_status": "POSTPROCESSING_STOP_AFTER_REPLICATION_COMPLETED",
        "stopped_phase": "copy_continuation_context",
        "error_type": "FileNotFoundError",
        "error": "context-copy destination directory did not exist",
        "base_replication_manifest_sha256": _sha_file(manifest_path),
        "replica_replication_manifest_sha256": _sha_file(replica_manifest_path),
        "reproducibility_status": manifest.get("reproducibility_status"),
        "experiment_rerun": False,
        "prediction_or_outcome_artifacts_changed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    stop["postprocessing_stop_sha256"] = _canonical_digest(stop)
    _write_json(stop_path, stop)
    return {
        "postprocessing_stop_sha256": stop["postprocessing_stop_sha256"],
        "base_reproducibility_status": stop["reproducibility_status"],
        "experiment_rerun": False,
    }


def _validated_base(
    primary_root: str | Path,
    replica_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    primary = _custody(primary_root)
    replica = _custody(replica_root)
    v2 = _custody(v2_root)
    prior_v1 = _custody(prior_v1_root)
    v3_packet = continuation_v3.verify_continuation(primary, v2, prior_v1)
    v3_receipt = _read_json(primary / continuation_v3.PRIMARY_RECEIPT_NAME)
    receipt_material = {key: value for key, value in v3_receipt.items() if key != "continuation_receipt_sha256"}
    if (
        v3_receipt.get("continuation_receipt_sha256") != _canonical_digest(receipt_material)
        or v3_receipt.get("continuation_packet_sha256") != v3_packet.get("continuation_packet_sha256")
    ):
        raise ValueError("v3 primary receipt digest or packet binding is invalid")

    manifest_path = primary / "replication-manifest.json"
    replica_manifest_path = replica / "replication-manifest.json"
    if _sha_file(manifest_path) != _sha_file(replica_manifest_path):
        raise ValueError("primary and replica base replication manifests differ")
    manifest = _read_json(manifest_path)
    manifest_material = {key: value for key, value in manifest.items() if key != "replication_sha256"}
    if manifest.get("replication_sha256") != _canonical_digest(manifest_material):
        raise ValueError("base replication manifest digest is invalid")
    checks = manifest.get("checks") or {}
    if (
        manifest.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or manifest.get("assurance_level") != "REPRODUCED_LOCAL"
        or not checks
        or not all(value is True for value in checks.values())
        or manifest.get("candidate_state_match") is not True
    ):
        raise ValueError("base replication is not a complete REPRODUCED_LOCAL result")
    primary_report = _verified_report(primary / "outcome-manifest.json")
    replica_report = _verified_report(replica / "outcome-manifest.json")
    pre_replication_identity = _pre_replication_report_identity(primary_report, v3_receipt)
    if (
        primary_report.get("outcome_status") != "FAIL"
        or replica_report.get("outcome_status") != "FAIL"
        or primary_report.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or primary_report.get("replication_manifest_sha256") != _sha_file(manifest_path)
        or manifest.get("replica_report_sha256") != _sha_file(replica / "outcome-manifest.json")
        or manifest.get("primary_report_sha256_before_replication") != pre_replication_identity["report_file_sha256"]
    ):
        raise ValueError("primary, replica, and replication receipts do not bind the same FAIL result")

    stop_path = primary / POSTPROCESS_STOP_NAME
    stop = _read_json(stop_path)
    stop_material = {key: value for key, value in stop.items() if key != "postprocessing_stop_sha256"}
    if (
        stop.get("postprocessing_stop_sha256") != _canonical_digest(stop_material)
        or stop.get("continuation_id") != continuation_v3.CONTINUATION_ID
        or stop.get("base_replication_manifest_sha256") != _sha_file(manifest_path)
        or stop.get("replica_replication_manifest_sha256") != _sha_file(replica_manifest_path)
        or stop.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or stop.get("experiment_rerun") is not False
    ):
        raise ValueError("v3 postprocessing stop does not bind the completed replication")
    primary_runtime = runner.verify_runtime()
    prep = _read_json(primary / "candidate-preparation.json")
    if (
        primary_runtime["locked_environment_pass"] is not True
        or primary_runtime["dependency_manifest_sha256"] != prep["runtime"]["dependency_manifest_sha256"]
        or manifest.get("checks", {}).get("dependency_manifest") is not True
    ):
        raise ValueError("clean replica runtime does not match the frozen dependency environment")
    return {
        "primary": primary,
        "replica": replica,
        "v2": v2,
        "prior_v1": prior_v1,
        "v3_packet": v3_packet,
        "v3_receipt": v3_receipt,
        "manifest": manifest,
        "primary_report": primary_report,
        "replica_report": replica_report,
        "stop": stop,
        "primary_manifest_sha256": _sha_file(manifest_path),
        "primary_report_sha256": _sha_file(primary / "outcome-manifest.json"),
        "replica_report_sha256": _sha_file(replica / "outcome-manifest.json"),
        "pre_replication_report_identity": pre_replication_identity,
    }


def freeze_binding(
    primary_root: str | Path,
    replica_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    primary = _custody(primary_root)
    if (primary / FREEZE_NAME).exists():
        raise FileExistsError("v4 replication-binding freeze is immutable")
    base = _validated_base(primary, replica_root, v2_root, prior_v1_root)
    runtime = runner.verify_runtime()
    payload = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": runner.ASSURANCE_OVERLAY,
        "v3_continuation_packet_sha256": base["v3_packet"]["continuation_packet_sha256"],
        "v3_primary_receipt_sha256": _sha_file(primary / continuation_v3.PRIMARY_RECEIPT_NAME),
        "v3_postprocessing_stop_sha256": base["stop"]["postprocessing_stop_sha256"],
        "base_replication_manifest_sha256": base["primary_manifest_sha256"],
        "primary_outcome_report_sha256": base["primary_report_sha256"],
        "replica_outcome_report_sha256": base["replica_report_sha256"],
        "pre_replication_report_file_sha256": base["pre_replication_report_identity"]["report_file_sha256"],
        "pre_replication_report_canonical_sha256": base["pre_replication_report_identity"]["canonical_report_sha256"],
        "outcome_status_primary": base["primary_report"]["outcome_status"],
        "outcome_status_replica": base["replica_report"]["outcome_status"],
        "reproducibility_status": base["manifest"]["reproducibility_status"],
        "replication_checks": base["manifest"]["checks"],
        "source_files": _source_rows(),
        "runtime": {
            "python": runtime["python"],
            "dependencies": runtime["dependencies"],
            "dependency_manifest_sha256": runtime["dependency_manifest_sha256"],
            "locked_environment_pass": runtime["locked_environment_pass"],
            "cpu_thread_count": runtime["cpu_thread_count"],
        },
        "repair_contract": {
            "experiment_rerun": False,
            "prediction_or_outcome_arrays_modified": False,
            "base_replication_manifest_modified": False,
            "result_status_changed": False,
            "scope": "bind existing completed REPRODUCED_LOCAL FAIL result and preserve v3 postprocessing stop",
        },
        "claim_ceiling": base["primary_report"].get("claim_ceiling"),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["binding_packet_sha256"] = _canonical_digest(payload)
    _write_json(primary / FREEZE_NAME, payload)
    return {
        "continuation_id": CONTINUATION_ID,
        "binding_packet_sha256": payload["binding_packet_sha256"],
        "base_replication_manifest_sha256": payload["base_replication_manifest_sha256"],
        "reproducibility_status": payload["reproducibility_status"],
        "outcome_status": payload["outcome_status_primary"],
    }


def verify_binding(
    primary_root: str | Path,
    replica_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    primary = _custody(primary_root)
    base = _validated_base(primary, replica_root, v2_root, prior_v1_root)
    payload = _read_json(primary / FREEZE_NAME)
    material = {key: value for key, value in payload.items() if key != "binding_packet_sha256"}
    if payload.get("binding_packet_sha256") != _canonical_digest(material):
        raise ValueError("v4 replication-binding packet digest is invalid")
    expected = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v3_continuation_packet_sha256": base["v3_packet"]["continuation_packet_sha256"],
        "v3_primary_receipt_sha256": _sha_file(primary / continuation_v3.PRIMARY_RECEIPT_NAME),
        "v3_postprocessing_stop_sha256": base["stop"]["postprocessing_stop_sha256"],
        "base_replication_manifest_sha256": base["primary_manifest_sha256"],
        "primary_outcome_report_sha256": base["primary_report_sha256"],
        "replica_outcome_report_sha256": base["replica_report_sha256"],
        "pre_replication_report_file_sha256": base["pre_replication_report_identity"]["report_file_sha256"],
        "pre_replication_report_canonical_sha256": base["pre_replication_report_identity"]["canonical_report_sha256"],
        "outcome_status_primary": base["primary_report"]["outcome_status"],
        "outcome_status_replica": base["replica_report"]["outcome_status"],
        "reproducibility_status": base["manifest"]["reproducibility_status"],
        "replication_checks": base["manifest"]["checks"],
        "source_files": _source_rows(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"v4 packet binding changed: {key}")
    if payload.get("runtime", {}).get("dependency_manifest_sha256") != runner.verify_runtime()["dependency_manifest_sha256"]:
        raise ValueError("v4 binding runtime differs from frozen dependencies")
    return payload


def finalize_binding(
    primary_root: str | Path,
    replica_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    primary = _custody(primary_root)
    replica = _custody(replica_root)
    v2 = _custody(v2_root)
    prior_v1 = _custody(prior_v1_root)
    packet = verify_binding(primary, replica, v2, prior_v1)
    if (primary / RECEIPT_NAME).exists() or (replica / RECEIPT_NAME).exists():
        raise FileExistsError("v4 replication binding receipt is immutable")

    context_dir = replica / "v9-continuation-v3-context"
    context_dir.mkdir(parents=True, exist_ok=True)
    source_dir = replica / "v9-continuation-v3-packet"
    source_dir.mkdir(parents=True, exist_ok=True)
    for relative in continuation_v3.SOURCE_FILES:
        source = runner._root() / relative
        shutil.copy2(source, source_dir / Path(relative).name)
    copies = (
        (primary, (
            continuation_v3.FREEZE_NAME,
            continuation_v3.PRIMARY_RECEIPT_NAME,
            continuation_v3.SNAPSHOT_NAME,
            continuation_v3.CANDIDATE_PREFLIGHT_NAME,
            POSTPROCESS_STOP_NAME,
        )),
        (v2, (continuation_v2.FREEZE_NAME, continuation_v2.STOP_NAME)),
        (prior_v1, (
            continuation_v2.continuation_v1.FREEZE_NAME,
            continuation_v2.continuation_v1.PRIMARY_STOP_NAME,
        )),
    )
    context_rows = []
    for source_root, relatives in copies:
        for relative in relatives:
            source = source_root / relative
            target = context_dir / Path(relative).name
            if not source.is_file():
                raise FileNotFoundError(f"required replication context is missing: {relative}")
            shutil.copy2(source, target)
            context_rows.append({"path": target.relative_to(replica).as_posix(), "bytes": target.stat().st_size, "sha256": _sha_file(target)})
    for relative in SOURCE_FILES:
        source = runner._root() / relative
        target = replica / "v9-continuation-v4-packet" / Path(relative).name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    context_manifest = {
        "continuation_id": CONTINUATION_ID,
        "binding_packet_sha256": packet["binding_packet_sha256"],
        "files": sorted(context_rows, key=lambda row: row["path"]),
    }
    context_manifest["context_sha256"] = _canonical_digest(context_manifest)
    _write_json(replica / "v9-continuation-v4-context.json", context_manifest)
    receipt = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "binding_packet_sha256": packet["binding_packet_sha256"],
        "v3_continuation_packet_sha256": packet["v3_continuation_packet_sha256"],
        "v3_postprocessing_stop_sha256": packet["v3_postprocessing_stop_sha256"],
        "base_replication_manifest_sha256": packet["base_replication_manifest_sha256"],
        "primary_outcome_report_sha256": packet["primary_outcome_report_sha256"],
        "replica_outcome_report_sha256": packet["replica_outcome_report_sha256"],
        "outcome_status": "FAIL",
        "reproducibility_status": "REPRODUCED_LOCAL",
        "assurance_level": "REPRODUCED_LOCAL",
        "reviewer_status": "NOT_SUBMITTED",
        "replication_checks": packet["replication_checks"],
        "same_owner_clean_runtime": True,
        "experiment_rerun": False,
        "prediction_or_outcome_arrays_modified": False,
        "claim_ceiling": packet["claim_ceiling"],
        "next_blocked_authority_gate": "blind external replication and independent signed review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["binding_receipt_sha256"] = _canonical_digest(receipt)
    _write_json(primary / RECEIPT_NAME, receipt)
    _write_json(replica / RECEIPT_NAME, receipt)
    verify_binding(primary, replica, v2, prior_v1)
    return {
        "outcome_status": receipt["outcome_status"],
        "reproducibility_status": receipt["reproducibility_status"],
        "assurance_level": receipt["assurance_level"],
        "binding_receipt_sha256": receipt["binding_receipt_sha256"],
        "replication_checks_all_pass": all(receipt["replication_checks"].values()),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record-stop", "freeze", "verify", "finalize"))
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--v2-root", required=True)
    parser.add_argument("--prior-v1-root", required=True)
    args = parser.parse_args()
    if args.command == "record-stop":
        result = record_postprocessing_stop(args.primary_root, args.replica_root)
    elif args.command == "freeze":
        result = freeze_binding(args.primary_root, args.replica_root, args.v2_root, args.prior_v1_root)
    elif args.command == "verify":
        payload = verify_binding(args.primary_root, args.replica_root, args.v2_root, args.prior_v1_root)
        result = {"continuation_id": payload["continuation_id"], "binding_packet_sha256": payload["binding_packet_sha256"]}
    else:
        result = finalize_binding(args.primary_root, args.replica_root, args.v2_root, args.prior_v1_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
