#!/usr/bin/env python3
"""Restore the frozen primary report after preserving discrepant clone bytes.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h.
"""
from __future__ import annotations
import hashlib, json, os, tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path("/home/dev/theseus-v9")
PACKET = ROOT / "v4h"
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h"
STATE_SLICE = PROTOCOL_ID

def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value

def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(path.name + ".v4h.tmp")
    if temporary.exists():
        raise ValueError(f"unexpected recovery temporary file exists: {temporary.name}")
    temporary.write_bytes(data)
    os.replace(temporary, path)

def _recover() -> dict[str, Any]:
    freeze = _read_json(PACKET / "freeze.json")
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material) or freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("state_slice") != STATE_SLICE:
        raise ValueError("V4h freeze identity or canonical digest mismatch")
    recovery, target = freeze["primary_report_recovery"], freeze["target"]
    archive_path = ROOT / "input" / "gman-v9-input.tar.gz"
    manifest_path = ROOT / "input" / "metadata" / "input-manifest.json"
    if not archive_path.is_file() or _sha_file(archive_path) != recovery["archive_sha256"] or archive_path.stat().st_size != recovery["archive_bytes"] or not manifest_path.is_file() or _sha_file(manifest_path) != freeze["input"]["input_manifest_sha256"]:
        raise ValueError("V4h frozen archive or input manifest identity mismatch")
    expected_sha = target["primary_outcome_manifest_sha256"]
    matches: list[tuple[PurePosixPath, bytes]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if not member.isfile() or member_path.name != "outcome-manifest.json":
                continue
            stream = archive.extractfile(member)
            if stream is not None:
                data = stream.read()
                if _sha_bytes(data) == expected_sha:
                    matches.append((member_path, data))
    if len(matches) != 1:
        raise ValueError("frozen primary report is not unique in the verified archive")
    member_path, expected_bytes = matches[0]
    if member_path.is_absolute() or ".." in member_path.parts or member_path.name != "outcome-manifest.json":
        raise ValueError("frozen primary report path is unsafe")
    relative_report = Path("input", *member_path.parts)
    report_path = ROOT / relative_report
    if not report_path.parent.is_dir() or not report_path.is_file():
        raise ValueError("V4h cloned primary report is absent; recovery will not create a replacement")
    expected_report = json.loads(expected_bytes.decode("utf-8"))
    if expected_report.get("outcome_status") != target["required_primary_outcome"]:
        raise ValueError("frozen archive primary outcome differs from the V9 target")
    quarantine_rel = Path(recovery["quarantine_relative_path"])
    if quarantine_rel.is_absolute() or ".." in quarantine_rel.parts:
        raise ValueError("V4h quarantine path is unsafe")
    quarantine_path = ROOT / quarantine_rel
    pre_bytes = report_path.read_bytes()
    pre_sha = _sha_bytes(pre_bytes)
    if pre_sha != expected_sha:
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        if quarantine_path.exists() and quarantine_path.read_bytes() != pre_bytes:
            raise ValueError("V4h quarantine already contains different bytes")
        if not quarantine_path.exists():
            _atomic_write(quarantine_path, pre_bytes)
        if quarantine_path.read_bytes() != pre_bytes:
            raise ValueError("V4h did not preserve the discrepant clone bytes")
        _atomic_write(report_path, expected_bytes)
    post_sha = _sha_file(report_path)
    if post_sha != expected_sha or report_path.stat().st_size != len(expected_bytes):
        raise ValueError("V4h restored report does not match its unique frozen archive member")
    preserved_sha = _sha_file(quarantine_path) if pre_sha != expected_sha else None
    if pre_sha != expected_sha and preserved_sha != pre_sha:
        raise ValueError("V4h preserved-report digest does not match the original clone bytes")
    receipt = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "archive_sha256": recovery["archive_sha256"], "archive_bytes": recovery["archive_bytes"],
        "input_manifest_sha256": freeze["input"]["input_manifest_sha256"],
        "matched_archive_member_count": 1, "archive_member_sha256": expected_sha,
        "primary_report_relative_path": relative_report.as_posix(),
        "pre_repair_report_sha256": pre_sha, "pre_repair_report_bytes": len(pre_bytes),
        "quarantine_relative_path": quarantine_rel.as_posix() if preserved_sha else None,
        "quarantine_sha256": preserved_sha,
        "quarantine_bytes": quarantine_path.stat().st_size if preserved_sha else 0,
        "restored_report_sha256": post_sha, "restored_report_bytes": len(expected_bytes),
        "all_checks_pass": True, "held_out_predictions_or_outcomes_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    _atomic_write(PACKET / "primary-report-repair-v4h.json", (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return receipt

if __name__ == "__main__":
    try:
        result = _recover()
    except Exception as error:
        freeze = _read_json(PACKET / "freeze.json")
        stop = {"protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
                "packet_sha256": freeze.get("packet_sha256"), "stop_status": "INCONCLUSIVE",
                "stopped_phase": "primary_report_recovery", "error_type": type(error).__name__,
                "error": str(error), "held_out_predictions_or_outcomes_opened": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat()}
        stop["stop_sha256"] = _canonical_digest(stop)
        (PACKET / "primary-report-repair-v4h-stop.json").write_text(json.dumps(stop, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"protocol_id": PROTOCOL_ID, "packet_sha256": result["packet_sha256"],
                      "receipt_sha256": result["receipt_sha256"], "all_checks_pass": result["all_checks_pass"],
                      "held_out_predictions_or_outcomes_opened": result["held_out_predictions_or_outcomes_opened"],
                      "pre_repair_report_sha256": result["pre_repair_report_sha256"],
                      "restored_report_sha256": result["restored_report_sha256"]}, sort_keys=True))

