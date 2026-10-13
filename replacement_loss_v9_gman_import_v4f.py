#!/usr/bin/env python3
"""Verify V4f custody and terminal V4e parent evidence before evaluation.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/dev/theseus-v9")
PACKET = ROOT / "v4f"
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f"
STATE_SLICE = PROTOCOL_ID
V4E_PACKET_SHA256 = "7b24806d2a6d3c97ca9579e10c1c0136a5415c08d940b69583cfa0f73886e914"
V4E_FREEZE_FILE_SHA256 = "44cfcbe29e5fdd000278304da8199c7910c858501c3ae3fcbd1ed3118c2ceb93"
V4E_STOP_FILE_SHA256 = "265f7dc9f966267a431d009f073f9ca8de3f8e36813bf5e966e0535d685cff4d"
V4E_STOP_DIGEST = "4c6ed337a6ccd555ac983afedc274c3aca9f33264165843b9024cceab3bfafb5"
V4E_TRANSPORT_FILE_SHA256 = "3192dbaada484be1b02975ad7c1078f72e966066c2ae870551665aa2a0b34cab"
V4E_TRANSPORT_DIGEST = "5776d2639ddb835e79718f215cf01ae0f65fa52f8a2d2ddd88c322a9f3ca12eb"
V4_SOURCE_SHA256 = "2ae1cd0f8b813c28d8815c1b1bfab01a46fcd65afa7d0da24aad81a78b22fc97"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _verify_packet_files(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    checked = []
    for key, filename in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
        ("importer", "importer.py"),
    ):
        row = freeze[key]
        path = PACKET / filename
        if (
            not path.is_file()
            or path.stat().st_size != row["bytes"]
            or _sha_file(path) != row["sha256"]
        ):
            raise ValueError(f"V4f frozen file mismatch: {key}")
        checked.append({"key": key, "bytes": row["bytes"], "sha256": row["sha256"], "verified": True})
    return checked


def _verify_v4e_parent(freeze: dict[str, Any]) -> dict[str, Any]:
    parent = freeze["parent_v4e"]
    parent_root = ROOT / "v4e"
    freeze_path = parent_root / "freeze.json"
    if not freeze_path.is_file() or _sha_file(freeze_path) != V4E_FREEZE_FILE_SHA256:
        raise ValueError("terminal V4e freeze file digest mismatch")
    parent_freeze = _read_json(freeze_path)
    material = {key: value for key, value in parent_freeze.items() if key != "packet_sha256"}
    if (
        parent_freeze.get("packet_sha256") != V4E_PACKET_SHA256
        or parent_freeze.get("packet_sha256") != _canonical_digest(material)
        or parent_freeze.get("source_snapshot", {}).get("snapshot_id") != "snap-t3r2t"
        or parent.get("packet_sha256") != V4E_PACKET_SHA256
    ):
        raise ValueError("terminal V4e freeze identity mismatch")

    stop_path = parent_root / "v4-preflight-stop.json"
    if not stop_path.is_file() or _sha_file(stop_path) != V4E_STOP_FILE_SHA256:
        raise ValueError("V4e terminal preflight stop file digest mismatch")
    stop = _read_json(stop_path)
    stop_material = {key: value for key, value in stop.items() if key != "v4_stop_sha256"}
    if (
        stop.get("v4_stop_sha256") != V4E_STOP_DIGEST
        or stop.get("v4_stop_sha256") != _canonical_digest(stop_material)
        or stop.get("packet_sha256") != V4E_PACKET_SHA256
        or stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("stopped_phase") != "preflight"
        or stop.get("error_type") != "ValueError"
        or "installed distributions differ" not in str(stop.get("error", ""))
        or (parent_root / "v4-run-receipt.json").exists()
    ):
        raise ValueError("V4e was not a terminal preflight-only stop")

    transport_path = parent_root / "transport-preflight-v4e.json"
    if not transport_path.is_file() or _sha_file(transport_path) != V4E_TRANSPORT_FILE_SHA256:
        raise ValueError("V4e transport receipt file digest mismatch")
    transport = _read_json(transport_path)
    transport_material = {key: value for key, value in transport.items() if key != "receipt_sha256"}
    if (
        transport.get("receipt_sha256") != V4E_TRANSPORT_DIGEST
        or transport.get("receipt_sha256") != _canonical_digest(transport_material)
        or transport.get("all_checks_pass") is not True
        or transport.get("held_out_predictions_or_outcomes_opened") is not False
        or transport.get("packet_sha256") != V4E_PACKET_SHA256
    ):
        raise ValueError("V4e transport receipt did not pass while held-out data stayed sealed")

    for key, filename in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
        ("importer", "importer.py"),
    ):
        row = parent_freeze[key]
        path = parent_root / filename
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V4e frozen parent file differs: {key}")

    implementation = ROOT / "v4" / "implementation.py"
    if not implementation.is_file() or _sha_file(implementation) != V4_SOURCE_SHA256:
        raise ValueError("digest-bound V4 implementation differs from the frozen parent")

    return {
        "packet_sha256": V4E_PACKET_SHA256,
        "freeze_file_sha256": V4E_FREEZE_FILE_SHA256,
        "source_snapshot_id": "snap-t3r2t",
        "stop_file_sha256": V4E_STOP_FILE_SHA256,
        "stop_receipt_digest": V4E_STOP_DIGEST,
        "transport_file_sha256": V4E_TRANSPORT_FILE_SHA256,
        "transport_receipt_digest": V4E_TRANSPORT_DIGEST,
        "evaluator_started": False,
        "held_out_predictions_or_outcomes_opened": False,
    }


def _verify() -> dict[str, Any]:
    freeze = _read_json(PACKET / "freeze.json")
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if (
        freeze.get("packet_sha256") != _canonical_digest(material)
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("state_slice") != STATE_SLICE
        or freeze.get("source_snapshot", {}).get("snapshot_id") != "snap-t3r2t"
        or freeze.get("runtime", {}).get("dependency_manifest_sha256")
        != "0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63"
    ):
        raise ValueError("V4f freeze identity, runtime, or canonical digest mismatch")
    packet_checks = _verify_packet_files(freeze)
    parent = _verify_v4e_parent(freeze)

    for row in freeze["custody_transfer"]["artifacts"]:
        path = ROOT / row["destination"]
        format_name = row["format"]
        if format_name == "preinstalled_file":
            okay = path.is_file() and path.stat().st_size == row["size_bytes"] and _sha_file(path) == row["sha256"]
        elif format_name == "file":
            okay = path.is_file()
        elif format_name == "tar":
            okay = path.is_dir()
        else:
            raise ValueError(f"unsupported frozen custody format: {format_name}")
        if not okay:
            raise ValueError(f"frozen V9 custody artifact missing: {row['key']}")

    input_spec = freeze["input"]
    archive = ROOT / "input" / "gman-v9-input.tar.gz"
    manifest = ROOT / "input" / "metadata" / "input-manifest.json"
    lock = ROOT / "input" / "source-checkout" / "requirements-v9.lock.txt"
    if (
        not archive.is_file()
        or _sha_file(archive) != input_spec["archive_sha256"]
        or not manifest.is_file()
        or _sha_file(manifest) != input_spec["input_manifest_sha256"]
        or not lock.is_file()
        or _sha_file(lock) != freeze["target_lock"]["sha256"]
        or lock.stat().st_size != freeze["target_lock"]["bytes"]
    ):
        raise ValueError("V4f V9 input archive, manifest, or target lock mismatch")

    implementation = ROOT / "v4" / "implementation.py"
    if _sha_file(implementation) != freeze["implementation"]["sha256"]:
        raise ValueError("V4 implementation digest differs from V4f freeze")
    v3_freeze = ROOT / "v3" / "freeze.json"
    if not v3_freeze.is_file() or _sha_file(v3_freeze) != freeze["v3_freeze_file_sha256"]:
        raise ValueError("V3 freeze differs from V4f binding")

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "parent_v4e": parent,
        "source_snapshot": freeze["source_snapshot"],
        "packet_file_checks": packet_checks,
        "reused_artifacts": [
            {"key": row["key"], "destination": row["destination"], "sha256": row["sha256"], "verified": True}
            for row in freeze["custody_transfer"]["artifacts"]
        ],
        "input_archive_sha256": input_spec["archive_sha256"],
        "input_manifest_sha256": input_spec["input_manifest_sha256"],
        "target_lock_sha256": _sha_file(lock),
        "implementation_sha256": _sha_file(implementation),
        "all_checks_pass": True,
        "held_out_predictions_or_outcomes_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    (PACKET / "transport-preflight-v4f.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


if __name__ == "__main__":
    result = _verify()
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": result["packet_sha256"],
        "receipt_sha256": result["receipt_sha256"],
        "all_checks_pass": result["all_checks_pass"],
        "held_out_predictions_or_outcomes_opened": result["held_out_predictions_or_outcomes_opened"],
    }, sort_keys=True))

