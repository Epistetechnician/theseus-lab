"""Verify terminal V4c custody and V4d bindings before evaluation.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4d.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/home/dev/theseus-v9")
PACKET = ROOT / "v4d"
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4d"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4d"
V4C_PACKET_SHA256 = "94a7b89d8154a59c0d7d878e16897d6e14425a1720bd6d3ec532f7e1c83702fc"


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


def _verify() -> dict[str, Any]:
    freeze_path = PACKET / "freeze.json"
    freeze = _read_json(freeze_path)
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if (
        freeze.get("packet_sha256") != _canonical_digest(material)
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("state_slice") != STATE_SLICE
        or freeze.get("parent_v4c", {}).get("packet_sha256") != V4C_PACKET_SHA256
    ):
        raise ValueError("V4d freeze identity or canonical digest mismatch")
    for key, filename in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
        ("importer", "importer.py"),
    ):
        row = freeze[key]
        path = PACKET / filename
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V4d frozen file failed its byte binding: {key}")

    parent = freeze["parent_v4c"]
    parent_root = ROOT / "v4"
    parent_freeze_path = parent_root / "freeze.json"
    if not parent_freeze_path.is_file() or _sha_file(parent_freeze_path) != parent["freeze_file_sha256"]:
        raise ValueError("V4c freeze file differs from the V4d parent binding")
    parent_freeze = _read_json(parent_freeze_path)
    parent_material = {key: value for key, value in parent_freeze.items() if key != "packet_sha256"}
    if (
        parent_freeze.get("packet_sha256") != V4C_PACKET_SHA256
        or parent_freeze.get("packet_sha256") != _canonical_digest(parent_material)
    ):
        raise ValueError("V4c parent packet is invalid")

    parent_checks = []
    for key, filename in (
        ("transport_receipt", "transport-preflight-v4c.json"),
        ("runtime_receipt", "runtime-preflight-v4.json"),
        ("preflight_receipt", "v4-preflight.json"),
        ("stop_receipt", "v4-run-stop.json"),
    ):
        row = parent[key]
        path = parent_root / filename
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V4c terminal evidence differs from binding: {filename}")
        parent_checks.append({"path": filename, "bytes": row["bytes"], "sha256": row["sha256"], "verified": True})

    v4c_transport = _read_json(parent_root / "transport-preflight-v4c.json")
    transport_material = {key: value for key, value in v4c_transport.items() if key != "receipt_sha256"}
    if (
        v4c_transport.get("receipt_sha256") != parent["transport_receipt_sha256"]
        or v4c_transport.get("receipt_sha256") != _canonical_digest(transport_material)
        or v4c_transport.get("all_checks_pass") is not True
    ):
        raise ValueError("V4c transport receipt digest or status mismatch")

    v4c_preflight = _read_json(parent_root / "v4-preflight.json")
    preflight_material = {key: value for key, value in v4c_preflight.items() if key != "preflight_sha256"}
    if (
        v4c_preflight.get("preflight_sha256") != parent["preflight_receipt_sha256"]
        or v4c_preflight.get("preflight_sha256") != _canonical_digest(preflight_material)
        or v4c_preflight.get("all_checks_pass") is not True
        or v4c_preflight.get("held_out_predictions_or_outcomes_opened") is not False
    ):
        raise ValueError("V4c preflight is not the bound unrevealed passing preflight")

    v4c_stop = _read_json(parent_root / "v4-run-stop.json")
    if (
        v4c_stop.get("v4_stop_sha256") != parent["stop_receipt_digest"]
        or v4c_stop.get("stop_status") != "INCONCLUSIVE"
        or v4c_stop.get("packet_sha256") != V4C_PACKET_SHA256
    ):
        raise ValueError("V4c stop receipt is not the bound terminal result")

    artifacts = freeze["custody_transfer"]["artifacts"]
    checked_artifacts = []
    for row in artifacts:
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
            raise ValueError(f"reused V4c custody artifact is missing: {row['key']}")
        checked_artifacts.append({"key": row["key"], "destination": row["destination"], "sha256": row["sha256"], "verified": True})

    input_spec = freeze["input"]
    archive = ROOT / "input" / "gman-v9-input.tar.gz"
    manifest = ROOT / "input" / "metadata" / "input-manifest.json"
    if not archive.is_file() or _sha_file(archive) != input_spec["archive_sha256"]:
        raise ValueError("frozen V9 input archive differs from V4d")
    if not manifest.is_file() or _sha_file(manifest) != input_spec["input_manifest_sha256"]:
        raise ValueError("frozen V9 input manifest differs from V4d")
    v3_freeze_path = ROOT / "v3" / "freeze.json"
    if not v3_freeze_path.is_file() or _sha_file(v3_freeze_path) != freeze["v3_freeze_file_sha256"]:
        raise ValueError("V3 packet freeze file differs from V4d")
    implementation = parent_root / "implementation.py"
    if _sha_file(implementation) != freeze["implementation"]["sha256"]:
        raise ValueError("V4d implementation reference differs from V4c")

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "parent_v4c_packet_sha256": V4C_PACKET_SHA256,
        "parent_v4c_receipt_checks": parent_checks,
        "artifacts": artifacts,
        "reused_artifacts": checked_artifacts,
        "input_archive_sha256": input_spec["archive_sha256"],
        "input_manifest_sha256": input_spec["input_manifest_sha256"],
        "implementation_sha256": _sha_file(implementation),
        "all_checks_pass": True,
        "held_out_predictions_or_outcomes_opened": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    output = PACKET / "transport-preflight-v4d.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    result = _verify()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "packet_sha256": result["packet_sha256"], "receipt_sha256": result["receipt_sha256"], "artifact_count": len(result["artifacts"]), "all_checks_pass": result["all_checks_pass"]}, sort_keys=True))
