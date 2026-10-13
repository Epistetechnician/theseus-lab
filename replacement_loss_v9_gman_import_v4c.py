"""Verify retained V4b artifacts and the missing V3 document for V4c.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c"
WORKSPACE = Path("/home/dev/theseus-v9")


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


def main() -> int:
    packet_root = WORKSPACE / "v4"
    freeze_path = packet_root / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V4c freeze canonical digest mismatch")
    importer = freeze.get("importer", {})
    if (
        importer.get("sha256") != _sha_file(Path(__file__))
        or importer.get("bytes") != Path(__file__).stat().st_size
    ):
        raise ValueError("V4c importer differs from its frozen digest")

    parent = freeze.get("parent_attempt", {})
    parent_receipt_path = packet_root / "transport-preflight-v4b.json"
    if _sha_file(parent_receipt_path) != parent.get("transport_receipt_file_sha256"):
        raise ValueError("V4b transport receipt file differs from the V4c binding")
    parent_receipt = json.loads(parent_receipt_path.read_text(encoding="utf-8"))
    parent_material = {key: value for key, value in parent_receipt.items() if key != "receipt_sha256"}
    if (
        parent_receipt.get("receipt_sha256") != parent.get("transport_receipt_sha256")
        or parent_receipt.get("receipt_sha256") != _canonical_digest(parent_material)
        or parent_receipt.get("packet_sha256") != parent.get("packet_sha256")
        or parent_receipt.get("all_checks_pass") is not True
    ):
        raise ValueError("terminal V4b custody receipt is not a passing bound receipt")

    artifacts = freeze.get("custody_transfer", {}).get("artifacts", [])
    downloaded = [row for row in artifacts if row.get("format") in ("file", "tar")]
    if parent_receipt.get("artifacts") != downloaded or len(downloaded) != 8:
        raise ValueError("V4c reused artifact manifest differs from the verified V4b transfer")

    prior_attempt_files = parent.get("stop_files", [])
    for row in prior_attempt_files:
        path = packet_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError("terminal V4b stop receipt differs from the V4c binding")

    checked = []
    for row in artifacts:
        destination = WORKSPACE / row["destination"]
        if row.get("format") == "preinstalled_file":
            if not destination.is_file() or destination.stat().st_size != row["size_bytes"] or _sha_file(destination) != row["sha256"]:
                raise ValueError("supplemental frozen V3 document failed its byte digest")
        elif row.get("format") == "file":
            if not destination.is_file():
                raise FileNotFoundError("previously verified V4b file is missing")
        elif row.get("format") == "tar":
            if not destination.is_dir():
                raise FileNotFoundError("previously verified V4b directory is missing")
        else:
            raise ValueError("unsupported V4c custody artifact format")
        checked.append(
            {
                "key": row["key"],
                "artifact_id": row.get("artifact_id"),
                "size_bytes": row["size_bytes"],
                "sha256": row["sha256"],
                "destination": row["destination"],
                "format": row["format"],
                "verified": True,
            }
        )

    archive = WORKSPACE / "input/gman-v9-input.tar.gz"
    archive_sha = freeze["input"]["archive_sha256"]
    if not archive.is_file() or _sha_file(archive) != archive_sha:
        raise ValueError("frozen V9 input archive differs from the V4c digest")
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "method": freeze["custody_transfer"]["method"],
        "parent_v4b_packet_sha256": parent["packet_sha256"],
        "parent_v4b_transport_receipt_sha256": parent_receipt["receipt_sha256"],
        "artifacts": artifacts,
        "reuse_and_supplement_checks": checked,
        "input_archive_sha256": archive_sha,
        "all_checks_pass": len(checked) == 9,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    receipt_path = packet_root / "transport-preflight-v4c.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "packet_sha256": freeze["packet_sha256"], "receipt_sha256": receipt["receipt_sha256"], "artifact_count": len(checked), "all_checks_pass": receipt["all_checks_pass"]}, sort_keys=True))
    return 0 if receipt["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
