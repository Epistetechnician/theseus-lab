"""Digest-check GMan artifact relay inputs before V4b evaluator preflight.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4b.
"""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4b"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4b"
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


def _download(url: str, path: Path, expected_size: int, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    digest = hashlib.sha256()
    size = 0
    request = urllib.request.Request(url, headers={"User-Agent": "theseus-v4b-import/1"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            size += len(chunk)
            if size > expected_size:
                raise ValueError("artifact exceeds its frozen byte count")
            digest.update(chunk)
            output.write(chunk)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError("downloaded artifact failed its frozen size or SHA-256 check")
    os.replace(temporary, path)


def _extract_safe(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("artifact archive contains an unsafe path")
            if not (member.isdir() or member.isfile()):
                raise ValueError("artifact archive contains a link or special file")
            target = (destination / Path(*relative.parts)).resolve()
            if target != root and root not in target.parents:
                raise ValueError("artifact member escapes its destination")
        archive.extractall(destination)


def main() -> int:
    packet_root = WORKSPACE / "v4"
    freeze_path = packet_root / "freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V4b freeze canonical digest mismatch")
    importer = freeze.get("importer", {})
    if (
        importer.get("sha256") != _sha_file(Path(__file__))
        or importer.get("bytes") != Path(__file__).stat().st_size
    ):
        raise ValueError("V4b importer differs from its frozen digest")
    transfer = freeze.get("custody_transfer", {})
    if transfer.get("method") != "GMAN_PRIVATE_ARTIFACT_RELAY":
        raise ValueError("V4b custody transfer method mismatch")

    rows = transfer.get("artifacts", [])
    if len(rows) != 8:
        raise ValueError("V4b transfer manifest must contain exactly eight artifacts")
    imported = []
    for row in rows:
        key = row["key"]
        url = os.environ.get("THESEUS_V4B_ARTIFACT_" + key.upper() + "_URL")
        if not url or not url.startswith("https://"):
            raise ValueError("a frozen artifact download URL is missing")
        destination = WORKSPACE / row["destination"]
        staged = packet_root / "transfer-cache" / (key + (".tar" if row["format"] == "tar" else ".blob"))
        _download(url, staged, int(row["size_bytes"]), str(row["sha256"]))
        if row["format"] == "tar":
            _extract_safe(staged, destination)
        elif row["format"] == "file":
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged, destination)
        else:
            raise ValueError("unsupported frozen artifact format")
        imported.append(
            {
                "key": key,
                "artifact_id": row["artifact_id"],
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
                "destination": str(row["destination"]),
                "format": row["format"],
                "verified": True,
            }
        )
        staged.unlink(missing_ok=True)

    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "method": transfer["method"],
        "artifacts": rows,
        "download_and_extraction_checks": imported,
        "all_checks_pass": len(imported) == 8 and all(item["verified"] for item in imported),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    receipt_path = packet_root / "transport-preflight-v4b.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol_id": PROTOCOL_ID, "packet_sha256": freeze["packet_sha256"], "receipt_sha256": receipt["receipt_sha256"], "artifact_count": len(imported), "all_checks_pass": receipt["all_checks_pass"]}, sort_keys=True))
    return 0 if receipt["all_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
