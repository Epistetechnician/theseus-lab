"""Prepare a minimal, digest-checked build context for the v8 Cloud Run replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_FILES = (
    ("cloud_run_v8_replay/Dockerfile", "Dockerfile"),
    ("cloud_run_v8_replay/cloud_run_entrypoint.py", "cloud_run_entrypoint.py"),
)
BASE_IMAGE = "python:3.12.13-slim-bookworm@sha256:6e13e65c55e33adf203d77ee371cf8bf5d81bd4902ef07565721f46bf44917af"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(raw)


def prepare(output: Path) -> dict[str, Any]:
    freeze_path = ROOT / "docs" / "replacement-loss-v8-freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    for frozen in freeze["packet_files"]:
        relative = str(frozen["path"])
        source = ROOT / relative
        data = source.read_bytes()
        digest = _sha256(data)
        if len(data) != frozen["bytes"] or digest != frozen["sha256"]:
            raise ValueError(f"frozen v8 source mismatch: {relative}")
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append({"path": relative, "bytes": len(data), "sha256": digest})

    freeze_relative = "docs/replacement-loss-v8-freeze.json"
    freeze_bytes = freeze_path.read_bytes()
    freeze_destination = output / freeze_relative
    freeze_destination.parent.mkdir(parents=True, exist_ok=True)
    freeze_destination.write_bytes(freeze_bytes)
    rows.append({
        "path": freeze_relative,
        "bytes": len(freeze_bytes),
        "sha256": _sha256(freeze_bytes),
    })

    for source_relative, destination_relative in ADAPTER_FILES:
        source = ROOT / source_relative
        data = source.read_bytes()
        destination = output / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        rows.append({
            "path": destination_relative,
            "bytes": len(data),
            "sha256": _sha256(data),
        })

    rows.sort(key=lambda row: row["path"])
    payload: dict[str, Any] = {
        "schema": "theseus-v8-cloud-run-container-source-v1",
        "state_slices": [
            "theseus-lab-sandbox-v1",
            "global-experiment-assurance-lifecycle-v1",
        ],
        "protocol_id": "theseus-replacement-loss-v8-ridge-primary-causal",
        "v8_packet_sha256": freeze["packet_sha256"],
        "base_image": BASE_IMAGE,
        "source_files": rows,
    }
    payload["container_source_sha256"] = _canonical_sha256(payload)
    (output / "container-source-manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = prepare(args.output.expanduser().resolve())
    print(json.dumps({
        "schema": result["schema"],
        "v8_packet_sha256": result["v8_packet_sha256"],
        "container_source_sha256": result["container_source_sha256"],
        "source_file_count": len(result["source_files"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
