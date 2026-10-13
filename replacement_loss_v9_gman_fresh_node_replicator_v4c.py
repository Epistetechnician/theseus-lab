"""V4c entry point binding the V4 evaluator to a fail-closed continuation.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c"
IMPLEMENTATION_SHA256 = "2ae1cd0f8b813c28d8815c1b1bfab01a46fcd65afa7d0da24aad81a78b22fc97"


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


def _load_implementation():
    packet_root = Path(__file__).resolve().parent
    freeze = json.loads((packet_root / "freeze.json").read_text(encoding="utf-8"))
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V4c freeze canonical digest mismatch")
    implementation_path = packet_root / "implementation.py"
    expected = freeze.get("implementation", {}).get("sha256")
    if expected != IMPLEMENTATION_SHA256 or _sha_file(implementation_path) != IMPLEMENTATION_SHA256:
        raise ValueError("V4c evaluator implementation differs from its frozen digest")
    spec = importlib.util.spec_from_file_location("theseus_v4c_frozen_implementation", implementation_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the digest-bound V4 evaluator implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    module.STATE_SLICE = STATE_SLICE

    original_verify = module._verify_v4_packet

    def verify_packet(workspace: Path):
        packet = original_verify(workspace)
        receipt_path = workspace / "v4" / "transport-preflight-v4c.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if receipt.get("receipt_sha256") != _canonical_digest(receipt_material):
            raise ValueError("V4c artifact transport receipt canonical digest mismatch")
        transfer = packet.get("custody_transfer", {})
        if (
            receipt.get("all_checks_pass") is not True
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("state_slice") != STATE_SLICE
            or receipt.get("packet_sha256") != packet.get("packet_sha256")
            or receipt.get("artifacts") != transfer.get("artifacts")
        ):
            raise ValueError("V4c artifact transport receipt differs from the frozen transfer manifest")
        return packet

    module._verify_v4_packet = verify_packet
    return module


if __name__ == "__main__":
    _load_implementation().main()
