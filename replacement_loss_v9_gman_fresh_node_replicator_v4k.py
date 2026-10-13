#!/usr/bin/env python3
"""V4k entry point loading the digest-bound V4h scientific implementation.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4k, and
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4k"
STATE_SLICE = PROTOCOL_ID
PACKET_NAME = "v4k"


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_parent() -> Any:
    packet = Path(__file__).resolve().parent
    freeze = json.loads((packet / "freeze.json").read_text(encoding="utf-8"))
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if (
        freeze.get("packet_sha256") != _canonical_digest(material)
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("state_slice") != STATE_SLICE
    ):
        raise ValueError("V4k canonical freeze identity mismatch")
    row = freeze["parent_helpers"]["runner"]
    path = packet / row["path"]
    content = path.read_bytes()
    if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("V4h parent runner helper digest mismatch")
    spec = importlib.util.spec_from_file_location("theseus_v4k_parent_runner_v4h", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the frozen V4h parent runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    module.STATE_SLICE = STATE_SLICE
    module.PACKET_NAME = PACKET_NAME
    return module


if __name__ == "__main__":
    _load_parent()._load_implementation().main()
