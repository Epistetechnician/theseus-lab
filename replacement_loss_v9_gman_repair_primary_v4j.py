#!/usr/bin/env python3
"""V4j identity adapter for the frozen V4h primary-report recovery.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j, and
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j"
STATE_SLICE = PROTOCOL_ID
ROOT = Path("/home/dev/theseus-v9")
PACKET = ROOT / "v4j"


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_parent() -> Any:
    freeze = json.loads((PACKET / "freeze.json").read_text(encoding="utf-8"))
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material) or freeze.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V4j canonical freeze identity mismatch")
    row = freeze["parent_helpers"]["repair"]
    path = PACKET / row["path"]
    content = path.read_bytes()
    if len(content) != row["bytes"] or hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError("V4h parent repair helper digest mismatch")
    spec = importlib.util.spec_from_file_location("theseus_v4j_parent_repair_v4h", path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the frozen V4h parent recovery helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    module.PACKET = PACKET
    module.PROTOCOL_ID = PROTOCOL_ID
    module.STATE_SLICE = STATE_SLICE
    return module


if __name__ == "__main__":
    parent = _load_parent()
    try:
        result = parent._recover()
    except Exception as error:
        freeze = parent._read_json(PACKET / "freeze.json")
        stop = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": freeze.get("packet_sha256"),
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": "primary_report_recovery",
            "error_type": type(error).__name__,
            "error": str(error),
            "held_out_predictions_or_outcomes_opened": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["stop_sha256"] = _canonical_digest(stop)
        (PACKET / "primary-report-repair-v4h-stop.json").write_text(
            json.dumps(stop, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": result["packet_sha256"],
        "receipt_sha256": result["receipt_sha256"],
        "all_checks_pass": result["all_checks_pass"],
        "held_out_predictions_or_outcomes_opened": result["held_out_predictions_or_outcomes_opened"],
        "restored_report_sha256": result["restored_report_sha256"],
    }, sort_keys=True))
