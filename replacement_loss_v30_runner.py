"""V30 recovery wrapper around the frozen V29 evaluator.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v30-nonlinear-synthetic-recovery-v1, and
global-experiment-assurance-lifecycle-v1. V30 changes only fixture tensor
dtype normalization and binds the failed V29 packet/receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import replacement_loss_v29_runner as core


ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path("/Volumes/PrimaryED/ResearchArtifacts")
V29_ROOT = ARTIFACTS / "theseus-lab-sandbox-v1-replacement-loss-v29"
V30_ROOT = ARTIFACTS / "theseus-lab-sandbox-v1-replacement-loss-v30"
V30_REPLICA = ARTIFACTS / "theseus-lab-sandbox-v1-replacement-loss-v30-replicator"
V29_FREEZE = ROOT / "docs/replacement-loss-v29-freeze.json"
V29_RESULTS = ROOT / "docs/replacement-loss-v29-results.md"
V29_FAILURE = V29_ROOT / "run-failure-receipt.json"
FREEZE_LOCAL = ROOT / "docs/replacement-loss-v30-freeze.json"
PROTOCOL_ID = "theseus-replacement-loss-v30-float32-recovery"
CONTINUATION_SLICE = "theseus-replacement-loss-v30-nonlinear-synthetic-recovery-v1"
PACKET_FILES = (
    "docs/replacement-loss-v30-float32-recovery.md",
    "replacement_loss_v30_runner.py",
    "requirements-v30.lock.txt",
    "replacement_loss_v29_runner.py",
    "docs/replacement-loss-v29-nonlinear-synthetic.md",
    "docs/replacement-loss-v29-freeze.json",
    "docs/replacement-loss-v29-results.md",
)
_ORIGINAL_SAMPLE_INPUTS = core._sample_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: Any) -> str:
    return core._digest(value)


def _normalize_float32_inputs(rng: np.random.Generator, stratum: str, count: int) -> np.ndarray:
    return np.asarray(_ORIGINAL_SAMPLE_INPUTS(rng, stratum, count), dtype=np.float32)


def _verify_v29_upstream() -> dict[str, Any]:
    local = json.loads(V29_FREEZE.read_text())
    external = json.loads((V29_ROOT / "freeze.json").read_text())
    if local != external or local.get("packet_sha256") != _digest({key: value for key, value in local.items() if key != "packet_sha256"}):
        raise ValueError("V29 frozen packet identity is invalid")
    for row in local["packet_files"]:
        relative = row["path"]
        if _sha(ROOT / relative) != row["sha256"] or _sha(V29_ROOT / "frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V29 upstream frozen source changed: {relative}")
    failure = json.loads(V29_FAILURE.read_text())
    if failure.get("packet_sha256") != local["packet_sha256"] or failure.get("status") != "INCONCLUSIVE":
        raise ValueError("V29 failure receipt is missing or does not bind its frozen packet")
    return {"packet_sha256": local["packet_sha256"], "failure_receipt_sha256": _sha(V29_FAILURE)}


# Rebind the frozen engine to the new packet and custody. The only algorithmic
# override is an explicit float32 conversion at the fixture boundary.
core.PROTOCOL_ID = PROTOCOL_ID
core.CONTINUATION_SLICE = CONTINUATION_SLICE
core.PRIMARY = V30_ROOT
core.REPLICA = V30_REPLICA
core.FREEZE_LOCAL = FREEZE_LOCAL
core.PACKET_FILES = PACKET_FILES
core._sample_inputs = _normalize_float32_inputs


def freeze() -> dict[str, Any]:
    upstream = _verify_v29_upstream()
    result = core.freeze()
    manifest = json.loads(FREEZE_LOCAL.read_text())
    manifest["upstream_v29_packet_sha256"] = upstream["packet_sha256"]
    manifest["upstream_v29_failure_receipt_sha256"] = upstream["failure_receipt_sha256"]
    manifest["recovery_change"] = "normalize every sampled input array to float32 before module execution"
    manifest.pop("packet_sha256", None)
    manifest["packet_sha256"] = _digest(manifest)
    (V30_ROOT / "freeze.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    FREEZE_LOCAL.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return {**result, "packet_sha256": manifest["packet_sha256"], "upstream_v29_packet_sha256": upstream["packet_sha256"]}


def verify() -> dict[str, Any]:
    upstream = _verify_v29_upstream()
    manifest = core.verify(V30_ROOT)
    if manifest.get("upstream_v29_packet_sha256") != upstream["packet_sha256"] or manifest.get("upstream_v29_failure_receipt_sha256") != upstream["failure_receipt_sha256"]:
        raise ValueError("V30 packet does not bind the exact V29 failure and source")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "verify", "run", "replicate"))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze()
    elif args.command == "verify":
        result = {"packet_sha256": verify()["packet_sha256"], "verified": True}
    elif args.command == "run":
        verify()
        result = core.run_primary()
    else:
        verify()
        result = core.replicate(V30_ROOT, V30_REPLICA)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
