#!/usr/bin/env python3
"""Write a digest-only V4i stop receipt after an evaluator timeout or error.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i, and
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i"
STATE_SLICE = PROTOCOL_ID


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _is_canonical_record(value: dict[str, Any], digest_key: str) -> bool:
    return value.get(digest_key) == _canonical_digest({key: item for key, item in value.items() if key != digest_key})


def _replica_root(workspace: Path, target_packet_sha: str) -> tuple[Path | None, list[Path]]:
    excluded = {"input", "venv", ".venv", ".cache", "__pycache__", "v4i", "v4e", "v4f", "v4g", "v4h"}
    matches: list[tuple[str, Path]] = []
    for path in workspace.rglob("candidate-preparation.json"):
        relative = path.relative_to(workspace)
        if any(part.lower() in excluded for part in relative.parts):
            continue
        if not any("replica" in part.lower() for part in relative.parts[:-1]):
            continue
        prep = _read_json(path)
        if prep and prep.get("packet_sha256") == target_packet_sha:
            matches.append((str(prep.get("created_at_utc", "")), path.parent))
    matches.sort(key=lambda row: (row[0], row[1].as_posix()))
    return (matches[-1][1] if matches else None), [row[1] for row in matches]


def _prediction_inventory(root: Path | None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if root is not None:
        for path in sorted(root.rglob("*.npz")):
            if not path.is_file():
                continue
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    material = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return {
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "inventory_sha256": hashlib.sha256(material).hexdigest(),
    }


def build_receipt(workspace: Path, exit_code: int, timeout_kind: str, timeout_seconds: int, evaluator_started: bool,
                  evaluator_started_utc: str | None, evaluator_ended_utc: str | None) -> dict[str, Any]:
    packet = workspace / "v4i"
    packet_freeze = _read_json(packet / "freeze.json")
    if not packet_freeze:
        raise ValueError("V4i freeze is missing")
    material = {key: value for key, value in packet_freeze.items() if key != "packet_sha256"}
    if (
        packet_freeze.get("packet_sha256") != _canonical_digest(material)
        or packet_freeze.get("protocol_id") != PROTOCOL_ID
        or packet_freeze.get("state_slice") != STATE_SLICE
    ):
        raise ValueError("V4i freeze identity or canonical digest mismatch")
    target_sha = packet_freeze["target"]["v9_packet_sha256"]
    replica, roots = _replica_root(workspace, target_sha)
    prediction = _prediction_inventory(replica)
    prep = _read_json(replica / "candidate-preparation.json") if replica else None
    prep_valid = bool(prep and prep.get("packet_sha256") == target_sha and _is_canonical_record(prep, "candidate_preparation_sha256"))
    candidate_freeze = _read_json(replica / "candidate-freeze.json") if replica else None
    candidate_frozen = bool(
        candidate_freeze and candidate_freeze.get("packet_sha256") == target_sha
        and _is_canonical_record(candidate_freeze, "candidate_freeze_sha256")
    )
    challenge = _read_json(replica / "challenge-manifest.json") if replica else None
    challenge_revealed = bool(
        challenge and candidate_frozen
        and challenge.get("candidate_freeze_sha256") == _sha_file(replica / "candidate-freeze.json")
        and (challenge.get("revealed_after_replica_candidate_freeze") is True
             or challenge.get("challenge_generation_started_after_candidate_freeze") is True)
    )
    prediction_manifest = bool(replica and (replica / "prediction-manifest.json").is_file())
    outcome_manifest = bool(replica and (replica / "outcome-manifest.json").is_file())
    replication_manifest = bool(replica and (replica / "replication-manifest.json").is_file())
    evaluation_status = (
        "completed" if outcome_manifest else
        "not_reached_by_frozen_call_order" if not prediction_manifest else
        "unknown_or_incomplete"
    )
    budget_guard = exit_code == 125 and not evaluator_started
    verdict = "INCONCLUSIVE_TIMEOUT" if exit_code == 124 or budget_guard else "INCONCLUSIVE_EVALUATOR_ERROR"
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet_freeze["packet_sha256"],
        "parent_v4h_packet_sha256": packet_freeze["parent_v4h"]["packet_sha256"],
        "reproducibility_status": verdict,
        "assurance_level": "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "stop_reason": (
            "node_budget_guard_before_evaluator" if budget_guard else
            f"{timeout_kind}_timeout_exit_124" if exit_code == 124 else
            "evaluator_error_or_early_stop"
        ),
        "timeout_kind": timeout_kind,
        "evaluator_started": evaluator_started,
        "observed_exit_code": exit_code,
        "timeout_limit_seconds": timeout_seconds,
        "evaluator_started_utc": evaluator_started_utc,
        "evaluator_ended_utc": evaluator_ended_utc,
        "candidate_preparation_completed": prep_valid,
        "candidate_freeze_completed": candidate_frozen,
        "challenge_fixtures_revealed_after_candidate_freeze": challenge_revealed and candidate_frozen,
        "prediction_inventory": prediction,
        "prediction_manifest_present": prediction_manifest,
        "replica_outcome_manifest_present": outcome_manifest,
        "replication_manifest_present": replication_manifest,
        "outcome_evaluation_status": evaluation_status,
        "replica_root_match_count": len(roots),
        "model_prompts_outputs_weights_and_evaluation_data_exported": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/dev/theseus-v9"))
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--timeout-kind", choices=("evaluator", "watchdog"), required=True)
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--evaluator-started", choices=("true", "false"), required=True)
    parser.add_argument("--started-utc")
    parser.add_argument("--ended-utc")
    args = parser.parse_args()
    receipt = build_receipt(args.workspace_root, args.exit_code, args.timeout_kind, args.timeout_seconds,
                            args.evaluator_started == "true", args.started_utc, args.ended_utc)
    receipt["receipt_sha256"] = _canonical_digest(receipt)
    path = args.workspace_root / "v4i" / "v4i-stop-receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result_path = os.environ.get("GMN_RESULT_PATH")
    if result_path:
        Path(result_path).write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in (
        "protocol_id", "packet_sha256", "reproducibility_status", "timeout_kind", "observed_exit_code",
        "prediction_inventory", "outcome_evaluation_status", "receipt_sha256",
    )}, sort_keys=True))


if __name__ == "__main__":
    main()
