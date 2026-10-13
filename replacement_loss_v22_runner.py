"""Frozen lifecycle for the V22 fresh-checkpoint replication.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v22-fresh-checkpoint-replication-v1, and
global-experiment-assurance-lifecycle-v1. Model and prompt data remain in
external owner custody. This runner delegates the unchanged treatment and
evaluator to the frozen V8 implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import replacement_loss_v8_runner as v8
import replacement_loss_v22_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v22-fresh-checkpoint-replication-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
V8_PACKET_SHA256 = "b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2"
PACKET_FILES = (
    "docs/replacement-loss-v22-fresh-checkpoint-replication.md",
    "replacement_loss_v22_fixtures.py",
    "replacement_loss_v22_runner.py",
    "requirements-v22.lock.txt",
)
ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V22_ROOT_NAME = "theseus-lab-sandbox-v1-replacement-loss-v22"
V22_FREEZE_PATH = "docs/replacement-loss-v22-freeze.json"
PRIOR_ROOT_PATTERN = re.compile(r"^theseus-lab-sandbox-v1-replacement-loss-v(?:[3-9]|[1-9][0-9])[a-z0-9-]*$")
PRIOR_SKIP_DIRS = {
    ".git", ".venv", "site-packages", "__pycache__", "models", "model",
    "weights", "candidates", "captures", "adaptation-captures", "predictions",
    "outcomes", "challenges", "prompts", "outputs", "frozen-packet",
    "v8-frozen-packet", "vendor",
}


def _root() -> Path:
    return Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _custody_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("V22 custody must be outside the repository")
    return root


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        source = _root() / relative
        if not source.is_file():
            raise FileNotFoundError(f"V22 packet input is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha(source), "bytes": source.stat().st_size})
    return rows


def freeze_packet(
    custody_root: str | Path,
    v8_custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    """Freeze V22 code and protocol before any fresh target inventory is read."""

    root = _custody_root(custody_root)
    v8_root = _custody_root(v8_custody_root)
    pinned_v7_root = _custody_root(v7_custody_root)
    local_freeze = _root() / V22_FREEZE_PATH
    external_manifest = root / "v22-frozen-packet-manifest.json"
    archive = root / "v22-frozen-packet"
    if local_freeze.exists() or external_manifest.exists() or archive.exists():
        raise FileExistsError("V22 packet is immutable; start a new version to change it")
    v8_receipt = v8.verify_packet(v8_root, pinned_v7_root)
    if v8_receipt.get("packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("upstream V8 packet is not the frozen positive-control packet")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("V22 custody root must be new or empty before freeze")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(v8_root / "v8-frozen-packet-manifest.json", root / "v8-frozen-packet-manifest.json")
    shutil.copytree(v8_root / "v8-frozen-packet", root / "v8-frozen-packet")
    source_rows = _source_rows()
    for row in source_rows:
        source = _root() / str(row["path"])
        target = archive / str(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    v8_freeze_path = _root() / "docs/replacement-loss-v8-freeze.json"
    v8_freeze = _read_json(v8_freeze_path)
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "created_at_utc": _now(),
        "freeze_state": "FROZEN_BEFORE_V22_CHECKPOINT_INVENTORY_OR_SELECTION",
        "packet_files": source_rows,
        "upstream_v8_packet_sha256": V8_PACKET_SHA256,
        "upstream_v8_freeze_file_sha256": _sha(v8_freeze_path),
        "upstream_v8_packet_file_count": len(v8_freeze["packet_files"]),
        "checkpoint_identities_selected": False,
        "candidate_preparation_started": False,
        "challenge_fixtures_generated": False,
        "method": "exact_frozen_v8_runner_with_v22_fixtures_and_post_freeze_checkpoint_manifest",
        "treatment_thresholds_controls": "unchanged_frozen_v8_packet",
        "runtime": {
            "python": "3.12.13",
            "platform": "local macOS arm64 CPU only",
            "dependencies": "requirements-v22.lock.txt",
            "clean_replica": "separate Python 3.12.13 environment with identical dependency inventory",
        },
        "resource_caps": {
            "max_case_wall_seconds": 7200,
            "max_total_wall_seconds": 64800,
            "max_peak_rss_bytes": 25769803776,
            "monetary_budget_usd": 0,
        },
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": [
            "unseen architecture families", "all architectures", "whole-model identity",
            "independent science", "consciousness", "alignment", "production readiness",
        ],
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    _write_json(external_manifest, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "upstream_v8_packet_sha256": V8_PACKET_SHA256,
        "packet_file_count": len(source_rows),
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
    }


def verify_packet(
    custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    pinned_v7_root = _custody_root(v7_custody_root)
    local_manifest = _read_json(_root() / V22_FREEZE_PATH)
    external_manifest = _read_json(root / "v22-frozen-packet-manifest.json")
    if local_manifest != external_manifest:
        raise ValueError("local and custodied V22 freeze manifests differ")
    if local_manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V22 protocol identity mismatch")
    expected = _canonical_digest({key: value for key, value in local_manifest.items() if key != "packet_sha256"})
    if local_manifest.get("packet_sha256") != expected:
        raise ValueError("V22 packet digest is invalid")
    for row in local_manifest["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen V22 source changed: {relative}")
        if _sha(root / "v22-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"archived V22 source changed: {relative}")
    v8_receipt = v8.verify_packet(root, pinned_v7_root)
    if v8_receipt.get("packet_sha256") != local_manifest.get("upstream_v8_packet_sha256"):
        raise ValueError("upstream V8 method packet identity mismatch")
    if _sha(_root() / "docs/replacement-loss-v8-freeze.json") != local_manifest.get("upstream_v8_freeze_file_sha256"):
        raise ValueError("upstream V8 freeze manifest changed after V22 freeze")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": local_manifest["packet_sha256"],
        "upstream_v8_packet_sha256": v8_receipt["packet_sha256"],
        "packet_file_count": len(local_manifest["packet_files"]),
        "verified": True,
    }


def _prior_experiment_roots() -> list[Path]:
    if not ARTIFACTS_ROOT.is_dir():
        return []
    roots = []
    for path in ARTIFACTS_ROOT.iterdir():
        if path.is_dir() and path.name != V22_ROOT_NAME and PRIOR_ROOT_PATTERN.match(path.name):
            roots.append(path)
    return sorted(roots)


def _prior_fixture_roots() -> list[Path]:
    selected: list[Path] = []
    for root in _prior_experiment_roots():
        for name in ("fixtures", "challenges", "prompts"):
            path = root / name
            if path.is_dir():
                selected.append(path)
    return selected


def _identity_values(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"model_id", "repo_id", "checkpoint_id"} and isinstance(child, str):
                normalized = child.strip().lower()
                if normalized:
                    found.add(normalized)
            found.update(_identity_values(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_identity_values(child))
    return found


def _prior_checkpoint_ledger() -> tuple[set[str], list[dict[str, Any]]]:
    identities = {str(value).lower() for value in v8.PRIOR_CHECKPOINT_IDS}
    rows: list[dict[str, Any]] = []
    for root in _prior_experiment_roots():
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in PRIOR_SKIP_DIRS]
            for filename in files:
                if not filename.endswith(".json"):
                    continue
                path = Path(current) / filename
                try:
                    if path.stat().st_size > 8 * 1024 * 1024:
                        continue
                    raw = path.read_bytes()
                    if b'"model_id"' not in raw and b'"repo_id"' not in raw and b'"checkpoint_id"' not in raw:
                        continue
                    parsed = json.loads(raw)
                except (OSError, json.JSONDecodeError):
                    continue
                values = _identity_values(parsed)
                if values:
                    identities.update(values)
                    rows.append({
                        "path": str(path.relative_to(ARTIFACTS_ROOT)),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "identity_count": len(values),
                    })
    rows.sort(key=lambda row: str(row["path"]))
    return identities, rows


@contextmanager
def _with_v22_profile(prior_ids: set[str] | None = None) -> Iterator[None]:
    old_fixtures = v8.fixtures
    old_ids = v8.PRIOR_CHECKPOINT_IDS
    try:
        v8.fixtures = fixtures
        if prior_ids is not None:
            v8.PRIOR_CHECKPOINT_IDS = set(old_ids) | set(prior_ids)
        yield
    finally:
        v8.fixtures = old_fixtures
        v8.PRIOR_CHECKPOINT_IDS = old_ids


def write_checkpoint_manifest(
    custody_root: str | Path,
    checkpoint_specs: list[dict[str, Any]],
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root, v7_custody_root)
    freeze = _read_json(_root() / V22_FREEZE_PATH)
    if datetime.now(timezone.utc) <= datetime.fromisoformat(str(freeze["created_at_utc"])):
        raise ValueError("fresh checkpoint selection must occur after V22 freeze")
    prior_ids, ledger_rows = _prior_checkpoint_ledger()
    if len(checkpoint_specs) != 3 or {str(item.get("family")) for item in checkpoint_specs} != set(v8.v3.FAMILIES):
        raise ValueError("select exactly one checkpoint per frozen OPT, GPT-2, and Llama family")
    proposed_ids = {str(item.get("model_id", "")).strip().lower() for item in checkpoint_specs}
    if "" in proposed_ids or proposed_ids & prior_ids:
        raise ValueError("checkpoint identity is missing or was used in V3-V21 protocol work")
    with _with_v22_profile(prior_ids):
        result = v8.write_checkpoint_manifest(root, checkpoint_specs, v7_custody_root)
    checkpoint_path = root / "checkpoints.json"
    selection = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "selected_at_utc": _now(),
        "selected_after_protocol_freeze": True,
        "not_used_for_protocol_design": True,
        "checkpoint_manifest_sha256": _sha(checkpoint_path),
        "prior_checkpoint_identity_count": len(prior_ids),
        "prior_checkpoint_ledger_source_count": len(ledger_rows),
        "prior_checkpoint_ledger_sha256": _canonical_digest(ledger_rows),
        "selected_checkpoint_count": 3,
    }
    _write_json(root / "v22-checkpoint-selection.json", selection)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "checkpoint_manifest_sha256": _sha(checkpoint_path),
        "selection_receipt_sha256": _sha(root / "v22-checkpoint-selection.json"),
        "checkpoint_count": 3,
        "prior_checkpoint_identity_count": len(prior_ids),
        "prior_checkpoint_ledger_source_count": len(ledger_rows),
    }


def _aggregate_report(root: Path, packet_sha: str, outcome: dict[str, Any]) -> dict[str, Any]:
    inner = _read_json(root / "replacement-loss-v8-report.json")
    selection_path = root / "v22-checkpoint-selection.json"
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "packet_sha256": packet_sha,
        "upstream_v8_packet_sha256": V8_PACKET_SHA256,
        "checkpoint_selection_receipt_sha256": _sha(selection_path),
        "checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
        "outcome_status": inner["outcome_status"],
        "assurance_level": inner["assurance_level"],
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "PENDING_SECOND_CLEAN_RUNTIME",
        "case_count": len(inner["cases"]),
        "cases": [
            {
                "family": row["family"],
                "seed_index": row["seed_index"],
                "seed": row["seed"],
                "selected_layer": row["selected_layer"],
                "case_status": row["case_status"],
                "blockers": row["blockers"],
                "treatment_effects": row["treatment_effects"],
                "treatment_behavior": row["treatment_behavior"],
                "treatment_activation_rescue": row["treatment_activation_rescue"],
                "control_direction_accuracy": row["control_direction_accuracy"],
                "candidate_state_digests": row["candidate_state_digests"],
                "prediction_array_sha256": row["prediction_array_sha256"],
                "outcome_array_sha256": row["outcome_array_sha256"],
                "full_case_elapsed_wall_seconds": row["full_case_elapsed_wall_seconds"],
            }
            for row in inner["cases"]
        ],
        "runtime": inner["runtime"],
        "budget_and_resource_measurements": inner["budget_and_resource_measurements"],
        "data_and_artifact_custody": "model assets, prompts, captures, candidates, predictions, outcomes, and row diagnostics remain in external owner custody",
        "known_limitations": [
            "three fresh checkpoint identities within the same three architecture families",
            "one MLP output projection per checkpoint",
            "synthetic prompts and same-owner reproduction",
            "no independent reviewer, human alignment evaluation, consciousness evidence, or production authorization",
            "monetary cost is zero by protocol; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": [
            "unseen architecture families", "all architectures", "whole-model mechanistic identity",
            "independent science", "consciousness", "alignment", "production readiness",
        ],
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "resource_blockers": inner.get("resource_blockers", []),
        "primary_runner_receipt": outcome,
        "created_at_utc": _now(),
    }
    _write_json(root / "replacement-loss-v22-report.json", report)
    return report


def run_primary(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root, v7_custody_root)
    selection = _read_json(root / "v22-checkpoint-selection.json")
    if selection.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("checkpoint selection does not bind the frozen V22 packet")
    prior_fixture_roots = _prior_fixture_roots()
    if not prior_fixture_roots:
        raise FileNotFoundError("no prior V3-V21 fixture custody roots were discovered")
    prior_ids, _ledger = _prior_checkpoint_ledger()
    with _with_v22_profile(prior_ids):
        outcome = v8.run_primary(root, root / "checkpoints.json", prior_fixture_roots, v7_custody_root)
    report = _aggregate_report(root, packet["packet_sha256"], outcome)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "assurance_level": report["assurance_level"],
        "case_count": report["case_count"],
        "report_sha256": _sha(root / "replacement-loss-v22-report.json"),
    }


def replicate_packet(
    primary_custody_root: str | Path,
    replica_custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    packet = verify_packet(primary, v7_custody_root)
    primary_report_path = primary / "replacement-loss-v22-report.json"
    if not primary_report_path.is_file():
        raise FileNotFoundError("V22 primary report is required before replication")
    primary_report_before = _read_json(primary_report_path)
    selection = _read_json(primary / "v22-checkpoint-selection.json")
    if primary_report_before.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V22 primary report does not bind this packet")
    if selection.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V22 selected checkpoints do not bind this packet")
    if primary_report_before.get("checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V22 primary checkpoint manifest changed after evaluation")
    prior_ids, _ledger = _prior_checkpoint_ledger()
    with _with_v22_profile(prior_ids):
        result = v8.replicate_packet(primary, replica, primary / "checkpoints.json", v7_custody_root)
    v8_report = _read_json(replica / "replacement-loss-v8-replication-report.json")
    primary_report = _read_json(primary_report_path)
    final = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": result.get("assurance_level", "INCONCLUSIVE"),
        "reproducibility_status": result.get("assurance_level", "INCONCLUSIVE"),
        "experiment_outcome_status": primary_report["outcome_status"],
        "reviewer_status": "NOT_SUBMITTED",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "candidate_reconstruction_match": bool(v8_report.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(v8_report.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(v8_report.get("dependencies_match")),
        "distinct_clean_runtime": bool(v8_report.get("distinct_clean_runtime")),
        "source_identity_match": bool(v8_report.get("source_identity_match")),
        "upstream_v8_replication_report_sha256": _sha(replica / "replacement-loss-v8-replication-report.json"),
        "primary_v22_report_sha256_before_reconciliation": _sha(primary_report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "created_at_utc": _now(),
    }
    replica_report = replica / "replacement-loss-v22-replication-report.json"
    _write_json(replica_report, final)
    final["replication_report_sha256"] = _sha(replica_report)
    primary_report["assurance_level"] = final["assurance_level"]
    primary_report["reproducibility_status"] = final["reproducibility_status"]
    primary_report["replication_report_sha256"] = _sha(replica_report)
    _write_json(primary_report_path, primary_report)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": final["assurance_level"],
        "experiment_outcome_status": final["experiment_outcome_status"],
        "replication_report_sha256": _sha(replica_report),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--custody-root", required=True)
    freeze.add_argument("--v8-custody-root", required=True)
    freeze.add_argument("--v7-custody-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--custody-root", required=True)
    verify.add_argument("--v7-custody-root", required=True)
    select = commands.add_parser("select")
    select.add_argument("--custody-root", required=True)
    select.add_argument("--v7-custody-root", required=True)
    select.add_argument("--checkpoint-specs", required=True, help="JSON file held outside the repository")
    run = commands.add_parser("run")
    run.add_argument("--custody-root", required=True)
    run.add_argument("--v7-custody-root", required=True)
    replicate = commands.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", required=True)
    replicate.add_argument("--replica-custody-root", required=True)
    replicate.add_argument("--v7-custody-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root, args.v8_custody_root, args.v7_custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root, args.v7_custody_root)
    elif args.command == "select":
        root = _custody_root(args.custody_root)
        specs_path = Path(args.checkpoint_specs).expanduser().resolve()
        if root not in specs_path.parents:
            raise ValueError("checkpoint specifications must stay in external V22 custody")
        value = json.loads(specs_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("checkpoint specifications must be a JSON list")
        result = write_checkpoint_manifest(root, value, args.v7_custody_root)
    elif args.command == "run":
        result = run_primary(args.custody_root, args.v7_custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root, args.v7_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
