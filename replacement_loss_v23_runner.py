"""Frozen lifecycle for V23 public checkpoint acquisition and replication.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v23-public-checkpoint-acquisition-v1, and
global-experiment-assurance-lifecycle-v1. Public model assets and all experiment
data remain in external owner custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import replacement_loss_v22_runner as v22
import replacement_loss_v23_hub_adapter as hub_adapter


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v23-public-checkpoint-acquisition-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = hub_adapter.PROTOCOL_ID
V22_PACKET_SHA256 = "924be878356660c79ffc3ec4b0aa32f34b3a32926f381cc10a184db459e6179d"
V22_SCREEN_SHA256 = "2acea39522dac40cc2f7fd110ceeca10dd79747b0f2ea8a85fdda754b73e4de6"
PACKET_FILES = (
    "docs/replacement-loss-v23-public-checkpoint-acquisition.md",
    "replacement_loss_v23_hub_adapter.py",
    "replacement_loss_v23_runner.py",
    "requirements-v23.lock.txt",
)
ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V22_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v22"
V23_ROOT_NAME = "theseus-lab-sandbox-v1-replacement-loss-v23"
V23_FREEZE_PATH = "docs/replacement-loss-v23-freeze.json"


def _root() -> Path:
    return Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


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
        raise ValueError("V23 custody must be outside the repository")
    return root


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V23 packet input is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    return rows


def _validate_v22_screen() -> dict[str, Any]:
    receipt_path = V22_ROOT / "v22-local-target-screening.json"
    receipt = _read_json(receipt_path)
    if _sha(receipt_path) != V22_SCREEN_SHA256:
        raise ValueError("V22 no-target screening receipt changed")
    body = {key: value for key, value in receipt.items() if key != "screening_receipt_sha256"}
    if receipt.get("screening_receipt_sha256") != _canonical_digest(body):
        raise ValueError("V22 screening receipt canonical digest is invalid")
    if receipt.get("status") != "NO_ELIGIBLE_FRESH_TARGET":
        raise ValueError("V23 continuation requires the recorded V22 no-target stop")
    return receipt


def freeze_packet(
    custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    pinned_v7_root = _custody_root(v7_custody_root)
    local_freeze = _root() / V23_FREEZE_PATH
    external_manifest = root / "v23-frozen-packet-manifest.json"
    archive = root / "v23-frozen-packet"
    if local_freeze.exists() or external_manifest.exists() or archive.exists():
        raise FileExistsError("V23 packet is immutable; use a new protocol version to change it")
    v22_receipt = v22.verify_packet(V22_ROOT, pinned_v7_root)
    if v22_receipt.get("packet_sha256") != V22_PACKET_SHA256:
        raise ValueError("upstream V22 packet identity changed")
    v22_screen = _validate_v22_screen()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("V23 custody root must be new or empty before freeze")
    root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V22_ROOT / "v22-frozen-packet-manifest.json", root / "v22-frozen-packet-manifest.json")
    shutil.copytree(V22_ROOT / "v22-frozen-packet", root / "v22-frozen-packet")
    shutil.copyfile(V22_ROOT / "v8-frozen-packet-manifest.json", root / "v8-frozen-packet-manifest.json")
    shutil.copytree(V22_ROOT / "v8-frozen-packet", root / "v8-frozen-packet")
    source_rows = _packet_rows()
    for row in source_rows:
        source = _root() / str(row["path"])
        target = archive / str(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_state": "FROZEN_BEFORE_PUBLIC_HUB_QUERY_OR_V23_CHECKPOINT_SELECTION",
        "packet_files": source_rows,
        "upstream_v22_packet_sha256": V22_PACKET_SHA256,
        "upstream_v22_freeze_file_sha256": _sha(_root() / "docs/replacement-loss-v22-freeze.json"),
        "upstream_v22_screen_receipt_sha256": V22_SCREEN_SHA256,
        "upstream_v8_packet_sha256": v22_receipt["upstream_v8_packet_sha256"],
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "treatment_evaluator_thresholds": "unchanged frozen V8 packet mediated by exact V22 runner",
        "dependency_manifest_sha256": _sha(_root() / "requirements-v23.lock.txt"),
        "runtime": {
            "python": "3.12.13",
            "platform": "local macOS arm64 CPU only",
            "network_scope": "public Hub metadata and pinned public model files only",
            "hub_client": "huggingface_hub==0.36.2",
            "auth": "token=False; no local or environment credential access",
        },
        "budget": {
            "monetary_usd": 0,
            "max_model_bytes_each": hub_adapter.MAX_MODEL_BYTES,
            "max_model_bytes_total": hub_adapter.MAX_TOTAL_BYTES,
            "max_case_wall_seconds": 7200,
            "max_total_wall_seconds": 64800,
            "max_peak_rss_bytes": 25769803776,
            "remote_workers": 0,
            "hosted_inference_calls": 0,
        },
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "independent science", "consciousness", "alignment", "production readiness"],
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    _write_json(external_manifest, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "upstream_v22_packet_sha256": V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": payload["upstream_v8_packet_sha256"],
        "packet_file_count": len(source_rows),
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
    }


def verify_packet(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    pinned_v7_root = _custody_root(v7_custody_root)
    local = _read_json(_root() / V23_FREEZE_PATH)
    external = _read_json(root / "v23-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("local and custodied V23 freeze manifests differ")
    expected = _canonical_digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("protocol_id") != PROTOCOL_ID or local.get("packet_sha256") != expected:
        raise ValueError("V23 packet identity or digest is invalid")
    for row in local["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen V23 source changed: {relative}")
        if _sha(root / "v23-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"archived V23 source changed: {relative}")
    v22_receipt = v22.verify_packet(root, pinned_v7_root)
    if v22_receipt.get("packet_sha256") != local.get("upstream_v22_packet_sha256"):
        raise ValueError("upstream V22 packet identity mismatch")
    if _sha(_root() / "docs/replacement-loss-v22-freeze.json") != local.get("upstream_v22_freeze_file_sha256"):
        raise ValueError("upstream V22 freeze manifest changed")
    v22_screen = _validate_v22_screen()
    if v22_screen.get("screening_receipt_sha256") is None:
        raise ValueError("upstream V22 screening receipt is incomplete")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": local["packet_sha256"],
        "upstream_v22_packet_sha256": v22_receipt["packet_sha256"],
        "upstream_v8_packet_sha256": v22_receipt["upstream_v8_packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def _v22_screen_ids() -> set[str]:
    screen = _validate_v22_screen()
    return {
        str(row.get("model_id", "")).strip().lower()
        for row in screen["target_screening"].get("compatible_family_configs", [])
        if row.get("model_id")
    }


def _v22_screen_ledger_row() -> dict[str, Any]:
    screen_ids = _v22_screen_ids()
    return {
        "path": "theseus-lab-sandbox-v1-replacement-loss-v22/v22-local-target-screening.json",
        "sha256": V22_SCREEN_SHA256,
        "identity_count": len(screen_ids),
    }


def _v22_prior_ids(current_root: Path | None = None) -> tuple[set[str], list[dict[str, Any]]]:
    if current_root is None:
        ids, rows = v22._prior_checkpoint_ledger()
    else:
        original_roots = v22._prior_experiment_roots
        try:
            v22._prior_experiment_roots = lambda: [
                path for path in original_roots() if path.resolve() != current_root.resolve()
            ]
            ids, rows = v22._prior_checkpoint_ledger()
        finally:
            v22._prior_experiment_roots = original_roots
    ids.update(_v22_screen_ids())
    rows = list(rows)
    screen_row = _v22_screen_ledger_row()
    if screen_row not in rows:
        rows.append(screen_row)
    return ids, rows


@contextmanager
def _v22_external_exclusions(current_root: Path, excluded_ids: set[str] | None = None) -> Iterator[None]:
    old_roots = v22._prior_experiment_roots
    old_ledger = v22._prior_checkpoint_ledger
    try:
        screen_ids = _v22_screen_ids()
        screen_row = _v22_screen_ledger_row()

        def prior_roots() -> list[Path]:
            return [path for path in old_roots() if path.resolve() != current_root.resolve()]

        def prior_ledger() -> tuple[set[str], list[dict[str, Any]]]:
            ids, rows = old_ledger()
            ids.update(screen_ids)
            if screen_row not in rows:
                rows.append(screen_row)
            if excluded_ids is not None:
                ids.update(excluded_ids)
            return ids, rows

        v22._prior_experiment_roots = prior_roots
        v22._prior_checkpoint_ledger = prior_ledger
        yield
    finally:
        v22._prior_experiment_roots = old_roots
        v22._prior_checkpoint_ledger = old_ledger


def acquire_checkpoints(
    custody_root: str | Path,
    v7_custody_root: str | Path,
    *,
    allow_public_hub_downloads: bool,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root, v7_custody_root)
    if allow_public_hub_downloads is not True:
        raise PermissionError("run requires explicit --allow-public-hub-downloads opt-in")
    with _v22_external_exclusions(root):
        prior_ids, prior_rows = _v22_prior_ids(root)
        screen = hub_adapter.screen_public_hub(prior_ids)
    screen["v23_packet_sha256"] = packet["packet_sha256"]
    screen["prior_checkpoint_identity_count"] = len(prior_ids)
    screen["prior_checkpoint_ledger_sha256"] = _canonical_digest(prior_rows)
    screen["upstream_v22_screen_sha256"] = V22_SCREEN_SHA256
    screen_path = root / "v23-hub-screen.json"
    _write_json(screen_path, screen)
    if screen["status"] != "READY_TO_DOWNLOAD":
        terminal = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "assurance_slice": ASSURANCE_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": "NO_ELIGIBLE_FRESH_TARGET",
            "assurance_level": "INTERNAL_PASS",
            "assurance_scope": "frozen packet verification and public metadata availability screen only",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_RUN",
            "hub_screen_sha256": _sha(screen_path),
            "eligible_families": [row["family"] for row in screen.get("selected_checkpoints", [])],
            "model_weights_downloaded": False,
            "candidate_preparation_started": False,
            "challenge_entropy_drawn": False,
            "primary_run_status": "NOT_RUN",
            "replication_status": "NOT_RUN",
            "monetary_cost_usd": 0,
            "next_blocked_authority_gate": "availability of eligible public base checkpoints within the frozen Hub search bound",
        }
        terminal["terminal_report_sha256"] = _canonical_digest(terminal)
        _write_json(root / "replacement-loss-v23-terminal-report.json", terminal)
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "hub_screen_sha256": _sha(screen_path)}
    disk = shutil.disk_usage(root)
    if disk.free < int(screen["selected_download_bytes"]) + 1_000_000_000:
        raise OSError("external custody free space is below the frozen download reserve")
    selection = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "hub_screen_sha256": _sha(screen_path),
        "selected_after_v23_freeze": True,
        "model_weights_downloaded": False,
        "selected_checkpoints": screen["selected_checkpoints"],
        "selected_at_utc": screen["created_at_utc"],
    }
    _write_json(root / "v23-checkpoint-selection.json", selection)
    downloaded = hub_adapter.download_selected(root, screen, allow_public_hub_downloads=True)
    download_manifest = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "selection_sha256": _sha(root / "v23-checkpoint-selection.json"),
        "downloaded": downloaded,
        "total_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloaded),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "v23-download-manifest.json", download_manifest)
    specs = [
        {
            "family": row["family"],
            "model_id": row["model_id"],
            "revision": row["revision"],
            "local_path": row["local_path"],
        }
        for row in downloaded
    ]
    with _v22_external_exclusions(root):
        v22_selection = v22.write_checkpoint_manifest(root, specs, v7_custody_root)
    binding = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "hub_screen_sha256": _sha(screen_path),
        "selection_sha256": _sha(root / "v23-checkpoint-selection.json"),
        "download_manifest_sha256": _sha(root / "v23-download-manifest.json"),
        "v22_checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
        "v22_selection_receipt": v22_selection,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "v23-selection-binding.json", binding)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": "TARGETS_ACQUIRED_READY_FOR_PRIMARY",
        "selected_checkpoint_count": 3,
        "downloaded_bytes": download_manifest["total_downloaded_bytes"],
        "selection_binding_sha256": _sha(root / "v23-selection-binding.json"),
    }


def _aggregate_v23(root: Path, packet_sha: str, primary_receipt: dict[str, Any]) -> dict[str, Any]:
    upstream = _read_json(root / "replacement-loss-v22-report.json")
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "packet_sha256": packet_sha,
        "upstream_v22_packet_sha256": V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "selection_binding_sha256": _sha(root / "v23-selection-binding.json"),
        "download_manifest_sha256": _sha(root / "v23-download-manifest.json"),
        "outcome_status": upstream["outcome_status"],
        "assurance_level": upstream["assurance_level"],
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "PENDING_SECOND_CLEAN_RUNTIME",
        "case_count": upstream["case_count"],
        "cases": upstream["cases"],
        "runtime": upstream["runtime"],
        "budget_and_resource_measurements": upstream["budget_and_resource_measurements"],
        "data_and_artifact_custody": "pinned model files and all experiment data remain in external owner custody",
        "known_limitations": [
            "three fresh checkpoint identities within three predeclared families",
            "one MLP output projection per checkpoint",
            "synthetic prompts and same-owner local replication",
            "public Hub discovery is bounded to the frozen queries and result limit",
            "monetary cost is zero; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "primary_runner_receipt": primary_receipt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "replacement-loss-v23-report.json", report)
    return report


def run_primary(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root, v7_custody_root)
    binding = _read_json(root / "v23-selection-binding.json")
    if binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V23 selected models do not bind the frozen packet")
    prior_ids, _rows = _v22_prior_ids(root)
    with _v22_external_exclusions(root, prior_ids):
        result = v22.run_primary(root, v7_custody_root)
    report = _aggregate_v23(root, packet["packet_sha256"], result)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "assurance_level": report["assurance_level"],
        "case_count": report["case_count"],
        "report_sha256": _sha(root / "replacement-loss-v23-report.json"),
    }


def replicate_packet(
    primary_custody_root: str | Path,
    replica_custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    packet = verify_packet(primary, v7_custody_root)
    primary_report_path = primary / "replacement-loss-v23-report.json"
    if not primary_report_path.is_file():
        raise FileNotFoundError("V23 primary report is required before reproduction")
    report_before = _read_json(primary_report_path)
    binding = _read_json(primary / "v23-selection-binding.json")
    if report_before.get("packet_sha256") != packet["packet_sha256"] or binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V23 primary output or target selection does not bind this packet")
    if binding.get("v22_checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V23 checkpoint manifest changed after primary evaluation")
    prior_ids, _rows = _v22_prior_ids(primary)
    with _v22_external_exclusions(primary, prior_ids):
        result = v22.replicate_packet(primary, replica, v7_custody_root)
    upstream_replica = _read_json(replica / "replacement-loss-v22-replication-report.json")
    final = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report_before["outcome_status"],
        "assurance_level": result.get("assurance_level", "INCONCLUSIVE"),
        "reproducibility_status": result.get("assurance_level", "INCONCLUSIVE"),
        "reviewer_status": "NOT_SUBMITTED",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "candidate_reconstruction_match": bool(upstream_replica.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(upstream_replica.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(upstream_replica.get("dependencies_match")),
        "distinct_clean_runtime": bool(upstream_replica.get("distinct_clean_runtime")),
        "source_identity_match": bool(upstream_replica.get("source_identity_match")),
        "upstream_v22_replication_report_sha256": _sha(replica / "replacement-loss-v22-replication-report.json"),
        "primary_v23_report_sha256_before_reconciliation": _sha(primary_report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    replica_path = replica / "replacement-loss-v23-replication-report.json"
    _write_json(replica_path, final)
    report_before["assurance_level"] = final["assurance_level"]
    report_before["reproducibility_status"] = final["reproducibility_status"]
    report_before["replication_report_sha256"] = _sha(replica_path)
    _write_json(primary_report_path, report_before)
    final["replication_report_sha256"] = _sha(replica_path)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": final["assurance_level"],
        "outcome_status": final["outcome_status"],
        "replication_report_sha256": _sha(replica_path),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--custody-root", required=True)
    freeze.add_argument("--v7-custody-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--custody-root", required=True)
    verify.add_argument("--v7-custody-root", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--custody-root", required=True)
    acquire.add_argument("--v7-custody-root", required=True)
    acquire.add_argument("--allow-public-hub-downloads", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--custody-root", required=True)
    run.add_argument("--v7-custody-root", required=True)
    replicate = commands.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", required=True)
    replicate.add_argument("--replica-custody-root", required=True)
    replicate.add_argument("--v7-custody-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root, args.v7_custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root, args.v7_custody_root)
    elif args.command == "acquire":
        result = acquire_checkpoints(
            args.custody_root,
            args.v7_custody_root,
            allow_public_hub_downloads=args.allow_public_hub_downloads,
        )
    elif args.command == "run":
        result = run_primary(args.custody_root, args.v7_custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root, args.v7_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
