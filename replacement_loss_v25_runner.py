"""Frozen lifecycle for metadata-complete V25 checkpoint acquisition.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v25-metadata-complete-acquisition-v1, and
global-experiment-assurance-lifecycle-v1. All data and model artifacts stay
under external owner custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V25_ROOT_NAME = "theseus-lab-sandbox-v1-replacement-loss-v25"
V25_ROOT = ARTIFACTS_ROOT / V25_ROOT_NAME
V25_REPLICA_ROOT = ARTIFACTS_ROOT / f"{V25_ROOT_NAME}-replicator"
os.environ["HF_HOME"] = str(V25_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V25_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V25_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V25_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import replacement_loss_v22_runner as v22
import replacement_loss_v23_runner as v23
import replacement_loss_v25_hub_adapter as hub_adapter


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v25-metadata-complete-acquisition-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
SOURCE_REVISION = "202f459d6d89df9a038d78d58e0af825d287fcd8"
PROTOCOL_ID = hub_adapter.PROTOCOL_ID
V24_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v24"
V24_PACKET_SHA256 = "1d1b99baa7dbe41f62a989a3653dc18b353f9d7fee5f8cbd6ed810796b162197"
V24_FREEZE_SHA256 = "caa82b873077c5789133d14c2ddf668cb2eaacf1e566a792822dda9f8444350c"
V24_SCREEN_SHA256 = "8764c4487c229140ebbbb3b77b4d8a4f0d8bee40bac8355ed607b48b6c31d68c"
V24_PROGRESS_SHA256 = "12eae1140782a0fde258520e89759ab27493b5a8c9560214460c5b2dde3f6041"
V24_TERMINAL_FILE_SHA256 = "b0c69983e023b7452c27e4107dcfc48951bfc5821eb882ffb08442ad249701e8"
V7_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v7"
FREEZE_PATH = "docs/replacement-loss-v25-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v25-metadata-complete-acquisition.md",
    "replacement_loss_v25_hub_adapter.py",
    "replacement_loss_v25_runner.py",
    "requirements-v25.lock.txt",
)


def _root() -> Path:
    return Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _external_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("V25 data custody must be outside the repository")
    return root


def _custody_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V25_ROOT.resolve():
        raise ValueError(f"V25 primary custody is pinned to {V25_ROOT}")
    return root


def _replica_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V25_REPLICA_ROOT.resolve():
        raise ValueError(f"V25 clean-replica custody is pinned to {V25_REPLICA_ROOT}")
    return root


def _v24_screen_ids(screen: dict[str, Any]) -> set[str]:
    found = set()
    for family in screen.get("family_screens", {}).values():
        for query in family.get("queries", []):
            for row in query.get("result_projection", []):
                model_id = str(row.get("model_id", "")).strip().lower()
                if model_id:
                    found.add(model_id)
    return found


def _verify_v24_evidence() -> dict[str, Any]:
    local_freeze = _read_json(_root() / "docs/replacement-loss-v24-freeze.json")
    external_freeze_path = V24_ROOT / "v24-frozen-packet-manifest.json"
    external_freeze = _read_json(external_freeze_path)
    if local_freeze != external_freeze or _sha(external_freeze_path) != V24_FREEZE_SHA256:
        raise ValueError("V24 freeze identity or external copy does not match the pinned digest")
    packet_sha = _canonical_digest({key: value for key, value in local_freeze.items() if key != "packet_sha256"})
    if packet_sha != V24_PACKET_SHA256 or local_freeze.get("packet_sha256") != packet_sha:
        raise ValueError("V24 packet self-digest is invalid")
    for row in local_freeze["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V24 local frozen source changed: {relative}")
        if _sha(V24_ROOT / "v24-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V24 archived source changed: {relative}")
    screen_path = V24_ROOT / "v24-hub-screen.json"
    progress_path = V24_ROOT / "v24-hub-screen-progress.json"
    terminal_path = V24_ROOT / "replacement-loss-v24-terminal-report.json"
    if _sha(screen_path) != V24_SCREEN_SHA256 or _sha(progress_path) != V24_PROGRESS_SHA256:
        raise ValueError("V24 screen evidence changed")
    if _sha(terminal_path) != V24_TERMINAL_FILE_SHA256:
        raise ValueError("V24 terminal receipt changed")
    screen = _read_json(screen_path)
    if screen.get("v24_packet_sha256") != V24_PACKET_SHA256:
        raise ValueError("V24 screen does not bind its frozen packet")
    terminal = _read_json(terminal_path)
    terminal_body = {key: value for key, value in terminal.items() if key != "terminal_report_sha256"}
    if terminal.get("terminal_report_sha256") != _canonical_digest(terminal_body):
        raise ValueError("V24 terminal receipt self-digest is invalid")
    if terminal.get("outcome_status") != "NO_ELIGIBLE_FRESH_TARGET" or terminal.get("model_weights_downloaded") is not False:
        raise ValueError("V25 continuation requires the audited V24 no-download terminal")
    ids = _v24_screen_ids(screen)
    if len(ids) != 5946:
        raise ValueError("V24 audit identity count differs from the frozen audit")
    if screen.get("selected_checkpoints"):
        raise ValueError("V24 unexpectedly selected a checkpoint")
    projections = [
        row
        for family in screen.get("family_screens", {}).values()
        for query in family.get("queries", [])
        for row in query.get("result_projection", [])
    ]
    has_valid_revision = any(
        len(str(row.get("revision", ""))) == 40
        and all(char in "0123456789abcdef" for char in str(row.get("revision", "")).lower())
        for row in projections
    )
    if len(projections) != 6000 or has_valid_revision:
        raise ValueError("V24 query receipt does not match the audited missing-SHA result")
    if any(V24_ROOT.glob("hub-cache/**/snapshots/**/*.safetensors")):
        raise ValueError("V24 evidence unexpectedly contains downloaded model weight snapshots")
    return {
        "packet_sha256": V24_PACKET_SHA256,
        "freeze_file_sha256": V24_FREEZE_SHA256,
        "screen_sha256": V24_SCREEN_SHA256,
        "progress_sha256": V24_PROGRESS_SHA256,
        "terminal_file_sha256": V24_TERMINAL_FILE_SHA256,
        "screen_identity_count": len(ids),
        "screen_identities": ids,
    }


def _prior_ledger(current_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    ids, rows = v23._v22_prior_ids(current_root)
    v24 = _verify_v24_evidence()
    if not v24["screen_identities"].issubset(ids):
        raise ValueError("prior identity ledger does not include every V24-screened identity")
    return ids, rows


def freeze_packet(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    local_freeze = _root() / FREEZE_PATH
    manifest_path = root / "v25-frozen-packet-manifest.json"
    archive = root / "v25-frozen-packet"
    if local_freeze.exists() or manifest_path.exists() or archive.exists():
        raise FileExistsError("V25 is immutable after freeze; use a new protocol version to change it")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("V25 custody root must be new or empty before freeze")
    evidence = _verify_v24_evidence()
    upstream = v23.verify_packet(V24_ROOT, V7_ROOT)
    source_rows = []
    for relative in PACKET_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V25 packet input is missing: {relative}")
        source_rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    root.mkdir(parents=True, exist_ok=True)
    for name in (
        "v24-frozen-packet-manifest.json",
        "v23-frozen-packet-manifest.json",
        "v22-frozen-packet-manifest.json",
        "v8-frozen-packet-manifest.json",
        "v24-hub-screen.json",
        "v24-hub-screen-progress.json",
        "replacement-loss-v24-terminal-report.json",
    ):
        source = V24_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"V24 external evidence missing: {name}")
        shutil.copyfile(source, root / name)
    for name in ("v24-frozen-packet", "v23-frozen-packet", "v22-frozen-packet", "v8-frozen-packet"):
        shutil.copytree(V24_ROOT / name, root / name)
    shutil.copyfile(_root() / "docs/replacement-loss-v24-results.md", root / "v24-audit-report.md")
    for row in source_rows:
        source = _root() / str(row["path"])
        target = archive / str(row["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    prior_ids, prior_rows = _prior_ledger(root)
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_state": "FROZEN_BEFORE_V25_PUBLIC_HUB_QUERY_OR_TARGET_SELECTION",
        "packet_files": source_rows,
        "upstream_v24_packet_sha256": evidence["packet_sha256"],
        "upstream_v24_freeze_file_sha256": evidence["freeze_file_sha256"],
        "upstream_v24_screen_sha256": evidence["screen_sha256"],
        "upstream_v24_progress_sha256": evidence["progress_sha256"],
        "upstream_v24_terminal_file_sha256": evidence["terminal_file_sha256"],
        "upstream_v24_audit_report_sha256": _sha(_root() / "docs/replacement-loss-v24-results.md"),
        "upstream_v22_packet_sha256": upstream["upstream_v22_packet_sha256"],
        "upstream_v8_packet_sha256": upstream["upstream_v8_packet_sha256"],
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "metadata_projection": hub_adapter.RESULT_PROPERTIES,
        "metadata_completeness_guard": "nonempty query results with zero valid full SHAs stop as INCONCLUSIVE_SETUP_OR_EXECUTION",
        "pre_v25_excluded_identity_count": len(prior_ids),
        "pre_v25_identity_ledger_sha256": _canonical_digest(prior_rows),
        "dependency_manifest_sha256": _sha(_root() / "requirements-v25.lock.txt"),
        "dependency_inventory": {
            "distribution_count": 26,
            "sha256": "cf90cf944be759e444dc187328b75b5e6632cd4c46401069c35cb90e5e6b2609",
            "primary_and_clean_replica_match": True,
        },
        "runtime": {
            "python": "3.12.13",
            "hub_client": "huggingface_hub==0.36.2",
            "execution": "local macOS arm64 CPU only",
            "auth": "token=False",
            "hub_cache_root": str(V25_ROOT / "hub-cache"),
            "hub_home_root": str(V25_ROOT / "hf-home"),
            "hub_assets_cache_root": str(V25_ROOT / "assets-cache"),
            "hub_xet_cache_root": str(V25_ROOT / "xet-cache"),
            "hub_telemetry": False,
            "hosted_inference": False,
            "remote_workers": 0,
        },
        "custody_roots": {"primary": str(V25_ROOT), "clean_replica": str(V25_REPLICA_ROOT)},
        "budget": {
            "monetary_usd": 0,
            "search_limit_per_query": hub_adapter.SEARCH_LIMIT,
            "max_model_bytes_each": hub_adapter.MAX_MODEL_BYTES,
            "max_model_bytes_total": hub_adapter.MAX_TOTAL_BYTES,
            "max_case_wall_seconds": 7200,
            "max_total_wall_seconds": 64800,
            "max_peak_rss_bytes": 25769803776,
        },
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "independent science", "consciousness", "alignment", "production readiness"],
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    _write_json(manifest_path, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": payload["packet_sha256"],
        "upstream_v24_packet_sha256": evidence["packet_sha256"],
        "pre_v25_excluded_identity_count": len(prior_ids),
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
    }


def verify_packet(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    local = _read_json(_root() / FREEZE_PATH)
    external = _read_json(root / "v25-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("local and external V25 freeze manifests differ")
    expected = _canonical_digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("packet_sha256") != expected or local.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V25 packet identity or self-digest is invalid")
    for row in local["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V25 frozen source changed: {relative}")
        if _sha(root / "v25-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V25 archived source changed: {relative}")
    evidence = _verify_v24_evidence()
    for key, value in (
        ("upstream_v24_packet_sha256", evidence["packet_sha256"]),
        ("upstream_v24_freeze_file_sha256", evidence["freeze_file_sha256"]),
        ("upstream_v24_screen_sha256", evidence["screen_sha256"]),
        ("upstream_v24_progress_sha256", evidence["progress_sha256"]),
        ("upstream_v24_terminal_file_sha256", evidence["terminal_file_sha256"]),
    ):
        if local.get(key) != value:
            raise ValueError(f"V25 upstream evidence binding mismatch: {key}")
    for source_name in ("v24-hub-screen.json", "v24-hub-screen-progress.json", "replacement-loss-v24-terminal-report.json"):
        if _sha(root / source_name) != _sha(V24_ROOT / source_name):
            raise ValueError(f"V25 archived V24 evidence changed: {source_name}")
    audit_path = _root() / "docs/replacement-loss-v24-results.md"
    if _sha(audit_path) != local.get("upstream_v24_audit_report_sha256") or _sha(root / "v24-audit-report.md") != _sha(audit_path):
        raise ValueError("V25 archived V24 audit report changed")
    upstream = v23.verify_packet(root, V7_ROOT)
    if upstream.get("upstream_v22_packet_sha256") != local.get("upstream_v22_packet_sha256"):
        raise ValueError("upstream V22 packet identity mismatch")
    prior_ids, prior_rows = _prior_ledger(root)
    if len(prior_ids) != local.get("pre_v25_excluded_identity_count"):
        raise ValueError("pre-V25 exclusion identity count changed")
    if _canonical_digest(prior_rows) != local.get("pre_v25_identity_ledger_sha256"):
        raise ValueError("pre-V25 exclusion ledger changed")
    return {
        "protocol_id": PROTOCOL_ID,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": local["packet_sha256"],
        "pre_v25_excluded_identity_count": len(prior_ids),
        "upstream_v24_packet_sha256": evidence["packet_sha256"],
        "upstream_v22_packet_sha256": upstream["upstream_v22_packet_sha256"],
        "upstream_v8_packet_sha256": upstream["upstream_v8_packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def _write_stop(
    root: Path,
    packet_sha: str,
    *,
    outcome_status: str,
    phase: str,
    error: Exception | None = None,
    **details: Any,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet_sha,
        "outcome_status": outcome_status,
        "assurance_level": "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "NOT_RUN",
        "phase": phase,
        "behavior_and_mechanism_diverged": "NOT_ASSESSED_NO_COMPLETE_EXPERIMENT",
        "data_and_artifact_custody": f"V25 artifacts remain under external custody {root}",
        "known_limitations": [
            "local same-owner work only",
            "three named model families; no unseen architecture test",
            "one MLP output projection per checkpoint",
            "frozen V8 method defines no formal behavior-versus-mechanism divergence statistic",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "next_blocked_authority_gate": "resolve the recorded stop in a new frozen continuation before making a scientific claim",
        "monetary_cost_usd": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    if error is not None:
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
    receipt["terminal_report_sha256"] = _canonical_digest(receipt)
    _write_json(root / "replacement-loss-v25-terminal-report.json", receipt)
    return receipt


def acquire_checkpoints(
    custody_root: str | Path,
    *,
    allow_public_hub_downloads: bool,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if allow_public_hub_downloads is not True:
        raise PermissionError("run requires explicit --allow-public-hub-downloads opt-in")
    progress_path = root / "v25-hub-screen-progress.json"
    one_shot = (
        progress_path,
        root / "v25-hub-screen.json",
        root / "v25-checkpoint-selection.json",
        root / "v25-download-manifest.json",
        root / "v25-selection-binding.json",
        root / "replacement-loss-v25-terminal-report.json",
        root / "replacement-loss-v25-report.json",
        root / "checkpoints.json",
    )
    existing = [path.name for path in one_shot if path.exists()]
    if existing:
        raise FileExistsError(f"V25 acquisition is one-shot; prior receipts exist: {existing}")
    prior_ids, prior_rows = _prior_ledger(root)

    def save_progress(receipt: dict[str, Any]) -> None:
        old = _read_json(progress_path) if progress_path.is_file() else {"completed_queries": []}
        rows = old.get("completed_queries", [])
        rows.append(receipt)
        _write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "completed_queries": rows,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    try:
        screen = hub_adapter.screen_public_hub(prior_ids, on_progress=save_progress)
    except Exception as exc:
        stopped = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase="PUBLIC_METADATA_SCREEN",
            error=exc,
            progress_receipt_sha256=_sha(progress_path) if progress_path.is_file() else None,
            prior_identity_ledger_sha256=_canonical_digest(prior_rows),
            model_weights_downloaded=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": stopped["outcome_status"], "terminal_report_sha256": stopped["terminal_report_sha256"]}
    screen["v25_packet_sha256"] = packet["packet_sha256"]
    screen["prior_checkpoint_identity_count"] = len(prior_ids)
    screen["prior_checkpoint_ledger_sha256"] = _canonical_digest(prior_rows)
    screen_path = root / "v25-hub-screen.json"
    _write_json(screen_path, screen)
    if screen["status"] != "READY_TO_DOWNLOAD":
        terminal = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "assurance_slice": ASSURANCE_SLICE,
            "source_revision": SOURCE_REVISION,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": "NO_ELIGIBLE_FRESH_TARGET",
            "assurance_level": "INTERNAL_PASS",
            "assurance_scope": "complete metadata-valid bounded public screen and frozen packet verification only",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_RUN",
            "behavior_and_mechanism_diverged": "NOT_ASSESSED_NO_MODEL_RUN",
            "data_and_artifact_custody": f"V25 packet and public metadata receipts remain under {root}; no model weights or prompts acquired",
            "known_limitations": ["frozen Hub result bounds only", "three named model families", "no scientific evaluation or formal divergence statistic was run"],
            "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
            "hub_screen_sha256": _sha(screen_path),
            "progress_receipt_sha256": _sha(progress_path) if progress_path.is_file() else None,
            "prior_identity_ledger_sha256": _canonical_digest(prior_rows),
            "eligible_checkpoint_count_by_family": {family: row["selected_identity_count"] for family, row in screen["family_screens"].items()},
            "model_weights_downloaded": False,
            "candidate_preparation_started": False,
            "challenge_entropy_drawn": False,
            "primary_run_status": "NOT_RUN",
            "replication_status": "NOT_RUN",
            "monetary_cost_usd": 0,
            "next_blocked_authority_gate": "availability of eligible fresh public base checkpoints within the frozen V25 query bound",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        terminal["terminal_report_sha256"] = _canonical_digest(terminal)
        _write_json(root / "replacement-loss-v25-terminal-report.json", terminal)
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "hub_screen_sha256": _sha(screen_path)}

    phase = "RESOURCE_CHECK"
    try:
        if shutil.disk_usage(root).free < int(screen["selected_download_bytes"]) + 1_000_000_000:
            raise OSError("external V25 custody lacks the frozen download reserve")
        selection = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selected_after_v25_freeze": True,
            "selected_checkpoints": screen["selected_checkpoints"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v25-checkpoint-selection.json", selection)
        phase = "MODEL_DOWNLOAD"
        downloaded = hub_adapter.download_selected(root, screen, allow_public_hub_downloads=True)
        downloads = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "selection_sha256": _sha(root / "v25-checkpoint-selection.json"),
            "downloaded": downloaded,
            "total_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloaded),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v25-download-manifest.json", downloads)
        specs = [{"family": row["family"], "model_id": row["model_id"], "revision": row["revision"], "local_path": row["local_path"]} for row in downloaded]
        phase = "CHECKPOINT_MANIFEST_BINDING"
        with v23._v22_external_exclusions(root, prior_ids):
            selection_result = v22.write_checkpoint_manifest(root, specs, V7_ROOT)
        binding = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selection_sha256": _sha(root / "v25-checkpoint-selection.json"),
            "download_manifest_sha256": _sha(root / "v25-download-manifest.json"),
            "v22_checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
            "v22_selection_receipt": selection_result,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v25-selection-binding.json", binding)
    except Exception as exc:
        stopped = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase=phase,
            error=exc,
            hub_screen_sha256=_sha(screen_path) if screen_path.is_file() else None,
            selection_sha256=_sha(root / "v25-checkpoint-selection.json") if (root / "v25-checkpoint-selection.json").is_file() else None,
            download_manifest_sha256=_sha(root / "v25-download-manifest.json") if (root / "v25-download-manifest.json").is_file() else None,
            checkpoint_manifest_sha256=_sha(root / "checkpoints.json") if (root / "checkpoints.json").is_file() else None,
            model_download_attempted=phase == "MODEL_DOWNLOAD" or (root / "v25-download-manifest.json").exists(),
            candidate_preparation_started=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": stopped["outcome_status"], "terminal_report_sha256": stopped["terminal_report_sha256"]}
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": "TARGETS_ACQUIRED_READY_FOR_PRIMARY",
        "checkpoint_count": 3,
        "downloaded_bytes": downloads["total_downloaded_bytes"],
        "selection_binding_sha256": _sha(root / "v25-selection-binding.json"),
    }


def _aggregate(root: Path, packet_sha: str, runner_receipt: dict[str, Any]) -> dict[str, Any]:
    upstream = _read_json(root / "replacement-loss-v22-report.json")
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet_sha,
        "upstream_v24_packet_sha256": V24_PACKET_SHA256,
        "upstream_v22_packet_sha256": v22.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "selection_binding_sha256": _sha(root / "v25-selection-binding.json"),
        "download_manifest_sha256": _sha(root / "v25-download-manifest.json"),
        "outcome_status": upstream["outcome_status"],
        "assurance_level": upstream["assurance_level"],
        "behavior_and_mechanism_diverged": "NOT_SEPARATELY_DEFINED_BY_FROZEN_V8_METHOD; behavior, causal effects, and activation rescue are separate metrics",
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "PENDING_SECOND_CLEAN_RUNTIME",
        "case_count": upstream["case_count"],
        "cases": upstream["cases"],
        "runtime": upstream["runtime"],
        "budget_and_resource_measurements": upstream["budget_and_resource_measurements"],
        "data_and_artifact_custody": f"model files and experiment data remain under external owner custody {root}",
        "known_limitations": [
            "three fresh checkpoint identities within three predeclared families",
            "one MLP output projection per checkpoint",
            "synthetic prompts and same-owner local reproduction",
            "Hub discovery is limited to the frozen queries and result limits",
            "monetary cost is zero; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "primary_runner_receipt": runner_receipt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "replacement-loss-v25-report.json", report)
    return report


def run_primary(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    binding = _read_json(root / "v25-selection-binding.json")
    if binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V25 selected checkpoints do not bind the frozen packet")
    prior_outputs = (root / "replacement-loss-v25-report.json", root / "replacement-loss-v22-report.json", root / "candidate-preparation.json", root / "candidate-freeze.json", root / "challenge-manifest.json")
    if any(path.exists() for path in prior_outputs):
        raise FileExistsError("V25 primary evaluation is one-shot; prior execution output exists")
    prior_ids, _rows = _prior_ledger(root)
    try:
        with v23._v22_external_exclusions(root, prior_ids):
            result = v22.run_primary(root, V7_ROOT)
        report = _aggregate(root, packet["packet_sha256"], result)
    except Exception as exc:
        stopped = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_EXECUTION",
            phase="PRIMARY_EVALUATION",
            error=exc,
            behavior_and_mechanism_diverged="NOT_ASSESSED_INCOMPLETE_EXPERIMENT",
            model_weights_downloaded=True,
            candidate_preparation_started=(root / "candidate-preparation.json").is_file(),
            candidate_freeze_sha256=_sha(root / "candidate-freeze.json") if (root / "candidate-freeze.json").is_file() else None,
            challenge_manifest_sha256=_sha(root / "challenge-manifest.json") if (root / "challenge-manifest.json").is_file() else None,
            primary_run_status="INCONCLUSIVE",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": stopped["outcome_status"], "terminal_report_sha256": stopped["terminal_report_sha256"]}
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": report["outcome_status"], "assurance_level": report["assurance_level"], "case_count": report["case_count"], "report_sha256": _sha(root / "replacement-loss-v25-report.json")}


def replicate_packet(primary_custody_root: str | Path, replica_custody_root: str | Path) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _replica_root(replica_custody_root)
    packet = verify_packet(primary)
    report_path = primary / "replacement-loss-v25-report.json"
    report_before = _read_json(report_path)
    binding = _read_json(primary / "v25-selection-binding.json")
    if report_before.get("packet_sha256") != packet["packet_sha256"] or binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V25 primary report or checkpoint selection does not bind the frozen packet")
    if binding.get("v22_checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V25 checkpoint manifest changed after primary evaluation")
    if report_before.get("outcome_status") not in {"PASS", "FAIL"}:
        raise ValueError("V25 reproduction requires a completed primary PASS or FAIL outcome")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("V25 clean reproduction is one-shot; replica custody already contains files")
    replica.mkdir(parents=True, exist_ok=True)
    prior_ids, _rows = _prior_ledger(primary)
    try:
        with v23._v22_external_exclusions(primary, prior_ids):
            result = v22.replicate_packet(primary, replica, V7_ROOT)
        upstream = _read_json(replica / "replacement-loss-v22-replication-report.json")
    except Exception as exc:
        stopped = _write_stop(
            replica,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_REPRODUCTION",
            phase="CLEAN_REPLICA_EVALUATION",
            error=exc,
            primary_v25_report_sha256=_sha(report_path),
            partial_v22_replication_report_sha256=_sha(replica / "replacement-loss-v22-replication-report.json") if (replica / "replacement-loss-v22-replication-report.json").is_file() else None,
            primary_run_status=report_before.get("outcome_status"),
            replication_status="INCONCLUSIVE",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": stopped["outcome_status"], "terminal_report_sha256": stopped["terminal_report_sha256"]}
    final = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report_before["outcome_status"],
        "assurance_level": result.get("assurance_level", "INCONCLUSIVE"),
        "reproducibility_status": result.get("assurance_level", "INCONCLUSIVE"),
        "reviewer_status": "NOT_SUBMITTED",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "behavior_and_mechanism_diverged": report_before.get("behavior_and_mechanism_diverged", "NOT_ASSESSED"),
        "candidate_reconstruction_match": bool(upstream.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(upstream.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(upstream.get("dependencies_match")),
        "distinct_clean_runtime": bool(upstream.get("distinct_clean_runtime")),
        "source_identity_match": bool(upstream.get("source_identity_match")),
        "data_and_artifact_custody": f"primary and clean-replica evidence remains under {primary} and {replica}",
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "independent science", "consciousness", "alignment", "production readiness"],
        "upstream_v22_replication_report_sha256": _sha(replica / "replacement-loss-v22-replication-report.json"),
        "primary_v25_report_sha256_before_reconciliation": _sha(report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    replication_path = replica / "replacement-loss-v25-replication-report.json"
    _write_json(replication_path, final)
    report_before["assurance_level"] = final["assurance_level"]
    report_before["reproducibility_status"] = final["reproducibility_status"]
    report_before["replication_report_sha256"] = _sha(replication_path)
    _write_json(report_path, report_before)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "assurance_level": final["assurance_level"], "outcome_status": final["outcome_status"], "replication_report_sha256": _sha(replication_path)}


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--custody-root", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--custody-root", required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--custody-root", required=True)
    acquire.add_argument("--allow-public-hub-downloads", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--custody-root", required=True)
    replicate = commands.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", required=True)
    replicate.add_argument("--replica-custody-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root)
    elif args.command == "acquire":
        result = acquire_checkpoints(args.custody_root, allow_public_hub_downloads=args.allow_public_hub_downloads)
    elif args.command == "run":
        result = run_primary(args.custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
