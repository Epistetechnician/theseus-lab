"""Frozen lifecycle for V24 public checkpoint acquisition and replication.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v24-public-checkpoint-acquisition-v1, and
global-experiment-assurance-lifecycle-v1. Public model assets and experiment
data stay in external owner custody.
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

# V24 must keep Hub metadata and model caches in external experiment custody.
# Set these before importing any Hub or Transformers module because their
# cache paths are resolved from the environment during import.
ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V24_ROOT_NAME = "theseus-lab-sandbox-v1-replacement-loss-v24"
V24_ROOT = ARTIFACTS_ROOT / V24_ROOT_NAME
V24_REPLICA_ROOT = ARTIFACTS_ROOT / f"{V24_ROOT_NAME}-replicator"
os.environ["HF_HOME"] = str(V24_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V24_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V24_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V24_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import replacement_loss_v22_runner as v22
import replacement_loss_v23_runner as v23
import replacement_loss_v24_hub_adapter as hub_adapter


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v24-public-checkpoint-acquisition-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
SOURCE_REVISION = "202f459d6d89df9a038d78d58e0af825d287fcd8"
PROTOCOL_ID = hub_adapter.PROTOCOL_ID
V23_PACKET_SHA256 = "7600d653f906db5411a6ab3121e078f8134e311e8fe0ecb37ef05552bdbe890e"
V23_STOP_SHA256 = "0bff6a0367ca5b39b1decc3bdcb229a38b7bd551891d80901f731c673b4123d9"
V23_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v23"
V24_FREEZE_PATH = "docs/replacement-loss-v24-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v24-public-checkpoint-acquisition.md",
    "replacement_loss_v24_hub_adapter.py",
    "replacement_loss_v24_runner.py",
    "requirements-v24.lock.txt",
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
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        "data_and_artifact_custody": f"all V24 model and experiment artifacts remain under external custody {root}",
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
    _write_json(root / "replacement-loss-v24-terminal-report.json", receipt)
    return receipt


def _external_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("V24 custody must be outside the repository")
    return root


def _custody_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V24_ROOT.resolve():
        raise ValueError(f"V24 primary custody is pinned to {V24_ROOT}")
    return root


def _replica_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V24_REPLICA_ROOT.resolve():
        raise ValueError(f"V24 clean-replica custody is pinned to {V24_REPLICA_ROOT}")
    return root


def _validate_v23_stop() -> dict[str, Any]:
    path = V23_ROOT / "v23-setup-stop.json"
    receipt = _read_json(path)
    if _sha(path) != V23_STOP_SHA256:
        raise ValueError("V23 setup-stop receipt changed")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _canonical_digest(body):
        raise ValueError("V23 setup-stop receipt canonical digest is invalid")
    if receipt.get("outcome_status") != "INCONCLUSIVE_SETUP_OR_EXECUTION" or receipt.get("hub_records_received") != 0:
        raise ValueError("V24 must continue only the V23 pre-identity setup stop")
    return receipt


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V24 packet input is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    return rows


def freeze_packet(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    pinned_v7_root = _external_root(v7_custody_root)
    local_freeze = _root() / V24_FREEZE_PATH
    external_manifest = root / "v24-frozen-packet-manifest.json"
    archive = root / "v24-frozen-packet"
    if local_freeze.exists() or external_manifest.exists() or archive.exists():
        raise FileExistsError("V24 packet is immutable; use a new version to change it")
    v23_receipt = v23.verify_packet(V23_ROOT, pinned_v7_root)
    if v23_receipt.get("packet_sha256") != V23_PACKET_SHA256:
        raise ValueError("upstream V23 packet identity changed")
    v23_stop = _validate_v23_stop()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("V24 custody root must be new or empty before freeze")
    root.mkdir(parents=True, exist_ok=True)
    for name in ("v23-frozen-packet-manifest.json", "v22-frozen-packet-manifest.json", "v8-frozen-packet-manifest.json"):
        shutil.copyfile(V23_ROOT / name, root / name)
    for name in ("v23-frozen-packet", "v22-frozen-packet", "v8-frozen-packet"):
        shutil.copytree(V23_ROOT / name, root / name)
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
        "source_revision": SOURCE_REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_state": "FROZEN_BEFORE_V24_PUBLIC_HUB_QUERY_OR_TARGET_SELECTION",
        "packet_files": source_rows,
        "upstream_v23_packet_sha256": V23_PACKET_SHA256,
        "upstream_v23_freeze_file_sha256": _sha(_root() / "docs/replacement-loss-v23-freeze.json"),
        "upstream_v23_setup_stop_sha256": V23_STOP_SHA256,
        "upstream_v22_packet_sha256": v23_receipt["upstream_v22_packet_sha256"],
        "upstream_v8_packet_sha256": v23_receipt["upstream_v8_packet_sha256"],
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "parameter_cap_method": "local exact safetensors metadata from frozen V23 checker; no unsupported list_models parameter",
        "dependency_manifest_sha256": _sha(_root() / "requirements-v24.lock.txt"),
        "dependency_inventory": {
            "distribution_count": 26,
            "sha256": "cf90cf944be759e444dc187328b75b5e6632cd4c46401069c35cb90e5e6b2609",
            "primary_and_clean_replica_match": True,
        },
        "runtime": {
            "python": "3.12.13",
            "hub_client": "huggingface_hub==0.36.2",
            "auth": "token=False",
            "execution": "local macOS arm64 CPU only",
            "hub_cache_root": str(V24_ROOT / "hub-cache"),
            "hub_home_root": str(V24_ROOT / "hf-home"),
            "hub_assets_cache_root": str(V24_ROOT / "assets-cache"),
            "hub_xet_cache_root": str(V24_ROOT / "xet-cache"),
            "hub_telemetry": False,
            "hosted_inference": False,
            "remote_workers": 0,
        },
        "custody_roots": {
            "primary": str(V24_ROOT),
            "clean_replica": str(V24_REPLICA_ROOT),
        },
        "budget": {
            "monetary_usd": 0,
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
    _write_json(external_manifest, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "upstream_v23_packet_sha256": V23_PACKET_SHA256,
        "upstream_v22_packet_sha256": payload["upstream_v22_packet_sha256"],
        "upstream_v8_packet_sha256": payload["upstream_v8_packet_sha256"],
        "packet_file_count": len(source_rows),
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
        "upstream_v23_status": v23_stop["outcome_status"],
    }


def verify_packet(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    pinned_v7_root = _external_root(v7_custody_root)
    local = _read_json(_root() / V24_FREEZE_PATH)
    external = _read_json(root / "v24-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("local and external V24 freeze manifests differ")
    expected = _canonical_digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("protocol_id") != PROTOCOL_ID or local.get("packet_sha256") != expected:
        raise ValueError("V24 packet identity or digest is invalid")
    for row in local["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen V24 source changed: {relative}")
        if _sha(root / "v24-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"archived V24 source changed: {relative}")
    v23_receipt = v23.verify_packet(root, pinned_v7_root)
    if v23_receipt.get("packet_sha256") != local.get("upstream_v23_packet_sha256"):
        raise ValueError("upstream V23 packet identity mismatch")
    if _sha(_root() / "docs/replacement-loss-v23-freeze.json") != local.get("upstream_v23_freeze_file_sha256"):
        raise ValueError("upstream V23 freeze manifest changed")
    stop = _validate_v23_stop()
    if stop.get("receipt_sha256") is None:
        raise ValueError("upstream V23 stop receipt lacks its canonical digest")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": local["packet_sha256"],
        "upstream_v23_packet_sha256": v23_receipt["packet_sha256"],
        "upstream_v22_packet_sha256": v23_receipt["upstream_v22_packet_sha256"],
        "upstream_v8_packet_sha256": v23_receipt["upstream_v8_packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def _v22_prior_ids(current_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    return v23._v22_prior_ids(current_root)


def _context(current_root: Path, extra_ids: set[str]):
    return v23._v22_external_exclusions(current_root, extra_ids)


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
    progress_path = root / "v24-hub-screen-progress.json"
    one_shot_outputs = (
        progress_path,
        root / "v24-hub-screen.json",
        root / "v24-checkpoint-selection.json",
        root / "v24-download-manifest.json",
        root / "v24-selection-binding.json",
        root / "replacement-loss-v24-terminal-report.json",
        root / "replacement-loss-v24-report.json",
        root / "checkpoints.json",
    )
    existing = [path.name for path in one_shot_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"V24 acquisition is one-shot; prior receipts exist: {existing}")
    prior_ids, prior_rows = _v22_prior_ids(root)

    def save_progress(receipt: dict[str, Any]) -> None:
        prior = _read_json(progress_path) if progress_path.is_file() else {"completed_queries": []}
        completed = prior.get("completed_queries", [])
        completed.append(receipt)
        _write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "completed_queries": completed,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    try:
        with _context(root, prior_ids):
            screen = hub_adapter.screen_public_hub(prior_ids, on_progress=save_progress)
    except Exception as exc:
        stopped = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase="PUBLIC_METADATA_SCREEN",
            error=exc,
            progress_receipt_sha256=_sha(progress_path) if progress_path.is_file() else None,
            prior_checkpoint_ledger_sha256=_canonical_digest(prior_rows),
            checkpoint_identities_selected=False,
            model_weights_downloaded=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": stopped["outcome_status"],
            "progress_receipt_sha256": stopped["progress_receipt_sha256"],
        }

    screen["v24_packet_sha256"] = packet["packet_sha256"]
    screen["prior_checkpoint_identity_count"] = len(prior_ids)
    screen["prior_checkpoint_ledger_sha256"] = _canonical_digest(prior_rows)
    screen_path = root / "v24-hub-screen.json"
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
            "assurance_scope": "frozen packet verification and complete bounded public metadata screen only",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_RUN",
            "behavior_and_mechanism_diverged": "NOT_ASSESSED_NO_MODEL_RUN",
            "data_and_artifact_custody": f"frozen packet and public metadata screen receipts remain under external custody {root}; no model weights or prompts acquired",
            "known_limitations": [
                "screen covers only frozen public Hub queries and result limits",
                "three named model families; no unseen architecture test",
                "no scientific evaluation or formal divergence statistic was run",
            ],
            "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
            "hub_screen_sha256": _sha(screen_path),
            "prior_checkpoint_ledger_sha256": _canonical_digest(prior_rows),
            "eligible_checkpoint_count_by_family": {
                family: row["selected_identity_count"]
                for family, row in screen["family_screens"].items()
            },
            "model_weights_downloaded": False,
            "candidate_preparation_started": False,
            "challenge_entropy_drawn": False,
            "primary_run_status": "NOT_RUN",
            "replication_status": "NOT_RUN",
            "monetary_cost_usd": 0,
            "next_blocked_authority_gate": "availability of eligible fresh public base checkpoints within the frozen query bound",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        terminal["terminal_report_sha256"] = _canonical_digest(terminal)
        _write_json(root / "replacement-loss-v24-terminal-report.json", terminal)
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": terminal["outcome_status"],
            "hub_screen_sha256": _sha(screen_path),
        }
    phase = "RESOURCE_CHECK"
    try:
        free = shutil.disk_usage(root).free
        if free < int(screen["selected_download_bytes"]) + 1_000_000_000:
            raise OSError("external V24 custody lacks the frozen download reserve")
        phase = "IDENTITY_SELECTION"
        selection = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selected_after_v24_freeze": True,
            "selected_checkpoints": screen["selected_checkpoints"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v24-checkpoint-selection.json", selection)
        phase = "MODEL_DOWNLOAD"
        downloaded = hub_adapter.download_selected(root, screen, allow_public_hub_downloads=True)
        downloads = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "selection_sha256": _sha(root / "v24-checkpoint-selection.json"),
            "downloaded": downloaded,
            "total_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloaded),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v24-download-manifest.json", downloads)
        specs = [
            {"family": row["family"], "model_id": row["model_id"], "revision": row["revision"], "local_path": row["local_path"]}
            for row in downloaded
        ]
        phase = "CHECKPOINT_MANIFEST_BINDING"
        with _context(root, prior_ids):
            selection_result = v22.write_checkpoint_manifest(root, specs, v7_custody_root)
        binding = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selection_sha256": _sha(root / "v24-checkpoint-selection.json"),
            "download_manifest_sha256": _sha(root / "v24-download-manifest.json"),
            "v22_checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
            "v22_selection_receipt": selection_result,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v24-selection-binding.json", binding)
    except Exception as exc:
        stopped = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase=phase,
            error=exc,
            hub_screen_sha256=_sha(screen_path) if screen_path.is_file() else None,
            progress_receipt_sha256=_sha(progress_path) if progress_path.is_file() else None,
            selection_sha256=_sha(root / "v24-checkpoint-selection.json") if (root / "v24-checkpoint-selection.json").is_file() else None,
            download_manifest_sha256=_sha(root / "v24-download-manifest.json") if (root / "v24-download-manifest.json").is_file() else None,
            checkpoint_manifest_sha256=_sha(root / "checkpoints.json") if (root / "checkpoints.json").is_file() else None,
            model_download_attempted=phase == "MODEL_DOWNLOAD" or (root / "v24-download-manifest.json").exists(),
            candidate_preparation_started=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
            monetary_cost_usd=0,
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": stopped["outcome_status"],
            "terminal_report_sha256": stopped["terminal_report_sha256"],
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": "TARGETS_ACQUIRED_READY_FOR_PRIMARY",
        "checkpoint_count": 3,
        "downloaded_bytes": downloads["total_downloaded_bytes"],
        "selection_binding_sha256": _sha(root / "v24-selection-binding.json"),
    }


def _aggregate(root: Path, packet_sha: str, primary_receipt: dict[str, Any]) -> dict[str, Any]:
    upstream = _read_json(root / "replacement-loss-v22-report.json")
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet_sha,
        "upstream_v23_packet_sha256": V23_PACKET_SHA256,
        "upstream_v22_packet_sha256": v23.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "selection_binding_sha256": _sha(root / "v24-selection-binding.json"),
        "download_manifest_sha256": _sha(root / "v24-download-manifest.json"),
        "outcome_status": upstream["outcome_status"],
        "assurance_level": upstream["assurance_level"],
        "behavior_and_mechanism_diverged": (
            "NOT_SEPARATELY_DEFINED_BY_FROZEN_V8_METHOD; behavior, causal effects, and activation rescue are reported as distinct metrics"
        ),
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
            "synthetic prompts and same-owner local reproduction",
            "Hub discovery is limited to the frozen queries and result limits",
            "monetary cost is zero; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "primary_runner_receipt": primary_receipt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "replacement-loss-v24-report.json", report)
    return report


def run_primary(custody_root: str | Path, v7_custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root, v7_custody_root)
    binding = _read_json(root / "v24-selection-binding.json")
    if binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V24 selected models do not bind the frozen packet")
    prior_outputs = (
        root / "replacement-loss-v24-report.json",
        root / "replacement-loss-v22-report.json",
        root / "candidate-preparation.json",
        root / "candidate-freeze.json",
        root / "challenge-manifest.json",
    )
    if any(path.exists() for path in prior_outputs):
        raise FileExistsError("V24 primary run is one-shot; prior execution output exists")
    prior_ids, _rows = _v22_prior_ids(root)
    try:
        with _context(root, prior_ids):
            result = v22.run_primary(root, v7_custody_root)
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
            partial_v22_report_sha256=_sha(root / "replacement-loss-v22-report.json") if (root / "replacement-loss-v22-report.json").is_file() else None,
            primary_run_status="INCONCLUSIVE",
            replication_status="NOT_RUN",
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": stopped["outcome_status"],
            "terminal_report_sha256": stopped["terminal_report_sha256"],
        }
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "assurance_level": report["assurance_level"],
        "case_count": report["case_count"],
        "report_sha256": _sha(root / "replacement-loss-v24-report.json"),
    }


def replicate_packet(
    primary_custody_root: str | Path,
    replica_custody_root: str | Path,
    v7_custody_root: str | Path,
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _replica_root(replica_custody_root)
    packet = verify_packet(primary, v7_custody_root)
    report_path = primary / "replacement-loss-v24-report.json"
    if not report_path.is_file():
        raise FileNotFoundError("V24 primary report is required before reproduction")
    report_before = _read_json(report_path)
    binding = _read_json(primary / "v24-selection-binding.json")
    if report_before.get("packet_sha256") != packet["packet_sha256"] or binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V24 primary output or selection does not bind the frozen packet")
    if binding.get("v22_checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V24 checkpoint manifest changed after primary evaluation")
    if report_before.get("outcome_status") not in {"PASS", "FAIL"}:
        raise ValueError("V24 reproduction requires a completed primary PASS or FAIL outcome")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("V24 reproduction is one-shot; clean-replica custody already contains files")
    replica.mkdir(parents=True, exist_ok=True)
    prior_ids, _rows = _v22_prior_ids(primary)
    try:
        with _context(primary, prior_ids):
            result = v22.replicate_packet(primary, replica, v7_custody_root)
        upstream = _read_json(replica / "replacement-loss-v22-replication-report.json")
    except Exception as exc:
        stopped = _write_stop(
            replica,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_REPRODUCTION",
            phase="CLEAN_REPLICA_EVALUATION",
            error=exc,
            behavior_and_mechanism_diverged=report_before.get("behavior_and_mechanism_diverged", "NOT_ASSESSED"),
            primary_v24_report_sha256=_sha(report_path),
            partial_v22_replication_report_sha256=(
                _sha(replica / "replacement-loss-v22-replication-report.json")
                if (replica / "replacement-loss-v22-replication-report.json").is_file()
                else None
            ),
            primary_run_status=report_before.get("outcome_status"),
            replication_status="INCONCLUSIVE",
        )
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "outcome_status": stopped["outcome_status"],
            "terminal_report_sha256": stopped["terminal_report_sha256"],
        }
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
        "data_and_artifact_custody": f"primary and clean-replica artifacts remain under external custody {primary} and {replica}",
        "known_limitations": [
            "three fresh checkpoint identities within three predeclared families",
            "one MLP output projection per checkpoint",
            "same-owner local reproduction, not independent science",
            "frozen V8 method defines no formal behavior-versus-mechanism divergence statistic",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "candidate_reconstruction_match": bool(upstream.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(upstream.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(upstream.get("dependencies_match")),
        "distinct_clean_runtime": bool(upstream.get("distinct_clean_runtime")),
        "source_identity_match": bool(upstream.get("source_identity_match")),
        "upstream_v22_replication_report_sha256": _sha(replica / "replacement-loss-v22-replication-report.json"),
        "primary_v24_report_sha256_before_reconciliation": _sha(report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    replica_path = replica / "replacement-loss-v24-replication-report.json"
    _write_json(replica_path, final)
    report_before["assurance_level"] = final["assurance_level"]
    report_before["reproducibility_status"] = final["reproducibility_status"]
    report_before["replication_report_sha256"] = _sha(replica_path)
    _write_json(report_path, report_before)
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
        result = acquire_checkpoints(args.custody_root, args.v7_custody_root, allow_public_hub_downloads=args.allow_public_hub_downloads)
    elif args.command == "run":
        result = run_primary(args.custody_root, args.v7_custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root, args.v7_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
