"""Frozen V26 checkpoint selection, execution, and local-reproduction lifecycle.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v26-fresh-checkpoint-recovery-v1, and
global-experiment-assurance-lifecycle-v1. All checkpoint and experiment data
remain under external owner custody.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V26_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v26"
V26_REPLICA_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v26-replicator"
V26_PRIMARY_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v26-primary-toolchain"
V26_REPLICA_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v26-replicator-toolchain"
V25_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v25"
V7_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v7"
os.environ["HF_HOME"] = str(V26_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V26_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V26_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V26_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import replacement_loss_v22_runner as v22
import replacement_loss_v23_runner as v23
import replacement_loss_v25_runner as v25
import replacement_loss_v26_hub_adapter as hub_adapter


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v26-fresh-checkpoint-recovery-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = hub_adapter.PROTOCOL_ID
SOURCE_REVISION = "202f459d6d89df9a038d78d58e0af825d287fcd8"
FREEZE_PATH = "docs/replacement-loss-v26-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v26-fresh-checkpoint-recovery.md",
    "replacement_loss_v26_hub_adapter.py",
    "replacement_loss_v26_runner.py",
    "requirements-v26.lock.txt",
)
UPSTREAM_CODE_FILES = (
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v8_fixtures.py",
    "replacement_loss_v8_runner.py",
    "replacement_loss_v22_fixtures.py",
    "replacement_loss_v22_runner.py",
    "replacement_loss_v23_runner.py",
    "replacement_loss_v23_hub_adapter.py",
    "replacement_loss_v25_runner.py",
    "replacement_loss_v25_hub_adapter.py",
)
V25_PACKET_SHA256 = "9360f1c15820ce504fd1b059a468da6bb5ce26b4be86bd2369a769eaa74b161a"
V25_TERMINAL_SHA256 = "f582972e99ca7cba71baf53c7f61b9aeda6df5dc2db2afde30e8f1f7d183f057"
V25_SCREEN_SHA256 = "2e6822ac2752f7437c9b863f8c1f599ee6ea7bb6cf3caefc470cc7885345cdb1"
V25_SELECTION_SHA256 = "ec44abfd319fa483b10103025886ab6799ad23a16d6ed482c10989e0d4560e7d"
V25_DOWNLOAD_SHA256 = "bce4d47d7ad6fb03497563830aa3a8e017ff9e3f7f721a196a83407281560868"
V25_CHECKPOINT_SHA256 = "29cda78cde658a2212ee18c80d567dbf05c2dd5b7f8ae2dae0fc5378bf62f8c1"


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
        raise ValueError("V26 custody must be outside the repository")
    return root


def _custody_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V26_ROOT.resolve():
        raise ValueError(f"V26 primary custody is pinned to {V26_ROOT}")
    return root


def _replica_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V26_REPLICA_ROOT.resolve():
        raise ValueError(f"V26 replica custody is pinned to {V26_REPLICA_ROOT}")
    return root


def _v25_result_ids(screen: dict[str, Any]) -> set[str]:
    return {
        str(row.get("model_id", "")).strip().lower()
        for family in screen.get("family_screens", {}).values()
        for query in family.get("queries", [])
        for row in query.get("result_projection", [])
        if str(row.get("model_id", "")).strip()
    }


def _verify_v25_static() -> dict[str, Any]:
    """Verify frozen V25 bytes without recomputing its now-extended global ledger."""

    local_freeze = _read_json(_root() / "docs/replacement-loss-v25-freeze.json")
    external_freeze_path = V25_ROOT / "v25-frozen-packet-manifest.json"
    external_freeze = _read_json(external_freeze_path)
    if local_freeze != external_freeze:
        raise ValueError("V25 local and external freeze manifests differ")
    packet_sha = _canonical_digest({key: value for key, value in local_freeze.items() if key != "packet_sha256"})
    if packet_sha != V25_PACKET_SHA256 or local_freeze.get("packet_sha256") != packet_sha:
        raise ValueError("V25 frozen packet self-digest is invalid")
    for row in local_freeze.get("packet_files", []):
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V25 source changed: {relative}")
        if _sha(V25_ROOT / "v25-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V25 archived source changed: {relative}")
    ancestor_packets = {}
    for version in (8, 22, 23, 24):
        name = f"v{version}"
        local_ancestor = _read_json(_root() / f"docs/replacement-loss-{name}-freeze.json")
        external_ancestor = _read_json(V25_ROOT / f"{name}-frozen-packet-manifest.json")
        if local_ancestor != external_ancestor:
            raise ValueError(f"V{version} local and external manifests differ")
        ancestor_sha = _canonical_digest({key: value for key, value in external_ancestor.items() if key != "packet_sha256"})
        if external_ancestor.get("packet_sha256") != ancestor_sha:
            raise ValueError(f"V{version} frozen packet self-digest is invalid")
        for row in external_ancestor.get("packet_files", []):
            relative = str(row["path"])
            if _sha(_root() / relative) != row["sha256"]:
                raise ValueError(f"V{version} local frozen source changed: {relative}")
            if _sha(V25_ROOT / f"{name}-frozen-packet" / relative) != row["sha256"]:
                raise ValueError(f"V{version} archived source changed: {relative}")
        ancestor_packets[name] = external_ancestor["packet_sha256"]
    if ancestor_packets["v22"] != v23.V22_PACKET_SHA256:
        raise ValueError("V25 upstream V22 packet differs from the pinned V23 identity")
    screen_path = V25_ROOT / "v25-hub-screen.json"
    selection_path = V25_ROOT / "v25-checkpoint-selection.json"
    download_path = V25_ROOT / "v25-download-manifest.json"
    checkpoint_path = V25_ROOT / "checkpoints.json"
    terminal_path = V25_ROOT / "replacement-loss-v25-terminal-report.json"
    expected_files = (
        (screen_path, V25_SCREEN_SHA256),
        (selection_path, V25_SELECTION_SHA256),
        (download_path, V25_DOWNLOAD_SHA256),
        (checkpoint_path, V25_CHECKPOINT_SHA256),
        (terminal_path, V25_TERMINAL_SHA256),
    )
    for path, expected in expected_files:
        if _sha(path) != expected:
            raise ValueError(f"V25 upstream evidence changed: {path.name}")
    screen = _read_json(screen_path)
    selection = _read_json(selection_path)
    terminal = _read_json(terminal_path)
    if screen.get("v25_packet_sha256") != V25_PACKET_SHA256:
        raise ValueError("V25 screen does not bind its frozen packet")
    if selection.get("packet_sha256") != V25_PACKET_SHA256 or len(selection.get("selected_checkpoints", [])) != 3:
        raise ValueError("V25 selected-checkpoint receipt is incomplete")
    terminal_body = {key: value for key, value in terminal.items() if key != "terminal_report_sha256"}
    if terminal.get("terminal_report_sha256") != _canonical_digest(terminal_body):
        raise ValueError("V25 terminal receipt self-digest is invalid")
    if terminal.get("outcome_status") != "INCONCLUSIVE_EXECUTION" or terminal.get("phase") != "PRIMARY_EVALUATION":
        raise ValueError("V25 terminal status differs from the recorded protobuf stop")
    ids = _v25_result_ids(screen)
    selected_ids = {str(row.get("model_id", "")).strip().lower() for row in selection["selected_checkpoints"]}
    if not selected_ids.issubset(ids) or len(selected_ids) != 3:
        raise ValueError("V25 selection is not contained in the V25 screened identity set")
    return {
        "packet_sha256": V25_PACKET_SHA256,
        "ancestor_packet_sha256": ancestor_packets,
        "screen_sha256": V25_SCREEN_SHA256,
        "selection_sha256": V25_SELECTION_SHA256,
        "download_manifest_sha256": V25_DOWNLOAD_SHA256,
        "checkpoint_manifest_sha256": V25_CHECKPOINT_SHA256,
        "terminal_sha256": V25_TERMINAL_SHA256,
        "screen_identity_count": len(ids),
        "selected_identity_count": len(selected_ids),
        "terminal_outcome": terminal["outcome_status"],
    }


def _prior_ledger(current_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    ids, rows = v23._v22_prior_ids(current_root)
    v25_ids = _v25_result_ids(_read_json(V25_ROOT / "v25-hub-screen.json"))
    if len(v25_ids) != _verify_v25_static()["screen_identity_count"]:
        raise ValueError("V25 identity projection changed")
    if not v25_ids.issubset(ids):
        raise ValueError("prior identity ledger omits V25 screened identities")
    return ids, rows


def _upstream_source_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in UPSTREAM_CODE_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V26 upstream code dependency is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    return rows


def _copy_verified_file(source: Path, destination: Path) -> None:
    expected = _sha(source)
    if destination.exists():
        if not destination.is_file() or _sha(destination) != expected:
            raise ValueError(f"preexisting V26 custody file does not match source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_verified_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir() or _tree_digest(source) != _tree_digest(destination):
            raise ValueError(f"preexisting V26 custody directory differs from source: {destination}")
        return
    shutil.copytree(source, destination)


def freeze_packet(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    local_freeze = _root() / FREEZE_PATH
    external_manifest = root / "v26-frozen-packet-manifest.json"
    archive = root / "v26-frozen-packet"
    if local_freeze.exists() or external_manifest.exists():
        raise FileExistsError("V26 is immutable after freeze; use a new protocol version to change it")
    evidence = _verify_v25_static()
    source_rows = []
    for relative in PACKET_FILES:
        source = _root() / relative
        if not source.is_file():
            raise FileNotFoundError(f"V26 packet input is missing: {relative}")
        source_rows.append({"path": relative, "sha256": _sha(source), "bytes": source.stat().st_size})

    root.mkdir(parents=True, exist_ok=True)
    upstream_files = (
        "v8-frozen-packet-manifest.json",
        "v22-frozen-packet-manifest.json",
        "v23-frozen-packet-manifest.json",
        "v24-frozen-packet-manifest.json",
        "v25-frozen-packet-manifest.json",
        "v24-hub-screen.json",
        "v24-hub-screen-progress.json",
        "replacement-loss-v24-terminal-report.json",
        "v25-hub-screen.json",
        "v25-hub-screen-progress.json",
        "v25-checkpoint-selection.json",
        "v25-download-manifest.json",
        "v25-selection-binding.json",
        "v8-disjointness-preflight.json",
        "replacement-loss-v25-terminal-report.json",
    )
    for name in upstream_files:
        source = V25_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"pinned V25 upstream evidence is missing: {name}")
        _copy_verified_file(source, root / name)
    _copy_verified_file(V25_ROOT / "checkpoints.json", root / "v25-checkpoints.json")
    for name in ("v8-frozen-packet", "v22-frozen-packet", "v23-frozen-packet", "v24-frozen-packet", "v25-frozen-packet"):
        _copy_verified_tree(V25_ROOT / name, root / name)
    for row in source_rows:
        relative = str(row["path"])
        target = archive / relative
        _copy_verified_file(_root() / relative, target)
    upstream_source_rows = _upstream_source_rows()
    for row in upstream_source_rows:
        relative = str(row["path"])
        target = archive / "upstream-source" / relative
        _copy_verified_file(_root() / relative, target)

    allowed_seed_names = {
        *upstream_files,
        "v25-checkpoints.json",
        "v8-frozen-packet", "v22-frozen-packet", "v23-frozen-packet", "v24-frozen-packet", "v25-frozen-packet",
        "v26-frozen-packet",
        "v26-prefreeze-attempt-001",
    }
    unexpected_seed = sorted(path.name for path in root.iterdir() if path.name not in allowed_seed_names)
    if unexpected_seed:
        raise ValueError(f"unexpected files in partial V26 setup custody: {unexpected_seed}")

    prior_ids, prior_rows = _prior_ledger(root)
    lock_path = _root() / "requirements-v26.lock.txt"
    expected_packages = [line.strip() for line in lock_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_state": "FROZEN_BEFORE_V26_PUBLIC_HUB_QUERY_OR_TARGET_SELECTION",
        "packet_files": source_rows,
        "upstream_code_dependencies": upstream_source_rows,
        "upstream_v25_packet_sha256": evidence["packet_sha256"],
        "upstream_v25_static_verification": evidence,
        "upstream_v25_dynamic_packet_verification": "VERIFIED_BEFORE_V26_PREFREEZE_STOP; later checks use static byte bindings because the historical ledger grows with future packets",
        "upstream_v25_freeze_manifest_sha256": _sha(V25_ROOT / "v25-frozen-packet-manifest.json"),
        "upstream_v25_screen_identity_ledger_sha256": _canonical_digest(sorted(_v25_result_ids(_read_json(V25_ROOT / "v25-hub-screen.json")))),
        "upstream_v22_packet_sha256": v23.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "pre_v26_excluded_identity_count": len(prior_ids),
        "pre_v26_identity_ledger_sha256": _canonical_digest(prior_rows),
        "prior_ledger_includes_all_v25_screen_results": True,
        "dependency_manifest_sha256": _sha(lock_path),
        "dependency_inventory": {
            "distribution_count": len(expected_packages),
            "packages": expected_packages,
            "primary_and_clean_replica_must_match": True,
        },
        "runtime": {
            "python": "3.12.13",
            "platform": "macOS arm64 local CPU only",
            "primary_runtime": str(V26_PRIMARY_RUNTIME),
            "clean_replica_runtime": str(V26_REPLICA_RUNTIME),
            "hub_client": "huggingface_hub==0.36.2",
            "protobuf": "protobuf==5.29.6",
            "auth": "token=False",
            "hub_telemetry": False,
            "hosted_inference": False,
            "remote_workers": 0,
        },
        "custody_roots": {"primary": str(V26_ROOT), "clean_replica": str(V26_REPLICA_ROOT)},
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
        "nonclaims": [
            "unseen architecture families", "all architectures", "whole-model mechanistic identity",
            "independent science", "consciousness", "alignment", "production readiness",
        ],
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    _write_json(external_manifest, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": payload["packet_sha256"],
        "source_revision": SOURCE_REVISION,
        "pre_v26_excluded_identity_count": len(prior_ids),
        "upstream_v25_screen_identity_count": evidence["screen_identity_count"],
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
    }


def verify_packet(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    local = _read_json(_root() / FREEZE_PATH)
    external = _read_json(root / "v26-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("local and external V26 freeze manifests differ")
    expected = _canonical_digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("packet_sha256") != expected or local.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V26 packet identity or self-digest is invalid")
    for row in local["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V26 frozen source changed: {relative}")
        if _sha(root / "v26-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V26 archived source changed: {relative}")
    for row in local["upstream_code_dependencies"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen upstream code dependency changed: {relative}")
        if _sha(root / "v26-frozen-packet" / "upstream-source" / relative) != row["sha256"]:
            raise ValueError(f"archived upstream code dependency changed: {relative}")
    evidence = _verify_v25_static()
    if evidence != local.get("upstream_v25_static_verification"):
        raise ValueError("static V25 evidence binding differs from V26 freeze")
    if _sha(V25_ROOT / "v25-frozen-packet-manifest.json") != local.get("upstream_v25_freeze_manifest_sha256"):
        raise ValueError("V25 frozen manifest changed")
    prior_ids, prior_rows = _prior_ledger(root)
    if len(prior_ids) != local.get("pre_v26_excluded_identity_count"):
        raise ValueError("pre-V26 exclusion identity count changed")
    if _canonical_digest(prior_rows) != local.get("pre_v26_identity_ledger_sha256"):
        raise ValueError("pre-V26 exclusion ledger changed")
    return {
        "protocol_id": PROTOCOL_ID,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": local["packet_sha256"],
        "pre_v26_excluded_identity_count": len(prior_ids),
        "upstream_v25_packet_sha256": evidence["packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def check_runtime(role: str) -> dict[str, Any]:
    if role not in {"primary", "replicator"}:
        raise ValueError("runtime role must be primary or replicator")
    verify_packet(V26_ROOT)
    freeze = _read_json(_root() / FREEZE_PATH)
    packet = freeze.get("packet_sha256")
    expected_root = V26_PRIMARY_RUNTIME if role == "primary" else V26_REPLICA_RUNTIME
    executable = Path(sys.executable).resolve()
    if expected_root.resolve() not in executable.parents:
        raise ValueError(f"runtime executable is outside the frozen {role} environment: {executable}")
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError(f"expected Python 3.12.13, found {platform.python_version()}")
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise ValueError("frozen V26 runtime requires local macOS arm64")
    lock = [line.strip() for line in (_root() / "requirements-v26.lock.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    actual = sorted(
        f"{re.sub(r'[-_.]+', '-', dist.metadata['Name']).lower()}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    expected = sorted(re.sub(r'[-_.]+', '-', line.split("==", 1)[0]).lower() + "==" + line.split("==", 1)[1] for line in lock)
    if actual != expected:
        raise ValueError(f"installed package inventory differs from V26 lock: actual={actual!r}")
    import google.protobuf  # noqa: F401
    import huggingface_hub  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    package_digest = _canonical_digest(actual)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet,
        "role": role,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "executable": str(executable),
        "runtime_root": str(expected_root),
        "package_inventory": actual,
        "package_inventory_sha256": package_digest,
        "requirements_lock_sha256": _sha(_root() / "requirements-v26.lock.txt"),
        "protobuf_version": importlib.metadata.version("protobuf"),
        "verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = _custody_root(V26_ROOT) / f"v26-{role}-runtime-check.json"
    _write_json(path, receipt)
    return receipt


def _verify_runtime_pair(root: Path) -> dict[str, Any]:
    primary_path = root / "v26-primary-runtime-check.json"
    replica_path = root / "v26-replicator-runtime-check.json"
    primary = _read_json(primary_path)
    replica = _read_json(replica_path)
    packet_sha = _read_json(root / "v26-frozen-packet-manifest.json")["packet_sha256"]
    if any(row.get("packet_sha256") != packet_sha or not row.get("verified") for row in (primary, replica)):
        raise ValueError("both runtime receipts must bind and pass the frozen V26 packet")
    if primary.get("package_inventory_sha256") != replica.get("package_inventory_sha256"):
        raise ValueError("primary and clean-replica package inventories differ")
    if primary.get("executable") == replica.get("executable") or primary.get("runtime_root") == replica.get("runtime_root"):
        raise ValueError("primary and clean replica must use distinct runtime paths")
    return {
        "primary_receipt_sha256": _sha(primary_path),
        "replica_receipt_sha256": _sha(replica_path),
        "package_inventory_sha256": primary["package_inventory_sha256"],
        "distinct_runtime_paths": True,
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
        "data_and_artifact_custody": f"V26 artifacts remain under external custody {root}",
        "known_limitations": [
            "same-owner local work only",
            "three named model families; no architecture-held-out test",
            "one MLP output projection per checkpoint",
            "frozen V8 method defines no formal behavior-versus-mechanism divergence statistic",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "next_blocked_authority_gate": "resolve the recorded V26 stop in a new frozen continuation before a scientific claim",
        "monetary_cost_usd": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    if error is not None:
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
    receipt["terminal_report_sha256"] = _canonical_digest(receipt)
    _write_json(root / "replacement-loss-v26-terminal-report.json", receipt)
    return receipt


def acquire_checkpoints(custody_root: str | Path, *, allow_public_hub_downloads: bool) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if allow_public_hub_downloads is not True:
        raise PermissionError("run requires explicit --allow-public-hub-downloads opt-in")
    runtime_pair = _verify_runtime_pair(root)
    progress_path = root / "v26-hub-screen-progress.json"
    one_shot = (
        progress_path,
        root / "v26-hub-screen.json",
        root / "v26-checkpoint-selection.json",
        root / "v26-download-manifest.json",
        root / "v26-selection-binding.json",
        root / "replacement-loss-v26-terminal-report.json",
        root / "checkpoints.json",
    )
    existing = [path.name for path in one_shot if path.exists()]
    if existing:
        raise FileExistsError(f"V26 acquisition is one-shot; prior receipts exist: {existing}")
    prior_ids, prior_rows = _prior_ledger(root)
    latest: dict[str, dict[str, Any]] = {}

    def save_progress(receipt: dict[str, Any]) -> None:
        latest[str(receipt["family"])] = receipt
        _write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "completed_family_snapshots": latest,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    try:
        screen = hub_adapter.screen_public_hub(prior_ids, on_progress=save_progress)
    except Exception as exc:
        terminal = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase="PUBLIC_METADATA_SCREEN",
            error=exc,
            progress_receipt_sha256=_sha(progress_path) if progress_path.is_file() else None,
            prior_identity_ledger_sha256=_canonical_digest(prior_rows),
            runtime_pair=runtime_pair,
            model_weights_downloaded=False,
            candidate_preparation_started=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"]}

    screen["v26_packet_sha256"] = packet["packet_sha256"]
    screen["prior_checkpoint_identity_count"] = len(prior_ids)
    screen["prior_checkpoint_ledger_sha256"] = _canonical_digest(prior_rows)
    screen_path = root / "v26-hub-screen.json"
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
            "assurance_scope": "complete metadata-valid bounded screen and frozen packet verification only",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_RUN",
            "behavior_and_mechanism_diverged": "NOT_ASSESSED_NO_MODEL_RUN",
            "data_and_artifact_custody": f"V26 metadata receipts remain under {root}; no model weights or prompts acquired",
            "known_limitations": ["frozen Hub result bounds only", "no scientific evaluation ran"],
            "claim_ceiling": "BOUNDED_PUBLIC_METADATA_SCREEN_ONLY",
            "hub_screen_sha256": _sha(screen_path),
            "progress_receipt_sha256": _sha(progress_path) if progress_path.is_file() else None,
            "prior_identity_ledger_sha256": _canonical_digest(prior_rows),
            "eligible_checkpoint_count_by_family": {family: row["selected_identity_count"] for family, row in screen["family_screens"].items()},
            "model_weights_downloaded": False,
            "candidate_preparation_started": False,
            "primary_run_status": "NOT_RUN",
            "replication_status": "NOT_RUN",
            "monetary_cost_usd": 0,
            "next_blocked_authority_gate": "availability of eligible fresh public base checkpoints beyond the frozen V26 query bounds",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        terminal["terminal_report_sha256"] = _canonical_digest(terminal)
        _write_json(root / "replacement-loss-v26-terminal-report.json", terminal)
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "hub_screen_sha256": _sha(screen_path)}

    phase = "RESOURCE_CHECK"
    try:
        reserve = int(screen["selected_download_bytes"]) + 1_000_000_000
        if shutil.disk_usage(root).free < reserve:
            raise OSError("external V26 custody lacks the frozen download reserve")
        selection = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selected_after_v26_freeze": True,
            "selected_checkpoints": screen["selected_checkpoints"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v26-checkpoint-selection.json", selection)
        phase = "MODEL_DOWNLOAD"
        downloaded = hub_adapter.download_selected(root, screen, allow_public_hub_downloads=True)
        download_manifest = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "selection_sha256": _sha(root / "v26-checkpoint-selection.json"),
            "downloaded": downloaded,
            "total_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloaded),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v26-download-manifest.json", download_manifest)
        specs = [{"family": row["family"], "model_id": row["model_id"], "revision": row["revision"], "local_path": row["local_path"]} for row in downloaded]
        phase = "CHECKPOINT_MANIFEST_BINDING"
        with v23._v22_external_exclusions(root, prior_ids):
            selection_result = v22.write_checkpoint_manifest(root, specs, V7_ROOT)
        binding = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selection_sha256": _sha(root / "v26-checkpoint-selection.json"),
            "download_manifest_sha256": _sha(root / "v26-download-manifest.json"),
            "v22_checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
            "v22_selection_receipt": selection_result,
            "runtime_pair": runtime_pair,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v26-selection-binding.json", binding)
    except Exception as exc:
        terminal = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase=phase,
            error=exc,
            hub_screen_sha256=_sha(screen_path),
            selection_sha256=_sha(root / "v26-checkpoint-selection.json") if (root / "v26-checkpoint-selection.json").is_file() else None,
            download_manifest_sha256=_sha(root / "v26-download-manifest.json") if (root / "v26-download-manifest.json").is_file() else None,
            checkpoint_manifest_sha256=_sha(root / "checkpoints.json") if (root / "checkpoints.json").is_file() else None,
            model_download_attempted=phase == "MODEL_DOWNLOAD" or (root / "v26-download-manifest.json").exists(),
            candidate_preparation_started=False,
            primary_run_status="NOT_RUN",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(root / "replacement-loss-v26-terminal-report.json")}
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": "TARGETS_ACQUIRED_READY_FOR_PRIMARY",
        "checkpoint_count": 3,
        "downloaded_bytes": download_manifest["total_downloaded_bytes"],
        "selection_binding_sha256": _sha(root / "v26-selection-binding.json"),
    }


def _aggregate(root: Path, packet_sha: str, runner_receipt: dict[str, Any]) -> dict[str, Any]:
    upstream = _read_json(root / "replacement-loss-v22-report.json")
    binding = _read_json(root / "v26-selection-binding.json")
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet_sha,
        "upstream_v25_packet_sha256": V25_PACKET_SHA256,
        "upstream_v22_packet_sha256": v23.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "selection_binding_sha256": _sha(root / "v26-selection-binding.json"),
        "download_manifest_sha256": _sha(root / "v26-download-manifest.json"),
        "outcome_status": upstream["outcome_status"],
        "assurance_level": upstream["assurance_level"],
        "behavior_and_mechanism_diverged": "NOT_SEPARATELY_DEFINED_BY_FROZEN_V8_METHOD; behavior, causal effects, and activation rescue are separate metrics",
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "PENDING_SECOND_CLEAN_RUNTIME",
        "case_count": upstream["case_count"],
        "cases": upstream["cases"],
        "runtime": upstream["runtime"],
        "runtime_pair": binding["runtime_pair"],
        "budget_and_resource_measurements": upstream["budget_and_resource_measurements"],
        "data_and_artifact_custody": f"models, prompts, captures, candidates, predictions, outcomes, and diagnostics remain under external owner custody {root}",
        "known_limitations": [
            "three fresh checkpoint identities within the same three architecture families",
            "one MLP output projection per checkpoint",
            "synthetic prompts and same-owner reproduction",
            "bounded public Hub queries defined by this packet",
            "no formal behavior-versus-mechanism divergence statistic in the frozen V8 method",
            "monetary cost is zero; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "primary_runner_receipt": runner_receipt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "replacement-loss-v26-report.json", report)
    return report


def run_primary(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    _verify_runtime_pair(root)
    binding = _read_json(root / "v26-selection-binding.json")
    if binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V26 selection binding does not match the frozen packet")
    prior_outputs = (
        root / "replacement-loss-v26-report.json",
        root / "replacement-loss-v22-report.json",
        root / "candidate-preparation.json",
        root / "candidate-freeze.json",
        root / "challenge-manifest.json",
    )
    if any(path.exists() for path in prior_outputs):
        raise FileExistsError("V26 primary evaluation is one-shot; prior execution output exists")
    prior_ids, _rows = _prior_ledger(root)
    try:
        with v23._v22_external_exclusions(root, prior_ids):
            result = v22.run_primary(root, V7_ROOT)
        report = _aggregate(root, packet["packet_sha256"], result)
    except Exception as exc:
        terminal = _write_stop(
            root,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_EXECUTION",
            phase="PRIMARY_EVALUATION",
            error=exc,
            model_weights_downloaded=True,
            candidate_preparation_started=(root / "candidate-preparation.json").is_file(),
            candidate_freeze_sha256=_sha(root / "candidate-freeze.json") if (root / "candidate-freeze.json").is_file() else None,
            challenge_manifest_sha256=_sha(root / "challenge-manifest.json") if (root / "challenge-manifest.json").is_file() else None,
            primary_run_status="INCONCLUSIVE",
            replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(root / "replacement-loss-v26-terminal-report.json")}
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "assurance_level": report["assurance_level"],
        "case_count": report["case_count"],
        "report_sha256": _sha(root / "replacement-loss-v26-report.json"),
    }


def replicate_packet(primary_custody_root: str | Path, replica_custody_root: str | Path) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _replica_root(replica_custody_root)
    packet = verify_packet(primary)
    runtime_pair = _verify_runtime_pair(primary)
    primary_report_path = primary / "replacement-loss-v26-report.json"
    report_before = _read_json(primary_report_path)
    binding = _read_json(primary / "v26-selection-binding.json")
    if report_before.get("packet_sha256") != packet["packet_sha256"] or binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V26 primary report or selection binding does not bind this packet")
    if binding.get("v22_checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V26 checkpoint manifest changed after primary evaluation")
    if report_before.get("outcome_status") not in {"PASS", "FAIL"}:
        raise ValueError("V26 reproduction requires a completed primary PASS or FAIL outcome")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("V26 clean reproduction is one-shot; replica custody already contains files")
    replica.mkdir(parents=True, exist_ok=True)
    prior_ids, _rows = _prior_ledger(primary)
    try:
        with v23._v22_external_exclusions(primary, prior_ids):
            result = v22.replicate_packet(primary, replica, V7_ROOT)
        upstream = _read_json(replica / "replacement-loss-v22-replication-report.json")
    except Exception as exc:
        terminal = _write_stop(
            replica,
            packet["packet_sha256"],
            outcome_status="INCONCLUSIVE_REPRODUCTION",
            phase="CLEAN_REPLICA_EVALUATION",
            error=exc,
            primary_v26_report_sha256=_sha(primary_report_path),
            primary_run_status=report_before.get("outcome_status"),
            replication_status="INCONCLUSIVE",
            runtime_pair=runtime_pair,
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(replica / "replacement-loss-v26-terminal-report.json")}
    reproduced = result.get("assurance_level") == "REPRODUCED_LOCAL"
    final = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report_before["outcome_status"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else result.get("assurance_level", "INCONCLUSIVE"),
        "reproducibility_status": "REPRODUCED_LOCAL" if reproduced else result.get("assurance_level", "INCONCLUSIVE"),
        "reviewer_status": "NOT_SUBMITTED",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "candidate_reconstruction_match": bool(upstream.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(upstream.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(upstream.get("dependencies_match")),
        "distinct_clean_runtime": bool(upstream.get("distinct_clean_runtime")) and runtime_pair["distinct_runtime_paths"],
        "source_identity_match": bool(upstream.get("source_identity_match")),
        "runtime_pair": runtime_pair,
        "data_and_artifact_custody": f"primary and clean-replica evidence remains under {primary} and {replica}",
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "independent science", "consciousness", "alignment", "production readiness"],
        "upstream_v22_replication_report_sha256": _sha(replica / "replacement-loss-v22-replication-report.json"),
        "primary_v26_report_sha256_before_reconciliation": _sha(primary_report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted in this local task",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    replication_path = replica / "replacement-loss-v26-replication-report.json"
    _write_json(replication_path, final)
    report_before["assurance_level"] = final["assurance_level"]
    report_before["reproducibility_status"] = final["reproducibility_status"]
    report_before["replication_report_sha256"] = _sha(replication_path)
    _write_json(primary_report_path, report_before)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": final["assurance_level"],
        "outcome_status": final["outcome_status"],
        "replication_report_sha256": _sha(replication_path),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--custody-root", default=str(V26_ROOT))
    verify = commands.add_parser("verify")
    verify.add_argument("--custody-root", default=str(V26_ROOT))
    runtime = commands.add_parser("runtime-check")
    runtime.add_argument("--role", choices=("primary", "replicator"), required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--custody-root", default=str(V26_ROOT))
    acquire.add_argument("--allow-public-hub-downloads", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--custody-root", default=str(V26_ROOT))
    replicate = commands.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", default=str(V26_ROOT))
    replicate.add_argument("--replica-custody-root", default=str(V26_REPLICA_ROOT))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root)
    elif args.command == "runtime-check":
        result = check_runtime(args.role)
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
