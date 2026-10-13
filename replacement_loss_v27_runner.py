"""V27 frozen lifecycle for fresh-checkpoint replacement-loss evaluation.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v27-fresh-checkpoint-recovery-v1, and
global-experiment-assurance-lifecycle-v1. All model and experiment artifacts
remain in external owner custody.
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
import site
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V27_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v27"
V27_REPLICA_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v27-replicator"
V27_PRIMARY_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v27-primary-toolchain"
V27_REPLICA_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v27-replicator-toolchain"
V26_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v26"
V25_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v25"
V7_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v7"
os.environ["HF_HOME"] = str(V27_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V27_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V27_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V27_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import replacement_loss_v22_runner as v22
import replacement_loss_v23_runner as v23
import replacement_loss_v26_runner as v26
import replacement_loss_v27_hub_adapter as hub_adapter

# Earlier frozen helpers set their own cache roots during import. V27 restores
# its exclusive external cache paths before any Hub operation.
os.environ["HF_HOME"] = str(V27_ROOT / "hf-home")
os.environ["HF_HUB_CACHE"] = str(V27_ROOT / "hub-cache")
os.environ["HF_ASSETS_CACHE"] = str(V27_ROOT / "assets-cache")
os.environ["HF_XET_CACHE"] = str(V27_ROOT / "xet-cache")
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v27-fresh-checkpoint-recovery-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = hub_adapter.PROTOCOL_ID
SOURCE_REVISION = "202f459d6d89df9a038d78d58e0af825d287fcd8"
FREEZE_PATH = "docs/replacement-loss-v27-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v27-fresh-checkpoint-recovery.md",
    "replacement_loss_v27_hub_adapter.py",
    "replacement_loss_v27_runner.py",
    "requirements-v27.lock.txt",
)
UPSTREAM_CODE_FILES = (
    "requirements.txt",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
    "replacement_loss_v6_runner.py",
    "replacement_loss_v7_runner.py",
    "replacement_loss_v8_fixtures.py",
    "replacement_loss_v8_runner.py",
    "replacement_loss_v22_fixtures.py",
    "replacement_loss_v22_runner.py",
    "replacement_loss_v23_runner.py",
    "replacement_loss_v23_hub_adapter.py",
    "replacement_loss_v25_runner.py",
    "replacement_loss_v25_hub_adapter.py",
    "replacement_loss_v26_runner.py",
    "replacement_loss_v26_hub_adapter.py",
)
V26_PACKET_SHA256 = "1b92319ef36a98dbec6d534fa9a2f13cd80df528dab64639be59c8d2e453cfdc"


def _root() -> Path:
    return Path(__file__).resolve().parent


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _external_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("V27 custody must stay outside the repository")
    return root


def _custody_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V27_ROOT.resolve():
        raise ValueError(f"V27 primary custody is pinned to {V27_ROOT}")
    return root


def _replica_root(path: str | Path) -> Path:
    root = _external_root(path)
    if root != V27_REPLICA_ROOT.resolve():
        raise ValueError(f"V27 replica custody is pinned to {V27_REPLICA_ROOT}")
    return root


def _v26_packet_static() -> dict[str, Any]:
    local = _read_json(_root() / "docs/replacement-loss-v26-freeze.json")
    external = _read_json(V26_ROOT / "v26-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("V26 local and external frozen manifests differ")
    packet_sha = _digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if packet_sha != V26_PACKET_SHA256 or local.get("packet_sha256") != packet_sha:
        raise ValueError("V26 frozen packet self-digest is invalid")
    for row in local.get("packet_files", []):
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V26 local frozen source changed: {relative}")
        if _sha(V26_ROOT / "v26-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V26 archived frozen source changed: {relative}")
    for row in local.get("upstream_code_dependencies", []):
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"V26 upstream method source changed: {relative}")
        if _sha(V26_ROOT / "v26-frozen-packet" / "upstream-source" / relative) != row["sha256"]:
            raise ValueError(f"V26 archived upstream method source changed: {relative}")
    v25 = v26._verify_v25_static()
    if v25 != local.get("upstream_v25_static_verification"):
        raise ValueError("V26 V25 evidence binding differs")
    attempt_paths = (
        V26_ROOT / "v26-prefreeze-attempt-001" / "receipt.json",
        V26_ROOT / "v26-runtime-prefreeze-attempt-001.json",
        V26_ROOT / "v26-runtime-copy-attempt-002.json",
        V26_ROOT / "v26-runtime-copy-followup-002.json",
    )
    attempts = []
    for path in attempt_paths:
        row = _read_json(path)
        body = {key: value for key, value in row.items() if key != "receipt_sha256"}
        if row.get("receipt_sha256") != _digest(body):
            raise ValueError(f"V26 failed setup receipt self-digest is invalid: {path.name}")
        if row.get("hub_queries_issued") is not False or row.get("model_weights_downloaded") is not False:
            raise ValueError("V26 preselection failure receipt claims model acquisition")
        attempts.append({"path": path.name, "sha256": _sha(path), "status": row["outcome_status"]})
    if (V26_ROOT / "v26-hub-screen.json").exists() or (V26_ROOT / "v26-checkpoint-selection.json").exists():
        raise ValueError("V26 unexpectedly performed Hub selection")
    return {"packet_sha256": packet_sha, "v25_evidence": v25, "prefreeze_receipts": attempts}


def _prior_ledger(current_root: Path) -> tuple[set[str], list[dict[str, Any]]]:
    return v23._v22_prior_ids(current_root)


def _copy_file(source: Path, destination: Path) -> None:
    expected = _sha(source)
    if destination.exists():
        if not destination.is_file() or _sha(destination) != expected:
            raise ValueError(f"existing V27 seed file differs: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        actual = {p.relative_to(destination).as_posix(): _sha(p) for p in destination.rglob("*") if p.is_file()}
        expected = {p.relative_to(source).as_posix(): _sha(p) for p in source.rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError(f"existing V27 seed directory differs: {destination}")
        return
    shutil.copytree(source, destination)


def freeze_packet(custody_root: str | Path = V27_ROOT) -> dict[str, Any]:
    root = _custody_root(custody_root)
    if (_root() / FREEZE_PATH).exists() or (root / "v27-frozen-packet-manifest.json").exists():
        raise FileExistsError("V27 is immutable after freeze; use a new protocol version to change it")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("V27 custody root must be new or empty before freeze")
    # Full dynamic verification occurs before V27 has a screen that could
    # extend V26's historical identity ledger.
    v26_full = v26.verify_packet(V26_ROOT)
    upstream = _v26_packet_static()
    sources = []
    for relative in PACKET_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V27 packet source missing: {relative}")
        sources.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    code_rows = []
    for relative in UPSTREAM_CODE_FILES:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"V27 upstream code dependency missing: {relative}")
        code_rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})

    root.mkdir(parents=True, exist_ok=True)
    files = (
        "v8-frozen-packet-manifest.json", "v22-frozen-packet-manifest.json",
        "v23-frozen-packet-manifest.json", "v24-frozen-packet-manifest.json",
        "v25-frozen-packet-manifest.json", "v26-frozen-packet-manifest.json",
        "v24-hub-screen.json", "v24-hub-screen-progress.json",
        "replacement-loss-v24-terminal-report.json", "v25-hub-screen.json",
        "v25-hub-screen-progress.json", "v25-checkpoint-selection.json",
        "v25-download-manifest.json", "v25-selection-binding.json",
        "v8-disjointness-preflight.json", "replacement-loss-v25-terminal-report.json",
        "v26-runtime-prefreeze-attempt-001.json", "v26-runtime-copy-attempt-002.json",
        "v26-runtime-copy-followup-002.json",
    )
    for name in files:
        _copy_file(V26_ROOT / name, root / name)
    _copy_file(V26_ROOT / "v25-checkpoints.json", root / "v25-checkpoints.json")
    for name in ("v8", "v22", "v23", "v24", "v25", "v26"):
        _copy_tree(V26_ROOT / f"{name}-frozen-packet", root / f"{name}-frozen-packet")
    for row in sources:
        relative = str(row["path"])
        _copy_file(_root() / relative, root / "v27-frozen-packet" / relative)
    for row in code_rows:
        relative = str(row["path"])
        _copy_file(_root() / relative, root / "v27-frozen-packet" / "upstream-source" / relative)

    prior_ids, prior_rows = _prior_ledger(root)
    lock = [line.strip() for line in (_root() / "requirements-v27.lock.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_state": "FROZEN_BEFORE_V27_PUBLIC_HUB_QUERY_OR_TARGET_SELECTION",
        "packet_files": sources,
        "upstream_code_dependencies": code_rows,
        "upstream_v25_packet_sha256": v26._verify_v25_static()["packet_sha256"],
        "upstream_v26_packet_sha256": upstream["packet_sha256"],
        "upstream_v26_dynamic_packet_verification": v26_full,
        "upstream_v26_static_evidence": upstream,
        "upstream_v22_packet_sha256": v23.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "pre_v27_excluded_identity_count": len(prior_ids),
        "pre_v27_identity_ledger_sha256": _digest(prior_rows),
        "dependency_manifest_sha256": _sha(_root() / "requirements-v27.lock.txt"),
        "dependency_inventory": {
            "distribution_count": len(lock),
            "packages": lock,
            "primary_and_clean_replica_must_match": True,
        },
        "runtime": {
            "python": "3.12.13",
            "platform": "macOS arm64 local CPU only",
            "primary_runtime": str(V27_PRIMARY_RUNTIME),
            "clean_replica_runtime": str(V27_REPLICA_RUNTIME),
            "isolation_contract": "distinct sys.prefix and site-packages; include-system-site-packages=false; shared base executable permitted",
            "hub_client": "huggingface_hub==0.36.2",
            "protobuf": "protobuf==5.29.6",
            "auth": "token=False",
            "telemetry": False,
            "hosted_inference": False,
            "remote_workers": 0,
        },
        "custody_roots": {"primary": str(V27_ROOT), "clean_replica": str(V27_REPLICA_ROOT)},
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
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
    }
    payload["packet_sha256"] = _digest(payload)
    _write_json(root / "v27-frozen-packet-manifest.json", payload)
    _write_json(_root() / FREEZE_PATH, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": payload["packet_sha256"],
        "source_revision": SOURCE_REVISION,
        "pre_v27_excluded_identity_count": len(prior_ids),
        "freeze_state": payload["freeze_state"],
        "checkpoint_identities_selected": False,
    }


def verify_packet(custody_root: str | Path = V27_ROOT) -> dict[str, Any]:
    root = _custody_root(custody_root)
    local = _read_json(_root() / FREEZE_PATH)
    external = _read_json(root / "v27-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("local and external V27 freeze manifests differ")
    packet_sha = _digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("packet_sha256") != packet_sha or local.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V27 packet identity or self-digest is invalid")
    for row in local["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"] or _sha(root / "v27-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"V27 frozen source changed: {relative}")
    for row in local["upstream_code_dependencies"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"] or _sha(root / "v27-frozen-packet" / "upstream-source" / relative) != row["sha256"]:
            raise ValueError(f"V27 upstream code source changed: {relative}")
    upstream = _v26_packet_static()
    if upstream != local.get("upstream_v26_static_evidence"):
        raise ValueError("V26 static evidence differs from the V27 freeze")
    prior_ids, prior_rows = _prior_ledger(root)
    if len(prior_ids) != local.get("pre_v27_excluded_identity_count") or _digest(prior_rows) != local.get("pre_v27_identity_ledger_sha256"):
        raise ValueError("V27 preselection identity ledger changed")
    return {
        "protocol_id": PROTOCOL_ID,
        "source_revision": SOURCE_REVISION,
        "packet_sha256": packet_sha,
        "pre_v27_excluded_identity_count": len(prior_ids),
        "upstream_v26_packet_sha256": upstream["packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def check_runtime(role: str) -> dict[str, Any]:
    if role not in {"primary", "replicator"}:
        raise ValueError("runtime role must be primary or replicator")
    verify_packet(V27_ROOT)
    expected_root = V27_PRIMARY_RUNTIME if role == "primary" else V27_REPLICA_RUNTIME
    prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    purelib = Path(sysconfig.get_paths()["purelib"]).resolve()
    if prefix != expected_root.resolve():
        raise ValueError(f"virtual environment sys.prefix differs from frozen {role} root: {prefix}")
    if prefix == base_prefix:
        raise ValueError("runtime is not an isolated virtual environment")
    if expected_root.resolve() not in purelib.parents:
        raise ValueError(f"site-packages are outside frozen {role} environment: {purelib}")
    if site.ENABLE_USER_SITE:
        raise ValueError("user site-packages are enabled in the frozen runtime")
    pyvenv_cfg = (expected_root / "pyvenv.cfg").read_text(encoding="utf-8").lower()
    if "include-system-site-packages = false" not in pyvenv_cfg:
        raise ValueError("virtual environment must disable system site-packages")
    if sys.version_info[:3] != (3, 12, 13):
        raise ValueError(f"expected Python 3.12.13, found {platform.python_version()}")
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise ValueError("frozen V27 runtime requires local macOS arm64")
    lock = [line.strip() for line in (_root() / "requirements-v27.lock.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    normalize = lambda value: re.sub(r"[-_.]+", "-", value).lower()
    expected = sorted(normalize(line.split("==", 1)[0]) + "==" + line.split("==", 1)[1] for line in lock)
    actual = sorted(
        f"{normalize(dist.metadata['Name'])}=={dist.version}"
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    if actual != expected:
        raise ValueError(f"installed package inventory differs from V27 lock: {actual!r}")
    import google.protobuf  # noqa: F401
    import huggingface_hub  # noqa: F401
    import torch  # noqa: F401
    import transformers  # noqa: F401
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": _read_json(_root() / FREEZE_PATH)["packet_sha256"],
        "role": role,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "sys_prefix": str(prefix),
        "sys_base_prefix": str(base_prefix),
        "sys_executable": str(Path(sys.executable).absolute()),
        "sys_executable_realpath": str(Path(sys.executable).resolve()),
        "site_packages": str(purelib),
        "package_inventory": actual,
        "package_inventory_sha256": _digest(actual),
        "requirements_lock_sha256": _sha(_root() / "requirements-v27.lock.txt"),
        "protobuf_version": importlib.metadata.version("protobuf"),
        "system_site_packages": False,
        "verified": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(V27_ROOT / f"v27-{role}-runtime-check.json", receipt)
    return receipt


def _verify_runtime_pair(root: Path) -> dict[str, Any]:
    primary_path = root / "v27-primary-runtime-check.json"
    replica_path = root / "v27-replicator-runtime-check.json"
    primary, replica = _read_json(primary_path), _read_json(replica_path)
    packet_sha = _read_json(root / "v27-frozen-packet-manifest.json")["packet_sha256"]
    if any(row.get("packet_sha256") != packet_sha or row.get("verified") is not True for row in (primary, replica)):
        raise ValueError("both V27 runtime receipts must pass and bind the frozen packet")
    if primary["package_inventory_sha256"] != replica["package_inventory_sha256"]:
        raise ValueError("V27 primary and replica package inventories differ")
    if primary["sys_prefix"] == replica["sys_prefix"] or primary["site_packages"] == replica["site_packages"]:
        raise ValueError("V27 primary and replica must use distinct isolated environments")
    if primary["sys_base_prefix"] != replica["sys_base_prefix"]:
        raise ValueError("V27 primary and replica must share the frozen Python 3.12.13 base build")
    return {
        "primary_receipt_sha256": _sha(primary_path),
        "replica_receipt_sha256": _sha(replica_path),
        "package_inventory_sha256": primary["package_inventory_sha256"],
        "distinct_sys_prefix_and_site_packages": True,
        "shared_base_prefix": primary["sys_base_prefix"],
    }


def _write_stop(root: Path, packet_sha: str, *, outcome_status: str, phase: str, error: Exception | None = None, **details: Any) -> dict[str, Any]:
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
        "data_and_artifact_custody": f"V27 artifacts remain under external custody {root}",
        "known_limitations": ["same-owner local work only", "three named architecture families", "one MLP output projection per checkpoint"],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "next_blocked_authority_gate": "resolve this V27 stop in a new frozen continuation before a scientific claim",
        "monetary_cost_usd": 0,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    if error is not None:
        receipt["error_type"] = type(error).__name__
        receipt["error"] = str(error)
    receipt["terminal_report_sha256"] = _digest(receipt)
    _write_json(root / "replacement-loss-v27-terminal-report.json", receipt)
    return receipt


def acquire_checkpoints(custody_root: str | Path = V27_ROOT, *, allow_public_hub_downloads: bool) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if allow_public_hub_downloads is not True:
        raise PermissionError("public model downloads require --allow-public-hub-downloads")
    runtime_pair = _verify_runtime_pair(root)
    progress_path = root / "v27-hub-screen-progress.json"
    one_shot = (
        progress_path, root / "v27-hub-screen.json", root / "v27-checkpoint-selection.json",
        root / "v27-download-manifest.json", root / "v27-selection-binding.json",
        root / "replacement-loss-v27-terminal-report.json", root / "checkpoints.json",
    )
    if any(path.exists() for path in one_shot):
        raise FileExistsError("V27 acquisition is one-shot; prior receipts exist")
    prior_ids, prior_rows = _prior_ledger(root)
    family_progress: dict[str, dict[str, Any]] = {}

    def save_progress(receipt: dict[str, Any]) -> None:
        family_progress[str(receipt["family"])] = receipt
        _write_json(progress_path, {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "completed_family_snapshots": family_progress,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    try:
        screen = hub_adapter.screen_public_hub(prior_ids, on_progress=save_progress)
    except Exception as exc:
        terminal = _write_stop(
            root, packet["packet_sha256"], outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase="PUBLIC_METADATA_SCREEN", error=exc,
            progress_receipt_sha256=_sha(progress_path) if progress_path.is_file() else None,
            prior_identity_ledger_sha256=_digest(prior_rows), runtime_pair=runtime_pair,
            model_weights_downloaded=False, candidate_preparation_started=False,
            primary_run_status="NOT_RUN", replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"]}
    screen["v27_packet_sha256"] = packet["packet_sha256"]
    screen["prior_checkpoint_identity_count"] = len(prior_ids)
    screen["prior_checkpoint_ledger_sha256"] = _digest(prior_rows)
    screen_path = root / "v27-hub-screen.json"
    _write_json(screen_path, screen)
    if screen["status"] != "READY_TO_DOWNLOAD":
        terminal = {
            "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE, "assurance_slice": ASSURANCE_SLICE,
            "source_revision": SOURCE_REVISION, "packet_sha256": packet["packet_sha256"],
            "outcome_status": "NO_ELIGIBLE_FRESH_TARGET", "assurance_level": "INTERNAL_PASS",
            "assurance_scope": "complete metadata-valid bounded V27 screen only",
            "reviewer_status": "NOT_SUBMITTED", "reproducibility_status": "NOT_RUN",
            "behavior_and_mechanism_diverged": "NOT_ASSESSED_NO_MODEL_RUN",
            "data_and_artifact_custody": f"metadata receipts remain under {root}; no weights or prompts acquired",
            "known_limitations": ["frozen Hub result bounds only", "no scientific evaluation ran"],
            "claim_ceiling": "BOUNDED_PUBLIC_METADATA_SCREEN_ONLY",
            "hub_screen_sha256": _sha(screen_path),
            "progress_receipt_sha256": _sha(progress_path) if progress_path.is_file() else None,
            "prior_identity_ledger_sha256": _digest(prior_rows),
            "eligible_checkpoint_count_by_family": {family: row["selected_identity_count"] for family, row in screen["family_screens"].items()},
            "model_weights_downloaded": False, "candidate_preparation_started": False,
            "primary_run_status": "NOT_RUN", "replication_status": "NOT_RUN",
            "monetary_cost_usd": 0,
            "next_blocked_authority_gate": "eligible fresh public checkpoints beyond the frozen V27 search bounds",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        terminal["terminal_report_sha256"] = _digest(terminal)
        _write_json(root / "replacement-loss-v27-terminal-report.json", terminal)
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "hub_screen_sha256": _sha(screen_path)}

    phase = "RESOURCE_CHECK"
    try:
        if shutil.disk_usage(root).free < int(screen["selected_download_bytes"]) + 1_000_000_000:
            raise OSError("external V27 custody lacks the frozen download reserve")
        selection = {
            "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
            "continuation_slice": CONTINUATION_SLICE, "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path), "selected_after_v27_freeze": True,
            "selected_checkpoints": screen["selected_checkpoints"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v27-checkpoint-selection.json", selection)
        phase = "MODEL_DOWNLOAD"
        downloaded = hub_adapter.download_selected(root, screen, allow_public_hub_downloads=True)
        downloads = {
            "protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"],
            "selection_sha256": _sha(root / "v27-checkpoint-selection.json"),
            "downloaded": downloaded,
            "total_downloaded_bytes": sum(int(row["downloaded_bytes"]) for row in downloaded),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v27-download-manifest.json", downloads)
        specs = [{key: row[key] for key in ("family", "model_id", "revision", "local_path")} for row in downloaded]
        phase = "CHECKPOINT_MANIFEST_BINDING"
        with v23._v22_external_exclusions(root, prior_ids):
            selection_result = v22.write_checkpoint_manifest(root, specs, V7_ROOT)
        binding = {
            "protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"],
            "hub_screen_sha256": _sha(screen_path),
            "selection_sha256": _sha(root / "v27-checkpoint-selection.json"),
            "download_manifest_sha256": _sha(root / "v27-download-manifest.json"),
            "v22_checkpoint_manifest_sha256": _sha(root / "checkpoints.json"),
            "v22_selection_receipt": selection_result, "runtime_pair": runtime_pair,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _write_json(root / "v27-selection-binding.json", binding)
    except Exception as exc:
        terminal = _write_stop(
            root, packet["packet_sha256"], outcome_status="INCONCLUSIVE_SETUP_OR_EXECUTION",
            phase=phase, error=exc, hub_screen_sha256=_sha(screen_path),
            selection_sha256=_sha(root / "v27-checkpoint-selection.json") if (root / "v27-checkpoint-selection.json").is_file() else None,
            download_manifest_sha256=_sha(root / "v27-download-manifest.json") if (root / "v27-download-manifest.json").is_file() else None,
            checkpoint_manifest_sha256=_sha(root / "checkpoints.json") if (root / "checkpoints.json").is_file() else None,
            model_download_attempted=phase == "MODEL_DOWNLOAD", candidate_preparation_started=False,
            primary_run_status="NOT_RUN", replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(root / "replacement-loss-v27-terminal-report.json")}
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": "TARGETS_ACQUIRED_READY_FOR_PRIMARY", "checkpoint_count": 3, "downloaded_bytes": downloads["total_downloaded_bytes"], "selection_binding_sha256": _sha(root / "v27-selection-binding.json")}


def _aggregate(root: Path, packet_sha: str, runner_receipt: dict[str, Any]) -> dict[str, Any]:
    upstream = _read_json(root / "replacement-loss-v22-report.json")
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE, "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION, "packet_sha256": packet_sha,
        "upstream_v25_packet_sha256": v26._verify_v25_static()["packet_sha256"],
        "upstream_v26_packet_sha256": V26_PACKET_SHA256,
        "upstream_v22_packet_sha256": v23.V22_PACKET_SHA256,
        "upstream_v8_packet_sha256": v22.V8_PACKET_SHA256,
        "selection_binding_sha256": _sha(root / "v27-selection-binding.json"),
        "download_manifest_sha256": _sha(root / "v27-download-manifest.json"),
        "outcome_status": upstream["outcome_status"], "assurance_level": upstream["assurance_level"],
        "behavior_and_mechanism_diverged": "NOT_SEPARATELY_DEFINED_BY_FROZEN_V8_METHOD; behavior, causal effects, and activation rescue are separate metrics",
        "reviewer_status": "NOT_SUBMITTED", "reproducibility_status": "PENDING_SECOND_CLEAN_RUNTIME",
        "case_count": upstream["case_count"], "cases": upstream["cases"],
        "runtime": upstream["runtime"],
        "runtime_pair": _read_json(root / "v27-selection-binding.json")["runtime_pair"],
        "budget_and_resource_measurements": upstream["budget_and_resource_measurements"],
        "data_and_artifact_custody": f"models, prompts, captures, candidates, predictions, outcomes, and diagnostics remain under external owner custody {root}",
        "known_limitations": [
            "three fresh identities in three named architecture families",
            "one MLP output projection per checkpoint", "synthetic prompts",
            "same-owner local reproduction", "bounded public Hub queries",
            "frozen method has no formal behavior-versus-mechanism divergence statistic",
            "monetary cost is zero; energy is not measured",
        ],
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "external independent scientific review; not submitted",
        "primary_runner_receipt": runner_receipt,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(root / "replacement-loss-v27-report.json", report)
    return report


def run_primary(custody_root: str | Path = V27_ROOT) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    runtime_pair = _verify_runtime_pair(root)
    binding = _read_json(root / "v27-selection-binding.json")
    if binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V27 selection binding does not match frozen packet")
    one_shot = (root / "replacement-loss-v27-report.json", root / "replacement-loss-v22-report.json", root / "candidate-preparation.json", root / "candidate-freeze.json", root / "challenge-manifest.json")
    if any(path.exists() for path in one_shot):
        raise FileExistsError("V27 primary evaluation is one-shot")
    prior_ids, _rows = _prior_ledger(root)
    try:
        with v23._v22_external_exclusions(root, prior_ids):
            result = v22.run_primary(root, V7_ROOT)
        report = _aggregate(root, packet["packet_sha256"], result)
    except Exception as exc:
        terminal = _write_stop(
            root, packet["packet_sha256"], outcome_status="INCONCLUSIVE_EXECUTION",
            phase="PRIMARY_EVALUATION", error=exc, runtime_pair=runtime_pair,
            model_weights_downloaded=True,
            candidate_preparation_started=(root / "candidate-preparation.json").is_file(),
            candidate_freeze_sha256=_sha(root / "candidate-freeze.json") if (root / "candidate-freeze.json").is_file() else None,
            challenge_manifest_sha256=_sha(root / "challenge-manifest.json") if (root / "challenge-manifest.json").is_file() else None,
            primary_run_status="INCONCLUSIVE", replication_status="NOT_RUN",
        )
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(root / "replacement-loss-v27-terminal-report.json")}
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": report["outcome_status"], "assurance_level": report["assurance_level"], "case_count": report["case_count"], "report_sha256": _sha(root / "replacement-loss-v27-report.json")}


def replicate_packet(primary_custody_root: str | Path = V27_ROOT, replica_custody_root: str | Path = V27_REPLICA_ROOT) -> dict[str, Any]:
    primary, replica = _custody_root(primary_custody_root), _replica_root(replica_custody_root)
    packet = verify_packet(primary)
    runtime_pair = _verify_runtime_pair(primary)
    report_path = primary / "replacement-loss-v27-report.json"
    report_before = _read_json(report_path)
    binding = _read_json(primary / "v27-selection-binding.json")
    if report_before.get("packet_sha256") != packet["packet_sha256"] or binding.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V27 report or selection binding does not match frozen packet")
    if binding.get("v22_checkpoint_manifest_sha256") != _sha(primary / "checkpoints.json"):
        raise ValueError("V27 selected checkpoint manifest changed")
    if report_before.get("outcome_status") not in {"PASS", "FAIL"}:
        raise ValueError("V27 reproduction requires a complete primary PASS or FAIL")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("V27 replica custody is one-shot and must be empty")
    replica.mkdir(parents=True, exist_ok=True)
    prior_ids, _rows = _prior_ledger(primary)
    try:
        with v23._v22_external_exclusions(primary, prior_ids):
            result = v22.replicate_packet(primary, replica, V7_ROOT)
        upstream = _read_json(replica / "replacement-loss-v22-replication-report.json")
    except Exception as exc:
        terminal = _write_stop(replica, packet["packet_sha256"], outcome_status="INCONCLUSIVE_REPRODUCTION", phase="CLEAN_REPLICA_EVALUATION", error=exc, runtime_pair=runtime_pair, primary_v27_report_sha256=_sha(report_path), primary_run_status=report_before.get("outcome_status"), replication_status="INCONCLUSIVE")
        return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "outcome_status": terminal["outcome_status"], "terminal_report_sha256": _sha(replica / "replacement-loss-v27-terminal-report.json")}
    all_checks = all(bool(upstream.get(key)) for key in (
        "candidate_reconstruction_match", "prediction_and_outcome_array_digests_match",
        "dependencies_match", "distinct_clean_runtime", "source_identity_match",
    )) and runtime_pair["distinct_sys_prefix_and_site_packages"]
    assurance = "REPRODUCED_LOCAL" if result.get("assurance_level") == "REPRODUCED_LOCAL" and all_checks else "INCONCLUSIVE"
    final = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE, "assurance_slice": ASSURANCE_SLICE,
        "source_revision": SOURCE_REVISION, "packet_sha256": packet["packet_sha256"],
        "outcome_status": report_before["outcome_status"], "assurance_level": assurance,
        "reproducibility_status": assurance, "reviewer_status": "NOT_SUBMITTED",
        "same_owner_second_environment": True, "independent_science_or_acceptance": False,
        "candidate_reconstruction_match": bool(upstream.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(upstream.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(upstream.get("dependencies_match")),
        "distinct_clean_runtime": bool(upstream.get("distinct_clean_runtime")) and runtime_pair["distinct_sys_prefix_and_site_packages"],
        "source_identity_match": bool(upstream.get("source_identity_match")), "runtime_pair": runtime_pair,
        "data_and_artifact_custody": f"primary and clean-replica evidence remains under {primary} and {replica}",
        "claim_ceiling": "FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES",
        "nonclaims": ["unseen architecture families", "all architectures", "independent science", "consciousness", "alignment", "production readiness"],
        "upstream_v22_replication_report_sha256": _sha(replica / "replacement-loss-v22-replication-report.json"),
        "primary_v27_report_sha256_before_reconciliation": _sha(report_path),
        "next_blocked_authority_gate": "external independent scientific review; not submitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    replication_path = replica / "replacement-loss-v27-replication-report.json"
    _write_json(replication_path, final)
    report_before["assurance_level"] = assurance
    report_before["reproducibility_status"] = assurance
    report_before["replication_report_sha256"] = _sha(replication_path)
    _write_json(report_path, report_before)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "assurance_level": assurance, "outcome_status": final["outcome_status"], "replication_report_sha256": _sha(replication_path)}


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    verify = commands.add_parser("verify")
    verify.add_argument("--custody-root", default=str(V27_ROOT))
    runtime = commands.add_parser("runtime-check")
    runtime.add_argument("--role", choices=("primary", "replicator"), required=True)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--custody-root", default=str(V27_ROOT))
    acquire.add_argument("--allow-public-hub-downloads", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--custody-root", default=str(V27_ROOT))
    replicate = commands.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", default=str(V27_ROOT))
    replicate.add_argument("--replica-custody-root", default=str(V27_REPLICA_ROOT))
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet()
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
