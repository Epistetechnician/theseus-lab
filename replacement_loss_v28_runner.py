"""Frozen V28 local lifecycle for fresh-checkpoint replacement-loss recovery.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v28-fresh-checkpoint-recovery-v1, and
global-experiment-assurance-lifecycle-v1. Prompt and model data remain in
external owner custody. The frozen V27 evaluator/orchestration is reused
through explicit V28 bindings; its internal compatibility filenames are
preserved in the external V28 receipt tree.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import site
import subprocess
import sys
import sysconfig
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import replacement_loss_v27_runner as core
import replacement_loss_v28_fixtures as fixtures
import replacement_loss_v28_hub_adapter as hub_adapter


ARTIFACTS_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
UPSTREAM_V27_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v27"
V28_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v28"
V28_REPLICA_ROOT = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v28-replicator"
V28_PRIMARY_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v28-primary-toolchain"
V28_REPLICA_RUNTIME = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v28-replicator-toolchain"
FIXTURE_PREFLIGHT_PATH = ARTIFACTS_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v28-fixture-preflight.json"
CONTINUATION_SLICE = "theseus-replacement-loss-v28-fresh-checkpoint-recovery-v1"
PROTOCOL_ID = "theseus-replacement-loss-v28-fresh-checkpoint-recovery"
STATE_SLICE = "theseus-lab-sandbox-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
SOURCE_REVISION = "202f459d6d89df9a038d78d58e0af825d287fcd8"
FREEZE_PATH = "docs/replacement-loss-v28-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v28-fresh-checkpoint-recovery.md",
    "replacement_loss_v28_fixtures.py",
    "replacement_loss_v28_hub_adapter.py",
    "replacement_loss_v28_runner.py",
    "requirements-v28.lock.txt",
)
UPSTREAM_CODE_FILES = tuple(dict.fromkeys((
    *core.PACKET_FILES,
    *core.UPSTREAM_CODE_FILES,
    "requirements-v27.lock.txt",
)))
V27_TERMINAL_NAME = "replacement-loss-v27-terminal-report.json"
V27_PACKET_SHA256 = "3321cc188a83f8ae5a394f49747309c02c7ce0f5e8caf31cb7a21f12ab9dbc5b"


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


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_rows(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in paths:
        path = _root() / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen packet input missing: {relative}")
        rows.append({"path": relative, "sha256": _sha(path), "bytes": path.stat().st_size})
    return rows


def _assert_external(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if resolved == _root() or _root() in resolved.parents:
        raise ValueError("V28 artifacts must remain outside the repository")
    return resolved


def _configure_core() -> None:
    """Bind the frozen V27 orchestration to this V28 packet and custody roots."""
    core.V27_ROOT = V28_ROOT
    core.V27_REPLICA_ROOT = V28_REPLICA_ROOT
    core.V27_PRIMARY_RUNTIME = V28_PRIMARY_RUNTIME
    core.V27_REPLICA_RUNTIME = V28_REPLICA_RUNTIME
    core.CONTINUATION_SLICE = CONTINUATION_SLICE
    core.PROTOCOL_ID = PROTOCOL_ID
    core.SOURCE_REVISION = SOURCE_REVISION
    core.FREEZE_PATH = FREEZE_PATH
    core.PACKET_FILES = PACKET_FILES
    core.UPSTREAM_CODE_FILES = UPSTREAM_CODE_FILES
    core.hub_adapter = hub_adapter
    core.v22.fixtures = fixtures
    core.verify_packet = verify_packet
    os.environ["HF_HOME"] = str(V28_ROOT / "hf-home")
    os.environ["HF_HUB_CACHE"] = str(V28_ROOT / "hub-cache")
    os.environ["HF_ASSETS_CACHE"] = str(V28_ROOT / "assets-cache")
    os.environ["HF_XET_CACHE"] = str(V28_ROOT / "xet-cache")
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


def _verify_v27_frozen_evidence() -> dict[str, Any]:
    """Verify V27 bytes and its stop without recomputing its stale global ledger."""
    local = _read_json(_root() / "docs/replacement-loss-v27-freeze.json")
    external = _read_json(UPSTREAM_V27_ROOT / "v27-frozen-packet-manifest.json")
    if local != external:
        raise ValueError("V27 local and external packet manifests differ")
    expected = _digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if expected != V27_PACKET_SHA256 or local.get("packet_sha256") != expected:
        raise ValueError("V27 packet self-digest differs")
    for group, archive_prefix in (("packet_files", "v27-frozen-packet"), ("upstream_code_dependencies", "v27-frozen-packet/upstream-source")):
        for row in local.get(group, []):
            relative = str(row["path"])
            if _sha(_root() / relative) != row["sha256"]:
                raise ValueError(f"V27 current source differs: {relative}")
            if _sha(UPSTREAM_V27_ROOT / archive_prefix / relative) != row["sha256"]:
                raise ValueError(f"V27 archived source differs: {relative}")
    if core._v26_packet_static() != local.get("upstream_v26_static_evidence"):
        raise ValueError("V27 static V26 evidence binding differs")

    terminal_path = UPSTREAM_V27_ROOT / V27_TERMINAL_NAME
    terminal = _read_json(terminal_path)
    terminal_body = {key: value for key, value in terminal.items() if key != "terminal_report_sha256"}
    if terminal.get("terminal_report_sha256") != _digest(terminal_body):
        raise ValueError("V27 stop receipt self-digest differs")
    if (terminal.get("outcome_status"), terminal.get("phase"), terminal.get("candidate_preparation_started")) != (
        "INCONCLUSIVE_EXECUTION", "PRIMARY_EVALUATION", False
    ):
        raise ValueError("V27 terminal disposition is not the recorded pre-candidate stop")
    forbidden = ("candidate-preparation.json", "candidate-freeze.json", "challenge-manifest.json", "replacement-loss-v27-report.json")
    if any((UPSTREAM_V27_ROOT / name).exists() for name in forbidden):
        raise ValueError("V27 contains unexpected post-stop candidate or evaluation artifacts")
    selection = _read_json(UPSTREAM_V27_ROOT / "v27-selection-binding.json")
    if selection.get("packet_sha256") != expected:
        raise ValueError("V27 checkpoint selection is not bound to its frozen packet")
    if selection.get("hub_screen_sha256") != _sha(UPSTREAM_V27_ROOT / "v27-hub-screen.json"):
        raise ValueError("V27 screen digest differs from the selection binding")
    if selection.get("v22_checkpoint_manifest_sha256") != _sha(UPSTREAM_V27_ROOT / "checkpoints.json"):
        raise ValueError("V27 checkpoint manifest digest differs from the selection binding")
    return {
        "packet_sha256": expected,
        "terminal_report_sha256": _sha(terminal_path),
        "terminal_status": terminal["outcome_status"],
        "terminal_phase": terminal["phase"],
        "candidate_preparation_started": False,
        "selected_identity_artifacts_sha256": {
            name: _sha(UPSTREAM_V27_ROOT / name)
            for name in ("v27-hub-screen.json", "v27-checkpoint-selection.json", "v27-download-manifest.json", "checkpoints.json")
        },
    }


def _fixture_payloads() -> list[dict[str, Any]]:
    visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible_selection")
    adaptation = [
        fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation_{fixtures.SEEDS[index]}")
        for index in range(len(fixtures.SEEDS))
    ]
    fixtures.validate_partition_disjointness(visible, *adaptation)
    for payload in adaptation:
        fixtures.assert_stratum_counts(payload, fixtures.ADAPTATION_PROMPTS)
    return [visible, *adaptation]


def preflight_fixtures() -> dict[str, Any]:
    """Check V28 fixtures against all previous custody before any model search."""
    if FIXTURE_PREFLIGHT_PATH.exists():
        raise FileExistsError("V28 fixture preflight is one-shot; preserve it and start a new protocol")
    prior_roots = core.v22._prior_fixture_roots()
    payloads = _fixture_payloads()
    import replacement_loss_v6_runner as v6

    receipt = v6._validate_all_disjoint(prior_roots, payloads)
    body = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "checked_before_freeze_and_target_selection": True,
        "fixture_rule_sha256": _sha(_root() / "replacement_loss_v28_fixtures.py"),
        "prior_fixture_root_count": len(prior_roots),
        "prior_fixture_count": receipt["prior_fixture_count"],
        "prior_fixture_digests_sha256": receipt["prior_fixture_digests_sha256"],
        "current_fixture_counts": [len(item["records"]) for item in payloads],
        "current_fixture_digests": receipt["current_fixture_digests"],
        "all_partitions_disjoint": receipt["all_partitions_disjoint"],
        "prompt_or_continuation_text_in_receipt": False,
        "created_at_utc": _now(),
    }
    body["receipt_sha256"] = _digest(body)
    _write_json(FIXTURE_PREFLIGHT_PATH, body)
    return {
        "protocol_id": PROTOCOL_ID,
        "receipt_sha256": body["receipt_sha256"],
        "prior_fixture_count": body["prior_fixture_count"],
        "current_fixture_count": sum(body["current_fixture_counts"]),
        "all_partitions_disjoint": True,
        "prompt_or_continuation_text_in_receipt": False,
    }


def _copy_file(source: Path, target: Path) -> None:
    if target.exists():
        if not target.is_file() or _sha(target) != _sha(source):
            raise ValueError(f"existing V28 frozen file differs: {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if target.exists():
        actual = {p.relative_to(target).as_posix(): _sha(p) for p in target.rglob("*") if p.is_file()}
        expected = {p.relative_to(source).as_posix(): _sha(p) for p in source.rglob("*") if p.is_file()}
        if actual != expected:
            raise ValueError(f"existing V28 frozen tree differs: {target.name}")
        return
    shutil.copytree(source, target)


def freeze_packet() -> dict[str, Any]:
    """Freeze exact V28 bytes only after successful fixture-disjointness preflight."""
    _assert_external(V28_ROOT)
    if (_root() / FREEZE_PATH).exists() or V28_ROOT.exists():
        raise FileExistsError("V28 freeze is one-shot; preserve any existing packet")
    fixture_receipt = _read_json(FIXTURE_PREFLIGHT_PATH)
    receipt_body = {key: value for key, value in fixture_receipt.items() if key != "receipt_sha256"}
    if fixture_receipt.get("receipt_sha256") != _digest(receipt_body) or fixture_receipt.get("all_partitions_disjoint") is not True:
        raise ValueError("fixture-disjointness receipt is missing, invalid, or failed")
    v27 = _verify_v27_frozen_evidence()
    if _sha(_root() / "requirements-v28.lock.txt") != _sha(_root() / "requirements-v27.lock.txt"):
        raise ValueError("V28 dependency inventory must remain identical to the qualified V27 inventory")
    packet_rows = _file_rows(PACKET_FILES)
    upstream_rows = _file_rows(UPSTREAM_CODE_FILES)
    prior_ids, prior_rows = core._prior_ledger(V28_ROOT)
    if len(prior_ids) <= 11_927:
        raise ValueError("fresh checkpoint exclusion ledger did not advance beyond V27")

    V28_ROOT.mkdir(parents=True, exist_ok=False)
    _copy_file(FIXTURE_PREFLIGHT_PATH, V28_ROOT / "v28-fixture-preflight.json")
    # These two frozen archives are the exact upstream method inputs required
    # by V22/V8. No V27 model weights or prompt data are copied.
    for name in ("v8-frozen-packet-manifest.json", "v22-frozen-packet-manifest.json"):
        _copy_file(UPSTREAM_V27_ROOT / name, V28_ROOT / name)
    for name in ("v8-frozen-packet", "v22-frozen-packet"):
        _copy_tree(UPSTREAM_V27_ROOT / name, V28_ROOT / name)
    for row in packet_rows:
        relative = str(row["path"])
        _copy_file(_root() / relative, V28_ROOT / "v28-frozen-packet" / relative)
    for row in upstream_rows:
        relative = str(row["path"])
        _copy_file(_root() / relative, V28_ROOT / "v28-frozen-packet" / "upstream-source" / relative)

    try:
        source_revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_root(), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = SOURCE_REVISION
    lock = [line.strip() for line in (_root() / "requirements-v28.lock.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "assurance_slice": ASSURANCE_SLICE,
        "source_revision": source_revision,
        "source_identity_note": "git HEAD plus exact packet and upstream source digests; checkout contains unrelated pre-existing dirty files",
        "created_at_utc": _now(),
        "freeze_state": "FROZEN_AFTER_FIXTURE_DISJOINTNESS_AND_BEFORE_V28_HUB_QUERY_OR_TARGET_SELECTION",
        "packet_files": packet_rows,
        "upstream_code_dependencies": upstream_rows,
        "fixture_preflight_receipt_sha256": _sha(FIXTURE_PREFLIGHT_PATH),
        "fixture_preflight_receipt_body_sha256": fixture_receipt["receipt_sha256"],
        "fixture_preflight_all_partitions_disjoint": True,
        "pre_v28_excluded_identity_count": len(prior_ids),
        "pre_v28_identity_ledger_sha256": _digest(prior_rows),
        "upstream_v27_packet_sha256": v27["packet_sha256"],
        "upstream_v27_terminal_report_sha256": v27["terminal_report_sha256"],
        "upstream_v27_static_evidence": v27,
        "upstream_v8_packet_sha256": core.v22.V8_PACKET_SHA256,
        "upstream_v22_packet_sha256": core.v23.V22_PACKET_SHA256,
        "checkpoint_identities_selected": False,
        "hub_queries_issued": False,
        "model_weights_downloaded": False,
        "dependency_manifest_sha256": _sha(_root() / "requirements-v28.lock.txt"),
        "dependency_inventory": {"distribution_count": len(lock), "packages": lock, "primary_and_clean_replica_must_match": True},
        "runtime": {
            "python": "3.12.13",
            "platform": "macOS arm64 local CPU only",
            "primary_runtime": str(V28_PRIMARY_RUNTIME),
            "clean_replica_runtime": str(V28_REPLICA_RUNTIME),
            "isolation_contract": "distinct sys.prefix and site-packages; include-system-site-packages=false; shared base executable permitted",
            "hub_client": "huggingface_hub==0.36.2",
            "protobuf": "protobuf==5.29.6",
            "auth": "token=False",
            "telemetry": False,
            "hosted_inference": False,
            "remote_workers": 0,
        },
        "custody_roots": {"primary": str(V28_ROOT), "clean_replica": str(V28_REPLICA_ROOT)},
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
        "nonclaims": ["unseen architecture families", "all architectures", "independently learned replacement", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
    }
    payload["packet_sha256"] = _digest(payload)
    _write_json(V28_ROOT / "v28-frozen-packet-manifest.json", payload)
    # V27 core's runtime/acquisition functions retain their immutable internal
    # filename contract; this byte-identical alias contains the V28 manifest.
    _write_json(V28_ROOT / "v27-frozen-packet-manifest.json", payload)
    _write_json(_root() / FREEZE_PATH, payload)
    _configure_core()
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "continuation_slice": CONTINUATION_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "pre_v28_excluded_identity_count": len(prior_ids),
        "fixture_preflight_receipt_sha256": _sha(FIXTURE_PREFLIGHT_PATH),
        "checkpoint_identities_selected": False,
        "freeze_state": payload["freeze_state"],
    }


def verify_packet(custody_root: str | Path = V28_ROOT) -> dict[str, Any]:
    root = _assert_external(custody_root)
    if root != V28_ROOT.resolve():
        raise ValueError("V28 primary custody root is pinned")
    local = _read_json(_root() / FREEZE_PATH)
    external = _read_json(root / "v28-frozen-packet-manifest.json")
    alias = _read_json(root / "v27-frozen-packet-manifest.json")
    if local != external or alias != external:
        raise ValueError("V28 local, external, and compatibility manifests differ")
    packet_sha = _digest({key: value for key, value in local.items() if key != "packet_sha256"})
    if local.get("packet_sha256") != packet_sha or local.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V28 packet identity or self-digest differs")
    for group, archive_prefix in (("packet_files", "v28-frozen-packet"), ("upstream_code_dependencies", "v28-frozen-packet/upstream-source")):
        for row in local[group]:
            relative = str(row["path"])
            if _sha(_root() / relative) != row["sha256"] or _sha(root / archive_prefix / relative) != row["sha256"]:
                raise ValueError(f"V28 frozen source differs: {relative}")
    fixture_path = root / "v28-fixture-preflight.json"
    if _sha(fixture_path) != local.get("fixture_preflight_receipt_sha256"):
        raise ValueError("V28 fixture preflight receipt changed")
    fixture_receipt = _read_json(fixture_path)
    fixture_body = {key: value for key, value in fixture_receipt.items() if key != "receipt_sha256"}
    if fixture_receipt.get("receipt_sha256") != _digest(fixture_body) or fixture_receipt.get("all_partitions_disjoint") is not True:
        raise ValueError("V28 fixture preflight receipt failed verification")
    v27 = _verify_v27_frozen_evidence()
    if v27 != local.get("upstream_v27_static_evidence"):
        raise ValueError("V27 frozen evidence binding differs")
    prior_ids, prior_rows = core._prior_ledger(root)
    if len(prior_ids) != local.get("pre_v28_excluded_identity_count") or _digest(prior_rows) != local.get("pre_v28_identity_ledger_sha256"):
        raise ValueError("V28 preselection identity ledger changed")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet_sha,
        "pre_v28_excluded_identity_count": len(prior_ids),
        "fixture_preflight_receipt_sha256": _sha(fixture_path),
        "upstream_v27_packet_sha256": v27["packet_sha256"],
        "packet_file_count": len(local["packet_files"]),
        "verified": True,
    }


def prepare_runtimes() -> dict[str, Any]:
    """Create two pinned isolated venvs and install the frozen lock once each."""
    verify_packet(V28_ROOT)
    for runtime in (V28_PRIMARY_RUNTIME, V28_REPLICA_RUNTIME):
        if runtime.exists():
            raise FileExistsError(f"V28 runtime already exists; do not repair it in place: {runtime.name}")
    v27_runtime_receipt = _read_json(UPSTREAM_V27_ROOT / "v27-primary-runtime-check.json")
    if v27_runtime_receipt.get("verified") is not True or v27_runtime_receipt.get("python") != "3.12.13":
        raise ValueError("the qualified V27 base Python receipt is unavailable")
    base_python = str(v27_runtime_receipt["sys_executable_realpath"])
    if not Path(base_python).is_file():
        raise FileNotFoundError("the qualified Python 3.12.13 base executable is unavailable")
    install_log = V28_ROOT / "v28-runtime-install.log"
    env = dict(os.environ)
    env.update({"PIP_CONFIG_FILE": os.devnull, "PIP_CACHE_DIR": str(V28_ROOT / "pip-cache"), "HF_HUB_DISABLE_TELEMETRY": "1"})
    with install_log.open("w", encoding="utf-8") as log:
        for runtime in (V28_PRIMARY_RUNTIME, V28_REPLICA_RUNTIME):
            subprocess.run([base_python, "-m", "venv", str(runtime)], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            python = runtime / "bin/python"
            subprocess.run([str(python), "-m", "pip", "install", "--isolated", "--no-input", "--disable-pip-version-check", "--index-url", "https://pypi.org/simple", "pip==25.0.1"], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
            subprocess.run([str(python), "-m", "pip", "install", "--isolated", "--no-input", "--disable-pip-version-check", "--index-url", "https://pypi.org/simple", "-r", str(_root() / "requirements-v28.lock.txt")], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": _read_json(V28_ROOT / "v28-frozen-packet-manifest.json")["packet_sha256"], "runtime_install_log_sha256": _sha(install_log), "runtimes_created": 2}


def check_runtime(role: str) -> dict[str, Any]:
    _configure_core()
    receipt = core.check_runtime(role)
    source = V28_ROOT / f"v27-{role}-runtime-check.json"
    alias = V28_ROOT / f"v28-{role}-runtime-check.json"
    _copy_file(source, alias)
    return {"protocol_id": PROTOCOL_ID, "role": role, "verified": receipt["verified"], "package_inventory_sha256": receipt["package_inventory_sha256"], "runtime_receipt_sha256": _sha(alias)}


def acquire(*, allow_public_hub_downloads: bool) -> dict[str, Any]:
    _configure_core()
    verify_packet(V28_ROOT)
    result = core.acquire_checkpoints(V28_ROOT, allow_public_hub_downloads=allow_public_hub_downloads)
    for source, target in (
        ("v27-hub-screen-progress.json", "v28-hub-screen-progress.json"),
        ("v27-hub-screen.json", "v28-hub-screen.json"),
        ("v27-checkpoint-selection.json", "v28-checkpoint-selection.json"),
        ("v27-download-manifest.json", "v28-download-manifest.json"),
        ("v27-selection-binding.json", "v28-selection-binding.json"),
        ("replacement-loss-v27-terminal-report.json", "replacement-loss-v28-terminal-report.json"),
        ("checkpoints.json", "v28-checkpoints.json"),
    ):
        source_path = V28_ROOT / source
        if source_path.is_file():
            _copy_file(source_path, V28_ROOT / target)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": result.get("packet_sha256"), "outcome_status": result.get("outcome_status"), "checkpoint_count": result.get("checkpoint_count"), "downloaded_bytes": result.get("downloaded_bytes"), "terminal_report_sha256": result.get("terminal_report_sha256"), "selection_binding_sha256": result.get("selection_binding_sha256")}


def run_primary() -> dict[str, Any]:
    _configure_core()
    verify_packet(V28_ROOT)
    result = core.run_primary(V28_ROOT)
    for source, target in (
        ("replacement-loss-v27-report.json", "replacement-loss-v28-report.json"),
        ("replacement-loss-v27-terminal-report.json", "replacement-loss-v28-terminal-report.json"),
    ):
        source_path = V28_ROOT / source
        if source_path.is_file():
            _copy_file(source_path, V28_ROOT / target)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": result.get("packet_sha256"), "outcome_status": result.get("outcome_status"), "assurance_level": result.get("assurance_level"), "case_count": result.get("case_count"), "report_sha256": result.get("report_sha256"), "terminal_report_sha256": result.get("terminal_report_sha256")}


def replicate() -> dict[str, Any]:
    _configure_core()
    verify_packet(V28_ROOT)
    result = core.replicate_packet(V28_ROOT, V28_REPLICA_ROOT)
    for source, target in (
        ("replacement-loss-v27-replication-report.json", "replacement-loss-v28-replication-report.json"),
        ("replacement-loss-v27-terminal-report.json", "replacement-loss-v28-terminal-report.json"),
        ("replacement-loss-v27-report.json", "replacement-loss-v28-report.json"),
    ):
        source_path = V28_REPLICA_ROOT / source if "replication" in source or "terminal" in source else V28_ROOT / source
        target_root = V28_REPLICA_ROOT if "replication" in source or "terminal" in source else V28_ROOT
        if source_path.is_file():
            _copy_file(source_path, target_root / target)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": result.get("packet_sha256"), "outcome_status": result.get("outcome_status"), "assurance_level": result.get("assurance_level"), "replication_report_sha256": result.get("replication_report_sha256"), "terminal_report_sha256": result.get("terminal_report_sha256")}


def status() -> dict[str, Any]:
    result: dict[str, Any] = {"protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "continuation_slice": CONTINUATION_SLICE, "external_custody_exists": V28_ROOT.is_dir()}
    if V28_ROOT.is_dir():
        for name in ("v28-frozen-packet-manifest.json", "v28-hub-screen.json", "v28-checkpoint-selection.json", "v28-download-manifest.json", "replacement-loss-v28-report.json", "replacement-loss-v28-replication-report.json", "replacement-loss-v28-terminal-report.json"):
            path = V28_ROOT / name
            if path.is_file():
                row: dict[str, Any] = {"sha256": _sha(path)}
                try:
                    value = _read_json(path)
                    for key in ("outcome_status", "assurance_level", "phase", "candidate_preparation_started"):
                        if key in value:
                            row[key] = value[key]
                    if "cases" in value:
                        row["case_count"] = len(value["cases"])
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                result[name] = row
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fixture-preflight")
    commands.add_parser("freeze")
    commands.add_parser("verify")
    commands.add_parser("prepare-runtimes")
    runtime = commands.add_parser("runtime-check")
    runtime.add_argument("--role", choices=("primary", "replicator"), required=True)
    acquire_parser = commands.add_parser("acquire")
    acquire_parser.add_argument("--allow-public-hub-downloads", action="store_true")
    commands.add_parser("run-primary")
    commands.add_parser("replicate")
    commands.add_parser("status")
    args = parser.parse_args()
    try:
        if args.command == "fixture-preflight":
            result = preflight_fixtures()
        elif args.command == "freeze":
            result = freeze_packet()
        elif args.command == "verify":
            _configure_core()
            result = verify_packet()
        elif args.command == "prepare-runtimes":
            result = prepare_runtimes()
        elif args.command == "runtime-check":
            result = check_runtime(args.role)
        elif args.command == "acquire":
            result = acquire(allow_public_hub_downloads=args.allow_public_hub_downloads)
        elif args.command == "run-primary":
            result = run_primary()
        elif args.command == "replicate":
            result = replicate()
        else:
            result = status()
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception as exc:
        # The custody report may contain diagnostics; stdout intentionally
        # emits only a class label so no prompt, completion, or model ID leaks.
        print(json.dumps({"protocol_id": PROTOCOL_ID, "command": args.command, "error_type": type(exc).__name__, "status": "INCONCLUSIVE_SETUP_OR_EXECUTION"}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
