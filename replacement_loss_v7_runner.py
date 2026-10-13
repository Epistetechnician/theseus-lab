"""Frozen-packet lifecycle for replacement-loss-v7.

State slice: theseus-lab-sandbox-v1. Model and prompt data stay in external
custody. The runner reuses pinned v6 orchestration through a narrow adapter.
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

import replacement_loss_v3_runner as v3
import replacement_loss_v6_runner as v6_core
import replacement_loss_v7 as v7
import replacement_loss_v7_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = v7.PROTOCOL_ID
V3_PACKET_SHA256 = "d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83"
V6_PROTOCOL_ID = "theseus-replacement-loss-v6-downstream-aware"
V6_PACKET_SHA256 = "9b7b669195bb4fb9153e872d42f3cb4fd235e9784fdc978ec936037eb176bc5a"
V6_CUSTODY_ROOT = Path(v6_core.DEFAULT_CUSTODY)
DEFAULT_CUSTODY = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v7")
DEFAULT_REPLICA = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v7-replicator")
V7_PACKET_FILES = (
    "docs/replacement-loss-v7.md",
    "replacement_loss_v7.py",
    "replacement_loss_v7_fixtures.py",
    "replacement_loss_v7_runner.py",
    "tests/test_replacement_loss_v7.py",
    "requirements.txt",
    *v6_core.V6_PACKET_FILES,
    "docs/replacement-loss-v6-freeze.json",
    *v3.PACKET_FILES,
    v3.PACKET_FREEZE,
)
PRIOR_CHECKPOINT_IDS = set(v6_core.PRIOR_CHECKPOINT_IDS) | {
    "facebook/opt-350m",
    "openai-community/gpt2-medium",
    "huggingfacetb/smollm-360m",
    "babylm/opt-125m-strict-2023",
    "babylm-community/babylm-baseline-100m-gpt2",
    "babylm/babyllama-100m-2024",
}
_ORIGINAL_BUILD_V6_REPORT = v6_core._build_v6_report
_CORE_PATCHES = (
    "PROTOCOL_ID",
    "V6_PACKET_FILES",
    "DEFAULT_CUSTODY",
    "DEFAULT_REPLICA",
    "PRIOR_CHECKPOINT_IDS",
    "v6",
    "fixtures",
    "verify_packet",
    "_validate_checkpoint_manifest",
    "write_checkpoint_manifest",
    "_add_v6_binding_to_challenge_manifest",
    "_build_v6_report",
)


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
        raise ValueError(f"expected a JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _custody_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("v7 custody must be outside the repository")
    return root


def _verify_pinned_v6_source_and_archive() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    local_freeze = _read_json(_root() / "docs" / "replacement-loss-v6-freeze.json")
    archive_manifest_path = V6_CUSTODY_ROOT / "v6-frozen-packet-manifest.json"
    archive_manifest = _read_json(archive_manifest_path)
    if local_freeze != archive_manifest:
        raise ValueError("v6 local and custodied freeze manifests differ")
    if local_freeze.get("protocol_id") != V6_PROTOCOL_ID:
        raise ValueError("pinned v6 protocol identity mismatch")
    if local_freeze.get("packet_sha256") != V6_PACKET_SHA256:
        raise ValueError("pinned v6 packet identity changed")
    if local_freeze.get("packet_sha256") != _canonical_digest(
        {key: value for key, value in local_freeze.items() if key != "packet_sha256"}
    ):
        raise ValueError("pinned v6 packet manifest digest is invalid")
    rows = list(local_freeze["packet_files"])
    for row in rows:
        relative = str(row["path"])
        source = _root() / relative
        archived = V6_CUSTODY_ROOT / "v6-frozen-packet" / relative
        if _sha(source) != row["sha256"] or _sha(archived) != row["sha256"]:
            raise ValueError(f"pinned v6 source or archive changed: {relative}")
    freeze_path = _root() / "docs" / "replacement-loss-v6-freeze.json"
    rows.append({
        "path": "docs/replacement-loss-v6-freeze.json",
        "sha256": _sha(freeze_path),
        "bytes": freeze_path.stat().st_size,
    })
    return local_freeze, rows


def _verify_pinned_v3_source() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verified = v3.verify_packet()
    if verified.get("packet_sha256") != V3_PACKET_SHA256:
        raise ValueError("pinned v3 packet identity changed")
    freeze = _read_json(_root() / v3.PACKET_FREEZE)
    if freeze.get("packet_sha256") != V3_PACKET_SHA256:
        raise ValueError("pinned v3 freeze manifest identity changed")
    rows = list(freeze["packet_files"])
    for row in rows:
        if _sha(_root() / str(row["path"])) != row["sha256"]:
            raise ValueError(f"pinned v3 source changed: {row['path']}")
    freeze_path = _root() / v3.PACKET_FREEZE
    rows.append({"path": v3.PACKET_FREEZE, "sha256": _sha(freeze_path), "bytes": freeze_path.stat().st_size})
    return freeze, rows


def freeze_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    """Freeze v7 and exact v3/v6 dependencies before model selection."""

    root = _custody_root(custody_root)
    local_freeze = _root() / "docs" / "replacement-loss-v7-freeze.json"
    archive_manifest = root / "v7-frozen-packet-manifest.json"
    archive = root / "v7-frozen-packet"
    if local_freeze.exists() or archive_manifest.exists() or archive.exists():
        raise FileExistsError("v7 packet freeze is immutable; continue with a new version")
    v6_freeze, _v6_rows = _verify_pinned_v6_source_and_archive()
    v3_freeze, _v3_rows = _verify_pinned_v3_source()
    files = tuple(dict.fromkeys(V7_PACKET_FILES))
    missing = [relative for relative in files if not (_root() / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v7 packet inputs are missing: {', '.join(missing)}")
    rows = []
    for relative in files:
        source = _root() / relative
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "sha256": _sha(source), "bytes": source.stat().st_size})
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V7_CHECKPOINT_SELECTION",
        "created_at_utc": _now(),
        "packet_files": rows,
        "v3_packet_sha256": v3_freeze["packet_sha256"],
        "v6_packet_sha256": v6_freeze["packet_sha256"],
        "checkpoint_identities_selected": False,
        "challenge_fixtures_generated": False,
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(archive_manifest, payload)
    _write_json(local_freeze, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "packet_file_count": len(rows),
        "freeze_state": payload["freeze_state"],
    }


def verify_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = _custody_root(custody_root)
    manifest = _read_json(root / "v7-frozen-packet-manifest.json")
    local_manifest = _read_json(_root() / "docs" / "replacement-loss-v7-freeze.json")
    if manifest != local_manifest:
        raise ValueError("local and custodied v7 freeze manifests differ")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v7 packet identity mismatch")
    expected = _canonical_digest({key: value for key, value in manifest.items() if key != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("v7 packet digest is invalid")
    if manifest.get("v3_packet_sha256") != V3_PACKET_SHA256 or manifest.get("v6_packet_sha256") != V6_PACKET_SHA256:
        raise ValueError("v7 pinned dependency identity mismatch")
    for row in manifest["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen v7 source changed: {relative}")
        if _sha(root / "v7-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"frozen v7 archive changed: {relative}")
    _verify_pinned_v6_source_and_archive()
    _verify_pinned_v3_source()
    return {"packet_sha256": manifest["packet_sha256"], "packet_file_count": len(manifest["packet_files"]), "verified": True}


def write_checkpoint_manifest(
    custody_root: str | Path,
    checkpoint_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    v3_freeze, _ = _verify_pinned_v3_source()
    v7_freeze = _read_json(_root() / "docs" / "replacement-loss-v7-freeze.json")
    frozen_at = datetime.fromisoformat(str(v7_freeze["created_at_utc"]))
    selected_at = datetime.now(timezone.utc)
    if frozen_at.tzinfo is None or frozen_at.utcoffset() is None or selected_at <= frozen_at:
        raise ValueError("checkpoint selection must occur after the timezone-aware v7 freeze")
    if len(checkpoint_specs) != len(v3.FAMILIES) or {row.get("family") for row in checkpoint_specs} != set(v3.FAMILIES):
        raise ValueError("checkpoint selection must contain exactly one entry per frozen family")
    destination = root / "checkpoints.json"
    if destination.exists():
        raise FileExistsError("v7 checkpoint selection is immutable once recorded")
    entries = []
    for row in checkpoint_specs:
        family = str(row["family"])
        model_id = str(row["model_id"]).strip()
        revision = str(row["revision"])
        local_path = Path(str(row["local_path"]))
        if local_path.is_absolute() or ".." in local_path.parts:
            raise ValueError("checkpoint path must be relative to external custody")
        if model_id.lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError(f"checkpoint identity was used in prior protocol work: {model_id}")
        if len(revision) != 40 or any(char not in "0123456789abcdefABCDEF" for char in revision):
            raise ValueError("checkpoint revision must be a full 40-character commit SHA")
        if not (root / local_path).is_dir():
            raise FileNotFoundError(f"custodied checkpoint directory is missing for {family}")
        entries.append({
            "family": family,
            "model_id": model_id,
            "revision": revision.lower(),
            "local_path": local_path.as_posix(),
            "selected_after_protocol_freeze": True,
            "not_used_for_protocol_design": True,
            "base_model": True,
        })
    payload = {
        "protocol_id": v3.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": v3_freeze["packet_sha256"],
        "v7_protocol_id": PROTOCOL_ID,
        "v7_packet_sha256": packet["packet_sha256"],
        "selected_at_utc": selected_at.isoformat(),
        "checkpoints": sorted(entries, key=lambda item: v3.FAMILIES.index(item["family"])),
    }
    temporary = root / ".checkpoint-selection-preflight.json"
    try:
        _write_json(temporary, payload)
        with _activate_v7():
            with v6_core._v6_profile():
                v3._load_checkpoints(temporary, root)
    finally:
        temporary.unlink(missing_ok=True)
    _write_json(destination, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v7_packet_sha256": packet["packet_sha256"],
        "checkpoint_manifest_sha256": _sha(destination),
        "checkpoint_count": len(entries),
    }


def _validate_checkpoint_manifest(custody: Path, manifest_path: Path, packet_sha: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("v7_protocol_id") != PROTOCOL_ID or manifest.get("v7_packet_sha256") != packet_sha:
        raise ValueError("checkpoint manifest does not bind the frozen v7 packet")
    if manifest_path.parent != custody:
        raise ValueError("checkpoint manifest must be directly under external custody")
    if manifest.get("v7_packet_sha256") != verify_packet(custody)["packet_sha256"]:
        raise ValueError("checkpoint selection does not follow v7 packet freeze")
    freeze = _read_json(_root() / "docs" / "replacement-loss-v7-freeze.json")
    selected_at = datetime.fromisoformat(str(manifest.get("selected_at_utc", "")))
    frozen_at = datetime.fromisoformat(str(freeze["created_at_utc"]))
    if (
        selected_at.tzinfo is None
        or selected_at.utcoffset() is None
        or frozen_at.tzinfo is None
        or frozen_at.utcoffset() is None
        or selected_at <= frozen_at
    ):
        raise ValueError("checkpoint identities must be selected after the frozen v7 packet")
    for item in manifest.get("checkpoints", []):
        if str(item.get("model_id", "")).lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError("a previously used checkpoint identity cannot be selected for v7")
        if item.get("not_used_for_protocol_design") is not True or item.get("base_model") is not True:
            raise ValueError("v7 requires fresh public base checkpoints selected after freeze")
    with v6_core._v6_profile():
        v3._load_checkpoints(manifest_path, custody)
    return manifest


def _add_v7_binding_to_challenge_manifest(custody: Path, packet_sha: str) -> dict[str, Any]:
    path = custody / "challenge-manifest.json"
    manifest = _read_json(path)
    manifest["v7_protocol_id"] = PROTOCOL_ID
    manifest["v7_packet_sha256"] = packet_sha
    manifest.pop("challenge_manifest_sha256", None)
    manifest["challenge_manifest_sha256"] = _canonical_digest(manifest)
    _write_json(path, manifest)
    return manifest


def _build_v7_report(custody: Path, packet_sha: str, disjointness: dict[str, Any]) -> dict[str, Any]:
    report = _ORIGINAL_BUILD_V6_REPORT(custody, packet_sha, disjointness)
    report.update({
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet_sha,
        "v7_packet_sha256": packet_sha,
        "pinned_v6_packet_sha256": V6_PACKET_SHA256,
        "runner_core": "replacement_loss_v6_runner.py",
        "runner_core_sha256": _sha(_root() / "replacement_loss_v6_runner.py"),
        "candidate_control_alias": {"v3_mode": "wrong_module", "v7_semantics": "output_only_ridge"},
        "claim_ceiling": "LOCAL_BEHAVIOR_AWARE_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ONLY",
        "known_limitations": [
            "one bounded MLP output projection per fresh checkpoint",
            "synthetic prompts and one checkpoint per architecture family",
            "same-owner local replication is not independent scientific authority",
            "monetary cost and energy are not measured",
            "v6 implementation is reused unchanged through this packet-pinned adapter",
        ],
        "next_blocked_authority_gate": "configured external reviewer with independent authority; no review task submitted",
    })
    _write_json(custody / "replacement-loss-v7-report.json", report)
    return report


@contextmanager
def _activate_v7() -> Iterator[None]:
    saved = {name: getattr(v6_core, name) for name in _CORE_PATCHES}
    try:
        v6_core.PROTOCOL_ID = PROTOCOL_ID
        v6_core.V6_PACKET_FILES = V7_PACKET_FILES
        v6_core.DEFAULT_CUSTODY = DEFAULT_CUSTODY
        v6_core.DEFAULT_REPLICA = DEFAULT_REPLICA
        v6_core.PRIOR_CHECKPOINT_IDS = set(v6_core.PRIOR_CHECKPOINT_IDS) | PRIOR_CHECKPOINT_IDS
        v6_core.v6 = v7
        v6_core.fixtures = fixtures
        v6_core.verify_packet = verify_packet
        v6_core._validate_checkpoint_manifest = _validate_checkpoint_manifest
        v6_core.write_checkpoint_manifest = write_checkpoint_manifest
        v6_core._add_v6_binding_to_challenge_manifest = _add_v7_binding_to_challenge_manifest
        v6_core._build_v6_report = _build_v7_report
        yield
    finally:
        for name, value in saved.items():
            setattr(v6_core, name, value)


def run_primary(
    custody_root: str | Path = DEFAULT_CUSTODY,
    checkpoint_manifest: str | Path | None = None,
    prior_custody_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    if checkpoint_manifest is None:
        raise ValueError("post-freeze v7 checkpoint manifest is required")
    if not prior_custody_roots:
        raise ValueError("v7 requires prior custodied fixture roots for overlap checks")
    with _activate_v7():
        result = v6_core.run_primary(root, checkpoint_manifest, prior_custody_roots)
    report_path = root / "replacement-loss-v7-report.json"
    report = _read_json(report_path)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": report["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "case_count": len(report["cases"]),
        "report_sha256": _sha(report_path),
        "core_receipt": result,
    }


def _ensure_core_replication_aliases(primary: Path) -> None:
    packet_dir = primary / "v6-frozen-packet"
    packet_manifest = primary / "v6-frozen-packet-manifest.json"
    if not packet_dir.exists():
        shutil.copytree(primary / "v7-frozen-packet", packet_dir)
    if not packet_manifest.exists():
        shutil.copyfile(primary / "v7-frozen-packet-manifest.json", packet_manifest)


def replicate_packet(
    primary_custody_root: str | Path = DEFAULT_CUSTODY,
    replica_custody_root: str | Path = DEFAULT_REPLICA,
    checkpoint_manifest: str | Path | None = None,
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    if checkpoint_manifest is None:
        checkpoint_manifest = primary / "checkpoints.json"
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    _ensure_core_replication_aliases(primary)
    with _activate_v7():
        result = v6_core.replicate_packet(primary, replica, manifest_path)
    source_report = _read_json(primary / "replacement-loss-v6-report.json")
    source_report.update({
        "protocol_id": PROTOCOL_ID,
        "v7_packet_sha256": source_report["packet_sha256"],
        "reproducibility_status": result["assurance_level"],
        "assurance_level": result["assurance_level"],
        "replication_report_sha256": result["replication_report_sha256"],
    })
    _write_json(primary / "replacement-loss-v7-report.json", source_report)
    replica_report = _read_json(replica / "replacement-loss-v6-replication-report.json")
    replica_report.update({
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": result["packet_sha256"],
        "pinned_v6_packet_sha256": V6_PACKET_SHA256,
        "v7_primary_report_sha256": _sha(primary / "replacement-loss-v7-report.json"),
        "assurance_level": result["assurance_level"],
        "reproducibility_status": result["assurance_level"],
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
    })
    replica_report_path = replica / "replacement-loss-v7-replication-report.json"
    _write_json(replica_report_path, replica_report)
    return {
        "protocol_id": PROTOCOL_ID,
        "assurance_level": result["assurance_level"],
        "experiment_outcome_status": source_report["outcome_status"],
        "packet_sha256": result["packet_sha256"],
        "replication_report_sha256": _sha(replica_report_path),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--custody-root", default=str(DEFAULT_CUSTODY))
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--custody-root", default=str(DEFAULT_CUSTODY))
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--custody-root", default=str(DEFAULT_CUSTODY))
    run_parser.add_argument("--checkpoint-manifest", required=True)
    run_parser.add_argument("--prior-custody-root", action="append", required=True)
    replicate_parser = subparsers.add_parser("replicate")
    replicate_parser.add_argument("--primary-custody-root", default=str(DEFAULT_CUSTODY))
    replicate_parser.add_argument("--replica-custody-root", default=str(DEFAULT_REPLICA))
    replicate_parser.add_argument("--checkpoint-manifest")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root)
    elif args.command == "run":
        result = run_primary(args.custody_root, args.checkpoint_manifest, args.prior_custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root, args.checkpoint_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
