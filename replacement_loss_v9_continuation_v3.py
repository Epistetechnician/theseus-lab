"""Digest-bound v3 continuation for the frozen replacement-loss-v9 packet.

State slice: theseus-lab-sandbox-v1.

V3 resumes after the complete challenge packet was sealed. It preserves the
exact candidate and challenge bytes and corrects a read-time digest comparison:
candidate stage digests identify tensor state, while the candidate freeze
separately binds serialized file bytes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import replacement_loss_v9_continuation_v2 as continuation_v2
import replacement_loss_v9_runner as runner

CONTINUATION_ID = "theseus-replacement-loss-v9-execution-continuation-v3"
STATE_SLICE = runner.STATE_SLICE
SNAPSHOT_NAME = "v9-continuation-v3-staged-snapshot.json"
CANDIDATE_PREFLIGHT_NAME = "v9-continuation-v3-candidate-integrity-preflight.json"
FREEZE_NAME = "v9-continuation-v3-freeze.json"
PRIMARY_RECEIPT_NAME = "v9-continuation-v3-primary.json"
STOP_NAME = "v9-continuation-v3-stop.json"
REPLICATION_RECEIPT_NAME = "v9-continuation-v3-replication.json"
SOURCE_FILES = (
    "replacement_loss_v9_continuation_v3.py",
    "tests/test_replacement_loss_v9_continuation_v3.py",
    "docs/replacement-loss-v9-continuation-v3.md",
)
OUTPUT_NAMES = {
    "prediction-manifest.json",
    "predictions",
    "outcome-manifest.json",
    "outcomes",
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return runner._canonical_digest(value)


def _read_json(path: Path) -> dict[str, Any]:
    return runner._read_json(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    runner._write_json(path, value)


def _custody(path: str | Path) -> Path:
    return runner._custody_root(path)


def _source_rows() -> list[dict[str, Any]]:
    repo = runner._root()
    rows = []
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"v3 continuation source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _file_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"v3 continuation custody may not contain symlinks: {path.name}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    return rows


def _validate_v2_stop(v2_root: Path, prior_v1_root: str | Path) -> dict[str, Any]:
    frozen = continuation_v2.verify_continuation(v2_root, prior_v1_root)
    stop_path = v2_root / continuation_v2.STOP_NAME
    if not stop_path.is_file():
        raise FileNotFoundError("v2 continuation stop receipt is missing")
    stop = _read_json(stop_path)
    stop_material = {key: value for key, value in stop.items() if key != "continuation_stop_sha256"}
    if stop.get("continuation_stop_sha256") != _canonical_digest(stop_material):
        raise ValueError("v2 continuation stop receipt digest is invalid")
    if (
        stop.get("continuation_id") != continuation_v2.CONTINUATION_ID
        or stop.get("continuation_packet_sha256") != frozen.get("continuation_packet_sha256")
        or stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("stopped_phase") != "prediction_sealing"
        or stop.get("error_type") != "ValueError"
        or stop.get("error") != "frozen candidate bytes changed"
    ):
        raise ValueError("v2 stop is not the expected frozen candidate digest-comparison failure")
    for relative in OUTPUT_NAMES:
        if (v2_root / relative).exists():
            raise FileExistsError(f"v2 stop unexpectedly left prediction or outcome material: {relative}")
    if not (v2_root / "challenge-manifest.json").is_file():
        raise FileNotFoundError("complete v2 challenge manifest is missing")
    return {
        "freeze": frozen,
        "stop": stop,
        "stop_sha256": _sha_file(stop_path),
        "freeze_sha256": _sha_file(v2_root / continuation_v2.FREEZE_NAME),
    }


def _validate_challenges(root: Path) -> dict[str, Any]:
    manifest_path = root / "challenge-manifest.json"
    manifest = _read_json(manifest_path)
    material = {key: value for key, value in manifest.items() if key != "challenge_manifest_sha256"}
    if manifest.get("challenge_manifest_sha256") != _canonical_digest(material):
        raise ValueError("v2 challenge manifest digest is invalid")
    if (
        manifest.get("candidate_freeze_sha256") != _sha_file(root / "candidate-freeze.json")
        or manifest.get("challenge_generation_started_after_candidate_freeze") is not True
        or manifest.get("fixture_count") != 3
        or len(manifest.get("cases", [])) != 9
    ):
        raise ValueError("v2 challenge manifest is incomplete or not bound to frozen candidates")
    expected = {
        (family, seed_index)
        for family in runner.FAMILIES
        for seed_index in range(len(runner.fixtures.SEEDS))
    }
    observed = {(row.get("family"), int(row.get("seed_index", -1))) for row in manifest["cases"]}
    if observed != expected or len(manifest.get("entropy_commitments", [])) != 3:
        raise ValueError("v2 challenge manifest does not cover the frozen nine-case design")
    for row in manifest["cases"]:
        challenge_dir = root / "challenges" / row["family"] / f"seed-{row['seed_index']}"
        prompt_path = challenge_dir / "prompts.json"
        intervention_path = challenge_dir / "interventions.npz"
        if (
            _sha_file(prompt_path) != row.get("prompt_file_sha256")
            or _sha_file(intervention_path) != row.get("interventions_file_sha256")
        ):
            raise ValueError("v2 challenge payload differs from its sealed manifest")
    for seed_index, commitment in enumerate(manifest["entropy_commitments"]):
        secret = root / "challenge-secrets" / f"seed-{seed_index}.bin"
        if not secret.is_file() or _sha_file(secret) != commitment.get("entropy_sha256"):
            raise ValueError("v2 challenge entropy differs from its sealed commitment")
    return manifest


def _load_candidate_with_state_digest(
    custody: str | Path,
    stage: dict[str, Any],
    freeze: dict[str, Any],
) -> Any:
    """Check frozen file bytes and compare the stage digest to loaded tensor state."""

    root = _custody(custody)
    relative = str(stage["candidate_path"])
    matches = [row for row in freeze.get("candidate_files", []) if row.get("path") == relative]
    if len(matches) != 1:
        raise ValueError("candidate is not uniquely covered by the candidate freeze")
    path = root / relative
    if _sha_file(path) != matches[0].get("sha256"):
        raise ValueError("frozen candidate file bytes changed")
    module = runner._load_module(
        path,
        int(stage["in_features"]),
        int(stage["out_features"]),
        bool(stage["use_bias"]),
    )
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    if runner.v3._state_digest(state) != stage.get("candidate_sha256"):
        raise ValueError("frozen candidate tensor-state digest changed")
    return module


def _candidate_loader_for_runtime():
    freezes: dict[Path, dict[str, Any]] = {}

    def load(custody: Path, stage: dict[str, Any]) -> Any:
        root = _custody(custody)
        if root not in freezes:
            packet = runner.verify_packet(root)
            freezes[root] = runner._verify_candidate_freeze(root, packet["packet_sha256"])
        return _load_candidate_with_state_digest(root, stage, freezes[root])

    return load


def _candidate_integrity_preflight(root: Path) -> dict[str, Any]:
    packet = runner.verify_packet(root)
    freeze = runner._verify_candidate_freeze(root, packet["packet_sha256"])
    preparation = _read_json(root / "candidate-preparation.json")
    checked = 0
    for trajectory in preparation.get("trajectories", []):
        for stage in trajectory.get("stages", []):
            module = _load_candidate_with_state_digest(root, stage, freeze)
            del module
            checked += 1
    if checked != 81 or len(freeze.get("candidate_files", [])) != 162:
        raise ValueError("candidate integrity preflight did not cover all frozen files and placements")
    receipt = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "candidate_stage_state_digest_checks": checked,
        "candidate_and_control_file_digest_checks": len(freeze["candidate_files"]),
        "all_checks_pass": True,
        "prediction_or_outcome_data_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["preflight_sha256"] = _canonical_digest(receipt)
    _write_json(root / CANDIDATE_PREFLIGHT_NAME, receipt)
    return receipt


def stage_challenge_custody(
    v2_root: str | Path,
    prior_v1_root: str | Path,
    custody_root: str | Path,
) -> dict[str, Any]:
    source = _custody(v2_root)
    destination = _custody(custody_root)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("v2 and v3 custody roots must be distinct and non-nested")
    previous = _validate_v2_stop(source, prior_v1_root)
    _validate_challenges(source)
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError("v3 custody staging is immutable and destination is not empty")
        destination.rmdir()
    shutil.copytree(source, destination, copy_function=shutil.copy2)
    for name in OUTPUT_NAMES:
        if (destination / name).exists():
            raise FileExistsError(f"v3 stage copied unexpected prediction or outcome material: {name}")
    continuation_v2.verify_continuation(destination, prior_v1_root)
    _validate_challenges(destination)
    preflight = _candidate_integrity_preflight(destination)
    snapshot = {
        "continuation_id": CONTINUATION_ID,
        "prior_v2_stop_sha256": previous["stop_sha256"],
        "challenge_manifest_sha256": _sha_file(destination / "challenge-manifest.json"),
        "files": _file_rows(destination),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    snapshot["snapshot_sha256"] = _canonical_digest(snapshot)
    _write_json(destination / SNAPSHOT_NAME, snapshot)
    return {
        "continuation_id": CONTINUATION_ID,
        "staged_file_count": len(snapshot["files"]),
        "challenge_case_count": 9,
        "candidate_integrity_preflight_sha256": preflight["preflight_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "challenge_manifest_sha256": snapshot["challenge_manifest_sha256"],
        "prediction_or_outcome_data_copied": False,
    }


def _verify_snapshot(root: Path, previous: dict[str, Any]) -> dict[str, Any]:
    snapshot = _read_json(root / SNAPSHOT_NAME)
    material = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if snapshot.get("snapshot_sha256") != _canonical_digest(material):
        raise ValueError("v3 staged snapshot digest is invalid")
    if snapshot.get("prior_v2_stop_sha256") != previous["stop_sha256"]:
        raise ValueError("v3 staged snapshot no longer binds the v2 stop")
    if snapshot.get("challenge_manifest_sha256") != _sha_file(root / "challenge-manifest.json"):
        raise ValueError("v3 staged snapshot no longer binds the complete challenge manifest")
    for row in snapshot.get("files", []):
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"staged v3 base file changed after copy: {row['path']}")
    return snapshot


def _validate_v3_base(
    root: Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
    *,
    require_no_outputs: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    previous = _validate_v2_stop(_custody(v2_root), prior_v1_root)
    v2_packet = continuation_v2.verify_continuation(v2_root, prior_v1_root)
    if _sha_file(root / continuation_v2.FREEZE_NAME) != previous["freeze_sha256"]:
        raise ValueError("v3 custody differs from the bound v2 freeze")
    if _sha_file(root / continuation_v2.STOP_NAME) != previous["stop_sha256"]:
        raise ValueError("v3 custody differs from the bound v2 stop")
    base = continuation_v2.verify_continuation(root, prior_v1_root)
    challenge = _validate_challenges(root)
    snapshot = _verify_snapshot(root, previous)
    preflight = _read_json(root / CANDIDATE_PREFLIGHT_NAME)
    preflight_material = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if (
        preflight.get("preflight_sha256") != _canonical_digest(preflight_material)
        or preflight.get("all_checks_pass") is not True
        or preflight.get("candidate_stage_state_digest_checks") != 81
        or preflight.get("candidate_and_control_file_digest_checks") != 162
    ):
        raise ValueError("v3 candidate integrity preflight is invalid")
    if require_no_outputs:
        for name in OUTPUT_NAMES:
            if (root / name).exists():
                raise FileExistsError(f"v3 resume requires no prediction or outcome output: {name}")
    return base, {
        "previous": previous,
        "v2_packet": v2_packet,
        "challenge": challenge,
        "snapshot": snapshot,
        "preflight": preflight,
    }


def freeze_continuation(
    custody_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    root = _custody(custody_root)
    if (root / FREEZE_NAME).exists():
        raise FileExistsError("v3 continuation freeze is immutable")
    base, context = _validate_v3_base(root, v2_root, prior_v1_root, require_no_outputs=True)
    runtime = runner.verify_runtime()
    previous = context["previous"]
    payload = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": runner.ASSURANCE_OVERLAY,
        "base_packet_sha256": base["base_packet_sha256"],
        "v2_continuation_packet_sha256": context["v2_packet"]["continuation_packet_sha256"],
        "v2_continuation_freeze_sha256": previous["freeze_sha256"],
        "v2_stop_sha256": previous["stop_sha256"],
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "challenge_manifest_sha256": _sha_file(root / "challenge-manifest.json"),
        "challenge_payload_inventory_sha256": context["snapshot"]["snapshot_sha256"],
        "candidate_integrity_preflight_sha256": context["preflight"]["preflight_sha256"],
        "continuation_sources": _source_rows(),
        "runtime": {
            "python": runtime["python"],
            "dependencies": runtime["dependencies"],
            "dependency_manifest_sha256": runtime["dependency_manifest_sha256"],
            "locked_environment_pass": runtime["locked_environment_pass"],
            "cpu_thread_count": runtime["cpu_thread_count"],
        },
        "repair_contract": {
            "challenge_entropy_regenerated": False,
            "candidate_refit": False,
            "candidate_file_bytes_checked_against_frozen_file_inventory": True,
            "candidate_tensor_state_checked_against_stage_digest": True,
            "prediction_and_outcome_functions_changed": False,
            "held_out_outcomes_opened_at_freeze": False,
        },
        "claim_ceiling": "LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    payload["continuation_packet_sha256"] = _canonical_digest(payload)
    _write_json(root / FREEZE_NAME, payload)
    return {
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": payload["continuation_packet_sha256"],
        "base_packet_sha256": payload["base_packet_sha256"],
        "candidate_freeze_sha256": payload["candidate_freeze_sha256"],
        "challenge_manifest_sha256": payload["challenge_manifest_sha256"],
        "v2_stop_sha256": payload["v2_stop_sha256"],
    }


def verify_continuation(
    custody_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    root = _custody(custody_root)
    base, context = _validate_v3_base(root, v2_root, prior_v1_root, require_no_outputs=False)
    payload = _read_json(root / FREEZE_NAME)
    material = {key: value for key, value in payload.items() if key != "continuation_packet_sha256"}
    if payload.get("continuation_packet_sha256") != _canonical_digest(material):
        raise ValueError("v3 continuation packet digest is invalid")
    previous = context["previous"]
    expected = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "base_packet_sha256": base["base_packet_sha256"],
        "v2_continuation_packet_sha256": context["v2_packet"]["continuation_packet_sha256"],
        "v2_continuation_freeze_sha256": previous["freeze_sha256"],
        "v2_stop_sha256": previous["stop_sha256"],
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "challenge_manifest_sha256": _sha_file(root / "challenge-manifest.json"),
        "challenge_payload_inventory_sha256": context["snapshot"]["snapshot_sha256"],
        "candidate_integrity_preflight_sha256": context["preflight"]["preflight_sha256"],
        "continuation_sources": _source_rows(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"v3 continuation packet binding changed: {key}")
    runtime = runner.verify_runtime()
    if (
        payload.get("runtime", {}).get("dependency_manifest_sha256") != runtime["dependency_manifest_sha256"]
        or payload.get("runtime", {}).get("locked_environment_pass") is not True
    ):
        raise ValueError("v3 continuation runtime differs from its frozen environment")
    return payload


def _with_compatible_candidate_loader(operation):
    original = runner._load_candidate
    runner._load_candidate = _candidate_loader_for_runtime()
    try:
        return operation()
    finally:
        runner._load_candidate = original


def continue_primary(
    custody_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
) -> dict[str, Any]:
    root = _custody(custody_root)
    packet = verify_continuation(root, v2_root, prior_v1_root)
    for name in OUTPUT_NAMES | {PRIMARY_RECEIPT_NAME, STOP_NAME}:
        if (root / name).exists():
            raise FileExistsError(f"v3 prediction continuation is immutable: {name}")
    previous = _validate_v2_stop(_custody(v2_root), prior_v1_root)
    original_stop = _read_json(root / "primary-run-stop.json")
    prior_v1_stop = _read_json(_custody(prior_v1_root) / continuation_v2.continuation_v1.PRIMARY_STOP_NAME)
    runner._ACTIVE_RUN_STARTED = time.monotonic()
    runner._ACTIVE_CASE_STARTED = None
    runner._ACTIVE_PHASE = "prediction_sealing"
    started = time.monotonic()
    try:
        prediction = _with_compatible_candidate_loader(lambda: runner.seal_predictions(root))
        runner._ACTIVE_PHASE = "source_outcome_evaluation"
        outcome = _with_compatible_candidate_loader(lambda: runner.evaluate_outcomes(root))
        report_path = root / "outcome-manifest.json"
        report = _read_json(report_path)
        v2_stop = previous["stop"]
        prior_elapsed = (
            float(original_stop["elapsed_wall_seconds"])
            + float(prior_v1_stop["elapsed_wall_seconds"])
            + float(v2_stop["elapsed_wall_seconds"])
        )
        total_elapsed = prior_elapsed + time.monotonic() - started
        current_runtime = report["runtime"]
        earlier_runtimes = [original_stop["runtime"], prior_v1_stop["runtime"], v2_stop["runtime"]]
        peak_rss = max([int(current_runtime["peak_rss_bytes"]), *[int(row["peak_rss_bytes"]) for row in earlier_runtimes]])
        total_cpu = (
            float(original_stop["runtime"]["process_cpu_seconds"])
            + float(prior_v1_stop["runtime"]["process_cpu_seconds"])
            + float(v2_stop["runtime"]["process_cpu_seconds"])
            + float(current_runtime["process_cpu_seconds"])
        )
        if total_elapsed > runner.MAX_TOTAL_SECONDS or peak_rss > runner.MAX_RSS_BYTES:
            report["outcome_status"] = "INCONCLUSIVE"
            report["assurance_level"] = "INCONCLUSIVE"
        report["reproducibility_status"] = "PENDING_CLEAN_LOCAL_REPRODUCTION"
        report["primary_protocol_wall_seconds"] = total_elapsed
        report["primary_budget_pass"] = report["outcome_status"] != "INCONCLUSIVE"
        report["runtime"]["peak_rss_bytes"] = peak_rss
        report["runtime"]["process_cpu_seconds"] = total_cpu
        report["runtime"]["initial_candidate_preparation_cpu_seconds"] = original_stop["runtime"]["process_cpu_seconds"]
        report["runtime"]["v1_failed_continuation_cpu_seconds"] = prior_v1_stop["runtime"]["process_cpu_seconds"]
        report["runtime"]["v2_challenge_continuation_cpu_seconds"] = v2_stop["runtime"]["process_cpu_seconds"]
        report["runtime"]["v3_prediction_continuation_cpu_seconds"] = current_runtime["process_cpu_seconds"]
        report["continuation_id"] = CONTINUATION_ID
        report["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
        report["v2_continuation_packet_sha256"] = packet["v2_continuation_packet_sha256"]
        report["v2_stop_sha256"] = previous["stop_sha256"]
        report["candidate_refit_during_continuation"] = False
        report["claim_ceiling"] = packet["claim_ceiling"]
        report.pop("report_sha256", None)
        report["report_sha256"] = _canonical_digest(report)
        _write_json(report_path, report)
        receipt = {
            "continuation_id": CONTINUATION_ID,
            "protocol_id": runner.PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "assurance_overlay": runner.ASSURANCE_OVERLAY,
            "continuation_packet_sha256": packet["continuation_packet_sha256"],
            "v2_continuation_packet_sha256": packet["v2_continuation_packet_sha256"],
            "base_packet_sha256": packet["base_packet_sha256"],
            "candidate_freeze_sha256": packet["candidate_freeze_sha256"],
            "challenge_manifest_sha256": packet["challenge_manifest_sha256"],
            "v2_stop_sha256": previous["stop_sha256"],
            "prediction_manifest_sha256": prediction["prediction_manifest_sha256"],
            "outcome_report_sha256": report["report_sha256"],
            "outcome_status": report["outcome_status"],
            "assurance_level": report["assurance_level"],
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
            "candidate_refit": False,
            "challenge_entropy_regenerated": False,
            "candidate_file_byte_and_tensor_state_checks_passed": True,
            "held_out_outcomes_opened_after_prediction_seal": True,
            "combined_primary_active_wall_seconds": total_elapsed,
            "combined_primary_cpu_seconds": total_cpu,
            "combined_peak_rss_bytes": peak_rss,
            "monetary_cost": "NONE_GMAN_OR_PAID_COMPUTE_USED",
            "energy": "NOT_MEASURED",
            "claim_ceiling": packet["claim_ceiling"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["continuation_receipt_sha256"] = _canonical_digest(receipt)
        _write_json(root / PRIMARY_RECEIPT_NAME, receipt)
        runner._write_results_document(root, report_path)
        return {
            "outcome_status": report["outcome_status"],
            "assurance_level": report["assurance_level"],
            "reproducibility_status": report["reproducibility_status"],
            "continuation_packet_sha256": packet["continuation_packet_sha256"],
            "continuation_receipt_sha256": receipt["continuation_receipt_sha256"],
            "report_sha256": report["report_sha256"],
        }
    except Exception as error:
        stop = {
            "continuation_id": CONTINUATION_ID,
            "continuation_packet_sha256": packet["continuation_packet_sha256"],
            "v2_stop_sha256": previous["stop_sha256"],
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": runner._ACTIVE_PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_wall_seconds": time.monotonic() - started,
            "runtime": runner._runtime_receipt(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["continuation_stop_sha256"] = _canonical_digest(stop)
        _write_json(root / STOP_NAME, stop)
        raise
    finally:
        runner._ACTIVE_RUN_STARTED = None
        runner._ACTIVE_CASE_STARTED = None
        runner._ACTIVE_PHASE = "idle"


def _copy_context(primary: Path, replica: Path, v2_root: Path, prior_v1_root: Path) -> None:
    packet_dir = replica / "v9-continuation-v3-packet"
    for relative in SOURCE_FILES:
        source = runner._root() / relative
        target = packet_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    contexts = (
        (primary, (FREEZE_NAME, PRIMARY_RECEIPT_NAME, "primary-run-stop.json", SNAPSHOT_NAME, CANDIDATE_PREFLIGHT_NAME)),
        (v2_root, (continuation_v2.FREEZE_NAME, continuation_v2.STOP_NAME)),
        (prior_v1_root, (continuation_v2.continuation_v1.FREEZE_NAME, continuation_v2.continuation_v1.PRIMARY_STOP_NAME)),
    )
    target_dir = replica / "v9-continuation-v3-context"
    for source_root, relatives in contexts:
        for relative in relatives:
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"v3 replication context file is missing: {relative}")
            shutil.copy2(source, target_dir / Path(relative).name)


def replicate(
    primary_root: str | Path,
    replica_root: str | Path,
    v2_root: str | Path,
    prior_v1_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    primary = _custody(primary_root)
    replica = _custody(replica_root)
    v2_root_path = _custody(v2_root)
    prior_v1 = _custody(prior_v1_root)
    packet = verify_continuation(primary, v2_root_path, prior_v1)
    primary_receipt = _read_json(primary / PRIMARY_RECEIPT_NAME)
    receipt_material = {key: value for key, value in primary_receipt.items() if key != "continuation_receipt_sha256"}
    if (
        primary_receipt.get("continuation_receipt_sha256") != _canonical_digest(receipt_material)
        or primary_receipt.get("continuation_packet_sha256") != packet["continuation_packet_sha256"]
    ):
        raise ValueError("v3 primary receipt does not bind its frozen continuation")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("v3 replica custody must be empty before replication")
    import replacement_loss_v9_replicator as replicator

    def execute_replica() -> dict[str, Any]:
        return replicator._run_replica_impl(primary, replica, prior_custody_roots)

    base_result = _with_compatible_candidate_loader(execute_replica)
    _copy_context(primary, replica, v2_root_path, prior_v1)
    replication_path = primary / "replication-manifest.json"
    replication = _read_json(replication_path)
    base_replication_sha = replication["replication_sha256"]
    replication.pop("replication_sha256", None)
    replication.update({
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": packet["continuation_packet_sha256"],
        "continuation_primary_receipt_sha256": _sha_file(primary / PRIMARY_RECEIPT_NAME),
        "v2_stop_sha256": packet["v2_stop_sha256"],
        "base_replication_manifest_sha256": base_replication_sha,
    })
    replication["replication_sha256"] = _canonical_digest(replication)
    _write_json(primary / "replication-manifest.json", replication)
    _write_json(replica / "replication-manifest.json", replication)
    continuation_replication = {
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": packet["continuation_packet_sha256"],
        "base_v9_replication_manifest_sha256": base_replication_sha,
        "bound_replication_manifest_sha256": replication["replication_sha256"],
        "reproducibility_status": base_result["reproducibility_status"],
        "assurance_level": base_result["assurance_level"],
        "checks": base_result["checks"],
        "reviewer_status": "NOT_SUBMITTED",
        "v2_stop_sha256": packet["v2_stop_sha256"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    continuation_replication["continuation_replication_sha256"] = _canonical_digest(continuation_replication)
    _write_json(primary / REPLICATION_RECEIPT_NAME, continuation_replication)
    _write_json(replica / REPLICATION_RECEIPT_NAME, continuation_replication)

    report_path = primary / "outcome-manifest.json"
    report = _read_json(report_path)
    report["reproducibility_status"] = base_result["reproducibility_status"]
    report["assurance_level"] = base_result["assurance_level"]
    report["replication_manifest_sha256"] = _sha_file(replication_path)
    report["continuation_replication_sha256"] = continuation_replication["continuation_replication_sha256"]
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_digest(report)
    _write_json(report_path, report)
    replica_report_path = replica / "outcome-manifest.json"
    replica_report = _read_json(replica_report_path)
    replica_report["continuation_id"] = CONTINUATION_ID
    replica_report["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
    replica_report["replication_manifest_sha256"] = _sha_file(replica / "replication-manifest.json")
    replica_report.pop("report_sha256", None)
    replica_report["report_sha256"] = _canonical_digest(replica_report)
    _write_json(replica_report_path, replica_report)
    runner._write_results_document(primary, report_path)
    return {
        "reproducibility_status": base_result["reproducibility_status"],
        "assurance_level": base_result["assurance_level"],
        "checks": base_result["checks"],
        "replication_sha256": replication["replication_sha256"],
        "continuation_replication_sha256": continuation_replication["continuation_replication_sha256"],
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage", "freeze", "verify", "continue-primary", "replicate"))
    parser.add_argument("--source-root")
    parser.add_argument("--custody-root")
    parser.add_argument("--v2-root")
    parser.add_argument("--prior-v1-root")
    parser.add_argument("--primary-root")
    parser.add_argument("--replica-root")
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    if args.command == "stage":
        if not args.source_root or not args.prior_v1_root or not args.custody_root:
            parser.error("stage requires --source-root, --prior-v1-root, and --custody-root")
        result = stage_challenge_custody(args.source_root, args.prior_v1_root, args.custody_root)
    elif args.command == "freeze":
        if not args.custody_root or not args.v2_root or not args.prior_v1_root:
            parser.error("freeze requires --custody-root, --v2-root, and --prior-v1-root")
        result = freeze_continuation(args.custody_root, args.v2_root, args.prior_v1_root)
    elif args.command == "verify":
        if not args.custody_root or not args.v2_root or not args.prior_v1_root:
            parser.error("verify requires --custody-root, --v2-root, and --prior-v1-root")
        payload = verify_continuation(args.custody_root, args.v2_root, args.prior_v1_root)
        result = {"continuation_id": payload["continuation_id"], "continuation_packet_sha256": payload["continuation_packet_sha256"]}
    elif args.command == "continue-primary":
        if not args.custody_root or not args.v2_root or not args.prior_v1_root:
            parser.error("continue-primary requires --custody-root, --v2-root, and --prior-v1-root")
        result = continue_primary(args.custody_root, args.v2_root, args.prior_v1_root)
    else:
        if not args.primary_root or not args.replica_root or not args.v2_root or not args.prior_v1_root:
            parser.error("replicate requires --primary-root, --replica-root, --v2-root, and --prior-v1-root")
        result = replicate(
            args.primary_root,
            args.replica_root,
            args.v2_root,
            args.prior_v1_root,
            args.prior_custody_root,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
