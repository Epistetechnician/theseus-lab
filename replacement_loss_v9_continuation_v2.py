"""Digest-bound v2 continuation for the frozen replacement-loss-v9 packet.

State slice: theseus-lab-sandbox-v1.

The first continuation stopped after creating partial held-out material. This
continuation starts from a clean external copy of the same frozen candidates,
binds the prior stop and partial-material digests, and fixes the challenge
capture layout at read time without changing frozen v9 source bytes.
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

import replacement_loss_v9_continuation as continuation_v1
import replacement_loss_v9_runner as runner

CONTINUATION_ID = "theseus-replacement-loss-v9-execution-continuation-v2"
STATE_SLICE = runner.STATE_SLICE
SNAPSHOT_NAME = "v9-continuation-v2-staged-snapshot.json"
SHAPE_PREFLIGHT_NAME = "v9-continuation-v2-shape-preflight.json"
FREEZE_NAME = "v9-continuation-v2-freeze.json"
PRIMARY_RECEIPT_NAME = "v9-continuation-v2-primary.json"
STOP_NAME = "v9-continuation-v2-stop.json"
REPLICATION_RECEIPT_NAME = "v9-continuation-v2-replication.json"
SOURCE_FILES = (
    "replacement_loss_v9_continuation_v2.py",
    "tests/test_replacement_loss_v9_continuation_v2.py",
    "docs/replacement-loss-v9-continuation-v2.md",
)
HELD_OUT_ROOT_NAMES = {
    "challenge-secrets",
    "challenges",
    "predictions",
    "outcomes",
    "challenge-manifest.json",
    "prediction-manifest.json",
    "outcome-manifest.json",
    "v9-final-disjointness-receipt.json",
}
PARTIAL_PRIOR_FILES = (
    "challenge-secrets/seed-0.bin",
    "fixtures/challenge-seed-0.json",
)


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
            raise FileNotFoundError(f"v2 continuation source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _root_file_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"staged continuation custody may not contain symlinks: {path.name}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    return rows


def _no_held_out_material(root: Path) -> None:
    present = sorted(name for name in HELD_OUT_ROOT_NAMES if (root / name).exists())
    if present:
        raise FileExistsError(f"clean v2 custody contains held-out material: {present}")
    challenge_fixtures = sorted((root / "fixtures").glob("challenge-seed-*.json"))
    if challenge_fixtures:
        raise FileExistsError("clean v2 custody contains a prior challenge fixture")


def _validate_prior_attempt(prior_root: str | Path) -> dict[str, Any]:
    root = _custody(prior_root)
    previous_freeze = continuation_v1.verify_continuation(root)
    stop_path = root / continuation_v1.PRIMARY_STOP_NAME
    if not stop_path.is_file():
        raise FileNotFoundError("v1 continuation stop receipt is missing")
    stop = _read_json(stop_path)
    stop_material = {key: value for key, value in stop.items() if key != "continuation_stop_sha256"}
    if stop.get("continuation_stop_sha256") != _canonical_digest(stop_material):
        raise ValueError("v1 continuation stop receipt digest is invalid")
    if (
        stop.get("continuation_id") != continuation_v1.CONTINUATION_ID
        or stop.get("continuation_packet_sha256") != previous_freeze.get("continuation_packet_sha256")
        or stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("stopped_phase") != "challenge_generation"
        or stop.get("error_type") != "IndexError"
        or stop.get("error") != "tuple index out of range"
    ):
        raise ValueError("v1 stop is not the expected challenge activation-layout failure")
    partial_rows = []
    for relative in PARTIAL_PRIOR_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"expected partial held-out artifact is missing: {relative}")
        partial_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    for relative in (
        "challenge-secrets/seed-1.bin",
        "challenge-secrets/seed-2.bin",
        "challenge-manifest.json",
        "challenges",
        "prediction-manifest.json",
        "predictions",
        "outcome-manifest.json",
        "outcomes",
    ):
        if (root / relative).exists():
            raise ValueError(f"unexpected held-out continuation artifact exists: {relative}")
    challenge_fixtures = sorted(path.name for path in (root / "fixtures").glob("challenge-seed-*.json"))
    if challenge_fixtures != ["challenge-seed-0.json"]:
        raise ValueError("v1 partial challenge fixture inventory differs from the recorded stop")
    return {
        "root": root,
        "freeze": previous_freeze,
        "stop": stop,
        "stop_sha256": _sha_file(stop_path),
        "freeze_sha256": _sha_file(root / continuation_v1.FREEZE_NAME),
        "partial_material": partial_rows,
    }


def _stage_ignore(source_root: Path):
    def ignore(directory: str, names: list[str]) -> list[str]:
        current = Path(directory).resolve()
        ignored = set()
        if current == source_root:
            ignored.update(name for name in names if name in HELD_OUT_ROOT_NAMES)
            ignored.update(name for name in names if name.startswith("v9-continuation-v2-"))
        if current.name == "fixtures":
            ignored.update(name for name in names if name.startswith("challenge-seed-") and name.endswith(".json"))
        return sorted(ignored)

    return ignore


def _restore_padded_batch_layout(flat: Any, attention_mask: Any) -> Any:
    """Restore [batch, sequence, hidden] from valid-token-major [tokens, hidden]."""

    mask = attention_mask.to(device=flat.device).bool()
    if flat.ndim != 2 or mask.ndim != 2 or flat.shape[0] != int(mask.sum().item()):
        raise ValueError("flattened projection inputs do not match the attention-mask token count")
    padded = flat.new_zeros((int(mask.shape[0]), int(mask.shape[1]), int(flat.shape[-1])))
    padded[mask] = flat
    return padded


def _capture_batched_layer_inputs(model: Any, family: str, inputs: dict[str, Any]) -> list[Any]:
    flattened = runner.v3._capture_all_layer_inputs(model, family, inputs)
    return [_restore_padded_batch_layout(layer, inputs["attention_mask"]) for layer in flattened]


def _visible_shape_preflight(root: Path) -> dict[str, Any]:
    packet = runner.verify_packet(root)
    checkpoint_specs = runner._load_checkpoint_specs(root, packet["packet_sha256"])
    visible_payload = _read_json(root / "fixtures/visible-selection.json")
    visible_records = visible_payload["records"][:4]
    rows = []
    for entry in checkpoint_specs:
        checkpoint = runner.v3.Checkpoint(
            family=entry["family"],
            model_id=entry["model_id"],
            revision=entry["revision"],
            local_path=(root / entry["local_path"]).resolve(),
            selected_after_protocol_freeze=True,
        )
        model, tokenizer = runner.v3._load_local_model(checkpoint)
        token_rows = runner._tokenized(visible_records, tokenizer, continuation=True)
        group, inputs = next(runner.v3._batches(
            token_rows,
            tokenizer.pad_token_id,
            batch_size=len(token_rows),
        ))
        flattened = runner.v3._capture_all_layer_inputs(model, entry["family"], inputs)
        restored = [_restore_padded_batch_layout(layer, inputs["attention_mask"]) for layer in flattened]
        mask = inputs["attention_mask"].bool()
        if len(restored) != runner.v3._layer_count(model, entry["family"]):
            raise ValueError("visible activation-layout preflight missed one or more model layers")
        for flat_layer, padded_layer in zip(flattened, restored, strict=True):
            if tuple(padded_layer.shape[:2]) != tuple(mask.shape) or not runner.torch.equal(
                padded_layer[mask], flat_layer
            ):
                raise ValueError("visible activation-layout preflight failed exact valid-token roundtrip")
        rows.append({
            "family": entry["family"],
            "model_revision": entry["revision"],
            "layer_count": len(restored),
            "batch_size": len(group),
            "sequence_width": int(mask.shape[1]),
            "valid_token_count": int(mask.sum().item()),
            "hidden_width": int(flattened[0].shape[-1]),
            "restored_layout": "batch_sequence_hidden",
            "valid_activation_roundtrip": True,
        })
        del model, tokenizer, flattened, restored
        gc.collect()
    receipt = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "visible_fixture_sha256": _sha_file(root / "fixtures/visible-selection.json"),
        "families": rows,
        "all_families_pass": len(rows) == 3 and all(item["valid_activation_roundtrip"] for item in rows),
        "held_out_data_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not receipt["all_families_pass"]:
        raise ValueError("visible activation-layout qualification did not pass all three families")
    receipt["preflight_sha256"] = _canonical_digest(receipt)
    _write_json(root / SHAPE_PREFLIGHT_NAME, receipt)
    return receipt


def stage_clean_custody(source_root: str | Path, custody_root: str | Path) -> dict[str, Any]:
    """Copy only frozen pre-challenge state; retain the v1 partial packet in place."""

    source = _custody(source_root)
    destination = _custody(custody_root)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("v1 and v2 custody roots must be distinct and non-nested")
    prior = _validate_prior_attempt(source)
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError("v2 custody staging is immutable and the destination is not empty")
        destination.rmdir()
    shutil.copytree(source, destination, ignore=_stage_ignore(source))
    _no_held_out_material(destination)
    continuation_v1._assert_stopped_at_unrevealed_boundary(destination)
    shape = _visible_shape_preflight(destination)
    snapshot = {
        "continuation_id": CONTINUATION_ID,
        "prior_v1_stop_sha256": prior["stop_sha256"],
        "prior_partial_material": prior["partial_material"],
        "files": _root_file_rows(destination),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    snapshot["snapshot_sha256"] = _canonical_digest(snapshot)
    _write_json(destination / SNAPSHOT_NAME, snapshot)
    return {
        "continuation_id": CONTINUATION_ID,
        "staged_file_count": len(snapshot["files"]),
        "shape_preflight_sha256": shape["preflight_sha256"],
        "snapshot_sha256": snapshot["snapshot_sha256"],
        "held_out_material_copied": False,
        "candidate_freeze_sha256": _sha_file(destination / "candidate-freeze.json"),
    }


def _verify_snapshot(root: Path, prior: dict[str, Any]) -> dict[str, Any]:
    snapshot = _read_json(root / SNAPSHOT_NAME)
    material = {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    if snapshot.get("snapshot_sha256") != _canonical_digest(material):
        raise ValueError("v2 staged snapshot digest is invalid")
    if snapshot.get("prior_v1_stop_sha256") != prior["stop_sha256"]:
        raise ValueError("v2 staged snapshot no longer binds the v1 stop")
    if snapshot.get("prior_partial_material") != prior["partial_material"]:
        raise ValueError("v2 staged snapshot no longer binds prior partial challenge material")
    for row in snapshot.get("files", []):
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"staged v2 base file changed after copy: {row['path']}")
    return snapshot


def _validate_v2_base(root: Path, prior_root: str | Path, *, require_clean: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    prior = _validate_prior_attempt(prior_root)
    base = continuation_v1._assert_stopped_at_unrevealed_boundary(
        root,
        require_no_held_out_material=require_clean,
    )
    snapshot = _verify_snapshot(root, prior)
    preflight = _read_json(root / SHAPE_PREFLIGHT_NAME)
    preflight_material = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if (
        preflight.get("preflight_sha256") != _canonical_digest(preflight_material)
        or preflight.get("all_families_pass") is not True
        or preflight.get("held_out_data_used") is not False
    ):
        raise ValueError("v2 visible activation-layout preflight is invalid")
    runner.verify_packet(root)
    return base, {"prior": prior, "snapshot": snapshot, "preflight": preflight}


def freeze_continuation(custody_root: str | Path, prior_root: str | Path) -> dict[str, Any]:
    root = _custody(custody_root)
    if (root / FREEZE_NAME).exists():
        raise FileExistsError("v2 continuation freeze is immutable")
    base, context = _validate_v2_base(root, prior_root, require_clean=True)
    runtime = runner.verify_runtime()
    prior = context["prior"]
    snapshot = context["snapshot"]
    payload = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": runner.ASSURANCE_OVERLAY,
        "base_packet_sha256": base["packet"]["packet_sha256"],
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "prediction_rules_sha256": _canonical_digest(base["candidate_freeze"]["prediction_rules"]),
        "prior_v1_continuation_freeze_sha256": prior["freeze_sha256"],
        "prior_v1_stop_sha256": prior["stop_sha256"],
        "prior_partial_material": prior["partial_material"],
        "staged_snapshot_sha256": _sha_file(root / SNAPSHOT_NAME),
        "staged_snapshot_digest": snapshot["snapshot_sha256"],
        "shape_preflight_sha256": context["preflight"]["preflight_sha256"],
        "continuation_sources": _source_rows(),
        "runtime": {
            "python": runtime["python"],
            "dependencies": runtime["dependencies"],
            "dependency_manifest_sha256": runtime["dependency_manifest_sha256"],
            "locked_environment_pass": runtime["locked_environment_pass"],
            "cpu_thread_count": runtime["cpu_thread_count"],
        },
        "repair_contract": {
            "read_time_candidate_freeze_alias": True,
            "challenge_capture_layout_restored_to_batch_sequence_hidden": True,
            "valid_activation_values_roundtrip_exactly": True,
            "candidate_refit": False,
            "checkpoint_or_threshold_change": False,
            "prediction_and_outcome_functions_changed": False,
            "prior_partial_challenge_material_reused": False,
            "challenge_entropy_exists_at_freeze": False,
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
        "prior_v1_stop_sha256": payload["prior_v1_stop_sha256"],
        "shape_preflight_sha256": payload["shape_preflight_sha256"],
    }


def verify_continuation(custody_root: str | Path, prior_root: str | Path) -> dict[str, Any]:
    root = _custody(custody_root)
    base, context = _validate_v2_base(root, prior_root, require_clean=False)
    payload = _read_json(root / FREEZE_NAME)
    material = {key: value for key, value in payload.items() if key != "continuation_packet_sha256"}
    if payload.get("continuation_packet_sha256") != _canonical_digest(material):
        raise ValueError("v2 continuation packet digest is invalid")
    prior = context["prior"]
    expected = {
        "continuation_id": CONTINUATION_ID,
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "base_packet_sha256": base["packet"]["packet_sha256"],
        "checkpoint_manifest_sha256": _sha_file(root / "v9-checkpoint-manifest.json"),
        "candidate_preparation_sha256": _sha_file(root / "candidate-preparation.json"),
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "prediction_rules_sha256": _canonical_digest(base["candidate_freeze"]["prediction_rules"]),
        "prior_v1_continuation_freeze_sha256": prior["freeze_sha256"],
        "prior_v1_stop_sha256": prior["stop_sha256"],
        "prior_partial_material": prior["partial_material"],
        "staged_snapshot_sha256": _sha_file(root / SNAPSHOT_NAME),
        "staged_snapshot_digest": context["snapshot"]["snapshot_sha256"],
        "shape_preflight_sha256": context["preflight"]["preflight_sha256"],
        "continuation_sources": _source_rows(),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"v2 continuation packet binding changed: {key}")
    runtime = runner.verify_runtime()
    if (
        payload.get("runtime", {}).get("dependency_manifest_sha256") != runtime["dependency_manifest_sha256"]
        or payload.get("runtime", {}).get("locked_environment_pass") is not True
    ):
        raise ValueError("v2 continuation runtime differs from its frozen environment")
    return payload


def _generate_challenges(root: Path, prior_roots: list[str | Path]) -> dict[str, Any]:
    packet = runner.verify_packet(root)
    runner._verify_candidate_freeze(root, packet["packet_sha256"])
    original_reader = runner._read_json
    original_capture = runner.v3._capture_all_layer_inputs
    target_freeze = (root / "candidate-freeze.json").resolve()

    def compatibility_reader(path: str | Path) -> dict[str, Any]:
        value = original_reader(Path(path))
        if Path(path).expanduser().resolve() == target_freeze:
            return continuation_v1._CandidateFreezeAlias(value)
        return value

    def batch_layout_capture(model: Any, family: str, inputs: dict[str, Any]) -> list[Any]:
        flattened = original_capture(model, family, inputs)
        return [_restore_padded_batch_layout(layer, inputs["attention_mask"]) for layer in flattened]

    runner._read_json = compatibility_reader
    runner.v3._capture_all_layer_inputs = batch_layout_capture
    try:
        return runner.generate_challenges(root, prior_roots)
    finally:
        runner._read_json = original_reader
        runner.v3._capture_all_layer_inputs = original_capture


def continue_primary(custody_root: str | Path, prior_root: str | Path, prior_roots: list[str | Path]) -> dict[str, Any]:
    root = _custody(custody_root)
    packet = verify_continuation(root, prior_root)
    _no_held_out_material(root)
    if (root / PRIMARY_RECEIPT_NAME).exists() or (root / STOP_NAME).exists():
        raise FileExistsError("v2 continuation execution is immutable")
    previous = _validate_prior_attempt(prior_root)
    original_stop = _read_json(root / "primary-run-stop.json")
    runner._ACTIVE_RUN_STARTED = time.monotonic()
    runner._ACTIVE_CASE_STARTED = None
    runner._ACTIVE_PHASE = "challenge_generation"
    started = time.monotonic()
    try:
        challenge = _generate_challenges(root, prior_roots)
        runner._ACTIVE_PHASE = "prediction_sealing"
        prediction = runner.seal_predictions(root)
        runner._ACTIVE_PHASE = "source_outcome_evaluation"
        runner.evaluate_outcomes(root)
        report_path = root / "outcome-manifest.json"
        report = _read_json(report_path)
        v1_stop = previous["stop"]
        prior_elapsed = float(original_stop["elapsed_wall_seconds"]) + float(v1_stop["elapsed_wall_seconds"])
        total_elapsed = prior_elapsed + time.monotonic() - started
        original_runtime = original_stop["runtime"]
        prior_runtime = v1_stop["runtime"]
        current_runtime = report["runtime"]
        peak_rss = max(
            int(original_runtime["peak_rss_bytes"]),
            int(prior_runtime["peak_rss_bytes"]),
            int(current_runtime["peak_rss_bytes"]),
        )
        total_cpu = (
            float(original_runtime["process_cpu_seconds"])
            + float(prior_runtime["process_cpu_seconds"])
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
        report["runtime"]["initial_candidate_preparation_cpu_seconds"] = original_runtime["process_cpu_seconds"]
        report["runtime"]["v1_failed_continuation_cpu_seconds"] = prior_runtime["process_cpu_seconds"]
        report["runtime"]["v2_continuation_cpu_seconds"] = current_runtime["process_cpu_seconds"]
        report["continuation_id"] = CONTINUATION_ID
        report["continuation_packet_sha256"] = packet["continuation_packet_sha256"]
        report["prior_v1_stop_sha256"] = previous["stop_sha256"]
        report["prior_partial_material_reused"] = False
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
            "base_packet_sha256": packet["base_packet_sha256"],
            "candidate_freeze_sha256": packet["candidate_freeze_sha256"],
            "prior_v1_stop_sha256": previous["stop_sha256"],
            "challenge_manifest_sha256": challenge["challenge_manifest_sha256"],
            "prediction_manifest_sha256": prediction["prediction_manifest_sha256"],
            "outcome_report_sha256": report["report_sha256"],
            "outcome_status": report["outcome_status"],
            "assurance_level": report["assurance_level"],
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
            "candidate_refit": False,
            "prior_partial_material_reused": False,
            "held_out_challenges_generated_after_v2_freeze": True,
            "original_outcomes_opened_after_prediction_seal": True,
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
            "prior_v1_stop_sha256": previous["stop_sha256"],
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


def _copy_continuation_context(primary: Path, replica: Path, prior_root: Path) -> None:
    packet_dir = replica / "v9-continuation-v2-packet"
    for relative in SOURCE_FILES:
        source = runner._root() / relative
        target = packet_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source_root, relatives in (
        (primary, (FREEZE_NAME, PRIMARY_RECEIPT_NAME, "primary-run-stop.json", SHAPE_PREFLIGHT_NAME, SNAPSHOT_NAME)),
        (prior_root, (continuation_v1.FREEZE_NAME, continuation_v1.PRIMARY_STOP_NAME)),
    ):
        for relative in relatives:
            source = source_root / relative
            if not source.is_file():
                raise FileNotFoundError(f"continuation context file is missing: {relative}")
            target = replica / "v9-continuation-v2-context" / Path(relative).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def replicate(
    primary_root: str | Path,
    replica_root: str | Path,
    prior_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    primary = _custody(primary_root)
    replica = _custody(replica_root)
    prior = _custody(prior_root)
    packet = verify_continuation(primary, prior)
    primary_receipt = _read_json(primary / PRIMARY_RECEIPT_NAME)
    receipt_material = {key: value for key, value in primary_receipt.items() if key != "continuation_receipt_sha256"}
    if (
        primary_receipt.get("continuation_receipt_sha256") != _canonical_digest(receipt_material)
        or primary_receipt.get("continuation_packet_sha256") != packet["continuation_packet_sha256"]
    ):
        raise ValueError("primary v2 receipt does not bind its frozen continuation")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("v2 replica custody must be empty before replication")
    import replacement_loss_v9_replicator as replicator

    base_result = replicator._run_replica_impl(primary, replica, prior_custody_roots)
    _copy_continuation_context(primary, replica, prior)
    replication_path = primary / "replication-manifest.json"
    replication = _read_json(replication_path)
    base_replication_sha = replication["replication_sha256"]
    replication.pop("replication_sha256", None)
    replication.update({
        "continuation_id": CONTINUATION_ID,
        "continuation_packet_sha256": packet["continuation_packet_sha256"],
        "continuation_primary_receipt_sha256": _sha_file(primary / PRIMARY_RECEIPT_NAME),
        "prior_v1_stop_sha256": packet["prior_v1_stop_sha256"],
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
        "prior_v1_stop_sha256": packet["prior_v1_stop_sha256"],
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
    parser.add_argument("--prior-root")
    parser.add_argument("--primary-root")
    parser.add_argument("--replica-root")
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    if args.command == "stage":
        if not args.source_root or not args.custody_root:
            parser.error("stage requires --source-root and --custody-root")
        result = stage_clean_custody(args.source_root, args.custody_root)
    elif args.command == "freeze":
        if not args.custody_root or not args.prior_root:
            parser.error("freeze requires --custody-root and --prior-root")
        result = freeze_continuation(args.custody_root, args.prior_root)
    elif args.command == "verify":
        if not args.custody_root or not args.prior_root:
            parser.error("verify requires --custody-root and --prior-root")
        payload = verify_continuation(args.custody_root, args.prior_root)
        result = {"continuation_id": payload["continuation_id"], "continuation_packet_sha256": payload["continuation_packet_sha256"]}
    elif args.command == "continue-primary":
        if not args.custody_root or not args.prior_root:
            parser.error("continue-primary requires --custody-root and --prior-root")
        result = continue_primary(args.custody_root, args.prior_root, args.prior_custody_root)
    else:
        if not args.primary_root or not args.replica_root or not args.prior_root:
            parser.error("replicate requires --primary-root, --replica-root, and --prior-root")
        result = replicate(args.primary_root, args.replica_root, args.prior_root, args.prior_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
