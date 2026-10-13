"""Post-run, content-addressed replication audit for replacement-loss V11.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v12-replication-normalization-v1, and
global-experiment-assurance-lifecycle-v1.

This read-only auditor reconciles already completed V11 primary and replica
custodies. It does not rerun model inference, generate fixtures, or modify V11.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v12-replication-normalization-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = "theseus-replacement-loss-v12-replication-normalization-audit"
FREEZE_FILE = "docs/replacement-loss-v12-freeze.json"
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_V11_PROTOCOL_ID = "theseus-replacement-loss-v11-known-opt-causal-fit-remediation"
SOURCE_V11_PACKET_SHA256 = "2bd4e9903b2611ac454a0ff26ffabb051231bd2469f6895aee67504d8c99c582"
SOURCE_V11_TARGET_ID = "grushaprasad/opt125m-rootp-surprisal-curriculum"
SOURCE_V11_TARGET_REVISION = "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
SOURCE_V11_TREE_SHA256 = "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466"
SOURCE_V11_WEIGHT_SHA256 = "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132"
SOURCE_V11_MANIFEST_SHA256 = "b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277"
CLAIM_CEILING = "LOCAL_REPRODUCIBILITY_OF_V11_FAIL_OUTCOME_AND_ARTIFACTS_UNDER_CONTENT_IDENTITY"
RESULT_FILE = "reconciliation-report.json"

V11_PACKET_FILES = (
    "AGENTS.md",
    "docs/replacement-loss-v11.md",
    "docs/replacement-loss-v9.md",
    "docs/replacement-loss-v9-results.md",
    "docs/replacement-loss-v9-freeze.json",
    "replacement_loss_v11_fixtures.py",
    "replacement_loss_v11_runner.py",
    "tests/test_replacement_loss_v11.py",
    "requirements-v11.lock.txt",
    "docs/replacement-loss-v10.md",
    "docs/replacement-loss-v10-freeze.json",
    "replacement_loss_v10_fixtures.py",
    "replacement_loss_v10_runner.py",
    "requirements-v10.lock.txt",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
)

PACKET_FILES = (
    "AGENTS.md",
    "docs/replacement-loss-v12-replication.md",
    "replacement_loss_v12_replication.py",
    "tests/test_replacement_loss_v12_replication.py",
    "requirements-v12.lock.txt",
    *V11_PACKET_FILES,
    "docs/replacement-loss-v11-freeze.json",
    "docs/replacement-loss-v11-results.md",
)

ARTIFACT_GROUP_COUNTS = {
    "fixtures": 4,
    "adaptation": 6,
    "candidates": 12,
    "challenge-entropy": 3,
    "challenges": 9,
    "predictions": 30,
}

PACKAGE_PINS = {
    "huggingface-hub": "0.36.2",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable V12 audit artifact already exists: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _packet_rows(root: Path = REPO_ROOT) -> list[dict[str, str]]:
    rows = []
    for relative in PACKET_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V12 frozen packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def freeze_packet() -> dict[str, Any]:
    destination = REPO_ROOT / FREEZE_FILE
    if destination.exists():
        raise FileExistsError("V12 packet freeze is immutable")
    rows = _packet_rows()
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_V12_POSTHOC_OUTPUT_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packet_files": rows,
        "packet_sha256": _canonical_digest(rows),
        "source_v11_protocol_id": SOURCE_V11_PROTOCOL_ID,
        "source_v11_packet_sha256": SOURCE_V11_PACKET_SHA256,
        "source_v11_results_file_sha256": _sha_file(REPO_ROOT / "docs/replacement-loss-v11-results.md"),
        "source_run_status": "COMPLETED_BEFORE_V12_AUDIT",
        "source_artifact_content_previously_observed": True,
        "new_model_execution": False,
        "new_fixture_generation": False,
        "predeclared_target": {
            "model_id": SOURCE_V11_TARGET_ID,
            "revision": SOURCE_V11_TARGET_REVISION,
            "tree_sha256": SOURCE_V11_TREE_SHA256,
            "weight_file_sha256": SOURCE_V11_WEIGHT_SHA256,
            "fresh_checkpoint_generalization_claimed": False,
        },
        "artifact_group_counts": ARTIFACT_GROUP_COUNTS,
        "identity_rule": "hash file content and relative paths; candidate identity binds logical model revision, source packet, inputs, fitted candidate rows, and stable runtime versions, excluding absolute executable and custody paths",
        "runtime_provenance_rule": "retain Python executable path, platform, package pins, dependency-lock digest, thread count, deterministic mode, host digest, owner, and environment-separation status outside stable candidate identity",
        "pass_conditions": [
            "both V11 phase ledgers report completed prepare, predict, and score phases",
            "V11 frozen packet archive and target-selection receipts verify in both custodies",
            "primary and replica target, candidate content, fixture, challenge, prediction, and scientific-result identities match",
            "every file in the six predeclared artifact groups matches by relative path, byte count, and SHA-256",
            "runtime versions and dependency pins match while executable identities are distinct",
            "the V11 frozen comparator status is preserved as INCONCLUSIVE when its path-bound manifest digests differ",
        ],
        "claim_ceiling": CLAIM_CEILING,
        "resource_limits": {
            "execution": "local_read_only_artifact_audit",
            "max_total_seconds": 1800,
            "max_rss_bytes": 2147483648,
            "paid_or_remote_compute": False,
            "model_inference": False,
            "provider_calls": False,
        },
    }
    _write_json(destination, material, immutable=True)
    return material


def verify_packet(root: Path = REPO_ROOT) -> dict[str, Any]:
    repo = root.resolve()
    payload = _read_json(repo / FREEZE_FILE)
    rows = _packet_rows(repo)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_V12_POSTHOC_OUTPUT_AUDIT"
        or payload.get("packet_files") != rows
        or payload.get("packet_sha256") != _canonical_digest(rows)
        or payload.get("source_v11_packet_sha256") != SOURCE_V11_PACKET_SHA256
        or payload.get("source_v11_results_file_sha256") != _sha_file(repo / "docs/replacement-loss-v11-results.md")
        or payload.get("source_artifact_content_previously_observed") is not True
        or payload.get("new_model_execution") is not False
        or payload.get("new_fixture_generation") is not False
    ):
        raise ValueError("V12 packet freeze verification failed")
    return payload


def _external_root(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        return path
    raise ValueError("V12 custody roots must stay outside the repository")


def archive_packet(custody_value: str | Path) -> dict[str, Any]:
    custody = _external_root(custody_value)
    packet = verify_packet()
    if custody.exists() and any(custody.iterdir()):
        raise FileExistsError("V12 archive root is not empty; preserve it and choose a new root")
    custody.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in (*PACKET_FILES, FREEZE_FILE):
        source = REPO_ROOT / relative
        target = custody / "frozen-packet" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "bytes": target.stat().st_size, "sha256": _sha_file(target)})
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "archive_files": rows,
    }
    manifest = {**material, "archive_sha256": _canonical_digest(material)}
    _write_json(custody / "frozen-packet-manifest.json", manifest, immutable=True)
    return manifest


def _verify_v12_archive(custody: Path, packet: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(custody / "frozen-packet-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    if manifest.get("archive_sha256") != _canonical_digest(material) or manifest.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("V12 external packet archive manifest failed digest verification")
    expected_paths = {*PACKET_FILES, FREEZE_FILE}
    if {row.get("path") for row in manifest.get("archive_files", [])} != expected_paths:
        raise ValueError("V12 external packet archive inventory differs")
    for row in manifest["archive_files"]:
        path = custody / "frozen-packet" / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V12 external packet archive bytes differ: {row['path']}")
    return manifest


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("source checkpoint tree must not contain symbolic links")
        if not path.is_file() or path.name.endswith(".lock") or ".cache" in path.parts:
            continue
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return _canonical_digest(rows), rows


def _verify_v11_packet_and_archive(primary: Path, replica: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    v11_freeze = _read_json(REPO_ROOT / "docs/replacement-loss-v11-freeze.json")
    rows = []
    for item in v11_freeze.get("packet_files", []):
        path = REPO_ROOT / item["path"]
        rows.append({"path": item["path"], "sha256": _sha_file(path)})
    if (
        v11_freeze.get("protocol_id") != SOURCE_V11_PROTOCOL_ID
        or v11_freeze.get("packet_sha256") != SOURCE_V11_PACKET_SHA256
        or v11_freeze.get("packet_files") != rows
    ):
        raise ValueError("the source V11 frozen packet no longer verifies")

    expected_archive_paths = {*V11_PACKET_FILES, "docs/replacement-loss-v11-freeze.json"}
    expected_archive_digests = {
        relative: _sha_file(REPO_ROOT / relative)
        for relative in expected_archive_paths
    }
    manifests = []
    for root in (primary, replica):
        manifest = _read_json(root / "frozen-packet-manifest.json")
        material = {key: value for key, value in manifest.items() if key != "archive_sha256"}
        if manifest.get("archive_sha256") != _canonical_digest(material) or manifest.get("packet_sha256") != SOURCE_V11_PACKET_SHA256:
            raise ValueError("source V11 archive manifest failed digest verification")
        if {row.get("path") for row in manifest.get("archive_files", [])} != expected_archive_paths:
            raise ValueError("source V11 archive file inventory differs")
        for row in manifest["archive_files"]:
            path = root / "frozen-packet" / row["path"]
            if (
                not path.is_file()
                or path.stat().st_size != row["bytes"]
                or _sha_file(path) != row["sha256"]
                or row["sha256"] != expected_archive_digests[row["path"]]
            ):
                raise ValueError(f"source V11 archive bytes differ: {row['path']}")
        manifests.append(manifest)
    if manifests[0]["archive_sha256"] != manifests[1]["archive_sha256"]:
        raise ValueError("source V11 archive digests differ across custodies")
    return v11_freeze, manifests[0]


def _verify_target_selection(root: Path, v11_freeze: dict[str, Any], v11_archive: dict[str, Any]) -> dict[str, Any]:
    selection = _read_json(root / "checkpoint-selection.json")
    selection_material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("selection_sha256") != _canonical_digest(selection_material)
        or selection.get("protocol_id") != SOURCE_V11_PROTOCOL_ID
        or selection.get("packet_sha256") != SOURCE_V11_PACKET_SHA256
        or selection.get("packet_archive_sha256") != v11_archive["archive_sha256"]
        or selection.get("target_predeclared_before_freeze") is not True
        or selection.get("target_reused_from_v9") is not True
        or selection.get("selection_performed") is not False
        or selection.get("model_id") != SOURCE_V11_TARGET_ID
        or selection.get("revision") != SOURCE_V11_TARGET_REVISION
        or selection.get("source_manifest_sha256") != SOURCE_V11_MANIFEST_SHA256
        or selection.get("source_artifact_tree_sha256") != SOURCE_V11_TREE_SHA256
    ):
        raise ValueError("source V11 target-selection receipt failed verification")

    source_manifest_path = Path(selection["source_manifest_path"]).expanduser().resolve()
    if _sha_file(source_manifest_path) != SOURCE_V11_MANIFEST_SHA256:
        raise ValueError("source V9 checkpoint manifest digest differs")
    source_manifest = _read_json(source_manifest_path)
    if source_manifest.get("checkpoint_manifest_sha256") != v11_freeze["reused_target"]["source_checkpoint_manifest_canonical_sha256"]:
        raise ValueError("source V9 checkpoint-manifest identity differs")
    matches = [row for row in source_manifest.get("checkpoints", []) if row.get("family") == "opt_relu"]
    if len(matches) != 1:
        raise ValueError("source V9 checkpoint manifest does not contain exactly one OPT target")
    source_row = matches[0]
    if (
        source_row.get("model_id") != SOURCE_V11_TARGET_ID
        or source_row.get("revision") != SOURCE_V11_TARGET_REVISION
        or source_row.get("license") != "apache-2.0"
        or selection.get("license_id") != "apache-2.0"
    ):
        raise ValueError("source V9 checkpoint identity differs")
    model_dir = (source_manifest_path.parent / source_row["local_path"]).resolve()
    tree_sha, tree_rows = _tree_digest(model_dir)
    if tree_sha != SOURCE_V11_TREE_SHA256 or tree_sha != selection["source_artifact_tree_sha256"]:
        raise ValueError("source V9 checkpoint tree digest differs")
    weight_rows = [row for row in tree_rows if Path(row["path"]).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}]
    if (
        len(weight_rows) != 1
        or weight_rows[0]["path"] != "pytorch_model.bin"
        or weight_rows[0]["sha256"] != SOURCE_V11_WEIGHT_SHA256
        or _canonical_digest(weight_rows) != selection["weights_sha256"]
        or model_dir != Path(selection["local_path"]).resolve()
    ):
        raise ValueError("source V9 model weight or custody path binding differs")
    for name, expected_sha in (("config.json", selection["config_sha256"]), ("README.md", selection["model_card_sha256"])):
        if _sha_file(model_dir / name) != expected_sha:
            raise ValueError(f"source checkpoint {name} digest differs")
    return selection


def _runtime_rows(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    ledger = _read_json(root / "phase-ledger.json")
    phases = ledger.get("phases", [])
    by_name = {row.get("phase"): row for row in phases}
    if set(by_name) != {"prepare", "predict", "score"} or any(by_name[name].get("status") != "COMPLETED" for name in by_name):
        raise ValueError("V11 run did not complete each frozen phase exactly once")
    elapsed = float(ledger.get("total_elapsed_seconds", float("inf")))
    peak_rss = max(int(row.get("peak_rss_bytes", 0)) for row in phases)
    if elapsed > 4 * 60 * 60 or peak_rss > 16 * 1024**3:
        raise ValueError("V11 run exceeded its frozen execution budget")
    runtimes = [row.get("runtime", {}) for row in phases]
    runtime = runtimes[0]
    if not runtime or any(row != runtime for row in runtimes[1:]):
        raise ValueError("runtime provenance changed across V11 phases")
    if (
        runtime.get("python") != "3.12.13"
        or runtime.get("torch_num_threads") != 1
        or runtime.get("deterministic_algorithms") is not True
        or runtime.get("packages") != PACKAGE_PINS
    ):
        raise ValueError("V11 execution runtime differs from the predeclared pinned environment")
    expected_lock_sha = _sha_file(REPO_ROOT / "requirements-v10.lock.txt")
    if runtime.get("dependency_lock_sha256") != expected_lock_sha:
        raise ValueError("V11 imported engine dependency-lock digest differs")
    return runtime, {"elapsed_seconds": elapsed, "peak_rss_bytes": peak_rss}


def stable_candidate_content(freeze: dict[str, Any], selection: dict[str, Any]) -> str:
    runtime = freeze["runtime"]
    stable_runtime = {
        key: runtime[key]
        for key in (
            "python",
            "platform",
            "packages",
            "dependency_lock_sha256",
            "torch_num_threads",
            "deterministic_algorithms",
        )
    }
    target_identity = {
        "model_id": selection["model_id"],
        "revision": selection["revision"],
        "resolved_revision": selection["resolved_revision"],
        "tree_sha256": selection["source_artifact_tree_sha256"],
        "config_sha256": selection["config_sha256"],
        "weights_sha256": selection["weights_sha256"],
        "weight_files": selection["weight_files"],
    }
    material = {
        "protocol_id": freeze["protocol_id"],
        "state_slice": freeze["state_slice"],
        "protocol_slice": freeze["protocol_slice"],
        "packet_sha256": freeze["packet_sha256"],
        "target_content_identity": target_identity,
        "visible_fixture_sha256": freeze["visible_fixture_sha256"],
        "visible_selection_only": freeze["visible_selection_only"],
        "selected_layers": freeze["selected_layers"],
        "visible_layer_variance": freeze["visible_layer_variance"],
        "visible_input_rms": freeze["visible_input_rms"],
        "adaptation_stress_coordinate": freeze["adaptation_stress_coordinate"],
        "layer_dimensions": freeze["layer_dimensions"],
        "adaptation_fixture_sha256": freeze["adaptation_fixture_sha256"],
        "adaptation_artifacts": freeze["adaptation_artifacts"],
        "candidate_count": freeze["candidate_count"],
        "candidates": freeze["candidates"],
        "causal_fit_updates": freeze["causal_fit_updates"],
        "source_revision": freeze["source_revision"],
        "stable_runtime": stable_runtime,
    }
    return _canonical_digest(material)


def stable_challenge_content(manifest: dict[str, Any], candidate_content_sha256: str) -> str:
    material = {
        "protocol_id": manifest["protocol_id"],
        "state_slice": manifest["state_slice"],
        "protocol_slice": manifest["protocol_slice"],
        "packet_sha256": manifest["packet_sha256"],
        "candidate_content_sha256": candidate_content_sha256,
        "challenge_generation_started_after_candidate_freeze": manifest["challenge_generation_started_after_candidate_freeze"],
        "entropy": manifest["entropy"],
        "challenge_rows": manifest["challenge_rows"],
    }
    return _canonical_digest(material)


def stable_prediction_content(
    manifest: dict[str, Any], candidate_content_sha256: str, challenge_content_sha256: str
) -> str:
    material = {
        "protocol_id": manifest["protocol_id"],
        "state_slice": manifest["state_slice"],
        "protocol_slice": manifest["protocol_slice"],
        "packet_sha256": manifest["packet_sha256"],
        "candidate_content_sha256": candidate_content_sha256,
        "challenge_content_sha256": challenge_content_sha256,
        "predictions": manifest["predictions"],
        "original_model_outcomes_opened": manifest["original_model_outcomes_opened"],
    }
    return _canonical_digest(material)


def _artifact_rows(root: Path, group: str) -> list[dict[str, Any]]:
    directory = root / group
    if not directory.is_dir():
        raise FileNotFoundError(f"V11 artifact group is missing: {group}")
    rows = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"V11 artifact group contains a symbolic link: {group}")
        if path.is_file():
            rows.append({"path": path.relative_to(directory).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    if len(rows) != ARTIFACT_GROUP_COUNTS[group]:
        raise ValueError(f"V11 artifact group count differs from the frozen audit rule: {group}")
    return rows


def _verify_candidate_manifest(root: Path, packet_sha256: str, selection: dict[str, Any]) -> dict[str, Any]:
    freeze = _read_json(root / "candidate-freeze.json")
    material = {key: value for key, value in freeze.items() if key not in {"candidate_set_sha256", "created_at_utc"}}
    if (
        freeze.get("candidate_set_sha256") != _canonical_digest(material)
        or freeze.get("packet_sha256") != packet_sha256
        or freeze.get("checkpoint_selection_sha256") != selection.get("selection_sha256")
        or int(freeze.get("candidate_count", -1)) != 12
        or freeze.get("selected_layers") != [0, 7]
    ):
        raise ValueError("V11 candidate freeze failed its internal digest or target binding")
    return freeze


def _verify_failure_cell_summary(report: dict[str, Any]) -> str:
    targets = ((0, "ordinary"), (7, "ordinary"), (0, "adversarial"))
    lookup = {(int(row["seed_index"]), int(row["layer"]), row["method"]): row for row in report["results"]}
    cells = []
    per_seed = []
    for seed_index in range(3):
        gains = []
        for layer, stratum in targets:
            treatment = lookup[(seed_index, layer, "causal_fit")]["per_stratum"][stratum]["next_logits_causal_effect"]
            baseline = lookup[(seed_index, layer, "activation_ridge")]["per_stratum"][stratum]["next_logits_causal_effect"]
            treatment_cosine = treatment.get("median_cosine")
            baseline_cosine = baseline.get("median_cosine")
            gain = float(treatment_cosine or 0.0) - float(baseline_cosine or 0.0)
            gains.append(gain)
            cells.append({
                "seed_index": seed_index,
                "layer": layer,
                "stratum": stratum,
                "causal_fit_median_cosine": treatment_cosine,
                "causal_fit_gate_pass": _cell_gate(treatment),
                "activation_ridge_median_cosine": baseline_cosine,
                "activation_ridge_gate_pass": _cell_gate(baseline),
                "paired_cosine_gain": gain,
            })
        per_seed.append({"seed_index": seed_index, "mean_paired_cosine_gain": sum(gains) / len(gains)})
    summary = {
        "endpoint": "V9 previously failed next-logit causal-effect cells: ordinary at layers 0 and 7; adversarial at layer 0",
        "role": "predeclared secondary descriptive endpoint; not an additional PASS gate",
        "causal_fit_cells_passing": sum(bool(row["causal_fit_gate_pass"]) for row in cells),
        "activation_ridge_cells_passing": sum(bool(row["activation_ridge_gate_pass"]) for row in cells),
        "cells": cells,
        "per_seed_mean_paired_cosine_gain": per_seed,
    }
    if report.get("prior_failure_cell_summary") != summary:
        raise ValueError("V11 prior-failure summary does not recompute")
    digest = _canonical_digest(summary)
    if report.get("prior_failure_cell_summary_sha256") != digest:
        raise ValueError("V11 prior-failure summary digest differs")
    return digest


def _cell_gate(metric: dict[str, Any]) -> bool:
    return (
        int(metric.get("resolvable_count", 0)) >= 20
        and float(metric.get("direction_accuracy", 0.0)) >= 0.80
        and float(metric.get("median_cosine", 0.0) or 0.0) >= 0.90
        and float(metric.get("median_normalized_error", float("inf"))) <= 0.20
    )


def _verify_outcome(root: Path, packet_sha256: str) -> tuple[dict[str, Any], str, str]:
    report = _read_json(root / "outcome-report.json")
    scientific_keys = (
        "scientific_outcome", "primary_comparison", "causal_fit_retention_pass",
        "activation_rescue_pass", "zero_intervention_control_pass", "results", "controls",
    )
    summary = {key: report[key] for key in scientific_keys}
    scientific_digest = _canonical_digest(summary)
    report_material = {key: value for key, value in report.items() if key not in {"outcome_report_sha256", "created_at_utc"}}
    if (
        report.get("packet_sha256") != packet_sha256
        or report.get("scientific_results_sha256") != scientific_digest
        or report.get("outcome_report_sha256") != _canonical_digest(report_material)
        or report.get("scientific_outcome") != "FAIL"
        or report.get("claim_ceiling") != "LOCAL_CAUSAL_FIT_REMEDIATION_ON_REUSED_V9_OPT_CHECKPOINT_WITH_NEW_SYNTHETIC_FIXTURES"
        or report.get("target_reuse", {}).get("fresh_checkpoint_generalization_claimed") is not False
    ):
        raise ValueError("V11 outcome report failed its internal digest, packet, or claim binding")
    failure_summary_digest = _verify_failure_cell_summary(report)
    return report, scientific_digest, failure_summary_digest


def _verify_challenge_and_prediction(
    root: Path, packet_sha256: str, freeze: dict[str, Any], candidate_content_sha256: str
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    challenge = _read_json(root / "challenge-manifest.json")
    challenge_material = {key: value for key, value in challenge.items() if key not in {"challenge_manifest_sha256", "created_at_utc"}}
    if (
        challenge.get("challenge_manifest_sha256") != _canonical_digest(challenge_material)
        or challenge.get("packet_sha256") != packet_sha256
        or challenge.get("candidate_set_sha256") != freeze.get("candidate_set_sha256")
        or challenge.get("challenge_generation_started_after_candidate_freeze") is not True
    ):
        raise ValueError("V11 challenge manifest failed its internal binding")
    challenge_content_sha = stable_challenge_content(challenge, candidate_content_sha256)

    prediction = _read_json(root / "prediction-manifest.json")
    prediction_material = {key: value for key, value in prediction.items() if key not in {"prediction_manifest_sha256", "created_at_utc"}}
    if (
        prediction.get("prediction_manifest_sha256") != _canonical_digest(prediction_material)
        or prediction.get("packet_sha256") != packet_sha256
        or prediction.get("candidate_set_sha256") != freeze.get("candidate_set_sha256")
        or prediction.get("challenge_manifest_sha256") != challenge.get("challenge_manifest_sha256")
        or prediction.get("original_model_outcomes_opened") is not False
        or len(prediction.get("predictions", [])) != 30
    ):
        raise ValueError("V11 prediction manifest failed its internal binding")
    prediction_content_sha = stable_prediction_content(prediction, candidate_content_sha256, challenge_content_sha)
    return challenge, prediction, challenge_content_sha, prediction_content_sha


def _runtime_summary(runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "python": runtime["python"],
        "python_executable_path_sha256": hashlib.sha256(runtime["python_executable"].encode("utf-8")).hexdigest(),
        "platform": runtime["platform"],
        "host_id_sha256": runtime["host_id_sha256"],
        "packages": runtime["packages"],
        "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        "torch_num_threads": runtime["torch_num_threads"],
        "deterministic_algorithms": runtime["deterministic_algorithms"],
    }


def audit(v11_primary_value: str | Path, v11_replica_value: str | Path, audit_custody_value: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    audit_custody = _external_root(audit_custody_value)
    _verify_v12_archive(audit_custody, packet)
    if (audit_custody / RESULT_FILE).exists():
        raise FileExistsError("V12 reconciliation report is immutable; use an empty audit custody root")

    primary = _external_root(v11_primary_value)
    replica = _external_root(v11_replica_value)
    if primary == replica:
        raise ValueError("V11 primary and replica custody roots must be distinct")
    v11_freeze, v11_archive = _verify_v11_packet_and_archive(primary, replica)
    primary_selection = _verify_target_selection(primary, v11_freeze, v11_archive)
    replica_selection = _verify_target_selection(replica, v11_freeze, v11_archive)
    if primary_selection.get("selection_sha256") != replica_selection.get("selection_sha256"):
        raise ValueError("V11 target-selection receipts differ")

    runtimes = []
    run_budgets = []
    freezes = []
    challenge_manifests = []
    prediction_manifests = []
    stable_candidates = []
    stable_challenges = []
    stable_predictions = []
    outcome_reports = []
    scientific_digests = []
    failure_digests = []
    for root, selection in ((primary, primary_selection), (replica, replica_selection)):
        runtime, budget = _runtime_rows(root)
        runtimes.append(runtime)
        run_budgets.append(budget)
        freeze = _verify_candidate_manifest(root, SOURCE_V11_PACKET_SHA256, selection)
        freezes.append(freeze)
        stable_candidates.append(stable_candidate_content(freeze, selection))
        challenge, prediction, challenge_sha, prediction_sha = _verify_challenge_and_prediction(
            root, SOURCE_V11_PACKET_SHA256, freeze, stable_candidates[-1]
        )
        challenge_manifests.append(challenge)
        prediction_manifests.append(prediction)
        stable_challenges.append(challenge_sha)
        stable_predictions.append(prediction_sha)
        report, scientific_sha, failure_sha = _verify_outcome(root, SOURCE_V11_PACKET_SHA256)
        outcome_reports.append(report)
        scientific_digests.append(scientific_sha)
        failure_digests.append(failure_sha)

    groups = {}
    groups_equal = True
    for group, expected_count in ARTIFACT_GROUP_COUNTS.items():
        primary_rows = _artifact_rows(primary, group)
        replica_rows = _artifact_rows(replica, group)
        identical = primary_rows == replica_rows
        groups_equal = groups_equal and identical
        groups[group] = {
            "expected_file_count": expected_count,
            "primary_file_count": len(primary_rows),
            "replica_file_count": len(replica_rows),
            "primary_tree_sha256": _canonical_digest(primary_rows),
            "replica_tree_sha256": _canonical_digest(replica_rows),
            "byte_identical": identical,
            "files": primary_rows,
        }

    official_report = _read_json(primary / "replication-report.json")
    if (
        official_report.get("assurance_level") != "INCONCLUSIVE"
        or official_report.get("exact_identity_and_output_digests_match") is not False
        or official_report.get("same_owner") is not True
        or official_report.get("separate_python_environment") is not True
        or freezes[0].get("candidate_set_sha256") == freezes[1].get("candidate_set_sha256")
        or freezes[0].get("candidates") != freezes[1].get("candidates")
        or official_report.get("digest_comparisons", {}).get("candidate_set_sha256")
        != [freezes[0].get("candidate_set_sha256"), freezes[1].get("candidate_set_sha256")]
    ):
        raise ValueError("the original V11 comparator finding differs from the predeclared audit case")

    executable_ids = [runtime.get("python_executable", "") for runtime in runtimes]
    separate_runtime = bool(executable_ids[0] and executable_ids[1] and executable_ids[0] != executable_ids[1])
    same_owner = primary.stat().st_uid == replica.stat().st_uid
    same_host = runtimes[0].get("host_id_sha256") == runtimes[1].get("host_id_sha256")
    comparable_runtime = all(
        runtimes[0].get(key) == runtimes[1].get(key)
        for key in ("python", "platform", "packages", "dependency_lock_sha256", "torch_num_threads", "deterministic_algorithms")
    )
    content_equal = (
        groups_equal
        and stable_candidates[0] == stable_candidates[1]
        and stable_challenges[0] == stable_challenges[1]
        and stable_predictions[0] == stable_predictions[1]
        and scientific_digests[0] == scientific_digests[1]
        and failure_digests[0] == failure_digests[1]
        and outcome_reports[0]["scientific_outcome"] == outcome_reports[1]["scientific_outcome"] == "FAIL"
    )
    reproduced = content_equal and separate_runtime and same_owner and same_host and comparable_runtime

    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "assurance_scope": "content-addressed same-owner local reproduction of the completed V11 FAIL outcome and artifacts",
        "reviewer_status": "NOT_SUBMITTED",
        "source_v11": {
            "protocol_id": SOURCE_V11_PROTOCOL_ID,
            "packet_sha256": SOURCE_V11_PACKET_SHA256,
            "archive_sha256": v11_archive["archive_sha256"],
            "original_comparator_assurance": official_report["assurance_level"],
            "original_comparator_exact_digest_match": official_report["exact_identity_and_output_digests_match"],
            "original_comparator_candidate_set_sha256": official_report["digest_comparisons"]["candidate_set_sha256"],
            "scientific_outcome": outcome_reports[0]["scientific_outcome"],
            "claim_ceiling": outcome_reports[0]["claim_ceiling"],
        },
        "target_identity": {
            "model_id": primary_selection["model_id"],
            "revision": primary_selection["revision"],
            "tree_sha256": primary_selection["source_artifact_tree_sha256"],
            "weights_sha256": primary_selection["weights_sha256"],
            "selection_sha256": primary_selection["selection_sha256"],
        },
        "content_identities": {
            "candidate_content_sha256": stable_candidates,
            "challenge_content_sha256": stable_challenges,
            "prediction_content_sha256": stable_predictions,
            "scientific_results_sha256": scientific_digests,
            "prior_failure_cell_summary_sha256": failure_digests,
            "candidate_content_match": stable_candidates[0] == stable_candidates[1],
            "challenge_content_match": stable_challenges[0] == stable_challenges[1],
            "prediction_content_match": stable_predictions[0] == stable_predictions[1],
            "scientific_results_match": scientific_digests[0] == scientific_digests[1],
            "prior_failure_summary_match": failure_digests[0] == failure_digests[1],
        },
        "artifact_groups": groups,
        "runtime_provenance": {
            "primary": _runtime_summary(runtimes[0]),
            "replica": _runtime_summary(runtimes[1]),
            "runtime_versions_and_pins_match": comparable_runtime,
            "separate_python_executable_paths": separate_runtime,
            "same_host": same_host,
            "same_owner": same_owner,
            "primary_run_budget": run_budgets[0],
            "replica_run_budget": run_budgets[1],
        },
        "decision_conditions": {
            "all_artifact_groups_byte_identical": groups_equal,
            "stable_content_identities_match": stable_candidates[0] == stable_candidates[1] and stable_challenges[0] == stable_challenges[1] and stable_predictions[0] == stable_predictions[1],
            "scientific_and_secondary_summaries_match": scientific_digests[0] == scientific_digests[1] and failure_digests[0] == failure_digests[1],
            "separate_runtime_same_owner_same_host": separate_runtime and same_owner and same_host,
            "runtime_versions_and_pins_match": comparable_runtime,
            "source_v11_status_preserved": official_report["assurance_level"] == "INCONCLUSIVE",
        },
        "claim_ceiling": CLAIM_CEILING,
        "nonclaims": [
            "this post-run reconciliation is not blind or independent external replication",
            "the source V11 scientific outcome remains FAIL",
            "no fresh-checkpoint or architecture-general claim is made",
            "no novelty, consciousness, alignment, human-equivalence, independent-science, or production claim is made",
        ],
        "auditor_runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    _write_json(audit_custody / RESULT_FILE, report, immutable=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    auditor = commands.add_parser("audit")
    auditor.add_argument("--v11-primary", required=True)
    auditor.add_argument("--v11-replica", required=True)
    auditor.add_argument("--custody", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    else:
        result = audit(args.v11_primary, args.v11_replica, args.custody)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
