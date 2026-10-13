"""Failure-informed, target-reuse causal-fit experiment runner.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v11-known-opt-causal-fit-v1, and
global-experiment-assurance-lifecycle-v1. The pinned V9 model remains in its
existing external custody; this runner writes only V11 artifacts to the
external custody path supplied on the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

import replacement_loss_v10_runner as engine
import replacement_loss_v11_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v11-known-opt-causal-fit-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v11-freeze.json"
REPO_ROOT = Path(__file__).resolve().parent
SOURCE_PROTOCOL_ID = "theseus-replacement-loss-v9-causal-ladder"
SOURCE_CHECKPOINT_MANIFEST_SHA256 = "b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277"
SOURCE_CHECKPOINT_CANONICAL_SHA256 = "dc7e198ac55a27b3cffc1a45b9bb73b7006afc2e2bd689403e223eda1cb1dfc7"
SOURCE_PACKET_SHA256 = "9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959"
TARGET_MODEL_ID = "grushaprasad/opt125m-rootp-surprisal-curriculum"
TARGET_REVISION = "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
TARGET_WEIGHT_FILE_SHA256 = "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132"
TARGET_WEIGHT_BYTES = 501020637
TARGET_TREE_SHA256 = "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466"
CLAIM_CEILING = "LOCAL_CAUSAL_FIT_REMEDIATION_ON_REUSED_V9_OPT_CHECKPOINT_WITH_NEW_SYNTHETIC_FIXTURES"
PACKET_FILES = (
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

_ORIGINAL_WRITE_JSON = engine._write_json
_ORIGINAL_VISIBLE_SELECTION = engine._visible_selection
_REPLICATION_PAIR: tuple[Path, Path] | None = None
_CONFIGURED = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return engine._canonical_digest(value)


def _packet_material(root: Path) -> list[dict[str, str]]:
    rows = []
    for relative in PACKET_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V11 packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def verify_packet(root: Path | None = None) -> dict[str, Any]:
    repo = (root or REPO_ROOT).resolve()
    payload = json.loads((repo / FREEZE_FILE).read_text(encoding="utf-8"))
    rows = _packet_material(repo)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_NEW_FIXTURES_AND_FITTING"
        or payload.get("packet_files") != rows
        or payload.get("packet_sha256") != _canonical_digest(rows)
        or payload.get("target_predeclared_before_freeze") is not True
        or payload.get("challenge_fixtures_generated") is not False
    ):
        raise ValueError("V11 packet freeze verification failed")
    target = payload.get("reused_target", {})
    if target.get("model_id") != TARGET_MODEL_ID or target.get("revision") != TARGET_REVISION:
        raise ValueError("V11 predeclared reused target identity changed")
    return payload


def freeze_packet() -> dict[str, Any]:
    destination = REPO_ROOT / FREEZE_FILE
    if destination.exists():
        raise FileExistsError("V11 packet freeze is immutable")
    rows = _packet_material(REPO_ROOT)
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_NEW_FIXTURES_AND_FITTING",
        "created_at_utc": _now(),
        "packet_files": rows,
        "packet_sha256": _canonical_digest(rows),
        "target_predeclared_before_freeze": True,
        "reused_target": {
            "source_protocol_id": SOURCE_PROTOCOL_ID,
            "source_packet_sha256": SOURCE_PACKET_SHA256,
            "source_checkpoint_manifest_file_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA256,
            "source_checkpoint_manifest_canonical_sha256": SOURCE_CHECKPOINT_CANONICAL_SHA256,
            "model_id": TARGET_MODEL_ID,
            "revision": TARGET_REVISION,
            "raw_weight_file_sha256": TARGET_WEIGHT_FILE_SHA256,
            "checkpoint_tree_sha256": TARGET_TREE_SHA256,
            "fresh_checkpoint_generalization_claimed": False,
        },
        "fixture_seed_count": len(fixtures.SEEDS),
        "methods": ["activation_ridge", "causal_fit"],
        "selected_layers": [0, 7],
        "claim_ceiling": CLAIM_CEILING,
        "challenge_fixtures_generated": False,
        "resource_limits": {
            "execution": "local_cpu_only",
            "max_total_seconds": engine.MAX_TOTAL_SECONDS,
            "max_rss_bytes": engine.MAX_RSS_BYTES,
            "max_parameters": engine.MAX_PARAMETERS,
            "max_weight_bytes": engine.MAX_WEIGHT_BYTES,
            "paid_or_remote_compute": False,
        },
    }
    _ORIGINAL_WRITE_JSON(destination, material, immutable=True)
    return material


def _source_checkpoint(source_manifest_path: str | Path) -> dict[str, Any]:
    manifest_path = Path(source_manifest_path).expanduser().resolve()
    raw_sha = _sha_file(manifest_path)
    if raw_sha != SOURCE_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("V9 source checkpoint manifest digest differs from the predeclared target")
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        source_manifest.get("protocol_id") != SOURCE_PROTOCOL_ID
        or source_manifest.get("packet_sha256") != SOURCE_PACKET_SHA256
        or source_manifest.get("checkpoint_manifest_sha256") != SOURCE_CHECKPOINT_CANONICAL_SHA256
    ):
        raise ValueError("V9 checkpoint manifest identity or packet binding differs")
    matches = [row for row in source_manifest.get("checkpoints", []) if row.get("family") == "opt_relu"]
    if len(matches) != 1:
        raise ValueError("V9 manifest must contain exactly one OPT/ReLU target")
    row = matches[0]
    if row.get("model_id") != TARGET_MODEL_ID or row.get("revision") != TARGET_REVISION or row.get("license") != "apache-2.0":
        raise ValueError("V9 source manifest does not describe the frozen reused target")
    model_dir = (manifest_path.parent / row["local_path"]).resolve()
    tree_sha, file_rows = engine.v3._tree_digest(model_dir)
    if tree_sha != TARGET_TREE_SHA256 or tree_sha != row.get("artifact_tree_sha256") or len(file_rows) != row.get("artifact_file_count"):
        raise ValueError("existing V9 checkpoint tree failed its frozen digest")
    weight_rows = [item for item in file_rows if Path(item["path"]).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}]
    if (
        len(weight_rows) != 1
        or weight_rows[0]["path"] != "pytorch_model.bin"
        or weight_rows[0]["sha256"] != TARGET_WEIGHT_FILE_SHA256
        or weight_rows[0]["bytes"] != TARGET_WEIGHT_BYTES
        or _canonical_digest(weight_rows) != row.get("weights_sha256")
    ):
        raise ValueError("existing V9 model-weight bytes differ from the frozen reused target")
    config_path = model_dir / "config.json"
    card_path = model_dir / "README.md"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    card_text = card_path.read_text(encoding="utf-8")
    frontmatter = card_text.split("---", 2)
    card_metadata = yaml.safe_load(frontmatter[1]) if len(frontmatter) == 3 else {}
    if (
        config.get("model_type") != "opt"
        or config.get("activation_function") != "relu"
        or int(config.get("num_hidden_layers", 0)) != 12
        or int(config.get("max_position_embeddings", 0)) < engine.MAX_SEQUENCE_LENGTH
        or int(config.get("vocab_size", 0)) <= 0
        or str((card_metadata or {}).get("license", "")).casefold() != "apache-2.0"
    ):
        raise ValueError("reused target no longer satisfies its recorded OPT/ReLU or pinned license metadata")
    return {
        "source_manifest_path": str(manifest_path),
        "source_manifest_sha256": raw_sha,
        "source_checkpoint_manifest_sha256": source_manifest["checkpoint_manifest_sha256"],
        "source_artifact_tree_sha256": tree_sha,
        "model_id": TARGET_MODEL_ID,
        "revision": TARGET_REVISION,
        "resolved_revision": TARGET_REVISION,
        "local_path": str(model_dir),
        "license_id": "apache-2.0",
        "license_source_url": row["license_evidence_url"],
        "model_visibility": "public_ungated_recorded_by_v9",
        "model_card_url": f"https://huggingface.co/{TARGET_MODEL_ID}/blob/{TARGET_REVISION}/README.md",
        "model_card_sha256": _sha_file(card_path),
        "model_card_metadata": card_metadata or {},
        "model_card_metadata_sha256": _canonical_digest(card_metadata or {}),
        "config_sha256": _sha_file(config_path),
        "config_architecture": {
            "model_type": config["model_type"],
            "activation_function": config["activation_function"],
            "num_hidden_layers": int(config["num_hidden_layers"]),
            "hidden_size": int(config["hidden_size"]),
            "ffn_dim": int(config["ffn_dim"]),
            "vocab_size": int(config["vocab_size"]),
        },
        "weight_files": weight_rows,
        "total_weight_bytes": sum(int(item["bytes"]) for item in weight_rows),
        "weights_sha256": _canonical_digest(weight_rows),
    }


def bind_reused_target(custody_path: str | Path, source_manifest_path: str | Path) -> dict[str, Any]:
    custody = engine._custody_root(custody_path)
    if (custody / "checkpoint-selection.json").exists():
        raise FileExistsError("V11 reused-target binding is immutable")
    packet = verify_packet()
    archive = engine.archive_packet(custody)
    target = _source_checkpoint(source_manifest_path)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "packet_archive_sha256": archive["archive_sha256"],
        "target_predeclared_before_freeze": True,
        "target_reused_from_v9": True,
        "selection_performed": False,
        "verification_completed_at_utc": _now(),
        **target,
    }
    payload["selection_sha256"] = _canonical_digest(payload)
    _ORIGINAL_WRITE_JSON(custody / "checkpoint-selection.json", payload, immutable=True)
    return payload


def _checkpoint_manifest(custody: Path) -> dict[str, Any]:
    packet = verify_packet()
    selection = json.loads((custody / "checkpoint-selection.json").read_text(encoding="utf-8"))
    archive = engine._verify_packet_archive(custody)
    material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("packet_sha256") != packet["packet_sha256"]
        or selection.get("packet_archive_sha256") != archive["archive_sha256"]
        or selection.get("target_predeclared_before_freeze") is not True
        or selection.get("target_reused_from_v9") is not True
        or selection.get("selection_performed") is not False
        or selection.get("model_id") != TARGET_MODEL_ID
        or selection.get("revision") != TARGET_REVISION
        or selection.get("source_manifest_sha256") != SOURCE_CHECKPOINT_MANIFEST_SHA256
        or selection.get("source_checkpoint_manifest_sha256") != SOURCE_CHECKPOINT_CANONICAL_SHA256
        or selection.get("source_artifact_tree_sha256") != TARGET_TREE_SHA256
        or selection.get("weights_sha256") != _canonical_digest(selection.get("weight_files", []))
        or selection.get("selection_sha256") != _canonical_digest(material)
    ):
        raise ValueError("V11 checkpoint binding is not the predeclared, intentionally reused V9 target")
    source = _source_checkpoint(selection["source_manifest_path"])
    for key in (
        "local_path", "model_id", "revision", "resolved_revision", "license_id",
        "license_source_url", "model_card_url", "model_card_sha256", "config_sha256",
        "config_architecture", "weight_files", "total_weight_bytes", "weights_sha256",
    ):
        if selection.get(key) != source.get(key):
            raise ValueError(f"V11 reused-target binding differs from V9 source evidence: {key}")
    return selection


def _load_checkpoint(custody: Path) -> tuple[Any, dict[str, Any]]:
    selection = _checkpoint_manifest(custody)
    model_path = Path(selection["local_path"]).resolve()
    tree_sha, files = engine.v3._tree_digest(model_path)
    if tree_sha != TARGET_TREE_SHA256:
        raise ValueError("V11 reused model tree failed digest verification")
    for row in selection["weight_files"]:
        path = model_path / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError("V11 reused model weight file failed digest verification")
    if _sha_file(model_path / "config.json") != selection["config_sha256"] or _sha_file(model_path / "README.md") != selection["model_card_sha256"]:
        raise ValueError("V11 reused model config or card failed digest verification")
    checkpoint = engine.v3.Checkpoint(
        family="opt_relu",
        model_id=TARGET_MODEL_ID,
        revision=TARGET_REVISION,
        local_path=model_path,
        selected_after_protocol_freeze=False,
    )
    return checkpoint, selection


def _visible_failure_layers(model: Any, tokenizer: Any, records: list[dict[str, Any]]) -> tuple[list[int], list[float], list[float]]:
    _selected, variances, visible_rms = _ORIGINAL_VISIBLE_SELECTION(model, tokenizer, records)
    if engine.v3._layer_count(model, "opt_relu") <= 7:
        raise ValueError("predeclared V9 failure layers 0 and 7 are absent from this target")
    return [0, 7], variances, visible_rms


def _cell_gate(metric: dict[str, Any]) -> bool:
    return (
        int(metric.get("resolvable_count", 0)) >= 20
        and float(metric.get("direction_accuracy", 0.0)) >= 0.80
        and float(metric.get("median_cosine", 0.0) or 0.0) >= 0.90
        and float(metric.get("median_normalized_error", float("inf"))) <= 0.20
    )


def _failure_cell_summary(report: dict[str, Any]) -> dict[str, Any]:
    targets = ((0, "ordinary"), (7, "ordinary"), (0, "adversarial"))
    lookup = {(int(row["seed_index"]), int(row["layer"]), row["method"]): row for row in report["results"]}
    cells = []
    per_seed = []
    for seed_index in range(len(fixtures.SEEDS)):
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
    return {
        "endpoint": "V9 previously failed next-logit causal-effect cells: ordinary at layers 0 and 7; adversarial at layer 0",
        "role": "predeclared secondary descriptive endpoint; not an additional PASS gate",
        "causal_fit_cells_passing": sum(bool(row["causal_fit_gate_pass"]) for row in cells),
        "activation_ridge_cells_passing": sum(bool(row["activation_ridge_gate_pass"]) for row in cells),
        "cells": cells,
        "per_seed_mean_paired_cosine_gain": per_seed,
    }


def _write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    if path.name == "outcome-report.json":
        value["claim_ceiling"] = CLAIM_CEILING
        value["target_reuse"] = {
            "status": "INTENTIONAL_REUSE_OF_V9_OPT_CHECKPOINT",
            "fresh_checkpoint_generalization_claimed": False,
            "source_v9_checkpoint_manifest_sha256": SOURCE_CHECKPOINT_MANIFEST_SHA256,
            "prior_failure_cell_summary_sha256": None,
        }
        summary = _failure_cell_summary(value)
        summary_digest = _canonical_digest(summary)
        value["prior_failure_cell_summary"] = summary
        value["prior_failure_cell_summary_sha256"] = summary_digest
        value["target_reuse"]["prior_failure_cell_summary_sha256"] = summary_digest
        value["known_limitations"] = list(value.get("known_limitations", [])) + [
            "the exact OPT/ReLU checkpoint was previously studied in V9 and is intentionally reused",
            "new synthetic fixtures test remediation on this checkpoint only; fresh-checkpoint generalization is untested",
        ]
        value["outcome_report_sha256"] = _canonical_digest({key: item for key, item in value.items() if key not in {"outcome_report_sha256", "created_at_utc"}})
    elif path.name == "replication-report.json" and _REPLICATION_PAIR is not None:
        primary_path, replica_path = _REPLICATION_PAIR
        primary_report = json.loads((primary_path / "outcome-report.json").read_text(encoding="utf-8"))
        replica_report = json.loads((replica_path / "outcome-report.json").read_text(encoding="utf-8"))
        left = primary_report.get("prior_failure_cell_summary_sha256")
        right = replica_report.get("prior_failure_cell_summary_sha256")
        matched = left == right and left is not None
        value["prior_failure_cell_summary_sha256_comparison"] = {"primary": left, "replica": right}
        value["prior_failure_cell_summary_match"] = matched
        value["exact_identity_and_output_digests_match"] = bool(value.get("exact_identity_and_output_digests_match")) and matched
        if not value["exact_identity_and_output_digests_match"]:
            value["assurance_level"] = "INCONCLUSIVE"
    _ORIGINAL_WRITE_JSON(path, value, immutable=immutable)


def _verify_completed_artifacts(custody: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_VERIFY_COMPLETED_ARTIFACTS(custody, freeze)
    report = json.loads((custody / "outcome-report.json").read_text(encoding="utf-8"))
    summary = _failure_cell_summary(report)
    digest = _canonical_digest(summary)
    if report.get("prior_failure_cell_summary") != summary or report.get("prior_failure_cell_summary_sha256") != digest:
        raise ValueError("V11 prior-failure-cell analysis failed recomputation")
    result["prior_failure_cell_summary_sha256"] = digest
    return result


_ORIGINAL_VERIFY_COMPLETED_ARTIFACTS = engine._verify_completed_artifacts


def configure_engine() -> Any:
    global _CONFIGURED
    if _CONFIGURED:
        return engine
    engine.STATE_SLICE = STATE_SLICE
    engine.PROTOCOL_SLICE = PROTOCOL_SLICE
    engine.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.PACKET_FREEZE = FREEZE_FILE
    engine.PACKET_FILES = PACKET_FILES
    engine.FAMILY = "opt_relu"
    engine.fixtures = fixtures
    engine._root = lambda: REPO_ROOT
    engine.verify_packet = verify_packet
    engine.freeze_packet = freeze_packet
    engine._checkpoint_manifest = _checkpoint_manifest
    engine._load_checkpoint = _load_checkpoint
    engine._visible_selection = _visible_failure_layers
    engine._write_json = _write_json
    engine._verify_completed_artifacts = _verify_completed_artifacts
    _CONFIGURED = True
    return engine


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    return runner.archive_packet(custody_path)


def prepare_replica(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    if (replica / "checkpoint-selection.json").exists():
        raise FileExistsError("replica target binding already exists; use a new custody root")
    runner._verify_packet_archive(primary)
    source_selection = _checkpoint_manifest(primary)
    runner.archive_packet(replica)
    _ORIGINAL_WRITE_JSON(replica / "checkpoint-selection.json", source_selection, immutable=True)
    _checkpoint_manifest(replica)
    return {"replica_packet_archive_sha256": runner._verify_packet_archive(replica)["archive_sha256"], "selection_sha256": source_selection["selection_sha256"]}


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    global _REPLICATION_PAIR
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    _REPLICATION_PAIR = (primary, replica)
    try:
        return runner.compare_replication(primary, replica)
    finally:
        _REPLICATION_PAIR = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    bind = commands.add_parser("bind-target")
    bind.add_argument("--custody", required=True)
    bind.add_argument("--source-manifest", required=True)
    for name in ("prepare", "predict", "score", "run-local"):
        command = commands.add_parser(name)
        command.add_argument("--custody", required=True)
        if name in {"predict", "run-local"}:
            command.add_argument("--entropy-source")
    replica = commands.add_parser("prepare-replica")
    replica.add_argument("--primary", required=True)
    replica.add_argument("--replica", required=True)
    compare = commands.add_parser("compare-replication")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    runner = configure_engine()
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "bind-target":
        result = bind_reused_target(args.custody, args.source_manifest)
    elif args.command == "prepare-replica":
        result = prepare_replica(args.primary, args.replica)
    elif args.command == "compare-replication":
        result = compare_replication(args.primary, args.replica)
    elif args.command == "prepare":
        result = runner.prepare(args.custody)
    elif args.command == "predict":
        result = runner.predict(args.custody, entropy_source=args.entropy_source)
    elif args.command == "score":
        result = runner.score(args.custody)
    else:
        result = runner.run_local(args.custody, entropy_source=args.entropy_source)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
