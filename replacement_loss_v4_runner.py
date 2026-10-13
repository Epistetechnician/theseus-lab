"""Frozen local runner for the paired compositional-support experiment.

State slice: theseus-lab-sandbox-v1.

The v3 implementation is imported as an immutable, digest-pinned dependency.
All model assets, prompts, captures, predictions, and outcomes stay in external
custody. Standard output contains aggregate state only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import replacement_loss_v3_runner as v3
import replacement_loss_v4 as v4
import replacement_loss_v4_fixtures as v4_fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v4-compositional-support"
FAMILY = "gpt2_gelu"
LAYER = 3
RIDGE = 1e-6
TOKENS_PER_PROMPT = 16
BATCH_SIZE = 4
CASE_WALL_CAP_SECONDS = 2 * 60 * 60
RSS_CAP_BYTES = 24 * 1024**3
V3_CUSTODY_DEFAULT = Path(
    "/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v3"
)
DEFAULT_CUSTODY = Path(
    "/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v4"
)
PACKET_FILES = (
    "docs/replacement-loss-v4.md",
    "docs/replacement-loss-v3-failure-analysis.md",
    "replacement_loss_v4.py",
    "replacement_loss_v4_fixtures.py",
    "replacement_loss_v4_runner.py",
    "tests/test_replacement_loss_v4.py",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
    "requirements.txt",
)
EXPECTED_V3_HELPER_SHA256 = {
    "replacement_loss_v3.py": "d0543988ca048edcdb5ee94d58b710ddb50b1de5567f1813e222979e93eff379",
    "replacement_loss_v3_fixtures.py": "521612e305884ff5d47689853e51d41be00b3fda60dbe0188e98bf7e47687ab0",
    "replacement_loss_v3_runner.py": "d6a2c14da6ad82f6d9624d460fc3531c7e8e8c956464607b2d1da3abfcdc3278",
    "requirements.txt": "7f8769a3acdacbfa9103b603751b028196ccdae3ef16420e7c9288ac079a064d",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _runtime() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "safetensors", "tokenizers", "torch", "transformers"):
        package = __import__(name)
        packages[name] = str(getattr(package, "__version__", "unknown"))
    inventory = sorted((str(k), str(v)) for k, v in packages.items())
    return {
        "python": sys.version.split()[0],
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "packages": packages,
        "dependency_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _packet_manifest_path(custody: Path) -> Path:
    return custody / "frozen-packet-manifest.json"


def freeze_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    """Archive and digest the exact protocol/code/dependency packet."""

    custody = Path(custody_root).resolve()
    if custody == _repo_root() or _repo_root() in custody.parents:
        raise ValueError("experiment custody must be outside the repository")
    if custody.exists() and any(custody.iterdir()):
        raise FileExistsError("v4 custody must be new or empty for packet freeze")
    archive = custody / "frozen-packet"
    rows = []
    for relative in PACKET_FILES:
        source = _repo_root() / relative
        if not source.is_file():
            raise FileNotFoundError(f"frozen packet input is missing: {relative}")
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        digest = _sha(source)
        if relative in EXPECTED_V3_HELPER_SHA256 and digest != EXPECTED_V3_HELPER_SHA256[relative]:
            raise ValueError(f"pinned v3 dependency changed: {relative}")
        rows.append({"path": relative, "sha256": digest, "bytes": source.stat().st_size})
    content = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V4_ADAPTATION_FIXTURES",
        "created_at_utc": _now(),
        "packet_files": rows,
        "checkpoint_identity": {
            "model_id": "openai-community/gpt2-medium",
            "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
            "reuse_scope": "same v3 target; no fresh-checkpoint or cross-checkpoint claim",
        },
    }
    content["packet_sha256"] = v3._canonical_digest(content)
    _write_json(_packet_manifest_path(custody), content)
    local_manifest = _repo_root() / "docs" / "replacement-loss-v4-freeze.json"
    _write_json(local_manifest, content)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": content["freeze_state"],
        "packet_sha256": content["packet_sha256"],
        "packet_file_count": len(rows),
        "custody_root": str(custody),
        "manifest_sha256": _sha(_packet_manifest_path(custody)),
    }


def verify_packet(custody_root: str | Path) -> dict[str, Any]:
    custody = Path(custody_root).resolve()
    manifest = _json(_packet_manifest_path(custody))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v4 packet identity mismatch")
    rows = manifest.get("packet_files")
    if not isinstance(rows, list) or len(rows) != len(PACKET_FILES):
        raise ValueError("v4 packet file set is malformed")
    for row in rows:
        relative = str(row["path"])
        source = _repo_root() / relative
        archive = custody / "frozen-packet" / relative
        if _sha(source) != row["sha256"] or _sha(archive) != row["sha256"]:
            raise ValueError(f"v4 packet bytes changed: {relative}")
    recorded = manifest.get("packet_sha256")
    recomputed = v3._canonical_digest({k: v for k, v in manifest.items() if k != "packet_sha256"})
    if recorded != recomputed:
        raise ValueError("v4 packet digest mismatch")
    return {"packet_sha256": recorded, "packet_file_count": len(rows), "verified": True}


def _model_selection(custody: Path, v3_custody: Path) -> dict[str, Any]:
    v3_packet = _json(v3_custody / "frozen-packet-archive.json")
    v3_report_path = v3_custody / "replacement-loss-v3-report.json"
    v3_report = _json(v3_report_path)
    entries = _json(v3_custody / "checkpoints.json")["checkpoints"]
    entry = next(row for row in entries if row["family"] == FAMILY)
    if entry["model_id"] != "openai-community/gpt2-medium" or entry["revision"] != "6dcaa7a952f72f9298047fd5137cd6e4f05f41da":
        raise ValueError("v3 GPT-2 target identity does not match the v4 predeclaration")
    case = next(row for row in v3_report["cases"] if row["family"] == FAMILY and row["seed_index"] == 0)
    model_dir = (v3_custody / entry["local_path"]).resolve()
    digest, _tree = v3._tree_digest(model_dir)
    if digest != case["checkpoint_tree_sha256"]:
        raise ValueError("GPT-2 checkpoint tree digest differs from v3 report")
    if _repo_root() in model_dir.parents:
        raise ValueError("model assets may not be stored in the repository")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": _json(_packet_manifest_path(custody))["packet_sha256"],
        "v3_packet_sha256": v3_packet["packet_sha256"],
        "v3_primary_report_sha256": _sha(v3_report_path),
        "family": FAMILY,
        "model_id": entry["model_id"],
        "revision": entry["revision"],
        "checkpoint_tree_sha256": digest,
        "checkpoint_path": str(model_dir),
        "target_projection": "transformer.h.3.mlp.c_proj",
        "selected_layer": LAYER,
        "target_selection_source": "all three v3 GPT-2 cases selected layer 3 before v4 was designed",
        "fresh_checkpoint": False,
        "created_at_utc": _now(),
    }


def _load_checkpoint(root: Path) -> v3.Checkpoint:
    selection = _json(root / "model-selection.json")
    checkpoint = v3.Checkpoint(
        family=FAMILY,
        model_id=selection["model_id"],
        revision=selection["revision"],
        local_path=Path(selection["checkpoint_path"]),
        selected_after_protocol_freeze=False,
    )
    current, _ = v3._tree_digest(checkpoint.local_path)
    if current != selection["checkpoint_tree_sha256"]:
        raise ValueError("checkpoint bytes changed after v4 model identity was recorded")
    return checkpoint


def _capture_fit_arm(
    model: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    seed: int,
    root: Path,
    arm: str,
) -> dict[str, Any]:
    encoded = v3._token_rows(tokenizer, records, with_continuation=False)
    inputs_rows: list[torch.Tensor] = []
    targets_rows: list[torch.Tensor] = []
    for group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
        _logits, captured_inputs, captured_outputs = v3._run_projection(
            model, FAMILY, LAYER, inputs, positions=None
        )
        assert captured_outputs is not None
        for index, row in enumerate(group):
            length = int(row["prompt_len"])
            if length < TOKENS_PER_PROMPT:
                raise ValueError("adaptation prompt is too short for the frozen 16-position rule")
            positions = np.rint(np.linspace(0, length - 1, TOKENS_PER_PROMPT)).astype(np.int64)
            if len(set(positions.tolist())) != TOKENS_PER_PROMPT:
                raise ValueError("position sampler did not select 16 distinct tokens")
            position_tensor = torch.as_tensor(positions, dtype=torch.long)
            inputs_rows.append(captured_inputs[index].index_select(0, position_tensor).float())
            targets_rows.append(captured_outputs[index].index_select(0, position_tensor).float())
    x, y = torch.cat(inputs_rows), torch.cat(targets_rows)
    if x.shape[0] != 128 * TOKENS_PER_PROMPT or y.shape[0] != x.shape[0]:
        raise ValueError("adaptation capture did not produce exactly 2,048 paired rows")
    projection = v3._projection(model, FAMILY, LAYER)
    in_features, out_features, use_bias = v3._module_dimensions(projection)
    candidate, fit_receipt = v3.fit_independent_linear_replacement(
        in_features,
        out_features,
        x,
        y,
        use_bias=use_bias,
        seed=seed,
        ridge=RIDGE,
    )
    capture_path = root / "adaptation-captures" / f"seed-{seed}" / f"{arm}.npz"
    v3._save_npz_deterministic(capture_path, target_inputs=x.numpy(), target_outputs=y.numpy())
    candidate_path = root / "candidates" / f"seed-{seed}" / f"{arm}.safetensors"
    candidate_digest = v3._save_linear(candidate_path, candidate)
    return {
        "arm": arm,
        "seed": seed,
        "prompt_count": len(records),
        "adaptation_rows": int(x.shape[0]),
        "capture_path": capture_path.relative_to(root).as_posix(),
        "capture_sha256": _sha(capture_path),
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_file_sha256": _sha(candidate_path),
        "candidate_state_sha256": candidate_digest,
        "fit": fit_receipt,
    }


def prepare_candidates(
    custody_root: str | Path = DEFAULT_CUSTODY,
    v3_custody_root: str | Path = V3_CUSTODY_DEFAULT,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "candidate-preparation.json").exists() or (root / "candidate-freeze.json").exists():
        raise FileExistsError("candidate preparation is immutable; use a fresh custody root for a new run")
    if any((root / name).exists() for name in ("adaptation-fixtures", "adaptation-captures", "candidates")):
        raise FileExistsError("partial candidate state exists; preserve it and use a fresh custody root")
    model_path = root / "model-selection.json"
    if not model_path.exists():
        _write_json(model_path, _model_selection(root, Path(v3_custody_root).resolve()))
    selection = _json(model_path)
    if selection.get("v4_packet_sha256") != _json(_packet_manifest_path(root))["packet_sha256"]:
        raise ValueError("model identity does not bind the frozen v4 packet")
    checkpoint = _load_checkpoint(root)
    model, tokenizer = v3._load_local_model(checkpoint)
    arms_receipts: list[dict[str, Any]] = []
    fixture_hashes: dict[str, str] = {}
    for seed_index, seed in enumerate(v4_fixtures.SEEDS):
        arms = v4_fixtures.generate_adaptation_fixtures(seed_index)
        fixture_path = root / "adaptation-fixtures" / f"seed-{seed}.json"
        _write_json(fixture_path, arms)
        fixture_hashes[str(seed)] = _sha(fixture_path)
        for arm, fixture in arms.items():
            records = fixture["records"]
            arms_receipts.append(
                _capture_fit_arm(model, tokenizer, records, seed, root, arm)
            )
    result = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": _json(_packet_manifest_path(root))["packet_sha256"],
        "model_selection_sha256": _sha(model_path),
        "adaptation_fixture_sha256": fixture_hashes,
        "arms": arms_receipts,
        "runtime": _runtime(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    if result["peak_rss_bytes"] > RSS_CAP_BYTES:
        result["resource_status"] = "INCONCLUSIVE_RSS_CAP_EXCEEDED"
    else:
        result["resource_status"] = "WITHIN_CAP"
    _write_json(root / "candidate-preparation.json", result)
    return {
        "status": result["resource_status"],
        "seed_count": len(v4_fixtures.SEEDS),
        "candidate_count": len(arms_receipts),
        "rows_per_candidate": 2048,
        "elapsed_wall_seconds": result["elapsed_wall_seconds"],
        "peak_rss_bytes": result["peak_rss_bytes"],
        "adaptation_fixture_sha256": fixture_hashes,
    }


def freeze_candidates(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("candidate freeze is immutable")
    if (root / "challenge-manifest.json").exists():
        raise FileExistsError("candidate freeze cannot follow challenge generation")
    preparation = _json(root / "candidate-preparation.json")
    if preparation.get("resource_status") != "WITHIN_CAP":
        raise ValueError("candidate preparation is incomplete or over the resource cap")
    artifacts = []
    for row in preparation["arms"]:
        for key in ("capture_path", "candidate_path"):
            relative = row[key]
            artifacts.append({"path": relative, "sha256": _sha(root / relative)})
    fixtures = [
        {"path": f"adaptation-fixtures/seed-{seed}.json", "sha256": _sha(root / f"adaptation-fixtures/seed-{seed}.json")}
        for seed in v4_fixtures.SEEDS
    ]
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": _json(_packet_manifest_path(root))["packet_sha256"],
        "candidate_state": "FROZEN_BEFORE_CHALLENGE_GENERATION",
        "created_at_utc": _now(),
        "candidate_files": sorted(artifacts, key=lambda item: item["path"]),
        "adaptation_fixtures": fixtures,
        "candidate_preparation_sha256": _sha(root / "candidate-preparation.json"),
        "model_selection_sha256": _sha(root / "model-selection.json"),
    }
    freeze["candidate_freeze_sha256"] = v3._canonical_digest(freeze)
    _write_json(root / "candidate-freeze.json", freeze)
    return {
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "candidate_count": len(preparation["arms"]),
        "adaptation_fixture_count": len(fixtures),
        "created_at_utc": freeze["created_at_utc"],
    }


def _verify_candidate_freeze(root: Path) -> dict[str, Any]:
    freeze = _json(root / "candidate-freeze.json")
    if freeze.get("v4_packet_sha256") != _json(_packet_manifest_path(root))["packet_sha256"]:
        raise ValueError("candidate freeze is for another packet")
    if _sha(root / "model-selection.json") != freeze.get("model_selection_sha256"):
        raise ValueError("model identity changed after candidate freeze")
    for item in freeze["candidate_files"] + freeze["adaptation_fixtures"]:
        if _sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"candidate/adaptation bytes changed: {Path(item['path']).name}")
    expected = freeze.get("candidate_freeze_sha256")
    recomputed = v3._canonical_digest({k: v for k, v in freeze.items() if k != "candidate_freeze_sha256"})
    if expected != recomputed:
        raise ValueError("candidate freeze digest mismatch")
    return freeze


def make_challenges(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    freeze = _verify_candidate_freeze(root)
    if (root / "challenge-manifest.json").exists() or (root / "challenges").exists():
        raise FileExistsError("challenge fixture already exists; packet is immutable")
    rows = []
    for seed_index, seed in enumerate(v4_fixtures.SEEDS):
        fixture = v4_fixtures.generate_challenge_fixture(seed_index)
        adaptation = _json(root / "adaptation-fixtures" / f"seed-{seed}.json")
        v4_fixtures.validate_fresh_disjointness(adaptation, fixture)
        # Reject reuse of any exact prompt/content from the prior GPT-2 v3 challenges.
        previous_prompts: set[str] = set()
        previous_content: set[str] = set()
        v3_root = V3_CUSTODY_DEFAULT
        for old_seed in range(3):
            old_path = v3_root / "challenges" / FAMILY / f"seed-{old_seed}" / "prompts.json"
            old = _json(old_path)
            for record in old["records"]:
                previous_prompts.add(str(record["input_text"]))
                previous_content.add(hashlib.sha256((str(record["input_text"]) + "\0" + str(record["continuation"])).encode("utf-8")).hexdigest())
        for record in fixture["records"]:
            content = hashlib.sha256((str(record["input_text"]) + "\0" + str(record["continuation"])).encode("utf-8")).hexdigest()
            if str(record["input_text"]) in previous_prompts or content in previous_content:
                raise ValueError("v4 challenge fixture reused prior v3 GPT-2 challenge content")
        path = root / "challenges" / f"seed-{seed}.json"
        _write_json(path, fixture)
        rows.append({
            "seed": seed,
            "path": path.relative_to(root).as_posix(),
            "file_sha256": _sha(path),
            "example_set_sha256": fixture["example_set_sha256"],
            "count": fixture["count"],
            "stratum_counts": fixture["stratum_counts"],
        })
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": freeze["v4_packet_sha256"],
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "candidate_freeze_verified": True,
        "challenge_generation_started_after_candidate_freeze": True,
        "challenge_entropy_retained": False,
        "created_at_utc": _now(),
        "cases": rows,
    }
    manifest["challenge_manifest_sha256"] = v3._canonical_digest(manifest)
    _write_json(root / "challenge-manifest.json", manifest)
    return {
        "case_count": len(rows),
        "record_count": sum(item["count"] for item in rows),
        "challenge_manifest_sha256": manifest["challenge_manifest_sha256"],
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
    }


def predict(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "prediction-freeze.json").exists() or (root / "predictions").exists():
        raise FileExistsError("predictions have already been sealed for this custody root")
    freeze = _verify_candidate_freeze(root)
    challenge_manifest = _json(root / "challenge-manifest.json")
    if not challenge_manifest.get("challenge_generation_started_after_candidate_freeze"):
        raise ValueError("challenge data does not follow candidate freeze")
    checkpoint = _load_checkpoint(root)
    model, tokenizer = v3._load_local_model(checkpoint)
    receipts = []
    for seed_index, seed in enumerate(v4_fixtures.SEEDS):
        fixture = _json(root / "challenges" / f"seed-{seed}.json")
        records = fixture["records"]
        encoded = v3._token_rows(tokenizer, records, with_continuation=True)
        arrays: dict[str, list[np.ndarray]] = {
            "example_ids": [], "strata": [],
            "ordinary_only_activation": [], "ordinary_only_next_logits": [], "ordinary_only_ce": [],
            "compositional_augmented_activation": [], "compositional_augmented_next_logits": [], "compositional_augmented_ce": [],
        }
        candidates = {}
        for arm in ("ordinary_only", "compositional_augmented"):
            candidate_path = root / "candidates" / f"seed-{seed}" / f"{arm}.safetensors"
            projection = v3._projection(model, FAMILY, LAYER)
            in_features, out_features, use_bias = v3._module_dimensions(projection)
            candidates[arm] = v3._load_linear(candidate_path, in_features, out_features, use_bias)
        for group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
            arrays["example_ids"].append(np.asarray([row["example_id"] for row in group], dtype="U24"))
            arrays["strata"].append(np.asarray([row["stratum"] for row in group], dtype="U24"))
            for arm, candidate in candidates.items():
                logits, activation = v3._model_forward_variant(
                    model, FAMILY, LAYER, candidate, group, inputs, deltas=None
                )
                arrays[f"{arm}_activation"].append(activation.numpy())
                arrays[f"{arm}_next_logits"].append(v3._next_logits(logits, group).numpy())
                arrays[f"{arm}_ce"].append(v3._continuation_cross_entropy(logits, group))
        payload = {name: np.concatenate(parts, axis=0) for name, parts in arrays.items()}
        if not np.array_equal(payload["example_ids"], np.asarray([r["example_id"] for r in records], dtype="U24")):
            raise ValueError("prediction row ordering differs from challenge fixture")
        prediction_path = root / "predictions" / f"seed-{seed}.npz"
        v3._save_npz_deterministic(prediction_path, **payload)
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "seed": seed,
            "packet_sha256": freeze["v4_packet_sha256"],
            "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
            "challenge_fixture_sha256": fixture["example_set_sha256"],
            "prediction_file_sha256": _sha(prediction_path),
            "prediction_query_count": len(records) * 2,
            "original_outcomes_opened": False,
            "sealed_at_utc": _now(),
        }
        receipt_path = root / "predictions" / f"seed-{seed}.receipt.json"
        _write_json(receipt_path, receipt)
        receipts.append({"seed": seed, "prediction_file_sha256": receipt["prediction_file_sha256"], "receipt_sha256": _sha(receipt_path)})
    seal = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": freeze["v4_packet_sha256"],
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "predictions": receipts,
        "all_predictions_sealed_before_original_outcomes": True,
        "created_at_utc": _now(),
    }
    seal["prediction_freeze_sha256"] = v3._canonical_digest(seal)
    _write_json(root / "prediction-freeze.json", seal)
    return {"seed_count": len(receipts), "prediction_freeze_sha256": seal["prediction_freeze_sha256"], "prediction_digests": receipts}


def evaluate(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "replacement-loss-v4-report.json").exists() or (root / "outcomes").exists():
        raise FileExistsError("evaluation receipt is immutable; use a new custody root for another run")
    freeze = _verify_candidate_freeze(root)
    prediction_freeze = _json(root / "prediction-freeze.json")
    if not prediction_freeze.get("all_predictions_sealed_before_original_outcomes"):
        raise ValueError("predictions were not sealed before original outcome access")
    checkpoint = _load_checkpoint(root)
    model, tokenizer = v3._load_local_model(checkpoint)
    cases = []
    outcome_hashes = []
    prediction_hashes = []
    for seed_index, seed in enumerate(v4_fixtures.SEEDS):
        case_started = time.monotonic()
        fixture = _json(root / "challenges" / f"seed-{seed}.json")
        records = fixture["records"]
        encoded = v3._token_rows(tokenizer, records, with_continuation=True)
        original_acts: list[np.ndarray] = []
        original_logits: list[np.ndarray] = []
        original_ce: list[np.ndarray] = []
        for group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
            logits, _inputs, activations = v3._model_forward_original(
                model, FAMILY, LAYER, group, inputs
            )
            original_acts.append(activations.numpy())
            original_logits.append(v3._next_logits(logits, group).numpy())
            original_ce.append(v3._continuation_cross_entropy(logits, group))
        baseline_activation = np.concatenate(original_acts)
        baseline_logits = np.concatenate(original_logits)
        baseline_ce = np.concatenate(original_ce)
        prediction_path = root / "predictions" / f"seed-{seed}.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            predictions = {name: archive[name] for name in archive.files}
        strata = predictions["strata"].astype(str)
        fixture_ids = np.asarray([row["example_id"] for row in records], dtype="U24")
        if not np.array_equal(predictions["example_ids"], fixture_ids):
            raise ValueError("prediction/outcome example IDs differ")
        candidate_acts = {arm: predictions[f"{arm}_activation"] for arm in ("ordinary_only", "compositional_augmented")}
        candidate_logits = {arm: predictions[f"{arm}_next_logits"] for arm in ("ordinary_only", "compositional_augmented")}
        candidate_ce = {arm: predictions[f"{arm}_ce"] for arm in ("ordinary_only", "compositional_augmented")}
        case_metrics = v4.evaluate_seed(
            strata, baseline_activation, baseline_logits, baseline_ce,
            candidate_acts, candidate_logits, candidate_ce,
        )
        outcome_path = root / "outcomes" / f"seed-{seed}.npz"
        v3._save_npz_deterministic(
            outcome_path,
            example_ids=fixture_ids,
            strata=strata.astype("U24"),
            original_module_activation=baseline_activation,
            original_next_logits=baseline_logits,
            original_ce=baseline_ce,
        )
        prediction_hashes.append({"seed": seed, "sha256": _sha(prediction_path)})
        outcome_hashes.append({"seed": seed, "sha256": _sha(outcome_path)})
        elapsed = time.monotonic() - case_started
        rss = _peak_rss_bytes()
        passed = v4.seed_pass(case_metrics)
        within_caps = elapsed <= CASE_WALL_CAP_SECONDS and rss <= RSS_CAP_BYTES
        cases.append({
            "seed": seed,
            "challenge_fixture_sha256": fixture["example_set_sha256"],
            "prediction_array_sha256": _sha(prediction_path),
            "outcome_array_sha256": _sha(outcome_path),
            "metrics": case_metrics,
            "case_status": "PASS" if passed and within_caps else "FAIL" if within_caps else "INCONCLUSIVE",
            "elapsed_wall_seconds": elapsed,
            "peak_rss_bytes": rss,
        })
    statuses = {row["case_status"] for row in cases}
    overall = "PASS" if statuses == {"PASS"} else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "FAIL"
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": freeze["v4_packet_sha256"],
        "v3_packet_sha256": _json(root / "model-selection.json")["v3_packet_sha256"],
        "v3_primary_report_sha256": _json(root / "model-selection.json")["v3_primary_report_sha256"],
        "source": {
            "git_revision": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=_repo_root(), check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "worktree_status_sha256": hashlib.sha256(
                subprocess.run(
                    ["git", "status", "--porcelain=v1"], cwd=_repo_root(), check=True,
                    capture_output=True, text=True,
                ).stdout.encode("utf-8")
            ).hexdigest(),
        },
        "assurance_level": "INCONCLUSIVE",
        "outcome_status": overall,
        "reproducibility_status": "PENDING_SEPARATE_RUNTIME_REPLICATION",
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "case_count": len(cases),
        "pass_count": sum(case["case_status"] == "PASS" for case in cases),
        "fail_count": sum(case["case_status"] == "FAIL" for case in cases),
        "inconclusive_count": sum(case["case_status"] == "INCONCLUSIVE" for case in cases),
        "cases": cases,
        "prediction_arrays": prediction_hashes,
        "outcome_arrays": outcome_hashes,
        "runtime": _runtime(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "claim_ceiling": "LOCAL_GPT2_COMPOSITIONAL_ADAPTATION_EFFECT",
        "nonclaims": ["fresh-checkpoint generalization", "all GPT-2 checkpoints", "all architectures", "independent science", "alignment", "consciousness", "production authorization"],
        "data_and_artifact_custody": "All prompt text, checkpoint weights, activation captures, candidate states, predictions, and row-level outcomes are in external owner custody.",
        "next_blocked_authority_gate": "Separate runtime reproduction, then configured independent reviewer if an independent claim is pursued.",
        "monetary_cost": "NOT_MEASURED",
        "energy": "NOT_MEASURED",
    }
    report_path = root / "replacement-loss-v4-report.json"
    _write_json(report_path, report)
    return {
        "assurance_level": report["assurance_level"],
        "outcome_status": overall,
        "case_count": len(cases),
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "inconclusive_count": report["inconclusive_count"],
        "report_path": str(report_path),
        "report_sha256": _sha(report_path),
        "cases": [{
            "seed": case["seed"],
            "status": case["case_status"],
            "compositional_comparison": case["metrics"]["paired_compositional_comparison"],
            "augmented_compositional": case["metrics"]["arms"]["compositional_augmented"]["per_stratum"]["compositional"],
        } for case in cases],
    }


def replicate(
    primary_root: str | Path,
    replica_root: str | Path,
    v3_custody_root: str | Path = V3_CUSTODY_DEFAULT,
) -> dict[str, Any]:
    started = time.monotonic()
    primary = Path(primary_root).resolve()
    replica = Path(replica_root).resolve()
    if primary == replica or primary in replica.parents or replica in primary.parents:
        raise ValueError("primary and replica custody roots must be separate")
    verify_packet(primary)
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("replica custody must be a new or empty external directory")
    replica.mkdir(parents=True, exist_ok=True)
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(_packet_manifest_path(primary), _packet_manifest_path(replica))
    shutil.copyfile(primary / "model-selection.json", replica / "model-selection.json")
    verify_packet(replica)
    prepare_candidates(replica, v3_custody_root)
    primary_freeze = _verify_candidate_freeze(primary)
    primary_candidate_hashes = {item["path"]: item["sha256"] for item in primary_freeze["candidate_files"]}
    replica_candidate_hashes = {
        item["path"]: _sha(replica / item["path"])
        for item in primary_freeze["candidate_files"]
    }
    candidates_match = primary_candidate_hashes == replica_candidate_hashes
    if not candidates_match:
        raise ValueError("replica candidate reconstruction differs from the frozen primary candidates")
    shutil.copyfile(primary / "candidate-freeze.json", replica / "candidate-freeze.json")
    replica_candidate_freeze = _verify_candidate_freeze(replica)
    for seed in v4_fixtures.SEEDS:
        source = primary / "challenges" / f"seed-{seed}.json"
        (replica / "challenges").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, replica / "challenges" / source.name)
    shutil.copyfile(primary / "challenge-manifest.json", replica / "challenge-manifest.json")
    prediction_result = predict(replica)
    evaluate(replica)
    primary_report = _json(primary / "replacement-loss-v4-report.json")
    replica_report = _json(replica / "replacement-loss-v4-report.json")
    prediction_match = primary_report["prediction_arrays"] == replica_report["prediction_arrays"]
    outcome_match = primary_report["outcome_arrays"] == replica_report["outcome_arrays"]
    source_match = (
        primary_report["v4_packet_sha256"] == replica_report["v4_packet_sha256"]
        and primary_freeze["candidate_freeze_sha256"] == replica_candidate_freeze["candidate_freeze_sha256"]
    )
    runtime_match = primary_report["runtime"]["dependency_inventory_sha256"] == replica_report["runtime"]["dependency_inventory_sha256"]
    distinct_runtime = primary_report["runtime"]["python_prefix"] != replica_report["runtime"]["python_prefix"]
    reproducible = candidates_match and prediction_match and outcome_match and source_match and runtime_match and distinct_runtime
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v4_packet_sha256": primary_report["v4_packet_sha256"],
        "primary_report_sha256": _sha(primary / "replacement-loss-v4-report.json"),
        "replica_report_sha256": _sha(replica / "replacement-loss-v4-report.json"),
        "candidate_reconstruction_match": candidates_match,
        "prediction_array_digests_match": prediction_match,
        "outcome_array_digests_match": outcome_match,
        "source_identity_match": source_match,
        "dependencies_match": runtime_match,
        "distinct_clean_runtime": distinct_runtime,
        "same_owner_second_environment": True,
        "assurance_level": "REPRODUCED_LOCAL" if reproducible else "INCONCLUSIVE",
        "reproducibility_status": "REPRODUCED_LOCAL" if reproducible else "INCONCLUSIVE",
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "primary_outcome_status": primary_report["outcome_status"],
        "replica_outcome_status": replica_report["outcome_status"],
        "primary_pass_count": primary_report["pass_count"],
        "replica_pass_count": replica_report["pass_count"],
        "next_blocked_authority_gate": "Configured external reviewer with independent authority; no acceptance recorded.",
        "nonclaims": primary_report["nonclaims"],
    }
    _write_json(replica / "replication-report.json", receipt)
    return {
        **receipt,
        "replication_report_path": str(replica / "replication-report.json"),
        "replication_report_sha256": _sha(replica / "replication-report.json"),
        "replica_elapsed_wall_seconds": time.monotonic() - started,
        "replica_prediction_freeze_sha256": prediction_result["prediction_freeze_sha256"],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the frozen GPT-2 compositional-support v4 protocol")
    parser.add_argument("command", choices=("freeze-packet", "prepare", "freeze-candidates", "make-challenges", "predict", "evaluate", "replicate"))
    parser.add_argument("--custody-root", default=str(DEFAULT_CUSTODY))
    parser.add_argument("--v3-custody-root", default=str(V3_CUSTODY_DEFAULT))
    parser.add_argument("--replica-root")
    args = parser.parse_args()
    if args.command == "freeze-packet":
        result = freeze_packet(args.custody_root)
    elif args.command == "prepare":
        result = prepare_candidates(args.custody_root, args.v3_custody_root)
    elif args.command == "freeze-candidates":
        result = freeze_candidates(args.custody_root)
    elif args.command == "make-challenges":
        result = make_challenges(args.custody_root)
    elif args.command == "predict":
        result = predict(args.custody_root)
    elif args.command == "evaluate":
        result = evaluate(args.custody_root)
    else:
        if not args.replica_root:
            parser.error("replicate requires --replica-root")
        result = replicate(args.custody_root, args.replica_root, args.v3_custody_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
