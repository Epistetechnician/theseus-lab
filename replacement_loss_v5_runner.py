"""Frozen local runner for replacement-loss-v5.

State slice: theseus-lab-sandbox-v1. Prompts, model assets, captures,
candidates, predictions, and outcomes remain in external owner custody.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import secrets
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
import replacement_loss_v5 as v5
import replacement_loss_v5_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v5-balanced-adversarial-support"
FAMILY = "gpt2_gelu"
LAYER = 3
RIDGE = 1e-6
TOKENS_PER_PROMPT = 16
BATCH_SIZE = 4
CASE_WALL_CAP_SECONDS = 2 * 60 * 60
RSS_CAP_BYTES = 24 * 1024**3
V3_CUSTODY_DEFAULT = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v3")
V4_CUSTODY_DEFAULT = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v4")
V4_DIAG_ROOT = V4_CUSTODY_DEFAULT / "failure-analysis-v1"
DEFAULT_CUSTODY = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v5")
V4_DIAG_PACKET_SHA256 = "49928db7c35e5890fd948e85c832633f1598e6f86b7b5be7aa32dd030b906311"
V4_DIAG_FREEZE_FILE_SHA256 = "23ff22024dd3c16eaf2028b85485bc2bd7b5bdc8a8e70c0b76f6fbbca26bf145"
V4_DIAG_REPORT_DIGEST = "b704c0033c10ce447006da283ff661ce76a6ecc4bf9b8d010567deec12dbff0f"
V4_DIAG_REPORT_FILE_SHA256 = "4b7517666b56fdd3e544406d022f2f4284ebd1d5533714c7a1e3e6574e54fd14"
V4_DIAG_RECONCILIATION_DIGEST = "f6a07ca9b50987b455624bb1009fa8b7fefbe7ab237ef3cc8b9d534cc285a677"
V4_DIAG_RECONCILIATION_FILE_SHA256 = "1606c49edbf55f603f75add8f4b43d406f83f494a9f347414db3ab7f33aaad31"
V3_PACKET_SHA256 = "d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83"
PACKET_FILES = (
    "docs/replacement-loss-v5.md",
    "docs/replacement-loss-v4-failure-analysis.md",
    "replacement_loss_v4_failure_analysis.py",
    "replacement_loss_v5.py",
    "replacement_loss_v5_fixtures.py",
    "replacement_loss_v5_runner.py",
    "tests/test_replacement_loss_v5.py",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
    "requirements.txt",
)
PINNED_V3_FILES = {
    "replacement_loss_v3.py": "d0543988ca048edcdb5ee94d58b710ddb50b1de5567f1813e222979e93eff379",
    "replacement_loss_v3_fixtures.py": "521612e305884ff5d47689853e51d41be00b3fda60dbe0188e98bf7e47687ab0",
    "replacement_loss_v3_runner.py": "d6a2c14da6ad82f6d9624d460fc3531c7e8e8c956464607b2d1da3abfcdc3278",
    "requirements.txt": "7f8769a3acdacbfa9103b603751b028196ccdae3ef16420e7c9288ac079a064d",
}


def _root() -> Path:
    return Path(__file__).resolve().parent


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


def _runtime() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "safetensors", "tokenizers", "torch", "transformers"):
        package = __import__(name)
        packages[name] = str(getattr(package, "__version__", "unknown"))
    inventory = sorted((str(key), str(value)) for key, value in packages.items())
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "platform": platform.platform(),
        "packages": packages,
        "dependency_inventory_sha256": hashlib.sha256(json.dumps(inventory, separators=(",", ":")).encode()).hexdigest(),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }


def _peak_rss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if platform.system() == "Darwin" else raw * 1024


def _manifest_path(root: Path) -> Path:
    return root / "frozen-packet-manifest.json"


def _verify_v4_diagnosis() -> None:
    freeze = _json(V4_DIAG_ROOT / "freeze.json")
    report_path = V4_DIAG_ROOT / "analysis-report.json"
    reconciliation_path = V4_DIAG_ROOT / "reconciliation.json"
    if freeze.get("analysis_packet_sha256") != V4_DIAG_PACKET_SHA256 or _sha(V4_DIAG_ROOT / "freeze.json") != V4_DIAG_FREEZE_FILE_SHA256:
        raise ValueError("v4 failure-diagnosis packet identity mismatch")
    if _sha(report_path) != V4_DIAG_REPORT_FILE_SHA256 or _sha(reconciliation_path) != V4_DIAG_RECONCILIATION_FILE_SHA256:
        raise ValueError("v4 failure-diagnosis evidence bytes differ from the predeclared identity")
    report = _json(report_path)
    reconciliation = _json(reconciliation_path)
    if report.get("analysis_report_sha256") != V4_DIAG_REPORT_DIGEST or reconciliation.get("reconciliation_sha256") != V4_DIAG_RECONCILIATION_DIGEST:
        raise ValueError("v4 failure-diagnosis canonical digest fields do not match")
    if report.get("assurance_level") != "INTERNAL_PASS" or reconciliation.get("assurance_level") != "REPRODUCED_LOCAL":
        raise ValueError("v4 failure-diagnosis assurance is below the v5 entry requirement")


def freeze_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    """Archive exact source and protocol bytes before generating v5 data."""

    root = Path(custody_root).resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("experiment custody must be outside the repository")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("v5 custody must be new or empty at packet freeze")
    _verify_v4_diagnosis()
    v3_manifest = _json(V3_CUSTODY_DEFAULT / "frozen-packet-archive.json")
    if v3_manifest.get("packet_sha256") != V3_PACKET_SHA256:
        raise ValueError("pinned v3 packet identity mismatch")
    archived = root / "frozen-packet"
    rows = []
    for relative in PACKET_FILES:
        source = _root() / relative
        if not source.is_file():
            raise FileNotFoundError(f"frozen packet input is missing: {relative}")
        target = archived / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = _sha(source)
        if relative in PINNED_V3_FILES and digest != PINNED_V3_FILES[relative]:
            raise ValueError(f"pinned v3 dependency changed: {relative}")
        rows.append({"path": relative, "sha256": digest, "bytes": source.stat().st_size})
    content = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V5_ADAPTATION_FIXTURES",
        "created_at_utc": _now(),
        "packet_files": rows,
        "checkpoint_identity": {
            "model_id": "openai-community/gpt2-medium",
            "revision": "6dcaa7a952f72f9298047fd5137cd6e4f05f41da",
            "reuse_scope": "same v3/v4 GPT-2 target; no new-checkpoint or architecture-general claim",
            "projection": "transformer.h.3.mlp.c_proj",
        },
        "v3_packet_sha256": V3_PACKET_SHA256,
        "v4_diagnosis_packet_sha256": V4_DIAG_PACKET_SHA256,
        "v4_diagnosis_report_digest": V4_DIAG_REPORT_DIGEST,
        "v4_diagnosis_report_file_sha256": V4_DIAG_REPORT_FILE_SHA256,
        "v4_diagnosis_reconciliation_digest": V4_DIAG_RECONCILIATION_DIGEST,
        "v4_diagnosis_reconciliation_file_sha256": V4_DIAG_RECONCILIATION_FILE_SHA256,
    }
    content["packet_sha256"] = v3._canonical_digest(content)
    _write_json(_manifest_path(root), content)
    _write_json(_root() / "docs" / "replacement-loss-v5-freeze.json", content)
    return {"protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "freeze_state": content["freeze_state"], "packet_sha256": content["packet_sha256"], "packet_file_count": len(rows), "custody_root": str(root)}


def verify_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    manifest = _json(_manifest_path(root))
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v5 packet identity mismatch")
    rows = manifest.get("packet_files")
    if not isinstance(rows, list) or len(rows) != len(PACKET_FILES):
        raise ValueError("v5 packet file set is malformed")
    if {row.get("path") for row in rows} != set(PACKET_FILES):
        raise ValueError("v5 packet file list differs from the declared source set")
    for row in rows:
        relative = str(row["path"])
        local, archived = _root() / relative, root / "frozen-packet" / relative
        if _sha(local) != row["sha256"] or _sha(archived) != row["sha256"]:
            raise ValueError(f"v5 frozen source bytes changed: {relative}")
        if relative in PINNED_V3_FILES and row["sha256"] != PINNED_V3_FILES[relative]:
            raise ValueError(f"v5 packet does not preserve pinned v3 dependency: {relative}")
    recorded = manifest.get("packet_sha256")
    recomputed = v3._canonical_digest({key: value for key, value in manifest.items() if key != "packet_sha256"})
    if recorded != recomputed:
        raise ValueError("v5 packet digest mismatch")
    return {"packet_sha256": recorded, "packet_file_count": len(rows), "verified": True}


def _model_selection(root: Path, v3_root: Path) -> dict[str, Any]:
    packet = _json(v3_root / "frozen-packet-archive.json")
    report_path = v3_root / "replacement-loss-v3-report.json"
    report = _json(report_path)
    checkpoints = _json(v3_root / "checkpoints.json")["checkpoints"]
    entry = next(row for row in checkpoints if row["family"] == FAMILY)
    if entry["model_id"] != "openai-community/gpt2-medium" or entry["revision"] != "6dcaa7a952f72f9298047fd5137cd6e4f05f41da":
        raise ValueError("selected checkpoint differs from the preregistered target")
    case = next(row for row in report["cases"] if row["family"] == FAMILY and row["seed_index"] == 0)
    checkpoint_path = (v3_root / entry["local_path"]).resolve()
    tree_digest, _ = v3._tree_digest(checkpoint_path)
    if tree_digest != case["checkpoint_tree_sha256"]:
        raise ValueError("custodied GPT-2 checkpoint tree digest differs from v3")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": _json(_manifest_path(root))["packet_sha256"],
        "v3_packet_sha256": packet["packet_sha256"],
        "v3_primary_report_sha256": _sha(report_path),
        "family": FAMILY,
        "model_id": entry["model_id"],
        "revision": entry["revision"],
        "checkpoint_tree_sha256": tree_digest,
        "checkpoint_path": str(checkpoint_path),
        "target_projection": "transformer.h.3.mlp.c_proj",
        "selected_layer": LAYER,
        "selected_after_protocol_freeze": False,
        "fresh_checkpoint": False,
    }


def _load_checkpoint(root: Path) -> v3.Checkpoint:
    selection = _json(root / "model-selection.json")
    checkpoint = v3.Checkpoint(FAMILY, selection["model_id"], selection["revision"], Path(selection["checkpoint_path"]), False)
    current, _ = v3._tree_digest(checkpoint.local_path)
    if current != selection["checkpoint_tree_sha256"]:
        raise ValueError("checkpoint bytes changed after v5 target identity freeze")
    return checkpoint


def _capture_fit_arm(model: torch.nn.Module, tokenizer: Any, records: list[dict[str, Any]], seed: int, root: Path, arm: str) -> dict[str, Any]:
    encoded = v3._token_rows(tokenizer, records, with_continuation=False)
    input_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for _group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
        _logits, captured_inputs, captured_outputs = v3._run_projection(model, FAMILY, LAYER, inputs, positions=None)
        if captured_outputs is None:
            raise RuntimeError("adaptation capture did not retain the target projection outputs")
        for index, row in enumerate(_group):
            length = int(row["prompt_len"])
            if length < TOKENS_PER_PROMPT:
                raise ValueError("adaptation prompt is too short for 16 distinct positions")
            positions = np.rint(np.linspace(0, length - 1, TOKENS_PER_PROMPT)).astype(np.int64)
            if len(set(positions.tolist())) != TOKENS_PER_PROMPT:
                raise ValueError("position sampler failed to select 16 distinct positions")
            select = torch.as_tensor(positions, dtype=torch.long)
            input_parts.append(captured_inputs[index].index_select(0, select).float())
            target_parts.append(captured_outputs[index].index_select(0, select).float())
    x, y = torch.cat(input_parts), torch.cat(target_parts)
    if x.shape[0] != 2048 or y.shape[0] != x.shape[0]:
        raise ValueError("each candidate must use exactly 2,048 paired projection rows")
    projection = v3._projection(model, FAMILY, LAYER)
    in_features, out_features, use_bias = v3._module_dimensions(projection)
    candidate, fit_receipt = v3.fit_independent_linear_replacement(
        in_features, out_features, x, y, use_bias=use_bias, seed=seed, ridge=RIDGE
    )
    capture_path = root / "adaptation-captures" / f"seed-{seed}" / f"{arm}.npz"
    v3._save_npz_deterministic(capture_path, target_inputs=x.numpy(), target_outputs=y.numpy())
    candidate_path = root / "candidates" / f"seed-{seed}" / f"{arm}.safetensors"
    state_digest = v3._save_linear(candidate_path, candidate)
    return {
        "arm": arm,
        "seed": seed,
        "prompt_count": len(records),
        "adaptation_rows": int(x.shape[0]),
        "capture_path": capture_path.relative_to(root).as_posix(),
        "capture_sha256": _sha(capture_path),
        "candidate_path": candidate_path.relative_to(root).as_posix(),
        "candidate_file_sha256": _sha(candidate_path),
        "candidate_state_sha256": state_digest,
        "fit": fit_receipt,
    }


def prepare_candidates(custody_root: str | Path = DEFAULT_CUSTODY, v3_custody_root: str | Path = V3_CUSTODY_DEFAULT) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(custody_root).resolve()
    verify_packet(root)
    if any((root / name).exists() for name in ("candidate-preparation.json", "candidate-freeze.json", "adaptation-fixtures", "adaptation-captures", "candidates")):
        raise FileExistsError("partial or completed candidate state exists; use a fresh custody root")
    selection = _model_selection(root, Path(v3_custody_root).resolve())
    _write_json(root / "model-selection.json", selection)
    torch.set_num_threads(10)
    model, tokenizer = v3._load_local_model(_load_checkpoint(root))
    receipts: list[dict[str, Any]] = []
    fixture_hashes: dict[str, str] = {}
    for seed_index, seed in enumerate(fixtures.SEEDS):
        adaptation = fixtures.generate_adaptation_fixtures(seed_index)
        fixture_path = root / "adaptation-fixtures" / f"seed-{seed}.json"
        _write_json(fixture_path, adaptation)
        fixture_hashes[str(seed)] = _sha(fixture_path)
        for arm in v5.ARMS:
            receipts.append(_capture_fit_arm(model, tokenizer, adaptation[arm]["records"], seed, root, arm))
    runtime = _runtime()
    elapsed = time.monotonic() - started
    rss = _peak_rss_bytes()
    result = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": _json(_manifest_path(root))["packet_sha256"],
        "model_selection_sha256": _sha(root / "model-selection.json"),
        "adaptation_fixture_sha256": fixture_hashes,
        "arms": receipts,
        "runtime": runtime,
        "elapsed_wall_seconds": elapsed,
        "peak_rss_bytes": rss,
        "resource_status": "WITHIN_CAP" if rss <= RSS_CAP_BYTES and all(row["adaptation_rows"] == 2048 for row in receipts) else "INCONCLUSIVE_RESOURCE_OR_ROW_CAP",
    }
    _write_json(root / "candidate-preparation.json", result)
    return {"status": result["resource_status"], "seed_count": len(fixtures.SEEDS), "candidate_count": len(receipts), "rows_per_candidate": 2048, "elapsed_wall_seconds": elapsed, "peak_rss_bytes": rss, "adaptation_fixture_sha256": fixture_hashes}


def freeze_candidates(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "candidate-freeze.json").exists() or (root / "challenge-manifest.json").exists():
        raise FileExistsError("candidate freeze is immutable and must precede challenges")
    preparation = _json(root / "candidate-preparation.json")
    if preparation.get("resource_status") != "WITHIN_CAP":
        raise ValueError("candidate preparation did not complete within the frozen caps")
    candidate_files = []
    for item in preparation["arms"]:
        candidate_files.extend({"path": item[key], "sha256": _sha(root / item[key])} for key in ("capture_path", "candidate_path"))
    adaptation_files = [{"path": f"adaptation-fixtures/seed-{seed}.json", "sha256": _sha(root / "adaptation-fixtures" / f"seed-{seed}.json")} for seed in fixtures.SEEDS]
    freeze = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": _json(_manifest_path(root))["packet_sha256"],
        "candidate_state": "FROZEN_BEFORE_CHALLENGE_GENERATION",
        "created_at_utc": _now(),
        "candidate_files": sorted(candidate_files, key=lambda row: row["path"]),
        "adaptation_fixtures": adaptation_files,
        "candidate_preparation_sha256": _sha(root / "candidate-preparation.json"),
        "model_selection_sha256": _sha(root / "model-selection.json"),
    }
    freeze["candidate_freeze_sha256"] = v3._canonical_digest(freeze)
    _write_json(root / "candidate-freeze.json", freeze)
    return {"candidate_freeze_sha256": freeze["candidate_freeze_sha256"], "candidate_count": len(preparation["arms"]), "adaptation_fixture_count": len(adaptation_files)}


def _verify_candidate_freeze(root: Path) -> dict[str, Any]:
    freeze = _json(root / "candidate-freeze.json")
    if freeze.get("v5_packet_sha256") != _json(_manifest_path(root))["packet_sha256"]:
        raise ValueError("candidate freeze is bound to a different v5 packet")
    if freeze.get("model_selection_sha256") != _sha(root / "model-selection.json"):
        raise ValueError("model selection changed after candidate freeze")
    for item in freeze["candidate_files"] + freeze["adaptation_fixtures"]:
        if _sha(root / item["path"]) != item["sha256"]:
            raise ValueError(f"candidate or adaptation fixture changed: {Path(item['path']).name}")
    expected = freeze.get("candidate_freeze_sha256")
    if expected != v3._canonical_digest({key: value for key, value in freeze.items() if key != "candidate_freeze_sha256"}):
        raise ValueError("candidate freeze digest mismatch")
    return freeze


def _collect_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        rows = value.get("records")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict) and "input_text" in row and "continuation" in row]
        result: list[dict[str, Any]] = []
        for child in value.values():
            result.extend(_collect_records(child))
        return result
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict) and "input_text" in row and "continuation" in row]
    return []


def _history_union(v3_root: Path, v4_root: Path) -> dict[str, Any]:
    historical: list[dict[str, Any]] = []
    v3_paths = [v3_root / "fixtures" / "visible-selection.json"]
    v3_paths += [v3_root / "fixtures" / "gpt2_gelu" / f"adaptation-seed-{index}.json" for index in range(3)]
    v3_paths += [v3_root / "challenges" / "gpt2_gelu" / f"seed-{index}" / "prompts.json" for index in range(3)]
    v4_paths = [v4_root / "adaptation-fixtures" / f"seed-{seed}.json" for seed in (20260927, 20260928, 20260929)]
    v4_paths += [v4_root / "challenges" / f"seed-{seed}.json" for seed in (20260927, 20260928, 20260929)]
    for path in v3_paths + v4_paths:
        if not path.is_file():
            raise FileNotFoundError(f"required historical disjointness fixture is missing: {path.name}")
        historical.extend(_collect_records(_json(path)))
    for index, seed in enumerate(fixtures.DEVELOPMENT_ADAPTATION_SEEDS):
        development = fixtures._generate_adaptation_fixtures_for_seed(index, seed)
        for arm in development.values():
            historical.extend(arm["records"])
    for index, seed in enumerate(fixtures.DEVELOPMENT_TEST_SEEDS):
        test_adaptation = fixtures._generate_adaptation_fixtures_for_seed(index, seed)
        historical.extend(fixtures.union_adaptation_fixture(test_adaptation)["records"])
    historical.extend(fixtures.generate_challenge_fixture(1, fixtures.TEST_CHALLENGE_ENTROPY)["records"])
    unique: dict[str, dict[str, Any]] = {}
    for row in historical:
        key = str(row["example_id"])
        previous = unique.get(key)
        if previous is not None and previous != row:
            raise ValueError("historical fixtures reuse an ID for different prompt content")
        unique[key] = row
    return {"records": list(unique.values())}


def make_challenges(custody_root: str | Path = DEFAULT_CUSTODY, v3_custody_root: str | Path = V3_CUSTODY_DEFAULT, v4_custody_root: str | Path = V4_CUSTODY_DEFAULT) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    freeze = _verify_candidate_freeze(root)
    if (root / "challenge-manifest.json").exists() or (root / "challenges").exists():
        raise FileExistsError("challenge generation is immutable; use a new custody root")
    v3_root, v4_root = Path(v3_custody_root).resolve(), Path(v4_custody_root).resolve()
    historical = _history_union(v3_root, v4_root)
    fixtures_dir = root / "adaptation-fixtures"
    challenge_rows = []
    for seed_index, seed in enumerate(fixtures.SEEDS):
        adaptation = _json(fixtures_dir / f"seed-{seed}.json")
        union = fixtures.union_adaptation_fixture(adaptation)
        entropy = secrets.token_bytes(32)
        challenge = fixtures.generate_challenge_fixture(seed_index, entropy)
        fixtures.validate_partition_disjointness([historical, union, challenge])
        path = root / "challenges" / f"seed-{seed}.json"
        _write_json(path, challenge)
        challenge_rows.append({"seed": seed, "path": path.relative_to(root).as_posix(), "sha256": _sha(path), "example_set_sha256": challenge["example_set_sha256"], "entropy_commitment_sha256": hashlib.sha256(entropy).hexdigest(), "count": challenge["count"], "stratum_counts": challenge["stratum_counts"]})
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": _json(_manifest_path(root))["packet_sha256"],
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "candidate_freeze_file_sha256": _sha(root / "candidate-freeze.json"),
        "challenge_generation_started_after_candidate_freeze": True,
        "history_disjointness_source": "v3 and v4 custodied fixtures plus v5 adaptation union",
        "historical_fixture_root_count": 13,
        "excluded_development_fixture_groups": 6,
        "challenges": challenge_rows,
        "created_at_utc": _now(),
    }
    manifest["challenge_manifest_sha256"] = v3._canonical_digest(manifest)
    _write_json(root / "challenge-manifest.json", manifest)
    return {"challenge_count": len(challenge_rows), "rows_per_seed": 60, "challenge_manifest_sha256": manifest["challenge_manifest_sha256"], "challenge_fixture_sha256": [{"seed": row["seed"], "sha256": row["example_set_sha256"]} for row in challenge_rows]}


def _verify_challenges(root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    manifest = _json(root / "challenge-manifest.json")
    if manifest.get("v5_packet_sha256") != freeze["v5_packet_sha256"] or manifest.get("candidate_freeze_sha256") != freeze["candidate_freeze_sha256"]:
        raise ValueError("challenge manifest is bound to a different packet or candidate")
    if not manifest.get("challenge_generation_started_after_candidate_freeze"):
        raise ValueError("challenges did not follow candidate freeze")
    if len(manifest.get("challenges", [])) != len(fixtures.SEEDS):
        raise ValueError("challenge fixture count differs from the frozen protocol")
    for item in manifest["challenges"]:
        if _sha(root / item["path"]) != item["sha256"]:
            raise ValueError("challenge fixture digest mismatch")
        fixture = _json(root / item["path"])
        if fixture.get("example_set_sha256") != item["example_set_sha256"] or fixture.get("stratum_counts") != {"adversarial": 20, "compositional": 20, "ordinary": 20}:
            raise ValueError("challenge fixture coverage or identity mismatch")
    expected = manifest.get("challenge_manifest_sha256")
    actual = v3._canonical_digest({key: value for key, value in manifest.items() if key != "challenge_manifest_sha256"})
    if expected != actual:
        raise ValueError("challenge manifest digest mismatch")
    return manifest


def predict(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "prediction-freeze.json").exists() or (root / "predictions").exists():
        raise FileExistsError("prediction sealing is immutable; use a fresh custody root")
    freeze = _verify_candidate_freeze(root)
    challenge_manifest = _verify_challenges(root, freeze)
    torch.set_num_threads(10)
    model, tokenizer = v3._load_local_model(_load_checkpoint(root))
    receipts = []
    for seed_index, seed in enumerate(fixtures.SEEDS):
        challenge = _json(root / "challenges" / f"seed-{seed}.json")
        records = challenge["records"]
        encoded = v3._token_rows(tokenizer, records, with_continuation=True)
        payload_parts: dict[str, list[np.ndarray]] = {"example_ids": [], "strata": []}
        for arm in v5.ARMS:
            payload_parts[f"{arm}_activation"] = []
            payload_parts[f"{arm}_next_logits"] = []
            payload_parts[f"{arm}_ce"] = []
        candidates = {}
        for arm in v5.ARMS:
            projection = v3._projection(model, FAMILY, LAYER)
            in_features, out_features, use_bias = v3._module_dimensions(projection)
            candidates[arm] = v3._load_linear(root / "candidates" / f"seed-{seed}" / f"{arm}.safetensors", in_features, out_features, use_bias)
        for group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
            payload_parts["example_ids"].append(np.asarray([row["example_id"] for row in group], dtype="U24"))
            payload_parts["strata"].append(np.asarray([row["stratum"] for row in group], dtype="U24"))
            for arm, candidate in candidates.items():
                logits, activation = v3._model_forward_variant(model, FAMILY, LAYER, candidate, group, inputs, deltas=None)
                payload_parts[f"{arm}_activation"].append(activation.numpy())
                payload_parts[f"{arm}_next_logits"].append(v3._next_logits(logits, group).numpy())
                payload_parts[f"{arm}_ce"].append(v3._continuation_cross_entropy(logits, group))
        payload = {name: np.concatenate(parts, axis=0) for name, parts in payload_parts.items()}
        expected_ids = np.asarray([row["example_id"] for row in records], dtype="U24")
        if not np.array_equal(payload["example_ids"], expected_ids):
            raise ValueError("prediction rows changed challenge order")
        prediction_path = root / "predictions" / f"seed-{seed}.npz"
        v3._save_npz_deterministic(prediction_path, **payload)
        receipt = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "seed": seed,
            "v5_packet_sha256": freeze["v5_packet_sha256"],
            "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
            "challenge_fixture_sha256": challenge["example_set_sha256"],
            "prediction_file_sha256": _sha(prediction_path),
            "prediction_query_count": len(records) * len(v5.ARMS),
            "original_outcomes_opened": False,
            "sealed_at_utc": _now(),
        }
        receipt_path = root / "predictions" / f"seed-{seed}.receipt.json"
        _write_json(receipt_path, receipt)
        receipts.append({"seed": seed, "prediction_file_sha256": _sha(prediction_path), "receipt_sha256": _sha(receipt_path)})
    seal = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": freeze["v5_packet_sha256"],
        "candidate_freeze_sha256": freeze["candidate_freeze_sha256"],
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "predictions": receipts,
        "all_predictions_sealed_before_original_outcomes": True,
        "created_at_utc": _now(),
    }
    seal["prediction_freeze_sha256"] = v3._canonical_digest(seal)
    _write_json(root / "prediction-freeze.json", seal)
    return {"seed_count": len(receipts), "prediction_freeze_sha256": seal["prediction_freeze_sha256"], "prediction_digests": receipts}


def _report_metrics_digest(report: dict[str, Any]) -> str:
    material = [{"seed": case["seed"], "status": case["case_status"], "metrics": case["metrics"]} for case in report["cases"]]
    return v3._canonical_digest(material)


def evaluate(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(custody_root).resolve()
    verify_packet(root)
    if (root / "replacement-loss-v5-report.json").exists() or (root / "outcomes").exists():
        raise FileExistsError("v5 evaluation is immutable; use a fresh custody root")
    freeze = _verify_candidate_freeze(root)
    challenge_manifest = _verify_challenges(root, freeze)
    prediction_freeze = _json(root / "prediction-freeze.json")
    if prediction_freeze.get("all_predictions_sealed_before_original_outcomes") is not True:
        raise ValueError("all candidate predictions must be sealed before outcomes are opened")
    torch.set_num_threads(10)
    model, tokenizer = v3._load_local_model(_load_checkpoint(root))
    cases, prediction_hashes, outcome_hashes = [], [], []
    for seed_index, seed in enumerate(fixtures.SEEDS):
        case_started = time.monotonic()
        challenge = _json(root / "challenges" / f"seed-{seed}.json")
        records = challenge["records"]
        encoded = v3._token_rows(tokenizer, records, with_continuation=True)
        acts, logits_rows, ce_rows = [], [], []
        for group, inputs in v3._batches(encoded, tokenizer.pad_token_id, batch_size=BATCH_SIZE):
            original_logits, _module_inputs, original_acts = v3._model_forward_original(model, FAMILY, LAYER, group, inputs)
            acts.append(original_acts.numpy())
            logits_rows.append(v3._next_logits(original_logits, group).numpy())
            ce_rows.append(v3._continuation_cross_entropy(original_logits, group))
        original_activation, original_logits, original_ce = np.concatenate(acts), np.concatenate(logits_rows), np.concatenate(ce_rows)
        prediction_path = root / "predictions" / f"seed-{seed}.npz"
        with np.load(prediction_path, allow_pickle=False) as data:
            predictions = {name: data[name] for name in data.files}
        strata = predictions["strata"].astype(str)
        expected_ids = np.asarray([row["example_id"] for row in records], dtype="U24")
        if not np.array_equal(predictions["example_ids"], expected_ids):
            raise ValueError("prediction and evaluator challenge row order differs")
        candidate_activation = {arm: predictions[f"{arm}_activation"] for arm in v5.ARMS}
        candidate_logits = {arm: predictions[f"{arm}_next_logits"] for arm in v5.ARMS}
        candidate_ce = {arm: predictions[f"{arm}_ce"] for arm in v5.ARMS}
        metrics = v5.evaluate_seed(strata, original_activation, original_logits, original_ce, candidate_activation, candidate_logits, candidate_ce)
        outcome_path = root / "outcomes" / f"seed-{seed}.npz"
        v3._save_npz_deterministic(outcome_path, example_ids=expected_ids, strata=strata.astype("U24"), original_module_activation=original_activation, original_next_logits=original_logits, original_ce=original_ce)
        elapsed, rss = time.monotonic() - case_started, _peak_rss_bytes()
        gate_pass = v5.seed_pass(metrics)
        within_caps = elapsed <= CASE_WALL_CAP_SECONDS and rss <= RSS_CAP_BYTES
        cases.append({
            "seed": seed,
            "challenge_fixture_sha256": challenge["example_set_sha256"],
            "prediction_array_sha256": _sha(prediction_path),
            "outcome_array_sha256": _sha(outcome_path),
            "metrics": metrics,
            "case_status": "PASS" if gate_pass and within_caps else "FAIL" if within_caps else "INCONCLUSIVE",
            "elapsed_wall_seconds": elapsed,
            "peak_rss_bytes": rss,
        })
        prediction_hashes.append({"seed": seed, "sha256": _sha(prediction_path)})
        outcome_hashes.append({"seed": seed, "sha256": _sha(outcome_path)})
    statuses = {case["case_status"] for case in cases}
    overall = "PASS" if statuses == {"PASS"} else "INCONCLUSIVE" if "INCONCLUSIVE" in statuses else "FAIL"
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": freeze["v5_packet_sha256"],
        "v4_diagnosis_packet_sha256": V4_DIAG_PACKET_SHA256,
        "v4_diagnosis_report_digest": V4_DIAG_REPORT_DIGEST,
        "v4_diagnosis_report_file_sha256": V4_DIAG_REPORT_FILE_SHA256,
        "source": {
            "git_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=_root(), check=True, capture_output=True, text=True).stdout.strip(),
            "worktree_status_sha256": hashlib.sha256(subprocess.run(["git", "status", "--porcelain=v1"], cwd=_root(), check=True, capture_output=True, text=True).stdout.encode()).hexdigest(),
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
        "prediction_freeze_sha256": prediction_freeze.get("prediction_freeze_sha256"),
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "runtime": _runtime(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": _peak_rss_bytes(),
        "claim_ceiling": "LOCAL_GPT2_BALANCED_ADVERSARIAL_ADAPTATION_EFFECT",
        "nonclaims": ["fresh-checkpoint generalization", "all GPT-2 checkpoints", "all architectures", "mechanistic understanding", "independent science", "alignment", "consciousness", "production authorization"],
        "data_and_artifact_custody": "prompts, model assets, captures, candidates, predictions, outcomes, and row-level diagnostics remain in external owner custody",
        "next_blocked_authority_gate": "clean second-runtime reproduction, then configured external reviewer for any independent claim",
        "monetary_cost": "NOT_MEASURED",
        "energy": "NOT_MEASURED",
    }
    _write_json(root / "replacement-loss-v5-report.json", report)
    return {
        "outcome_status": overall,
        "case_count": len(cases),
        "pass_count": report["pass_count"],
        "fail_count": report["fail_count"],
        "inconclusive_count": report["inconclusive_count"],
        "report_path": str(root / "replacement-loss-v5-report.json"),
        "metrics_digest": _report_metrics_digest(report),
        "cases": [{"seed": case["seed"], "status": case["case_status"], "paired_comparisons": case["metrics"]["paired_comparisons"]} for case in cases],
    }


def replicate(primary_root: str | Path, replica_root: str | Path, v3_custody_root: str | Path = V3_CUSTODY_DEFAULT, v4_custody_root: str | Path = V4_CUSTODY_DEFAULT) -> dict[str, Any]:
    """Rebuild candidates and rerun the frozen challenge in another runtime."""

    started = time.monotonic()
    primary, replica = Path(primary_root).resolve(), Path(replica_root).resolve()
    if primary == replica or primary in replica.parents or replica in primary.parents:
        raise ValueError("primary and replica custody roots must be separate")
    verify_packet(primary)
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("replica custody must be new or empty")
    replica.mkdir(parents=True, exist_ok=True)
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(_manifest_path(primary), _manifest_path(replica))
    shutil.copyfile(primary / "model-selection.json", replica / "model-selection.json")
    verify_packet(replica)
    prepare_candidates(replica, v3_custody_root)
    primary_freeze = _verify_candidate_freeze(primary)
    replica_candidates = {row["path"]: _sha(replica / row["path"]) for row in primary_freeze["candidate_files"]}
    primary_candidates = {row["path"]: row["sha256"] for row in primary_freeze["candidate_files"]}
    adaptation_match = all(_sha(replica / row["path"]) == row["sha256"] for row in primary_freeze["adaptation_fixtures"])
    if primary_candidates != replica_candidates or not adaptation_match:
        raise ValueError("replica candidate state differs from the primary frozen candidate state")
    shutil.copyfile(primary / "candidate-freeze.json", replica / "candidate-freeze.json")
    _verify_candidate_freeze(replica)
    for seed in fixtures.SEEDS:
        target = replica / "challenges" / f"seed-{seed}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary / "challenges" / target.name, target)
    shutil.copyfile(primary / "challenge-manifest.json", replica / "challenge-manifest.json")
    prediction_result = predict(replica)
    evaluate(replica)
    primary_report = _json(primary / "replacement-loss-v5-report.json")
    replica_report = _json(replica / "replacement-loss-v5-report.json")
    prediction_match = primary_report["prediction_arrays"] == replica_report["prediction_arrays"]
    outcome_match = primary_report["outcome_arrays"] == replica_report["outcome_arrays"]
    metrics_match = _report_metrics_digest(primary_report) == _report_metrics_digest(replica_report)
    source_match = primary_report["v5_packet_sha256"] == replica_report["v5_packet_sha256"] and primary_freeze["candidate_freeze_sha256"] == _verify_candidate_freeze(replica)["candidate_freeze_sha256"]
    runtime_primary, runtime_replica = primary_report["runtime"], replica_report["runtime"]
    dependencies_match = runtime_primary["dependency_inventory_sha256"] == runtime_replica["dependency_inventory_sha256"]
    distinct_runtime = runtime_primary["python_prefix"] != runtime_replica["python_prefix"]
    reproducible = all((prediction_match, outcome_match, metrics_match, source_match, dependencies_match, distinct_runtime, adaptation_match))
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v5_packet_sha256": primary_report["v5_packet_sha256"],
        "primary_report_sha256": _sha(primary / "replacement-loss-v5-report.json"),
        "replica_report_sha256": _sha(replica / "replacement-loss-v5-report.json"),
        "candidate_reconstruction_match": primary_candidates == replica_candidates,
        "adaptation_fixture_digests_match": adaptation_match,
        "prediction_array_digests_match": prediction_match,
        "outcome_array_digests_match": outcome_match,
        "aggregate_metrics_match": metrics_match,
        "source_identity_match": source_match,
        "dependencies_match": dependencies_match,
        "distinct_clean_runtime": distinct_runtime,
        "same_owner_second_environment": True,
        "assurance_level": "REPRODUCED_LOCAL" if reproducible else "INCONCLUSIVE",
        "reproducibility_status": "REPRODUCED_LOCAL" if reproducible else "INCONCLUSIVE",
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "primary_outcome_status": primary_report["outcome_status"],
        "replica_outcome_status": replica_report["outcome_status"],
        "next_blocked_authority_gate": "Configured external reviewer with independent authority; no acceptance recorded.",
        "nonclaims": primary_report["nonclaims"],
    }
    _write_json(replica / "replication-report.json", receipt)
    return {**receipt, "replication_report_path": str(replica / "replication-report.json"), "replication_report_sha256": _sha(replica / "replication-report.json"), "replica_elapsed_wall_seconds": time.monotonic() - started, "replica_prediction_freeze_sha256": prediction_result["prediction_freeze_sha256"]}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run frozen replacement-loss-v5 stages")
    parser.add_argument("command", choices=("freeze-packet", "prepare", "freeze-candidates", "make-challenges", "predict", "evaluate", "replicate"))
    parser.add_argument("--custody-root", default=str(DEFAULT_CUSTODY))
    parser.add_argument("--v3-custody-root", default=str(V3_CUSTODY_DEFAULT))
    parser.add_argument("--v4-custody-root", default=str(V4_CUSTODY_DEFAULT))
    parser.add_argument("--replica-root")
    args = parser.parse_args()
    if args.command == "freeze-packet":
        result = freeze_packet(args.custody_root)
    elif args.command == "prepare":
        result = prepare_candidates(args.custody_root, args.v3_custody_root)
    elif args.command == "freeze-candidates":
        result = freeze_candidates(args.custody_root)
    elif args.command == "make-challenges":
        result = make_challenges(args.custody_root, args.v3_custody_root, args.v4_custody_root)
    elif args.command == "predict":
        result = predict(args.custody_root)
    elif args.command == "evaluate":
        result = evaluate(args.custody_root)
    else:
        if not args.replica_root:
            parser.error("replicate requires --replica-root")
        result = replicate(args.custody_root, args.replica_root, args.v3_custody_root, args.v4_custody_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
