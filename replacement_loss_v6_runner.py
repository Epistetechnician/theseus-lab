"""Frozen-packet orchestration for replacement-loss-v6.

State slice: theseus-lab-sandbox-v1. v3's exact capture and blind evaluator
are pinned dependencies; the v6 wrapper changes candidate fitting and report
interpretation without editing v3-v5 source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import replacement_loss_v3_runner as v3
import replacement_loss_v6 as v6
import replacement_loss_v6_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v6-downstream-aware"
V3_PACKET_SHA256 = "d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83"
V6_PACKET_FILES = (
    "docs/replacement-loss-v6.md",
    "replacement_loss_v6.py",
    "replacement_loss_v6_fixtures.py",
    "replacement_loss_v6_runner.py",
    "tests/test_replacement_loss_v6.py",
    "requirements.txt",
)
V3_PACKET_FILES = (*v3.PACKET_FILES, v3.PACKET_FREEZE)
DEFAULT_CUSTODY = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v6")
DEFAULT_REPLICA = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v6-replicator")
PRIOR_CHECKPOINT_IDS = {
    "gpt2", "distilgpt2", "openai-community/gpt2", "openai-community/distilgpt2",
    "openai-community/gpt2-medium", "openai-community/gpt2-large", "gpt2-large",
    "facebook/opt-125m", "opt-125m", "facebook/opt-350m", "facebook/opt-1.3b",
    "opt-1.3b", "tinyllama/tinyllama-1.1b-chat-v1.0",
    "huggingfacetb/smollm-135m", "huggingfacetb/smollm2-135m",
    "huggingfacetb/smollm-360m", "google-t5/t5-base",
    "state-spaces/mamba-370m-hf",
}
_ORIGINAL_FIT_CASE = v3._fit_case


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
        raise ValueError("v6 custody must be outside the repository")
    return root


def _verify_v3_packet() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    verified = v3.verify_packet()
    if verified["packet_sha256"] != V3_PACKET_SHA256:
        raise ValueError("pinned v3 packet identity changed")
    freeze = _read_json(_root() / v3.PACKET_FREEZE)
    if freeze.get("packet_sha256") != V3_PACKET_SHA256:
        raise ValueError("v3 freeze manifest identity changed")
    rows = list(freeze["packet_files"])
    for row in rows:
        if _sha(_root() / str(row["path"])) != row["sha256"]:
            raise ValueError(f"pinned v3 source changed: {row['path']}")
    freeze_path = _root() / v3.PACKET_FREEZE
    return freeze, rows + [{"path": v3.PACKET_FREEZE, "sha256": _sha(freeze_path), "bytes": freeze_path.stat().st_size}]


def freeze_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    """Freeze and archive exact v6 and pinned v3 bytes before model selection."""

    root = _custody_root(custody_root)
    local_freeze = _root() / "docs" / "replacement-loss-v6-freeze.json"
    archive_manifest = root / "v6-frozen-packet-manifest.json"
    archive = root / "v6-frozen-packet"
    if local_freeze.exists() or archive_manifest.exists() or archive.exists():
        raise FileExistsError("v6 packet freeze is immutable; use a new protocol version")
    v3_freeze, v3_rows = _verify_v3_packet()
    all_files = tuple(dict.fromkeys((*V6_PACKET_FILES, *(row["path"] for row in v3_rows))))
    missing = [relative for relative in all_files if not (_root() / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v6 packet inputs are missing: {', '.join(missing)}")
    packet_rows: list[dict[str, Any]] = []
    for relative in all_files:
        source = _root() / relative
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        packet_rows.append({"path": relative, "sha256": _sha(source), "bytes": source.stat().st_size})
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V6_CHECKPOINT_SELECTION",
        "created_at_utc": _now(),
        "packet_files": packet_rows,
        "v3_packet_sha256": v3_freeze["packet_sha256"],
        "checkpoint_identities_selected": False,
        "challenge_fixtures_generated": False,
    }
    payload["packet_sha256"] = _canonical_digest(payload)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(archive_manifest, payload)
    _write_json(local_freeze, payload)
    return {"protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "packet_sha256": payload["packet_sha256"], "packet_file_count": len(packet_rows)}


def verify_packet(custody_root: str | Path = DEFAULT_CUSTODY) -> dict[str, Any]:
    """Verify local packet and external archived bytes against the freeze."""

    root = _custody_root(custody_root)
    manifest = _read_json(root / "v6-frozen-packet-manifest.json")
    local_manifest = _read_json(_root() / "docs" / "replacement-loss-v6-freeze.json")
    if manifest != local_manifest:
        raise ValueError("local and custodied v6 freeze manifests differ")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v6 packet identity mismatch")
    recorded = manifest.get("packet_sha256")
    expected = _canonical_digest({key: value for key, value in manifest.items() if key != "packet_sha256"})
    if recorded != expected or manifest.get("v3_packet_sha256") != V3_PACKET_SHA256:
        raise ValueError("v6 packet digest or pinned v3 identity mismatch")
    for row in manifest["packet_files"]:
        relative = str(row["path"])
        local = _root() / relative
        archived = root / "v6-frozen-packet" / relative
        if _sha(local) != row["sha256"] or _sha(archived) != row["sha256"]:
            raise ValueError(f"frozen v6 packet bytes changed: {relative}")
    _verify_v3_packet()
    return {"packet_sha256": recorded, "packet_file_count": len(manifest["packet_files"]), "verified": True}


def write_checkpoint_manifest(
    custody_root: str | Path,
    checkpoint_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Bind exact checkpoint identities after v6 packet freeze."""

    root = _custody_root(custody_root)
    v6_manifest = verify_packet(root)
    v3_freeze, _v3_rows = _verify_v3_packet()
    destination = root / "checkpoints.json"
    if destination.exists():
        raise FileExistsError("checkpoint selection is immutable once recorded")
    expected_families = set(v3.FAMILIES)
    if len(checkpoint_specs) != len(expected_families) or {row.get("family") for row in checkpoint_specs} != expected_families:
        raise ValueError("checkpoint selection must contain exactly one entry per frozen family")
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
            "revision": revision,
            "local_path": local_path.as_posix(),
            "selected_after_protocol_freeze": True,
            "not_used_for_protocol_design": True,
            "base_model": True,
        })
    payload = {
        "protocol_id": v3.PROTOCOL_ID,
        "v6_protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": v3_freeze["packet_sha256"],
        "v6_packet_sha256": v6_manifest["packet_sha256"],
        "selected_at_utc": _now(),
        "checkpoints": sorted(entries, key=lambda item: v3.FAMILIES.index(item["family"])),
    }
    temporary = root / ".checkpoint-selection-preflight.json"
    try:
        _write_json(temporary, payload)
        with _v6_profile():
            v3._load_checkpoints(temporary, root)
    finally:
        temporary.unlink(missing_ok=True)
    _write_json(destination, payload)
    return {"protocol_id": PROTOCOL_ID, "v6_packet_sha256": v6_manifest["packet_sha256"], "checkpoint_manifest_sha256": _sha(destination), "checkpoint_count": len(entries)}


def _fit_case_v6(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint: Any,
    layer: int,
    seed_index: int,
    fixture_records: list[dict[str, Any]],
    custody: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case = _ORIGINAL_FIT_CASE(model, tokenizer, checkpoint, layer, seed_index, fixture_records, custody)
    by_name = {row["name"]: row for row in case["modules"]}
    ridge_row = by_name["treatment"]
    ridge_source = custody / ridge_row["file"]
    baseline_path = ridge_source.parent / "output-only-ridge.safetensors"
    shutil.copyfile(ridge_source, baseline_path)
    ridge_fit = case["treatment_fit"]

    treatment, training_receipt = v6.train_behavior_aware_replacement(
        model,
        tokenizer,
        checkpoint.family,
        layer,
        fixture_records,
        seed=fixtures.SEEDS[seed_index],
    )
    treatment_path = custody / by_name["treatment"]["file"]
    treatment_state_sha = v3._save_linear(treatment_path, treatment)
    if treatment_state_sha != training_receipt["candidate_state_sha256"]:
        raise ValueError("treatment state digest differs between fitter and safetensors receipt")

    variants, permutation, aligned_error = v3._make_variant_modules(treatment, fixtures.SEEDS[seed_index] + 303)
    raw_row = by_name["raw_index_permutation"]
    aligned_row = by_name["mapping_aligned_permutation"]
    raw_path, aligned_path = custody / raw_row["file"], custody / aligned_row["file"]
    raw_sha = v3._save_linear(raw_path, variants["raw_index_permutation"])
    aligned_sha = v3._save_linear(aligned_path, variants["raw_index_permutation"])

    original_wrong_path = custody / by_name["wrong_module"]["file"]
    original_wrong_sha = by_name["wrong_module"]["state_sha256"]
    if original_wrong_path != baseline_path:
        original_wrong_path.unlink(missing_ok=True)
    ridge_candidate = v3._load_linear(
        baseline_path,
        int(case["input_features"]),
        int(case["output_features"]),
        bool(case["use_bias"]),
    )
    baseline_sha = v3._state_digest(ridge_candidate.state_dict())

    by_name["treatment"]["state_sha256"] = treatment_state_sha
    by_name["wrong_module"].update({
        "file": baseline_path.relative_to(custody).as_posix(),
        "state_sha256": baseline_sha,
        "name": "wrong_module",
    })
    by_name["raw_index_permutation"]["state_sha256"] = raw_sha
    by_name["mapping_aligned_permutation"]["state_sha256"] = aligned_sha
    case["treatment_fit"] = training_receipt
    case["output_only_ridge_fit"] = ridge_fit
    case["wrong_module_fit"] = ridge_fit
    case["v6_control_aliases"] = {"wrong_module": "output_only_ridge"}
    case["v6_unused_wrong_module_candidate_state_sha256"] = original_wrong_sha
    case["raw_index_permutation"] = permutation
    case["mapping_aligned_equivalence_max_abs_error"] = aligned_error
    case["fit_elapsed_wall_seconds"] = float(case["fit_elapsed_wall_seconds"]) + (time.monotonic() - started)
    return case


@contextmanager
def _v6_profile() -> Iterator[None]:
    """Apply v6 fixtures, seeds, exclusions, and candidate fitter temporarily."""

    names = (
        "SEEDS", "PRIOR_CHECKPOINT_IDS", "generate_visible_fixture",
        "generate_adaptation_fixture", "generate_challenge_fixture",
        "fixture_payload", "validate_partition_disjointness", "_fit_case",
    )
    saved = {name: getattr(v3, name) for name in names}
    try:
        v3.SEEDS = fixtures.SEEDS
        v3.PRIOR_CHECKPOINT_IDS = set(v3.PRIOR_CHECKPOINT_IDS) | PRIOR_CHECKPOINT_IDS
        v3.generate_visible_fixture = fixtures.generate_visible_fixture
        v3.generate_adaptation_fixture = fixtures.generate_adaptation_fixture
        v3.generate_challenge_fixture = fixtures.generate_challenge_fixture
        v3.fixture_payload = fixtures.fixture_payload
        v3.validate_partition_disjointness = fixtures.validate_partition_disjointness
        v3._fit_case = _fit_case_v6
        yield
    finally:
        for name, value in saved.items():
            setattr(v3, name, value)


def _fixture_payloads_for_preflight() -> list[dict[str, Any]]:
    visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible_selection")
    adaptation = [
        fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation_{fixtures.SEEDS[index]}")
        for index in range(len(fixtures.SEEDS))
    ]
    fixtures.validate_partition_disjointness(visible, *adaptation)
    for index, payload in enumerate(adaptation):
        fixtures.assert_stratum_counts(payload, fixtures.ADAPTATION_PROMPTS)
    return [visible, *adaptation]


def _read_prior_fixture_payloads(custody_roots: list[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    payloads_by_digest: dict[str, dict[str, Any]] = {}
    for value in custody_roots:
        root = Path(value).expanduser().resolve()
        if not root.is_dir() or root == _root() or _root() in root.parents:
            raise ValueError("each prior custody root must be an existing external directory")
        for path in root.rglob("*.json"):
            try:
                payload = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            records = payload.get("records")
            if not isinstance(records, list) or not records:
                continue
            if not all(isinstance(row, dict) and {"example_id", "input_text", "continuation"} <= row.keys() for row in records):
                continue
            digest = str(payload.get("example_set_sha256") or _canonical_digest(records))
            payloads_by_digest.setdefault(digest, {"records": records, "example_set_sha256": digest})
    if not payloads_by_digest:
        raise ValueError("no prior custodied prompt fixtures were found for disjointness checks")
    return list(payloads_by_digest.values()), sorted(payloads_by_digest)


def _validate_all_disjoint(
    custody_roots: list[str | Path],
    current_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    prior_payloads, prior_digests = _read_prior_fixture_payloads(custody_roots)
    # Historical protocols can intentionally share a fixture across multiple
    # arms. Check current v6 partitions against each other, then compare them
    # with the union of historical identities/content.
    fixtures.validate_partition_disjointness(*current_payloads)
    prior_ids: set[str] = set()
    prior_prompts: set[str] = set()
    prior_content: set[str] = set()
    for payload in prior_payloads:
        for row in payload["records"]:
            prior_ids.add(str(row["example_id"]))
            prompt = str(row["input_text"])
            prior_prompts.add(prompt)
            prior_content.add(_canonical_digest([prompt, str(row["continuation"])]))
    for payload in current_payloads:
        for row in payload["records"]:
            prompt = str(row["input_text"])
            if (
                str(row["example_id"]) in prior_ids
                or prompt in prior_prompts
                or _canonical_digest([prompt, str(row["continuation"])]) in prior_content
            ):
                raise ValueError("current v6 fixtures overlap prior custodied fixture content")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "prior_fixture_count": len(prior_payloads),
        "prior_fixture_digests_sha256": _canonical_digest(prior_digests),
        "current_fixture_digests": [str(item["example_set_sha256"]) for item in current_payloads],
        "all_partitions_disjoint": True,
        "checked_at_utc": _now(),
    }


def _validate_checkpoint_manifest(custody: Path, manifest_path: Path, packet_sha: str) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("v6_protocol_id") != PROTOCOL_ID or manifest.get("v6_packet_sha256") != packet_sha:
        raise ValueError("checkpoint manifest does not bind the frozen v6 packet")
    if manifest_path.parent != custody:
        raise ValueError("checkpoint manifest must be directly under external custody")
    if manifest.get("v6_packet_sha256") != verify_packet(custody)["packet_sha256"]:
        raise ValueError("checkpoint selection does not follow v6 packet freeze")
    for item in manifest.get("checkpoints", []):
        if str(item.get("model_id", "")).lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError("a previously used checkpoint identity cannot be selected for v6")
        if item.get("not_used_for_protocol_design") is not True or item.get("base_model") is not True:
            raise ValueError("v6 requires fresh public base checkpoints selected after freeze")
    with _v6_profile():
        v3._load_checkpoints(manifest_path, custody)
    return manifest


def _add_v6_binding_to_challenge_manifest(custody: Path, packet_sha: str) -> dict[str, Any]:
    path = custody / "challenge-manifest.json"
    manifest = _read_json(path)
    manifest["v6_protocol_id"] = PROTOCOL_ID
    manifest["v6_packet_sha256"] = packet_sha
    manifest.pop("challenge_manifest_sha256", None)
    manifest["challenge_manifest_sha256"] = _canonical_digest(manifest)
    _write_json(path, manifest)
    return manifest


def _build_v6_report(custody: Path, packet_sha: str, disjointness: dict[str, Any]) -> dict[str, Any]:
    inner_path = custody / "replacement-loss-v3-report.json"
    inner = _read_json(inner_path)
    preparation = _read_json(custody / "candidate-preparation.json")
    candidate_freeze = _read_json(custody / "candidate-freeze.json")
    challenge_manifest = _read_json(custody / "challenge-manifest.json")
    cases = []
    improvements = []
    family_improvements: dict[str, list[float]] = {family: [] for family in v3.FAMILIES}
    for case in inner["cases"]:
        behavior = case["behavior"]
        treatment_error = float(behavior["treatment"]["paired_next_token_normalized_logit_error_max"])
        ridge_error = float(behavior["wrong_module"]["paired_next_token_normalized_logit_error_max"])
        improvement = ridge_error - treatment_error
        improvements.append(improvement)
        family_improvements[str(case["family"])].append(improvement)
        aligned = case["mapping_aligned_permutation_diagnostic"]
        aligned_blockers = []
        if aligned["module_effect_max_abs_error_vs_treatment"] > v3.EQUIVALENCE_MAX_ABS_ERROR:
            aligned_blockers.append("mapping_aligned_module_equivalence")
        if aligned["logit_effect_max_abs_error_vs_treatment"] > v3.EQUIVALENCE_MAX_ABS_ERROR:
            aligned_blockers.append("mapping_aligned_logit_equivalence")
        blockers = list(case["blockers"])
        blockers.extend(item for item in aligned_blockers if item not in blockers)
        cases.append({
            "family": case["family"],
            "seed_index": case["seed_index"],
            "seed": case["seed"],
            "v3_treatment_case_status": case["case_status"],
            "case_status": "FAIL" if aligned_blockers and case["case_status"] == "PASS" else case["case_status"],
            "blockers": blockers,
            "treatment_max_normalized_next_logit_error": treatment_error,
            "ridge_max_normalized_next_logit_error": ridge_error,
            "ridge_minus_treatment_error": improvement,
            "treatment_activation_rescue": case["activation_rescue"],
            "treatment_primary_effects": case["primary_effects"]["treatment"],
            "aligned_permutation_diagnostic": aligned,
            "full_case_elapsed_wall_seconds": case["full_case_elapsed_wall_seconds"],
            "prediction_array_sha256": case["prediction_array_sha256"],
            "outcome_array_sha256": case["outcome_array_sha256"],
        })
    if len(improvements) != 9:
        raise ValueError("v6 result must contain all nine frozen checkpoint-by-seed cases")
    lower = v3.bootstrap_mean_difference_lower_bound(
        np.asarray(improvements, dtype=np.float64),
        np.zeros(len(improvements), dtype=np.float64),
        seed=20261125,
        draws=10_000,
        quantile=0.025,
    )
    mean_improvement = float(np.mean(improvements))
    family_means = {family: float(np.mean(values)) for family, values in family_improvements.items()}
    improvement_pass = (
        mean_improvement >= 0.005
        and float(lower) > 0
        and all(value >= 0 for value in family_means.values())
    )
    all_case_gates = all(row["case_status"] == "PASS" for row in inner["cases"])
    inner_outcome = str(inner.get("outcome_status"))
    outcome = "PASS" if inner_outcome == "PASS" and improvement_pass else (
        "INCONCLUSIVE" if inner_outcome in {"INCOMPLETE", "INCONCLUSIVE"} else "FAIL"
    )
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_level": "INTERNAL_PASS" if outcome == "PASS" else "INCONCLUSIVE",
        "outcome_status": outcome,
        "v3_gate_outcome_status": inner_outcome,
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "reproducibility_status": "PENDING_SEPARATE_REPLICATOR",
        "packet_sha256": packet_sha,
        "pinned_v3_packet_sha256": V3_PACKET_SHA256,
        "source_identity": inner.get("source"),
        "checkpoint_manifest_sha256": inner.get("checkpoint_manifest_sha256"),
        "candidate_files_sha256": inner.get("candidate_files_sha256"),
        "candidate_preparation_sha256": candidate_freeze.get("candidate_preparation_sha256"),
        "candidate_freeze_sha256": _sha(custody / "candidate-freeze.json"),
        "challenge_manifest_sha256": inner.get("challenge_manifest_sha256"),
        "fixture_identities": {
            "visible_selection_sha256": sorted({
                str(row["visible_fixture_sha256"]) for row in preparation["cases"]
            }),
            "adaptation_sha256_by_case": [
                {
                    "family": row["family"],
                    "seed_index": row["seed_index"],
                    "sha256": row["adaptation_fixture_sha256"],
                }
                for row in preparation["cases"]
            ],
            "challenge_sha256_by_case": [
                {
                    "family": row["family"],
                    "seed_index": row["seed_index"],
                    "sha256": row["fixture_sha256"],
                }
                for row in challenge_manifest["cases"]
            ],
        },
        "v3_evaluator_report_sha256": _sha(inner_path),
        "candidate_control_alias": {"v3_mode": "wrong_module", "v6_semantics": "output_only_ridge"},
        "all_treatment_case_gates_pass": all_case_gates,
        "all_v6_case_gates_pass": all(row["case_status"] == "PASS" for row in cases),
        "all_aligned_equivalence_gates_pass": all(
            row["aligned_permutation_diagnostic"]["module_effect_max_abs_error_vs_treatment"] <= 1e-4
            and row["aligned_permutation_diagnostic"]["logit_effect_max_abs_error_vs_treatment"] <= 1e-4
            for row in cases
        ),
        "incremental_behavior_metric": {
            "definition": "ridge maximum normalized next-logit error minus treatment maximum normalized next-logit error, clustered by family and seed",
            "case_count": len(improvements),
            "mean_improvement": mean_improvement,
            "bootstrap_95_percent_lower": float(lower),
            "family_mean_improvement": family_means,
            "required_mean_improvement": 0.005,
            "required_lower_bound": 0.0,
            "pass": improvement_pass,
        },
        "cases": cases,
        "runtime_receipts": {
            "candidate_preparation": preparation.get("runtime"),
            "candidate_freeze": candidate_freeze.get("runtime"),
            "challenge_generation": challenge_manifest.get("runtime"),
            "evaluation": inner.get("runtime"),
        },
        "budget_and_resource_measurements": {
            "limits": {
                "max_peak_rss_bytes": 24 * 1024**3,
                "max_case_wall_seconds": 2 * 60 * 60,
                "max_total_wall_seconds": 18 * 60 * 60,
            },
            "measured_peak_rss_bytes": inner.get("stage_peak_rss_bytes"),
            "total_protocol_stage_wall_seconds": inner.get("total_protocol_stage_wall_seconds"),
            "total_process_cpu_seconds": inner.get("process_cpu_seconds_across_stages"),
            "resource_blockers": inner.get("resource_blockers", []),
            "per_case_wall_seconds": [
                {
                    "family": row["family"],
                    "seed_index": row["seed_index"],
                    "seconds": row["full_case_elapsed_wall_seconds"],
                }
                for row in cases
            ],
            "measured_case_cap_blockers": [
                {
                    "family": row["family"],
                    "seed_index": row["seed_index"],
                    "blocker": "total_case_wall_cap",
                }
                for row in inner["cases"]
                if "total_case_wall_cap" in row["blockers"]
            ],
        },
        "disjointness_receipt_sha256": _sha(custody / "v6-disjointness-receipt.json"),
        "data_and_artifact_custody": "weights, prompts, captures, candidates, predictions, outcomes, and row-level diagnostics remain in external owner custody",
        "known_limitations": [
            "one bounded MLP output projection per fresh checkpoint",
            "synthetic prompts and one checkpoint per architecture family",
            "same-owner local replication is not independent scientific authority",
            "monetary cost and energy are not measured",
        ],
        "claim_ceiling": "LOCAL_BEHAVIOR_AWARE_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ONLY",
        "nonclaims": ["all architectures", "general mechanistic understanding", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "configured external reviewer with independent authority; no review task submitted",
        "created_at_utc": _now(),
    }
    _write_json(custody / "replacement-loss-v6-report.json", report)
    return report


def run_primary(
    custody_root: str | Path = DEFAULT_CUSTODY,
    checkpoint_manifest: str | Path | None = None,
    prior_custody_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if checkpoint_manifest is None:
        raise ValueError("post-freeze checkpoint manifest is required")
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    if prior_custody_roots is None or not prior_custody_roots:
        raise ValueError("v6 requires prior custodied fixture roots for overlap checks")
    _validate_checkpoint_manifest(root, manifest_path, packet["packet_sha256"])
    preflight_payloads = _fixture_payloads_for_preflight()
    disjointness = _validate_all_disjoint(prior_custody_roots, preflight_payloads)
    _write_json(root / "v6-disjointness-preflight.json", disjointness)

    with _v6_profile():
        v3.archive_frozen_packet(root)
        v3.prepare_candidates(root, manifest_path)
        preparation_path = root / "candidate-preparation.json"
        preparation = _read_json(preparation_path)
        preparation["v6_packet_sha256"] = packet["packet_sha256"]
        preparation["v6_treatment_method"] = "fresh_affine_adamw_activation_plus_next_token_kl"
        _write_json(preparation_path, preparation)
        v3.freeze_candidates(root)
        v3.make_challenges(root, manifest_path)
        challenge_manifest = _add_v6_binding_to_challenge_manifest(root, packet["packet_sha256"])

        current = _fixture_payloads_for_preflight()
        for row in challenge_manifest["cases"]:
            prompts = _read_json(root / "challenges" / row["family"] / f"seed-{row['seed_index']}" / "prompts.json")
            current.append(prompts)
        disjointness = _validate_all_disjoint(prior_custody_roots, current)
        _write_json(root / "v6-disjointness-receipt.json", disjointness)
        v3.evaluate_packet(root, manifest_path)

    result = _build_v6_report(root, packet["packet_sha256"], disjointness)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": result["outcome_status"],
        "case_count": len(result["cases"]),
        "report_sha256": _sha(root / "replacement-loss-v6-report.json"),
    }


def replicate_packet(
    primary_custody_root: str | Path = DEFAULT_CUSTODY,
    replica_custody_root: str | Path = DEFAULT_REPLICA,
    checkpoint_manifest: str | Path | None = None,
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    packet = verify_packet(primary)
    if checkpoint_manifest is None:
        checkpoint_manifest = primary / "checkpoints.json"
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    _validate_checkpoint_manifest(primary, manifest_path, packet["packet_sha256"])
    primary_report_path = primary / "replacement-loss-v6-report.json"
    if not primary_report_path.is_file():
        raise FileNotFoundError("primary v6 aggregate report is required before replication")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("v6 replica custody must be new or empty")
    replica.mkdir(parents=True, exist_ok=True)
    inner_replica = replica / "v3-pinned-evaluator"
    with _v6_profile():
        v3_result = v3.replicate_packet(primary, inner_replica, manifest_path)
    inner_report_path = inner_replica / "replication-report.json"
    inner_report = _read_json(inner_report_path)
    shutil.copytree(primary / "v6-frozen-packet", replica / "v6-frozen-packet")
    shutil.copyfile(primary / "v6-frozen-packet-manifest.json", replica / "v6-frozen-packet-manifest.json")
    shutil.copyfile(manifest_path, replica / "checkpoints.json")
    shutil.copyfile(primary / "replacement-loss-v6-report.json", replica / "replacement-loss-v6-primary-report.json")
    shutil.copyfile(primary / "v6-disjointness-receipt.json", replica / "v6-disjointness-receipt.json")
    output_match = bool(inner_report.get("prediction_and_outcome_array_digests_match"))
    candidate_match = bool(inner_report.get("candidate_reconstruction_match"))
    runtime_match = bool(inner_report.get("dependencies_match")) and bool(inner_report.get("distinct_clean_runtime"))
    source_match = bool(inner_report.get("source_identity_match"))
    assurance = "REPRODUCED_LOCAL" if all((output_match, candidate_match, runtime_match, source_match)) else "INCONCLUSIVE"
    primary_report = _read_json(primary_report_path)
    replica_eval_path = inner_replica / "replacement-loss-v3-report.json"
    reconciliation = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_level": assurance,
        "reproducibility_status": assurance,
        "experiment_outcome_status": primary_report["outcome_status"],
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "packet_sha256": packet["packet_sha256"],
        "pinned_v3_packet_sha256": V3_PACKET_SHA256,
        "candidate_reconstruction_match": candidate_match,
        "prediction_and_outcome_array_digests_match": output_match,
        "dependencies_match": bool(inner_report.get("dependencies_match")),
        "distinct_clean_runtime": bool(inner_report.get("distinct_clean_runtime")),
        "source_identity_match": source_match,
        "primary_report_sha256": _sha(primary_report_path),
        "replica_evaluation_report_sha256": _sha(replica_eval_path),
        "pinned_evaluator_replication_report_sha256": _sha(inner_report_path),
        "replication_checks": inner_report,
        "next_blocked_authority_gate": "configured external reviewer with independent authority; no review task submitted",
        "created_at_utc": _now(),
    }
    _write_json(replica / "replacement-loss-v6-replication-report.json", reconciliation)
    primary_report["assurance_level"] = assurance
    primary_report["reproducibility_status"] = assurance
    primary_report["replication_report_sha256"] = _sha(replica / "replacement-loss-v6-replication-report.json")
    _write_json(primary_report_path, primary_report)
    return {
        "protocol_id": PROTOCOL_ID,
        "assurance_level": assurance,
        "experiment_outcome_status": primary_report["outcome_status"],
        "packet_sha256": packet["packet_sha256"],
        "replication_report_sha256": _sha(replica / "replacement-loss-v6-replication-report.json"),
        "replication_tool_status": v3_result.get("assurance_level"),
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
