"""Frozen-packet lifecycle for replacement-loss-v8.

State slice: theseus-lab-sandbox-v1. Model and prompt data stay in external
owner custody. Standard output contains aggregate receipts and digests only.
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

import replacement_loss_v3_runner as v3
import replacement_loss_v6_runner as v6_core
import replacement_loss_v3_fixtures as v3_fixtures
import replacement_loss_v6_fixtures as v6_fixtures
import replacement_loss_v7_fixtures as v7_fixtures
import replacement_loss_v8_fixtures as v8_fixtures
import replacement_loss_v8s_fixtures as v8s_fixtures
import replacement_loss_v8s2_fixtures as v8s2_fixtures
import replacement_loss_v8s3_fixtures as v8s3_fixtures
import replacement_loss_v8s4_fixtures as v8s4_fixtures
import replacement_loss_v7_runner as v7_core
import replacement_loss_v8s5_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
MAX_PARAMETER_COUNT = 9_000_000_000
PROTOCOL_ID = fixtures.PROTOCOL_ID
V3_PACKET_SHA256 = "d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83"
V6_PACKET_SHA256 = "9b7b669195bb4fb9153e872d42f3cb4fd235e9784fdc978ec936037eb176bc5a"
V7_PACKET_SHA256 = "39737cbef36a9eb84ed23a236de3f6c5de7c4a4168715f06d11203d6d5918918"
V7_PACKET_FILES = (*v7_core.V7_PACKET_FILES, "docs/replacement-loss-v7-freeze.json")
V8S5_PACKET_FILES = tuple(dict.fromkeys((
    "docs/replacement-loss-v8-scale-r5.md",
    "replacement_loss_v8s5_fixtures.py",
    "replacement_loss_v8s5_runner.py",
    "tests/test_replacement_loss_v8s5.py",
    "requirements.txt",
    "docs/replacement-loss-v8-scale-r4.md",
    "docs/replacement-loss-v8s4-freeze.json",
    "replacement_loss_v8s4_fixtures.py",
    "replacement_loss_v8s4_runner.py",
    "tests/test_replacement_loss_v8s4.py",
    "docs/replacement-loss-v8-scale-r3.md",
    "docs/replacement-loss-v8s3-freeze.json",
    "replacement_loss_v8s3_fixtures.py",
    "replacement_loss_v8s3_runner.py",
    "tests/test_replacement_loss_v8s3.py",
    "docs/replacement-loss-v8-scale-r2.md",
    "docs/replacement-loss-v8s2-freeze.json",
    "replacement_loss_v8s2_fixtures.py",
    "replacement_loss_v8s2_runner.py",
    "tests/test_replacement_loss_v8s2.py",
    "docs/replacement-loss-v8-scale.md",
    "docs/replacement-loss-v8s-freeze.json",
    "replacement_loss_v8s_fixtures.py",
    "replacement_loss_v8s_runner.py",
    "tests/test_replacement_loss_v8s.py",
    "docs/replacement-loss-v8.md",
    "docs/replacement-loss-v8-freeze.json",
    "replacement_loss_v8_fixtures.py",
    "replacement_loss_v8_runner.py",
    "tests/test_replacement_loss_v8.py",
    *V7_PACKET_FILES,
)))
PRIOR_CHECKPOINT_IDS = set(v7_core.PRIOR_CHECKPOINT_IDS) | {
    "babylm/opt-125m-strict-small-2023",
    "erwanf/gpt2-mini",
    "pgryko/babyllama-10m",
}
_ORIGINAL_FIT_CASE = v3._fit_case
BASE_ADAPTATION_PROMPTS = {"ordinary": 64, "compositional": 32, "adversarial": 32}
# Preregistered follow-up arms. Every arm keeps the frozen v3 gates unchanged; only
# the challenge intervention magnitude and the adaptation prompt count vary.
ARMS = {
    "N128": {"mode": "natural", "adaptation_prompts": 128},
    "N256": {"mode": "natural", "adaptation_prompts": 256},
    "N512": {"mode": "natural", "adaptation_prompts": 512},
    "T256": {"mode": "template", "adaptation_multiplier": 2},
}
MAX_PEAK_RSS_BYTES = 30 * 1024**3
V3_RSS_LITERAL_BLOCKER = "process_peak_rss_exceeded_24_GiB"


@contextmanager
def _arm_profile(arm: str) -> Iterator[None]:
    """Apply one arm's adaptation source and count, restoring afterwards. Challenge prompts are unchanged."""

    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    settings = ARMS[arm]
    saved = (dict(fixtures.ADAPTATION_PROMPTS), fixtures.ADAPTATION_MODE, fixtures.NATURAL_PROMPTS)
    try:
        fixtures.ADAPTATION_MODE = str(settings["mode"])
        if settings["mode"] == "natural":
            fixtures.NATURAL_PROMPTS = int(settings["adaptation_prompts"])
        else:
            fixtures.ADAPTATION_PROMPTS = {
                stratum: count * int(settings["adaptation_multiplier"])
                for stratum, count in BASE_ADAPTATION_PROMPTS.items()
            }
        yield
    finally:
        fixtures.ADAPTATION_PROMPTS, fixtures.ADAPTATION_MODE, fixtures.NATURAL_PROMPTS = saved


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
        raise ValueError("v8 custody must be outside the repository")
    return root


def _verify_pinned_v7() -> dict[str, Any]:
    """Verify pinned v7 and v8 source bytes from repository freezes (no custody)."""

    v7_freeze = _read_json(_root() / "docs" / "replacement-loss-v7-freeze.json")
    if v7_freeze.get("packet_sha256") != V7_PACKET_SHA256:
        raise ValueError("pinned v7 packet identity changed")
    if v7_freeze.get("v3_packet_sha256") != V3_PACKET_SHA256 or v7_freeze.get("v6_packet_sha256") != V6_PACKET_SHA256:
        raise ValueError("pinned v7 dependency identity changed")
    for freeze_name in ("replacement-loss-v7-freeze.json", "replacement-loss-v8-freeze.json", "replacement-loss-v8s-freeze.json", "replacement-loss-v8s2-freeze.json", "replacement-loss-v8s3-freeze.json", "replacement-loss-v8s4-freeze.json"):
        freeze = _read_json(_root() / "docs" / freeze_name)
        for row in freeze["packet_files"]:
            if _sha(_root() / str(row["path"])) != row["sha256"]:
                raise ValueError(f"pinned source changed: {row['path']}")
    return v7_freeze


def _generated_prior_payloads(_custody_roots: list[str | Path]) -> tuple[list[dict[str, Any]], list[str]]:
    """Regenerate v3, v6, v7 and v8 visible and adaptation fixtures for overlap checks."""

    payloads: dict[str, dict[str, Any]] = {}
    for module in (v3_fixtures, v6_fixtures, v7_fixtures, v8_fixtures, v8s_fixtures, v8s2_fixtures, v8s3_fixtures, v8s4_fixtures):
        batches = [(module.generate_visible_fixture(), "visible")]
        batches += [(module.generate_adaptation_fixture(index), f"adaptation_{index}") for index in range(3)]
        for records, kind in batches:
            payload = module.fixture_payload(records, fixture_kind=kind)
            payloads.setdefault(str(payload["example_set_sha256"]), payload)
    return list(payloads.values()), sorted(payloads)


def freeze_packet(
    custody_root: str | Path
) -> dict[str, Any]:
    """Freeze the complete v8 protocol before checkpoint selection."""

    root = _custody_root(custody_root)
    _verify_pinned_v7()
    local_freeze = _root() / "docs" / "replacement-loss-v8s5-freeze.json"
    archive_manifest = root / "v8s5-frozen-packet-manifest.json"
    archive = root / "v8s5-frozen-packet"
    if local_freeze.exists() or archive_manifest.exists() or archive.exists():
        raise FileExistsError("v8 packet freeze is immutable; continue with a new version")
    missing = [relative for relative in V8S5_PACKET_FILES if not (_root() / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v8 packet inputs are missing: {', '.join(missing)}")
    rows = []
    for relative in V8S5_PACKET_FILES:
        source = _root() / relative
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "sha256": _sha(source), "bytes": source.stat().st_size})
    v7_freeze = _read_json(_root() / "docs" / "replacement-loss-v7-freeze.json")
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V8S5_CHECKPOINT_SELECTION",
        "created_at_utc": _now(),
        "packet_files": rows,
        "v3_packet_sha256": V3_PACKET_SHA256,
        "v6_packet_sha256": V6_PACKET_SHA256,
        "v7_packet_sha256": V7_PACKET_SHA256,
        "checkpoint_identities_selected": False,
        "challenge_fixtures_generated": False,
        "treatment_method": "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        "threshold_source": "pinned_v3_causal_behavior_activation_and_control_gates",
        "pinned_v7_freeze_sha256": _sha(_root() / "docs" / "replacement-loss-v7-freeze.json"),
        "pinned_v7_created_at_utc": v7_freeze["created_at_utc"],
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


def verify_packet(
    custody_root: str | Path
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    _verify_pinned_v7()
    manifest = _read_json(root / "v8s5-frozen-packet-manifest.json")
    local_manifest = _read_json(_root() / "docs" / "replacement-loss-v8s5-freeze.json")
    if manifest != local_manifest:
        raise ValueError("local and custodied v8 freeze manifests differ")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v8 packet identity mismatch")
    expected = _canonical_digest({key: value for key, value in manifest.items() if key != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("v8 packet digest is invalid")
    for key, expected_sha in (
        ("v3_packet_sha256", V3_PACKET_SHA256),
        ("v6_packet_sha256", V6_PACKET_SHA256),
        ("v7_packet_sha256", V7_PACKET_SHA256),
    ):
        if manifest.get(key) != expected_sha:
            raise ValueError(f"v8 pinned dependency identity mismatch: {key}")
    for row in manifest["packet_files"]:
        relative = str(row["path"])
        if _sha(_root() / relative) != row["sha256"]:
            raise ValueError(f"frozen v8 source changed: {relative}")
        if _sha(root / "v8s5-frozen-packet" / relative) != row["sha256"]:
            raise ValueError(f"v8 archive changed: {relative}")
    return {
        "packet_sha256": manifest["packet_sha256"],
        "packet_file_count": len(manifest["packet_files"]),
        "verified": True,
    }


@contextmanager
def _activate_v8s5() -> Iterator[None]:
    names = (
        "PROTOCOL_ID", "V6_PACKET_FILES", "DEFAULT_CUSTODY", "DEFAULT_REPLICA",
        "PRIOR_CHECKPOINT_IDS", "fixtures", "verify_packet",
        "_validate_checkpoint_manifest", "write_checkpoint_manifest",
        "_add_v6_binding_to_challenge_manifest", "_build_v6_report", "_fit_case_v6",
        "_read_prior_fixture_payloads",
    )
    saved = {name: getattr(v6_core, name) for name in names}
    saved_cap = v3.MAX_PARAMETER_COUNT
    try:
        v6_core.PROTOCOL_ID = PROTOCOL_ID
        v6_core.V6_PACKET_FILES = V8S5_PACKET_FILES
        v6_core.DEFAULT_CUSTODY = Path("/tmp/theseus-v8s5-unused-default")
        v6_core.DEFAULT_REPLICA = Path("/tmp/theseus-v8s5-unused-replica")
        v6_core.PRIOR_CHECKPOINT_IDS = set(v6_core.PRIOR_CHECKPOINT_IDS) | PRIOR_CHECKPOINT_IDS
        v6_core.fixtures = fixtures
        v6_core.verify_packet = verify_packet
        v6_core._validate_checkpoint_manifest = _validate_checkpoint_manifest
        v6_core.write_checkpoint_manifest = write_checkpoint_manifest
        v6_core._add_v6_binding_to_challenge_manifest = _add_v8s5_binding_to_challenge_manifest
        v6_core._build_v6_report = _build_v8s5_report
        v6_core._fit_case_v6 = _fit_case_v8s5
        v6_core._read_prior_fixture_payloads = _generated_prior_payloads
        v3.MAX_PARAMETER_COUNT = MAX_PARAMETER_COUNT
        yield
    finally:
        for name, value in saved.items():
            setattr(v6_core, name, value)
        v3.MAX_PARAMETER_COUNT = saved_cap


def write_checkpoint_manifest(
    custody_root: str | Path,
    checkpoint_specs: list[dict[str, Any]]
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if len(checkpoint_specs) != len(v3.FAMILIES) or {row.get("family") for row in checkpoint_specs} != set(v3.FAMILIES):
        raise ValueError("v8 selection must contain exactly one checkpoint per frozen family")
    destination = root / "checkpoints.json"
    if destination.exists():
        raise FileExistsError("v8 checkpoint selection is immutable once recorded")
    freeze = _read_json(_root() / "docs" / "replacement-loss-v8s5-freeze.json")
    frozen_at = datetime.fromisoformat(str(freeze["created_at_utc"]))
    selected_at = datetime.now(timezone.utc)
    if selected_at <= frozen_at:
        raise ValueError("v8 checkpoint identities must be selected after packet freeze")
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
        if len(revision) != 40 or any(ch not in "0123456789abcdefABCDEF" for ch in revision):
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
    v3_freeze = _read_json(_root() / v3.PACKET_FREEZE)
    payload = {
        "protocol_id": v3.PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": v3_freeze["packet_sha256"],
        "v8s5_protocol_id": PROTOCOL_ID,
        "v8s5_packet_sha256": packet["packet_sha256"],
        "selected_at_utc": selected_at.isoformat(),
        "checkpoints": sorted(entries, key=lambda item: v3.FAMILIES.index(item["family"])),
    }
    temporary = root / ".v8s5-checkpoint-selection-preflight.json"
    try:
        with _activate_v8s5():
            with v6_core._v6_profile():
                temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                v3._load_checkpoints(temporary, root)
    finally:
        temporary.unlink(missing_ok=True)
    _write_json(destination, payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "v8s5_packet_sha256": packet["packet_sha256"],
        "checkpoint_manifest_sha256": _sha(destination),
        "checkpoint_count": len(entries),
    }


def _validate_checkpoint_manifest(
    custody: Path,
    manifest_path: Path,
    packet_sha: str
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if manifest.get("v8s5_protocol_id") != PROTOCOL_ID or manifest.get("v8s5_packet_sha256") != packet_sha:
        raise ValueError("checkpoint manifest does not bind the frozen v8 packet")
    if manifest_path.resolve().parent != custody.resolve():
        raise ValueError("checkpoint manifest must be directly under external custody")
    freeze = _read_json(_root() / "docs" / "replacement-loss-v8s5-freeze.json")
    selected_at = datetime.fromisoformat(str(manifest.get("selected_at_utc", "")))
    frozen_at = datetime.fromisoformat(str(freeze["created_at_utc"]))
    if selected_at.tzinfo is None or selected_at <= frozen_at:
        raise ValueError("v8 checkpoint selection must follow the timezone-aware packet freeze")
    for item in manifest.get("checkpoints", []):
        if str(item.get("model_id", "")).lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError("a checkpoint selected in v3-v7 cannot be reused for v8")
        if item.get("not_used_for_protocol_design") is not True or item.get("base_model") is not True:
            raise ValueError("v8 requires fresh public base checkpoints selected after freeze")
    with v6_core._v6_profile():
        v3._load_checkpoints(manifest_path, custody)
    return manifest


def _fit_case_v8s5(
    model: Any,
    tokenizer: Any,
    checkpoint: Any,
    layer: int,
    seed_index: int,
    fixture_records: list[dict[str, Any]],
    custody: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    case = _ORIGINAL_FIT_CASE(model, tokenizer, checkpoint, layer, seed_index, fixture_records, custody)
    case["v8s5_treatment_method"] = "pinned_v3_closed_form_output_ridge_from_adaptation_pairs"
    case["v8s5_adaptation_only"] = True
    case["v8s5_no_target_parameter_copy"] = True
    case["fit_elapsed_wall_seconds"] = float(case["fit_elapsed_wall_seconds"]) + (time.monotonic() - started)
    return case


def _add_v8s5_binding_to_challenge_manifest(custody: Path, packet_sha: str) -> dict[str, Any]:
    path = custody / "challenge-manifest.json"
    manifest = _read_json(path)
    manifest["v8s5_protocol_id"] = PROTOCOL_ID
    manifest["v8s5_packet_sha256"] = packet_sha
    manifest.pop("challenge_manifest_sha256", None)
    manifest["challenge_manifest_sha256"] = _canonical_digest(manifest)
    _write_json(path, manifest)
    return manifest


def _reclassify_outcome(inner: dict[str, Any]) -> dict[str, Any]:
    """Recompute the v3 outcome with this packet's 30 GiB memory limit.

    The frozen v3 evaluator hard-codes a 24 GiB check. v3 stays immutable, so this
    packet re-derives the outcome from the evaluator's own recorded case results,
    cluster-bootstrap values and equivalence diagnostics using the identical gate
    conditions, and replaces only the memory limit (24 -> 30 GiB).
    """

    peak = int(inner.get("stage_peak_rss_bytes") or 0)
    blockers = [b for b in inner.get("resource_blockers", []) if b != V3_RSS_LITERAL_BLOCKER]
    if peak > MAX_PEAK_RSS_BYTES:
        blockers.append("process_peak_rss_exceeded_30_GiB")
    cases = inner["cases"]
    bootstrap = inner["cluster_bootstrap"]
    gate_pass = (
        len(cases) == 9
        and all(row["case_status"] == "PASS" for row in cases)
        and bootstrap["treatment_direction_accuracy_95_percent_lower"] > 0.70
        and bootstrap["treatment_direction_accuracy_mean"] >= 0.80
        and all(value > 0 for value in bootstrap["paired_treatment_minus_control_95_percent_lower"].values())
        and all(
            row["mapping_aligned_permutation_diagnostic"]["module_effect_max_abs_error_vs_treatment"] <= v3.EQUIVALENCE_MAX_ABS_ERROR
            and row["mapping_aligned_permutation_diagnostic"]["logit_effect_max_abs_error_vs_treatment"] <= v3.EQUIVALENCE_MAX_ABS_ERROR
            for row in cases
        )
    )
    passed = gate_pass and not blockers
    if passed:
        status = "PASS"
    elif blockers or any(row["case_status"] == "INCONCLUSIVE" for row in cases):
        status = "INCOMPLETE"
    else:
        status = "FAIL"
    return {
        "outcome_status": status,
        "assurance_level": "INTERNAL_PASS" if passed else "INCONCLUSIVE",
        "resource_blockers": blockers,
        "gates_pass_ignoring_resources": gate_pass,
    }


def _build_v8s5_report(custody: Path, packet_sha: str, disjointness: dict[str, Any]) -> dict[str, Any]:
    inner = _read_json(custody / "replacement-loss-v3-report.json")
    reclassified = _reclassify_outcome(inner)
    preparation = _read_json(custody / "candidate-preparation.json")
    candidate_freeze = _read_json(custody / "candidate-freeze.json")
    challenge_manifest = _read_json(custody / "challenge-manifest.json")
    cases = []
    for row in inner["cases"]:
        prep = next(item for item in preparation["cases"] if item["family"] == row["family"] and item["seed_index"] == row["seed_index"])
        cases.append({
            "family": row["family"],
            "model_id": row["model_id"],
            "revision": row["revision"],
            "seed_index": row["seed_index"],
            "seed": row["seed"],
            "selected_layer": row["selected_layer"],
            "case_status": row["case_status"],
            "blockers": row["blockers"],
            "treatment_effects": row["primary_effects"]["treatment"],
            "treatment_behavior": row["behavior"]["treatment"],
            "treatment_activation_rescue": row["activation_rescue"],
            "control_direction_accuracy": row["case_scores_for_cluster_bootstrap"],
            "treatment_fit": prep["treatment_fit"],
            "candidate_state_digests": {item["name"]: item["state_sha256"] for item in prep["modules"]},
            "prediction_array_sha256": row["prediction_array_sha256"],
            "outcome_array_sha256": row["outcome_array_sha256"],
            "full_case_elapsed_wall_seconds": row["full_case_elapsed_wall_seconds"],
        })
    if len(cases) != 9:
        raise ValueError("v8 result must contain all nine frozen cases")
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet_sha,
        "pinned_v3_packet_sha256": V3_PACKET_SHA256,
        "pinned_v6_packet_sha256": V6_PACKET_SHA256,
        "pinned_v7_packet_sha256": V7_PACKET_SHA256,
        "source_identity": inner["source"],
        "checkpoint_manifest_sha256": inner["checkpoint_manifest_sha256"],
        "candidate_files_sha256": inner["candidate_files_sha256"],
        "candidate_preparation_sha256": candidate_freeze["candidate_preparation_sha256"],
        "candidate_freeze_sha256": _sha(custody / "candidate-freeze.json"),
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "disjointness_receipt_sha256": _sha(custody / "v8s5-disjointness-receipt.json"),
        "outcome_status": reclassified["outcome_status"],
        "assurance_level": reclassified["assurance_level"],
        "gates_pass_ignoring_resources": reclassified["gates_pass_ignoring_resources"],
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "reproducibility_status": "PENDING_SEPARATE_REPLICATOR",
        "treatment_method": "pinned_v3_closed_form_output_ridge_from_adaptation_pairs",
        "treatment_fit_uses_target_weights": False,
        "v3_evaluator_outcome_status": inner["outcome_status"],
        "all_treatment_case_gates_pass": all(row["case_status"] == "PASS" for row in inner["cases"]),
        "cases": cases,
        "runtime": inner["runtime"],
        "resource_blockers": reclassified["resource_blockers"],
        "monetary_cost": inner.get("monetary_cost", "NOT_MEASURED"),
        "energy": inner.get("energy", "NOT_MEASURED"),
        "budget_and_resource_measurements": {
            "limits": {"max_case_wall_seconds": 7200, "max_peak_rss_bytes": MAX_PEAK_RSS_BYTES, "max_total_wall_seconds": 64800},
            "stage_peak_rss_bytes": inner.get("stage_peak_rss_bytes"),
            "total_process_cpu_seconds": inner.get("process_cpu_seconds_across_stages"),
            "total_protocol_stage_wall_seconds": inner.get("total_protocol_stage_wall_seconds"),
            "case_wall_cap_blockers": [
                {"family": row["family"], "seed_index": row["seed_index"], "blockers": row["blockers"]}
                for row in inner["cases"] if "total_case_wall_cap" in row["blockers"]
            ],
        },
        "data_and_artifact_custody": "weights, prompts, captures, candidates, predictions, outcomes, and row-level diagnostics remain in external owner custody",
        "known_limitations": [
            "one MLP output projection per fresh checkpoint",
            "synthetic prompts and one checkpoint per architecture family",
            "same-owner local replication is not independent scientific authority",
            "monetary cost and energy are not measured",
        ],
        "claim_ceiling": "LOCAL_INDEPENDENT_AFFINE_PROJECTION_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION",
        "nonclaims": ["all architectures", "whole-model mechanistic identity", "independent science", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "configured external reviewer with independent authority; no review task submitted",
        "created_at_utc": _now(),
    }
    _write_json(custody / "replacement-loss-v8s5-report.json", report)
    return report


def _fixture_payloads_for_preflight() -> list[dict[str, Any]]:
    visible = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible_selection")
    adaptation = [
        fixtures.fixture_payload(fixtures.generate_adaptation_fixture(index), fixture_kind=f"adaptation_{fixtures.SEEDS[index]}")
        for index in range(len(fixtures.SEEDS))
    ]
    fixtures.validate_partition_disjointness(visible, *adaptation)
    return [visible, *adaptation]


def run_primary(
    custody_root: str | Path,
    checkpoint_manifest: str | Path,
    arm: str,
    prior_custody_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    with _arm_profile(arm):
        result = _run_primary_inner(custody_root, checkpoint_manifest, prior_custody_roots)
    report_path = _custody_root(custody_root) / "replacement-loss-v8s5-report.json"
    report = _read_json(report_path)
    report["arm"] = arm
    report["arm_settings"] = ARMS[arm]
    _write_json(report_path, report)
    result["arm"] = arm
    result["report_sha256"] = _sha(report_path)
    return result


def attach_custody(custody_root: str | Path) -> dict[str, Any]:
    """Create an arm's custody archive from the frozen repository bytes (verified against the freeze)."""

    root = _custody_root(custody_root)
    _verify_pinned_v7()
    manifest = _read_json(_root() / "docs" / "replacement-loss-v8s5-freeze.json")
    expected = _canonical_digest({key: value for key, value in manifest.items() if key != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("v8s5 freeze digest is invalid")
    archive = root / "v8s5-frozen-packet"
    if archive.exists():
        raise FileExistsError("custody archive already exists")
    for row in manifest["packet_files"]:
        relative = str(row["path"])
        source = _root() / relative
        if _sha(source) != row["sha256"]:
            raise ValueError(f"frozen source changed: {relative}")
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    _write_json(root / "v8s5-frozen-packet-manifest.json", manifest)
    return {"attached": True, "packet_sha256": manifest["packet_sha256"], "file_count": len(manifest["packet_files"])}


def _run_primary_inner(
    custody_root: str | Path,
    checkpoint_manifest: str | Path,
    prior_custody_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    root = _custody_root(custody_root)
    packet = verify_packet(root)
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    prior_custody_roots = prior_custody_roots or []
    with _activate_v8s5():
        _validate_checkpoint_manifest(root, manifest_path, packet["packet_sha256"])
        preflight = _fixture_payloads_for_preflight()
        disjointness = v6_core._validate_all_disjoint(prior_custody_roots, preflight)
        _write_json(root / "v8s5-disjointness-preflight.json", disjointness)
        with v6_core._v6_profile():
            v3.archive_frozen_packet(root)
            v3.prepare_candidates(root, manifest_path)
            preparation_path = root / "candidate-preparation.json"
            preparation = _read_json(preparation_path)
            preparation["v8s5_packet_sha256"] = packet["packet_sha256"]
            preparation["v8s5_treatment_method"] = "pinned_v3_closed_form_output_ridge_from_adaptation_pairs"
            _write_json(preparation_path, preparation)
            v3.freeze_candidates(root)
            v3.make_challenges(root, manifest_path)
            challenge_manifest = _add_v8s5_binding_to_challenge_manifest(root, packet["packet_sha256"])
            current = _fixture_payloads_for_preflight()
            for row in challenge_manifest["cases"]:
                prompt_payload = _read_json(root / "challenges" / row["family"] / f"seed-{row['seed_index']}" / "prompts.json")
                current.append(prompt_payload)
            disjointness = v6_core._validate_all_disjoint(prior_custody_roots, current)
            _write_json(root / "v8s5-disjointness-receipt.json", disjointness)
            v3.evaluate_packet(root, manifest_path)
    report = _build_v8s5_report(root, packet["packet_sha256"], disjointness)
    report_path = root / "replacement-loss-v8s5-report.json"
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "outcome_status": report["outcome_status"],
        "case_count": len(report["cases"]),
        "report_sha256": _sha(report_path),
    }


def replicate_packet(
    primary_custody_root: str | Path,
    replica_custody_root: str | Path,
    checkpoint_manifest: str | Path
) -> dict[str, Any]:
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    packet = verify_packet(primary)
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    _validate_checkpoint_manifest(primary, manifest_path, packet["packet_sha256"])
    primary_report_path = primary / "replacement-loss-v8s5-report.json"
    if not primary_report_path.is_file():
        raise FileNotFoundError("v8 primary aggregate report is required before replication")
    with _activate_v8s5():
        with v6_core._v6_profile():
            result = v3.replicate_packet(primary, replica, manifest_path)
    inner_replica_report_path = replica / "replication-report.json"
    inner_report = _read_json(inner_replica_report_path)
    primary_report = _read_json(primary_report_path)
    assurance = result.get("assurance_level", "INCONCLUSIVE")
    checks_pass = all(bool(inner_report.get(key)) for key in (
        "candidate_reconstruction_match",
        "prediction_and_outcome_array_digests_match",
        "dependencies_match",
        "distinct_clean_runtime",
        "source_identity_match",
    ))
    if not checks_pass:
        assurance = "INCONCLUSIVE"
    final = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": assurance,
        "reproducibility_status": assurance,
        "experiment_outcome_status": primary_report["outcome_status"],
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "v3_replication_checks": inner_report,
        "candidate_reconstruction_match": bool(inner_report.get("candidate_reconstruction_match")),
        "prediction_and_outcome_array_digests_match": bool(inner_report.get("prediction_and_outcome_array_digests_match")),
        "dependencies_match": bool(inner_report.get("dependencies_match")),
        "distinct_clean_runtime": bool(inner_report.get("distinct_clean_runtime")),
        "source_identity_match": bool(inner_report.get("source_identity_match")),
        "primary_v8s5_report_sha256_before_reconciliation": _sha(primary_report_path),
        "pinned_v3_replication_report_sha256": _sha(inner_replica_report_path),
        "next_blocked_authority_gate": "configured external reviewer with independent authority; no review task submitted",
        "created_at_utc": _now(),
    }
    replica_report_path = replica / "replacement-loss-v8s5-replication-report.json"
    _write_json(replica_report_path, final)
    final["replication_report_sha256"] = _sha(replica_report_path)
    primary_report["assurance_level"] = assurance
    primary_report["reproducibility_status"] = assurance
    primary_report["replication_report_sha256"] = _sha(replica_report_path)
    _write_json(primary_report_path, primary_report)
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": assurance,
        "experiment_outcome_status": primary_report["outcome_status"],
        "replication_report_sha256": _sha(replica_report_path),
    }


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--custody-root", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--custody-root", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--custody-root", required=True)
    run.add_argument("--checkpoint-manifest", required=True)
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--prior-custody-root", action="append", default=[])
    attach = subparsers.add_parser("attach")
    attach.add_argument("--custody-root", required=True)
    replicate = subparsers.add_parser("replicate")
    replicate.add_argument("--primary-custody-root", required=True)
    replicate.add_argument("--replica-custody-root", required=True)
    replicate.add_argument("--checkpoint-manifest", required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root)
    elif args.command == "attach":
        result = attach_custody(args.custody_root)
    elif args.command == "run":
        result = run_primary(args.custody_root, args.checkpoint_manifest, args.arm, args.prior_custody_root)
    else:
        result = replicate_packet(args.primary_custody_root, args.replica_custody_root, args.checkpoint_manifest)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
