"""Frozen paired-endpoint replacement experiment and local replay.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v21-paired-endpoint-supervision-v1, and
global-experiment-assurance-lifecycle-v1.

Model prompts and row-level artifacts stay in external owner custody.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import replacement_loss_v20_runner as v20
import replacement_loss_v20b_runner as v20b
import replacement_loss_v21_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
DEV_PROTOCOL_ID = fixtures.DEV_PROTOCOL_ID
CONFIRM_PROTOCOL_ID = fixtures.CONFIRM_PROTOCOL_ID
CLAIM_CEILING = "LOCAL_PAIRED_ENDPOINT_SUPERVISION_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT"
REPO_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = v20.RESEARCH_ROOT
DEV_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21-development-primary"
DEV_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21-development-replica"
CONFIRM_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21-confirmation-primary"
CONFIRM_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21-confirmation-replica"
PROTOCOL_FILE = "docs/replacement-loss-v21-paired-endpoint-supervision.md"
FIXTURE_FILE = "replacement_loss_v21_fixtures.py"
RUNNER_FILE = "replacement_loss_v21_runner.py"
LOCK_FILE = "requirements-v21.lock.txt"
DEV_FREEZE_FILE = "docs/replacement-loss-v21-development-freeze.json"
CONFIRM_FREEZE_FILE = "docs/replacement-loss-v21-confirmation-freeze.json"
CONFIRM_CONFIG_FILE = "docs/replacement-loss-v21-confirmation-config.json"
DEV_RESULT_JSON = "docs/replacement-loss-v21-development-results.json"
DEV_RESULT_MD = "docs/replacement-loss-v21-development-results.md"
DEV_REPLICATION_JSON = "docs/replacement-loss-v21-development-replication.json"
CONFIRM_RESULT_JSON = "docs/replacement-loss-v21-confirmation-results.json"
CONFIRM_RESULT_MD = "docs/replacement-loss-v21-confirmation-results.md"
CONFIRM_REPLICATION_JSON = "docs/replacement-loss-v21-confirmation-replication.json"
IDENTIFIABILITY_NOTE = "docs/causal-effect-identifiability-boundary-v1.md"
V20B_RESULT_JSON = "docs/replacement-loss-v20b-development-results.json"
V20B_RESULT_MD = "docs/replacement-loss-v20b-development-results.md"
V20B_REPLICATION_JSON = "docs/replacement-loss-v20b-development-replication.json"

_V20B_PACKET_FILES = tuple(dict.fromkeys((
    "AGENTS.md",
    IDENTIFIABILITY_NOTE,
    PROTOCOL_FILE,
    FIXTURE_FILE,
    RUNNER_FILE,
    LOCK_FILE,
    "docs/replacement-loss-v20b-development-freeze.json",
    V20B_RESULT_JSON,
    V20B_RESULT_MD,
    V20B_REPLICATION_JSON,
    *v20b.DEV_PACKET_FILES,
)))
DEV_PACKET_FILES = _V20B_PACKET_FILES
CONFIRM_PACKET_FILES = tuple(dict.fromkeys((
    *DEV_PACKET_FILES,
    DEV_FREEZE_FILE,
    CONFIRM_CONFIG_FILE,
    DEV_RESULT_JSON,
    DEV_RESULT_MD,
    DEV_REPLICATION_JSON,
)))

_V20_WRITE_JSON = v20._write_json
_V20_ORIGINAL_TRAINING_SET = v20._training_set
_V20_ORIGINAL_TRAIN_PAIR = v20._train_pair
_V20B_LOAD_DEPENDENCIES = v20b._load_dependencies
_RUNTIME_PATCHED = False
_V20B_REFERENCE_WRAPPED = False


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _rows(files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in files:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V21 frozen source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _parent_binding() -> dict[str, Any]:
    parent = v20b.verify_development_packet()
    result_path = REPO_ROOT / V20B_RESULT_JSON
    replay_path = REPO_ROOT / V20B_REPLICATION_JSON
    result, replay = _read_json(result_path), _read_json(replay_path)
    if (
        result.get("qualification_status") != "NO_QUALIFIED_DEVELOPMENT_CANDIDATE"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_results_match") is not True
        or replay.get("qualification_status") != "NO_QUALIFIED_DEVELOPMENT_CANDIDATE"
        or result.get("scientific_results_sha256") != replay.get("scientific_results_sha256")
    ):
        raise ValueError("V21 requires the exact reproduced V20B no-candidate disposition")
    return {
        "protocol_id": parent["protocol_id"],
        "packet_sha256": parent["packet_sha256"],
        "freeze_sha256": parent["freeze_sha256"],
        "result_file_sha256": _sha_file(result_path),
        "result_sha256": result["scientific_results_sha256"],
        "replication_file_sha256": _sha_file(replay_path),
        "replication_report_sha256": replay["report_sha256"],
        "qualification_status": result["qualification_status"],
        "reproducibility_status": result["reproducibility_status"],
        "v20b_data_reused": False,
    }


def freeze_development() -> dict[str, Any]:
    path = REPO_ROOT / DEV_FREEZE_FILE
    if path.exists():
        return verify_development_packet()
    parent = _parent_binding()
    packet_rows = _rows(DEV_PACKET_FILES)
    packet = {
        "protocol_id": DEV_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": packet_rows,
        "packet_sha256": _digest(packet_rows),
        "parent_negative_result": parent,
        "target_identity": {
            "model_id": "grushaprasad/opt125m-rootp-surprisal-curriculum",
            "revision": "8af2a2926bc772599ccb3b4e7ad7b40b35259a63",
            "tree_sha256": "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466",
            "weight_sha256": "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132",
            "target_known_before_protocol": True,
            "fresh_checkpoint_claim": False,
            "target_layer": 8,
            "wrong_layer": 9,
        },
        "method": {
            "kind": "paired_endpoint_value_supervision",
            "uses_finite_effect_targets_in_gradient": False,
            "fits_both_teacher_endpoints": True,
            "seeds": list(fixtures.DEVELOPMENT_SEEDS),
            "confirmation_seeds": list(fixtures.CONFIRMATION_SEEDS),
            "strata": list(fixtures.STRATA),
            "adaptation_prompts_per_stratum": fixtures.ADAPTATION_PER_STRATUM,
            "fit_prompts_per_stratum": fixtures.FIT_PER_STRATUM,
            "validation_prompts_per_stratum": fixtures.VALIDATION_PER_STRATUM,
            "optimizer_updates_pointwise_base": 1000,
            "optimizer_updates_endpoint_residual": 1000,
            "low_rank": 16,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 128,
            "gradient_clip_norm": 1.0,
        },
        "qualification": {
            "cells": 9,
            "effect_ratio_max": 0.90,
            "value_ratio_max": 1.25,
            "continuation_ce_increase_max": 0.05,
            "normalized_next_logit_error_max": 0.05,
            "all_cells_must_pass_primary_and_replica": True,
            "challenge_entropy_before_qualification": False,
        },
        "custody_and_limits": {
            "prompt_and_model_artifacts": "external owner custody",
            "runtime": "pinned Python 3.12.13; deterministic CPU; one Torch thread",
            "wall_cap_seconds": 14400,
            "rss_cap_bytes": 17179869184,
            "paid_or_remote_execution": False,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    payload = json.dumps(packet, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError("V21 development freeze exists with different bytes")
    if not path.exists():
        path.write_bytes(payload)
    return packet


def verify_development_packet() -> dict[str, Any]:
    frozen = _read_json(REPO_ROOT / DEV_FREEZE_FILE)
    current, parent = _rows(DEV_PACKET_FILES), _parent_binding()
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != DEV_PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current)
        or frozen.get("parent_negative_result") != parent
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V21 development packet identity or frozen bytes differ")
    return frozen


def freeze_confirmation() -> dict[str, Any]:
    development = verify_development_packet()
    result, replay = _read_json(REPO_ROOT / DEV_RESULT_JSON), _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    if (
        result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_results_match") is not True
    ):
        raise ValueError("V21 confirmation requires qualified development and exact local replay")
    config = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "development_packet_sha256": development["packet_sha256"],
        "development_freeze_sha256": development["freeze_sha256"],
        "development_result_sha256": result["scientific_results_sha256"],
        "development_replay_sha256": replay["report_sha256"],
        "seeds": list(fixtures.CONFIRMATION_SEEDS),
        "challenge_unlock": "exact primary and replica prechallenge match and every confirmation cell qualifies",
    }
    config["config_sha256"] = _digest(config)
    config_path = REPO_ROOT / CONFIRM_CONFIG_FILE
    payload = json.dumps(config, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if config_path.exists() and config_path.read_bytes() != payload:
        raise FileExistsError("V21 confirmation config exists with different bytes")
    if not config_path.exists():
        config_path.write_bytes(payload)
    packet_rows = _rows(CONFIRM_PACKET_FILES)
    packet = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": packet_rows,
        "packet_sha256": _digest(packet_rows),
        "development_binding": {
            "packet_sha256": development["packet_sha256"],
            "result_sha256": result["scientific_results_sha256"],
            "replay_sha256": replay["report_sha256"],
            "qualification_status": "QUALIFIED",
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    freeze_path = REPO_ROOT / CONFIRM_FREEZE_FILE
    payload = json.dumps(packet, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if freeze_path.exists() and freeze_path.read_bytes() != payload:
        return verify_confirmation_packet()
    if not freeze_path.exists():
        freeze_path.write_bytes(payload)
    return packet


def verify_confirmation_packet() -> dict[str, Any]:
    development = verify_development_packet()
    config = _read_json(REPO_ROOT / CONFIRM_CONFIG_FILE)
    result, replay = _read_json(REPO_ROOT / DEV_RESULT_JSON), _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    config_material = {key: value for key, value in config.items() if key != "config_sha256"}
    if (
        config.get("config_sha256") != _digest(config_material)
        or result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
    ):
        raise ValueError("V21 confirmation entry gate is invalid")
    frozen = _read_json(REPO_ROOT / CONFIRM_FREEZE_FILE)
    current = _rows(CONFIRM_PACKET_FILES)
    expected_binding = {
        "packet_sha256": development["packet_sha256"],
        "result_sha256": result["scientific_results_sha256"],
        "replay_sha256": replay["report_sha256"],
        "qualification_status": "QUALIFIED",
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != CONFIRM_PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current)
        or frozen.get("development_binding") != expected_binding
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V21 confirmation packet identity or frozen bytes differ")
    return frozen


def _packet(stage: str):
    if stage == "development":
        return DEV_PROTOCOL_ID, DEV_FREEZE_FILE, DEV_PACKET_FILES, verify_development_packet()
    if stage == "confirmation":
        return CONFIRM_PROTOCOL_ID, CONFIRM_FREEZE_FILE, CONFIRM_PACKET_FILES, verify_confirmation_packet()
    raise ValueError("stage must be development or confirmation")


def _training_set_with_endpoint(layer: Any, captures: Any, records: Any, prompt_set: set[int], *, seed: int, label: str):
    data = _V20_ORIGINAL_TRAINING_SET(layer, captures, records, prompt_set, seed=seed, label=label)
    import numpy as np

    shifted = np.asarray(data["x"] + data["delta"], dtype=np.float32)
    zeros = np.zeros_like(data["delta"], dtype=np.float32)
    endpoint, _ = v20.v17._teacher_effects(layer, shifted, zeros)
    endpoint = endpoint.astype(np.float32, copy=False)
    reconstructed = data["y"] + data["effect"]
    if endpoint.shape != reconstructed.shape or not np.allclose(endpoint, reconstructed, rtol=1e-5, atol=1e-6):
        raise ValueError("V21 direct counterfactual endpoint differs from baseline plus teacher effect")
    data["counterfactual_y"] = endpoint
    return data


def _train_pair_with_endpoint(*args: Any, **kwargs: Any):
    treatment, pointwise, random_control, receipt = _V20_ORIGINAL_TRAIN_PAIR(*args, **kwargs)
    if "causal_residual" in receipt:
        receipt["paired_endpoint_residual"] = receipt.pop("causal_residual")
    return treatment, pointwise, random_control, receipt


def _train_endpoint_residual(base: Any, data: dict[str, Any], *, seed: int, root: Path, started: float):
    _load_v21_dependencies()
    import numpy as np
    import torch
    import torch.nn.functional as F

    x, delta, y, y_cf = (
        torch.from_numpy(data[key]) for key in ("x", "delta", "y", "counterfactual_y")
    )
    residual = v20._factorized_candidate(base, seed)
    parameters = [residual.a1, residual.b1, residual.a2, residual.b2]
    optimizer = torch.optim.AdamW(parameters, lr=v20.LEARNING_RATE, weight_decay=v20.WEIGHT_DECAY)
    endpoint_norm = max(float((y.square().mean() + y_cf.square().mean()).item() / 2.0), 1e-12)
    effect_norm = max(float(torch.from_numpy(data["effect"]).square().mean().item()), 1e-12)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x21E0D1)
    draws = np.empty((v20.TRAIN_UPDATES, v20.TRAIN_BATCH_SIZE), dtype=np.int32)
    residual.train()
    for update in range(v20.TRAIN_UPDATES):
        indices = torch.randint(0, x.shape[0], (v20.TRAIN_BATCH_SIZE,), generator=generator)
        draws[update] = indices.numpy()
        xb, db = x.index_select(0, indices), delta.index_select(0, indices)
        yb, ycfb = y.index_select(0, indices), y_cf.index_select(0, indices)
        prediction = residual(xb)
        counterfactual_prediction = residual(xb + db)
        loss = (F.mse_loss(prediction, yb) + F.mse_loss(counterfactual_prediction, ycfb)) / (2.0 * endpoint_norm)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V21 paired endpoint objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, v20.GRADIENT_CLIP)
        optimizer.step()
        if (update + 1) % 100 == 0:
            v20._check_caps(root, started)
    residual.eval()
    dense = v20.v16.WholeMLPCandidate()
    dense.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        dense.fc1.weight.copy_(base.fc1.weight + residual.b1 @ residual.a1)
        dense.fc2.weight.copy_(base.fc2.weight + residual.b2 @ residual.a2)
        probe = x[: min(32, len(x))]
        merge_error = float((residual(probe) - dense(probe)).abs().max().item())
        fit_y_error = float(F.mse_loss(dense(x), y).item() / endpoint_norm)
        fit_cf_error = float(F.mse_loss(dense(x + delta), y_cf).item() / endpoint_norm)
        fit_effect_error = float(F.mse_loss(dense(x + delta) - dense(x), torch.from_numpy(data["effect"])).item() / effect_norm)
    if merge_error > 1e-5:
        raise ValueError("V21 merged dense candidate differs from its factorized module")
    dense.eval()
    endpoint_bytes = np.concatenate((data["y"], data["counterfactual_y"]), axis=0).astype("<f4", copy=False).tobytes()
    return dense, {
        "base_state_sha256": v20.v16._module_state_digest(base),
        "residual_state_sha256": v20.v16._module_state_digest(residual),
        "merged_state_sha256": v20.v16._module_state_digest(dense),
        "factor_rank": v20.RANK,
        "updates": v20.TRAIN_UPDATES,
        "batch_size": v20.TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(draws.astype("<i4", copy=False).tobytes()).hexdigest(),
        "objective": "equal-weight normalized squared error at teacher factual and counterfactual endpoints; no finite-effect term or base-drift anchor",
        "endpoint_normalizer": endpoint_norm,
        "fit_factual_endpoint_mse_normalized": fit_y_error,
        "fit_counterfactual_endpoint_mse_normalized": fit_cf_error,
        "fit_effect_mse_ratio_zero": fit_effect_error,
        "endpoint_target_sha256": hashlib.sha256(endpoint_bytes).hexdigest(),
        "learning_rate": v20.LEARNING_RATE,
        "weight_decay": v20.WEIGHT_DECAY,
        "gradient_clip_norm": v20.GRADIENT_CLIP,
        "factorized_to_merged_probe_max_abs_error": merge_error,
        "source_parameters_copied": False,
        "pointwise_base_parameters_frozen": True,
        "fit_inputs_sha256": hashlib.sha256(data["x"].astype("<f4", copy=False).tobytes()).hexdigest(),
        "fit_directions_sha256": hashlib.sha256(data["delta"].astype("<f4", copy=False).tobytes()).hexdigest(),
    }


def _reference_payloads_with_v20b() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible, prior = _V20B_REFERENCE()
    root = RESEARCH_ROOT / v20b.DEV_PRIMARY_NAME
    if not root.is_dir():
        raise FileNotFoundError("V21 requires the completed V20B development fixtures for disjointness checks")
    extra = [
        _read_json(root / "fixtures" / f"adaptation-seed-{index}.json")
        for index in range(len(fixtures.DEVELOPMENT_SEEDS))
    ]
    return visible, [*prior, *extra]


_V20B_REFERENCE = None


def _load_v21_dependencies() -> None:
    global _RUNTIME_PATCHED, _V20B_REFERENCE
    _V20B_LOAD_DEPENDENCIES()
    if _RUNTIME_PATCHED:
        return
    v20.fixtures = fixtures
    v20.v19.fixtures = fixtures
    _V20B_REFERENCE = v20.v19._reference_payloads
    v20.v19._reference_payloads = _reference_payloads_with_v20b
    v20._training_set = _training_set_with_endpoint
    v20._train_pair = _train_pair_with_endpoint
    v20._train_residual = _train_endpoint_residual
    _RUNTIME_PATCHED = True


def _write_json_v21(path: Path, value: Any, *, immutable: bool = False) -> None:
    if Path(path).name == "candidate-freeze.json" and isinstance(value, dict):
        training = value.get("training")
        if isinstance(training, dict):
            training["residual_objective"] = (
                "equal-weight normalized squared error at teacher factual and counterfactual endpoints; "
                "no finite-effect term or base-drift anchor"
            )
            training["endpoint_target_semantics"] = "teacher outputs m(x) and m(x+delta) are separately supervised"
            value.pop("candidate_set_sha256", None)
            value.pop("candidate_freeze_sha256", None)
            value["candidate_set_sha256"] = _digest(value)
            value["candidate_freeze_sha256"] = _digest({key: item for key, item in value.items() if key != "candidate_freeze_sha256"})
    _V20_WRITE_JSON(path, value, immutable=immutable)


def _python_exec(executable: Path, arguments: list[str], *, log_path: Path) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(f"pinned V21 runtime is missing: {executable}")
    completed = subprocess.run(
        [str(executable), str(Path(__file__).resolve()), *arguments],
        check=False, capture_output=True, text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"V21 pinned runtime exited {completed.returncode}; see external-custody log {log_path.name}")
    decoder = json.JSONDecoder()
    for index, character in enumerate(completed.stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[index:])
        except json.JSONDecodeError:
            continue
        if not completed.stdout[index + end :].strip() and isinstance(value, dict):
            return value
    raise ValueError("V21 pinned runtime did not return one JSON receipt")


def _rewrite_markdown(writer, *args: Any, **kwargs: Any):
    payload = writer(*args, **kwargs)
    for relative in (DEV_RESULT_MD, CONFIRM_RESULT_MD):
        path = REPO_ROOT / relative
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            text = text.replace("Replacement-loss V20 ", "Replacement-loss V21 ")
            text = text.replace("theseus-replacement-loss-v20-low-rank-causal-residual-v1", PROTOCOL_SLICE)
            path.write_text(text, encoding="utf-8")
    return payload


def _write_development_results_v21(replay: dict[str, Any], primary: Path) -> dict[str, Any]:
    payload = _rewrite_markdown(v20._write_development_results_original, replay, primary)
    diverged = False
    for cell in payload["selection"]["cells"]:
        behavior_pass = (
            cell["validation_cross_entropy_increase"] <= 0.05
            and cell["median_normalized_next_logit_error"] <= 0.05
        )
        mechanism_pass = cell["effect_mse_ratio_zero"] <= 0.90
        diverged = diverged or behavior_pass != mechanism_pass
    payload["behavior_and_mechanism_diverged"] = diverged
    payload.pop("result_document_sha256", None)
    payload["result_document_sha256"] = _digest(payload)
    _write_json_v21(REPO_ROOT / DEV_RESULT_JSON, payload)
    markdown_path = REPO_ROOT / DEV_RESULT_MD
    markdown = markdown_path.read_text(encoding="utf-8")
    markdown += f"\nBehavior and mechanism diverged: `{str(diverged).lower()}`.\n"
    markdown_path.write_text(markdown, encoding="utf-8")
    return payload


def _install() -> None:
    v20.STATE_SLICE = STATE_SLICE
    v20.PROTOCOL_SLICE = PROTOCOL_SLICE
    v20.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    v20.DEV_PROTOCOL_ID = DEV_PROTOCOL_ID
    v20.CONFIRM_PROTOCOL_ID = CONFIRM_PROTOCOL_ID
    v20.DEV_FREEZE_FILE = DEV_FREEZE_FILE
    v20.CONFIRM_FREEZE_FILE = CONFIRM_FREEZE_FILE
    v20.CONFIRM_CONFIG_FILE = CONFIRM_CONFIG_FILE
    v20.DEV_RESULT_JSON = DEV_RESULT_JSON
    v20.DEV_RESULT_MD = DEV_RESULT_MD
    v20.DEV_REPLICATION_JSON = DEV_REPLICATION_JSON
    v20.CONFIRM_RESULT_JSON = CONFIRM_RESULT_JSON
    v20.CONFIRM_RESULT_MD = CONFIRM_RESULT_MD
    v20.CONFIRM_REPLICATION_JSON = CONFIRM_REPLICATION_JSON
    v20.CLAIM_CEILING = CLAIM_CEILING
    v20.DEV_PRIMARY_NAME = DEV_PRIMARY_NAME
    v20.DEV_REPLICA_NAME = DEV_REPLICA_NAME
    v20.CONFIRM_PRIMARY_NAME = CONFIRM_PRIMARY_NAME
    v20.CONFIRM_REPLICA_NAME = CONFIRM_REPLICA_NAME
    v20.PROTOCOL_FILE = PROTOCOL_FILE
    v20.FIXTURE_FILE = FIXTURE_FILE
    v20.RUNNER_FILE = RUNNER_FILE
    v20.LOCK_FILE = LOCK_FILE
    v20.REPO_ROOT = REPO_ROOT
    v20.RESEARCH_ROOT = RESEARCH_ROOT
    v20.fixtures = fixtures
    v20._load_dependencies = _load_v21_dependencies
    v20._python_exec = _python_exec
    v20._packet = _packet
    v20.freeze_development = freeze_development
    v20.verify_development_packet = verify_development_packet
    v20.freeze_confirmation = freeze_confirmation
    v20.verify_confirmation_packet = verify_confirmation_packet
    v20._write_json = _write_json_v21
    v20._write_development_results = _write_development_results_v21
    v20._write_confirmation_results = lambda replay, **kwargs: _rewrite_markdown(
        v20._write_confirmation_results_original, replay, **kwargs,
    )


def main(argv: list[str] | None = None) -> int:
    if not hasattr(v20, "_write_development_results_original"):
        v20._write_development_results_original = v20._write_development_results
    if not hasattr(v20, "_write_confirmation_results_original"):
        v20._write_confirmation_results_original = v20._write_confirmation_results
    v20b._install()
    _install()
    return v20.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
