"""Frozen local runner for causal-supervised whole-MLP replacement.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v17-causal-supervised-whole-mlp-v1, and
global-experiment-assurance-lifecycle-v1.

All prompt-bearing and row-level artifacts stay in external owner custody.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import secrets
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import replacement_loss_v16_runner as v16
import replacement_loss_v17_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v17-freeze.json"
CLAIM_CEILING = "LOCAL_CAUSAL_SUPERVISION_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT"
TARGET_LAYER = 8
ADJACENT_LAYER = 9
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v17-causal-supervision-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v17-causal-supervision-replica"
RESEARCH_ROOT = v16.RESEARCH_ROOT
REPO_ROOT = Path(__file__).resolve().parent
TRAIN_UPDATES = 1000
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
INFERENCE_BATCH_SIZE = 4
PACKET_FILES = (
    "AGENTS.md",
    "docs/replacement-loss-v17-causal-supervision.md",
    "replacement_loss_v17_fixtures.py",
    "replacement_loss_v17_runner.py",
    "requirements-v17.lock.txt",
    "docs/replacement-loss-v16-freeze.json",
    "docs/replacement-loss-v16-results.md",
    "docs/replacement-loss-v16-whole-mlp.md",
    "replacement_loss_v16_fixtures.py",
    "replacement_loss_v16_runner.py",
    "requirements-v16.lock.txt",
    "replacement_loss_v3_fixtures.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    v16._write_json(path, value, immutable=immutable)


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V17 frozen packet source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def freeze_packet() -> dict[str, Any]:
    path = REPO_ROOT / FREEZE_FILE
    if path.exists():
        return verify_packet()
    v16.verify_packet()
    files = _packet_rows()
    packet = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": files,
        "packet_sha256": _digest(files),
        "parent_protocol": {
            "protocol_id": v16.PROTOCOL_ID,
            "packet_sha256": v16.verify_packet()["packet_sha256"],
            "scientific_outcome": "FAIL",
            "selected_layer": TARGET_LAYER,
            "post_result_follow_up": True,
        },
        "target_identity": {
            "model_id": v16.TARGET_MODEL_ID,
            "revision": v16.TARGET_REVISION,
            "tree_sha256": v16.TARGET_TREE_SHA256,
            "weight_sha256": v16.TARGET_WEIGHT_SHA256,
            "target_known_before_protocol": True,
            "fresh_checkpoint_claim": False,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_packet() -> dict[str, Any]:
    v16.verify_packet()
    path = REPO_ROOT / FREEZE_FILE
    if not path.is_file():
        raise FileNotFoundError("V17 packet is not frozen")
    frozen = _read_json(path)
    current = _packet_rows()
    if frozen.get("packet_files") != current or frozen.get("packet_sha256") != _digest(current):
        raise ValueError("V17 packet source bytes differ from the frozen packet")
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _digest(material):
        raise ValueError("V17 freeze digest is invalid")
    if (
        frozen.get("protocol_id") != PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("parent_protocol") != {
            "protocol_id": v16.PROTOCOL_ID,
            "packet_sha256": v16.verify_packet()["packet_sha256"],
            "scientific_outcome": "FAIL",
            "selected_layer": TARGET_LAYER,
            "post_result_follow_up": True,
        }
    ):
        raise ValueError("V17 frozen protocol identity or parent binding differs")
    return frozen


def _custody_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.parent != RESEARCH_ROOT or root.name not in {PRIMARY_NAME, REPLICA_NAME} or root.is_symlink():
        raise ValueError("custody root must be one of the two frozen V17 PrimaryED roots")
    return root


def _external_packet_archive(root: Path) -> dict[str, Any]:
    packet = verify_packet()
    archive = root / "frozen-packet"
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("external V17 archive binds a different packet")
    for row in packet["packet_files"]:
        path = archive / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"external V17 packet file differs: {row['path']}")
    freeze_copy = archive / FREEZE_FILE
    if not freeze_copy.is_file() or _sha_file(freeze_copy) != _sha_file(REPO_ROOT / FREEZE_FILE):
        raise ValueError("external V17 archive has a different freeze record")
    expected = list(packet["packet_files"])
    expected.append({"path": FREEZE_FILE, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected or manifest.get("archive_sha256") != _digest(expected):
        raise ValueError("external V17 archive manifest is invalid")
    return manifest


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _custody_root(custody_path)
    if root.exists():
        raise FileExistsError("V17 custody root already exists; preserve it and choose a fresh root")
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "frozen-packet"
    for row in packet["packet_files"]:
        source = REPO_ROOT / row["path"]
        destination = archive / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    freeze_source = REPO_ROOT / FREEZE_FILE
    freeze_destination = archive / FREEZE_FILE
    freeze_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(freeze_source, freeze_destination)
    rows = [{"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]} for row in packet["packet_files"]]
    rows.append({"path": FREEZE_FILE, "sha256": _sha_file(freeze_destination), "bytes": freeze_destination.stat().st_size})
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"],
        "files": rows,
        "archive_sha256": _digest(rows),
    }
    _write_json(root / "frozen-packet-manifest.json", manifest, immutable=True)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "archive_sha256": manifest["archive_sha256"], "file_count": len(rows)}


def _ensure_root(value: str | Path) -> Path:
    root = _custody_root(value)
    if not root.is_dir():
        raise FileNotFoundError("V17 external custody root is missing")
    if (root / "run-stop.json").exists():
        raise RuntimeError("this V17 custody root has a terminal stop receipt; do not reuse it")
    _external_packet_archive(root)
    return root


def _runtime_receipt() -> dict[str, Any]:
    receipt = v16._runtime_receipt()
    v16_lock = _parse_lock(REPO_ROOT / "requirements-v16.lock.txt")
    v17_lock = _parse_lock(REPO_ROOT / "requirements-v17.lock.txt")
    if v16_lock != v17_lock:
        raise ValueError("V17 dependency pins differ from the reproduced V16 runtimes")
    receipt["dependency_lock_sha256"] = _sha_file(REPO_ROOT / "requirements-v17.lock.txt")
    return receipt


def _parse_lock(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise ValueError(f"invalid exact dependency pin in {path.name}")
        result[name.casefold().replace("_", "-")] = version
    return result


def _tree_receipt() -> dict[str, Any]:
    receipt = v16._target_tree_receipt()
    if receipt["tree_sha256"] != v16.TARGET_TREE_SHA256 or receipt["weight_sha256"] != v16.TARGET_WEIGHT_SHA256:
        raise ValueError("V17 checkpoint differs from the predeclared V16 target")
    return receipt


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    v16._save_npz_deterministic(path, **arrays)


def _split(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed ^ 0x17D15A)
    fit: list[int] = []
    validation: list[int] = []
    for stratum in fixtures.STRATA:
        indices = np.asarray([i for i, row in enumerate(records) if row["stratum"] == stratum], dtype=np.int32)
        if len(indices) != fixtures.PER_STRATUM_COUNT:
            raise ValueError("V17 adaptation fixture count differs from frozen strata")
        rng.shuffle(indices)
        fit.extend(int(value) for value in indices[:24])
        validation.extend(int(value) for value in indices[24:])
    return {"fit_prompt_indices": fit, "validation_prompt_indices": validation}


def _directions(
    inputs: np.ndarray,
    prompt_indices: np.ndarray,
    records: list[dict[str, Any]],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed ^ 0x17CA05A1)
    deltas = np.zeros_like(inputs, dtype=np.float32)
    for index, (activation, prompt_index) in enumerate(zip(inputs, prompt_indices, strict=True)):
        stratum = records[int(prompt_index)]["stratum"]
        rms = max(float(np.sqrt(np.mean(np.square(activation, dtype=np.float64)))), 1e-12)
        if stratum == "adversarial":
            vector = np.zeros_like(activation, dtype=np.float64)
            vector[int(np.argmax(np.abs(activation)))] = 1.0 if rng.integers(0, 2) else -1.0
        else:
            vector = rng.normal(size=activation.shape).astype(np.float64)
            if stratum == "compositional":
                vector += rng.normal(size=activation.shape)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            raise ValueError("V17 causal-training direction has zero norm")
        deltas[index] = (vector * (0.01 * rms / norm)).astype(np.float32)
    return deltas


def _heldout_directions(
    inputs: np.ndarray,
    records: list[dict[str, Any]],
    rng: np.random.Generator,
) -> np.ndarray:
    deltas = np.zeros_like(inputs, dtype=np.float32)
    for index, (activation, record) in enumerate(zip(inputs, records, strict=True)):
        rms = max(float(np.sqrt(np.mean(np.square(activation, dtype=np.float64)))), 1e-12)
        if record["stratum"] == "adversarial":
            vector = np.zeros_like(activation, dtype=np.float64)
            vector[int(np.argmax(np.abs(activation)))] = 1.0 if rng.integers(0, 2) else -1.0
        else:
            vector = rng.normal(size=activation.shape).astype(np.float64)
            if record["stratum"] == "compositional":
                vector += rng.normal(size=activation.shape)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            raise ValueError("V17 held-out challenge direction has zero norm")
        deltas[index] = (vector * (0.01 * rms / norm)).astype(np.float32)
    return deltas


def _teacher_effects(layer: torch.nn.Module, inputs: np.ndarray, deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base_chunks = []
    effect_chunks = []
    with torch.inference_mode():
        for start in range(0, len(inputs), 1024):
            x = torch.from_numpy(np.asarray(inputs[start : start + 1024], dtype=np.float32))
            delta = torch.from_numpy(np.asarray(deltas[start : start + 1024], dtype=np.float32))
            y = layer.fc2(F.relu(layer.fc1(x)))
            y_changed = layer.fc2(F.relu(layer.fc1(x + delta)))
            base_chunks.append(y.cpu().numpy())
            effect_chunks.append((y_changed - y).cpu().numpy())
    return np.concatenate(base_chunks, axis=0), np.concatenate(effect_chunks, axis=0)


def _training_set(
    layer: torch.nn.Module,
    captures: dict[str, np.ndarray],
    records: list[dict[str, Any]],
    prompt_set: set[int],
    *,
    seed: int,
    label: str,
) -> dict[str, np.ndarray]:
    prompt_indices = captures["prompt_index"]
    mask = np.isin(prompt_indices, np.asarray(sorted(prompt_set), dtype=np.int32))
    x = captures[f"x_{label}"][mask]
    captured_y = captures[f"y_{label}"][mask]
    selected_prompts = prompt_indices[mask]
    directions = _directions(x, selected_prompts, records, seed=seed)
    teacher_y, teacher_effect = _teacher_effects(layer, x, directions)
    if teacher_y.shape != captured_y.shape or not np.allclose(teacher_y, captured_y, rtol=1e-5, atol=1e-6):
        raise ValueError("V17 direct teacher function differs from captured feed-forward outputs")
    return {"x": x, "y": teacher_y.astype(np.float32), "delta": directions, "effect": teacher_effect.astype(np.float32)}


def _train_candidate(
    data: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    seed: int,
    causal_weight: float,
    root: Path,
) -> tuple[torch.nn.Module, torch.nn.Module, dict[str, Any]]:
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    delta = torch.from_numpy(data["delta"])
    effect = torch.from_numpy(data["effect"])
    xv = torch.from_numpy(validation["x"])
    yv = torch.from_numpy(validation["y"])
    deltav = torch.from_numpy(validation["delta"])
    effectv = torch.from_numpy(validation["effect"])
    candidate = v16._new_candidate(seed)
    random_control = copy.deepcopy(candidate)
    initialization_sha = v16._module_state_digest(candidate)
    value_normalizer = max(float(y.square().mean().item()), 1e-12)
    effect_normalizer = max(float(effect.square().mean().item()), 1e-12)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x17A5C3)
    draws = np.empty((TRAIN_UPDATES, TRAIN_BATCH_SIZE), dtype=np.int32)
    candidate.train()
    for update in range(TRAIN_UPDATES):
        indices = torch.randint(0, x.shape[0], (TRAIN_BATCH_SIZE,), generator=generator)
        draws[update] = indices.numpy()
        x_batch = x.index_select(0, indices)
        y_batch = y.index_select(0, indices)
        delta_batch = delta.index_select(0, indices)
        effect_batch = effect.index_select(0, indices)
        predicted = candidate(x_batch)
        value_loss = F.mse_loss(predicted, y_batch) / value_normalizer
        if causal_weight > 0.0:
            predicted_effect = candidate(x_batch + delta_batch) - predicted
            causal_loss = F.mse_loss(predicted_effect, effect_batch) / effect_normalizer
            loss = (1.0 - causal_weight) * value_loss + causal_weight * causal_loss
        else:
            loss = value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V17 candidate training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if (update + 1) % 100 == 0:
            v16._check_caps(root)
    candidate.eval()
    random_control.eval()
    with torch.inference_mode():
        fit_value = float(F.mse_loss(candidate(x), y).item() / value_normalizer)
        fit_effect = float(F.mse_loss(candidate(x + delta) - candidate(x), effect).item() / effect_normalizer)
        validation_value = float(F.mse_loss(candidate(xv), yv).item() / value_normalizer)
        validation_effect = float(F.mse_loss(candidate(xv + deltav) - candidate(xv), effectv).item() / effect_normalizer)
    receipt = {
        "initialization_sha256": initialization_sha,
        "candidate_state_sha256": v16._module_state_digest(candidate),
        "fit_token_rows": int(len(x)),
        "validation_token_rows": int(len(xv)),
        "updates": TRAIN_UPDATES,
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(draws.astype("<i4", copy=False).tobytes()).hexdigest(),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP,
        "causal_loss_weight": causal_weight,
        "value_normalizer": value_normalizer,
        "effect_normalizer": effect_normalizer,
        "normalized_fit_value_mse": fit_value,
        "normalized_fit_effect_mse": fit_effect,
        "normalized_validation_value_mse": validation_value,
        "normalized_validation_effect_mse": validation_effect,
        "source_parameters_copied": False,
        "fit_input_sha256": hashlib.sha256(data["x"].astype("<f4", copy=False).tobytes()).hexdigest(),
        "fit_effect_target_sha256": hashlib.sha256(data["effect"].astype("<f4", copy=False).tobytes()).hexdigest(),
    }
    return candidate, random_control, receipt


def _artifact_rows(root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V17 preparation artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _candidate_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "candidate-freeze.json")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("V17 candidate freeze binds a different protocol packet")
    material = {key: value for key, value in manifest.items() if key not in {"candidate_set_sha256", "candidate_freeze_sha256"}}
    if manifest.get("candidate_set_sha256") != _digest(material):
        raise ValueError("V17 candidate set digest is invalid")
    freeze_material = {key: value for key, value in manifest.items() if key != "candidate_freeze_sha256"}
    if manifest.get("candidate_freeze_sha256") != _digest(freeze_material):
        raise ValueError("V17 candidate freeze digest is invalid")
    expected = {(seed, method) for seed in range(len(fixtures.SEEDS)) for method in ("causal", "pointwise", "wrong_layer", "random")}
    rows = manifest.get("candidates", [])
    if len(rows) != len(expected) or {(row.get("seed_index"), row.get("method")) for row in rows} != expected:
        raise ValueError("V17 candidate freeze omits or duplicates a method/seed candidate")
    expected_artifacts = {
        "fixtures/v16-visible-reference.json",
        *(f"fixtures/adaptation-seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
        *(f"splits/seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
        *(f"adaptation/seed-{seed}.npz" for seed in range(len(fixtures.SEEDS))),
    }
    artifacts = manifest.get("preparation_artifacts", [])
    if len(artifacts) != len(expected_artifacts) or {row.get("path") for row in artifacts} != expected_artifacts:
        raise ValueError("V17 candidate freeze omits or duplicates preparation provenance")
    for artifact in artifacts:
        path = root / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or _sha_file(path) != artifact["sha256"]:
            raise ValueError("V17 preparation artifact differs from its frozen manifest")
    for row in rows:
        path = root / row["path"]
        if not path.is_file() or _sha_file(path) != row["file_sha256"]:
            raise ValueError("V17 candidate bytes differ from their frozen manifest")
    return manifest


def _save_candidate(path: Path, candidate: torch.nn.Module) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in candidate.state_dict().items()}
    save_file(state, str(path))
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": v16._module_state_digest(candidate),
        "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        "architecture": "Linear-ReLU-Linear",
    }


def prepare(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _ensure_root(custody_path)
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("V17 candidates already frozen; do not rerun preparation in place")
    started = time.monotonic()
    v16._phase_ledger(root)
    try:
        _tree_receipt()
        runtime = _runtime_receipt()
        model, tokenizer = v16._load_model()
        layers = v16._layers(model)
        if len(layers) != 12 or TARGET_LAYER != 8 or ADJACENT_LAYER != 9:
            raise ValueError("V17 fixed target blocks differ from the frozen OPT follow-up contract")
        v16_root = v16.RESEARCH_ROOT / v16.PRIMARY_NAME
        v16._external_packet_archive(v16_root)
        visible_path = v16_root / "fixtures" / "visible.json"
        selection_path = v16_root / "block-selection.json"
        visible_v16 = _read_json(visible_path)
        selection_v16 = _read_json(selection_path)
        if selection_v16.get("target_layer") != TARGET_LAYER or selection_v16.get("visible_fixture_sha256") != visible_v16.get("example_set_sha256"):
            raise ValueError("V17 preselected block is not bound to the archived V16 visible-only selection")
        adaptations = []
        records_by_seed = []
        splits = []
        for seed_index, seed in enumerate(fixtures.SEEDS):
            rows = fixtures.generate_adaptation_fixture(seed_index)
            payload = fixtures.fixture_payload(rows, fixture_kind=f"adaptation-seed-{seed_index}")
            adaptations.append(payload)
            records_by_seed.append(payload["records"])
            splits.append(_split(payload["records"], seed))
        fixtures.validate_partition_disjointness(visible_v16, *adaptations)
        _write_json(root / "fixtures" / "v16-visible-reference.json", {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": v16.PROTOCOL_ID,
            "source_example_set_sha256": visible_v16["example_set_sha256"],
            "source_count": visible_v16["count"],
            "source_selection_sha256": _sha_file(selection_path),
        }, immutable=True)
        for seed_index, payload in enumerate(adaptations):
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
            _write_json(root / "splits" / f"seed-{seed_index}.json", splits[seed_index], immutable=True)

        all_candidates = []
        for seed_index, (records, split) in enumerate(zip(records_by_seed, splits, strict=True)):
            v16._check_caps(root, started)
            captures = v16._capture_adaptation(model, tokenizer, records, TARGET_LAYER, ADJACENT_LAYER)
            _save_npz(root / "adaptation" / f"seed-{seed_index}.npz", **captures)
            fit_prompts = set(split["fit_prompt_indices"])
            validation_prompts = set(split["validation_prompt_indices"])
            prompt_ids = captures["prompt_index"]
            fit_mask = np.isin(prompt_ids, np.asarray(sorted(fit_prompts), dtype=np.int32))
            validation_mask = np.isin(prompt_ids, np.asarray(sorted(validation_prompts), dtype=np.int32))
            if not fit_mask.any() or not validation_mask.any() or np.any(fit_mask & validation_mask):
                raise ValueError("V17 prompt-level fit and validation partitions are invalid")
            fit_rows = {key: value[fit_mask] for key, value in captures.items()}
            validation_rows = {key: value[validation_mask] for key, value in captures.items()}
            training_seed = fixtures.SEEDS[seed_index] ^ 0x17B10C

            target_fit = _training_set(layers[TARGET_LAYER], fit_rows, records, fit_prompts, seed=training_seed, label="target")
            target_val = _training_set(layers[TARGET_LAYER], validation_rows, records, validation_prompts, seed=training_seed ^ 0x55A, label="target")
            adjacent_fit = _training_set(layers[ADJACENT_LAYER], fit_rows, records, fit_prompts, seed=training_seed ^ 0x9A1, label="adjacent")
            adjacent_val = _training_set(layers[ADJACENT_LAYER], validation_rows, records, validation_prompts, seed=training_seed ^ 0x9A1 ^ 0x55A, label="adjacent")
            if len(target_fit["x"]) != int(fit_mask.sum()) or len(target_val["x"]) != int(validation_mask.sum()):
                raise ValueError("V17 causal-training rows do not match captured token provenance")

            causal, random_control, causal_receipt = _train_candidate(target_fit, target_val, seed=training_seed, causal_weight=0.5, root=root)
            pointwise, pointwise_random, pointwise_receipt = _train_candidate(target_fit, target_val, seed=training_seed, causal_weight=0.0, root=root)
            if v16._module_state_digest(random_control) != v16._module_state_digest(pointwise_random):
                raise ValueError("V17 causal and pointwise controls did not share the frozen initialization")
            wrong, _wrong_random, wrong_receipt = _train_candidate(adjacent_fit, adjacent_val, seed=training_seed ^ 0x9A1, causal_weight=0.5, root=root)
            method_items = (
                ("causal", causal, causal_receipt),
                ("pointwise", pointwise, pointwise_receipt),
                ("wrong_layer", wrong, wrong_receipt),
                ("random", random_control, {"source_parameters_copied": False, "trained": False, "initialization_sha256": v16._module_state_digest(random_control)}),
            )
            for method, candidate, receipt in method_items:
                saved = _save_candidate(root / "candidates" / f"seed-{seed_index}-{method}.safetensors", candidate)
                all_candidates.append({"seed_index": seed_index, "method": method, **saved, "fit_receipt": receipt})
            del captures, causal, pointwise, wrong, random_control, pointwise_random, _wrong_random
            del target_fit, target_val, adjacent_fit, adjacent_val, fit_rows, validation_rows

        relatives = [
            "fixtures/v16-visible-reference.json",
            *(f"fixtures/adaptation-seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
            *(f"splits/seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
            *(f"adaptation/seed-{seed}.npz" for seed in range(len(fixtures.SEEDS))),
        ]
        material = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "target_tree_sha256": v16.TARGET_TREE_SHA256,
            "target_layer": TARGET_LAYER,
            "adjacent_control_layer": ADJACENT_LAYER,
            "preparation_artifacts": _artifact_rows(root, relatives),
            "candidates": all_candidates,
            "training": {
                "architecture": "Linear(768,3072)-ReLU-Linear(3072,768)",
                "updates": TRAIN_UPDATES,
                "batch_size": TRAIN_BATCH_SIZE,
                "causal_objective": "0.5 normalized pointwise MSE + 0.5 normalized finite-intervention effect MSE",
                "pointwise_control_objective": "normalized pointwise MSE only",
            },
            "runtime_versions": {key: runtime[key] for key in (
                "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "repository_head",
                "deterministic_algorithms", "torch_num_threads",
            )},
        }
        material["candidate_set_sha256"] = _digest(material)
        material["candidate_freeze_sha256"] = _digest(material)
        _write_json(root / "candidate-freeze.json", material, immutable=True)
        _write_json(root / "runtime.json", runtime, immutable=True)
        v16._record_phase(root, "prepare", started, "COMPLETED", details={
            "seed_count": len(fixtures.SEEDS),
            "visible_reference_count": visible_v16["count"],
            "target_layer": TARGET_LAYER,
            "candidate_count": len(all_candidates),
            "candidate_set_sha256": material["candidate_set_sha256"],
        })
        return {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "phase": "prepare",
            "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "candidate_set_sha256": material["candidate_set_sha256"],
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
            "data_and_artifact_custody": "adaptation prompts, captures, finite intervention targets, and candidates remain in external owner custody",
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": PROTOCOL_ID, "phase": "prepare", "status": "INCONCLUSIVE",
            "error_type": type(error).__name__, "error": str(error),
            "packet_sha256": verify_packet()["packet_sha256"], "timestamp_epoch": time.time(),
        }, immutable=True)
        v16._record_phase(root, "prepare", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _candidate_for(root: Path, seed_index: int, method: str, manifest: dict[str, Any]) -> torch.nn.Module:
    if method == "no_op":
        return v16.NoOpCandidate().eval()
    row = next((item for item in manifest["candidates"] if item["seed_index"] == seed_index and item["method"] == method), None)
    if row is None:
        raise ValueError(f"V17 candidate freeze lacks {method} for seed {seed_index}")
    candidate = v16.WholeMLPCandidate()
    path = root / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("V17 candidate bytes differ from the frozen candidate set")
    candidate.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if v16._module_state_digest(candidate) != row["state_sha256"]:
        raise ValueError("V17 candidate tensor state digest differs")
    return candidate.eval()


def _entropy_manifest(root: Path, candidate_set_sha: str) -> dict[str, Any]:
    manifest = _read_json(root / "challenge-entropy-manifest.json")
    rows = manifest.get("entropy", [])
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidate_set_sha
        or manifest.get("entropy_manifest_sha256") != _digest(rows)
        or [row.get("seed_index") for row in rows] != list(range(len(fixtures.SEEDS)))
    ):
        raise ValueError("V17 entropy manifest is invalid or bound to a different candidate set")
    for row in rows:
        path = root / "challenge-entropy" / f"seed-{row['seed_index']}.bin"
        if not path.is_file() or path.stat().st_size != 32 or _sha_file(path) != row["sha256"]:
            raise ValueError("V17 challenge entropy bytes differ from their manifest")
    return manifest


def _verify_fixture(root: Path, path: Path, kind: str, count: int) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("records")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("fixture_kind") != kind
        or payload.get("prompt_rules_version") != fixtures.PROMPT_RULES_VERSION
        or not isinstance(rows, list) or len(rows) != count or payload.get("count") != count
    ):
        raise ValueError("V17 fixture identity or size differs")
    digests = [hashlib.sha256(_canonical_bytes(row)).hexdigest() for row in rows]
    expected = hashlib.sha256("\n".join(sorted(digests)).encode("ascii")).hexdigest()
    if expected != payload.get("example_set_sha256") or len({row.get("example_id") for row in rows}) != count:
        raise ValueError("V17 fixture content digest is invalid")
    expected_counts = {name: fixtures.PER_STRATUM_COUNT for name in fixtures.STRATA} if kind.startswith("adaptation-") else {name: fixtures.CHALLENGES_PER_STRATUM for name in fixtures.STRATA}
    actual_counts = {name: sum(row.get("stratum") == name for row in rows) for name in sorted(expected_counts)}
    if actual_counts != dict(sorted(expected_counts.items())) or payload.get("stratum_counts") != actual_counts:
        raise ValueError("V17 fixture stratum distribution differs")
    return payload


def _prediction_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "prediction-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "prediction_set_sha256"}
    if manifest.get("prediction_set_sha256") != _digest(material):
        raise ValueError("V17 prediction-set digest is invalid")
    candidates = _candidate_manifest(root)
    entropy = _entropy_manifest(root, candidates["candidate_set_sha256"])
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidates["candidate_set_sha256"]
        or manifest.get("entropy_manifest_sha256") != entropy["entropy_manifest_sha256"]
    ):
        raise ValueError("V17 predictions do not bind the frozen packet, candidates, and entropy")
    v16_root = v16.RESEARCH_ROOT / v16.PRIMARY_NAME
    v16._external_packet_archive(v16_root)
    visible_path = v16_root / "fixtures" / "visible.json"
    selection_path = v16_root / "block-selection.json"
    visible = _read_json(visible_path)
    visible_reference = _read_json(root / "fixtures" / "v16-visible-reference.json")
    if (
        visible_reference.get("source_example_set_sha256") != visible.get("example_set_sha256")
        or visible_reference.get("source_selection_sha256") != _sha_file(selection_path)
    ):
        raise ValueError("V17 visible-selection reference differs from the archived V16 selection inputs")
    adaptation_payloads = []
    for seed_index in range(len(fixtures.SEEDS)):
        adaptation_payloads.append(_verify_fixture(root, root / "fixtures" / f"adaptation-seed-{seed_index}.json", f"adaptation-seed-{seed_index}", fixtures.PER_STRATUM_COUNT * len(fixtures.STRATA)))
    rows = manifest.get("challenge_rows", [])
    if [row.get("seed_index") for row in rows] != list(range(len(fixtures.SEEDS))):
        raise ValueError("V17 challenge manifest has missing or unordered seed rows")
    for row in rows:
        seed_index = int(row["seed_index"])
        challenge = _verify_fixture(root, root / "fixtures" / f"challenge-seed-{seed_index}.json", f"challenge-seed-{seed_index}", fixtures.CHALLENGES_PER_STRATUM * len(fixtures.STRATA))
        fixtures.validate_partition_disjointness(visible, *adaptation_payloads, challenge)
        if row.get("challenge_sha256") != challenge["example_set_sha256"] or row.get("challenge_count") != challenge["count"]:
            raise ValueError("V17 challenge fixture differs from its sealed prediction row")
        path = root / "predictions" / f"seed-{seed_index}.npz"
        if not path.is_file() or _sha_file(path) != row["prediction_file_sha256"]:
            raise ValueError("V17 prediction tensor file differs from the sealed manifest")
    if not visible.get("records") or manifest.get("target_layer") != TARGET_LAYER:
        raise ValueError("V17 target selection differs from the frozen visible V16 follow-up")
    return manifest


def _predict_seed(root: Path, seed_index: int, entropy: bytes, candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    records = fixtures.generate_challenge_fixture(seed_index, entropy)
    payload = fixtures.fixture_payload(records, fixture_kind=f"challenge-seed-{seed_index}")
    adaptation = _read_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json")
    fixtures.validate_partition_disjointness(adaptation, payload)
    _write_json(root / "fixtures" / f"challenge-seed-{seed_index}.json", payload, immutable=True)
    model, tokenizer = v16._load_model()
    rows = [v16._encode(tokenizer, record, with_continuation=True) for record in payload["records"]]
    methods = ("causal", "pointwise", "wrong_layer", "random", "no_op")
    outputs: dict[str, dict[str, list[np.ndarray]]] = {
        method: {name: [] for name in ("module_delta", "logit_delta", "module_base", "boundary_logits", "continuation_ce")}
        for method in methods
    }
    direction_chunks = []
    direction_seed = int.from_bytes(hashlib.sha256(entropy + b"v17-heldout-directions").digest()[:8], "big")
    direction_rng = np.random.default_rng(direction_seed)
    for start in range(0, len(rows), INFERENCE_BATCH_SIZE):
        v16._check_caps(root)
        group = rows[start : start + INFERENCE_BATCH_SIZE]
        positions = [int(row["prompt_len"]) - 1 for row in group]
        causal_candidate = _candidate_for(root, seed_index, "causal", candidate_manifest)
        causal_base = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=causal_candidate, positions=positions, capture_inputs=True, capture_outputs=True)
        directions = _heldout_directions(causal_base["module_input"], group, direction_rng)
        direction_chunks.append(directions)
        for method in methods:
            candidate = causal_candidate if method == "causal" else _candidate_for(root, seed_index, method, candidate_manifest)
            base = causal_base if method == "causal" else v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=candidate, positions=positions, capture_outputs=True)
            changed = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=candidate, positions=positions, deltas=directions, capture_outputs=True)
            outputs[method]["module_delta"].append((changed["module_output"] - base["module_output"]).astype(np.float32))
            outputs[method]["logit_delta"].append((changed["boundary_logits"] - base["boundary_logits"]).astype(np.float32))
            outputs[method]["module_base"].append(base["module_output"].astype(np.float32))
            outputs[method]["boundary_logits"].append(base["boundary_logits"].astype(np.float32))
            outputs[method]["continuation_ce"].append(base["continuation_ce"].astype(np.float32))
    arrays: dict[str, np.ndarray] = {"directions": np.concatenate(direction_chunks, axis=0)}
    for method, metrics in outputs.items():
        for metric, chunks in metrics.items():
            arrays[f"{method}_{metric}"] = np.concatenate(chunks, axis=0)
    path = root / "predictions" / f"seed-{seed_index}.npz"
    _save_npz(path, **arrays)
    return {"seed_index": seed_index, "challenge_count": len(rows), "challenge_sha256": payload["example_set_sha256"], "prediction_file_sha256": _sha_file(path), "prediction_file_bytes": path.stat().st_size}


def predict(custody_path: str | Path, *, entropy_from: str | Path | None = None) -> dict[str, Any]:
    verify_packet()
    root = _ensure_root(custody_path)
    if not (root / "candidate-freeze.json").is_file():
        raise FileNotFoundError("V17 candidates must be frozen before challenge generation")
    if (root / "prediction-manifest.json").exists():
        raise FileExistsError("V17 predictions already sealed; do not overwrite")
    started = time.monotonic()
    v16._phase_ledger(root)
    try:
        candidates = _candidate_manifest(root)
        entropy_root = root / "challenge-entropy"
        entropy_root.mkdir(parents=True, exist_ok=True)
        if entropy_from is None:
            for seed_index in range(len(fixtures.SEEDS)):
                path = entropy_root / f"seed-{seed_index}.bin"
                if path.exists():
                    raise FileExistsError("V17 entropy exists; do not redraw or overwrite")
                path.write_bytes(secrets.token_bytes(32))
        else:
            source = _ensure_root(entropy_from)
            source_candidates = _candidate_manifest(source)
            _entropy_manifest(source, source_candidates["candidate_set_sha256"])
            for seed_index in range(len(fixtures.SEEDS)):
                source_path = source / "challenge-entropy" / f"seed-{seed_index}.bin"
                target_path = entropy_root / source_path.name
                if target_path.exists():
                    raise FileExistsError("replica entropy destination exists")
                shutil.copyfile(source_path, target_path)
        entropy_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            raw = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            if len(raw) != 32:
                raise ValueError("V17 entropy must contain 256 bits per seed")
            entropy_rows.append({"seed_index": seed_index, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": 32})
        entropy_record = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "entropy": entropy_rows,
            "entropy_manifest_sha256": _digest(entropy_rows),
        }
        _write_json(root / "challenge-entropy-manifest.json", entropy_record, immutable=True)
        _entropy_manifest(root, candidates["candidate_set_sha256"])
        challenge_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            v16._check_caps(root, started)
            secret = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            challenge_rows.append(_predict_seed(root, seed_index, secret, candidates))
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "entropy_manifest_sha256": entropy_record["entropy_manifest_sha256"],
            "target_layer": TARGET_LAYER,
            "challenge_rows": challenge_rows,
        }
        manifest["prediction_set_sha256"] = _digest(manifest)
        _write_json(root / "prediction-manifest.json", manifest, immutable=True)
        _prediction_manifest(root)
        v16._record_phase(root, "predict", started, "COMPLETED", details={"challenge_count": sum(row["challenge_count"] for row in challenge_rows), "prediction_set_sha256": manifest["prediction_set_sha256"]})
        return {
            "protocol_id": PROTOCOL_ID, "phase": "predict", "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS", "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED", "challenge_count": sum(row["challenge_count"] for row in challenge_rows),
            "prediction_set_sha256": manifest["prediction_set_sha256"], "candidate_was_frozen_before_challenge": True,
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {"protocol_id": PROTOCOL_ID, "phase": "predict", "status": "INCONCLUSIVE", "error_type": type(error).__name__, "error": str(error), "packet_sha256": verify_packet()["packet_sha256"], "timestamp_epoch": time.time()}, immutable=True)
        v16._record_phase(root, "predict", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _score_seed(root: Path, seed_index: int) -> dict[str, Any]:
    payload = _verify_fixture(root, root / "fixtures" / f"challenge-seed-{seed_index}.json", f"challenge-seed-{seed_index}", fixtures.CHALLENGES_PER_STRATUM * len(fixtures.STRATA))
    model, tokenizer = v16._load_model()
    rows = [v16._encode(tokenizer, record, with_continuation=True) for record in payload["records"]]
    with np.load(root / "predictions" / f"seed-{seed_index}.npz", allow_pickle=False) as source:
        prediction = {name: source[name] for name in source.files}
    methods = ("causal", "pointwise", "wrong_layer", "random", "no_op")
    metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    activation = []
    behavior = []
    oracle_errors = []
    candidates = _candidate_manifest(root)
    for start in range(0, len(rows), INFERENCE_BATCH_SIZE):
        v16._check_caps(root)
        group = rows[start : start + INFERENCE_BATCH_SIZE]
        pos = [int(row["prompt_len"]) - 1 for row in group]
        original = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=None, positions=pos, capture_inputs=True, capture_outputs=True, capture_all_outputs=True)
        directions = prediction["directions"][start : start + len(group)]
        changed_teacher = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=None, positions=pos, deltas=directions, capture_outputs=True)
        teacher_module_delta = changed_teacher["module_output"] - original["module_output"]
        teacher_logit_delta = changed_teacher["boundary_logits"] - original["boundary_logits"]
        scale_module = np.maximum(np.linalg.norm(original["module_output"], axis=1), 1e-12)
        scale_logits = np.maximum(np.linalg.norm(original["boundary_logits"], axis=1), 1e-12)
        group_records = payload["records"][start : start + len(group)]
        for method in methods:
            predicted_module = prediction[f"{method}_module_delta"][start : start + len(group)]
            predicted_logits = prediction[f"{method}_logit_delta"][start : start + len(group)]
            module_rows = v16._metric_rows(predicted_module, teacher_module_delta, scale_module)
            logit_rows = v16._metric_rows(predicted_logits, teacher_logit_delta, scale_logits)
            metrics[method].extend({"stratum": row["stratum"], "module": module, "logit": logit} for row, module, logit in zip(group_records, module_rows, logit_rows, strict=True))
        candidate_module = prediction["causal_module_base"][start : start + len(group)]
        for row, replacement, teacher in zip(group_records, candidate_module, original["module_output"], strict=True):
            a = np.asarray(replacement, dtype=np.float64)
            b = np.asarray(teacher, dtype=np.float64)
            activation.append({"stratum": row["stratum"], "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30)), "relative_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))})
        baseline_logits = prediction["causal_boundary_logits"][start : start + len(group)]
        baseline_ce = prediction["causal_continuation_ce"][start : start + len(group)]
        for i, row in enumerate(group_records):
            behavior.append({
                "stratum": row["stratum"],
                "cross_entropy_increase": float(baseline_ce[i] - original["continuation_ce"][i]),
                "normalized_next_logit_error": float(np.linalg.norm(baseline_logits[i].astype(np.float64) - original["boundary_logits"][i].astype(np.float64)) / max(np.linalg.norm(original["boundary_logits"][i]), 1e-12)),
            })
        causal = _candidate_for(root, seed_index, "causal", candidates)
        oracle = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=causal, positions=pos, capture_outputs=False, restore_all_outputs=original["all_module_output"])
        oracle_errors.append(float((original["all_logits"][original["attention_mask"].bool()] - oracle["all_logits"][original["attention_mask"].bool()]).abs().max().item()))
    result = {"seed_index": seed_index, "strata": {}}
    for stratum in fixtures.STRATA:
        result["strata"][stratum] = {}
        for method in methods:
            selected = [row for row in metrics[method] if row["stratum"] == stratum]
            result["strata"][stratum][method] = {
                "module_effect": v16._summarize_effect([row["module"] for row in selected]),
                "next_logit_effect": v16._summarize_effect([row["logit"] for row in selected]),
            }
        selected_activation = [row for row in activation if row["stratum"] == stratum]
        selected_behavior = [row for row in behavior if row["stratum"] == stratum]
        causal_logit = result["strata"][stratum]["causal"]["next_logit_effect"]
        pointwise_logit = result["strata"][stratum]["pointwise"]["next_logit_effect"]
        result["strata"][stratum]["activation"] = {
            "median_cosine": float(np.median([row["cosine"] for row in selected_activation])),
            "median_relative_error": float(np.median([row["relative_error"] for row in selected_activation])),
        }
        result["strata"][stratum]["behavior"] = {
            "mean_cross_entropy_increase": float(np.mean([row["cross_entropy_increase"] for row in selected_behavior])),
            "median_normalized_next_logit_error": float(np.median([row["normalized_next_logit_error"] for row in selected_behavior])),
        }
        result["strata"][stratum]["causal_vs_pointwise"] = {
            "next_logit_median_cosine_gain": (
                float(causal_logit["median_cosine"] - pointwise_logit["median_cosine"])
                if causal_logit["median_cosine"] is not None and pointwise_logit["median_cosine"] is not None else None
            ),
            "next_logit_relative_error_reduction": (
                float(pointwise_logit["median_normalized_error"] - causal_logit["median_normalized_error"])
                if causal_logit["median_normalized_error"] is not None and pointwise_logit["median_normalized_error"] is not None else None
            ),
        }
    result["oracle_max_absolute_logit_error"] = max(oracle_errors)
    return result


def _scientific_outcome(results: list[dict[str, Any]]) -> str:
    incomplete = False
    passed = True
    for seed in results:
        for stratum in seed["strata"].values():
            causal = stratum["causal"]
            for metric in (causal["module_effect"], causal["next_logit_effect"]):
                if metric["resolvable_count"] < 16:
                    incomplete = True
                passed &= v16._passes_effect(metric)
            passed &= stratum["activation"]["median_cosine"] >= 0.90
            passed &= stratum["activation"]["median_relative_error"] <= 0.20
            passed &= stratum["behavior"]["mean_cross_entropy_increase"] <= 0.05
            passed &= stratum["behavior"]["median_normalized_next_logit_error"] <= 0.05
            comparison = stratum["causal_vs_pointwise"]
            if comparison["next_logit_median_cosine_gain"] is None or comparison["next_logit_relative_error_reduction"] is None:
                incomplete = True
                passed = False
            else:
                passed &= comparison["next_logit_median_cosine_gain"] >= 0.05
                passed &= comparison["next_logit_relative_error_reduction"] >= 0.10
            for name in ("wrong_layer", "random", "no_op"):
                control_cosine = stratum[name]["next_logit_effect"]["median_cosine"]
                treatment_cosine = causal["next_logit_effect"]["median_cosine"]
                if control_cosine is None or treatment_cosine is None:
                    incomplete = True
                    passed = False
                else:
                    passed &= treatment_cosine - control_cosine >= 0.05
        passed &= seed["oracle_max_absolute_logit_error"] <= 1e-5
    return "INCONCLUSIVE" if incomplete else "PASS" if passed else "FAIL"


def _diverged(results: list[dict[str, Any]]) -> bool | None:
    values = []
    for seed in results:
        for row in seed["strata"].values():
            values.extend((
                (row["behavior"]["mean_cross_entropy_increase"], 0.05),
                (row["behavior"]["median_normalized_next_logit_error"], 0.05),
                (row["activation"]["median_relative_error"], 0.20),
                (row["causal"]["module_effect"]["median_normalized_error"], 0.20),
                (row["causal"]["next_logit_effect"]["median_normalized_error"], 0.20),
            ))
    if any(value is None for value, _threshold in values):
        return None
    return any(float(value) > threshold for value, threshold in values)


def score(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _ensure_root(custody_path)
    if not (root / "prediction-manifest.json").is_file():
        raise FileNotFoundError("V17 sealed predictions must exist before scoring")
    if (root / "outcome-report.json").exists():
        raise FileExistsError("V17 outcomes already exist; do not rescore in place")
    started = time.monotonic()
    v16._phase_ledger(root)
    try:
        candidates = _candidate_manifest(root)
        predictions = _prediction_manifest(root)
        results = [_score_seed(root, seed_index) for seed_index in range(len(fixtures.SEEDS))]
        outcome = _scientific_outcome(results)
        report = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "prediction_set_sha256": predictions["prediction_set_sha256"],
            "target_identity": {
                "model_id": v16.TARGET_MODEL_ID,
                "revision": v16.TARGET_REVISION,
                "tree_sha256": v16.TARGET_TREE_SHA256,
                "weight_sha256": v16.TARGET_WEIGHT_SHA256,
                "target_layer": TARGET_LAYER,
                "selection_source": "V16 visible-only selection fixed before V17 protocol",
            },
            "scientific_outcome": outcome,
            "behavior_and_mechanism_diverged": _diverged(results),
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "blind_status": "NOT_BLIND_LOCAL_CUSTODY_RUN",
            "results": results,
            "claim_ceiling": CLAIM_CEILING,
            "data_and_artifact_custody": "V17 prompts, causal targets, captures, candidates, entropy, predictions, and row-level outcomes remain in external owner custody; V16 challenge data are excluded from V17 training",
            "runtime": _runtime_receipt(),
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
            "limitations": [
                "post-V16 outcome-aware follow-up on one reused OPT/ReLU checkpoint and the previously selected block 8",
                "synthetic prompts and three descriptive seeds do not constitute architecture or checkpoint replication",
                "finite-intervention supervision tests local response prediction, not metaphysical identity or consciousness",
                "oracle teacher-output restoration is a pipeline control, not learned rescue",
                "same-owner second-runtime replay is not blind replication or independent acceptance",
            ],
            "next_blocked_authority_gate": "blind external replication and an authorized independent reviewer are required for independent scientific acceptance",
        }
        scientific = {key: value for key, value in report.items() if key not in {"assurance_level", "reviewer_status", "reproducibility_status", "blind_status", "runtime"}}
        report["scientific_results_sha256"] = _digest(scientific)
        report["report_sha256"] = _digest(report)
        _write_json(root / "outcome-report.json", report, immutable=True)
        v16._record_phase(root, "score", started, "COMPLETED", details={"scientific_outcome": outcome, "scientific_results_sha256": report["scientific_results_sha256"]})
        return {
            "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "protocol_slice": PROTOCOL_SLICE,
            "phase": "score", "status": "COMPLETED", "scientific_outcome": outcome,
            "scientific_results_sha256": report["scientific_results_sha256"], "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED", "reproducibility_status": "NOT_YET_COMPARED",
            "behavior_and_mechanism_diverged": report["behavior_and_mechanism_diverged"],
            "claim_ceiling": CLAIM_CEILING,
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {"protocol_id": PROTOCOL_ID, "phase": "score", "status": "INCONCLUSIVE", "error_type": type(error).__name__, "error": str(error), "packet_sha256": verify_packet()["packet_sha256"], "timestamp_epoch": time.time()}, immutable=True)
        v16._record_phase(root, "score", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def preflight() -> dict[str, Any]:
    packet = verify_packet()
    tree = _tree_receipt()
    _runtime_receipt()
    model, _tokenizer = v16._load_model()
    layers = v16._layers(model)
    if len(layers) != 12 or layers[TARGET_LAYER].fc1.in_features != 768 or layers[TARGET_LAYER].fc2.out_features != 768:
        raise ValueError("V17 target layer dimensions differ from the frozen contract")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "target_identity": {"model_id": v16.TARGET_MODEL_ID, "revision": v16.TARGET_REVISION, **tree},
        "target_layer": TARGET_LAYER,
        "architecture": "OPT/ReLU Linear(768,3072)-ReLU-Linear(3072,768)",
        "runtime": _runtime_receipt(),
        "wall_cap_seconds": MAX_WALL_SECONDS,
        "rss_cap_bytes": MAX_RSS_BYTES,
    }


def _files(root: Path, prefix: str) -> list[str]:
    path = root / prefix
    return [item.relative_to(root).as_posix() for item in sorted(path.rglob("*")) if item.is_file()] if path.exists() else []


def _file_group(root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V17 replication artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path)
    replica = _ensure_root(replica_path)
    primary_archive = _external_packet_archive(primary)
    replica_archive = _external_packet_archive(replica)
    if primary_archive["archive_sha256"] != replica_archive["archive_sha256"]:
        raise ValueError("V17 primary and replica packet archives differ")
    left_report = _read_json(primary / "outcome-report.json")
    right_report = _read_json(replica / "outcome-report.json")
    groups_to_paths = {
        "fixtures": _files(primary, "fixtures"),
        "splits": _files(primary, "splits"),
        "captures": _files(primary, "adaptation"),
        "candidates": _files(primary, "candidates") + ["candidate-freeze.json"],
        "challenge_entropy": _files(primary, "challenge-entropy") + ["challenge-entropy-manifest.json"],
        "predictions": _files(primary, "predictions") + ["prediction-manifest.json"],
    }
    groups = {}
    for name, paths in groups_to_paths.items():
        left = _file_group(primary, paths)
        right = _file_group(replica, paths)
        groups[name] = {"primary_count": len(left), "replica_count": len(right), "primary_sha256": _digest(left), "replica_sha256": _digest(right), "byte_identical": left == right}
    results_match = left_report["scientific_results_sha256"] == right_report["scientific_results_sha256"]
    left_runtime = _read_json(primary / "runtime.json")
    right_runtime = _read_json(replica / "runtime.json")
    pin_fields = ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "deterministic_algorithms", "torch_num_threads")
    pins_match = all(left_runtime.get(field) == right_runtime.get(field) for field in pin_fields)
    separate = left_runtime.get("python_executable") != right_runtime.get("python_executable")
    same_host = left_runtime.get("host_id_sha256") == right_runtime.get("host_id_sha256")
    same_owner = left_runtime.get("uid") == right_runtime.get("uid")
    identical = all(row["byte_identical"] for row in groups.values())
    reproduced = identical and results_match and pins_match and separate and same_host and same_owner
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": verify_packet()["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "scientific_outcome": left_report["scientific_outcome"] if results_match else "INCONCLUSIVE",
        "scientific_results_match": results_match,
        "scientific_results_sha256": [left_report["scientific_results_sha256"], right_report["scientific_results_sha256"]],
        "artifact_groups": groups,
        "runtime_pins_match": pins_match,
        "separate_python_runtimes": separate,
        "same_host_and_owner": same_host and same_owner,
        "archive_sha256": [primary_archive["archive_sha256"], replica_archive["archive_sha256"]],
        "claim_ceiling": CLAIM_CEILING,
        "custody": "prompts, captures, causal targets, candidates, entropy, predictions, and row-level results remain in external owner custody",
        "next_blocked_authority_gate": "blind external replication and authorized independent review",
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "replication-report.json", report, immutable=True)
    left_report["reproducibility_status"] = report["assurance_level"]
    left_report["replication_report_sha256"] = report["report_sha256"]
    left_report.pop("report_sha256", None)
    left_report["report_sha256"] = _digest(left_report)
    _write_json(primary / "outcome-report.json", left_report)
    return report


def replicate(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path)
    replica = _custody_root(replica_path)
    if replica.exists():
        raise FileExistsError("V17 replica root exists; preserve it and choose a fresh root")
    replica.mkdir(parents=True, exist_ok=False)
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(primary / "frozen-packet-manifest.json", replica / "frozen-packet-manifest.json")
    archive = _external_packet_archive(replica)
    prepare(replica)
    predict(replica, entropy_from=primary)
    score(replica)
    comparison = compare_replication(primary, replica)
    return {"protocol_id": PROTOCOL_ID, "replication_status": comparison["assurance_level"], "scientific_outcome": comparison["scientific_outcome"], "archive_sha256": archive["archive_sha256"], "replication_report_sha256": comparison["report_sha256"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "preflight"):
        commands.add_parser(name)
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--custody", required=True)
    predict_command = commands.add_parser("predict")
    predict_command.add_argument("--custody", required=True)
    predict_command.add_argument("--entropy-from")
    score_command = commands.add_parser("score")
    score_command.add_argument("--custody", required=True)
    replicate_command = commands.add_parser("replicate")
    replicate_command.add_argument("--primary", required=True)
    replicate_command.add_argument("--replica", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "verify":
        packet = verify_packet()
        result = {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "freeze_sha256": packet["freeze_sha256"], "packet_file_count": len(packet["packet_files"])}
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "prepare":
        result = prepare(args.custody)
    elif args.command == "predict":
        result = predict(args.custody, entropy_from=args.entropy_from)
    elif args.command == "score":
        result = score(args.custody)
    elif args.command == "replicate":
        result = replicate(args.primary, args.replica)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
