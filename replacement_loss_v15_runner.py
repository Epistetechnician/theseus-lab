"""Custody-bound V15 adaptation-only capacity/optimization diagnostic.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1, and
global-experiment-assurance-lifecycle-v1. Synthetic prompts, activations,
candidate weights, and row-level metrics stay in external owner custody.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import replacement_loss_v13_runner as v13
import replacement_loss_v14_runner as v14
import replacement_loss_v15_fixtures as fixtures
import replacement_loss_v3 as v3_core


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v15-freeze.json"
CLAIM_CEILING = "LOCAL_ADAPTATION_DIAGNOSTIC_OF_RANK_AND_OPTIMIZATION_FOR_ONE_REUSED_OPT_CHECKPOINT"
SOURCE_V14_PACKET_SHA256 = "3e6db084afbd7d9cfd2d072214d916886d06a85b9ff47471dd1452257c51485f"
SOURCE_V14_SELECTION_SHA256 = "020aa9b9f74d897d38d960807c4db91af5fe4fb17a1da1fd61afb4a02c7f970f"
SOURCE_V14_PRIMARY = Path(
    "/Volumes/PrimaryED/ResearchArtifacts/"
    "theseus-lab-sandbox-v1-replacement-loss-v14-paired-primary"
)
ARTIFACT_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15-capacity-optimization-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15-capacity-optimization-replica"
TARGET_LAYERS = (0, 7)
RANKS = (512, 768)
CHECKPOINT_STEPS = (200, 800)
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MAX_TOTAL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
MAX_PARAMETERS = 200_000_000
MAX_WEIGHT_BYTES = 800 * 1024**2
SIGNAL_MINIMUM_RELATIVE_REDUCTION = 0.10
SIGNAL_MINIMUM_CELLS = 5

_REPO_ROOT = Path(__file__).resolve().parent
_V14_FREEZE = json.loads((_REPO_ROOT / "docs/replacement-loss-v14-freeze.json").read_text(encoding="utf-8"))
_SOURCE_V14_DEPENDENCIES = tuple(row["path"] for row in _V14_FREEZE["packet_files"])
PACKET_FILES = tuple(dict.fromkeys((
    "AGENTS.md",
    "docs/replacement-loss-v15-capacity-optimization-diagnostic.md",
    "replacement_loss_v15_fixtures.py",
    "replacement_loss_v15_runner.py",
    "requirements-v15.lock.txt",
    "docs/replacement-loss-v14-freeze.json",
    "docs/replacement-loss-v14-results.md",
    "replacement_loss_v14_fixtures.py",
    "replacement_loss_v14_runner.py",
    "requirements-v14.lock.txt",
    *_SOURCE_V14_DEPENDENCIES,
)))


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable V15 artifact already exists: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _packet_rows() -> list[dict[str, str]]:
    rows = []
    for relative in PACKET_FILES:
        path = _REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V15 packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def freeze_packet() -> dict[str, Any]:
    destination = _REPO_ROOT / FREEZE_FILE
    if destination.exists():
        raise FileExistsError("V15 packet freeze is immutable")
    source_v14 = v14.verify_packet()
    if source_v14["packet_sha256"] != SOURCE_V14_PACKET_SHA256:
        raise ValueError("V15 source V14 packet digest differs from the predeclared diagnostic")
    rows = _packet_rows()
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_V15_FIXTURE_GENERATION_AND_TRAINING",
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "source_v14_packet_sha256": SOURCE_V14_PACKET_SHA256,
        "source_v14_freeze_sha256": _sha_file(_REPO_ROOT / "docs/replacement-loss-v14-freeze.json"),
        "source_v14_results_sha256": _sha_file(_REPO_ROOT / "docs/replacement-loss-v14-results.md"),
        "source_v14_selection_sha256": SOURCE_V14_SELECTION_SHA256,
        "source_v14_scientific_outcome": "FAIL",
        "source_v14_reproduction_assurance": "REPRODUCED_LOCAL",
        "fixture_generation_started": False,
        "candidate_training_started": False,
        "challenge_generation_started": False,
        "visible_fixture_seed": fixtures.VISIBLE_SEED,
        "adaptation_seeds": list(fixtures.SEEDS),
        "selected_layers": list(TARGET_LAYERS),
        "ranks": list(RANKS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "validation_per_stratum": fixtures.VALIDATION_PER_STRATUM,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_norm_clip": GRADIENT_CLIP,
        "optimizer": "AdamW",
        "objective": "normalized activation MSE plus normalized next-logit intervention-effect MSE, unit weights",
        "validation_signal_thresholds": {
            "minimum_relative_reduction": SIGNAL_MINIMUM_RELATIVE_REDUCTION,
            "minimum_positive_seed_layer_cells": SIGNAL_MINIMUM_CELLS,
            "cell_count": len(fixtures.SEEDS) * len(TARGET_LAYERS),
            "inferential_claim": False,
        },
        "resource_limits": {
            "execution": "local_cpu_only",
            "max_total_seconds": MAX_TOTAL_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "max_base_model_parameters": MAX_PARAMETERS,
            "max_checkpoint_bytes": MAX_WEIGHT_BYTES,
            "paid_or_remote_compute": False,
            "provider_calls": False,
            "gman": False,
        },
        "claim_ceiling": CLAIM_CEILING,
        "nonclaims": [
            "no held-out challenge or continuation behavior evaluation",
            "no learned activation-rescue or replacement-retention conclusion",
            "no fresh-checkpoint or architecture-general evidence",
            "no independent scientific, alignment, consciousness, or production claim",
        ],
    }
    material["freeze_sha256"] = _digest(material)
    _write_json(destination, material, immutable=True)
    return material


def verify_packet() -> dict[str, Any]:
    payload = _json(_REPO_ROOT / FREEZE_FILE)
    source_v14 = v14.verify_packet()
    material = {key: value for key, value in payload.items() if key != "freeze_sha256"}
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_V15_FIXTURE_GENERATION_AND_TRAINING"
        or payload.get("packet_files") != _packet_rows()
        or payload.get("packet_sha256") != _digest(_packet_rows())
        or payload.get("source_v14_packet_sha256") != source_v14["packet_sha256"]
        or source_v14["packet_sha256"] != SOURCE_V14_PACKET_SHA256
        or payload.get("source_v14_freeze_sha256") != _sha_file(_REPO_ROOT / "docs/replacement-loss-v14-freeze.json")
        or payload.get("source_v14_results_sha256") != _sha_file(_REPO_ROOT / "docs/replacement-loss-v14-results.md")
        or payload.get("source_v14_selection_sha256") != SOURCE_V14_SELECTION_SHA256
        or payload.get("source_v14_scientific_outcome") != "FAIL"
        or payload.get("fixture_generation_started") is not False
        or payload.get("candidate_training_started") is not False
        or payload.get("challenge_generation_started") is not False
        or payload.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V15 frozen packet verification failed")
    return payload


def _custody_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.parent != ARTIFACT_ROOT.resolve() or root.name not in {PRIMARY_NAME, REPLICA_NAME}:
        raise ValueError("V15 custody must be one of the two predeclared PrimaryED roots")
    return root


def _source_selection() -> dict[str, Any]:
    selection = _json(SOURCE_V14_PRIMARY / "checkpoint-selection.json")
    material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("selection_sha256") != SOURCE_V14_SELECTION_SHA256
        or _digest(material) != SOURCE_V14_SELECTION_SHA256
        or selection.get("packet_sha256") != SOURCE_V14_PACKET_SHA256
        or selection.get("model_id") != "grushaprasad/opt125m-rootp-surprisal-curriculum"
        or selection.get("revision") != "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
        or selection.get("selection_performed") is not False
    ):
        raise ValueError("V15 source target selection differs from the frozen V14 OPT identity")
    model_path = Path(selection["local_path"]).resolve()
    if _sha_file(model_path / "config.json") != selection["config_sha256"]:
        raise ValueError("V15 source checkpoint config differs from its bound V14 selection")
    if _sha_file(model_path / "README.md") != selection["model_card_sha256"]:
        raise ValueError("V15 source checkpoint model card differs from its bound V14 selection")
    if _digest(selection["weight_files"]) != selection["weights_sha256"]:
        raise ValueError("V15 source checkpoint weight inventory digest is invalid")
    if sum(int(row["bytes"]) for row in selection["weight_files"]) > MAX_WEIGHT_BYTES:
        raise ValueError("V15 source checkpoint exceeds the frozen byte limit")
    for row in selection["weight_files"]:
        path = model_path / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha_file(path) != row["sha256"]:
            raise ValueError("V15 source checkpoint weight bytes differ from the bound V14 selection")
    return selection


def preflight() -> dict[str, Any]:
    packet = verify_packet()
    selection = _source_selection()
    runtime = _runtime()
    if runtime["packages"].get("torch") != "2.6.0" or runtime["packages"].get("transformers") != "4.57.3":
        raise ValueError("V15 runtime packages do not match the predeclared lock")
    if not ARTIFACT_ROOT.is_dir():
        raise FileNotFoundError("V15 external custody volume is not mounted")
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "source_v14_outcome": "FAIL",
        "target_identity": {
            "model_id": selection["model_id"],
            "revision": selection["revision"],
            "tree_sha256": selection["source_artifact_tree_sha256"],
            "selection_sha256": selection["selection_sha256"],
            "weight_bytes": selection["total_weight_bytes"],
        },
        "runtime": runtime,
        "custody_volume_available": True,
        "fixture_generation_started": False,
        "candidate_training_started": False,
        "challenge_generation_started": False,
        "paid_or_remote_compute": False,
    }


def _runtime() -> dict[str, Any]:
    package_names = ("huggingface-hub", "numpy", "safetensors", "tokenizers", "torch", "transformers")
    packages = {name: importlib.metadata.version(name) for name in package_names}
    runtime = v13._ORIGINALS["verify_runtime"]()
    runtime.update({
        "packages": packages,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "host_id_sha256": hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest(),
        "dependency_lock_sha256": _sha_file(_REPO_ROOT / "requirements-v15.lock.txt"),
        "packet_sha256": verify_packet()["packet_sha256"],
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })
    return runtime


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_path)
    packet = verify_packet()
    if root.exists():
        raise FileExistsError("V15 custody root already exists; preserve it and select a new authorized root")
    root.mkdir(mode=0o700, parents=True)
    rows = []
    try:
        for relative in PACKET_FILES:
            source = _REPO_ROOT / relative
            destination = root / "frozen-packet" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            rows.append({"path": relative, "bytes": destination.stat().st_size, "sha256": _sha_file(destination)})
        freeze_source = _REPO_ROOT / FREEZE_FILE
        shutil.copyfile(freeze_source, root / "frozen-packet" / FREEZE_FILE)
        freeze_sha = _sha_file(root / "frozen-packet" / FREEZE_FILE)
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "freeze_file_sha256": freeze_sha,
            "files": rows,
        }
        manifest["archive_sha256"] = _digest(manifest)
        _write_json(root / "archive-manifest.json", manifest, immutable=True)
        return {"archive_sha256": manifest["archive_sha256"], "packet_sha256": packet["packet_sha256"], "file_count": len(rows)}
    except BaseException:
        _write_json(root / "archive-stop.json", {
            "protocol_id": PROTOCOL_ID,
            "status": "INCONCLUSIVE",
            "phase": "archive",
            "error_type": "ArchiveError",
        }, immutable=True)
        raise


def _verify_archive(root: Path) -> dict[str, Any]:
    manifest = _json(root / "archive-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    if manifest.get("archive_sha256") != _digest(material) or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("V15 frozen packet archive digest is invalid")
    actual = []
    for row in manifest["files"]:
        path = root / "frozen-packet" / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or _sha_file(path) != row["sha256"]:
            raise ValueError("V15 archived packet file differs from the frozen manifest")
        actual.append({"path": row["path"], "sha256": row["sha256"]})
    if actual != _packet_rows():
        raise ValueError("V15 archived packet membership differs from the frozen packet")
    freeze_path = root / "frozen-packet" / FREEZE_FILE
    if not freeze_path.is_file() or _sha_file(freeze_path) != manifest["freeze_file_sha256"]:
        raise ValueError("V15 archived freeze record is invalid")
    return manifest


def bind_target(custody_path: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_path)
    _verify_archive(root)
    destination = root / "checkpoint-selection.json"
    if destination.exists():
        raise FileExistsError("V15 target selection is immutable")
    selection = _source_selection()
    _write_json(destination, selection, immutable=True)
    return {
        "selection_sha256": selection["selection_sha256"],
        "model_id": selection["model_id"],
        "revision": selection["revision"],
        "tree_sha256": selection["source_artifact_tree_sha256"],
        "weight_bytes": selection["total_weight_bytes"],
    }


class DiagnosticStack(torch.nn.Module):
    """Factorized GELU projection with a frozen diagnostic rank."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.first = torch.nn.Linear(in_features, rank, bias=True, dtype=torch.float32)
        self.activation = torch.nn.GELU()
        self.second = torch.nn.Linear(rank, out_features, bias=True, dtype=torch.float32)
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self.rank = int(rank)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def bias(self) -> torch.Tensor:
        return self.second.bias

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.activation(self.first(value)))


def _candidate_seed(seed_index: int, layer: int, rank: int) -> int:
    payload = f"{PROTOCOL_ID}\0{seed_index}\0{layer}\0rank-{rank}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def _split_rows(rows: list[dict[str, Any]], seed_index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    material: dict[str, Any] = {"seed_index": seed_index, "strata": {}}
    for stratum in fixtures.STRATA:
        group = [row for row in rows if row["stratum"] == stratum]
        ordered = sorted(group, key=lambda row: hashlib.sha256(
            f"{PROTOCOL_ID}\0split\0{seed_index}\0{stratum}\0{row['example_id']}".encode("utf-8")
        ).hexdigest())
        count = fixtures.VALIDATION_PER_STRATUM
        if len(ordered) != fixtures.ADAPTATION_PER_STRATUM or count >= len(ordered):
            raise ValueError("V15 adaptation fixture does not match the frozen stratified split")
        valid = ordered[:count]
        fitted = ordered[count:]
        validation.extend(valid)
        train.extend(fitted)
        material["strata"][stratum] = {
            "fit_count": len(fitted),
            "validation_count": len(valid),
            "fit_ids": [row["example_id"] for row in fitted],
            "validation_ids": [row["example_id"] for row in valid],
        }
    if set(row["example_id"] for row in train) & set(row["example_id"] for row in validation):
        raise ValueError("V15 fit and validation rows overlap")
    material["fit_ids_sha256"] = _digest(sorted(row["example_id"] for row in train))
    material["validation_ids_sha256"] = _digest(sorted(row["example_id"] for row in validation))
    return train, validation, material


def _collate(engine: Any, rows: list[dict[str, Any]], pad_token_id: int):
    return engine._collate_training(rows, pad_token_id)


def _evaluate_rows(
    engine: Any,
    model: torch.nn.Module,
    layer: int,
    rows: list[dict[str, Any]],
    pad_token_id: int,
    activation_scale: float,
) -> dict[str, float]:
    activation_values: list[float] = []
    causal_values: list[float] = []
    candidate = v3_core._projection(model, "opt_relu", layer)
    candidate.eval()
    with torch.no_grad():
        for start in range(0, len(rows), TRAIN_BATCH_SIZE):
            batch_rows = rows[start:start + TRAIN_BATCH_SIZE]
            inputs, positions, deltas, targets, teacher_effects = _collate(engine, batch_rows, pad_token_id)
            base_logits, predicted_y = engine._candidate_forward(model, layer, inputs, positions, None)
            changed_logits, _changed_y = engine._candidate_forward(model, layer, inputs, positions, deltas)
            predicted_effect = changed_logits - base_logits
            activation_values.extend(((predicted_y - targets).square().mean(dim=1) / activation_scale).cpu().tolist())
            effect_scales = teacher_effects.square().mean(dim=1).sqrt().clamp_min(1e-5)
            causal_values.extend((((predicted_effect - teacher_effects) / effect_scales[:, None]).square().mean(dim=1)).cpu().tolist())
    if not activation_values:
        raise ValueError("V15 evaluation partition is empty")
    activation_loss = float(np.mean(activation_values))
    causal_loss = float(np.mean(causal_values))
    return {
        "normalized_activation_mse": activation_loss,
        "normalized_causal_mse": causal_loss,
        "combined_objective": activation_loss + causal_loss,
        "row_count": len(rows),
    }


def _train_rank(
    engine: Any,
    model: torch.nn.Module,
    layer: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
    fit_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    pad_token_id: int,
    seed: int,
    rank: int,
    seed_index: int,
    custody: Path,
    adaptation_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (in_features, out_features, use_bias) != (3072, 768, True):
        raise ValueError("V15 target projection dimensions differ from the frozen OPT fc2 boundary")
    if len(fit_rows) < TRAIN_BATCH_SIZE or not validation_rows:
        raise ValueError("V15 fit and validation partitions are undersized")
    source_module = v3_core._projection(model, "opt_relu", layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        candidate = DiagnosticStack(in_features, out_features, rank).to(device="cpu", dtype=torch.float32)
    init_sha = v3_core._module_digest(candidate)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    v3_core._set_projection(model, "opt_relu", layer, candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    batch_seed = int(seed_index * 1_000_003 + layer * 1009 + 0x15A11)
    order = np.random.default_rng(batch_seed).permutation(len(fit_rows))
    batch_order = []
    target_values = torch.as_tensor(np.stack([row["target_y"] for row in fit_rows]), dtype=torch.float32)
    activation_scale = float(target_values.square().mean().clamp_min(1e-6).item())
    metric_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    def checkpoint(step: int) -> None:
        candidate.eval()
        train_metrics = _evaluate_rows(engine, model, layer, fit_rows, pad_token_id, activation_scale)
        validation_metrics = _evaluate_rows(engine, model, layer, validation_rows, pad_token_id, activation_scale)
        metric_rows.append({
            "seed_index": seed_index,
            "seed": fixtures.SEEDS[seed_index],
            "layer": layer,
            "rank": rank,
            "updates": step,
            "fit": train_metrics,
            "validation": validation_metrics,
        })
        if step:
            name = f"rank-{rank}-step-{step}"
            relative = f"candidates/seed-{seed_index}/layer-{layer}/{name}.safetensors"
            path = custody / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            state = {key: tensor.detach().cpu().contiguous() for key, tensor in candidate.state_dict().items()}
            save_file(state, str(path))
            candidate_rows.append({
                "seed_index": seed_index,
                "seed": fixtures.SEEDS[seed_index],
                "layer": layer,
                "rank": rank,
                "updates": step,
                "architecture": "Linear-GELU-Linear",
                "in_features": in_features,
                "out_features": out_features,
                "use_bias": True,
                "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
                "path": relative,
                "file_sha256": _sha_file(path),
                "state_sha256": v3_core._module_digest(candidate),
                "initialization_sha256": init_sha,
                "batch_order_sha256": hashlib.sha256(np.asarray(batch_order, dtype="<i8").tobytes()).hexdigest(),
                "adaptation_sha256": adaptation_sha256,
            })
        candidate.train()

    try:
        model.eval()
        candidate.train()
        checkpoint(0)
        for update in range(1, max(CHECKPOINT_STEPS) + 1):
            engine._check_budget()
            start = ((update - 1) * TRAIN_BATCH_SIZE) % len(order)
            indexes = order[(start + np.arange(TRAIN_BATCH_SIZE)) % len(order)]
            batch_order.extend(int(index) for index in indexes)
            batch_rows = [fit_rows[int(index)] for index in indexes]
            inputs, positions, deltas, targets, teacher_effects = _collate(engine, batch_rows, pad_token_id)
            base_logits, predicted_y = engine._candidate_forward(model, layer, inputs, positions, None)
            changed_logits, _changed_y = engine._candidate_forward(model, layer, inputs, positions, deltas)
            predicted_effect = changed_logits - base_logits
            effect_scales = teacher_effects.square().mean(dim=1).sqrt().clamp_min(1e-5)
            activation_loss = (predicted_y - targets).square().mean() / activation_scale
            causal_loss = ((predicted_effect - teacher_effects) / effect_scales[:, None]).square().mean()
            loss = activation_loss + causal_loss
            if not torch.isfinite(loss):
                raise ValueError("V15 diagnostic fit produced a non-finite objective")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP)
            optimizer.step()
            if update in CHECKPOINT_STEPS:
                checkpoint(update)
        return metric_rows, candidate_rows
    finally:
        v3_core._set_projection(model, "opt_relu", layer, source_module)
        gc.collect()


def _host_id_sha256() -> str:
    return hashlib.sha256(socket.gethostname().encode("utf-8")).hexdigest()


def diagnose(custody_path: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_path)
    packet = verify_packet()
    archive = _verify_archive(root)
    selection = _json(root / "checkpoint-selection.json")
    verified_selection = _source_selection()
    if selection.get("selection_sha256") != SOURCE_V14_SELECTION_SHA256 or selection != verified_selection:
        raise ValueError("V15 custody target selection differs from the frozen V14 target")
    if (root / "phase-ledger.json").exists() or (root / "run-stop.json").exists():
        raise FileExistsError("V15 diagnostic custody is immutable once execution starts")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    runtime = _runtime()
    engine = v14.configure_engine()
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.PROTOCOL_SLICE = PROTOCOL_SLICE
    engine.fixtures = fixtures
    engine.FAMILY = "opt_relu"
    engine.TRAIN_BATCH_SIZE = TRAIN_BATCH_SIZE
    engine.MAX_TOTAL_SECONDS = MAX_TOTAL_SECONDS
    engine.MAX_RSS_BYTES = MAX_RSS_BYTES
    engine._verify_runtime = lambda: runtime
    engine._begin(root, "diagnostic")
    started = time.monotonic()
    _write_json(root / "phase-ledger.json", {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "archive_sha256": archive["archive_sha256"],
        "phases": [{"phase": "diagnostic", "status": "RUNNING", "runtime": runtime}],
        "total_elapsed_seconds": 0.0,
    }, immutable=True)
    try:
        model_path = Path(selection["local_path"]).resolve()
        checkpoint = v3_core.Checkpoint(
            family="opt_relu",
            model_id=selection["model_id"],
            revision=selection["revision"],
            local_path=model_path,
            selected_after_protocol_freeze=False,
        )
        model, tokenizer = v3_core._load_local_model(checkpoint)
        model.eval()
        model_parameters = sum(parameter.numel() for parameter in model.parameters())
        if model_parameters > MAX_PARAMETERS:
            raise ValueError("V15 target checkpoint exceeds the frozen parameter limit")
        visible_payload = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        _write_json(root / "fixtures" / "visible.json", visible_payload, immutable=True)
        visible_rows = engine._token_rows(visible_payload["records"], tokenizer, continuation=False)
        visible_inputs = []
        with torch.no_grad():
            for _group, inputs in v3_core._batches(visible_rows, tokenizer.pad_token_id, batch_size=8):
                values = v3_core._capture_all_layer_inputs(model, "opt_relu", inputs)
                visible_inputs.append(values)
        visible_rms = {}
        for layer in TARGET_LAYERS:
            combined = torch.cat([batch[layer].float().cpu() for batch in visible_inputs])
            visible_rms[str(layer)] = float(combined.square().mean().sqrt().item())
            if not np.isfinite(visible_rms[str(layer)]) or visible_rms[str(layer)] <= 0:
                raise ValueError("V15 visible-only activation scale is invalid")

        adaptation_payloads = []
        for seed_index in range(len(fixtures.SEEDS)):
            payload = fixtures.fixture_payload(
                fixtures.generate_adaptation_fixture(seed_index),
                fixture_kind=f"adaptation_seed_{seed_index}",
            )
            fixtures.validate_partition_disjointness(visible_payload, *adaptation_payloads, payload)
            adaptation_payloads.append(payload)
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)

        all_metric_rows: list[dict[str, Any]] = []
        all_candidate_rows: list[dict[str, Any]] = []
        split_rows: list[dict[str, Any]] = []
        adaptation_rows: list[dict[str, Any]] = []
        fixture_rows = [{
            "path": "fixtures/visible.json",
            "sha256": _sha_file(root / "fixtures" / "visible.json"),
            "example_set_sha256": visible_payload["example_set_sha256"],
            "count": visible_payload["count"],
        }]
        fixture_rows.extend({
            "path": f"fixtures/adaptation-seed-{seed_index}.json",
            "sha256": _sha_file(root / "fixtures" / f"adaptation-seed-{seed_index}.json"),
            "example_set_sha256": payload["example_set_sha256"],
            "count": payload["count"],
        } for seed_index, payload in enumerate(adaptation_payloads))
        for seed_index, payload in enumerate(adaptation_payloads):
            fit_prompts, validation_prompts, split_material = _split_rows(payload["records"], seed_index)
            fit_ids = [row["example_id"] for row in fit_prompts]
            validation_ids = [row["example_id"] for row in validation_prompts]
            if set(fit_ids) & set(validation_ids):
                raise ValueError("V15 pre-capture fit and validation prompts overlap")
            split_rows.append(split_material)
            _write_json(root / "splits" / f"seed-{seed_index}.json", split_material, immutable=True)
            for layer in TARGET_LAYERS:
                _x_fit, _y_fit, fit_rows = engine._capture_adaptation_layer(
                    model, tokenizer, layer, fit_prompts, seed_index
                )
                _x_validation, _y_validation, validation_rows = engine._capture_adaptation_layer(
                    model, tokenizer, layer, validation_prompts, seed_index
                )
                captured_rows = fit_rows + validation_rows
                if [row["example_id"] for row in fit_rows] != fit_ids or [row["example_id"] for row in validation_rows] != validation_ids:
                    raise ValueError("V15 adaptation capture changed the frozen pre-capture split order")
                capture_path = root / "adaptation" / f"seed-{seed_index}" / f"layer-{layer}.npz"
                v3_core._save_npz_deterministic(
                    capture_path,
                    targets=np.stack([row["target_y"] for row in captured_rows]),
                    deltas=np.stack([row["delta"] for row in captured_rows]),
                    teacher_logit_effects=np.stack([row["teacher_effect"] for row in captured_rows]),
                    example_ids=np.asarray([row["example_id"] for row in captured_rows], dtype="U24"),
                    strata=np.asarray([row["stratum"] for row in captured_rows], dtype="U24"),
                    fit_ids=np.asarray(sorted(row["example_id"] for row in fit_rows), dtype="U24"),
                    validation_ids=np.asarray(sorted(row["example_id"] for row in validation_rows), dtype="U24"),
                )
                adaptation_sha = _sha_file(capture_path)
                adaptation_rows.append({
                    "seed_index": seed_index,
                    "layer": layer,
                    "path": capture_path.relative_to(root).as_posix(),
                    "sha256": adaptation_sha,
                    "row_count": len(captured_rows),
                    "fit_count": len(fit_rows),
                    "validation_count": len(validation_rows),
                })
                in_features, out_features, use_bias = v3_core._module_dimensions(v3_core._projection(model, "opt_relu", layer))
                for rank in RANKS:
                    seed = _candidate_seed(seed_index, layer, rank)
                    metrics, candidates = _train_rank(
                        engine, model, layer, in_features, out_features, use_bias,
                        fit_rows, validation_rows, int(tokenizer.pad_token_id), seed,
                        rank, seed_index, root, adaptation_sha,
                    )
                    all_metric_rows.extend(metrics)
                    all_candidate_rows.extend(candidates)
                    gc.collect()
                    engine._check_budget()
        del model, tokenizer
        gc.collect()
        expected_candidates = len(fixtures.SEEDS) * len(TARGET_LAYERS) * len(RANKS) * len(CHECKPOINT_STEPS)
        if len(all_candidate_rows) != expected_candidates:
            raise ValueError("V15 candidate set does not cover every seed/layer/rank/update cell")
        candidate_material = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "archive_sha256": archive["archive_sha256"],
            "target_selection_sha256": selection["selection_sha256"],
            "target_tree_sha256": selection["source_artifact_tree_sha256"],
            "visible_fixture_sha256": visible_payload["example_set_sha256"],
            "visible_activation_rms": visible_rms,
            "fixture_artifacts": fixture_rows,
            "adaptation_artifacts": adaptation_rows,
            "split_artifacts": [{
                "seed_index": row["seed_index"],
                "path": f"splits/seed-{row['seed_index']}.json",
                "sha256": _sha_file(root / "splits" / f"seed-{row['seed_index']}.json"),
                "fit_ids_sha256": row["fit_ids_sha256"],
                "validation_ids_sha256": row["validation_ids_sha256"],
            } for row in split_rows],
            "candidate_count": expected_candidates,
            "ranks": list(RANKS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "candidates": all_candidate_rows,
            "training_metrics": all_metric_rows,
        }
        candidate_material["candidate_set_sha256"] = _digest(candidate_material)
        _write_json(root / "candidate-freeze.json", candidate_material, immutable=True)
        diagnostic = _classify(all_metric_rows)
        diagnostic_payload = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidate_material["candidate_set_sha256"],
            "target_selection_sha256": selection["selection_sha256"],
            "claim_ceiling": CLAIM_CEILING,
            "diagnostic_only": True,
            "no_challenge_generation": True,
            "no_heldout_model_behavior_evaluation": True,
            "classification": diagnostic,
            "cell_metrics": all_metric_rows,
            "teacher_target_integrity_reference": {
                "activation_target_identity_error": 0.0,
                "causal_effect_target_identity_error": 0.0,
                "interpretation": "algebraic target identity only; not learned replacement evidence",
            },
            "data_and_artifact_custody": "synthetic prompts, captures, candidate tensors, split identifiers, and metric rows remain in external owner custody",
        }
        diagnostic_payload["diagnostic_results_sha256"] = _digest(diagnostic_payload)
        _write_json(root / "diagnostic-results.json", diagnostic_payload, immutable=True)
        elapsed = time.monotonic() - started
        peak_rss = engine._peak_rss_bytes()
        phase = {
            "phase": "diagnostic",
            "status": "COMPLETED",
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss,
            "runtime": runtime,
            "candidate_count": expected_candidates,
            "fixture_seed_count": len(fixtures.SEEDS),
            "selected_layers": list(TARGET_LAYERS),
            "candidate_training_started": True,
            "challenge_generation_started": False,
        }
        _write_json(root / "phase-ledger.json", {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "archive_sha256": archive["archive_sha256"],
            "phases": [phase],
            "total_elapsed_seconds": elapsed,
        })
        report = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "source_revision": "202f459d6d89df9a038d78d58e0af825d287fcd8",
            "packet_sha256": packet["packet_sha256"],
            "archive_sha256": archive["archive_sha256"],
            "target_identity": {
                "model_id": selection["model_id"],
                "revision": selection["revision"],
                "tree_sha256": selection["source_artifact_tree_sha256"],
                "selection_sha256": selection["selection_sha256"],
            },
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "blind_status": "NOT_BLIND_SAME_OWNER_DIAGNOSTIC",
            "reproducibility_status": "NOT_YET_COMPARED",
            "diagnostic_classification": diagnostic,
            "diagnostic_results_sha256": diagnostic_payload["diagnostic_results_sha256"],
            "candidate_count": expected_candidates,
            "runtime": runtime,
            "phase_ledger": phase,
            "data_and_artifact_custody": diagnostic_payload["data_and_artifact_custody"],
            "known_limitations": [
                "one reused outcome-aware OPT/ReLU checkpoint and fixed layers 0 and 7",
                "three synthetic fixture seeds and six nested seed/layer cells",
                "adaptation validation is not a fresh causal challenge or behavior holdout",
                "the diagnostic cannot establish learned activation rescue or replacement retention",
                "same-owner local work is not independent scientific acceptance",
            ],
            "claim_ceiling": CLAIM_CEILING,
            "next_blocked_authority_gate": "fresh frozen held-out causal and behavior evaluation is required before any replacement-retention claim; independent science additionally requires blind replication and an authorized reviewer",
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
        }
        report["report_sha256"] = _digest(report)
        _write_json(root / "diagnostic-report.json", report, immutable=True)
        return report
    except BaseException as error:
        elapsed = time.monotonic() - started
        phase = {
            "phase": "diagnostic",
            "status": "STOPPED",
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": engine._peak_rss_bytes(),
            "runtime": runtime,
        }
        ledger_path = root / "phase-ledger.json"
        previous = _json(ledger_path) if ledger_path.exists() else {"phases": []}
        previous["phases"] = [phase]
        previous["total_elapsed_seconds"] = elapsed
        _write_json(ledger_path, previous)
        _write_json(root / "run-stop.json", {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "phase": "diagnostic",
            "status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error_message": str(error)[:500],
            "packet_sha256": packet["packet_sha256"],
        }, immutable=True)
        raise


def _classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = {(int(r["seed_index"]), int(r["layer"]), int(r["rank"]), int(r["updates"])): r for r in rows}
    cell_count = len(fixtures.SEEDS) * len(TARGET_LAYERS)
    if len(lookup) != cell_count * len(RANKS) * (len(CHECKPOINT_STEPS) + 1):
        raise ValueError("V15 metric table has missing or duplicate factorial cells")
    optimization: dict[str, Any] = {}
    capacity: dict[str, Any] = {}
    overfit: dict[str, Any] = {}
    for rank in RANKS:
        gains = []
        overfit_cells = []
        for seed_index in range(len(fixtures.SEEDS)):
            for layer in TARGET_LAYERS:
                early = lookup[(seed_index, layer, rank, CHECKPOINT_STEPS[0])]
                late = lookup[(seed_index, layer, rank, CHECKPOINT_STEPS[1])]
                early_val = float(early["validation"]["combined_objective"])
                late_val = float(late["validation"]["combined_objective"])
                early_fit = float(early["fit"]["combined_objective"])
                late_fit = float(late["fit"]["combined_objective"])
                gains.append((early_val - late_val) / max(abs(early_val), 1e-12))
                fit_gain = (early_fit - late_fit) / max(abs(early_fit), 1e-12)
                val_change = (late_val - early_val) / max(abs(early_val), 1e-12)
                overfit_cells.append(fit_gain >= SIGNAL_MINIMUM_RELATIVE_REDUCTION and val_change >= SIGNAL_MINIMUM_RELATIVE_REDUCTION)
        optimization[str(rank)] = {
            "mean_validation_relative_reduction": float(np.mean(gains)),
            "positive_cells": sum(value > 0 for value in gains),
            "cell_count": cell_count,
            "signal": float(np.mean(gains)) >= SIGNAL_MINIMUM_RELATIVE_REDUCTION and sum(value > 0 for value in gains) >= SIGNAL_MINIMUM_CELLS,
        }
        overfit[str(rank)] = {
            "cells_with_fit_down_and_validation_up_by_threshold": sum(overfit_cells),
            "cell_count": cell_count,
            "signal": sum(overfit_cells) >= SIGNAL_MINIMUM_CELLS,
        }
    for step in CHECKPOINT_STEPS:
        gains = []
        for seed_index in range(len(fixtures.SEEDS)):
            for layer in TARGET_LAYERS:
                smaller = float(lookup[(seed_index, layer, RANKS[0], step)]["validation"]["combined_objective"])
                larger = float(lookup[(seed_index, layer, RANKS[1], step)]["validation"]["combined_objective"])
                gains.append((smaller - larger) / max(abs(smaller), 1e-12))
        capacity[str(step)] = {
            "mean_validation_relative_reduction_rank_768_vs_512": float(np.mean(gains)),
            "positive_cells": sum(value > 0 for value in gains),
            "cell_count": cell_count,
            "signal": float(np.mean(gains)) >= SIGNAL_MINIMUM_RELATIVE_REDUCTION and sum(value > 0 for value in gains) >= SIGNAL_MINIMUM_CELLS,
        }
    optimization_signal = any(row["signal"] for row in optimization.values())
    capacity_signal = any(row["signal"] for row in capacity.values())
    overfit_signal = any(row["signal"] for row in overfit.values())
    signals = [name for name, active in (("OPTIMIZATION_SIGNAL", optimization_signal), ("CAPACITY_SIGNAL", capacity_signal), ("OVERFIT_SIGNAL", overfit_signal)) if active]
    if not signals:
        classification = "NO_CLEAR_SIGNAL"
    elif len(signals) == 1:
        classification = signals[0]
    else:
        classification = "MULTIPLE_SIGNALS"
    return {
        "classification": classification,
        "signals": signals,
        "optimization": optimization,
        "capacity": capacity,
        "overfit": overfit,
        "meaning": "descriptive diagnostic only; six nested cells do not support an inferential or replacement-retention claim",
    }


def _artifact_group(root: Path, name: str) -> list[dict[str, Any]]:
    base = root / name
    if not base.is_dir():
        raise FileNotFoundError(f"V15 artifact group is missing: {name}")
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(item for item in base.rglob("*") if item.is_file())
    ]


def _verify_run_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    candidate_freeze = _json(root / "candidate-freeze.json")
    candidate_material = {key: value for key, value in candidate_freeze.items() if key != "candidate_set_sha256"}
    if candidate_freeze.get("candidate_set_sha256") != _digest(candidate_material):
        raise ValueError("V15 candidate-set digest is invalid")
    if candidate_freeze.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("V15 candidates are bound to a different frozen packet")
    if (
        candidate_freeze.get("target_selection_sha256") != SOURCE_V14_SELECTION_SHA256
        or candidate_freeze.get("target_tree_sha256") != _source_selection()["source_artifact_tree_sha256"]
    ):
        raise ValueError("V15 candidate set is bound to a different target checkpoint")
    fixture_rows = candidate_freeze.get("fixture_artifacts", [])
    expected_fixture_paths = ["fixtures/visible.json"] + [
        f"fixtures/adaptation-seed-{seed_index}.json" for seed_index in range(len(fixtures.SEEDS))
    ]
    if [row.get("path") for row in fixture_rows] != expected_fixture_paths:
        raise ValueError("V15 fixture manifest does not cover the frozen fixture set")
    fixture_payloads = []
    fixture_ids_by_path: dict[str, set[str]] = {}
    for fixture_row in fixture_rows:
        fixture_path = root / fixture_row["path"]
        if not fixture_path.is_file() or _sha_file(fixture_path) != fixture_row.get("sha256"):
            raise ValueError("V15 fixture bytes differ from the candidate manifest")
        payload = _json(fixture_path)
        records = payload.get("records")
        if (
            payload.get("protocol_id") != PROTOCOL_ID
            or payload.get("state_slice") != STATE_SLICE
            or payload.get("protocol_slice") != PROTOCOL_SLICE
            or payload.get("count") != fixture_row.get("count")
            or payload.get("example_set_sha256") != fixture_row.get("example_set_sha256")
            or not isinstance(records, list)
            or len(records) != payload.get("count")
        ):
            raise ValueError("V15 fixture payload metadata differs from its frozen manifest")
        row_digests = [hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest() for row in records]
        if hashlib.sha256("\n".join(sorted(row_digests)).encode("ascii")).hexdigest() != payload["example_set_sha256"]:
            raise ValueError("V15 fixture record contents differ from their declared set digest")
        fixture_ids_by_path[fixture_row["path"]] = {str(row["example_id"]) for row in records}
        fixture_payloads.append(payload)
    fixtures.validate_partition_disjointness(*fixture_payloads)
    split_artifacts = candidate_freeze.get("split_artifacts", [])
    if len(split_artifacts) != len(fixtures.SEEDS):
        raise ValueError("V15 split manifest does not cover every adaptation seed")
    split_by_seed = {}
    for row in split_artifacts:
        seed_index = int(row["seed_index"])
        expected_path = f"splits/seed-{seed_index}.json"
        path = root / expected_path
        if row.get("path") != expected_path or not path.is_file() or _sha_file(path) != row.get("sha256"):
            raise ValueError("V15 split bytes differ from the candidate manifest")
        split = _json(path)
        if (
            split.get("seed_index") != seed_index
            or split.get("fit_ids_sha256") != row.get("fit_ids_sha256")
            or split.get("validation_ids_sha256") != row.get("validation_ids_sha256")
        ):
            raise ValueError("V15 split identifiers differ from their frozen manifest")
        split_ids = {
            identifier
            for values in split.get("strata", {}).values()
            for identifier in values.get("fit_ids", []) + values.get("validation_ids", [])
        }
        if split_ids != fixture_ids_by_path[f"fixtures/adaptation-seed-{seed_index}.json"]:
            raise ValueError("V15 pre-capture split does not partition its frozen adaptation fixture")
        split_by_seed[seed_index] = split
    adaptation_rows = candidate_freeze.get("adaptation_artifacts", [])
    expected_adaptation = {(seed_index, layer) for seed_index in range(len(fixtures.SEEDS)) for layer in TARGET_LAYERS}
    actual_adaptation = {(int(row["seed_index"]), int(row["layer"])) for row in adaptation_rows}
    if actual_adaptation != expected_adaptation:
        raise ValueError("V15 adaptation manifest does not cover every seed/layer capture")
    for row in adaptation_rows:
        path = root / row["path"]
        if not path.is_file() or _sha_file(path) != row.get("sha256"):
            raise ValueError("V15 adaptation capture bytes differ from the candidate manifest")
        if row.get("row_count") != 96 or row.get("fit_count") != 72 or row.get("validation_count") != 24:
            raise ValueError("V15 adaptation capture row counts differ from the frozen split")
        split = split_by_seed[int(row["seed_index"])]
        fit_ids = {identifier for values in split["strata"].values() for identifier in values["fit_ids"]}
        validation_ids = {identifier for values in split["strata"].values() for identifier in values["validation_ids"]}
        if len(fit_ids) != 72 or len(validation_ids) != 24 or fit_ids & validation_ids:
            raise ValueError("V15 frozen fit and validation identifier sets are invalid")
        with np.load(path, allow_pickle=False) as capture:
            captured_ids = {str(identifier) for identifier in capture["example_ids"].tolist()}
            captured_fit = {str(identifier) for identifier in capture["fit_ids"].tolist()}
            captured_validation = {str(identifier) for identifier in capture["validation_ids"].tolist()}
            strata = [str(value) for value in capture["strata"].tolist()]
            if (
                captured_ids != fit_ids | validation_ids
                or captured_fit != fit_ids
                or captured_validation != validation_ids
                or len(captured_ids) != 96
                or any(strata.count(name) != 32 for name in fixtures.STRATA)
            ):
                raise ValueError("V15 adaptation capture does not honor the pre-capture stratified split")
    expected = {
        (seed_index, layer, rank, step)
        for seed_index in range(len(fixtures.SEEDS))
        for layer in TARGET_LAYERS
        for rank in RANKS
        for step in CHECKPOINT_STEPS
    }
    rows = candidate_freeze.get("candidates", [])
    actual = {(int(row["seed_index"]), int(row["layer"]), int(row["rank"]), int(row["updates"])) for row in rows}
    if actual != expected or len(rows) != len(expected) or candidate_freeze.get("candidate_count") != len(expected):
        raise ValueError("V15 candidate manifest does not cover every frozen cell")
    for row in rows:
        path = root / row["path"]
        if not path.is_file() or _sha_file(path) != row["file_sha256"]:
            raise ValueError("V15 candidate bytes differ from the candidate manifest")
        candidate = DiagnosticStack(int(row["in_features"]), int(row["out_features"]), int(row["rank"]))
        candidate.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        if v3_core._module_digest(candidate) != row["state_sha256"]:
            raise ValueError("V15 candidate tensor digest differs from the candidate manifest")
        if sum(parameter.numel() for parameter in candidate.parameters()) != int(row["parameter_count"]):
            raise ValueError("V15 candidate parameter count differs from the candidate manifest")
    diagnostic = _json(root / "diagnostic-results.json")
    diagnostic_material = {key: value for key, value in diagnostic.items() if key != "diagnostic_results_sha256"}
    if diagnostic.get("diagnostic_results_sha256") != _digest(diagnostic_material):
        raise ValueError("V15 diagnostic-result digest is invalid")
    metrics = diagnostic.get("cell_metrics", [])
    if metrics != candidate_freeze.get("training_metrics") or diagnostic.get("classification") != _classify(metrics):
        raise ValueError("V15 diagnostic metrics differ from the sealed candidate training record")
    report = _json(root / "diagnostic-report.json")
    report_material = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != _digest(report_material):
        raise ValueError("V15 diagnostic report digest is invalid")
    if report.get("candidate_count") != len(rows) or report.get("diagnostic_results_sha256") != diagnostic["diagnostic_results_sha256"]:
        raise ValueError("V15 diagnostic report differs from sealed results")
    phases = _json(root / "phase-ledger.json").get("phases", [])
    if len(phases) != 1 or phases[0].get("phase") != "diagnostic" or phases[0].get("status") != "COMPLETED":
        raise ValueError("V15 diagnostic execution did not complete cleanly")
    if any((root / name).exists() for name in ("challenges", "challenge-entropy", "predictions", "outcomes")):
        raise ValueError("V15 adaptation diagnostic unexpectedly generated held-out challenges or predictions")
    return candidate_freeze, diagnostic, report


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _custody_root(primary_path)
    replica = _custody_root(replica_path)
    primary_archive = _verify_archive(primary)
    replica_archive = _verify_archive(replica)
    primary_selection = _json(primary / "checkpoint-selection.json")
    replica_selection = _json(replica / "checkpoint-selection.json")
    _primary_candidates, primary_metrics, primary_run = _verify_run_artifacts(primary)
    _replica_candidates, replica_metrics, replica_run = _verify_run_artifacts(replica)
    groups = {}
    for name in ("adaptation", "candidates", "fixtures", "splits"):
        left = _artifact_group(primary, name)
        right = _artifact_group(replica, name)
        groups[name] = {
            "primary_file_count": len(left),
            "replica_file_count": len(right),
            "primary_tree_sha256": _digest(left),
            "replica_tree_sha256": _digest(right),
            "byte_identical": left == right,
        }
    for name, relative in (("manifest", "candidate-freeze.json"), ("diagnostics", "diagnostic-results.json")):
        left = [{"path": relative, "bytes": (primary / relative).stat().st_size, "sha256": _sha_file(primary / relative)}]
        right = [{"path": relative, "bytes": (replica / relative).stat().st_size, "sha256": _sha_file(replica / relative)}]
        groups[name] = {
            "primary_file_count": len(left),
            "replica_file_count": len(right),
            "primary_tree_sha256": _digest(left),
            "replica_tree_sha256": _digest(right),
            "byte_identical": left == right,
        }
    result_match = primary_metrics.get("diagnostic_results_sha256") == replica_metrics.get("diagnostic_results_sha256")
    package_fields = ("python", "platform", "packages", "dependency_lock_sha256")
    runtime_pins_match = all(primary_run["runtime"].get(key) == replica_run["runtime"].get(key) for key in package_fields)
    separate_runtimes = primary_run["runtime"].get("python_executable") != replica_run["runtime"].get("python_executable")
    same_host = primary_run["runtime"].get("host_id_sha256") == replica_run["runtime"].get("host_id_sha256")
    same_owner = primary.stat().st_uid == replica.stat().st_uid
    all_groups_match = all(row["byte_identical"] for row in groups.values())
    reproduced = all_groups_match and result_match and runtime_pins_match and separate_runtimes and same_host and same_owner and primary_selection == replica_selection and primary_archive["archive_sha256"] == replica_archive["archive_sha256"]
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": verify_packet()["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "diagnostic_classification": primary_run["diagnostic_classification"] if result_match else "INCONCLUSIVE",
        "diagnostic_results_match": result_match,
        "diagnostic_results_sha256": [primary_metrics["diagnostic_results_sha256"], replica_metrics["diagnostic_results_sha256"]],
        "artifact_groups": groups,
        "runtime_pins_match": runtime_pins_match,
        "separate_python_runtimes": separate_runtimes,
        "same_host_and_owner": same_host and same_owner,
        "candidate_set_sha256": [primary_metrics["candidate_set_sha256"], replica_metrics["candidate_set_sha256"]],
        "target_selection_sha256": SOURCE_V14_SELECTION_SHA256,
        "data_and_artifact_custody": "synthetic prompts, activations, candidates, splits, and cell metrics remain in external owner custody",
        "claim_ceiling": CLAIM_CEILING,
        "limitations": [
            "single reused outcome-aware OPT/ReLU target; fixed layers 0 and 7",
            "three seeds and six nested seed/layer cells; descriptive only",
            "no challenge outcomes, continuation behavior, or learned rescue were evaluated",
            "same-owner local replay is not blind or independent acceptance",
        ],
        "next_blocked_authority_gate": "fresh frozen held-out causal and behavior evaluation must precede replacement-retention claims; independent science needs a blind replicator and authorized reviewer",
        "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "replication-report.json", report, immutable=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    commands.add_parser("verify")
    commands.add_parser("preflight")
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    bind = commands.add_parser("bind-target")
    bind.add_argument("--custody", required=True)
    run = commands.add_parser("diagnose")
    run.add_argument("--custody", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "verify":
        packet = verify_packet()
        result = {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "freeze_sha256": packet["freeze_sha256"], "packet_file_count": len(packet["packet_files"])}
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "bind-target":
        result = bind_target(args.custody)
    elif args.command == "diagnose":
        result = diagnose(args.custody)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
