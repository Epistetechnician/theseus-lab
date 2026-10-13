"""Custody-bound V13 nonlinear causal-replacement runner.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v13-nonlinear-causal-replacement-v1, and
global-experiment-assurance-lifecycle-v1. Reuses the V10 execution engine
under a new frozen packet; prompt and model artifacts stay in external custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import replacement_loss_v10_runner as engine
import replacement_loss_v11_runner as v11
import replacement_loss_v13_fixtures as fixtures
import replacement_loss_v3 as v3_core
import replacement_loss_v3_runner as v3


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v13-nonlinear-causal-replacement-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v13-freeze.json"
SOURCE_V11_PACKET_SHA256 = "2bd4e9903b2611ac454a0ff26ffabb051231bd2469f6895aee67504d8c99c582"
SOURCE_V11_ARCHIVE_SHA256 = "8b92f9f762c172c2c6ae49c12447b3906124fc4e6378b9793bd6cd6f0083f08c"
SOURCE_V11_SELECTION_SHA256 = "f5078a88b8094f58fcc1de282d52afe779ae672bc7d8c2ffe6dca1e8941b057f"
SOURCE_V11_TARGET_TREE_SHA256 = "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466"
SOURCE_V11_WEIGHT_FILE_SHA256 = "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132"
SOURCE_V11_MODEL_ID = "grushaprasad/opt125m-rootp-surprisal-curriculum"
SOURCE_V11_REVISION = "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
SOURCE_V11_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v11-primary"
CLAIM_CEILING = "LOCAL_NONLINEAR_CAUSAL_REPLACEMENT_IMPROVEMENT_OVER_ACTIVATION_RIDGE_ON_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS"
HIDDEN_DIM = 512
CAUSAL_UPDATES = 200
CAUSAL_LEARNING_RATE = 3e-4
CAUSAL_WEIGHT_DECAY = 1e-4
CAUSAL_GRAD_CLIP = 1.0
TARGET_LAYERS = (0, 7)
MAX_TOTAL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3

PACKET_FILES = (
    "AGENTS.md",
    "docs/replacement-loss-v13.md",
    "replacement_loss_v13_fixtures.py",
    "replacement_loss_v13_runner.py",
    "tests/test_replacement_loss_v13.py",
    "requirements-v13.lock.txt",
    "docs/replacement-loss-v12-replication.md",
    "docs/replacement-loss-v12-replication-results.md",
    "replacement_loss_v12_replication.py",
    "tests/test_replacement_loss_v12_replication.py",
    "requirements-v12.lock.txt",
    "docs/replacement-loss-v11.md",
    "docs/replacement-loss-v11-freeze.json",
    "docs/replacement-loss-v11-results.md",
    "replacement_loss_v11_fixtures.py",
    "replacement_loss_v11_runner.py",
    "tests/test_replacement_loss_v11.py",
    "requirements-v11.lock.txt",
    "docs/replacement-loss-v10.md",
    "docs/replacement-loss-v10-freeze.json",
    "replacement_loss_v10_fixtures.py",
    "replacement_loss_v10_runner.py",
    "tests/test_replacement_loss_v10.py",
    "requirements-v10.lock.txt",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
)

_REPO_ROOT = Path(__file__).resolve().parent
_ORIGINALS = {
    "freeze_packet": engine.freeze_packet,
    "verify_packet": engine.verify_packet,
    "verify_runtime": engine._verify_runtime,
    "visible_selection": engine._visible_selection,
    "fit_causal_candidate": engine._fit_causal_candidate,
    "save_candidate": engine._save_candidate,
    "load_candidate": engine._load_candidate,
    "write_json": engine._write_json,
    "verify_completed_artifacts": engine._verify_completed_artifacts,
}
_CONFIGURED = False


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
        raise ValueError(f"expected JSON object in {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable V13 artifact already exists: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _packet_rows(root: Path = _REPO_ROOT) -> list[dict[str, str]]:
    rows = []
    for relative in PACKET_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V13 packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def freeze_packet() -> dict[str, Any]:
    destination = _REPO_ROOT / FREEZE_FILE
    if destination.exists():
        raise FileExistsError("V13 packet freeze is immutable")
    v11_freeze = _read_json(_REPO_ROOT / "docs/replacement-loss-v11-freeze.json")
    if v11_freeze.get("packet_sha256") != SOURCE_V11_PACKET_SHA256:
        raise ValueError("V13 source V11 freeze packet digest differs from the predeclared target")
    v11_results_sha = _sha_file(_REPO_ROOT / "docs/replacement-loss-v11-results.md")
    rows = _packet_rows()
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_V13_FIXTURE_GENERATION_AND_CANDIDATE_TRAINING",
        "created_at_utc": engine._now(),
        "packet_files": rows,
        "packet_sha256": _canonical_digest(rows),
        "source_v11_packet_sha256": SOURCE_V11_PACKET_SHA256,
        "source_v11_results_file_sha256": v11_results_sha,
        "source_v11_freeze_sha256": _sha_file(_REPO_ROOT / "docs/replacement-loss-v11-freeze.json"),
        "source_v11_scientific_outcome": "FAIL",
        "source_v11_original_comparator": "INCONCLUSIVE",
        "target_predeclared": {
            "model_id": SOURCE_V11_MODEL_ID,
            "revision": SOURCE_V11_REVISION,
            "tree_sha256": SOURCE_V11_TARGET_TREE_SHA256,
            "weight_file_sha256": SOURCE_V11_WEIGHT_FILE_SHA256,
            "reused_from_v11": True,
            "fresh_checkpoint_claimed": False,
        },
        "fixture_generation_started": False,
        "challenge_generation_started": False,
        "candidate_training_started": False,
        "fixture_seed_count": len(fixtures.SEEDS),
        "selected_layers": list(TARGET_LAYERS),
        "methods": ["activation_ridge", "causal_fit_nonlinear_mlp"],
        "replacement_architecture": {
            "input_width": 3072,
            "hidden_width": HIDDEN_DIM,
            "output_width": 768,
            "activation": "GELU",
            "parameters": 1_967_360,
            "source_weights_used": False,
        },
        "primary_metric": "mean seed-clustered difference in held-out next-logit intervention-effect cosine, causal_fit_nonlinear_mlp minus activation_ridge",
        "primary_minimum_effect": 0.05,
        "primary_lower_confidence_bound": 0.95,
        "bootstrap_draws": 10_000,
        "bootstrap_seed": 202609399,
        "claim_ceiling": CLAIM_CEILING,
        "resource_limits": {
            "execution": "local_cpu_only",
            "max_total_seconds": MAX_TOTAL_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "max_parameters": 200_000_000,
            "max_weight_bytes": 800 * 1024**2,
            "paid_or_remote_compute": False,
            "provider_calls": False,
        },
    }
    _write_json(destination, material, immutable=True)
    return material


def verify_packet(root: Path = _REPO_ROOT) -> dict[str, Any]:
    repo = root.resolve()
    payload = _read_json(repo / FREEZE_FILE)
    rows = _packet_rows(repo)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_V13_FIXTURE_GENERATION_AND_CANDIDATE_TRAINING"
        or payload.get("packet_files") != rows
        or payload.get("packet_sha256") != _canonical_digest(rows)
        or payload.get("source_v11_packet_sha256") != SOURCE_V11_PACKET_SHA256
        or payload.get("source_v11_freeze_sha256") != _sha_file(repo / "docs/replacement-loss-v11-freeze.json")
        or payload.get("source_v11_results_file_sha256") != _sha_file(repo / "docs/replacement-loss-v11-results.md")
        or payload.get("source_v11_scientific_outcome") != "FAIL"
        or payload.get("source_v11_original_comparator") != "INCONCLUSIVE"
        or payload.get("fixture_generation_started") is not False
        or payload.get("challenge_generation_started") is not False
        or payload.get("candidate_training_started") is not False
    ):
        raise ValueError("V13 packet freeze verification failed")
    return payload


def _external_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    try:
        root.relative_to(_REPO_ROOT.resolve())
    except ValueError:
        return root
    raise ValueError("V13 model and experiment artifacts must stay outside the repository")


class NonlinearReplacement(torch.nn.Module):
    """Fresh low-parameter nonlinear map matching one OPT fc2 boundary."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int = HIDDEN_DIM) -> None:
        super().__init__()
        self.first = torch.nn.Linear(in_features, hidden_dim, bias=True, dtype=torch.float32)
        self.activation = torch.nn.GELU()
        self.second = torch.nn.Linear(hidden_dim, out_features, bias=True, dtype=torch.float32)
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self.hidden_dim = int(hidden_dim)

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


def _new_nonlinear(in_features: int, out_features: int, seed: int) -> NonlinearReplacement:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return NonlinearReplacement(in_features, out_features, HIDDEN_DIM).to(device="cpu", dtype=torch.float32)


def _fit_nonlinear_candidate(
    model: torch.nn.Module,
    layer: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
    training_rows: list[dict[str, Any]],
    pad_token_id: int,
    *,
    seed: int,
    updates: int = CAUSAL_UPDATES,
    learning_rate: float = CAUSAL_LEARNING_RATE,
    weight_decay: float = CAUSAL_WEIGHT_DECAY,
) -> tuple[NonlinearReplacement, dict[str, Any]]:
    if not use_bias or updates < 1 or len(training_rows) < engine.TRAIN_BATCH_SIZE:
        raise ValueError("V13 requires a biased source projection, positive updates, and one full adaptation batch")
    if (in_features, out_features) != (3072, 768):
        raise ValueError("V13 frozen nonlinear module dimensions do not match OPT fc2")
    target_values = torch.as_tensor(np.stack([row["target_y"] for row in training_rows]), dtype=torch.float32)
    source_module = v3._projection(model, engine.FAMILY, layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidate = _new_nonlinear(in_features, out_features, seed)
    initialization_digest = v3_core._module_digest(candidate)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    v3._set_projection(model, engine.FAMILY, layer, candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = np.random.default_rng(int(seed) ^ 0xD13C0A5)
    order = generator.permutation(len(training_rows))
    activation_scale = float(target_values.square().mean().clamp_min(1e-6).item())
    final_activation_loss: float | None = None
    final_causal_loss: float | None = None
    try:
        model.eval()
        candidate.train()
        for update in range(updates):
            engine._check_budget()
            start = (update * engine.TRAIN_BATCH_SIZE) % len(order)
            indexes = order[(start + np.arange(engine.TRAIN_BATCH_SIZE)) % len(order)]
            batch_rows = [training_rows[int(index)] for index in indexes]
            inputs, positions, deltas, targets, teacher_effects = engine._collate_training(batch_rows, pad_token_id)
            base_logits, predicted_y = engine._candidate_forward(model, layer, inputs, positions, None)
            changed_logits, _changed_y = engine._candidate_forward(model, layer, inputs, positions, deltas)
            predicted_effect = changed_logits - base_logits
            effect_scales = teacher_effects.square().mean(dim=1).sqrt().clamp_min(1e-5)
            activation_loss = (predicted_y - targets).square().mean() / activation_scale
            causal_loss = ((predicted_effect - teacher_effects) / effect_scales[:, None]).square().mean()
            loss = activation_loss + causal_loss
            if not torch.isfinite(loss):
                raise ValueError("V13 nonlinear training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), CAUSAL_GRAD_CLIP)
            optimizer.step()
            final_activation_loss = float(activation_loss.detach().item())
            final_causal_loss = float(causal_loss.detach().item())
        candidate.eval()
        receipt = {
            "method": "causal_fit_nonlinear_mlp",
            "architecture": "Linear-GELU-Linear",
            "initialization": "fresh_random_without_source_weights",
            "initialization_sha256": initialization_digest,
            "fitted_candidate_sha256": v3_core._module_digest(candidate),
            "adaptation_rows": len(training_rows),
            "updates": updates,
            "batch_size": engine.TRAIN_BATCH_SIZE,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_norm_clip": CAUSAL_GRAD_CLIP,
            "activation_loss_weight": 1.0,
            "causal_loss_weight": 1.0,
            "activation_normalization_mean_square": activation_scale,
            "final_update_normalized_activation_mse": final_activation_loss,
            "final_update_normalized_causal_mse": final_causal_loss,
            "optimizer": "AdamW",
            "seed": int(seed),
            "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        }
        return candidate, receipt
    finally:
        v3._set_projection(model, engine.FAMILY, layer, source_module)
        engine.gc.collect()


def _save_candidate(path: Path, module: torch.nn.Module) -> dict[str, Any]:
    if not isinstance(module, NonlinearReplacement):
        return _ORIGINALS["save_candidate"](path, module)
    if engine._CUSTODY is None:
        raise RuntimeError("V13 candidate save was called outside a custody-bound phase")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    save_file(state, str(path))
    return {
        "path": path.relative_to(engine._CUSTODY).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": v3_core._module_digest(module),
        "in_features": module.in_features,
        "out_features": module.out_features,
        "use_bias": True,
        "architecture": "Linear-GELU-Linear",
        "hidden_dim": module.hidden_dim,
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
    }


def _load_candidate(custody: Path, row: dict[str, Any]) -> torch.nn.Module:
    if row.get("architecture") != "Linear-GELU-Linear":
        return _ORIGINALS["load_candidate"](custody, row)
    path = custody / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("V13 nonlinear candidate bytes differ from the frozen manifest")
    module = NonlinearReplacement(int(row["in_features"]), int(row["out_features"]), int(row["hidden_dim"]))
    module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if v3_core._module_digest(module) != row["state_sha256"]:
        raise ValueError("V13 nonlinear candidate tensor-state digest differs")
    if sum(parameter.numel() for parameter in module.parameters()) != int(row["parameter_count"]):
        raise ValueError("V13 nonlinear candidate parameter count differs")
    module.eval()
    return module


def _fixed_layer_selection(model: torch.nn.Module, tokenizer: Any, records: list[dict[str, Any]]) -> tuple[list[int], list[float], list[float]]:
    layers, variances, rms = _ORIGINALS["visible_selection"](model, tokenizer, records)
    del layers
    if engine.v3._layer_count(model, engine.FAMILY) <= max(TARGET_LAYERS):
        raise ValueError("V13 predeclared OPT layers are absent")
    return list(TARGET_LAYERS), variances, rms


def _verify_v11_archive(source_root: Path) -> dict[str, Any]:
    manifest_path = source_root / "frozen-packet-manifest.json"
    manifest = _read_json(manifest_path)
    material = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    if (
        manifest.get("protocol_id") != v11.PROTOCOL_ID
        or manifest.get("packet_sha256") != SOURCE_V11_PACKET_SHA256
        or manifest.get("archive_sha256") != SOURCE_V11_ARCHIVE_SHA256
        or manifest.get("archive_sha256") != _canonical_digest(material)
    ):
        raise ValueError("V11 frozen packet archive identity failed V13 source verification")
    expected = {*v11.PACKET_FILES, v11.FREEZE_FILE}
    if {row.get("path") for row in manifest.get("archive_files", [])} != expected:
        raise ValueError("V11 archived packet inventory differs from its frozen manifest")
    for row in manifest["archive_files"]:
        path = source_root / "frozen-packet" / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V11 archived packet bytes differ: {row['path']}")
    freeze = _read_json(source_root / "frozen-packet" / v11.FREEZE_FILE)
    if freeze.get("packet_sha256") != SOURCE_V11_PACKET_SHA256:
        raise ValueError("V11 freeze record does not bind its declared packet")
    return manifest


def _verify_v11_target(source_root_value: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_root = Path(source_root_value).expanduser().resolve()
    archive = _verify_v11_archive(source_root)
    selection = _read_json(source_root / "checkpoint-selection.json")
    material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("protocol_id") != v11.PROTOCOL_ID
        or selection.get("packet_sha256") != SOURCE_V11_PACKET_SHA256
        or selection.get("packet_archive_sha256") != archive["archive_sha256"]
        or selection.get("selection_sha256") != SOURCE_V11_SELECTION_SHA256
        or selection.get("selection_sha256") != _canonical_digest(material)
        or selection.get("model_id") != SOURCE_V11_MODEL_ID
        or selection.get("revision") != SOURCE_V11_REVISION
        or selection.get("source_artifact_tree_sha256") != SOURCE_V11_TARGET_TREE_SHA256
    ):
        raise ValueError("V11 target-selection receipt does not match the V13 predeclared target")
    architecture = selection.get("config_architecture", {})
    if (
        architecture.get("model_type") != "opt"
        or architecture.get("activation_function") != "relu"
        or int(architecture.get("hidden_size", 0)) != 768
        or int(architecture.get("ffn_dim", 0)) != 3072
        or int(architecture.get("num_hidden_layers", 0)) <= max(TARGET_LAYERS)
    ):
        raise ValueError("V11 target config differs from the frozen OPT/ReLU V13 replacement dimensions")
    model_path = Path(selection["local_path"]).expanduser().resolve()
    tree_sha, _tree_rows = v3._tree_digest(model_path)
    if tree_sha != SOURCE_V11_TARGET_TREE_SHA256:
        raise ValueError("V13 reused checkpoint tree digest differs from the frozen V11 target")
    if _canonical_digest(selection.get("weight_files", [])) != selection.get("weights_sha256"):
        raise ValueError("V11 target weight inventory digest is invalid")
    for row in selection["weight_files"]:
        path = model_path / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError("V13 reused checkpoint weight file differs from its V11 receipt")
    if not any(row["sha256"] == SOURCE_V11_WEIGHT_FILE_SHA256 for row in selection["weight_files"]):
        raise ValueError("V13 reused checkpoint weight digest differs from the predeclared model")
    outcome = _read_json(source_root / "outcome-report.json")
    comparator = _read_json(source_root / "replication-report.json")
    if outcome.get("scientific_outcome") != "FAIL" or comparator.get("assurance_level") != "INCONCLUSIVE":
        raise ValueError("V13 source V11 result statuses differ from the predeclared source record")
    return selection, {"archive": archive, "outcome": outcome, "comparator": comparator}


def _checkpoint_manifest(custody: Path) -> dict[str, Any]:
    packet = verify_packet()
    selection = _read_json(custody / "checkpoint-selection.json")
    archive = engine._verify_packet_archive(custody)
    material = {key: value for key, value in selection.items() if key != "selection_sha256"}
    if (
        selection.get("packet_sha256") != packet["packet_sha256"]
        or selection.get("packet_archive_sha256") != archive["archive_sha256"]
        or selection.get("selection_sha256") != _canonical_digest(material)
        or selection.get("target_predeclared_before_freeze") is not True
        or selection.get("target_reused_from_v11") is not True
        or selection.get("selection_performed") is not False
    ):
        raise ValueError("V13 target binding is not tied to its frozen packet and archive")
    source, source_status = _verify_v11_target(selection["source_v11_custody_root"])
    for key in ("model_id", "revision", "local_path", "source_artifact_tree_sha256", "weight_files", "weights_sha256"):
        if selection.get(key) != source.get(key):
            raise ValueError(f"V13 target binding differs from V11 source selection: {key}")
    if selection.get("source_v11_selection_sha256") != source["selection_sha256"]:
        raise ValueError("V13 target binding references a different V11 selection")
    if source_status["archive"]["archive_sha256"] != SOURCE_V11_ARCHIVE_SHA256:
        raise ValueError("V13 target binding references a different V11 packet archive")
    return selection


def _load_checkpoint(custody: Path) -> tuple[Any, dict[str, Any]]:
    selection = _checkpoint_manifest(custody)
    model_path = Path(selection["local_path"]).resolve()
    checkpoint = v3.Checkpoint(
        family="opt_relu",
        model_id=SOURCE_V11_MODEL_ID,
        revision=SOURCE_V11_REVISION,
        local_path=model_path,
        selected_after_protocol_freeze=False,
    )
    return checkpoint, selection


def _runtime() -> dict[str, Any]:
    runtime = _ORIGINALS["verify_runtime"]()
    runtime["dependency_lock_sha256"] = _sha_file(_REPO_ROOT / "requirements-v13.lock.txt")
    runtime["packet_sha256"] = verify_packet()["packet_sha256"]
    return runtime


def configure_engine() -> Any:
    global _CONFIGURED
    if _CONFIGURED:
        return engine
    engine.STATE_SLICE = STATE_SLICE
    engine.PROTOCOL_SLICE = PROTOCOL_SLICE
    engine.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    engine.PROTOCOL_ID = PROTOCOL_ID
    engine.FAMILY = "opt_relu"
    engine.METHODS = ("activation_ridge", "causal_fit")
    engine.PACKET_FREEZE = FREEZE_FILE
    engine.PACKET_FILES = PACKET_FILES
    engine.CAUSAL_FIT_UPDATES = CAUSAL_UPDATES
    engine.CAUSAL_FIT_LEARNING_RATE = CAUSAL_LEARNING_RATE
    engine.CAUSAL_FIT_WEIGHT_DECAY = CAUSAL_WEIGHT_DECAY
    engine.MAX_TOTAL_SECONDS = MAX_TOTAL_SECONDS
    engine.MAX_RSS_BYTES = MAX_RSS_BYTES
    engine.fixtures = fixtures
    engine._root = lambda: _REPO_ROOT
    engine.verify_packet = verify_packet
    engine.freeze_packet = freeze_packet
    engine._verify_runtime = _runtime
    engine._visible_selection = _fixed_layer_selection
    engine._fit_causal_candidate = _fit_nonlinear_candidate
    engine._save_candidate = _save_candidate
    engine._load_candidate = _load_candidate
    engine._checkpoint_manifest = _checkpoint_manifest
    engine._load_checkpoint = _load_checkpoint
    engine._write_json = _ORIGINALS["write_json"]
    engine._verify_completed_artifacts = _ORIGINALS["verify_completed_artifacts"]
    _CONFIGURED = True
    return engine


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    return runner.archive_packet(custody_path)


def bind_target(custody_path: str | Path, source_v11_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    custody = runner._custody_root(custody_path)
    if (custody / "checkpoint-selection.json").exists():
        raise FileExistsError("V13 target binding is immutable")
    packet = verify_packet()
    archive = runner.archive_packet(custody)
    source, _status = _verify_v11_target(source_v11_path)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "packet_archive_sha256": archive["archive_sha256"],
        "target_predeclared_before_freeze": True,
        "target_reused_from_v11": True,
        "selection_performed": False,
        "source_v11_custody_root": str(Path(source_v11_path).expanduser().resolve()),
        "source_v11_packet_sha256": SOURCE_V11_PACKET_SHA256,
        "source_v11_archive_sha256": SOURCE_V11_ARCHIVE_SHA256,
        "source_v11_selection_sha256": source["selection_sha256"],
        "source_v11_scientific_outcome": "FAIL",
        "source_v11_original_comparator": "INCONCLUSIVE",
        "model_id": source["model_id"],
        "revision": source["revision"],
        "resolved_revision": source["resolved_revision"],
        "local_path": source["local_path"],
        "license_id": source["license_id"],
        "license_source_url": source["license_source_url"],
        "config_sha256": source["config_sha256"],
        "config_architecture": source["config_architecture"],
        "model_card_sha256": source["model_card_sha256"],
        "source_artifact_tree_sha256": source["source_artifact_tree_sha256"],
        "weight_files": source["weight_files"],
        "total_weight_bytes": source["total_weight_bytes"],
        "weights_sha256": source["weights_sha256"],
    }
    payload["selection_sha256"] = _canonical_digest(payload)
    _write_json(custody / "checkpoint-selection.json", payload, immutable=True)
    _checkpoint_manifest(custody)
    return payload


def prepare_replica(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    if (replica / "checkpoint-selection.json").exists():
        raise FileExistsError("V13 replica target binding already exists")
    runner._verify_packet_archive(primary)
    source_selection = _checkpoint_manifest(primary)
    runner.archive_packet(replica)
    _write_json(replica / "checkpoint-selection.json", source_selection, immutable=True)
    _checkpoint_manifest(replica)
    return {
        "replica_packet_archive_sha256": runner._verify_packet_archive(replica)["archive_sha256"],
        "selection_sha256": source_selection["selection_sha256"],
        "target_tree_sha256": source_selection["source_artifact_tree_sha256"],
    }


def score(custody_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    custody = runner._custody_root(custody_path)
    result = runner.score(custody)
    report_path = custody / "outcome-report.json"
    report = _read_json(report_path)
    report["claim_ceiling"] = CLAIM_CEILING
    report["replacement_architecture"] = {
        "method_name": "causal_fit",
        "architecture": "Linear-GELU-Linear",
        "hidden_dim": HIDDEN_DIM,
        "source_weights_used": False,
        "updates": CAUSAL_UPDATES,
        "learning_rate": CAUSAL_LEARNING_RATE,
        "weight_decay": CAUSAL_WEIGHT_DECAY,
        "gradient_norm_clip": CAUSAL_GRAD_CLIP,
    }
    report["source_v11_context"] = {
        "packet_sha256": SOURCE_V11_PACKET_SHA256,
        "scientific_outcome": "FAIL",
        "original_comparator_assurance": "INCONCLUSIVE",
        "not_a_paired_baseline": True,
    }
    report["known_limitations"] = [
        "one reused OPT/ReLU checkpoint and six synthetic fixture seeds",
        "layers 0 and 7 were fixed from previously reported causal-effect misses",
        "same-fixture affine causal-fit was not included, so the result does not isolate the nonlinear architecture effect",
        "same-owner local execution is not blind replication or independent scientific acceptance",
    ]
    report["next_blocked_authority_gate"] = "a separate blind replicator and independent reviewer are required for independent science; alignment, consciousness, and production gates remain separate"
    report_material = {key: value for key, value in report.items() if key not in {"outcome_report_sha256", "created_at_utc"}}
    report["outcome_report_sha256"] = _canonical_digest(report_material)
    _write_json(report_path, report)
    return {**result, "claim_ceiling": CLAIM_CEILING, "outcome_report_sha256": report["outcome_report_sha256"]}


def _group_rows(root: Path, relative: str) -> list[dict[str, Any]]:
    base = root / relative
    if not base.is_dir():
        raise FileNotFoundError(f"V13 replication artifact group is missing: {relative}")
    rows = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        rows.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def stable_candidate_identity(freeze: dict[str, Any], selection: dict[str, Any]) -> str:
    candidates = []
    for row in sorted(freeze["candidates"], key=lambda item: (int(item["seed_index"]), int(item["layer"]), str(item["method"]))):
        candidates.append({
            "seed_index": int(row["seed_index"]),
            "seed": int(row["seed"]),
            "layer": int(row["layer"]),
            "method": row["method"],
            "file_sha256": row["file_sha256"],
            "state_sha256": row["state_sha256"],
            "architecture": row.get("architecture", "affine-linear"),
            "fit_receipt": row["fit_receipt"],
            "adaptation_sha256": row["adaptation_sha256"],
        })
    material = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": freeze["packet_sha256"],
        "target": {
            "model_id": selection["model_id"],
            "revision": selection["revision"],
            "tree_sha256": selection["source_artifact_tree_sha256"],
            "weights_sha256": selection["weights_sha256"],
            "selection_sha256": selection["selection_sha256"],
        },
        "visible_fixture_sha256": freeze["visible_fixture_sha256"],
        "selected_layers": freeze["selected_layers"],
        "adaptation_fixture_sha256": freeze["adaptation_fixture_sha256"],
        "candidates": candidates,
    }
    return _canonical_digest(material)


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    primary_freeze, primary_selection = runner._candidate_freeze(primary)
    replica_freeze, replica_selection = runner._candidate_freeze(replica)
    runner._verify_candidates(primary, primary_freeze)
    runner._verify_candidates(replica, replica_freeze)
    primary_artifacts = runner._verify_completed_artifacts(primary, primary_freeze)
    replica_artifacts = runner._verify_completed_artifacts(replica, replica_freeze)
    group_dirs = {
        "fixtures": "fixtures",
        "adaptation": "adaptation",
        "candidates": "candidates",
        "challenge_entropy": "challenge-entropy",
        "challenges": "challenges",
        "predictions": "predictions",
        "outcomes": "outcomes",
    }
    groups = {}
    all_groups_match = True
    for name, directory in group_dirs.items():
        left = _group_rows(primary, directory)
        right = _group_rows(replica, directory)
        matched = left == right
        all_groups_match = all_groups_match and matched
        groups[name] = {
            "primary_file_count": len(left),
            "replica_file_count": len(right),
            "primary_tree_sha256": _canonical_digest(left),
            "replica_tree_sha256": _canonical_digest(right),
            "byte_identical": matched,
        }

    primary_candidate_id = stable_candidate_identity(primary_freeze, primary_selection)
    replica_candidate_id = stable_candidate_identity(replica_freeze, replica_selection)
    challenge_manifest_primary = _read_json(primary / "challenge-manifest.json")
    challenge_manifest_replica = _read_json(replica / "challenge-manifest.json")
    prediction_manifest_primary = _read_json(primary / "prediction-manifest.json")
    prediction_manifest_replica = _read_json(replica / "prediction-manifest.json")
    challenge_ids = []
    prediction_ids = []
    for root, candidate_id, challenge_manifest, prediction_manifest in (
        (primary, primary_candidate_id, challenge_manifest_primary, prediction_manifest_primary),
        (replica, replica_candidate_id, challenge_manifest_replica, prediction_manifest_replica),
    ):
        challenge_id = _canonical_digest({
            "candidate_content_sha256": candidate_id,
            "entropy": challenge_manifest["entropy"],
            "challenge_rows": challenge_manifest["challenge_rows"],
        })
        challenge_ids.append(challenge_id)
        prediction_rows = [
            {key: row[key] for key in ("seed_index", "layer", "method", "control", "sha256") if key in row}
            for row in prediction_manifest["predictions"]
        ]
        prediction_ids.append(_canonical_digest({"candidate_content_sha256": candidate_id, "challenge_content_sha256": challenge_id, "predictions": prediction_rows}))

    primary_report = _read_json(primary / "outcome-report.json")
    replica_report = _read_json(replica / "outcome-report.json")
    scientific_ids = [primary_report["scientific_results_sha256"], replica_report["scientific_results_sha256"]]
    runtimes = [primary_report["runtime"], replica_report["runtime"]]
    pins_match = all(runtimes[0].get(key) == runtimes[1].get(key) for key in ("python", "platform", "packages", "dependency_lock_sha256"))
    separate_runtime = runtimes[0].get("python_executable") != runtimes[1].get("python_executable")
    same_owner = primary.stat().st_uid == replica.stat().st_uid
    same_host = runtimes[0].get("host_id_sha256") == runtimes[1].get("host_id_sha256")
    source_status_preserved = all(
        report.get("source_v11_context", {}).get("scientific_outcome") == "FAIL"
        and report.get("source_v11_context", {}).get("original_comparator_assurance") == "INCONCLUSIVE"
        for report in (primary_report, replica_report)
    )
    identities_match = (
        primary_candidate_id == replica_candidate_id
        and challenge_ids[0] == challenge_ids[1]
        and prediction_ids[0] == prediction_ids[1]
        and scientific_ids[0] == scientific_ids[1]
        and primary_report["scientific_outcome"] == replica_report["scientific_outcome"]
    )
    reproduced = all_groups_match and identities_match and pins_match and separate_runtime and same_owner and same_host and source_status_preserved
    packet = verify_packet()
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "assurance_scope": "same-owner, same-host reproduction of V13 nonlinear causal-replacement artifacts and outcome across distinct pinned Python runtimes",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPRODUCTION",
        "scientific_outcome": primary_report["scientific_outcome"] if identities_match else "INCONCLUSIVE",
        "claim_ceiling": CLAIM_CEILING,
        "source_v11": {
            "packet_sha256": SOURCE_V11_PACKET_SHA256,
            "archive_sha256": SOURCE_V11_ARCHIVE_SHA256,
            "scientific_outcome": "FAIL",
            "original_comparator_assurance": "INCONCLUSIVE",
        },
        "target_identity": {
            "model_id": SOURCE_V11_MODEL_ID,
            "revision": SOURCE_V11_REVISION,
            "tree_sha256": SOURCE_V11_TARGET_TREE_SHA256,
            "raw_weight_file_sha256": SOURCE_V11_WEIGHT_FILE_SHA256,
            "selection_sha256": SOURCE_V11_SELECTION_SHA256,
        },
        "content_identities": {
            "candidate_sha256": [primary_candidate_id, replica_candidate_id],
            "challenge_sha256": challenge_ids,
            "prediction_sha256": prediction_ids,
            "scientific_results_sha256": scientific_ids,
            "candidate_content_match": primary_candidate_id == replica_candidate_id,
            "challenge_content_match": challenge_ids[0] == challenge_ids[1],
            "prediction_content_match": prediction_ids[0] == prediction_ids[1],
            "scientific_results_match": scientific_ids[0] == scientific_ids[1],
            "source_v11_status_preserved": source_status_preserved,
        },
        "artifact_groups": groups,
        "decision_conditions": {
            "all_artifact_groups_byte_identical": all_groups_match,
            "content_identities_match": identities_match,
            "runtime_versions_and_pins_match": pins_match,
            "separate_python_runtimes": separate_runtime,
            "same_owner_and_host": same_owner and same_host,
            "source_v11_status_preserved": source_status_preserved,
        },
        "runtime_provenance": {
            "primary": {
                "python": runtimes[0]["python"],
                "platform": runtimes[0]["platform"],
                "dependency_lock_sha256": runtimes[0]["dependency_lock_sha256"],
                "python_executable_path_sha256": hashlib.sha256(str(runtimes[0]["python_executable"]).encode()).hexdigest(),
                "host_id_sha256": runtimes[0]["host_id_sha256"],
                "packages": runtimes[0]["packages"],
                "run_budget": primary_artifacts["runtime"],
            },
            "replica": {
                "python": runtimes[1]["python"],
                "platform": runtimes[1]["platform"],
                "dependency_lock_sha256": runtimes[1]["dependency_lock_sha256"],
                "python_executable_path_sha256": hashlib.sha256(str(runtimes[1]["python_executable"]).encode()).hexdigest(),
                "host_id_sha256": runtimes[1]["host_id_sha256"],
                "packages": runtimes[1]["packages"],
                "run_budget": replica_artifacts["runtime"],
            },
        },
        "data_and_artifact_custody": "synthetic prompts, checkpoint files, captures, candidates, entropy, predictions, and row-level outcomes remain in external owner custody",
        "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
        "limitations": [
            "one reused OPT/ReLU checkpoint; no fresh-checkpoint generalization",
            "layers 0 and 7 were fixed from prior causal-effect misses",
            "not a blind replication or independent acceptance",
            "the comparator does not isolate nonlinear architecture from loss and optimizer changes",
            "no architecture-general, consciousness, alignment, or production claim",
        ],
        "next_blocked_authority_gate": "separate blind replication and independent reviewer authority for independent science",
    }
    report["report_sha256"] = _canonical_digest(report)
    _write_json(primary / "replication-report.json", report, immutable=True)
    return report


def main(argv: list[str] | None = None) -> int:
    runner = configure_engine()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    bind = commands.add_parser("bind-target")
    bind.add_argument("--custody", required=True)
    bind.add_argument("--source-v11", required=True)
    for name in ("prepare", "predict", "score"):
        command = commands.add_parser(name)
        command.add_argument("--custody", required=True)
        if name == "predict":
            command.add_argument("--entropy-source")
    replica = commands.add_parser("prepare-replica")
    replica.add_argument("--primary", required=True)
    replica.add_argument("--replica", required=True)
    compare = commands.add_parser("compare-replication")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "bind-target":
        result = bind_target(args.custody, args.source_v11)
    elif args.command == "prepare":
        result = runner.prepare(args.custody)
    elif args.command == "predict":
        result = runner.predict(args.custody, entropy_source=args.entropy_source)
    elif args.command == "score":
        result = score(args.custody)
    elif args.command == "prepare-replica":
        result = prepare_replica(args.primary, args.replica)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
