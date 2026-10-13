"""Custody-bound V14 paired activation ablation runner.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v14-paired-activation-ablation-v1, and
global-experiment-assurance-lifecycle-v1. Uses the V10 execution engine and
V13 checkpoint verification under a new frozen protocol. Payload artifacts
remain in external owner custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file, save_file

import replacement_loss_v13_runner as v13
import replacement_loss_v14_fixtures as fixtures
import replacement_loss_v3 as v3_core


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v14-paired-activation-ablation-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v14-freeze.json"
CLAIM_CEILING = "LOCAL_GELU_ACTIVATION_EFFECT_ON_MATCHED_LOW_RANK_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS"
SOURCE_V13_PACKET_SHA256 = "93c1ecfce1f39b76433ebcf2452d2723824ce41de5e00ab6df6cd7e314af839c"
SOURCE_V13_SCIENTIFIC_RESULTS_SHA256 = "3312c3052caaa6f00b0fe999d84222048cd3ad809464490a7b2095430df81aea"
SOURCE_V11_SELECTION_SHA256 = "f5078a88b8094f58fcc1de282d52afe779ae672bc7d8c2ffe6dca1e8941b057f"
HIDDEN_DIM = 512
UPDATES = 200
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
TARGET_LAYERS = (0, 7)
BOOTSTRAP_SEED = 202610099
BOOTSTRAP_DRAWS = 10_000
METHOD_KEY_MAPPING = {
    "activation_ridge": {
        "result_name": "affine_fit",
        "architecture": "Linear-Identity-Linear",
        "meaning": "paired AdamW-fitted factorized affine replacement; not ridge regression",
    },
    "causal_fit": {
        "result_name": "gelu_fit",
        "architecture": "Linear-GELU-Linear",
        "meaning": "paired AdamW-fitted nonlinear replacement",
    },
}

_REPO_ROOT = Path(__file__).resolve().parent
_PACKET_DEPENDENCIES = tuple(v13.PACKET_FILES) + (
    "docs/replacement-loss-v13-freeze.json",
    "docs/replacement-loss-v13-results.md",
)
PACKET_FILES = tuple(dict.fromkeys((
    "AGENTS.md",
    "docs/replacement-loss-v14.md",
    "replacement_loss_v14_fixtures.py",
    "replacement_loss_v14_runner.py",
    "tests/test_replacement_loss_v14.py",
    "requirements-v14.lock.txt",
    *_PACKET_DEPENDENCIES,
)))

_CURRENT_CONTEXT: dict[str, Any] = {}
_ENGINE: Any = None
_FIT_LINEAR_ORIGINAL = v3_core.fit_independent_linear_replacement
_CAPTURE_ORIGINAL = v13.engine._capture_adaptation_layer


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
        raise FileExistsError(f"immutable V14 artifact already exists: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _packet_rows(root: Path = _REPO_ROOT) -> list[dict[str, str]]:
    rows = []
    for relative in PACKET_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V14 packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def freeze_packet() -> dict[str, Any]:
    destination = _REPO_ROOT / FREEZE_FILE
    if destination.exists():
        raise FileExistsError("V14 packet freeze is immutable")
    source_v13_freeze = _read_json(_REPO_ROOT / "docs/replacement-loss-v13-freeze.json")
    if source_v13_freeze.get("packet_sha256") != SOURCE_V13_PACKET_SHA256:
        raise ValueError("V14 source V13 packet digest differs from the predeclared follow-up")
    rows = _packet_rows()
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_V14_FIXTURE_GENERATION_AND_CANDIDATE_TRAINING",
        "created_at_utc": v13.engine._now(),
        "packet_files": rows,
        "packet_sha256": _canonical_digest(rows),
        "source_v13_packet_sha256": SOURCE_V13_PACKET_SHA256,
        "source_v13_freeze_sha256": _sha_file(_REPO_ROOT / "docs/replacement-loss-v13-freeze.json"),
        "source_v13_results_file_sha256": _sha_file(_REPO_ROOT / "docs/replacement-loss-v13-results.md"),
        "source_v13_scientific_results_sha256": SOURCE_V13_SCIENTIFIC_RESULTS_SHA256,
        "source_v13_scientific_outcome": "FAIL",
        "source_v13_reproduction_assurance": "REPRODUCED_LOCAL",
        "source_v11_selection_sha256": SOURCE_V11_SELECTION_SHA256,
        "fixture_generation_started": False,
        "challenge_generation_started": False,
        "candidate_training_started": False,
        "fixture_seed_count": len(fixtures.SEEDS),
        "selected_layers": list(TARGET_LAYERS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_cluster": "fixture_seed",
        "primary_metric": "mean seed-clustered held-out next-logit intervention-effect cosine difference, GELU minus matched identity activation",
        "primary_minimum_effect": 0.05,
        "primary_lower_confidence_bound": 0.95,
        "method_key_mapping": METHOD_KEY_MAPPING,
        "paired_training": {
            "same_initial_parameter_tensors": True,
            "same_adaptation_rows_and_batch_order": True,
            "same_objective": "normalized activation MSE plus normalized next-logit intervention-effect MSE, unit weights",
            "optimizer": "AdamW",
            "updates": UPDATES,
            "batch_size": 8,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_norm_clip": GRADIENT_CLIP,
            "only_forward_difference": "GELU versus identity activation",
        },
        "replacement_architecture": {
            "input_width": 3072,
            "hidden_width": HIDDEN_DIM,
            "output_width": 768,
            "parameter_count_each": 1_967_360,
            "source_weights_used": False,
        },
        "resource_limits": {
            "execution": "local_cpu_only",
            "max_total_seconds": 4 * 60 * 60,
            "max_rss_bytes": 16 * 1024**3,
            "max_base_model_parameters": 200_000_000,
            "max_checkpoint_bytes": 800 * 1024**2,
            "paid_or_remote_compute": False,
            "provider_calls": False,
        },
        "claim_ceiling": CLAIM_CEILING,
    }
    material["freeze_sha256"] = _canonical_digest({key: value for key, value in material.items() if key != "freeze_sha256"})
    _write_json(destination, material, immutable=True)
    return material


def verify_packet(root: Path = _REPO_ROOT) -> dict[str, Any]:
    payload = _read_json(root.resolve() / FREEZE_FILE)
    rows = _packet_rows(root.resolve())
    material = {key: value for key, value in payload.items() if key not in {"freeze_sha256"}}
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_V14_FIXTURE_GENERATION_AND_CANDIDATE_TRAINING"
        or payload.get("packet_files") != rows
        or payload.get("packet_sha256") != _canonical_digest(rows)
        or payload.get("source_v13_packet_sha256") != SOURCE_V13_PACKET_SHA256
        or payload.get("source_v13_scientific_results_sha256") != SOURCE_V13_SCIENTIFIC_RESULTS_SHA256
        or payload.get("source_v13_scientific_outcome") != "FAIL"
        or payload.get("source_v13_reproduction_assurance") != "REPRODUCED_LOCAL"
        or payload.get("source_v11_selection_sha256") != SOURCE_V11_SELECTION_SHA256
        or payload.get("fixture_generation_started") is not False
        or payload.get("challenge_generation_started") is not False
        or payload.get("candidate_training_started") is not False
        or payload.get("freeze_sha256") != _canonical_digest(material)
    ):
        raise ValueError("V14 packet freeze verification failed")
    return payload


def _external_root(value: str | Path) -> Path:
    return v13._external_root(value)


class PairedStackReplacement(torch.nn.Module):
    """Same-shaped factorized map with identity or GELU activation."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int, *, gelu: bool) -> None:
        super().__init__()
        self.first = torch.nn.Linear(in_features, hidden_dim, bias=True, dtype=torch.float32)
        self.activation = torch.nn.GELU() if gelu else torch.nn.Identity()
        self.second = torch.nn.Linear(hidden_dim, out_features, bias=True, dtype=torch.float32)
        self._in_features = int(in_features)
        self._out_features = int(out_features)
        self.hidden_dim = int(hidden_dim)
        self.gelu = bool(gelu)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    @property
    def bias(self) -> torch.Tensor:
        return self.second.bias

    @property
    def architecture(self) -> str:
        return "Linear-GELU-Linear" if self.gelu else "Linear-Identity-Linear"

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(self.activation(self.first(value)))


def _new_stack(in_features: int, out_features: int, seed: int, *, gelu: bool) -> PairedStackReplacement:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return PairedStackReplacement(in_features, out_features, HIDDEN_DIM, gelu=gelu).to(device="cpu", dtype=torch.float32)


def _train_stack_candidate(
    model: torch.nn.Module,
    layer: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
    training_rows: list[dict[str, Any]],
    pad_token_id: int,
    *,
    seed: int,
    gelu: bool,
    updates: int = UPDATES,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[PairedStackReplacement, dict[str, Any]]:
    runner = configure_engine()
    if (in_features, out_features, use_bias) != (3072, 768, True):
        raise ValueError("V14 target dimensions differ from the frozen OPT fc2 boundary")
    if updates < 1 or len(training_rows) < runner.TRAIN_BATCH_SIZE:
        raise ValueError("V14 requires a positive update count and a full adaptation batch")
    target_values = torch.as_tensor(np.stack([row["target_y"] for row in training_rows]), dtype=torch.float32)
    source_module = runner.v3._projection(model, runner.FAMILY, layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidate = _new_stack(in_features, out_features, seed, gelu=gelu)
    initialization_digest = v3_core._module_digest(candidate)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    runner.v3._set_projection(model, runner.FAMILY, layer, candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    batch_seed = int(seed) ^ 0xA14F17
    generator = np.random.default_rng(batch_seed)
    order = generator.permutation(len(training_rows))
    batch_trace = []
    activation_scale = float(target_values.square().mean().clamp_min(1e-6).item())
    final_activation_loss: float | None = None
    final_causal_loss: float | None = None
    try:
        model.eval()
        candidate.train()
        for update in range(updates):
            runner._check_budget()
            start = (update * runner.TRAIN_BATCH_SIZE) % len(order)
            indexes = order[(start + np.arange(runner.TRAIN_BATCH_SIZE)) % len(order)]
            batch_trace.extend(int(index) for index in indexes)
            batch_rows = [training_rows[int(index)] for index in indexes]
            inputs, positions, deltas, targets, teacher_effects = runner._collate_training(batch_rows, pad_token_id)
            base_logits, predicted_y = runner._candidate_forward(model, layer, inputs, positions, None)
            changed_logits, _changed_y = runner._candidate_forward(model, layer, inputs, positions, deltas)
            predicted_effect = changed_logits - base_logits
            effect_scales = teacher_effects.square().mean(dim=1).sqrt().clamp_min(1e-5)
            activation_loss = (predicted_y - targets).square().mean() / activation_scale
            causal_loss = ((predicted_effect - teacher_effects) / effect_scales[:, None]).square().mean()
            loss = activation_loss + causal_loss
            if not torch.isfinite(loss):
                raise ValueError("V14 paired candidate produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP)
            optimizer.step()
            final_activation_loss = float(activation_loss.detach().item())
            final_causal_loss = float(causal_loss.detach().item())
        candidate.eval()
        receipt = {
            "method": "paired_gelu_fit" if gelu else "paired_affine_fit",
            "architecture": candidate.architecture,
            "initialization": "shared_random_parameter_tensors_without_source_weights",
            "initialization_sha256": initialization_digest,
            "fitted_candidate_sha256": v3_core._module_digest(candidate),
            "adaptation_rows": len(training_rows),
            "updates": updates,
            "batch_size": runner.TRAIN_BATCH_SIZE,
            "batch_order_sha256": hashlib.sha256(np.asarray(batch_trace, dtype="<i8").tobytes()).hexdigest(),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_norm_clip": GRADIENT_CLIP,
            "activation_loss_weight": 1.0,
            "causal_loss_weight": 1.0,
            "activation_normalization_mean_square": activation_scale,
            "final_update_normalized_activation_mse": final_activation_loss,
            "final_update_normalized_causal_mse": final_causal_loss,
            "optimizer": "AdamW",
            "paired_seed": int(seed),
            "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
            "only_forward_difference": "GELU" if gelu else "identity",
        }
        return candidate, receipt
    finally:
        runner.v3._set_projection(model, runner.FAMILY, layer, source_module)
        runner.gc.collect()


def _paired_affine_fit(
    in_features: int,
    out_features: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    use_bias: bool,
    seed: int,
    ridge: float,
) -> tuple[PairedStackReplacement, dict[str, Any]]:
    del ridge
    context = _CURRENT_CONTEXT
    if not context or context.get("training_rows") is None:
        raise RuntimeError("paired affine fitting has no adaptation capture context")
    if not torch.equal(inputs, context["inputs"]) or not torch.equal(targets, context["targets"]):
        raise ValueError("paired affine inputs differ from the captured adaptation partition")
    return _train_stack_candidate(
        context["model"],
        context["layer"],
        in_features,
        out_features,
        use_bias,
        context["training_rows"],
        context["pad_token_id"],
        seed=seed,
        gelu=False,
    )


def _fit_gelu_candidate(
    model: torch.nn.Module,
    layer: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
    training_rows: list[dict[str, Any]],
    pad_token_id: int,
    *,
    seed: int,
    updates: int = UPDATES,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[PairedStackReplacement, dict[str, Any]]:
    context = _CURRENT_CONTEXT
    if not context or context.get("layer") != layer or context.get("training_rows") is not training_rows:
        raise ValueError("paired GELU fit does not match the current adaptation capture")
    return _train_stack_candidate(
        model, layer, in_features, out_features, use_bias, training_rows, pad_token_id,
        seed=seed, gelu=True, updates=updates, learning_rate=learning_rate, weight_decay=weight_decay,
    )


def _capture_adaptation_layer(model: torch.nn.Module, tokenizer: Any, layer: int, rows: list[dict[str, Any]], seed_index: int):
    runner = configure_engine()
    inputs, targets, training_rows = _CAPTURE_ORIGINAL(model, tokenizer, layer, rows, seed_index)
    _CURRENT_CONTEXT.clear()
    _CURRENT_CONTEXT.update({
        "model": model,
        "tokenizer": tokenizer,
        "pad_token_id": int(tokenizer.pad_token_id),
        "layer": layer,
        "seed_index": seed_index,
        "inputs": inputs,
        "targets": targets,
        "training_rows": training_rows,
    })
    return inputs, targets, training_rows


def _paired_candidate_seed(seed_index: int, layer: int, method: str) -> int:
    paired_method = "same-initialization-pair" if method in {"activation_ridge", "causal_fit"} else method
    digest = hashlib.sha256(f"{PROTOCOL_ID}\0{seed_index}\0{layer}\0{paired_method}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _save_candidate(path: Path, module: torch.nn.Module) -> dict[str, Any]:
    runner = configure_engine()
    if not isinstance(module, PairedStackReplacement):
        return v13._ORIGINALS["save_candidate"](path, module)
    if runner._CUSTODY is None:
        raise RuntimeError("V14 candidate save was called outside a custody-bound phase")
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    save_file(state, str(path))
    return {
        "path": path.relative_to(runner._CUSTODY).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": v3_core._module_digest(module),
        "in_features": module.in_features,
        "out_features": module.out_features,
        "use_bias": True,
        "architecture": module.architecture,
        "hidden_dim": module.hidden_dim,
        "parameter_count": sum(parameter.numel() for parameter in module.parameters()),
    }


def _load_candidate(custody: Path, row: dict[str, Any]) -> torch.nn.Module:
    architecture = row.get("architecture")
    if architecture not in {"Linear-Identity-Linear", "Linear-GELU-Linear"}:
        return v13._ORIGINALS["load_candidate"](custody, row)
    path = custody / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("V14 paired candidate bytes differ from their frozen manifest")
    module = PairedStackReplacement(
        int(row["in_features"]), int(row["out_features"]), int(row["hidden_dim"]),
        gelu=architecture == "Linear-GELU-Linear",
    )
    module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if v3_core._module_digest(module) != row["state_sha256"]:
        raise ValueError("V14 paired candidate tensor digest differs")
    if sum(parameter.numel() for parameter in module.parameters()) != int(row["parameter_count"]):
        raise ValueError("V14 paired candidate parameter count differs")
    module.eval()
    return module


def _runtime() -> dict[str, Any]:
    runtime = v13._ORIGINALS["verify_runtime"]()
    runtime["dependency_lock_sha256"] = _sha_file(_REPO_ROOT / "requirements-v14.lock.txt")
    runtime["packet_sha256"] = verify_packet()["packet_sha256"]
    return runtime


def configure_engine() -> Any:
    global _ENGINE
    if _ENGINE is not None:
        return _ENGINE
    v13.STATE_SLICE = STATE_SLICE
    v13.PROTOCOL_SLICE = PROTOCOL_SLICE
    v13.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    v13.PROTOCOL_ID = PROTOCOL_ID
    v13.FREEZE_FILE = FREEZE_FILE
    v13.PACKET_FILES = PACKET_FILES
    v13.CLAIM_CEILING = CLAIM_CEILING
    v13.fixtures = fixtures
    v13._REPO_ROOT = _REPO_ROOT
    v13.verify_packet = verify_packet
    v13._runtime = _runtime
    v13._fit_nonlinear_candidate = _fit_gelu_candidate
    v13._save_candidate = _save_candidate
    v13._load_candidate = _load_candidate
    v13._CONFIGURED = False
    runner = v13.configure_engine()
    runner.METHODS = ("activation_ridge", "causal_fit")
    runner.FAMILY = "opt_relu"
    runner.CAUSAL_FIT_UPDATES = UPDATES
    runner.CAUSAL_FIT_LEARNING_RATE = LEARNING_RATE
    runner.CAUSAL_FIT_WEIGHT_DECAY = WEIGHT_DECAY
    runner.BOOTSTRAP_SEED = BOOTSTRAP_SEED
    runner.BOOTSTRAP_DRAWS = BOOTSTRAP_DRAWS
    runner._fit_causal_candidate = _fit_gelu_candidate
    runner._capture_adaptation_layer = _capture_adaptation_layer
    runner._candidate_seed = _paired_candidate_seed
    runner._save_candidate = _save_candidate
    runner._load_candidate = _load_candidate
    runner._verify_runtime = _runtime
    runner.verify_packet = verify_packet
    runner._root = lambda: _REPO_ROOT
    runner.PACKET_FREEZE = FREEZE_FILE
    runner.PACKET_FILES = PACKET_FILES
    runner.fixtures = fixtures
    v3_core.fit_independent_linear_replacement = _paired_affine_fit
    _ENGINE = runner
    return runner


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    return configure_engine().archive_packet(custody_path)


def bind_target(custody_path: str | Path, source_v11_path: str | Path) -> dict[str, Any]:
    configure_engine()
    return v13.bind_target(custody_path, source_v11_path)


def _record_pair_mapping_before_candidate_seal() -> Any:
    runner = configure_engine()
    original_write = runner._write_json

    def write(path: Path, value: dict[str, Any], *, immutable: bool = False) -> None:
        if Path(path).name == "candidate-freeze.json":
            value = dict(value)
            value["method_key_mapping"] = METHOD_KEY_MAPPING
            value["paired_training"] = {
                "same_initial_parameter_tensors": True,
                "same_adaptation_rows_and_batch_order": True,
                "updates": UPDATES,
                "optimizer": "AdamW",
                "only_forward_difference": "GELU versus identity activation",
            }
            material = {key: item for key, item in value.items() if key not in {"candidate_set_sha256", "created_at_utc"}}
            value["candidate_set_sha256"] = _canonical_digest(material)
        original_write(path, value, immutable=immutable)

    runner._write_json = write
    return original_write


def prepare(custody_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    original_write = _record_pair_mapping_before_candidate_seal()
    try:
        result = runner.prepare(custody_path)
        freeze, _selection = runner._candidate_freeze(runner._custody_root(custody_path))
        rows = {(int(row["seed_index"]), int(row["layer"])): row for row in freeze["candidates"] if row["method"] == "activation_ridge"}
        for seed_index in range(len(fixtures.SEEDS)):
            for layer in TARGET_LAYERS:
                affine = rows[(seed_index, layer)]
                nonlinear = next(row for row in freeze["candidates"] if int(row["seed_index"]) == seed_index and int(row["layer"]) == layer and row["method"] == "causal_fit")
                if affine["fit_receipt"].get("initialization_sha256") != nonlinear["fit_receipt"].get("initialization_sha256"):
                    raise ValueError("paired candidates do not share identical initialization tensors")
                if affine["fit_receipt"].get("batch_order_sha256") != nonlinear["fit_receipt"].get("batch_order_sha256"):
                    raise ValueError("paired candidates do not share the same minibatch order")
        return result
    finally:
        runner._write_json = original_write


def predict(custody_path: str | Path, *, entropy_source: str | Path | None = None) -> dict[str, Any]:
    return configure_engine().predict(custody_path, entropy_source=entropy_source)


def score(custody_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    custody = runner._custody_root(custody_path)
    result = runner.score(custody)
    path = custody / "outcome-report.json"
    report = _read_json(path)
    for row in report["results"]:
        row["method"] = METHOD_KEY_MAPPING[row["method"]]["result_name"]
    comparison = report["primary_comparison"]
    for row in comparison["per_seed"]:
        row["gelu_fit"] = row.pop("causal_fit")
        row["affine_fit"] = row.pop("activation_ridge")
    affine_rows = [row for row in report["results"] if row["method"] == "affine_fit"]
    gelu_rows = [row for row in report["results"] if row["method"] == "gelu_fit"]
    affine_retention = bool(affine_rows) and all(row["retention_pass"] for row in affine_rows)
    gelu_retention = bool(gelu_rows) and all(row["retention_pass"] for row in gelu_rows)
    affine_rescue = bool(affine_rows) and all(row["activation_rescue_pass"] for row in affine_rows)
    gelu_rescue = bool(gelu_rows) and all(row["activation_rescue_pass"] for row in gelu_rows)
    paired_retention = affine_retention and gelu_retention
    comparative_pass = comparison["lower_95_bound"] > 0.05
    rescue_pass = affine_rescue and gelu_rescue
    zero_pass = bool(report["zero_intervention_control_pass"])
    report["primary_comparison"]["affine_fit_retention_pass"] = affine_retention
    report["primary_comparison"]["gelu_fit_retention_pass"] = gelu_retention
    report["primary_comparison"]["both_fit_retention_pass"] = paired_retention
    report["primary_comparison"]["both_activation_rescue_pass"] = rescue_pass
    report["causal_fit_retention_pass"] = gelu_retention
    report["activation_rescue_pass"] = rescue_pass
    report["scientific_outcome"] = "PASS" if comparative_pass and paired_retention and rescue_pass and zero_pass else "FAIL"
    report["method_key_mapping"] = METHOD_KEY_MAPPING
    report["paired_training"] = {
        "same_initial_parameter_tensors": True,
        "same_adaptation_rows_and_batch_order": True,
        "same_objective": "normalized activation MSE plus normalized next-logit intervention-effect MSE, unit weights",
        "optimizer": "AdamW",
        "updates": UPDATES,
        "batch_size": runner.TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_norm_clip": GRADIENT_CLIP,
        "only_forward_difference": "GELU versus identity activation",
    }
    report["claim_ceiling"] = CLAIM_CEILING
    report["source_v11_context"] = {
        "packet_sha256": v13.SOURCE_V11_PACKET_SHA256,
        "scientific_outcome": "FAIL",
        "original_comparator_assurance": "INCONCLUSIVE",
    }
    report["source_v13_context"] = {
        "packet_sha256": SOURCE_V13_PACKET_SHA256,
        "scientific_results_sha256": SOURCE_V13_SCIENTIFIC_RESULTS_SHA256,
        "scientific_outcome": "FAIL",
        "reproduction_assurance": "REPRODUCED_LOCAL",
    }
    report["known_limitations"] = [
        "one reused OPT/ReLU checkpoint and six fresh synthetic fixture seeds",
        "layers 0 and 7 were fixed from prior causal-effect failures",
        "the factorized affine comparator is rank limited to 512 and is not a full-rank projection",
        "same-owner local execution is not blind replication or independent scientific acceptance",
    ]
    report["next_blocked_authority_gate"] = "separate blind replication and an authorized independent reviewer are required for independent science; alignment, consciousness, and production gates remain separate"
    ledger_path = custody / "phase-ledger.json"
    ledger = _read_json(ledger_path)
    for phase in ledger["phases"]:
        if phase.get("phase") == "score":
            phase["scientific_outcome"] = report["scientific_outcome"]
    report["phase_ledger"] = ledger
    summary_keys = (
        "scientific_outcome", "primary_comparison", "causal_fit_retention_pass",
        "activation_rescue_pass", "zero_intervention_control_pass", "results", "controls",
    )
    scientific_summary = {key: report[key] for key in summary_keys}
    report["scientific_results_sha256"] = _canonical_digest(scientific_summary)
    report["outcome_report_sha256"] = _canonical_digest({key: value for key, value in report.items() if key not in {"outcome_report_sha256", "created_at_utc"}})
    _write_json(path, report)
    _write_json(ledger_path, ledger)
    return {
        **result,
        "scientific_outcome": report["scientific_outcome"],
        "assurance_level": "INTERNAL_PASS",
        "comparative_mean_difference": comparison["mean_paired_difference"],
        "comparative_lower_95": comparison["lower_95_bound"],
        "affine_fit_retention_pass": affine_retention,
        "gelu_fit_retention_pass": gelu_retention,
        "activation_rescue_pass": rescue_pass,
        "outcome_report_sha256": report["outcome_report_sha256"],
    }


def prepare_replica(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    if (replica / "checkpoint-selection.json").exists():
        raise FileExistsError("V14 replica target binding already exists")
    runner._verify_packet_archive(primary)
    source_selection = v13._checkpoint_manifest(primary)
    runner.archive_packet(replica)
    _write_json(replica / "checkpoint-selection.json", source_selection, immutable=True)
    v13._checkpoint_manifest(replica)
    return {
        "replica_packet_archive_sha256": runner._verify_packet_archive(replica)["archive_sha256"],
        "selection_sha256": source_selection["selection_sha256"],
        "target_tree_sha256": source_selection["source_artifact_tree_sha256"],
    }


def _group_rows(root: Path, relative: str) -> list[dict[str, Any]]:
    base = root / relative
    if not base.is_dir():
        raise FileNotFoundError(f"V14 replication artifact group is missing: {relative}")
    return [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)}
        for path in sorted(item for item in base.rglob("*") if item.is_file())
    ]


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    runner = configure_engine()
    primary = runner._custody_root(primary_path)
    replica = runner._custody_root(replica_path)
    primary_freeze, primary_selection = runner._candidate_freeze(primary)
    replica_freeze, replica_selection = runner._candidate_freeze(replica)
    runner._verify_candidates(primary, primary_freeze)
    runner._verify_candidates(replica, replica_freeze)
    primary_complete = runner._verify_completed_artifacts(primary, primary_freeze)
    replica_complete = runner._verify_completed_artifacts(replica, replica_freeze)
    groups: dict[str, Any] = {}
    for name in ("adaptation", "candidates", "challenge-entropy", "challenges", "fixtures", "outcomes", "predictions"):
        left = _group_rows(primary, name)
        right = _group_rows(replica, name)
        groups[name] = {
            "primary_file_count": len(left),
            "replica_file_count": len(right),
            "primary_tree_sha256": _canonical_digest(left),
            "replica_tree_sha256": _canonical_digest(right),
            "byte_identical": left == right,
        }
    all_groups_match = all(row["byte_identical"] for row in groups.values())
    primary_candidate = v13.stable_candidate_identity(primary_freeze, primary_selection)
    replica_candidate = v13.stable_candidate_identity(replica_freeze, replica_selection)
    challenge_ids = []
    prediction_ids = []
    for root, candidate_id in ((primary, primary_candidate), (replica, replica_candidate)):
        challenge = _read_json(root / "challenge-manifest.json")
        predictions = _read_json(root / "prediction-manifest.json")
        challenge_id = _canonical_digest({
            "candidate_content_sha256": candidate_id,
            "entropy": challenge["entropy"],
            "challenge_rows": challenge["challenge_rows"],
        })
        challenge_ids.append(challenge_id)
        prediction_rows = [
            {key: row[key] for key in ("seed_index", "layer", "method", "control", "sha256") if key in row}
            for row in predictions["predictions"]
        ]
        prediction_ids.append(_canonical_digest({
            "candidate_content_sha256": candidate_id,
            "challenge_content_sha256": challenge_id,
            "predictions": prediction_rows,
        }))
    primary_report = _read_json(primary / "outcome-report.json")
    replica_report = _read_json(replica / "outcome-report.json")
    scientific_ids = [primary_report["scientific_results_sha256"], replica_report["scientific_results_sha256"]]
    runtimes = [primary_report["runtime"], replica_report["runtime"]]
    pins_match = all(runtimes[0].get(key) == runtimes[1].get(key) for key in ("python", "platform", "packages", "dependency_lock_sha256"))
    separate_runtime = runtimes[0].get("python_executable") != runtimes[1].get("python_executable")
    same_owner = primary.stat().st_uid == replica.stat().st_uid
    same_host = runtimes[0].get("host_id_sha256") == runtimes[1].get("host_id_sha256")
    identities_match = (
        primary_candidate == replica_candidate
        and challenge_ids[0] == challenge_ids[1]
        and prediction_ids[0] == prediction_ids[1]
        and scientific_ids[0] == scientific_ids[1]
        and primary_report["scientific_outcome"] == replica_report["scientific_outcome"]
    )
    source_status_match = all(
        report.get("source_v11_context", {}).get("scientific_outcome") == "FAIL"
        and report.get("source_v11_context", {}).get("original_comparator_assurance") == "INCONCLUSIVE"
        and report.get("source_v13_context", {}).get("scientific_outcome") == "FAIL"
        and report.get("source_v13_context", {}).get("reproduction_assurance") == "REPRODUCED_LOCAL"
        for report in (primary_report, replica_report)
    )
    reproduced = all_groups_match and identities_match and pins_match and separate_runtime and same_owner and same_host and source_status_match
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": verify_packet()["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "assurance_scope": "same-owner, same-host replay of paired affine and GELU replacement artifacts across distinct pinned Python runtimes",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPRODUCTION",
        "scientific_outcome": primary_report["scientific_outcome"] if identities_match else "INCONCLUSIVE",
        "claim_ceiling": CLAIM_CEILING,
        "target_identity": {
            "model_id": primary_selection["model_id"],
            "revision": primary_selection["revision"],
            "tree_sha256": primary_selection["source_artifact_tree_sha256"],
            "selection_sha256": primary_selection["selection_sha256"],
        },
        "content_identities": {
            "candidate_sha256": [primary_candidate, replica_candidate],
            "challenge_sha256": challenge_ids,
            "prediction_sha256": prediction_ids,
            "scientific_results_sha256": scientific_ids,
            "candidate_content_match": primary_candidate == replica_candidate,
            "challenge_content_match": challenge_ids[0] == challenge_ids[1],
            "prediction_content_match": prediction_ids[0] == prediction_ids[1],
            "scientific_results_match": scientific_ids[0] == scientific_ids[1],
        },
        "artifact_groups": groups,
        "decision_conditions": {
            "all_artifact_groups_byte_identical": all_groups_match,
            "content_identities_match": identities_match,
            "runtime_versions_and_pins_match": pins_match,
            "separate_python_runtimes": separate_runtime,
            "same_owner_and_host": same_owner and same_host,
            "source_v11_v13_statuses_preserved": source_status_match,
        },
        "runtime_provenance": {
            "primary": {
                "python": runtimes[0]["python"],
                "platform": runtimes[0]["platform"],
                "dependency_lock_sha256": runtimes[0]["dependency_lock_sha256"],
                "python_executable_path_sha256": hashlib.sha256(str(runtimes[0]["python_executable"]).encode()).hexdigest(),
                "host_id_sha256": runtimes[0]["host_id_sha256"],
                "packages": runtimes[0]["packages"],
                "run_budget": primary_complete["runtime"],
            },
            "replica": {
                "python": runtimes[1]["python"],
                "platform": runtimes[1]["platform"],
                "dependency_lock_sha256": runtimes[1]["dependency_lock_sha256"],
                "python_executable_path_sha256": hashlib.sha256(str(runtimes[1]["python_executable"]).encode()).hexdigest(),
                "host_id_sha256": runtimes[1]["host_id_sha256"],
                "packages": runtimes[1]["packages"],
                "run_budget": replica_complete["runtime"],
            },
        },
        "data_and_artifact_custody": "synthetic prompts, checkpoint files, captures, candidates, entropy, predictions, and row-level outcomes remain in external owner custody",
        "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
        "limitations": [
            "one reused OPT/ReLU checkpoint; no fresh-checkpoint generalization",
            "layers 0 and 7 were fixed from prior causal-effect failures",
            "same-owner local replay is not blind replication or independent acceptance",
            "factorized affine replacement is limited to rank 512",
            "no architecture-general, consciousness, alignment, or production claim",
        ],
        "next_blocked_authority_gate": "separate blind replication and an authorized independent reviewer are required for independent science",
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
        result = prepare(args.custody)
    elif args.command == "predict":
        result = predict(args.custody, entropy_source=args.entropy_source)
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
