"""Local-only architecture checks for exact additive-module splitting.

The runner never downloads models. It loads a digest-pinned checkpoint from an
owner-custodied local directory, selects one additive output-projection
channel using visible prompts, and checks two independent split
implementations on held-out prompts. The model weights remain outside this
repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer


STATE_SLICE = "theseus-lab-sandbox-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = "theseus-additive-module-validity-v1"
ALPHA = 0.5
LOCAL_RELATIVE_ERROR_MAX = 1e-12
MODEL_RELATIVE_ERROR_MAX = 1e-5
CUSTODY_ROOT = Path.home() / ".codex/custody/theseus-lab-sandbox-v1-architecture-validity-v1"

FAMILIES = (
    "opt_relu",
    "gpt2_gelu",
    "llama_silu_gated",
    "encoder_decoder_t5",
    "state_space_mamba",
)

VISIBLE_PROMPTS = (
    "A triangle has",
    "Water freezes at",
    "A compass points",
    "A violin has",
)

HELDOUT_PROMPTS = (
    "The planet closest to the sun is",
    "A calendar year has twelve",
    "A bicycle normally has two",
    "The opposite of ancient is",
)


@dataclass(frozen=True)
class CheckpointSpec:
    family: str
    model_id: str
    revision: str
    path: Path
    selected_after_protocol_freeze: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_digest(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or ".cache" in path.parts or path.name.endswith(".lock"):
            continue
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        files.append({"path": relative, "sha256": digest.hexdigest(), "bytes": size})
    return _json_digest(files), files


def _worktree_identity(repo_root: Path) -> dict[str, str]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"revision": revision, "worktree_status_sha256": _sha256(status.encode())}


def _packet_material(repo_root: Path) -> list[dict[str, str]]:
    packet_files = (
        Path("docs/additive-module-theorem.md"),
        Path("additive_module_validity.py"),
        Path("tests/test_additive_module_validity.py"),
        Path("requirements.txt"),
    )
    material = []
    for relative in packet_files:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"protocol packet file is missing: {relative}")
        material.append({"path": relative.as_posix(), "sha256": _sha256(path.read_bytes())})
    return material


def _packet_digest(repo_root: Path) -> str:
    return _json_digest(_packet_material(repo_root))


def _verify_packet_freeze(repo_root: Path) -> str:
    freeze_path = repo_root / "docs/additive-module-validity-v1-freeze.json"
    freeze = json.loads(freeze_path.read_text())
    material = _packet_material(repo_root)
    actual_packet_digest = _json_digest(material)
    if (
        freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("packet_sha256") != actual_packet_digest
        or freeze.get("files") != material
        or freeze.get("freeze_state") != "FROZEN_BEFORE_CHECKPOINT_SELECTION"
    ):
        raise ValueError("the local packet does not match its frozen digest manifest")
    return actual_packet_digest


def _load_manifest(path: Path, custody_root: Path, repo_root: Path) -> list[CheckpointSpec]:
    payload = json.loads(path.read_text())
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint manifest protocol_id does not match")
    entries = payload.get("checkpoints")
    if not isinstance(entries, list):
        raise ValueError("checkpoint manifest must contain a checkpoints list")
    specs: list[CheckpointSpec] = []
    for entry in entries:
        relative = Path(str(entry["path"]))
        if relative.is_absolute():
            raise ValueError("checkpoint manifest paths must be relative to the owner custody root")
        resolved = (custody_root / relative).resolve()
        try:
            resolved.relative_to(custody_root.resolve())
        except ValueError as error:
            raise ValueError("checkpoint manifest path escapes the owner custody root") from error
        try:
            resolved.relative_to(repo_root.resolve())
        except ValueError:
            pass
        else:
            raise ValueError("model assets may not be stored inside the repository")
        specs.append(
        CheckpointSpec(
            family=str(entry["family"]),
            model_id=str(entry["model_id"]),
            revision=str(entry["revision"]),
            path=resolved,
            selected_after_protocol_freeze=entry.get("selected_after_protocol_freeze") is True,
        )
        )
    if {spec.family for spec in specs} != set(FAMILIES) or len(specs) != len(FAMILIES):
        raise ValueError(f"manifest must contain exactly one checkpoint for each family: {FAMILIES}")
    if any(not spec.selected_after_protocol_freeze for spec in specs):
        raise ValueError("all checkpoint identities must be attested as selected after protocol freeze")
    if any(len(spec.revision) != 40 or any(ch not in "0123456789abcdef" for ch in spec.revision.lower()) for spec in specs):
        raise ValueError("each checkpoint revision must be a full 40-character commit SHA")
    if len({(spec.model_id, spec.revision) for spec in specs}) != len(specs):
        raise ValueError("checkpoint identities must be unique")
    return sorted(specs, key=lambda spec: FAMILIES.index(spec.family))


def _projection(model: torch.nn.Module, family: str, layer: int = 0) -> torch.nn.Module:
    if family == "opt_relu":
        return model.model.decoder.layers[layer].fc2
    if family == "gpt2_gelu":
        return model.transformer.h[layer].mlp.c_proj
    if family == "llama_silu_gated":
        return model.model.layers[layer].mlp.down_proj
    if family == "encoder_decoder_t5":
        return model.encoder.block[layer].layer[1].DenseReluDense.wo
    if family == "state_space_mamba":
        return model.backbone.layers[layer].mixer.out_proj
    raise ValueError(f"unsupported model family: {family}")


def _projection_layout(module: torch.nn.Module) -> str:
    if module.__class__.__name__ == "Conv1D":
        return "in_out"
    if isinstance(module, torch.nn.Linear):
        return "out_in"
    raise TypeError(f"unsupported output projection type: {module.__class__.__name__}")


def _config_blockers(config: Any, family: str) -> list[str]:
    blockers: list[str] = []
    model_type = getattr(config, "model_type", None)
    if family == "opt_relu" and (model_type != "opt" or getattr(config, "activation_function", None) != "relu"):
        blockers.append("expected OPT with ReLU activation")
    elif family == "gpt2_gelu" and (
        model_type != "gpt2" or getattr(config, "activation_function", None) not in {"gelu", "gelu_new", "gelu_fast"}
    ):
        blockers.append("expected GPT-2 with GELU-family activation")
    elif family == "llama_silu_gated" and (model_type != "llama" or getattr(config, "hidden_act", None) != "silu"):
        blockers.append("expected Llama-family model with SiLU-gated MLP")
    elif family == "encoder_decoder_t5" and (model_type != "t5" or not getattr(config, "is_encoder_decoder", False)):
        blockers.append("expected T5 encoder-decoder architecture")
    elif family == "state_space_mamba" and model_type != "mamba":
        blockers.append("expected Mamba state-space architecture")
    return blockers


def _model_inputs(tokenizer: Any, model: torch.nn.Module, family: str, prompt: str) -> dict[str, torch.Tensor]:
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    if family == "encoder_decoder_t5":
        start_id = getattr(model.config, "decoder_start_token_id", None)
        if start_id is None:
            start_id = tokenizer.pad_token_id
        if start_id is None:
            raise ValueError("T5 checkpoint has no decoder start or pad token")
        inputs["decoder_input_ids"] = torch.tensor([[int(start_id)]], dtype=torch.long)
    return inputs


def _tokenized_fixture_digest(tokenizer: Any, model: torch.nn.Module, family: str, prompts: tuple[str, ...]) -> str:
    material = []
    for prompt in prompts:
        inputs = _model_inputs(tokenizer, model, family, prompt)
        material.append({
            name: {"shape": list(tensor.shape), "values": tensor.tolist()}
            for name, tensor in sorted(inputs.items())
        })
    return _json_digest(material)


def _forward_logits(model: torch.nn.Module, inputs: dict[str, torch.Tensor]) -> np.ndarray:
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, return_dict=True)
    logits = output.logits[0, -1].detach().cpu().numpy()
    return np.asarray(logits, dtype=np.float64)


def _capture_inputs(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    projection: torch.nn.Module,
    prompts: tuple[str, ...],
) -> list[np.ndarray]:
    captured: list[np.ndarray] = []

    def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        captured.append(args[0].detach().cpu().numpy().copy())

    handle = projection.register_forward_pre_hook(pre_hook)
    try:
        for prompt in prompts:
            _forward_logits(model, _model_inputs(tokenizer, model, family, prompt))
    finally:
        handle.remove()
    if len(captured) != len(prompts):
        raise RuntimeError("selected projection did not execute once per visible prompt")
    return captured


def _select_channel(visible_inputs: list[np.ndarray]) -> tuple[int, dict[str, Any]]:
    flat = np.concatenate([item.reshape(-1, item.shape[-1]) for item in visible_inputs], axis=0)
    active_counts = np.count_nonzero(np.abs(flat) > 1e-8, axis=0)
    magnitude_sums = np.sum(np.abs(flat), axis=0, dtype=np.float64)
    order = sorted(range(flat.shape[1]), key=lambda index: (-int(active_counts[index]), -float(magnitude_sums[index]), index))
    active = [index for index in order if active_counts[index] > 0]
    if not active:
        raise ValueError("visible inputs contain no active additive channels")
    selected = int(active[0])
    return selected, {
        "selection_source": "visible_projection_inputs_only",
        "selected_index": selected,
        "visible_active_token_count": int(active_counts[selected]),
    }


def _weight_vector(weight: torch.Tensor, channel: int, layout: str) -> torch.Tensor:
    return weight[channel, :] if layout == "in_out" else weight[:, channel]


def _torch_split_output(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    channel: int,
    alpha: float,
    layout: str,
) -> torch.Tensor:
    vector = _weight_vector(weight, channel, layout)
    scalar = inputs[..., channel:channel + 1]
    other_weight = weight.clone()
    if layout == "in_out":
        other_weight[channel, :] = 0
        result = inputs @ other_weight
    else:
        other_weight[:, channel] = 0
        result = torch.nn.functional.linear(inputs, other_weight, bias)
    if layout == "in_out" and bias is not None:
        result = result + bias
    child_a = scalar * (alpha * vector)
    child_b = scalar * ((1.0 - alpha) * vector)
    return result + child_a + child_b


def _numpy_split_output(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    channel: int,
    alpha: float,
    layout: str,
) -> np.ndarray:
    inputs64 = np.asarray(inputs, dtype=np.float64)
    weight64 = np.asarray(weight, dtype=np.float64)
    vector = weight64[channel, :] if layout == "in_out" else weight64[:, channel]
    other_weight = weight64.copy()
    if layout == "in_out":
        other_weight[channel, :] = 0
        result = inputs64 @ other_weight
    else:
        other_weight[:, channel] = 0
        result = inputs64 @ other_weight.T
    if bias is not None:
        result = result + np.asarray(bias, dtype=np.float64)
    scalar = inputs64[..., channel:channel + 1]
    child_a = scalar * (alpha * vector)
    child_b = scalar * ((1.0 - alpha) * vector)
    return result + child_a + child_b


def _relative_error(left: np.ndarray, right: np.ndarray) -> float:
    left64 = np.asarray(left, dtype=np.float64)
    right64 = np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(left64 - right64) / max(float(np.linalg.norm(left64)), 1e-12))


def _output_digest(outputs: list[np.ndarray]) -> str:
    flattened = [np.asarray(output, dtype="<f8").reshape(-1) for output in outputs]
    packed = np.concatenate(flattened) if flattened else np.asarray([], dtype="<f8")
    return _sha256(packed.tobytes())


def _run_split_path(
    model: torch.nn.Module,
    tokenizer: Any,
    spec: CheckpointSpec,
    projection: torch.nn.Module,
    channel: int,
    implementation: str,
) -> list[np.ndarray]:
    if implementation not in {"torch", "numpy"}:
        raise ValueError(f"unknown split implementation: {implementation}")
    layout = _projection_layout(projection)
    weight = projection.weight.detach()
    bias = projection.bias.detach() if getattr(projection, "bias", None) is not None else None

    original_forward = projection.forward
    if implementation == "torch":
        def replacement_forward(inputs: torch.Tensor) -> torch.Tensor:
            return _torch_split_output(inputs, weight, bias, channel, ALPHA, layout)
    else:
        def replacement_forward(inputs: torch.Tensor) -> torch.Tensor:
            transformed = _numpy_split_output(
                inputs.detach().cpu().numpy(),
                weight.cpu().numpy(),
                bias.cpu().numpy() if bias is not None else None,
                channel,
                ALPHA,
                layout,
            )
            return torch.as_tensor(transformed, dtype=inputs.dtype, device=inputs.device)

    projection.forward = replacement_forward
    try:
        outputs = [
            _forward_logits(model, _model_inputs(tokenizer, model, spec.family, prompt))
            for prompt in HELDOUT_PROMPTS
        ]
    finally:
        projection.forward = original_forward
    return outputs


def _local_split_error(
    heldout_inputs: list[np.ndarray], weight: torch.Tensor, channel: int, layout: str
) -> float:
    vector = _weight_vector(weight, channel, layout).detach().cpu().numpy().astype(np.float64)
    largest_error = 0.0
    for inputs in heldout_inputs:
        values = np.asarray(inputs, dtype=np.float64)
        scalar = values[..., channel:channel + 1]
        original = scalar * vector
        children = scalar * (ALPHA * vector) + scalar * ((1.0 - ALPHA) * vector)
        largest_error = max(largest_error, _relative_error(original, children))
    return largest_error


def _load_checkpoint(spec: CheckpointSpec) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    if not spec.path.is_dir():
        raise FileNotFoundError(f"checkpoint directory missing for family {spec.family}")
    tree_digest, artifacts = _tree_digest(spec.path)
    config_path = spec.path / "config.json"
    config_digest = _sha256(config_path.read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(spec.path, local_files_only=True, trust_remote_code=False)
    model_cls = AutoModelForSeq2SeqLM if spec.family == "encoder_decoder_t5" else AutoModelForCausalLM
    has_safetensors = any(path.suffix == ".safetensors" for path in spec.path.glob("**/*") if path.is_file())
    model = model_cls.from_pretrained(
        spec.path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=has_safetensors,
        torch_dtype=torch.float32,
    )
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    blockers = _config_blockers(model.config, spec.family)
    if blockers:
        raise ValueError("; ".join(blockers))
    return model, tokenizer, {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "checkpoint_tree_sha256": tree_digest,
        "config_sha256": config_digest,
        "artifact_count": len(artifacts),
        "artifact_manifest_sha256": _json_digest(artifacts),
        "selected_after_protocol_freeze": spec.selected_after_protocol_freeze,
        "local_files_only": True,
        "trust_remote_code": False,
        "precision": "float32",
        "custody_path_disclosed": False,
    }


def _run_one(spec: CheckpointSpec) -> dict[str, Any]:
    model, tokenizer, checkpoint = _load_checkpoint(spec)
    projection = _projection(model, spec.family)
    layout = _projection_layout(projection)
    visible_inputs = _capture_inputs(model, tokenizer, spec.family, projection, VISIBLE_PROMPTS)
    channel, selection = _select_channel(visible_inputs)
    heldout_inputs = _capture_inputs(model, tokenizer, spec.family, projection, HELDOUT_PROMPTS)
    baseline = [_forward_logits(model, _model_inputs(tokenizer, model, spec.family, prompt)) for prompt in HELDOUT_PROMPTS]

    torch_outputs = _run_split_path(model, tokenizer, spec, projection, channel, "torch")
    numpy_outputs = _run_split_path(model, tokenizer, spec, projection, channel, "numpy")
    local_error = _local_split_error(heldout_inputs, projection.weight, channel, layout)
    torch_errors = [_relative_error(before, after) for before, after in zip(baseline, torch_outputs)]
    numpy_errors = [_relative_error(before, after) for before, after in zip(baseline, numpy_outputs)]
    cross_errors = [_relative_error(left, right) for left, right in zip(torch_outputs, numpy_outputs)]
    top1_torch = [int(np.argmax(before) == np.argmax(after)) for before, after in zip(baseline, torch_outputs)]
    top1_numpy = [int(np.argmax(before) == np.argmax(after)) for before, after in zip(baseline, numpy_outputs)]
    metrics = {
        "heldout_prompt_count": len(HELDOUT_PROMPTS),
        "local_split_relative_error_max": local_error,
        "torch_full_model_relative_error_max": max(torch_errors, default=0.0),
        "numpy_full_model_relative_error_max": max(numpy_errors, default=0.0),
        "torch_vs_numpy_relative_error_max": max(cross_errors, default=0.0),
        "torch_top1_agreement": float(np.mean(top1_torch)),
        "numpy_top1_agreement": float(np.mean(top1_numpy)),
    }
    passed = bool(
        local_error <= LOCAL_RELATIVE_ERROR_MAX
        and metrics["torch_full_model_relative_error_max"] <= MODEL_RELATIVE_ERROR_MAX
        and metrics["numpy_full_model_relative_error_max"] <= MODEL_RELATIVE_ERROR_MAX
        and metrics["torch_vs_numpy_relative_error_max"] <= MODEL_RELATIVE_ERROR_MAX
    )
    return {
        "family": spec.family,
        "checkpoint": checkpoint,
        "projection_layout": layout,
        "selected_projection": projection.__class__.__name__,
        "selection": selection,
        "mechanism_lineage": {
            "parent_component": f"{spec.family}:layer0:channel{channel}",
            "child_components": [
                f"{spec.family}:layer0:channel{channel}:alpha",
                f"{spec.family}:layer0:channel{channel}:one_minus_alpha",
            ],
            "parent_active_contribution_removed": True,
            "component_count_delta": 1,
            "checkpoint_parameters_mutated": False,
            "runtime_projection_forward_replaced": True,
        },
        "visible_fixture_sha256": _json_digest(VISIBLE_PROMPTS),
        "heldout_fixture_sha256": _json_digest(HELDOUT_PROMPTS),
        "tokenized_visible_fixture_sha256": _tokenized_fixture_digest(tokenizer, model, spec.family, VISIBLE_PROMPTS),
        "tokenized_heldout_fixture_sha256": _tokenized_fixture_digest(tokenizer, model, spec.family, HELDOUT_PROMPTS),
        "alpha": ALPHA,
        "thresholds": {
            "local_relative_error_max": LOCAL_RELATIVE_ERROR_MAX,
            "full_model_relative_error_max": MODEL_RELATIVE_ERROR_MAX,
            "top1_agreement": "reported_diagnostic_only",
        },
        "metrics": metrics,
        "behavior_diverged": bool(max(metrics["torch_full_model_relative_error_max"], metrics["numpy_full_model_relative_error_max"]) > MODEL_RELATIVE_ERROR_MAX),
        "mechanism_diverged": True,
        "heldout_output_digests": {
            "baseline": _output_digest(baseline),
            "torch_split": _output_digest(torch_outputs),
            "numpy_split": _output_digest(numpy_outputs),
        },
        "status": "PASS" if passed else "INCONCLUSIVE",
    }


def _runtime_metadata() -> dict[str, Any]:
    distributions = sorted(
        (distribution.metadata.get("Name", ""), distribution.version)
        for distribution in importlib.metadata.distributions()
    )
    identity = {
        "executable": sys.executable,
        "prefix": sys.prefix,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "numpy": np.__version__,
        "distributions": distributions,
    }
    return {
        "python": identity["python"],
        "platform": identity["platform"],
        "torch": identity["torch"],
        "transformers": identity["transformers"],
        "numpy": identity["numpy"],
        "device": "cpu",
        "dependency_manifest_sha256": _json_digest(distributions),
        "runtime_identity_sha256": _json_digest(identity),
    }


def _runtime_blockers(repo_root: Path) -> list[dict[str, str]]:
    blockers = []
    for line in (repo_root / "requirements.txt").read_text().splitlines():
        requirement = line.strip()
        if not requirement or requirement.startswith("#") or "==" not in requirement:
            continue
        name, expected = requirement.split("==", 1)
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "MISSING"
        if actual != expected:
            blockers.append({"package": name, "expected": expected, "actual": actual})
    return blockers


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def run(manifest_path: Path, repo_root: Path, custody_root: Path = CUSTODY_ROOT) -> dict[str, Any]:
    try:
        manifest_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("checkpoint manifest must remain outside the repository")
    packet_digest = _verify_packet_freeze(repo_root)
    specs = _load_manifest(manifest_path, custody_root, repo_root)
    started = time.perf_counter()
    runtime = _runtime_metadata()
    runtime_blockers = _runtime_blockers(repo_root)
    reports: list[dict[str, Any]] = []
    for spec in specs:
        if runtime_blockers:
            reports.append({
                "family": spec.family,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "status": "INCONCLUSIVE",
                "error_detail": "runtime does not match the frozen requirements",
                "custody_path_disclosed": False,
            })
            continue
        try:
            reports.append(_run_one(spec))
        except Exception as error:
            reports.append({
                "family": spec.family,
                "model_id": spec.model_id,
                "revision": spec.revision,
                "status": "INCONCLUSIVE",
                "error_type": error.__class__.__name__,
                "error_detail": "checkpoint validation or local model execution failed; inspect owner-side diagnostics",
                "custody_path_disclosed": False,
            })
    all_passed = len(reports) == len(FAMILIES) and all(item.get("status") == "PASS" for item in reports)
    return {
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet_digest,
        "source": _worktree_identity(repo_root),
        "execution_id": str(uuid.uuid4()),
        "runtime": runtime,
        "runtime_blockers": runtime_blockers,
        "resource_use": {
            "wall_seconds": round(time.perf_counter() - started, 6),
            "process_peak_rss_bytes": _peak_rss_bytes(),
            "accelerator_hours": 0.0,
            "energy_joules": None,
            "energy_measurement_available": False,
            "paid_execution": False,
            "external_provider_called": False,
        },
        "assurance_level": "INTERNAL_PASS" if all_passed else "INCONCLUSIVE",
        "reviewer_status": "NOT_PERFORMED",
        "reproducibility_status": "NOT_REPRODUCED",
        "claim_ceiling": "LOCAL_ADDITIVE_MODULE_ADAPTER_VALIDATION_ONLY",
        "data_artifact_custody": "checkpoint bytes remain in the external owner-custody root; no prompt data or model weights are written to this repository",
        "known_limitations": [
            "five finite architecture strata and four held-out prompts per checkpoint",
            "the theorem is exact over field arithmetic; model outputs are compared in float32",
            "checkpoint freshness is an owner-custodian attestation",
            "synthetic adapter smoke checks are not checkpoint replication",
        ],
        "nonclaims": [
            "all architectures",
            "independent replacement learning",
            "mechanistic understanding",
            "consciousness",
            "alignment validation",
            "independent science",
            "production readiness",
        ],
        "next_blocked_authority_gate": "clean second-runtime reproduction followed by SEGREGATED_INTERNAL_REVIEW",
        "controls": {
            "baseline": "unmodified checkpoint output on the held-out prompt set",
            "independent_implementation": "NumPy full projection replacement compared with the PyTorch replacement",
            "negative_controls": "not used; this is an exact algebra/adapter validity check, not a mechanistic-learning experiment",
        },
        "status": "INTERNAL_PASS" if all_passed else "INCONCLUSIVE",
        "reports": reports,
    }


def compare_reproductions(primary: dict[str, Any], replica: dict[str, Any]) -> dict[str, Any]:
    blockers: list[str] = []
    for key in ("state_slice", "protocol_id", "packet_sha256"):
        if primary.get(key) != replica.get(key):
            blockers.append(f"{key}_mismatch")
    if primary.get("source") != replica.get("source"):
        blockers.append("source_identity_mismatch")
    primary_dependencies = primary.get("runtime", {}).get("dependency_manifest_sha256")
    replica_dependencies = replica.get("runtime", {}).get("dependency_manifest_sha256")
    if not primary_dependencies or primary_dependencies != replica_dependencies:
        blockers.append("dependency_manifest_mismatch")
    primary_runtime = primary.get("runtime", {}).get("runtime_identity_sha256")
    replica_runtime = replica.get("runtime", {}).get("runtime_identity_sha256")
    if not primary_runtime or not replica_runtime or primary_runtime == replica_runtime:
        blockers.append("runtime_not_distinct")
    if primary.get("assurance_level") != "INTERNAL_PASS" or replica.get("assurance_level") != "INTERNAL_PASS":
        blockers.append("one_or_both_runs_not_internal_pass")
    if primary.get("runtime_blockers") or replica.get("runtime_blockers"):
        blockers.append("runtime_requirements_mismatch")

    primary_cases = {case.get("family"): case for case in primary.get("reports", [])}
    replica_cases = {case.get("family"): case for case in replica.get("reports", [])}
    if set(primary_cases) != set(FAMILIES) or set(replica_cases) != set(FAMILIES):
        blockers.append("model_class_coverage_mismatch")
    case_comparisons: list[dict[str, Any]] = []
    for family in FAMILIES:
        left = primary_cases.get(family, {})
        right = replica_cases.get(family, {})
        identity_fields = (
            "family",
            "checkpoint",
            "selection",
            "visible_fixture_sha256",
            "heldout_fixture_sha256",
            "tokenized_visible_fixture_sha256",
            "tokenized_heldout_fixture_sha256",
            "thresholds",
        )
        same_identity = all(left.get(field) == right.get(field) for field in identity_fields)
        if not same_identity:
            blockers.append(f"{family}_target_identity_mismatch")
        numeric_deltas: dict[str, float] = {}
        left_metrics = left.get("metrics", {})
        right_metrics = right.get("metrics", {})
        for metric in (
            "local_split_relative_error_max",
            "torch_full_model_relative_error_max",
            "numpy_full_model_relative_error_max",
            "torch_vs_numpy_relative_error_max",
        ):
            if metric not in left_metrics or metric not in right_metrics:
                blockers.append(f"{family}_{metric}_missing")
                continue
            numeric_deltas[metric] = abs(float(left_metrics[metric]) - float(right_metrics[metric]))
            if numeric_deltas[metric] > MODEL_RELATIVE_ERROR_MAX:
                blockers.append(f"{family}_{metric}_reproduction_delta_exceeded")
        if left.get("status") != "PASS" or right.get("status") != "PASS":
            blockers.append(f"{family}_behavior_gate_mismatch")
        metrics_match = all(delta <= MODEL_RELATIVE_ERROR_MAX for delta in numeric_deltas.values())
        case_passed = bool(
            same_identity and metrics_match
            and left.get("status") == "PASS" and right.get("status") == "PASS"
        )
        case_comparisons.append({
            "family": family,
            "same_frozen_identity": same_identity,
            "metrics_within_reproduction_tolerance": metrics_match,
            "metrics_deltas": numeric_deltas,
            "output_digests_match": left.get("heldout_output_digests") == right.get("heldout_output_digests"),
            "primary_output_digests": left.get("heldout_output_digests"),
            "replica_output_digests": right.get("heldout_output_digests"),
            "status": "PASS" if case_passed else "INCONCLUSIVE",
        })

    reproduced = not blockers
    return {
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": primary.get("packet_sha256"),
        "source": primary.get("source"),
        "primary_execution_id": primary.get("execution_id"),
        "replica_execution_id": replica.get("execution_id"),
        "primary_runtime_identity_sha256": primary_runtime,
        "replica_runtime_identity_sha256": replica_runtime,
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_PERFORMED",
        "reproducibility_status": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_ADDITIVE_MODULE_ADAPTER_VALIDATION_ONLY",
        "data_artifact_custody": "checkpoint bytes remain in external owner custody; this comparison contains only digests and aggregate metrics",
        "known_limitations": [
            "reproduction is same-owner local reproduction, not independent review",
            "finite architecture coverage does not justify all architectures",
        ],
        "controls": "same unmodified checkpoint baseline and two independent split implementations",
        "nonclaims": ["all architectures", "independent scientific acceptance", "mechanistic understanding", "consciousness", "alignment", "production readiness"],
        "next_blocked_authority_gate": "SEGREGATED_INTERNAL_REVIEW by a separate blind reviewer",
        "blockers": blockers,
        "cases": case_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="owner-custodied JSON manifest of five local checkpoints")
    parser.add_argument("--compare-first", type=Path, help="first run report for reproduction comparison")
    parser.add_argument("--compare-second", type=Path, help="clean second-runtime report for reproduction comparison")
    parser.add_argument("--custody-root", type=Path, default=CUSTODY_ROOT)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--report", type=Path, default=None, help="optional aggregate-safe report path outside the repository")
    args = parser.parse_args()
    if args.report is not None:
        try:
            args.report.resolve().relative_to(args.repo_root.resolve())
        except ValueError:
            pass
        else:
            parser.error("reports must be persisted outside the repository")
    if args.compare_first or args.compare_second:
        if not args.compare_first or not args.compare_second or args.manifest:
            parser.error("provide both comparison reports and omit --manifest")
        result = compare_reproductions(
            json.loads(args.compare_first.read_text()),
            json.loads(args.compare_second.read_text()),
        )
    else:
        if not args.manifest:
            parser.error("--manifest is required for a checkpoint run")
        result = run(args.manifest, args.repo_root.resolve(), args.custody_root.resolve())
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print(encoded)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    return 0 if result["assurance_level"] == "INTERNAL_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
