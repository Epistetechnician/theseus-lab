"""Frozen local capture, replacement, prediction, and evaluation runner.

State slice: theseus-lab-sandbox-v1.

All model files, prompts, captures, prediction arrays, and row-level outcomes
must remain under an owner-approved external custody root. Standard output is
restricted to aggregate receipts and digests.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from replacement_loss_v3 import (
    STATE_SLICE,
    bootstrap_mean_difference_lower_bound,
    causal_effect_metrics,
    fit_independent_linear_replacement,
)
from replacement_loss_v3_fixtures import (
    CHALLENGES_PER_STRATUM,
    PROTOCOL_ID,
    STRATA,
    VISIBLE_COUNT,
    fixture_payload,
    generate_adaptation_fixture,
    generate_challenge_fixture,
    generate_visible_fixture,
    validate_partition_disjointness,
)


ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PACKET_FREEZE = "docs/replacement-loss-v3-freeze.json"
FAMILIES = ("opt_relu", "gpt2_gelu", "llama_silu_gated")
SEEDS = (20260924, 20260925, 20260926)
MAX_PARAMETER_COUNT = 500_000_000
PRIOR_CHECKPOINT_IDS = {
    "gpt2",
    "distilgpt2",
    "openai-community/gpt2",
    "openai-community/distilgpt2",
    "openai-community/gpt2-large",
    "gpt2-large",
    "facebook/opt-125m",
    "opt-125m",
    "facebook/opt-1.3b",
    "tinyllama/tinyllama-1.1b-chat-v1.0",
    "huggingfacetb/smollm-135m",
    "huggingfacetb/smollm2-135m",
}
MIN_ADAPTATION_ROWS = 512
RIDGE = 1e-6
BATCH_SIZE = 8
EQUIVALENCE_MAX_ABS_ERROR = 1e-4
PACKET_FILES = (
    "docs/replacement-loss-v3.md",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
    "tests/test_replacement_loss_v3.py",
    "tests/test_replacement_loss_v3_runner.py",
    "requirements.txt",
)


@dataclass(frozen=True)
class Checkpoint:
    family: str
    model_id: str
    revision: str
    local_path: Path
    selected_after_protocol_freeze: bool


class _ProjectionInputsCaptured(Exception):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_digest(value: Any) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock") or ".cache" in path.parts:
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
        rows.append({"path": path.relative_to(root).as_posix(), "sha256": digest.hexdigest(), "bytes": size})
    return _canonical_digest(rows), rows


def _custody_storage_bytes(root: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.name.endswith(".lock"):
            file_count += 1
            total_bytes += path.stat().st_size
    return {"file_count": file_count, "total_bytes": total_bytes}


def _save_npz_deterministic(path: Path, **arrays: np.ndarray) -> None:
    """Write stable NPZ bytes so a clean rerun can compare artifact digests."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in sorted(arrays.items()):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path.name}")
    return payload


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _custody_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    try:
        root.relative_to(_repo_root())
    except ValueError:
        return root
    raise ValueError("custody root must be outside the repository")


def _custodied_manifest_path(path: str | Path, custody: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(custody.resolve())
    except ValueError as error:
        raise ValueError("checkpoint manifest must be inside external custody") from error
    return resolved


def freeze_packet(repo_root: Path | None = None) -> dict[str, Any]:
    """Digest the exact implementation and preregistration bytes before model selection."""

    repo = (repo_root or _repo_root()).resolve()
    material: list[dict[str, str]] = []
    for relative in PACKET_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen packet input is missing: {relative}")
        material.append({"path": relative, "sha256": _sha256(path.read_bytes())})
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_CHECKPOINT_SELECTION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packet_files": material,
        "packet_sha256": _canonical_digest(material),
        "checkpoint_identities_selected": False,
        "challenge_fixtures_generated": False,
    }
    destination = repo / PACKET_FREEZE
    if destination.exists():
        raise FileExistsError("packet freeze is immutable; create a new protocol version to change it")
    _write_json(destination, payload)
    return payload


def archive_frozen_packet(custody_root: str | Path) -> dict[str, Any]:
    """Copy the exact digest-listed protocol bytes into external custody."""

    custody = _custody_root(custody_root)
    freeze = verify_packet()
    destination = custody / "frozen-packet"
    if destination.exists():
        raise FileExistsError("frozen packet archive is immutable")
    rows: list[dict[str, str]] = []
    for relative in (*PACKET_FILES, PACKET_FREEZE):
        source = _repo_root() / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "sha256": _sha256(target.read_bytes())})
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "archive_files": rows,
        "archive_sha256": _canonical_digest(rows),
    }
    _write_json(custody / "frozen-packet-archive.json", payload)
    return {"packet_sha256": payload["packet_sha256"], "archive_sha256": payload["archive_sha256"]}


def verify_packet(repo_root: Path | None = None) -> dict[str, Any]:
    repo = (repo_root or _repo_root()).resolve()
    freeze_path = repo / PACKET_FREEZE
    freeze = _load_json(freeze_path)
    material = [
        {"path": relative, "sha256": _sha256((repo / relative).read_bytes())}
        for relative in PACKET_FILES
    ]
    if (
        freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("state_slice") != STATE_SLICE
        or freeze.get("freeze_state") != "FROZEN_BEFORE_CHECKPOINT_SELECTION"
        or freeze.get("packet_files") != material
        or freeze.get("packet_sha256") != _canonical_digest(material)
        or freeze.get("checkpoint_identities_selected") is not False
        or freeze.get("challenge_fixtures_generated") is not False
    ):
        raise ValueError("source or preregistration bytes do not match the preselection freeze")
    return freeze


def _load_checkpoints(manifest_path: Path, custody: Path) -> list[Checkpoint]:
    payload = _load_json(manifest_path)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("state_slice") != STATE_SLICE:
        raise ValueError("checkpoint manifest protocol or state slice mismatch")
    if payload.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("checkpoint manifest does not bind the frozen packet")
    selected_at = payload.get("selected_at_utc")
    if not isinstance(selected_at, str):
        raise ValueError("checkpoint manifest must record a post-freeze selection time")
    selected_datetime = datetime.fromisoformat(selected_at)
    freeze_datetime = datetime.fromisoformat(verify_packet()["created_at_utc"])
    if (
        selected_datetime.tzinfo is None
        or selected_datetime.utcoffset() is None
        or freeze_datetime.tzinfo is None
        or freeze_datetime.utcoffset() is None
        or selected_datetime <= freeze_datetime
    ):
        raise ValueError("checkpoint identities must be selected after the frozen packet")
    entries = payload.get("checkpoints")
    if not isinstance(entries, list) or len(entries) != len(FAMILIES):
        raise ValueError("checkpoint manifest must contain exactly three entries")
    checkpoints: list[Checkpoint] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("checkpoint entries must be objects")
        family = str(entry.get("family"))
        revision = str(entry.get("revision", ""))
        model_id = str(entry.get("model_id", ""))
        local = Path(str(entry.get("local_path", "")))
        if family not in FAMILIES or not model_id:
            raise ValueError("unsupported or missing checkpoint family/model ID")
        if model_id.strip().lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError("checkpoint was used in prior Theseus-lab experiments and is not fresh")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
            raise ValueError("checkpoint revision must be a full 40-character commit SHA")
        if entry.get("not_used_for_protocol_design") is not True:
            raise ValueError("checkpoint must be attested as unseen during protocol design")
        if entry.get("base_model") is not True:
            raise ValueError("checkpoint must be a public base model, not an instruction-tuned derivative")
        if local.is_absolute():
            raise ValueError("checkpoint paths must be relative to external custody")
        resolved = (custody / local).resolve()
        try:
            resolved.relative_to(custody.resolve())
        except ValueError as error:
            raise ValueError("checkpoint path escapes the custody root") from error
        try:
            resolved.relative_to(_repo_root())
        except ValueError:
            pass
        else:
            raise ValueError("model assets may not be stored in the repository")
        checkpoints.append(
            Checkpoint(
                family=family,
                model_id=model_id,
                revision=revision,
                local_path=resolved,
                selected_after_protocol_freeze=entry.get("selected_after_protocol_freeze") is True,
            )
        )
    if {item.family for item in checkpoints} != set(FAMILIES):
        raise ValueError("checkpoint manifest must cover OPT, GPT-2, and Llama families exactly once")
    if any(not item.selected_after_protocol_freeze for item in checkpoints):
        raise ValueError("each checkpoint must be attested as selected after packet freeze")
    if len({(item.model_id, item.revision) for item in checkpoints}) != len(FAMILIES):
        raise ValueError("checkpoint identities must be unique")
    return sorted(checkpoints, key=lambda item: FAMILIES.index(item.family))


def _projection(model: torch.nn.Module, family: str, layer: int) -> torch.nn.Module:
    if family == "opt_relu":
        return model.model.decoder.layers[layer].fc2
    if family == "gpt2_gelu":
        return model.transformer.h[layer].mlp.c_proj
    if family == "llama_silu_gated":
        return model.model.layers[layer].mlp.down_proj
    raise ValueError(f"unsupported family: {family}")


def _projection_parent(model: torch.nn.Module, family: str, layer: int) -> tuple[Any, str]:
    if family == "opt_relu":
        return model.model.decoder.layers[layer], "fc2"
    if family == "gpt2_gelu":
        return model.transformer.h[layer].mlp, "c_proj"
    if family == "llama_silu_gated":
        return model.model.layers[layer].mlp, "down_proj"
    raise ValueError(f"unsupported family: {family}")


def _set_projection(model: torch.nn.Module, family: str, layer: int, module: torch.nn.Module) -> None:
    parent, name = _projection_parent(model, family, layer)
    setattr(parent, name, module)


def _layer_count(model: torch.nn.Module, family: str) -> int:
    if family == "opt_relu":
        return len(model.model.decoder.layers)
    if family == "gpt2_gelu":
        return len(model.transformer.h)
    if family == "llama_silu_gated":
        return len(model.model.layers)
    raise ValueError(f"unsupported family: {family}")


def _config_blockers(config: Any, family: str) -> list[str]:
    if family == "opt_relu":
        good = getattr(config, "model_type", None) == "opt" and getattr(config, "activation_function", None) == "relu"
    elif family == "gpt2_gelu":
        good = getattr(config, "model_type", None) == "gpt2" and getattr(config, "activation_function", None) in {
            "gelu", "gelu_new", "gelu_fast"
        }
    elif family == "llama_silu_gated":
        good = (
            getattr(config, "model_type", None) == "llama"
            and getattr(config, "hidden_act", None) == "silu"
        )
    else:
        return ["unsupported family"]
    return [] if good else [f"config does not satisfy frozen {family} architecture contract"]


def _load_local_model(checkpoint: Checkpoint) -> tuple[torch.nn.Module, Any]:
    if not checkpoint.local_path.is_dir():
        raise FileNotFoundError(f"custodied checkpoint directory missing for {checkpoint.family}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint.local_path, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer has no pad or eos token for deterministic batch padding")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint.local_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    blockers = _config_blockers(model.config, checkpoint.family)
    if blockers:
        raise ValueError("; ".join(blockers))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > MAX_PARAMETER_COUNT:
        raise ValueError("checkpoint exceeds the frozen 500-million-parameter ceiling")
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    return model, tokenizer


def _token_rows(tokenizer: Any, records: list[dict[str, Any]], with_continuation: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt_ids = tokenizer(record["input_text"], add_special_tokens=True)["input_ids"]
        continuation_ids = (
            tokenizer(record["continuation"], add_special_tokens=False)["input_ids"]
            if with_continuation
            else []
        )
        if not prompt_ids or (with_continuation and not continuation_ids):
            raise ValueError("all prompts and challenge continuations must tokenize to non-empty sequences")
        rows.append(
            {
                **record,
                "ids": [int(item) for item in prompt_ids + continuation_ids],
                "prompt_len": len(prompt_ids),
                "continuation_ids": [int(item) for item in continuation_ids],
            }
        )
    return rows


def _batches(rows: list[dict[str, Any]], pad_id: int, batch_size: int = BATCH_SIZE) -> Iterable[tuple[list[dict[str, Any]], dict[str, torch.Tensor]]]:
    for start in range(0, len(rows), batch_size):
        group = rows[start : start + batch_size]
        max_length = max(len(row["ids"]) for row in group)
        ids = torch.full((len(group), max_length), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(group), max_length), dtype=torch.long)
        for index, row in enumerate(group):
            length = len(row["ids"])
            ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long)
            mask[index, :length] = 1
        yield group, {"input_ids": ids, "attention_mask": mask}


def _activation_bth(activation: torch.Tensor, batch: int, sequence: int) -> torch.Tensor:
    if activation.ndim == 3 and tuple(activation.shape[:2]) == (batch, sequence):
        return activation
    if activation.ndim == 2 and activation.shape[0] == batch * sequence:
        return activation.reshape(batch, sequence, activation.shape[-1])
    raise ValueError(
        f"unsupported projection activation layout {tuple(activation.shape)} for batch={batch}, sequence={sequence}"
    )


def _run_projection(
    model: torch.nn.Module,
    family: str,
    layer: int,
    inputs: dict[str, torch.Tensor],
    positions: list[int] | None = None,
    deltas: torch.Tensor | None = None,
    abort_after_input: bool = False,
    input_permutation: list[int] | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None]:
    module = _projection(model, family, layer)
    input_capture: list[torch.Tensor] = []
    output_capture: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        activation = args[0]
        batch_size, sequence_length = inputs["input_ids"].shape
        activation_bth = _activation_bth(activation, batch_size, sequence_length)
        if positions is None:
            input_capture.append(activation_bth.detach())
        else:
            input_capture.append(torch.stack([activation_bth[i, pos].detach() for i, pos in enumerate(positions)]))
        if abort_after_input:
            raise _ProjectionInputsCaptured()
        if deltas is None and input_permutation is None:
            return None
        changed = activation_bth
        permutation_tensor = None
        if input_permutation is not None:
            permutation_tensor = torch.as_tensor(input_permutation, device=changed.device)
            changed = changed.index_select(-1, permutation_tensor)
        if deltas is not None:
            changed = changed.clone()
            local_deltas = deltas
            if permutation_tensor is not None:
                local_deltas = local_deltas.index_select(-1, permutation_tensor.cpu())
            for index, position in enumerate(positions or []):
                changed[index, position] += local_deltas[index].to(device=changed.device, dtype=changed.dtype)
        if activation.ndim == 2:
            changed = changed.reshape_as(activation)
        return (changed, *args[1:])

    def post_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        batch_size, sequence_length = inputs["input_ids"].shape
        tensor_bth = _activation_bth(tensor, batch_size, sequence_length)
        if positions is None:
            output_capture.append(tensor_bth.detach())
        else:
            output_capture.append(torch.stack([tensor_bth[i, pos].detach() for i, pos in enumerate(positions)]))

    pre_handle = module.register_forward_pre_hook(pre_hook)
    post_handle = module.register_forward_hook(post_hook)
    logits: torch.Tensor | None = None
    try:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False, return_dict=True).logits.detach()
    except _ProjectionInputsCaptured:
        pass
    finally:
        pre_handle.remove()
        post_handle.remove()
    if not input_capture:
        raise RuntimeError("target projection pre-hook was not called")
    if not abort_after_input and not output_capture:
        raise RuntimeError("target projection output hook was not called")
    return logits, input_capture[0], output_capture[0] if output_capture else None


def _capture_all_layer_inputs(
    model: torch.nn.Module,
    family: str,
    inputs: dict[str, torch.Tensor],
) -> list[torch.Tensor]:
    captures: list[torch.Tensor | None] = [None] * _layer_count(model, family)
    handles = []
    for layer in range(len(captures)):
        projection = _projection(model, family, layer)

        def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, index: int = layer) -> None:
            batch, sequence = inputs["input_ids"].shape
            captures[index] = _activation_bth(args[0], batch, sequence).detach()

        handles.append(projection.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            model(**inputs, use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    mask = inputs["attention_mask"].bool()
    return [item[mask] for item in captures if item is not None]


def _module_dimensions(module: torch.nn.Module) -> tuple[int, int, bool]:
    if module.__class__.__name__ == "Conv1D":
        return int(module.weight.shape[0]), int(module.weight.shape[1]), module.bias is not None
    if isinstance(module, torch.nn.Linear):
        return int(module.in_features), int(module.out_features), module.bias is not None
    raise TypeError(f"unsupported projection module {module.__class__.__name__}")


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_linear(path: Path, module: torch.nn.Linear) -> str:
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return _state_digest(state)


def _load_linear(path: Path, in_features: int, out_features: int, use_bias: bool) -> torch.nn.Linear:
    module = torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32)
    module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    module.eval()
    return module


def _make_variant_modules(
    treatment: torch.nn.Linear,
    permutation_seed: int,
) -> tuple[dict[str, torch.nn.Linear], list[int], float]:
    generator = np.random.default_rng(permutation_seed)
    permutation = generator.permutation(treatment.in_features).astype(np.int64)
    raw = torch.nn.Linear(
        treatment.in_features,
        treatment.out_features,
        bias=treatment.bias is not None,
        dtype=treatment.weight.dtype,
    )
    with torch.no_grad():
        raw.weight.copy_(treatment.weight[:, torch.as_tensor(permutation)])
        if raw.bias is not None:
            raw.bias.copy_(treatment.bias)
    aligned_input = torch.from_numpy(
        generator.normal(size=(5, treatment.in_features)).astype(np.float32)
    ).to(dtype=treatment.weight.dtype)
    permuted_input = aligned_input[:, torch.as_tensor(permutation)]
    aligned_weight = treatment.weight[:, torch.as_tensor(permutation)]
    aligned_output = F.linear(permuted_input, aligned_weight, treatment.bias)
    original_output = treatment(aligned_input)
    equiv_error = float(torch.max(torch.abs(aligned_output - original_output)).item())
    return {"raw_index_permutation": raw}, permutation.tolist(), equiv_error


def _selected_positions(group: list[dict[str, Any]]) -> list[int]:
    return [int(row["prompt_len"] - 1) for row in group]


def _continuation_cross_entropy(logits: torch.Tensor, group: list[dict[str, Any]]) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(group):
        labels = row["continuation_ids"]
        start = int(row["prompt_len"] - 1)
        count = len(labels)
        row_logits = logits[index, start : start + count].float()
        target = torch.tensor(labels, dtype=torch.long, device=row_logits.device)
        values.append(float(F.cross_entropy(row_logits, target, reduction="mean").item()))
    return np.asarray(values, dtype=np.float64)


def _next_logits(logits: torch.Tensor, group: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([logits[index, int(row["prompt_len"] - 1)].detach().cpu() for index, row in enumerate(group)])


def _capture_visible_layer(model: torch.nn.Module, tokenizer: Any, family: str, records: list[dict[str, Any]]) -> tuple[int, list[float]]:
    encoded = _token_rows(tokenizer, records, with_continuation=False)
    sums: list[list[torch.Tensor]] = [[] for _ in range(_layer_count(model, family))]
    for _group, inputs in _batches(encoded, tokenizer.pad_token_id):
        activations = _capture_all_layer_inputs(model, family, inputs)
        for layer, activation in enumerate(activations):
            sums[layer].append(activation.float())
    if len(records) < VISIBLE_COUNT:
        raise ValueError("visible fixture has fewer examples than predeclared")
    variances = [float(torch.cat(parts, dim=0).var(dim=0, unbiased=False).mean().item()) for parts in sums]
    selected = max(range(len(variances)), key=lambda index: (variances[index], -index))
    return selected, variances


def _fit_case(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint: Checkpoint,
    layer: int,
    seed_index: int,
    fixture_records: list[dict[str, Any]],
    custody: Path,
) -> dict[str, Any]:
    fit_started = time.monotonic()
    adjacent = layer + 1 if layer + 1 < _layer_count(model, checkpoint.family) else layer - 1
    if adjacent < 0 or adjacent == layer:
        raise ValueError("wrong-module control requires at least two MLP projections")
    encoded = _token_rows(tokenizer, fixture_records, with_continuation=False)
    target_x: list[torch.Tensor] = []
    target_y: list[torch.Tensor] = []
    adjacent_x: list[torch.Tensor] = []
    adjacent_y: list[torch.Tensor] = []
    target_projection = _projection(model, checkpoint.family, layer)
    adjacent_projection = _projection(model, checkpoint.family, adjacent)
    captured: dict[str, list[torch.Tensor]] = {"target_x": [], "target_y": [], "adjacent_x": [], "adjacent_y": []}
    handles = []
    for name, module in (("target", target_projection), ("adjacent", adjacent_projection)):
        def pre(_module: torch.nn.Module, args: tuple[Any, ...], *, key: str = name) -> None:
            captured[f"{key}_x"].append(args[0].detach())

        def post(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any, *, key: str = name) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            captured[f"{key}_y"].append(tensor.detach())

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    try:
        for _group, inputs in _batches(encoded, tokenizer.pad_token_id):
            with torch.no_grad():
                model(**inputs, use_cache=False, return_dict=True)
            mask = inputs["attention_mask"].bool()
            # Hook tensors are captured in model layer order, so append each batch
            # after one forward and remove padded positions with this batch mask.
            for name in ("target_x", "target_y", "adjacent_x", "adjacent_y"):
                tensor = captured[name][-1]
                captured[name][-1] = _activation_bth(tensor, *mask.shape)[mask]
    finally:
        for handle in handles:
            handle.remove()
    # Hooks append once per batch; the last entry was already unpadded above.
    target_x = captured["target_x"]
    target_y = captured["target_y"]
    adjacent_x = captured["adjacent_x"]
    adjacent_y = captured["adjacent_y"]
    tx, ty = torch.cat(target_x).float(), torch.cat(target_y).float()
    ax, ay = torch.cat(adjacent_x).float(), torch.cat(adjacent_y).float()
    if len(tx) < MIN_ADAPTATION_ROWS:
        raise ValueError(f"adaptation capture yielded {len(tx)} rows; need {MIN_ADAPTATION_ROWS}")
    if not (len(tx) == len(ty) == len(ax) == len(ay)):
        raise ValueError("selected and adjacent projections did not capture paired rows")
    in_features, out_features, use_bias = _module_dimensions(target_projection)
    if (ax.shape[1], ay.shape[1]) != (in_features, out_features):
        raise ValueError("adjacent projection dimensions do not match the selected projection")

    fixture_dir = custody / "adaptation-captures" / checkpoint.family
    fixture_dir.mkdir(parents=True, exist_ok=True)
    capture_path = fixture_dir / f"seed-{seed_index}.npz"
    _save_npz_deterministic(
        capture_path,
        target_inputs=tx.numpy(),
        target_outputs=ty.numpy(),
        adjacent_inputs=ax.numpy(),
        adjacent_outputs=ay.numpy(),
    )
    rng = np.random.default_rng(SEEDS[seed_index] ^ 0x5A17)
    shuffled_indices = rng.permutation(len(ty))
    treatment, treatment_receipt = fit_independent_linear_replacement(
        in_features, out_features, tx, ty, use_bias=use_bias, seed=SEEDS[seed_index], ridge=RIDGE
    )
    wrong, wrong_receipt = fit_independent_linear_replacement(
        in_features,
        out_features,
        ax,
        ay,
        use_bias=use_bias,
        seed=SEEDS[seed_index] + 101,
        ridge=RIDGE,
    )
    random_control, random_receipt = fit_independent_linear_replacement(
        in_features,
        out_features,
        tx,
        ty[torch.as_tensor(shuffled_indices)],
        use_bias=use_bias,
        seed=SEEDS[seed_index] + 202,
        ridge=RIDGE,
    )
    permutation_control, permutation, aligned_error = _make_variant_modules(
        treatment, SEEDS[seed_index] + 303
    )
    noop = torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32)
    with torch.no_grad():
        noop.weight.zero_()
        if noop.bias is not None:
            noop.bias.zero_()
    controls = {
        "treatment": treatment,
        "wrong_module": wrong,
        "random": random_control,
        "raw_index_permutation": permutation_control["raw_index_permutation"],
        "mapping_aligned_permutation": permutation_control["raw_index_permutation"],
        "no_op": noop,
    }
    candidate_dir = custody / "candidates" / checkpoint.family / f"seed-{seed_index}"
    module_rows = []
    for name, module in controls.items():
        file = candidate_dir / f"{name}.safetensors"
        digest = _save_linear(file, module)
        module_rows.append({"name": name, "file": file.relative_to(custody).as_posix(), "state_sha256": digest})
    return {
        "family": checkpoint.family,
        "seed_index": seed_index,
        "seed": SEEDS[seed_index],
        "selected_layer": layer,
        "adjacent_layer": adjacent,
        "adaptation_fixture_sha256": _canonical_digest(fixture_records),
        "adaptation_capture_file": capture_path.relative_to(custody).as_posix(),
        "adaptation_capture_sha256": _sha256(capture_path.read_bytes()),
        "adaptation_row_count": int(len(tx)),
        "input_features": in_features,
        "output_features": out_features,
        "use_bias": use_bias,
        "treatment_fit": treatment_receipt,
        "wrong_module_fit": wrong_receipt,
        "random_fit": random_receipt,
        "random_target_permutation_sha256": _sha256(shuffled_indices.astype("<i8").tobytes()),
        "raw_index_permutation": permutation,
        "mapping_aligned_equivalence_max_abs_error": aligned_error,
        "modules": module_rows,
        "fit_elapsed_wall_seconds": time.monotonic() - fit_started,
    }


def prepare_candidates(custody_root: str | Path, checkpoint_manifest: str | Path) -> dict[str, Any]:
    stage_started = time.monotonic()
    freeze = verify_packet()
    custody = _custody_root(custody_root)
    custody.mkdir(parents=True, exist_ok=True)
    if (custody / "candidate-preparation.json").exists():
        raise FileExistsError("candidate preparation is immutable; do not rerun after candidate creation")
    manifest_path = _custodied_manifest_path(checkpoint_manifest, custody)
    checkpoints = _load_checkpoints(manifest_path, custody)
    visible_fixture = generate_visible_fixture()
    visible_records = [asdict(row) for row in visible_fixture]
    visible_payload = fixture_payload(visible_fixture, fixture_kind="visible_selection")
    _write_json(custody / "fixtures/visible-selection.json", visible_payload)
    adaptation_payloads = {
        seed_index: fixture_payload(
            generate_adaptation_fixture(seed_index), fixture_kind=f"adaptation_seed_{seed_index}"
        )
        for seed_index in range(len(SEEDS))
    }
    validate_partition_disjointness(visible_payload, *adaptation_payloads.values())
    candidate_cases = []
    checkpoint_rows = []
    for checkpoint in checkpoints:
        family_started = time.monotonic()
        checkpoint_digest, checkpoint_files = _tree_digest(checkpoint.local_path)
        model, tokenizer = _load_local_model(checkpoint)
        selected_layer, variances = _capture_visible_layer(
            model, tokenizer, checkpoint.family, visible_records
        )
        visible_tokenized_digest = _canonical_digest(
            [row["ids"] for row in _token_rows(tokenizer, visible_records, with_continuation=False)]
        )
        family_case_rows = []
        for seed_index in range(len(SEEDS)):
            adaptation_payload = adaptation_payloads[seed_index]
            _write_json(
                custody / "fixtures" / checkpoint.family / f"adaptation-seed-{seed_index}.json",
                adaptation_payload,
            )
            validate_partition_disjointness(visible_payload, adaptation_payload)
            case = _fit_case(
                model,
                tokenizer,
                checkpoint,
                selected_layer,
                seed_index,
                adaptation_payload["records"],
                custody,
            )
            case.update(
                {
                    "checkpoint_model_id": checkpoint.model_id,
                    "checkpoint_revision": checkpoint.revision,
                    "checkpoint_tree_sha256": checkpoint_digest,
                    "visible_fixture_sha256": visible_payload["example_set_sha256"],
                    "visible_tokenized_sha256": visible_tokenized_digest,
                    "selection_layer_variances": variances,
                }
            )
            family_case_rows.append(case)
            candidate_cases.append(case)
        shared_preparation_seconds = max(
            0.0,
            time.monotonic() - family_started
            - sum(float(row["fit_elapsed_wall_seconds"]) for row in family_case_rows),
        )
        for row in family_case_rows:
            row["candidate_preparation_shared_allocated_seconds"] = shared_preparation_seconds / len(SEEDS)
        checkpoint_rows.append(
            {
                "family": checkpoint.family,
                "model_id": checkpoint.model_id,
                "revision": checkpoint.revision,
                "checkpoint_tree_sha256": checkpoint_digest,
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "checkpoint_files": checkpoint_files,
                "selected_layer": selected_layer,
                "visible_fixture_sha256": visible_payload["example_set_sha256"],
                "cases": family_case_rows,
            }
        )
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "stage_status": "CANDIDATES_PREPARED" if len(candidate_cases) == 9 else "INCOMPLETE",
        "packet_sha256": freeze["packet_sha256"],
        "checkpoint_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_count": len(checkpoint_rows),
        "case_count": len(candidate_cases),
        "elapsed_wall_seconds": time.monotonic() - stage_started,
        "runtime": _runtime_receipt(),
        "checkpoints": checkpoint_rows,
        "cases": candidate_cases,
    }
    _write_json(custody / "candidate-preparation.json", result)
    return {key: result[key] for key in ("protocol_id", "stage_status", "packet_sha256", "checkpoint_count", "case_count")}


def freeze_candidates(custody_root: str | Path) -> dict[str, Any]:
    stage_started = time.monotonic()
    custody = _custody_root(custody_root)
    freeze = verify_packet()
    preparation = _load_json(custody / "candidate-preparation.json")
    if preparation.get("packet_sha256") != freeze["packet_sha256"] or preparation.get("case_count") != 9:
        raise ValueError("candidate preparation is incomplete or bound to a different packet")
    files: list[dict[str, Any]] = []
    for case in preparation["cases"]:
        for module in case["modules"]:
            path = custody / module["file"]
            files.append({"path": module["file"], "sha256": _sha256(path.read_bytes())})
        capture = custody / case["adaptation_capture_file"]
        files.append({"path": case["adaptation_capture_file"], "sha256": _sha256(capture.read_bytes())})
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "candidate_preparation_sha256": _sha256((custody / "candidate-preparation.json").read_bytes()),
        "candidate_files": files,
        "candidate_files_sha256": _canonical_digest(files),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "challenge_fixtures_generated": False,
        "elapsed_wall_seconds": time.monotonic() - stage_started,
        "runtime": _runtime_receipt(),
    }
    destination = custody / "candidate-freeze.json"
    if destination.exists():
        raise FileExistsError("candidate freeze is immutable")
    _write_json(destination, payload)
    return {key: payload[key] for key in ("packet_sha256", "candidate_files_sha256", "created_at_utc")}


def _verify_candidate_freeze(custody: Path) -> dict[str, Any]:
    payload = _load_json(custody / "candidate-freeze.json")
    preparation_path = custody / "candidate-preparation.json"
    if (
        payload.get("packet_sha256") != verify_packet()["packet_sha256"]
        or payload.get("challenge_fixtures_generated") is not False
        or payload.get("candidate_files_sha256") != _canonical_digest(payload.get("candidate_files", []))
        or _sha256(preparation_path.read_bytes()) != payload.get("candidate_preparation_sha256")
    ):
        raise ValueError("candidate freeze does not bind the frozen protocol and candidate files")
    for row in payload["candidate_files"]:
        if _sha256((custody / row["path"]).read_bytes()) != row["sha256"]:
            raise ValueError("candidate file changed after freeze")
    return payload


def _direction(record: dict[str, Any], activation: np.ndarray, entropy: bytes) -> np.ndarray:
    seed_bytes = hashlib.sha256(entropy + str(record["example_id"]).encode("ascii")).digest()[:8]
    rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
    norm = float(np.sqrt(np.mean(np.square(activation, dtype=np.float64)))) * 0.01
    if norm <= 0:
        raise ValueError("cannot form intervention from a zero-RMS projection input")
    width = activation.shape[-1]
    if record["stratum"] == "ordinary":
        direction = rng.normal(size=width)
    elif record["stratum"] == "compositional":
        first = rng.normal(size=width)
        second = rng.normal(size=width)
        first /= max(float(np.linalg.norm(first)), 1e-30)
        second /= max(float(np.linalg.norm(second)), 1e-30)
        direction = first + second
    elif record["stratum"] == "adversarial":
        direction = np.zeros(width, dtype=np.float64)
        direction[int(np.argmax(np.abs(activation)))] = float(rng.choice((-1.0, 1.0)))
    else:
        raise ValueError(f"unknown challenge stratum: {record['stratum']}")
    direction /= max(float(np.linalg.norm(direction)), 1e-30)
    return (direction * norm).astype(np.float32)


def make_challenges(custody_root: str | Path, checkpoint_manifest: str | Path) -> dict[str, Any]:
    stage_started = time.monotonic()
    custody = _custody_root(custody_root)
    freeze = _verify_candidate_freeze(custody)
    if (custody / "challenge-manifest.json").exists():
        raise FileExistsError("challenge packet is immutable; do not regenerate held-out fixtures")
    checkpoints = _load_checkpoints(_custodied_manifest_path(checkpoint_manifest, custody), custody)
    preparation = _load_json(custody / "candidate-preparation.json")
    visible_payload = _load_json(custody / "fixtures/visible-selection.json")
    adaptation_payloads = [
        _load_json(custody / "fixtures" / FAMILIES[0] / f"adaptation-seed-{seed_index}.json")
        for seed_index in range(len(SEEDS))
    ]
    summaries = []
    all_challenge_payloads: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        family_started = time.monotonic()
        case_rows = [row for row in preparation["cases"] if row["family"] == checkpoint.family]
        model, tokenizer = _load_local_model(checkpoint)
        for case in case_rows:
            challenge_case_started = time.monotonic()
            records = generate_challenge_fixture(int(case["seed_index"]))
            payload = fixture_payload(records, fixture_kind=f"challenge_{checkpoint.family}_seed_{case['seed_index']}")
            validate_partition_disjointness(
                visible_payload, *adaptation_payloads, *all_challenge_payloads, payload
            )
            token_rows = _token_rows(tokenizer, payload["records"], with_continuation=True)
            activations: list[np.ndarray] = []
            for group, inputs in _batches(token_rows, tokenizer.pad_token_id):
                positions = _selected_positions(group)
                _logits, selected, _outputs = _run_projection(
                    model, checkpoint.family, int(case["selected_layer"]), inputs,
                    positions=positions, abort_after_input=True,
                )
                activations.extend(item.astype(np.float32) for item in selected.cpu().numpy())
            if len(activations) != len(payload["records"]):
                raise ValueError("challenge activation capture did not cover every record")
            entropy = os.urandom(32)
            deltas = np.stack([_direction(row, activation, entropy) for row, activation in zip(payload["records"], activations)])
            challenge_dir = custody / "challenges" / checkpoint.family / f"seed-{case['seed_index']}"
            _write_json(challenge_dir / "prompts.json", payload)
            _save_npz_deterministic(challenge_dir / "interventions.npz", deltas=deltas)
            record = {
                "family": checkpoint.family,
                "seed_index": case["seed_index"],
                "fixture_sha256": payload["example_set_sha256"],
                "prompt_file_sha256": _sha256((challenge_dir / "prompts.json").read_bytes()),
                "interventions_file_sha256": _sha256((challenge_dir / "interventions.npz").read_bytes()),
                "record_count": len(payload["records"]),
                "stratum_counts": payload["stratum_counts"],
                "challenge_entropy_retained": False,
                "challenge_case_elapsed_wall_seconds": time.monotonic() - challenge_case_started,
            }
            summaries.append(record)
            all_challenge_payloads.append(payload)
        shared_challenge_seconds = max(
            0.0,
            time.monotonic() - family_started
            - sum(float(row["challenge_case_elapsed_wall_seconds"]) for row in summaries if row["family"] == checkpoint.family),
        )
        family_summaries = [row for row in summaries if row["family"] == checkpoint.family]
        for row in family_summaries:
            row["challenge_shared_allocated_seconds"] = shared_challenge_seconds / len(SEEDS)
        del model
        del tokenizer
    result = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": verify_packet()["packet_sha256"],
        "candidate_files_sha256": freeze["candidate_files_sha256"],
        "candidate_freeze_file_sha256": _sha256((custody / "candidate-freeze.json").read_bytes()),
        "challenge_generation_started_after_candidate_freeze": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": summaries,
        "elapsed_wall_seconds": time.monotonic() - stage_started,
        "runtime": _runtime_receipt(),
    }
    result["challenge_manifest_sha256"] = _canonical_digest(result)
    _write_json(custody / "challenge-manifest.json", result)
    return {
        "case_count": len(summaries),
        "challenge_manifest_sha256": result["challenge_manifest_sha256"],
        "record_count_per_case": CHALLENGES_PER_STRATUM * len(STRATA),
    }


def _module_map(custody: Path, case: dict[str, Any]) -> dict[str, torch.nn.Linear]:
    result: dict[str, torch.nn.Linear] = {}
    for item in case["modules"]:
        result[item["name"]] = _load_linear(
            custody / item["file"], int(case["input_features"]), int(case["output_features"]), bool(case["use_bias"])
        )
    return result


def _model_forward_variant(
    model: torch.nn.Module,
    family: str,
    layer: int,
    replacement: torch.nn.Module,
    group: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    deltas: torch.Tensor | None,
    input_permutation: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    original = _projection(model, family, layer)
    _set_projection(model, family, layer, replacement)
    try:
        logits, _in, outputs = _run_projection(
            model,
            family,
            layer,
            inputs,
            positions=_selected_positions(group),
            deltas=deltas,
            input_permutation=input_permutation,
        )
    finally:
        _set_projection(model, family, layer, original)
    if logits is None or outputs is None:
        raise RuntimeError("variant model forward did not return logits and projection outputs")
    return logits.cpu(), outputs.cpu()


def _model_forward_original(
    model: torch.nn.Module,
    family: str,
    layer: int,
    group: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    deltas: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits, activations_in, activations_out = _run_projection(
        model,
        family,
        layer,
        inputs,
        positions=_selected_positions(group),
        deltas=deltas,
    )
    if logits is None or activations_out is None:
        raise RuntimeError("original model forward did not return logits and boundary activations")
    return logits.cpu(), activations_in.cpu(), activations_out.cpu()


def _metrics_for_effect_rows(
    predicted: np.ndarray,
    observed: np.ndarray,
    floors: np.ndarray,
    scales: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        causal_effect_metrics(p, o, resolution_floor=float(floor), scale=float(scale))
        for p, o, floor, scale in zip(predicted, observed, floors, scales)
    ]


def _aggregate_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolvable = [row for row in rows if row["resolvable"]]
    if not resolvable:
        return {"count": len(rows), "resolvable_count": 0, "direction_accuracy": None, "median_cosine": None, "median_normalized_error": None}
    return {
        "count": len(rows),
        "resolvable_count": len(resolvable),
        "direction_accuracy": float(np.mean([row["direction_correct"] for row in resolvable])),
        "median_cosine": float(np.median([row["cosine_similarity"] for row in resolvable])),
        "median_normalized_error": float(np.median([row["normalized_error"] for row in resolvable])),
    }


def _case_evaluation(
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint: Checkpoint,
    case: dict[str, Any],
    custody: Path,
    challenge_manifest_row: dict[str, Any],
) -> dict[str, Any]:
    case_started = time.monotonic()
    family, seed_index, layer = checkpoint.family, int(case["seed_index"]), int(case["selected_layer"])
    challenge_dir = custody / "challenges" / family / f"seed-{seed_index}"
    prompt_file = challenge_dir / "prompts.json"
    intervention_file = challenge_dir / "interventions.npz"
    if _sha256(prompt_file.read_bytes()) != challenge_manifest_row["prompt_file_sha256"]:
        raise ValueError("challenge prompt fixture digest mismatch")
    if _sha256(intervention_file.read_bytes()) != challenge_manifest_row["interventions_file_sha256"]:
        raise ValueError("challenge intervention fixture digest mismatch")
    fixture = _load_json(prompt_file)
    if challenge_manifest_row.get("fixture_sha256") != fixture.get("example_set_sha256"):
        raise ValueError("challenge fixture identity mismatch")
    rows = _token_rows(tokenizer, fixture["records"], with_continuation=True)
    if len(rows) != CHALLENGES_PER_STRATUM * len(STRATA):
        raise ValueError("challenge fixture row count differs from preregistration")
    for stratum in STRATA:
        if sum(row["stratum"] == stratum for row in rows) != CHALLENGES_PER_STRATUM:
            raise ValueError(f"challenge fixture lacks required {stratum} coverage")
    deltas = np.load(intervention_file)["deltas"].astype(np.float32)
    if deltas.shape != (len(rows), int(case["input_features"])):
        raise ValueError("challenge intervention dimensions do not match selected projection")
    modules = _module_map(custody, case)
    prediction_arrays: dict[str, list[np.ndarray]] = {
        f"{name}_{suffix}": []
        for name in ("treatment", "wrong_module", "random", "raw_index_permutation", "mapping_aligned_permutation", "no_op")
        for suffix in ("module_effect", "boundary_output", "logit_effect", "next_logits", "ce")
    }
    predictions_per_row: dict[str, list[dict[str, Any]]] = {name: [] for name in ("treatment", "wrong_module", "random", "raw_index_permutation", "mapping_aligned_permutation", "no_op")}
    variant_keys = {
        "treatment": "treatment",
        "wrong_module": "wrong_module",
        "random": "random",
        "raw_index_permutation": "raw_index_permutation",
        "mapping_aligned_permutation": "mapping_aligned_permutation",
        "no_op": "no_op",
    }
    cursor = 0
    started = time.monotonic()
    # Prediction queries are committed before any original-model outcome pass.
    for group, inputs in _batches(rows, tokenizer.pad_token_id):
        count = len(group)
        local_deltas = torch.from_numpy(deltas[cursor : cursor + count])
        for name, key in variant_keys.items():
            module = modules[key]
            input_permutation = (
                case["raw_index_permutation"] if name == "mapping_aligned_permutation" else None
            )
            base_logits, base_outputs = _model_forward_variant(
                model, family, layer, module, group, inputs, deltas=None,
                input_permutation=input_permutation,
            )
            if name == "no_op":
                changed_logits = base_logits
                changed_outputs = base_outputs
            else:
                changed_logits, changed_outputs = _model_forward_variant(
                    model, family, layer, module, group, inputs, deltas=local_deltas,
                    input_permutation=input_permutation,
                )
            base_next = _next_logits(base_logits, group).numpy()
            changed_next = _next_logits(changed_logits, group).numpy()
            base_ce = _continuation_cross_entropy(base_logits, group)
            module_effect = changed_outputs.numpy() - base_outputs.numpy()
            logit_effect = changed_next - base_next
            for local_index, row in enumerate(group):
                predictions_per_row[name].append(
                    {
                        "example_id": row["example_id"],
                        "stratum": row["stratum"],
                        "predicted_module_effect": module_effect[local_index],
                        "predicted_logit_effect": logit_effect[local_index],
                        "candidate_boundary_output": base_outputs.numpy()[local_index],
                        "candidate_next_logits": base_next[local_index],
                        "candidate_ce": base_ce[local_index],
                    }
                )
            prediction_arrays[f"{name}_module_effect"].append(module_effect)
            prediction_arrays[f"{name}_boundary_output"].append(base_outputs.numpy())
            prediction_arrays[f"{name}_logit_effect"].append(logit_effect)
            prediction_arrays[f"{name}_next_logits"].append(base_next)
            prediction_arrays[f"{name}_ce"].append(base_ce)
        cursor += count
    prediction_duration = time.monotonic() - started
    compact_arrays = {
        name: np.concatenate(parts, axis=0) if parts else np.empty((0,), dtype=np.float32)
        for name, parts in prediction_arrays.items()
    }
    compact_arrays["example_ids"] = np.asarray([row["example_id"] for row in rows], dtype="U24")
    compact_arrays["strata"] = np.asarray([row["stratum"] for row in rows], dtype="U24")
    prediction_path = custody / "predictions" / family / f"seed-{seed_index}.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz_deterministic(prediction_path, **compact_arrays)
    prediction_digest = _sha256(prediction_path.read_bytes())
    prediction_receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "family": family,
        "seed_index": seed_index,
        "packet_sha256": verify_packet()["packet_sha256"],
        "candidate_files_sha256": _load_json(custody / "candidate-freeze.json")["candidate_files_sha256"],
        "challenge_fixture_sha256": fixture["example_set_sha256"],
        "prediction_file_sha256": prediction_digest,
        "prediction_query_count": len(rows),
        "prediction_duration_seconds": prediction_duration,
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "original_outcomes_opened": False,
    }
    receipt_path = custody / "predictions" / family / f"seed-{seed_index}.receipt.json"
    _write_json(receipt_path, prediction_receipt)

    # Original baseline is run twice after prediction sealing, then intervention outcomes once.
    observed_inputs: list[np.ndarray] = []
    observed_activations: list[np.ndarray] = []
    baseline_logits_rows: list[np.ndarray] = []
    repeat_logits_rows: list[np.ndarray] = []
    intervention_activations: list[np.ndarray] = []
    intervention_logits_rows: list[np.ndarray] = []
    original_ce: list[np.ndarray] = []
    original_top: list[np.ndarray] = []
    original_logits_for_behavior: list[np.ndarray] = []
    cursor = 0
    for group, inputs in _batches(rows, tokenizer.pad_token_id):
        count = len(group)
        baseline_logits, activation_inputs, activation_outputs = _model_forward_original(
            model, family, layer, group, inputs
        )
        repeat_logits, _repeat_inputs, _repeat_outputs = _model_forward_original(
            model, family, layer, group, inputs
        )
        local_deltas = torch.from_numpy(deltas[cursor : cursor + count])
        changed_logits, _changed_inputs, changed_outputs = _model_forward_original(
            model, family, layer, group, inputs, deltas=local_deltas
        )
        observed_inputs.append(activation_inputs.numpy())
        observed_activations.append(activation_outputs.numpy())
        baseline_logits_rows.append(_next_logits(baseline_logits, group).numpy())
        original_ce.append(_continuation_cross_entropy(baseline_logits, group))
        original_top.append(np.argmax(_next_logits(baseline_logits, group).numpy(), axis=1))
        original_logits_for_behavior.append(_next_logits(baseline_logits, group).numpy())
        repeat_logits_rows.append(_next_logits(repeat_logits, group).numpy())
        intervention_logits_rows.append(_next_logits(changed_logits, group).numpy())
        intervention_activations.append(changed_outputs.numpy())
        cursor += count
    original_next = np.concatenate(baseline_logits_rows, axis=0)
    repeat_next = np.concatenate(repeat_logits_rows, axis=0)
    observed_logit_effect = np.concatenate(intervention_logits_rows, axis=0) - original_next
    baseline_activation = np.concatenate(observed_activations, axis=0)
    observed_module_effect = np.concatenate(intervention_activations, axis=0) - baseline_activation
    resolution_floors = np.maximum(1e-12, 1e-8 * np.linalg.norm(original_next - repeat_next, axis=1))
    scales = np.maximum(np.linalg.norm(original_next, axis=1), 1e-12)
    metrics_by_mode: dict[str, Any] = {}
    for name, values in predictions_per_row.items():
        predicted_module = np.stack([value["predicted_module_effect"] for value in values])
        predicted_logits = np.stack([value["predicted_logit_effect"] for value in values])
        module_metrics = _metrics_for_effect_rows(predicted_module, observed_module_effect, np.full(len(rows), 1e-12), scales)
        logit_metrics = _metrics_for_effect_rows(predicted_logits, observed_logit_effect, resolution_floors, scales)
        metrics_by_mode[name] = {
            "module_boundary": _aggregate_effect(module_metrics),
            "logits": _aggregate_effect(logit_metrics),
            "per_stratum": {},
        }
        for stratum in STRATA:
            indexes = [index for index, row in enumerate(rows) if row["stratum"] == stratum]
            metrics_by_mode[name]["per_stratum"][stratum] = {
                "module_boundary": _aggregate_effect([module_metrics[index] for index in indexes]),
                "logits": _aggregate_effect([logit_metrics[index] for index in indexes]),
            }

    orig_ce = np.concatenate(original_ce)
    orig_top = np.concatenate(original_top)
    orig_next = np.concatenate(original_logits_for_behavior)
    behavior_by_mode: dict[str, Any] = {}
    for name, values in predictions_per_row.items():
        candidate_ce = np.asarray([value["candidate_ce"] for value in values], dtype=np.float64)
        candidate_next = np.stack([value["candidate_next_logits"] for value in values])
        candidate_top = np.argmax(candidate_next, axis=1)
        row_relerr = np.linalg.norm(candidate_next - orig_next, axis=1) / np.maximum(
            np.linalg.norm(orig_next, axis=1), 1e-12
        )
        behavior_by_mode[name] = {
            "continuation_token_count": int(sum(len(row["continuation_ids"]) for row in rows)),
            "mean_cross_entropy_nats_per_token": float(np.average(candidate_ce, weights=[len(row["continuation_ids"]) for row in rows])),
            "original_mean_cross_entropy_nats_per_token": float(np.average(orig_ce, weights=[len(row["continuation_ids"]) for row in rows])),
            "cross_entropy_delta_nats_per_token": float(
                np.average(candidate_ce - orig_ce, weights=[len(row["continuation_ids"]) for row in rows])
            ),
            "next_token_top1_agreement": float(np.mean(candidate_top == orig_top)),
            "paired_next_token_normalized_logit_error_max": float(np.max(row_relerr)),
            "per_stratum": {},
        }
        for stratum in STRATA:
            indexes = np.asarray([index for index, row in enumerate(rows) if row["stratum"] == stratum], dtype=int)
            weights = np.asarray([len(rows[index]["continuation_ids"]) for index in indexes], dtype=np.float64)
            behavior_by_mode[name]["per_stratum"][stratum] = {
                "cross_entropy_delta_nats_per_token": float(np.average(candidate_ce[indexes] - orig_ce[indexes], weights=weights)),
                "next_token_top1_agreement": float(np.mean(candidate_top[indexes] == orig_top[indexes])),
                "paired_next_token_normalized_logit_error_max": float(np.max(row_relerr[indexes])),
            }

    for mode in ("treatment", "wrong_module", "random", "raw_index_permutation", "no_op"):
        candidate_outputs = compact_arrays[f"{mode}_boundary_output"]
        cosine_rows: list[float] = []
        error_rows: list[float] = []
        for predicted, observed in zip(candidate_outputs, baseline_activation):
            denom = max(float(np.linalg.norm(predicted) * np.linalg.norm(observed)), 1e-30)
            cosine_rows.append(float(np.dot(predicted, observed) / denom))
            error_rows.append(float(np.linalg.norm(predicted - observed) / max(float(np.linalg.norm(observed)), 1e-12)))
        if mode == "treatment":
            treatment_activation = {
                "median_cosine": float(np.median(cosine_rows)),
                "median_normalized_error": float(np.median(error_rows)),
                "per_stratum": {},
            }
            for stratum in STRATA:
                indexes = [index for index, row in enumerate(rows) if row["stratum"] == stratum]
                treatment_activation["per_stratum"][stratum] = {
                    "median_cosine": float(np.median(np.asarray(cosine_rows)[indexes])),
                    "median_normalized_error": float(np.median(np.asarray(error_rows)[indexes])),
                }
    aligned_prediction_error = float(
        np.max(np.abs(compact_arrays["treatment_module_effect"] - compact_arrays["mapping_aligned_permutation_module_effect"]))
    )
    aligned_logit_error = float(
        np.max(np.abs(compact_arrays["treatment_logit_effect"] - compact_arrays["mapping_aligned_permutation_logit_effect"]))
    )
    per_case_score = {
        name: float(metrics_by_mode[name]["logits"]["direction_accuracy"] or 0.0)
        for name in metrics_by_mode
    }
    treatment_pass = _passes_effect(
        metrics_by_mode["treatment"]["logits"], minimum_resolvable=60
    ) and _passes_effect(
        metrics_by_mode["treatment"]["module_boundary"], minimum_resolvable=60
    )
    stratum_pass = all(
        _passes_effect(metrics_by_mode["treatment"]["per_stratum"][stratum]["logits"], minimum_resolvable=20)
        and _passes_effect(
            metrics_by_mode["treatment"]["per_stratum"][stratum]["module_boundary"],
            minimum_resolvable=20,
        )
        for stratum in STRATA
    )
    behavior_pass = all(
        _passes_behavior(behavior_by_mode["treatment"]["per_stratum"][stratum])
        for stratum in STRATA
    ) and _passes_behavior(behavior_by_mode["treatment"])
    activation_pass = (
        treatment_activation["median_cosine"] >= 0.90
        and treatment_activation["median_normalized_error"] <= 0.20
        and all(
            treatment_activation["per_stratum"][stratum]["median_cosine"] >= 0.90
            and treatment_activation["per_stratum"][stratum]["median_normalized_error"] <= 0.20
            for stratum in STRATA
        )
    )
    case_elapsed = time.monotonic() - case_started
    coverage_pass = all(
        metrics_by_mode["treatment"]["per_stratum"][stratum]["logits"]["resolvable_count"] >= 20
        for stratum in STRATA
    )
    case_within_wall_cap = case_elapsed <= 2 * 60 * 60
    status = (
        "PASS"
        if treatment_pass and stratum_pass and behavior_pass and activation_pass and case_within_wall_cap
        else "INCONCLUSIVE"
        if not coverage_pass or not case_within_wall_cap
        else "FAIL"
    )
    outcome_path = custody / "outcomes" / family / f"seed-{seed_index}.npz"
    outcome_path.parent.mkdir(parents=True, exist_ok=True)
    _save_npz_deterministic(
        outcome_path,
        example_ids=np.asarray([row["example_id"] for row in rows], dtype="U24"),
        strata=np.asarray([row["stratum"] for row in rows], dtype="U24"),
        original_next_logits=original_next,
        original_logit_effect=observed_logit_effect,
        original_module_activation=baseline_activation,
        original_module_effect=observed_module_effect,
        observed_module_inputs=np.concatenate(observed_inputs, axis=0),
        resolution_floors=resolution_floors,
        baseline_logit_scales=scales,
    )
    return {
        "family": family,
        "model_id": checkpoint.model_id,
        "revision": checkpoint.revision,
        "checkpoint_tree_sha256": case["checkpoint_tree_sha256"],
        "seed_index": seed_index,
        "seed": SEEDS[seed_index],
        "selected_layer": layer,
        "challenge_fixture_sha256": fixture["example_set_sha256"],
        "prediction_receipt_sha256": _sha256(receipt_path.read_bytes()),
        "prediction_array_sha256": prediction_digest,
        "outcome_array_sha256": _sha256(outcome_path.read_bytes()),
        "prediction_outcome_order_verified": True,
        "challenge_counts": fixture["stratum_counts"],
        "thresholds": {
            "direction_accuracy_min": 0.80,
            "cluster_bootstrap_direction_lower_min": 0.70,
            "median_cosine_min": 0.90,
            "median_normalized_error_max": 0.20,
            "behavior_ce_delta_max": 0.05,
            "behavior_next_logit_error_max": 0.05,
            "activation_cosine_min": 0.90,
            "activation_normalized_error_max": 0.20,
        },
        "primary_effects": metrics_by_mode,
        "behavior": behavior_by_mode,
        "activation_rescue": treatment_activation,
        "mapping_aligned_permutation_diagnostic": {
            "module_effect_max_abs_error_vs_treatment": aligned_prediction_error,
            "logit_effect_max_abs_error_vs_treatment": aligned_logit_error,
            "required_max_abs_error": EQUIVALENCE_MAX_ABS_ERROR,
        },
        "case_scores_for_cluster_bootstrap": per_case_score,
        "case_status": status,
        "case_elapsed_wall_seconds": case_elapsed,
        "blockers": [] if status == "PASS" else [
            name for name, passed in (
                ("resolvable_challenge_coverage", coverage_pass),
                ("primary_treatment_thresholds", treatment_pass),
                ("stratum_primary_thresholds", stratum_pass),
                ("heldout_behavior_thresholds", behavior_pass),
                ("activation_rescue_thresholds", activation_pass),
                ("case_wall_cap", case_within_wall_cap),
            ) if not passed
        ],
    }


def _passes_effect(summary: dict[str, Any], *, minimum_resolvable: int = 1) -> bool:
    return (
        summary.get("resolvable_count", 0) >= minimum_resolvable
        and summary.get("direction_accuracy", 0.0) >= 0.80
        and summary.get("median_cosine", -1.0) >= 0.90
        and summary.get("median_normalized_error", float("inf")) <= 0.20
    )


def _passes_behavior(summary: dict[str, Any]) -> bool:
    return (
        summary.get("cross_entropy_delta_nats_per_token", float("inf")) <= 0.05
        and summary.get("paired_next_token_normalized_logit_error_max", float("inf")) <= 0.05
    )


def _runtime_receipt() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "transformers", "safetensors", "tokenizers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak *= 1024
    dependency_rows = sorted(
        {
            f"{str(distribution.metadata.get('Name', 'unknown')).strip().lower()}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        }
    )
    dependency_digest = _sha256("\n".join(dependency_rows).encode("utf-8"))
    runtime = {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).absolute()),
        "python_real_executable": str(Path(sys.executable).resolve()),
        "python_prefix": str(Path(sys.prefix).resolve()),
        "platform": platform.platform(),
        "packages": packages,
        "dependency_inventory_sha256": dependency_digest,
        "dependency_inventory_count": len(dependency_rows),
        "package_manifest_sha256": _sha256((_repo_root() / "requirements.txt").read_bytes()),
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "peak_rss_bytes": peak,
        "process_cpu_seconds": float(usage.ru_utime + usage.ru_stime),
    }
    runtime["runtime_identity_sha256"] = _canonical_digest(
        {
            "python_executable": runtime["python_executable"],
            "python_real_executable": runtime["python_real_executable"],
            "python_prefix": runtime["python_prefix"],
            "python": runtime["python"],
            "platform": runtime["platform"],
            "packages": packages,
            "dependency_inventory_sha256": dependency_digest,
        }
    )
    return runtime


def _source_identity() -> dict[str, str]:
    root = _repo_root()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"git_revision": revision, "worktree_status_sha256": _sha256(status.encode("utf-8"))}


def evaluate_packet(
    custody_root: str | Path,
    checkpoint_manifest: str | Path,
    *,
    checkpoint_specs: list[Checkpoint] | None = None,
) -> dict[str, Any]:
    stage_started = time.monotonic()
    freeze = verify_packet()
    custody = _custody_root(custody_root)
    candidate_freeze = _verify_candidate_freeze(custody)
    challenge_manifest = _load_json(custody / "challenge-manifest.json")
    if (
        challenge_manifest.get("packet_sha256") != freeze["packet_sha256"]
        or challenge_manifest.get("candidate_files_sha256") != candidate_freeze["candidate_files_sha256"]
        or len(challenge_manifest.get("cases", [])) != 9
        or challenge_manifest.get("candidate_freeze_file_sha256")
        != _sha256((custody / "candidate-freeze.json").read_bytes())
        or challenge_manifest.get("challenge_generation_started_after_candidate_freeze") is not True
    ):
        raise ValueError("challenge manifest is incomplete or not bound to the frozen candidate packet")
    challenge_material = dict(challenge_manifest)
    challenge_digest = challenge_material.pop("challenge_manifest_sha256", None)
    if challenge_digest != _canonical_digest(challenge_material):
        raise ValueError("challenge manifest digest mismatch")
    expected_case_keys = {(family, seed_index) for family in FAMILIES for seed_index in range(len(SEEDS))}
    actual_case_keys = {(row.get("family"), row.get("seed_index")) for row in challenge_manifest["cases"]}
    if actual_case_keys != expected_case_keys:
        raise ValueError("challenge manifest must bind exactly three seeds for each family")
    if datetime.fromisoformat(challenge_manifest["created_at_utc"]) <= datetime.fromisoformat(
        candidate_freeze["created_at_utc"]
    ):
        raise ValueError("held-out challenge fixtures must be generated after candidate freeze")
    if checkpoint_specs is None:
        manifest_path = _custodied_manifest_path(checkpoint_manifest, custody)
        checkpoints = _load_checkpoints(manifest_path, custody)
    else:
        manifest_path = Path(checkpoint_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError("replication checkpoint manifest is missing")
        checkpoints = sorted(checkpoint_specs, key=lambda item: FAMILIES.index(item.family))
        if {item.family for item in checkpoints} != set(FAMILIES):
            raise ValueError("replication checkpoint set does not cover all frozen families")
    preparation = _load_json(custody / "candidate-preparation.json")
    report_cases = []
    started = time.monotonic()
    for checkpoint in checkpoints:
        checkpoint_load_started = time.monotonic()
        model, tokenizer = _load_local_model(checkpoint)
        checkpoint_load_seconds = time.monotonic() - checkpoint_load_started
        case_rows = [row for row in preparation["cases"] if row["family"] == checkpoint.family]
        challenge_rows = [row for row in challenge_manifest["cases"] if row["family"] == checkpoint.family]
        for case in case_rows:
            challenge_row = next(row for row in challenge_rows if row["seed_index"] == case["seed_index"])
            evaluated = _case_evaluation(model, tokenizer, checkpoint, case, custody, challenge_row)
            evaluated["evaluation_model_load_allocated_seconds"] = checkpoint_load_seconds / len(SEEDS)
            prep_case = case
            challenge_case = challenge_row
            evaluated["full_case_elapsed_wall_seconds"] = (
                float(prep_case.get("fit_elapsed_wall_seconds", 0.0))
                + float(prep_case.get("candidate_preparation_shared_allocated_seconds", 0.0))
                + float(challenge_case.get("challenge_case_elapsed_wall_seconds", 0.0))
                + float(challenge_case.get("challenge_shared_allocated_seconds", 0.0))
                + float(evaluated["case_elapsed_wall_seconds"])
                + float(evaluated["evaluation_model_load_allocated_seconds"])
            )
            if evaluated["full_case_elapsed_wall_seconds"] > 2 * 60 * 60:
                evaluated["case_status"] = "INCONCLUSIVE"
                evaluated["blockers"].append("total_case_wall_cap")
            report_cases.append(evaluated)
        del model
        del tokenizer
    modes = ("raw_index_permutation", "wrong_module", "random", "no_op")
    treatment_scores = np.asarray([row["case_scores_for_cluster_bootstrap"]["treatment"] for row in report_cases])
    treatment_bootstrap_lower = bootstrap_mean_difference_lower_bound(
        treatment_scores, np.zeros_like(treatment_scores), seed=20260924, draws=10_000, quantile=0.025
    )
    control_comparisons = {}
    for index, mode in enumerate(modes):
        control_scores = np.asarray([row["case_scores_for_cluster_bootstrap"][mode] for row in report_cases])
        control_comparisons[mode] = bootstrap_mean_difference_lower_bound(
            treatment_scores, control_scores, seed=20260925 + index, draws=10_000, quantile=0.025
        )
    global_pass = (
        len(report_cases) == 9
        and all(row["case_status"] == "PASS" for row in report_cases)
        and treatment_bootstrap_lower > 0.70
        and treatment_scores.mean() >= 0.80
        and all(value > 0 for value in control_comparisons.values())
        and all(row["mapping_aligned_permutation_diagnostic"]["module_effect_max_abs_error_vs_treatment"] <= EQUIVALENCE_MAX_ABS_ERROR for row in report_cases)
        and all(row["mapping_aligned_permutation_diagnostic"]["logit_effect_max_abs_error_vs_treatment"] <= EQUIVALENCE_MAX_ABS_ERROR for row in report_cases)
    )
    elapsed = time.monotonic() - started
    runtime = _runtime_receipt()
    preparation = _load_json(custody / "candidate-preparation.json")
    challenge_stage = _load_json(custody / "challenge-manifest.json")
    candidate_stage = _load_json(custody / "candidate-freeze.json")
    total_wall = (
        float(preparation.get("elapsed_wall_seconds", 0.0))
        + float(candidate_stage.get("elapsed_wall_seconds", 0.0))
        + float(challenge_stage.get("elapsed_wall_seconds", 0.0))
        + (time.monotonic() - stage_started)
    )
    stage_runtimes = [
        preparation.get("runtime", {}),
        candidate_stage.get("runtime", {}),
        challenge_stage.get("runtime", {}),
        runtime,
    ]
    peak_rss = max((int(item.get("peak_rss_bytes", 0)) for item in stage_runtimes), default=0)
    total_cpu = sum(float(item.get("process_cpu_seconds", 0.0)) for item in stage_runtimes)
    resource_blockers = []
    if peak_rss > 24 * 1024**3:
        resource_blockers.append("process_peak_rss_exceeded_24_GiB")
    if total_wall > 18 * 60 * 60:
        resource_blockers.append("total_wall_time_exceeded_18_hours")
    if resource_blockers:
        global_pass = False
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "assurance_level": "INTERNAL_PASS" if global_pass else "INCONCLUSIVE",
        "outcome_status": "PASS" if global_pass else (
            "INCOMPLETE"
            if resource_blockers or any(row["case_status"] == "INCONCLUSIVE" for row in report_cases)
            else "FAIL"
        ),
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "reproducibility_status": "PENDING_SEPARATE_REPLICATOR",
        "claim_ceiling": "LOCAL_INDEPENDENT_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ONLY",
        "nonclaims": ["independent science", "alignment", "consciousness", "production authorization", "all architectures"],
        "packet_sha256": freeze["packet_sha256"],
        "source": _source_identity(),
        "candidate_files_sha256": candidate_freeze["candidate_files_sha256"],
        "candidate_freeze_file_sha256": _sha256((custody / "candidate-freeze.json").read_bytes()),
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "checkpoint_manifest_sha256": _sha256(manifest_path.read_bytes()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_wall_seconds": elapsed,
        "total_protocol_stage_wall_seconds": total_wall,
        "runtime": runtime,
        "stage_peak_rss_bytes": peak_rss,
        "custody_storage_before_report": _custody_storage_bytes(custody),
        "process_cpu_seconds_across_stages": total_cpu,
        "monetary_cost": "NOT_MEASURED",
        "energy": "NOT_MEASURED",
        "cluster_bootstrap": {
            "cluster_definition": "checkpoint-family x seed case",
            "cluster_count": len(report_cases),
            "treatment_direction_accuracy_mean": float(treatment_scores.mean()),
            "treatment_direction_accuracy_95_percent_lower": treatment_bootstrap_lower,
            "paired_treatment_minus_control_95_percent_lower": control_comparisons,
        },
        "resource_blockers": resource_blockers,
        "data_and_artifact_custody": "model weights, prompts, captures, predictions, and row-level outcomes are external under the supplied owner custody root",
        "cases": report_cases,
        "known_limitations": [
            "one bounded MLP output projection per checkpoint, not a whole model subsystem",
            "synthetic prompt fixtures and same-owner local checkpoints",
            "local replication and separate internal replication do not confer independent scientific authority",
            "local CPU energy and monetary cost are not measured by this runner",
        ],
        "next_blocked_authority_gate": (
            "frozen digest-bound packet review by an external reviewer with independent authority"
            if global_pass
            else "local replacement and causal-prediction gates did not pass; independent science review remains closed"
        ),
    }
    _write_json(custody / "replacement-loss-v3-report.json", report)
    return {
        "assurance_level": report["assurance_level"],
        "case_count": len(report_cases),
        "passed_case_count": sum(row["case_status"] == "PASS" for row in report_cases),
        "packet_sha256": report["packet_sha256"],
        "report_sha256": _sha256((custody / "replacement-loss-v3-report.json").read_bytes()),
        "next_blocked_authority_gate": report["next_blocked_authority_gate"],
    }


def _copy_custody_file(source_root: Path, destination_root: Path, relative: str | Path) -> dict[str, str]:
    source = source_root / relative
    destination = destination_root / relative
    if not source.is_file():
        raise FileNotFoundError(f"replication input is missing: {Path(relative).name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"path": Path(relative).as_posix(), "sha256": _sha256(destination.read_bytes())}


def replicate_packet(
    primary_custody_root: str | Path,
    replica_custody_root: str | Path,
    checkpoint_manifest: str | Path,
) -> dict[str, Any]:
    """Rebuild candidates and rerun sealed challenges without opening primary outcomes first."""

    started = time.monotonic()
    primary = _custody_root(primary_custody_root)
    replica = _custody_root(replica_custody_root)
    try:
        replica.relative_to(primary)
    except ValueError:
        pass
    else:
        raise ValueError("replica custody must be separate from primary custody")
    try:
        primary.relative_to(replica)
    except ValueError:
        pass
    else:
        raise ValueError("replica custody cannot contain primary custody")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("replica custody must be a new or empty external directory")
    freeze = verify_packet()
    primary_candidate_freeze = _verify_candidate_freeze(primary)
    primary_manifest = _custodied_manifest_path(checkpoint_manifest, primary)
    checkpoints = _load_checkpoints(primary_manifest, primary)
    preparation = _load_json(primary / "candidate-preparation.json")
    if preparation.get("packet_sha256") != freeze["packet_sha256"]:
        raise ValueError("primary inputs do not bind the frozen protocol packet")
    replica.mkdir(parents=True, exist_ok=True)
    copied_inputs: list[dict[str, str]] = []
    for relative in (
        "candidate-preparation.json",
        "candidate-freeze.json",
        "fixtures/visible-selection.json",
    ):
        copied_inputs.append(_copy_custody_file(primary, replica, relative))
    copied_inputs.append(
        _copy_custody_file(primary, replica, primary_manifest.relative_to(primary))
    )
    if (primary / "frozen-packet-archive.json").is_file():
        copied_inputs.append(_copy_custody_file(primary, replica, "frozen-packet-archive.json"))
        source_packet_dir = primary / "frozen-packet"
        replica_packet_dir = replica / "frozen-packet"
        shutil.copytree(source_packet_dir, replica_packet_dir)
    for case in preparation["cases"]:
        family = str(case["family"])
        seed_index = int(case["seed_index"])
        for relative in (
            Path("fixtures") / family / f"adaptation-seed-{seed_index}.json",
        ):
            copied_inputs.append(_copy_custody_file(primary, replica, relative))

    reconstruction_cases: list[dict[str, Any]] = []
    blockers: list[str] = []
    for checkpoint in checkpoints:
        expected_checkpoint = next(row for row in preparation["checkpoints"] if row["family"] == checkpoint.family)
        tree_digest, _file_rows = _tree_digest(checkpoint.local_path)
        if tree_digest != expected_checkpoint["checkpoint_tree_sha256"]:
            blockers.append(f"{checkpoint.family}:checkpoint_tree_digest_mismatch")
        model, tokenizer = _load_local_model(checkpoint)
        visible_payload = _load_json(replica / "fixtures/visible-selection.json")
        visible_records = visible_payload["records"]
        selected_layer, variances = _capture_visible_layer(model, tokenizer, checkpoint.family, visible_records)
        if selected_layer != expected_checkpoint["selected_layer"]:
            blockers.append(f"{checkpoint.family}:visible_layer_selection_mismatch")
        expected_variances = expected_checkpoint["cases"][0]["selection_layer_variances"]
        variance_error = float(
            np.max(np.abs(np.asarray(variances, dtype=np.float64) - np.asarray(expected_variances, dtype=np.float64)))
        )
        if variance_error > 1e-7:
            blockers.append(f"{checkpoint.family}:visible_selection_variance_mismatch")
        visible_tokenized_digest = _canonical_digest(
            [row["ids"] for row in _token_rows(tokenizer, visible_records, with_continuation=False)]
        )
        for case in (row for row in preparation["cases"] if row["family"] == checkpoint.family):
            seed_index = int(case["seed_index"])
            adaptation_payload = _load_json(
                replica / "fixtures" / checkpoint.family / f"adaptation-seed-{seed_index}.json"
            )
            if _canonical_digest(adaptation_payload["records"]) != case["adaptation_fixture_sha256"]:
                blockers.append(f"{checkpoint.family}/seed-{seed_index}:adaptation_fixture_mismatch")
            if visible_tokenized_digest != case["visible_tokenized_sha256"]:
                blockers.append(f"{checkpoint.family}/seed-{seed_index}:visible_tokenization_mismatch")
            rebuilt = _fit_case(
                model,
                tokenizer,
                checkpoint,
                int(case["selected_layer"]),
                seed_index,
                adaptation_payload["records"],
                replica,
            )
            module_expected = {row["name"]: row["state_sha256"] for row in case["modules"]}
            module_actual = {row["name"]: row["state_sha256"] for row in rebuilt["modules"]}
            capture_match = rebuilt["adaptation_capture_sha256"] == case["adaptation_capture_sha256"]
            module_match = module_actual == module_expected
            if not capture_match:
                blockers.append(f"{checkpoint.family}/seed-{seed_index}:adaptation_capture_mismatch")
            if not module_match:
                blockers.append(f"{checkpoint.family}/seed-{seed_index}:candidate_weight_mismatch")
            reconstruction_cases.append(
                {
                    "family": checkpoint.family,
                    "seed_index": seed_index,
                    "selected_layer": selected_layer,
                    "selection_variance_max_abs_error": variance_error,
                    "adaptation_row_count": rebuilt["adaptation_row_count"],
                    "adaptation_capture_digest_match": capture_match,
                    "candidate_state_digests_match": module_match,
                    "candidate_state_sha256": module_actual,
                }
            )
        del model
        del tokenizer
    reconstruction = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "candidate_files_sha256": primary_candidate_freeze["candidate_files_sha256"],
        "source_checkpoint_manifest_sha256": _sha256(primary_manifest.read_bytes()),
        "reconstruction_cases": reconstruction_cases,
        "copied_blind_inputs": copied_inputs,
        "blockers": blockers,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(replica / "candidate-reconstruction.json", reconstruction)
    if blockers:
        result = {
            **reconstruction,
            "assurance_level": "INCONCLUSIVE",
            "reproducibility_status": "INCONCLUSIVE",
            "elapsed_wall_seconds": time.monotonic() - started,
            "next_blocked_authority_gate": "candidate or fixture digest mismatch prevents exact reproduction",
        }
        _write_json(replica / "replication-report.json", result)
        return {"assurance_level": result["assurance_level"], "blockers": blockers}

    _verify_candidate_freeze(replica)
    challenge_manifest = _load_json(primary / "challenge-manifest.json")
    if challenge_manifest.get("candidate_files_sha256") != primary_candidate_freeze["candidate_files_sha256"]:
        raise ValueError("primary challenge packet is not bound to the reconstructed candidates")
    copied_inputs.append(_copy_custody_file(primary, replica, "challenge-manifest.json"))
    for challenge_case in challenge_manifest["cases"]:
        family = str(challenge_case["family"])
        seed_index = int(challenge_case["seed_index"])
        for relative in (
            Path("challenges") / family / f"seed-{seed_index}" / "prompts.json",
            Path("challenges") / family / f"seed-{seed_index}" / "interventions.npz",
        ):
            copied_inputs.append(_copy_custody_file(primary, replica, relative))
    reconstruction["copied_blind_inputs"] = copied_inputs
    _write_json(replica / "candidate-reconstruction.json", reconstruction)
    eval_summary = evaluate_packet(
        replica,
        primary_manifest,
        checkpoint_specs=checkpoints,
    )
    primary_report_path = primary / "replacement-loss-v3-report.json"
    if not primary_report_path.is_file():
        raise FileNotFoundError("primary aggregate report is required after the replica has sealed its own outcomes")
    primary_report = _load_json(primary_report_path)
    replica_report_path = replica / "replacement-loss-v3-report.json"
    replica_report = _load_json(replica_report_path)
    primary_cases = {(row["family"], row["seed_index"]): row for row in primary_report["cases"]}
    replica_cases = {(row["family"], row["seed_index"]): row for row in replica_report["cases"]}
    output_digest_rows = []
    output_digests_match = set(primary_cases) == set(replica_cases)
    for key in sorted(set(primary_cases) & set(replica_cases)):
        original = primary_cases[key]
        reproduced = replica_cases[key]
        matching = (
            original["prediction_array_sha256"] == reproduced["prediction_array_sha256"]
            and original["outcome_array_sha256"] == reproduced["outcome_array_sha256"]
            and original["challenge_fixture_sha256"] == reproduced["challenge_fixture_sha256"]
        )
        output_digests_match = output_digests_match and matching
        output_digest_rows.append(
            {
                "family": key[0],
                "seed_index": key[1],
                "prediction_arrays_match": original["prediction_array_sha256"] == reproduced["prediction_array_sha256"],
                "outcome_arrays_match": original["outcome_array_sha256"] == reproduced["outcome_array_sha256"],
                "challenge_fixture_match": original["challenge_fixture_sha256"] == reproduced["challenge_fixture_sha256"],
            }
        )
    primary_runtime = primary_report["runtime"]
    replica_runtime = replica_report["runtime"]
    dependency_match = (
        primary_runtime["python"] == replica_runtime["python"]
        and primary_runtime["platform"] == replica_runtime["platform"]
        and primary_runtime["packages"] == replica_runtime["packages"]
        and primary_runtime["dependency_inventory_sha256"]
        == replica_runtime["dependency_inventory_sha256"]
        and primary_runtime["package_manifest_sha256"] == replica_runtime["package_manifest_sha256"]
    )
    distinct_environment = (
        primary_runtime["python_prefix"] != replica_runtime["python_prefix"]
        and primary_runtime["runtime_identity_sha256"] != replica_runtime["runtime_identity_sha256"]
    )
    source_match = primary_report["source"] == replica_report["source"]
    assurance = (
        "REPRODUCED_LOCAL"
        if output_digests_match and dependency_match and distinct_environment and source_match
        else "INCONCLUSIVE"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_level": assurance,
        "reproducibility_status": assurance,
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "same_owner_second_environment": True,
        "independent_science_or_acceptance": False,
        "packet_sha256": freeze["packet_sha256"],
        "candidate_files_sha256": primary_candidate_freeze["candidate_files_sha256"],
        "candidate_reconstruction_match": True,
        "prediction_and_outcome_array_digests_match": output_digests_match,
        "dependencies_match": dependency_match,
        "distinct_clean_runtime": distinct_environment,
        "source_identity_match": source_match,
        "primary_report_sha256": _sha256(primary_report_path.read_bytes()),
        "replica_report_sha256": _sha256(replica_report_path.read_bytes()),
        "replica_evaluation_summary": eval_summary,
        "replication_cases": output_digest_rows,
        "runtime": replica_runtime,
        "elapsed_wall_seconds": time.monotonic() - started,
        "custody_storage": _custody_storage_bytes(replica),
        "nonclaims": ["independent science", "alignment", "consciousness", "production authorization"],
        "next_blocked_authority_gate": "external reviewer with independent authority",
    }
    _write_json(replica / "replication-report.json", result)
    return {
        "assurance_level": assurance,
        "candidate_reconstruction_match": True,
        "prediction_and_outcome_array_digests_match": output_digests_match,
        "dependencies_match": dependency_match,
        "distinct_clean_runtime": distinct_environment,
        "replication_report_sha256": _sha256((replica / "replication-report.json").read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen replacement-loss-v3 local protocol stages")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze-packet")
    for command in ("archive-packet", "prepare", "freeze-candidates", "make-challenges", "evaluate", "replicate"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--custody-root", required=True)
        if command in ("prepare", "make-challenges", "evaluate", "replicate"):
            sub.add_argument("--checkpoint-manifest", required=True)
        if command == "replicate":
            sub.add_argument("--replica-root", required=True)
    args = parser.parse_args()
    if args.command == "freeze-packet":
        result = freeze_packet()
    elif args.command == "archive-packet":
        result = archive_frozen_packet(args.custody_root)
    elif args.command == "prepare":
        result = prepare_candidates(args.custody_root, args.checkpoint_manifest)
    elif args.command == "freeze-candidates":
        result = freeze_candidates(args.custody_root)
    elif args.command == "make-challenges":
        result = make_challenges(args.custody_root, args.checkpoint_manifest)
    elif args.command == "replicate":
        result = replicate_packet(args.custody_root, args.replica_root, args.checkpoint_manifest)
    else:
        result = evaluate_packet(args.custody_root, args.checkpoint_manifest)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
