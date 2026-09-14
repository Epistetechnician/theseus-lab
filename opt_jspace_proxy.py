"""Bounded local Jacobian-lens proxy for the pinned OPT checkpoint."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2_experiment import (
    HELDOUT_PROMPTS,
    VISIBLE_PROMPTS,
    bootstrap_ci,
    full_digest,
)


FIT_PROMPTS = VISIBLE_PROMPTS
EVALUATION_PROMPTS = HELDOUT_PROMPTS
TARGET_TOKENS = (" the", " is", " of", " and")
PERMUTATION_SHIFT = 17
PRUNE_COUNT = 128
JSPACE_COSINE_MIN = 0.999
JSPACE_RELATIVE_L2_MAX = 1e-3


def selected_layers(layer_count: int) -> tuple[int, ...]:
    candidates = (0, 5, 11) if layer_count >= 12 else (0, layer_count // 2, layer_count - 1)
    return tuple(dict.fromkeys(index for index in candidates if 0 <= index < layer_count))


def _input_ids(tokenizer: AutoTokenizer, prompt: str) -> torch.Tensor:
    return tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]


def target_ids(tokenizer: AutoTokenizer) -> tuple[int, ...]:
    ids: list[int] = []
    for token in TARGET_TOKENS:
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"target token is not a single tokenizer unit: {token!r}")
        ids.append(int(encoded[0]))
    return tuple(ids)


def _layer(model: torch.nn.Module, index: int) -> torch.nn.Module:
    return model.model.decoder.layers[index]


def last_mlp_activation(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
) -> torch.Tensor:
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[object, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().clone())

    handle = _layer(model, layer).activation_fn.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(input_ids=_input_ids(tokenizer, prompt), return_dict=True)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("OPT MLP activation probe did not capture exactly one tensor")
    activation = captured[0]
    return activation[-1] if activation.ndim == 2 else activation[0, -1]


def prompt_jacobian(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    selected_targets: tuple[int, ...],
) -> np.ndarray:
    """Estimate d(selected final logits) / d(last post-activation MLP vector)."""

    base = last_mlp_activation(model, tokenizer, prompt, layer)
    current: torch.Tensor | None = None

    def replace_last_activation(
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        if current is None:
            raise RuntimeError("OPT Jacobian probe did not provide an activation vector")
        replacement = output.clone()
        if replacement.ndim == 2:
            replacement[-1, :] = current
        else:
            replacement[:, -1, :] = current
        return replacement

    handle = _layer(model, layer).activation_fn.register_forward_hook(replace_last_activation)
    input_ids = _input_ids(tokenizer, prompt)

    def logits_from_activation(activation: torch.Tensor) -> torch.Tensor:
        nonlocal current
        current = activation
        output = model(input_ids=input_ids, return_dict=True)
        return output.logits[0, -1, list(selected_targets)]

    try:
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(
                logits_from_activation,
                base,
                vectorize=False,
            )
    finally:
        handle.remove()
    return jacobian.detach().cpu().numpy().astype(np.float32)


def _permutation(width: int, shift: int) -> np.ndarray:
    return np.roll(np.arange(width), shift % width)


@contextmanager
def permute_mlp(model: torch.nn.Module, layer: int, shift: int):
    block = _layer(model, layer)
    fc1_weight, fc1_bias, fc2_weight = block.fc1.weight, block.fc1.bias, block.fc2.weight
    order = _permutation(fc1_bias.shape[0], shift)
    with torch.no_grad():
        original = (fc1_weight.detach().clone(), fc1_bias.detach().clone(), fc2_weight.detach().clone())
        fc1_weight.copy_(fc1_weight[order, :])
        fc1_bias.copy_(fc1_bias[order])
        fc2_weight.copy_(fc2_weight[:, order])
    try:
        yield order
    finally:
        with torch.no_grad():
            fc1_weight.copy_(original[0])
            fc1_bias.copy_(original[1])
            fc2_weight.copy_(original[2])


@contextmanager
def prune_mlp(model: torch.nn.Module, layer: int, count: int):
    block = _layer(model, layer)
    if count < 1 or count > block.fc1.bias.shape[0]:
        raise ValueError("pruning count is outside the OPT MLP width")
    with torch.no_grad():
        original = (block.fc1.weight.detach().clone(), block.fc1.bias.detach().clone())
        block.fc1.weight[:count, :].zero_()
        block.fc1.bias[:count].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            block.fc1.weight.copy_(original[0])
            block.fc1.bias.copy_(original[1])


def _metric_values(before: list[np.ndarray], after: list[np.ndarray]) -> tuple[list[float], list[float], list[float]]:
    relative_l2: list[float] = []
    cosine: list[float] = []
    max_abs: list[float] = []
    for left, right in zip(before, after):
        relative_l2.append(float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12)))
        cosine.append(float(np.clip(
            np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12),
            -1.0,
            1.0,
        )))
        max_abs.append(float(np.max(np.abs(left - right))))
    return relative_l2, cosine, max_abs


def vector_metrics(before: list[np.ndarray], after: list[np.ndarray], seed_offset: int) -> dict[str, object]:
    relative_l2, cosine, max_abs = _metric_values(before, after)
    return {
        "prompt_count": len(before),
        "mean_relative_l2": round(float(np.mean(relative_l2)), 8),
        "mean_relative_l2_ci95": bootstrap_ci(relative_l2, seed_offset),
        "mean_cosine": round(float(np.mean(cosine)), 8),
        "mean_cosine_ci95": bootstrap_ci(cosine, seed_offset + 1),
        "mean_max_abs": round(float(np.mean(max_abs)), 8),
        "mean_max_abs_ci95": bootstrap_ci(max_abs, seed_offset + 2),
    }


def _coordinates(jacobian: np.ndarray, activations: list[torch.Tensor]) -> list[np.ndarray]:
    return [jacobian @ activation.cpu().numpy() for activation in activations]


def analyze_jspace(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    selected_targets = target_ids(tokenizer)
    layer_reports: list[dict[str, object]] = []

    for layer in selected_layers(len(model.model.decoder.layers)):
        fit_jacobians = [
            prompt_jacobian(model, tokenizer, prompt, layer, selected_targets)
            for prompt in FIT_PROMPTS
        ]
        mean_jacobian = np.mean(fit_jacobians, axis=0)
        baseline_activations = [
            last_mlp_activation(model, tokenizer, prompt, layer)
            for prompt in EVALUATION_PROMPTS
        ]
        baseline_coordinates = _coordinates(mean_jacobian, baseline_activations)
        with permute_mlp(model, layer, PERMUTATION_SHIFT) as order:
            permuted_activations = [
                last_mlp_activation(model, tokenizer, prompt, layer)
                for prompt in EVALUATION_PROMPTS
            ]
        inverse = np.argsort(order)
        aligned_activations = [activation[inverse] for activation in permuted_activations]
        with prune_mlp(model, layer, PRUNE_COUNT):
            pruned_activations = [
                last_mlp_activation(model, tokenizer, prompt, layer)
                for prompt in EVALUATION_PROMPTS
            ]
        raw_permutation_metrics = vector_metrics(
            baseline_coordinates,
            _coordinates(mean_jacobian, permuted_activations),
            layer * 100,
        )
        aligned_permutation_metrics = vector_metrics(
            baseline_coordinates,
            _coordinates(mean_jacobian, aligned_activations),
            layer * 100 + 1,
        )
        pruning_metrics = vector_metrics(
            baseline_coordinates,
            _coordinates(mean_jacobian, pruned_activations),
            layer * 100 + 10,
        )
        aligned_permutation_metrics["pass"] = bool(
            aligned_permutation_metrics["mean_relative_l2"] < JSPACE_RELATIVE_L2_MAX
            and aligned_permutation_metrics["mean_cosine"] >= JSPACE_COSINE_MIN
        )
        raw_permutation_metrics["coordinate_change_observed"] = bool(
            raw_permutation_metrics["mean_relative_l2"] >= JSPACE_RELATIVE_L2_MAX
        )
        pruning_metrics["divergent"] = bool(
            pruning_metrics["mean_relative_l2"] >= JSPACE_RELATIVE_L2_MAX
        )
        layer_reports.append({
            "layer": layer,
            "coordinate_space": "post_relu_mlp_activation",
            "fit_jacobian_shape": list(mean_jacobian.shape),
            "raw_permutation": raw_permutation_metrics,
            "aligned_permutation": aligned_permutation_metrics,
            "pruning": pruning_metrics,
        })

    permutation_stability = all(bool(report["aligned_permutation"]["pass"]) for report in layer_reports)
    return {
        "kind": "local_jacobian_lens_proxy",
        "model_id": model_id or model_dir.name,
        "coordinate_space": "post_relu_mlp_activation",
        "fit_prompt_count": len(FIT_PROMPTS),
        "evaluation_prompt_count": len(EVALUATION_PROMPTS),
        "fit_prompt_digest": full_digest(FIT_PROMPTS),
        "evaluation_prompt_digest": full_digest(EVALUATION_PROMPTS),
        "target_tokens": list(TARGET_TOKENS),
        "target_token_ids": list(selected_targets),
        "target_digest": full_digest(list(selected_targets)),
        "layers": layer_reports,
        "permutation_stability": permutation_stability,
        "pruning_divergence_observed": any(bool(report["pruning"]["divergent"]) for report in layer_reports),
        "raw_coordinate_change_observed": all(bool(report["raw_permutation"]["coordinate_change_observed"]) for report in layer_reports),
        "status": "LOCAL_PROXY_PASS" if permutation_stability else "LOCAL_PROXY_INCONCLUSIVE",
        "claim_ceiling": "LOCAL_JACOBIAN_PROXY_OBSERVATION_ONLY",
    }
