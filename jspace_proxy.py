"""Bounded local Jacobian-lens proxy for the Theseus experiment."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2_experiment import bootstrap_ci, full_digest


FIT_PROMPTS = (
    "The capital of France is",
    "The opposite of hot is",
    "The largest planet is",
    "A common color of grass is",
    "The first month of the year is",
    "Birds can usually",
    "Three times three equals",
    "The sun rises in the",
)
EVALUATION_PROMPTS = (
    "A triangle has three",
    "Water freezes at",
    "A year has twelve",
    "The chemical symbol for water is",
    "Before noon comes",
    "A mammal breathes",
    "The Pacific is an",
    "When rain falls, the ground gets",
)
LAYERS = (0, 5, 11)
TARGET_TOKENS = (" the", " is", " of", " and")
PERMUTATION_SHIFT = 17
PRUNE_COUNT = 128
JSPACE_COSINE_MIN = 0.999
JSPACE_RELATIVE_L2_MAX = 1e-3


def _input_ids(tokenizer: AutoTokenizer, prompt: str) -> torch.Tensor:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    return encoded["input_ids"]


def target_ids(tokenizer: AutoTokenizer) -> tuple[int, ...]:
    ids: list[int] = []
    for token in TARGET_TOKENS:
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"target token is not a single tokenizer unit: {token!r}")
        ids.append(int(encoded[0]))
    return tuple(ids)


def last_hidden_state(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
) -> torch.Tensor:
    with torch.no_grad():
        output = model(
            input_ids=_input_ids(tokenizer, prompt),
            output_hidden_states=True,
            return_dict=True,
        )
    return output.hidden_states[layer + 1][0, -1].detach().clone()


def prompt_jacobian(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    layer: int,
    selected_targets: tuple[int, ...],
) -> np.ndarray:
    """Estimate d(selected final logits) / d(last residual at ``layer``)."""

    base = last_hidden_state(model, tokenizer, prompt, layer)
    current: torch.Tensor | None = None

    def replace_last_residual(_module: torch.nn.Module, _inputs: tuple[object, ...], output: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        if current is None:
            raise RuntimeError("Jacobian probe did not provide a residual vector")
        replacement = output[0].clone()
        replacement[:, -1, :] = current
        return (replacement, *output[1:])

    handle = model.transformer.h[layer].register_forward_hook(replace_last_residual)
    input_ids = _input_ids(tokenizer, prompt)

    def logits_from_residual(residual: torch.Tensor) -> torch.Tensor:
        nonlocal current
        current = residual
        output = model(input_ids=input_ids, return_dict=True)
        return output.logits[0, -1, list(selected_targets)]

    try:
        with torch.enable_grad():
            jacobian = torch.autograd.functional.jacobian(
                logits_from_residual,
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
    block = model.transformer.h[layer]
    fc_weight = block.mlp.c_fc.weight
    fc_bias = block.mlp.c_fc.bias
    projection = block.mlp.c_proj.weight
    order = _permutation(fc_bias.shape[0], shift)
    with torch.no_grad():
        original = (fc_weight.detach().clone(), fc_bias.detach().clone(), projection.detach().clone())
        fc_weight.copy_(fc_weight[:, order])
        fc_bias.copy_(fc_bias[order])
        projection.copy_(projection[order, :])
    try:
        yield order
    finally:
        with torch.no_grad():
            fc_weight.copy_(original[0])
            fc_bias.copy_(original[1])
            projection.copy_(original[2])


@contextmanager
def prune_mlp(model: torch.nn.Module, layer: int, count: int):
    block = model.transformer.h[layer]
    fc_weight = block.mlp.c_fc.weight
    fc_bias = block.mlp.c_fc.bias
    if count < 1 or count > fc_bias.shape[0]:
        raise ValueError("pruning count is outside the MLP width")
    with torch.no_grad():
        original = (fc_weight.detach().clone(), fc_bias.detach().clone())
        fc_weight[:, :count].zero_()
        fc_bias[:count].zero_()
    try:
        yield
    finally:
        with torch.no_grad():
            fc_weight.copy_(original[0])
            fc_bias.copy_(original[1])


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


def _coordinates(jacobian: np.ndarray, hidden_states: list[torch.Tensor]) -> list[np.ndarray]:
    return [jacobian @ hidden.cpu().numpy() for hidden in hidden_states]


def analyze_jspace(model_dir: Path) -> dict[str, object]:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    selected_targets = target_ids(tokenizer)
    layer_reports: list[dict[str, object]] = []

    for layer in LAYERS:
        fit_jacobians = [
            prompt_jacobian(model, tokenizer, prompt, layer, selected_targets)
            for prompt in FIT_PROMPTS
        ]
        mean_jacobian = np.mean(fit_jacobians, axis=0)
        baseline_hidden = [
            last_hidden_state(model, tokenizer, prompt, layer)
            for prompt in EVALUATION_PROMPTS
        ]
        baseline_coordinates = _coordinates(mean_jacobian, baseline_hidden)
        with permute_mlp(model, layer, PERMUTATION_SHIFT):
            permuted_hidden = [
                last_hidden_state(model, tokenizer, prompt, layer)
                for prompt in EVALUATION_PROMPTS
            ]
        with prune_mlp(model, layer, PRUNE_COUNT):
            pruned_hidden = [
                last_hidden_state(model, tokenizer, prompt, layer)
                for prompt in EVALUATION_PROMPTS
            ]
        permutation_metrics = vector_metrics(
            baseline_coordinates,
            _coordinates(mean_jacobian, permuted_hidden),
            layer * 100,
        )
        pruning_metrics = vector_metrics(
            baseline_coordinates,
            _coordinates(mean_jacobian, pruned_hidden),
            layer * 100 + 10,
        )
        permutation_metrics["pass"] = bool(
            permutation_metrics["mean_relative_l2"] < JSPACE_RELATIVE_L2_MAX
            and permutation_metrics["mean_cosine"] >= JSPACE_COSINE_MIN
        )
        pruning_metrics["divergent"] = bool(
            pruning_metrics["mean_relative_l2"] >= JSPACE_RELATIVE_L2_MAX
        )
        layer_reports.append({
            "layer": layer,
            "fit_jacobian_shape": list(mean_jacobian.shape),
            "permutation": permutation_metrics,
            "pruning": pruning_metrics,
        })

    return {
        "kind": "local_jacobian_lens_proxy",
        "fit_prompt_count": len(FIT_PROMPTS),
        "evaluation_prompt_count": len(EVALUATION_PROMPTS),
        "fit_prompt_digest": full_digest(FIT_PROMPTS),
        "evaluation_prompt_digest": full_digest(EVALUATION_PROMPTS),
        "target_tokens": list(TARGET_TOKENS),
        "target_token_ids": list(selected_targets),
        "target_digest": full_digest(list(selected_targets)),
        "layers": layer_reports,
        "permutation_stability": all(bool(report["permutation"]["pass"]) for report in layer_reports),
        "pruning_divergence_observed": any(bool(report["pruning"]["divergent"]) for report in layer_reports),
        "status": "LOCAL_PROXY_PASS" if all(bool(report["permutation"]["pass"]) for report in layer_reports) else "LOCAL_PROXY_INCONCLUSIVE",
        "claim_ceiling": "LOCAL_JACOBIAN_PROXY_OBSERVATION_ONLY",
    }
