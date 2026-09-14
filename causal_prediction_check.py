"""Held-out causal prediction, activation patching, and rescue for SmolLM2.

This is a local exploratory protocol.  It predicts intervention effects from
the original checkpoint, evaluates those predictions on held-out prompts, and
records a deliberately narrow evidence ceiling.  It does not establish
mechanistic understanding, alignment, or production readiness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

from gpt2_experiment import bootstrap_ci, full_digest
from reference_check import permutation
from smollm2_reference_check import FRESH_HELDOUT_PROMPTS, FRESH_VISIBLE_PROMPTS
from smollm_experiment import NumpySmolLM, SmolLMCheckpoint, tensor_digest
from smollm_reference_check import (
    canonicalize_gated_mlp,
    canonicalize_gqa,
    infer_gated_mlp_permutation,
    infer_gqa_permutation,
)


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "smollm2-causal-prediction-v1"
SELECTED_LAYERS = (0, 14, 29)
CHANNELS_PER_LAYER = 3
GQA_GROUPS = (0, 1, 2)
MLP_TRANSFORM_SHIFT = 17
GQA_TRANSFORM_SHIFT = 1
RANDOM_SEED = 20260914
BOOTSTRAP_SEED_OFFSET = 61000
PREDICTION_RELATIVE_L2_MAX = 1e-4
RESCUE_RELATIVE_L2_MAX = 1e-4
CAUSAL_PREDICTION_WIN_RATE_MIN = 0.8
TARGET_TOKENS = (" the", " is", " of", " and")


def _target_ids(tokenizer: AutoTokenizer) -> list[int]:
    ids: list[int] = []
    for token in TARGET_TOKENS:
        encoded = tokenizer(token, add_special_tokens=False)["input_ids"]
        if len(encoded) != 1:
            raise ValueError(f"target token is not a single tokenizer unit: {token!r}")
        ids.append(int(encoded[0]))
    if len(set(ids)) != len(ids):
        raise ValueError("target tokens must have distinct ids")
    return ids


def _prompt_ids(tokenizer: AutoTokenizer, prompts: Iterable[str]) -> list[list[int]]:
    encoded = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    if any(not ids for ids in encoded):
        raise ValueError("all causal probes must tokenize to a non-empty sequence")
    return [[int(token) for token in ids] for ids in encoded]


def _target_logits(
    model: NumpySmolLM,
    token_ids: list[int],
    target_ids: list[int],
    **kwargs: object,
) -> np.ndarray:
    logits = model.forward(token_ids, precision="float64", **kwargs)
    return np.asarray(logits[target_ids], dtype=np.float64)


def _relative_l2(before: np.ndarray, after: np.ndarray) -> float:
    return float(np.linalg.norm(before - after) / max(np.linalg.norm(before), 1e-12))


def _pair_summary(
    actual: list[np.ndarray],
    predicted: list[np.ndarray],
    seed_offset: int,
) -> dict[str, object]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("paired causal observations must be non-empty and equal length")
    relative_l2 = [_relative_l2(left, right) for left, right in zip(actual, predicted)]
    absolute_l2 = [float(np.linalg.norm(left - right)) for left, right in zip(actual, predicted)]
    cosine = [
        float(np.clip(
            np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12),
            -1.0,
            1.0,
        ))
        for left, right in zip(actual, predicted)
    ]
    return {
        "observation_count": len(actual),
        "mean_relative_l2": round(float(np.mean(relative_l2)), 10),
        "relative_l2_ci95": bootstrap_ci(relative_l2, seed_offset),
        "mean_absolute_l2": round(float(np.mean(absolute_l2)), 10),
        "mean_cosine": round(float(np.mean(cosine)), 10),
        "cosine_ci95": bootstrap_ci(cosine, seed_offset + 1),
    }


def _win_rate(
    canonical: list[float],
    comparator: list[float],
    seed_offset: int,
) -> dict[str, object]:
    wins = [int(left < right) for left, right in zip(canonical, comparator)]
    return {
        "comparison_count": len(wins),
        "canonical_win_rate": round(float(np.mean(wins)), 10),
        "canonical_win_rate_ci95": bootstrap_ci(wins, seed_offset),
    }


def _select_channels(
    model: NumpySmolLM,
    visible_traces: list[list[np.ndarray]],
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, object]]]:
    selected: dict[int, tuple[int, ...]] = {}
    details: dict[int, dict[str, object]] = {}
    for layer in SELECTED_LAYERS:
        matrix = np.stack([trace[layer] for trace in visible_traces])
        active_count = np.count_nonzero(np.abs(matrix) > 1e-8, axis=0)
        magnitude_sum = np.sum(np.abs(matrix), axis=0)
        order = sorted(
            range(model.ffn_dim),
            key=lambda index: (-int(active_count[index]), -float(magnitude_sum[index]), index),
        )
        channels = tuple(int(index) for index in order[:CHANNELS_PER_LAYER])
        if any(int(active_count[index]) == 0 for index in channels):
            raise ValueError(f"selected inactive SmolLM channels at layer {layer}")
        selected[layer] = channels
        details[layer] = {
            "channels": list(channels),
            "visible_active_probe_counts": [int(active_count[index]) for index in channels],
            "visible_absolute_activation_sums": [
                round(float(magnitude_sum[index]), 10) for index in channels
            ],
        }
    return selected, details


def _blind_mapping(
    model: NumpySmolLM,
    kind: str,
    layer: int,
    width: int,
    shift: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Infer the operational mapping without consuming the recorded shift."""

    recorded_permutation = permutation(width, shift)
    if kind == "mlp":
        before = model.gated_mlp_components(layer)
        after = [before[0][:, recorded_permutation], before[1][:, recorded_permutation], before[2][recorded_permutation, :]]
        canonical_before, _ = canonicalize_gated_mlp(*before)
        canonical_after, _ = canonicalize_gated_mlp(*after)
        inferred = infer_gated_mlp_permutation(before, after)
        before_digest = tensor_digest({"gate": before[0], "up": before[1], "down": before[2]})
        after_digest = tensor_digest({"gate": after[0], "up": after[1], "down": after[2]})
        canonical_before_digest = tensor_digest({"gate": canonical_before[0], "up": canonical_before[1], "down": canonical_before[2]})
        canonical_after_digest = tensor_digest({"gate": canonical_after[0], "up": canonical_after[1], "down": canonical_after[2]})
    elif kind == "gqa":
        before = model.gqa_components(layer)
        query_indexes, kv_indexes = model._group_indexes(recorded_permutation)
        after = [before[0][query_indexes, :], before[1][kv_indexes, :], before[2][kv_indexes, :], before[3][:, query_indexes]]
        canonical_before, _ = canonicalize_gqa(
            *before, model.head_dim, model.n_kv_head, model.query_heads_per_kv,
        )
        canonical_after, _ = canonicalize_gqa(
            *after, model.head_dim, model.n_kv_head, model.query_heads_per_kv,
        )
        inferred = infer_gqa_permutation(
            before,
            after,
            model.head_dim,
            model.n_kv_head,
            model.query_heads_per_kv,
        )
        before_digest = tensor_digest({"query": before[0], "key": before[1], "value": before[2], "output": before[3]})
        after_digest = tensor_digest({"query": after[0], "key": after[1], "value": after[2], "output": after[3]})
        canonical_before_digest = tensor_digest({"query": canonical_before[0], "key": canonical_before[1], "value": canonical_before[2], "output": canonical_before[3]})
        canonical_after_digest = tensor_digest({"query": canonical_after[0], "key": canonical_after[1], "value": canonical_after[2], "output": canonical_after[3]})
    else:
        raise ValueError(f"unknown component kind: {kind}")
    if not np.array_equal(inferred, recorded_permutation):
        raise ValueError(f"blind {kind} mapping did not recover the recorded transformation")
    if canonical_before_digest != canonical_after_digest:
        raise ValueError(f"blind {kind} canonical digests did not match")
    return inferred, {
        "raw_digest_diff": before_digest != after_digest,
        "canonical_digest_match": canonical_before_digest == canonical_after_digest,
        "before_digest": before_digest,
        "after_digest": after_digest,
        "canonical_before_digest": canonical_before_digest,
        "canonical_after_digest": canonical_after_digest,
        "inferred_mapping_digest": full_digest(inferred.tolist()),
        "mapping_source": "blind_exact_component_signature_inference",
    }


def _component_effects(
    model: NumpySmolLM,
    prompt_ids: list[list[int]],
    baseline: list[np.ndarray],
    target_ids: list[int],
    kind: str,
    layer: int,
    index: int,
) -> list[np.ndarray]:
    effects: list[np.ndarray] = []
    for ids, original in zip(prompt_ids, baseline):
        if kind == "mlp":
            altered = _target_logits(
                model,
                ids,
                target_ids,
                ablate_channel=index,
                ablate_channel_layer=layer,
            )
        elif kind == "gqa":
            altered = _target_logits(
                model,
                ids,
                target_ids,
                ablate_head_group=index,
                ablate_head_group_layer=layer,
            )
        else:
            raise ValueError(f"unknown component kind: {kind}")
        effects.append(original - altered)
    return effects


def _target_output_cache(
    model: NumpySmolLM,
    prompt_ids: list[list[int]],
    target_ids: list[int],
    **kwargs: object,
) -> list[np.ndarray]:
    return [_target_logits(model, ids, target_ids, **kwargs) for ids in prompt_ids]


def _mapping_prediction(
    model: NumpySmolLM,
    prompt_ids: list[list[int]],
    baseline: list[np.ndarray],
    heldout_offset: int,
    selected: dict[int, tuple[int, ...]],
    kind: str,
    width: int,
    shift: int,
    seed: int,
    target_ids: list[int],
) -> dict[str, object]:
    random_generator = np.random.default_rng(seed)
    random_source = random_generator.permutation(width)
    mapping_by_layer: dict[int, tuple[np.ndarray, dict[str, object]]] = {
        layer: _blind_mapping(model, kind, layer, width, shift)
        for layer in SELECTED_LAYERS
    }
    actual_vectors: list[np.ndarray] = []
    canonical_vectors: list[np.ndarray] = []
    raw_vectors: list[np.ndarray] = []
    random_vectors: list[np.ndarray] = []
    case_summaries: list[dict[str, object]] = []
    all_baseline_effects: dict[tuple[int, int], list[np.ndarray]] = {}
    needed_indexes = set()
    for layer in SELECTED_LAYERS:
        transform, _ = mapping_by_layer[layer]
        inverse = np.argsort(transform)
        indexes = selected[layer] if kind == "mlp" else GQA_GROUPS
        for index in indexes:
            needed_indexes.add(int(index))
        for index in indexes:
            needed_indexes.add(int(inverse[index]))
            needed_indexes.add(int(random_source[inverse[index]]))
    for layer in SELECTED_LAYERS:
        for index in sorted(needed_indexes):
            all_baseline_effects[(layer, index)] = _component_effects(
                model,
                prompt_ids,
                baseline,
                target_ids,
                kind,
                layer,
                index,
            )

    for layer in SELECTED_LAYERS:
        transform, mapping_validation = mapping_by_layer[layer]
        inverse = np.argsort(transform)
        indexes = selected[layer] if kind == "mlp" else GQA_GROUPS
        transformed_kwargs: dict[str, object] = (
            {"permutation": transform, "permutation_layer": layer}
            if kind == "mlp"
            else {"attention_permutation": transform, "attention_permutation_layer": layer}
        )
        transformed_base = _target_output_cache(model, prompt_ids, target_ids, **transformed_kwargs)
        for old_index in indexes:
            new_index = int(inverse[old_index])
            random_index = int(random_source[new_index])
            actual: list[np.ndarray] = []
            canonical: list[np.ndarray] = []
            raw: list[np.ndarray] = []
            random: list[np.ndarray] = []
            for prompt_index in range(heldout_offset, len(prompt_ids)):
                ids = prompt_ids[prompt_index]
                if kind == "mlp":
                    altered = _target_logits(
                        model,
                        ids,
                        target_ids,
                        **transformed_kwargs,
                        ablate_channel=new_index,
                        ablate_channel_layer=layer,
                    )
                else:
                    altered = _target_logits(
                        model,
                        ids,
                        target_ids,
                        **transformed_kwargs,
                        ablate_head_group=new_index,
                        ablate_head_group_layer=layer,
                    )
                actual.append(transformed_base[prompt_index] - altered)
                canonical.append(all_baseline_effects[(layer, old_index)][prompt_index])
                raw.append(all_baseline_effects[(layer, new_index)][prompt_index])
                random.append(all_baseline_effects[(layer, random_index)][prompt_index])
            actual_vectors.extend(actual)
            canonical_vectors.extend(canonical)
            raw_vectors.extend(raw)
            random_vectors.extend(random)
            canonical_errors = [_relative_l2(left, right) for left, right in zip(actual, canonical)]
            raw_errors = [_relative_l2(left, right) for left, right in zip(actual, raw)]
            random_errors = [_relative_l2(left, right) for left, right in zip(actual, random)]
            case_summaries.append({
                "layer": layer,
                "mapping_validation": mapping_validation,
                "original_component": int(old_index),
                "transformed_component": new_index,
                "random_source_component": random_index,
                "canonical_mean_relative_l2": round(float(np.mean(canonical_errors)), 10),
                "raw_mean_relative_l2": round(float(np.mean(raw_errors)), 10),
                "random_mean_relative_l2": round(float(np.mean(random_errors)), 10),
            })
    canonical_errors = [_relative_l2(left, right) for left, right in zip(actual_vectors, canonical_vectors)]
    raw_errors = [_relative_l2(left, right) for left, right in zip(actual_vectors, raw_vectors)]
    random_errors = [_relative_l2(left, right) for left, right in zip(actual_vectors, random_vectors)]
    summary = {
        "kind": kind,
        "component_case_count": len(case_summaries),
        "heldout_probe_count": len(prompt_ids) - heldout_offset,
        "canonical": _pair_summary(actual_vectors, canonical_vectors, BOOTSTRAP_SEED_OFFSET + (0 if kind == "mlp" else 100)),
        "raw_coordinate": _pair_summary(actual_vectors, raw_vectors, BOOTSTRAP_SEED_OFFSET + (10 if kind == "mlp" else 110)),
        "random_mapping": _pair_summary(actual_vectors, random_vectors, BOOTSTRAP_SEED_OFFSET + (20 if kind == "mlp" else 120)),
        "canonical_beats_raw": _win_rate(canonical_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + (30 if kind == "mlp" else 130)),
        "canonical_beats_random": _win_rate(canonical_errors, random_errors, BOOTSTRAP_SEED_OFFSET + (40 if kind == "mlp" else 140)),
        "mapping_validation_by_layer": {
            str(layer): validation for layer, (_, validation) in mapping_by_layer.items()
        },
        "cases": case_summaries,
    }
    return summary


def _activation_patching(
    model: NumpySmolLM,
    prompt_ids: list[list[int]],
    heldout_offset: int,
    baseline: list[np.ndarray],
    traces: list[list[np.ndarray]],
    selected: dict[int, tuple[int, ...]],
    target_ids: list[int],
) -> dict[str, object]:
    aligned_effects: list[np.ndarray] = []
    raw_effects: list[np.ndarray] = []
    original_effects: list[np.ndarray] = []
    cases: list[dict[str, object]] = []
    for layer in SELECTED_LAYERS:
        transform, mapping_validation = _blind_mapping(
            model, "mlp", layer, model.ffn_dim, MLP_TRANSFORM_SHIFT,
        )
        inverse = np.argsort(transform)
        transformed_kwargs = {"permutation": transform, "permutation_layer": layer}
        transformed_base = _target_output_cache(model, prompt_ids, target_ids, **transformed_kwargs)
        for old_index in selected[layer]:
            new_index = int(inverse[old_index])
            for prompt_index in range(heldout_offset, len(prompt_ids)):
                target_activation = traces[prompt_index][layer]
                source_activation = traces[0][layer]
                original_patch_activation = target_activation.copy()
                original_patch_activation[old_index] = source_activation[old_index]
                original_patched = _target_logits(
                    model,
                    prompt_ids[prompt_index],
                    target_ids,
                    patch_mlp_activation=original_patch_activation,
                    patch_mlp_activation_layer=layer,
                )
                transformed_activation = target_activation[transform]
                aligned_patch_activation = transformed_activation.copy()
                aligned_patch_activation[new_index] = source_activation[old_index]
                raw_patch_activation = transformed_activation.copy()
                raw_patch_activation[old_index] = source_activation[old_index]
                aligned_patched = _target_logits(
                    model,
                    prompt_ids[prompt_index],
                    target_ids,
                    **transformed_kwargs,
                    patch_mlp_activation=aligned_patch_activation,
                    patch_mlp_activation_layer=layer,
                )
                raw_patched = _target_logits(
                    model,
                    prompt_ids[prompt_index],
                    target_ids,
                    **transformed_kwargs,
                    patch_mlp_activation=raw_patch_activation,
                    patch_mlp_activation_layer=layer,
                )
                original_effect = original_patched - baseline[prompt_index]
                aligned_effect = aligned_patched - transformed_base[prompt_index]
                raw_effect = raw_patched - transformed_base[prompt_index]
                original_effects.append(original_effect)
                aligned_effects.append(aligned_effect)
                raw_effects.append(raw_effect)
                cases.append({
                    "layer": layer,
                    "mapping_validation": mapping_validation,
                    "original_channel": int(old_index),
                    "transformed_channel": new_index,
                    "heldout_index": prompt_index - heldout_offset,
                    "aligned_relative_l2": round(_relative_l2(original_effect, aligned_effect), 10),
                    "raw_relative_l2": round(_relative_l2(original_effect, raw_effect), 10),
                })
    aligned_errors = [_relative_l2(left, right) for left, right in zip(original_effects, aligned_effects)]
    raw_errors = [_relative_l2(left, right) for left, right in zip(original_effects, raw_effects)]
    return {
        "source_probe_index": 0,
        "source_probe_digest": full_digest(FRESH_VISIBLE_PROMPTS[0]),
        "component_case_count": len(selected) * CHANNELS_PER_LAYER,
        "observation_count": len(cases),
        "aligned_canonical": _pair_summary(original_effects, aligned_effects, BOOTSTRAP_SEED_OFFSET + 200),
        "raw_coordinate": _pair_summary(original_effects, raw_effects, BOOTSTRAP_SEED_OFFSET + 210),
        "canonical_beats_raw": _win_rate(aligned_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + 220),
        "cases": cases,
    }


def _activation_rescue(
    model: NumpySmolLM,
    prompt_ids: list[list[int]],
    heldout_offset: int,
    baseline: list[np.ndarray],
    traces: list[list[np.ndarray]],
    target_ids: list[int],
) -> dict[str, object]:
    aligned_outputs: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    original_outputs: list[np.ndarray] = []
    cases: list[dict[str, object]] = []
    for layer in SELECTED_LAYERS:
        transform, mapping_validation = _blind_mapping(
            model, "mlp", layer, model.ffn_dim, MLP_TRANSFORM_SHIFT,
        )
        transformed_kwargs = {"permutation": transform, "permutation_layer": layer}
        for prompt_index in range(heldout_offset, len(prompt_ids)):
            original_activation = traces[prompt_index][layer]
            transformed_activation = original_activation[transform]
            aligned = _target_logits(
                model,
                prompt_ids[prompt_index],
                target_ids,
                **transformed_kwargs,
                patch_mlp_activation=transformed_activation,
                patch_mlp_activation_layer=layer,
            )
            raw = _target_logits(
                model,
                prompt_ids[prompt_index],
                target_ids,
                **transformed_kwargs,
                patch_mlp_activation=original_activation,
                patch_mlp_activation_layer=layer,
            )
            original = baseline[prompt_index]
            aligned_outputs.append(aligned)
            raw_outputs.append(raw)
            original_outputs.append(original)
            cases.append({
                "layer": layer,
                "mapping_validation": mapping_validation,
                "heldout_index": prompt_index - heldout_offset,
                "aligned_relative_l2": round(_relative_l2(original, aligned), 10),
                "raw_relative_l2": round(_relative_l2(original, raw), 10),
            })
    aligned_errors = [_relative_l2(left, right) for left, right in zip(original_outputs, aligned_outputs)]
    raw_errors = [_relative_l2(left, right) for left, right in zip(original_outputs, raw_outputs)]
    return {
        "layer_count": len(SELECTED_LAYERS),
        "heldout_probe_count": len(prompt_ids) - heldout_offset,
        "observation_count": len(cases),
        "aligned_canonical": _pair_summary(original_outputs, aligned_outputs, BOOTSTRAP_SEED_OFFSET + 300),
        "raw_coordinate": _pair_summary(original_outputs, raw_outputs, BOOTSTRAP_SEED_OFFSET + 310),
        "canonical_beats_raw": _win_rate(aligned_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + 320),
        "cases": cases,
    }


def run(model_dir: Path, model_id: str = "smollm2-135m") -> dict[str, object]:
    checkpoint = SmolLMCheckpoint(model_dir, model_id)
    model = NumpySmolLM(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    visible_ids = _prompt_ids(tokenizer, FRESH_VISIBLE_PROMPTS)
    heldout_ids = _prompt_ids(tokenizer, FRESH_HELDOUT_PROMPTS)
    prompt_ids = visible_ids + heldout_ids
    target_ids = _target_ids(tokenizer)
    heldout_offset = len(visible_ids)
    baseline = _target_output_cache(model, prompt_ids, target_ids)
    traces = [model.mlp_activation_trace(ids) for ids in prompt_ids]
    selected, selection_details = _select_channels(model, traces[:heldout_offset])
    mlp_prediction = _mapping_prediction(
        model,
        prompt_ids,
        baseline,
        heldout_offset,
        selected,
        "mlp",
        model.ffn_dim,
        MLP_TRANSFORM_SHIFT,
        RANDOM_SEED,
        target_ids,
    )
    gqa_prediction = _mapping_prediction(
        model,
        prompt_ids,
        baseline,
        heldout_offset,
        selected,
        "gqa",
        model.n_kv_head,
        GQA_TRANSFORM_SHIFT,
        RANDOM_SEED + 1,
        target_ids,
    )
    patching = _activation_patching(
        model,
        prompt_ids,
        heldout_offset,
        baseline,
        traces,
        selected,
        target_ids,
    )
    rescue = _activation_rescue(
        model,
        prompt_ids,
        heldout_offset,
        baseline,
        traces,
        target_ids,
    )
    canonical_prediction_pass = all(
        float(summary["canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(summary["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
        and float(summary["canonical_beats_random"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
        for summary in (mlp_prediction, gqa_prediction)
    )
    patch_pass = (
        float(patching["aligned_canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(patching["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    rescue_pass = (
        float(rescue["aligned_canonical"]["mean_relative_l2"]) < RESCUE_RELATIVE_L2_MAX
        and float(rescue["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    overall_pass = canonical_prediction_pass and patch_pass and rescue_pass
    return {
        "status": "CAUSAL_PREDICTION_AND_RESCUE_PASS" if overall_pass else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_CAUSAL_PREDICTION_AND_RESCUE_OBSERVATION_ONLY",
        "state_slice": STATE_SLICE,
        "protocol_id": PROTOCOL_ID,
        "pre_registered_protocol": {
            "model_id": checkpoint.model_id,
            "probe_suite_name": "smollm2_fresh_v1",
            "selection_probe_set": "8 visible probes only",
            "evaluation_probe_set": "64 held-out probes only",
            "selected_layers": list(SELECTED_LAYERS),
            "mlp_channels_per_layer": CHANNELS_PER_LAYER,
            "gqa_groups": list(GQA_GROUPS),
            "mlp_transformation": f"permutation_shift_{MLP_TRANSFORM_SHIFT}",
            "gqa_transformation": f"permutation_shift_{GQA_TRANSFORM_SHIFT}",
            "random_mapping_seed": RANDOM_SEED,
            "bootstrap_seed_offset": BOOTSTRAP_SEED_OFFSET,
            "target_logit_tokens": list(TARGET_TOKENS),
            "prediction_relative_l2_max": PREDICTION_RELATIVE_L2_MAX,
            "rescue_relative_l2_max": RESCUE_RELATIVE_L2_MAX,
            "canonical_win_rate_min": CAUSAL_PREDICTION_WIN_RATE_MIN,
        },
        "checkpoint": {
            "weights_loaded": True,
            "loader": "independent_numpy_llama_from_safetensors",
            "checkpoint_digest": checkpoint.checkpoint_digest,
            "model_id": checkpoint.model_id,
            "revision": checkpoint.revision,
            "runtime_id": checkpoint.runtime_id,
            "tensor_count": len(checkpoint.tensors),
        },
        "probe_set_digest": full_digest({
            "visible": list(FRESH_VISIBLE_PROMPTS),
            "heldout": list(FRESH_HELDOUT_PROMPTS),
            "target_tokens": list(TARGET_TOKENS),
        }),
        "selection": {
            "visible_probe_count": heldout_offset,
            "visible_probe_digest": full_digest(list(FRESH_VISIBLE_PROMPTS)),
            "channels_by_layer": {str(layer): list(channels) for layer, channels in selected.items()},
            "details_by_layer": {str(layer): details for layer, details in selection_details.items()},
        },
        "prediction": {
            "mlp": mlp_prediction,
            "gqa": gqa_prediction,
            "pass": canonical_prediction_pass,
        },
        "activation_patching": {"result": patching, "pass": patch_pass},
        "activation_rescue": {"result": rescue, "pass": rescue_pass},
        "evidence": {
            "model_weights_loaded": True,
            "provider_called": False,
            "real_assessment_data_consumed": False,
            "assessment_status": "local_synthetic_probe_suite_only",
            "external_review_status": "not_run",
        },
        "nonclaims": [
            "not mechanistic understanding",
            "not architecture-general proof",
            "not alignment validation",
            "not independent review",
            "not production readiness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="smollm2-135m")
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
