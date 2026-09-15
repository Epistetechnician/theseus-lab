"""Fresh-contract causal prediction, patching, and rescue for OPT-125M."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from causal_prediction_check import (
    _pair_summary,
    _prompt_ids,
    _relative_l2,
    _target_ids,
    _target_logits,
    _target_output_cache,
    _win_rate,
)
from gpt2_experiment import full_digest
from opt_experiment import NumpyOPT, OPTCheckpoint, tensor_digest
from reference_check import (
    canonicalize_attention,
    canonicalize_mlp,
    infer_attention_permutation,
    infer_mlp_permutation,
    permutation,
)


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "opt125m-causal-prediction-v1"
FRESH_VISIBLE_PROMPTS = (
    "A compass needle points",
    "The planet closest to the Sun is",
    "A hexagon has six",
    "The substance plants release during photosynthesis is",
    "An adult human has two",
    "A violin is played with a",
    "The freezing point of water in Celsius is",
    "A calendar contains months",
)
FRESH_HELDOUT_PROMPTS = (
    "A lighthouse helps ships find the",
    "A spider has eight",
    "The opposite of shallow is",
    "A compass direction between north and east is",
    "A rectangle has four right",
    "A caterpillar becomes a",
    "The largest planet in our solar system is",
    "A thermometer measures",
    "The process by which plants make food is",
    "A violin has four",
    "A calendar year has twelve",
    "The chemical symbol for oxygen is",
    "A baby dog is called a",
    "A triangle with three equal sides is",
    "A book is made of pages and a",
    "Rain falls from",
    "A baker makes",
    "A map uses symbols to show",
    "The opposite of victory is",
    "A person who paints is an",
    "The sun is a",
    "A river flows toward the",
    "The organ that pumps blood is the",
    "A week begins after the",
    "A square has four equal",
    "A bird uses wings to",
    "A person who studies stars is an",
    "The color of fresh grass is usually",
    "A key is used to open a",
    "A clock has hands that show the",
    "A group of stars is called a",
    "A whale is a kind of",
    "The opposite of noisy is",
    "A bicycle is powered by",
    "A farmer plants",
    "A mirror shows a",
    "A doctor works in a",
    "The moon orbits the",
    "An octagon has eight",
    "A raincoat protects against",
    "A ruler measures",
    "A plane flies through the",
    "The season after autumn is",
    "A library lends",
    "A magnet attracts some kinds of",
    "A cup holds a",
    "A seed can grow into a",
    "A person who writes books is an",
    "The opposite of early is",
    "A bridge goes over a",
    "A microscope makes small things look",
    "An island is surrounded by",
    "A stove is used to heat",
    "The number before ten is",
    "A helmet protects the",
    "A train travels on",
    "A camera takes a",
    "The Earth has one natural",
    "A musician may play a",
    "A desert gets little",
    "A museum contains",
    "The opposite of hard is",
    "A clock tells the",
    "A comet has a tail",
)
SELECTED_LAYERS = (0, 5, 11)
CHANNELS_PER_LAYER = 3
SELECTED_HEADS = (0, 1, 2)
MLP_TRANSFORM_SHIFT = 17
HEAD_TRANSFORM_SHIFT = 1
RANDOM_SEED = 20260915
BOOTSTRAP_SEED_OFFSET = 62000
PREDICTION_RELATIVE_L2_MAX = 1e-4
RESCUE_RELATIVE_L2_MAX = 1e-4
CAUSAL_PREDICTION_WIN_RATE_MIN = 0.8
TARGET_TOKENS = (" the", " is", " of", " and")


def _component_effects(
    model: NumpyOPT,
    prompt_ids: list[list[int]],
    baseline: list[np.ndarray],
    target_ids: list[int],
    kind: str,
    layer: int,
    index: int,
) -> list[np.ndarray]:
    effects: list[np.ndarray] = []
    for ids, original in zip(prompt_ids, baseline):
        kwargs = (
            {"ablate_channel": index, "ablate_channel_layer": layer}
            if kind == "mlp"
            else {"ablate_head": index, "ablate_head_layer": layer}
        )
        altered = _target_logits(model, ids, target_ids, **kwargs)
        effects.append(original - altered)
    return effects


def _blind_mapping(
    model: NumpyOPT,
    kind: str,
    layer: int,
    width: int,
    shift: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Infer the mapping from component signatures, without using the shift."""

    recorded_permutation = permutation(width, shift)
    if kind == "mlp":
        before = model.mlp_components(layer)
        after = [before[0][:, recorded_permutation], before[1][recorded_permutation], before[2][recorded_permutation, :]]
        canonical_before, _ = canonicalize_mlp(*before)
        canonical_after, _ = canonicalize_mlp(*after)
        inferred = infer_mlp_permutation(before, after)
        before_digest = tensor_digest({"weight": before[0], "bias": before[1], "projection": before[2]})
        after_digest = tensor_digest({"weight": after[0], "bias": after[1], "projection": after[2]})
        canonical_before_digest = tensor_digest({"weight": canonical_before[0], "bias": canonical_before[1], "projection": canonical_before[2]})
        canonical_after_digest = tensor_digest({"weight": canonical_after[0], "bias": canonical_after[1], "projection": canonical_after[2]})
    elif kind == "head":
        before = model.attention_components(layer)
        head_indexes = np.arange(model.n_head)[:, None] * model.head_dim + np.arange(model.head_dim)
        flat = head_indexes[recorded_permutation].reshape(-1)
        qkv_order = np.concatenate((flat, model.n_embd + flat, 2 * model.n_embd + flat))
        after = [before[0][:, qkv_order], before[1][qkv_order], before[2][flat, :]]
        canonical_before, _ = canonicalize_attention(
            *before, model.head_dim, model.n_embd, model.n_head,
        )
        canonical_after, _ = canonicalize_attention(
            *after, model.head_dim, model.n_embd, model.n_head,
        )
        inferred = infer_attention_permutation(
            before, after, model.head_dim, model.n_embd, model.n_head,
        )
        before_digest = tensor_digest({"qkv_weight": before[0], "qkv_bias": before[1], "projection": before[2]})
        after_digest = tensor_digest({"qkv_weight": after[0], "qkv_bias": after[1], "projection": after[2]})
        canonical_before_digest = tensor_digest({"qkv_weight": canonical_before[0], "qkv_bias": canonical_before[1], "projection": canonical_before[2]})
        canonical_after_digest = tensor_digest({"qkv_weight": canonical_after[0], "qkv_bias": canonical_after[1], "projection": canonical_after[2]})
    else:
        raise ValueError(f"unknown component kind: {kind}")
    if not np.array_equal(inferred, recorded_permutation):
        raise ValueError(f"blind {kind} mapping did not recover the recorded transformation")
    if before_digest == after_digest or canonical_before_digest != canonical_after_digest:
        raise ValueError(f"blind {kind} digest criterion failed")
    return inferred, {
        "raw_digest_diff": True,
        "canonical_digest_match": True,
        "before_digest": before_digest,
        "after_digest": after_digest,
        "canonical_before_digest": canonical_before_digest,
        "canonical_after_digest": canonical_after_digest,
        "inferred_mapping_digest": full_digest(inferred.tolist()),
        "mapping_source": "blind_exact_component_signature_inference",
    }


def _select_channels(
    model: NumpyOPT,
    visible_traces: list[list[np.ndarray]],
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, object]]]:
    selected: dict[int, tuple[int, ...]] = {}
    details: dict[int, dict[str, object]] = {}
    for layer in SELECTED_LAYERS:
        matrix = np.stack([trace[layer] for trace in visible_traces])
        active_count = np.count_nonzero(matrix > 0.0, axis=0)
        activation_sum = np.sum(matrix, axis=0)
        order = sorted(
            range(model.ffn_dim),
            key=lambda index: (-int(active_count[index]), -float(activation_sum[index]), index),
        )
        channels = tuple(int(index) for index in order[:CHANNELS_PER_LAYER])
        if any(int(active_count[index]) == 0 for index in channels):
            raise ValueError(f"selected inactive OPT channels at layer {layer}")
        selected[layer] = channels
        details[layer] = {
            "channels": list(channels),
            "visible_active_probe_counts": [int(active_count[index]) for index in channels],
            "visible_activation_sums": [round(float(activation_sum[index]), 10) for index in channels],
        }
    return selected, details


def _mapping_prediction(
    model: NumpyOPT,
    prompt_ids: list[list[int]],
    baseline: list[np.ndarray],
    target_ids: list[int],
    heldout_offset: int,
    selected: dict[int, tuple[int, ...]],
    kind: str,
    width: int,
    shift: int,
    seed: int,
) -> dict[str, object]:
    random_source = np.random.default_rng(seed).permutation(width)
    mapping_by_layer = {
        layer: _blind_mapping(model, kind, layer, width, shift)
        for layer in SELECTED_LAYERS
    }
    effects: dict[tuple[int, int], list[np.ndarray]] = {}
    needed_by_layer: dict[int, set[int]] = {}
    for layer in SELECTED_LAYERS:
        transform, _ = mapping_by_layer[layer]
        inverse = np.argsort(transform)
        indexes = selected[layer] if kind == "mlp" else SELECTED_HEADS
        needed = {int(index) for index in indexes}
        needed.update(int(inverse[index]) for index in indexes)
        needed.update(int(random_source[inverse[index]]) for index in indexes)
        needed_by_layer[layer] = needed
        for index in sorted(needed):
            effects[(layer, index)] = _component_effects(
                model, prompt_ids, baseline, target_ids, kind, layer, index,
            )
    actual_vectors: list[np.ndarray] = []
    canonical_vectors: list[np.ndarray] = []
    raw_vectors: list[np.ndarray] = []
    random_vectors: list[np.ndarray] = []
    canonical_errors: list[float] = []
    raw_errors: list[float] = []
    random_errors: list[float] = []
    layer_summaries: list[dict[str, object]] = []
    for layer in SELECTED_LAYERS:
        transform, validation = mapping_by_layer[layer]
        inverse = np.argsort(transform)
        indexes = selected[layer] if kind == "mlp" else SELECTED_HEADS
        transformed_kwargs = (
            {"permutation": transform, "permutation_layer": layer}
            if kind == "mlp"
            else {"attention_permutation": transform, "attention_permutation_layer": layer}
        )
        transformed_base = _target_output_cache(model, prompt_ids, target_ids, **transformed_kwargs)
        layer_actual: list[np.ndarray] = []
        layer_canonical: list[np.ndarray] = []
        layer_raw: list[np.ndarray] = []
        layer_random: list[np.ndarray] = []
        for old_index in indexes:
            new_index = int(inverse[old_index])
            random_index = int(random_source[new_index])
            for prompt_index in range(heldout_offset, len(prompt_ids)):
                intervention_kwargs = dict(transformed_kwargs)
                if kind == "mlp":
                    intervention_kwargs.update({"ablate_channel": new_index, "ablate_channel_layer": layer})
                else:
                    intervention_kwargs.update({"ablate_head": new_index, "ablate_head_layer": layer})
                altered = _target_logits(model, prompt_ids[prompt_index], target_ids, **intervention_kwargs)
                actual = transformed_base[prompt_index] - altered
                canonical = effects[(layer, int(old_index))][prompt_index]
                raw = effects[(layer, new_index)][prompt_index]
                random = effects[(layer, random_index)][prompt_index]
                layer_actual.append(actual)
                layer_canonical.append(canonical)
                layer_raw.append(raw)
                layer_random.append(random)
        actual_vectors.extend(layer_actual)
        canonical_vectors.extend(layer_canonical)
        raw_vectors.extend(layer_raw)
        random_vectors.extend(layer_random)
        canonical_errors.extend(_relative_l2(left, right) for left, right in zip(layer_actual, layer_canonical))
        raw_errors.extend(_relative_l2(left, right) for left, right in zip(layer_actual, layer_raw))
        random_errors.extend(_relative_l2(left, right) for left, right in zip(layer_actual, layer_random))
        layer_summaries.append({
            "layer": layer,
            "component_case_count": len(indexes),
            "mapping_validation": validation,
            "canonical_mean_relative_l2": round(float(np.mean([_relative_l2(left, right) for left, right in zip(layer_actual, layer_canonical)])), 10),
            "raw_mean_relative_l2": round(float(np.mean([_relative_l2(left, right) for left, right in zip(layer_actual, layer_raw)])), 10),
            "random_mean_relative_l2": round(float(np.mean([_relative_l2(left, right) for left, right in zip(layer_actual, layer_random)])), 10),
        })
    return {
        "kind": kind,
        "component_case_count": len(SELECTED_LAYERS) * len(selected[SELECTED_LAYERS[0]] if kind == "mlp" else SELECTED_HEADS),
        "heldout_probe_count": len(prompt_ids) - heldout_offset,
        "canonical": _pair_summary(actual_vectors, canonical_vectors, BOOTSTRAP_SEED_OFFSET + (0 if kind == "mlp" else 100)),
        "raw_coordinate": _pair_summary(actual_vectors, raw_vectors, BOOTSTRAP_SEED_OFFSET + (10 if kind == "mlp" else 110)),
        "random_mapping": _pair_summary(actual_vectors, random_vectors, BOOTSTRAP_SEED_OFFSET + (20 if kind == "mlp" else 120)),
        "canonical_beats_raw": _win_rate(canonical_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + (30 if kind == "mlp" else 130)),
        "canonical_beats_random": _win_rate(canonical_errors, random_errors, BOOTSTRAP_SEED_OFFSET + (40 if kind == "mlp" else 140)),
        "mapping_validation_by_layer": {str(layer): validation for layer, (_, validation) in mapping_by_layer.items()},
        "layers": layer_summaries,
    }


def _activation_patching(
    model: NumpyOPT,
    prompt_ids: list[list[int]],
    target_ids: list[int],
    heldout_offset: int,
    baseline: list[np.ndarray],
    traces: list[list[np.ndarray]],
    selected: dict[int, tuple[int, ...]],
) -> dict[str, object]:
    original_effects: list[np.ndarray] = []
    aligned_effects: list[np.ndarray] = []
    raw_effects: list[np.ndarray] = []
    for layer in SELECTED_LAYERS:
        transform, _ = _blind_mapping(model, "mlp", layer, model.ffn_dim, MLP_TRANSFORM_SHIFT)
        inverse = np.argsort(transform)
        transformed_kwargs = {"permutation": transform, "permutation_layer": layer}
        transformed_base = _target_output_cache(model, prompt_ids, target_ids, **transformed_kwargs)
        for old_index in selected[layer]:
            new_index = int(inverse[old_index])
            for prompt_index in range(heldout_offset, len(prompt_ids)):
                target_activation = traces[prompt_index][layer]
                source_activation = traces[0][layer]
                original_patch = target_activation.copy()
                original_patch[old_index] = source_activation[old_index]
                original_patched = _target_logits(
                    model, prompt_ids[prompt_index], target_ids,
                    patch_mlp_activation=original_patch,
                    patch_mlp_activation_layer=layer,
                )
                transformed_activation = target_activation[transform]
                aligned_patch = transformed_activation.copy()
                aligned_patch[new_index] = source_activation[old_index]
                raw_patch = transformed_activation.copy()
                raw_patch[old_index] = source_activation[old_index]
                aligned_patched = _target_logits(
                    model, prompt_ids[prompt_index], target_ids,
                    **transformed_kwargs,
                    patch_mlp_activation=aligned_patch,
                    patch_mlp_activation_layer=layer,
                )
                raw_patched = _target_logits(
                    model, prompt_ids[prompt_index], target_ids,
                    **transformed_kwargs,
                    patch_mlp_activation=raw_patch,
                    patch_mlp_activation_layer=layer,
                )
                original_effects.append(original_patched - baseline[prompt_index])
                aligned_effects.append(aligned_patched - transformed_base[prompt_index])
                raw_effects.append(raw_patched - transformed_base[prompt_index])
    aligned_errors = [_relative_l2(left, right) for left, right in zip(original_effects, aligned_effects)]
    raw_errors = [_relative_l2(left, right) for left, right in zip(original_effects, raw_effects)]
    return {
        "observation_count": len(original_effects),
        "aligned_canonical": _pair_summary(original_effects, aligned_effects, BOOTSTRAP_SEED_OFFSET + 200),
        "raw_coordinate": _pair_summary(original_effects, raw_effects, BOOTSTRAP_SEED_OFFSET + 210),
        "canonical_beats_raw": _win_rate(aligned_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + 220),
    }


def _activation_rescue(
    model: NumpyOPT,
    prompt_ids: list[list[int]],
    target_ids: list[int],
    heldout_offset: int,
    baseline: list[np.ndarray],
    traces: list[list[np.ndarray]],
) -> dict[str, object]:
    original_outputs: list[np.ndarray] = []
    aligned_outputs: list[np.ndarray] = []
    raw_outputs: list[np.ndarray] = []
    for layer in SELECTED_LAYERS:
        transform, _ = _blind_mapping(model, "mlp", layer, model.ffn_dim, MLP_TRANSFORM_SHIFT)
        transformed_kwargs = {"permutation": transform, "permutation_layer": layer}
        for prompt_index in range(heldout_offset, len(prompt_ids)):
            activation = traces[prompt_index][layer]
            transformed_activation = activation[transform]
            aligned = _target_logits(
                model, prompt_ids[prompt_index], target_ids,
                **transformed_kwargs,
                patch_mlp_activation=transformed_activation,
                patch_mlp_activation_layer=layer,
            )
            raw = _target_logits(
                model, prompt_ids[prompt_index], target_ids,
                **transformed_kwargs,
                patch_mlp_activation=activation,
                patch_mlp_activation_layer=layer,
            )
            original_outputs.append(baseline[prompt_index])
            aligned_outputs.append(aligned)
            raw_outputs.append(raw)
    aligned_errors = [_relative_l2(left, right) for left, right in zip(original_outputs, aligned_outputs)]
    raw_errors = [_relative_l2(left, right) for left, right in zip(original_outputs, raw_outputs)]
    return {
        "observation_count": len(original_outputs),
        "aligned_canonical": _pair_summary(original_outputs, aligned_outputs, BOOTSTRAP_SEED_OFFSET + 300),
        "raw_coordinate": _pair_summary(original_outputs, raw_outputs, BOOTSTRAP_SEED_OFFSET + 310),
        "canonical_beats_raw": _win_rate(aligned_errors, raw_errors, BOOTSTRAP_SEED_OFFSET + 320),
    }


def run(model_dir: Path, model_id: str = "opt-125m") -> dict[str, object]:
    checkpoint = OPTCheckpoint(model_dir, model_id)
    model = NumpyOPT(checkpoint)
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
        model, prompt_ids, baseline, target_ids, heldout_offset, selected,
        "mlp", model.ffn_dim, MLP_TRANSFORM_SHIFT, RANDOM_SEED,
    )
    head_prediction = _mapping_prediction(
        model, prompt_ids, baseline, target_ids, heldout_offset, selected,
        "head", model.n_head, HEAD_TRANSFORM_SHIFT, RANDOM_SEED + 1,
    )
    patching = _activation_patching(
        model, prompt_ids, target_ids, heldout_offset, baseline, traces, selected,
    )
    rescue = _activation_rescue(
        model, prompt_ids, target_ids, heldout_offset, baseline, traces,
    )
    prediction_pass = all(
        float(summary["canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(summary["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
        and float(summary["canonical_beats_random"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
        for summary in (mlp_prediction, head_prediction)
    )
    patch_pass = (
        float(patching["aligned_canonical"]["mean_relative_l2"]) < PREDICTION_RELATIVE_L2_MAX
        and float(patching["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    rescue_pass = (
        float(rescue["aligned_canonical"]["mean_relative_l2"]) < RESCUE_RELATIVE_L2_MAX
        and float(rescue["canonical_beats_raw"]["canonical_win_rate"]) >= CAUSAL_PREDICTION_WIN_RATE_MIN
    )
    return {
        "status": "OPT_CAUSAL_PREDICTION_AND_RESCUE_PASS" if prediction_pass and patch_pass and rescue_pass else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_OPT_CAUSAL_PREDICTION_AND_RESCUE_OBSERVATION_ONLY",
        "state_slice": STATE_SLICE,
        "protocol_id": PROTOCOL_ID,
        "pre_registered_protocol": {
            "model_id": checkpoint.model_id,
            "probe_suite_name": "opt125m_fresh_v1",
            "selection_probe_set": "8 visible probes only",
            "evaluation_probe_set": "64 held-out probes only",
            "selected_layers": list(SELECTED_LAYERS),
            "mlp_channels_per_layer": CHANNELS_PER_LAYER,
            "selected_attention_heads": list(SELECTED_HEADS),
            "mlp_transformation": f"permutation_shift_{MLP_TRANSFORM_SHIFT}",
            "attention_transformation": f"permutation_shift_{HEAD_TRANSFORM_SHIFT}",
            "random_mapping_seed": RANDOM_SEED,
            "bootstrap_seed_offset": BOOTSTRAP_SEED_OFFSET,
            "target_logit_tokens": list(TARGET_TOKENS),
            "prediction_relative_l2_max": PREDICTION_RELATIVE_L2_MAX,
            "rescue_relative_l2_max": RESCUE_RELATIVE_L2_MAX,
            "canonical_win_rate_min": CAUSAL_PREDICTION_WIN_RATE_MIN,
        },
        "checkpoint": {
            "weights_loaded": True,
            "loader": "independent_numpy_opt_from_pytorch_state_dict",
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
        "prediction": {"mlp": mlp_prediction, "attention_head": head_prediction, "pass": prediction_pass},
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
    parser.add_argument("--model-id", default="opt-125m")
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
