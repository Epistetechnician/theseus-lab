"""Independent reference and symmetry sweep for the local SmolLM adapter."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from gpt2_experiment import (
    HELDOUT_PROMPTS,
    VISIBLE_PROMPTS,
    compare_logits,
    full_digest,
    vector_similarity,
)
from reference_check import (
    ALIGNED_INTERVENTION_COSINE_MIN,
    ALIGNED_INTERVENTION_MAX_ABS,
    ATLAS_CHANNEL_CANDIDATES,
    ATTENTION_SHIFTS,
    BEHAVIOR_CHANGE_RELATIVE_L2_MIN,
    MLP_SHIFTS,
    PRUNE_COUNT,
    REPLACEMENT_CHANNEL,
    REFERENCE_RELATIVE_L2_MAX,
    SYMMETRY_RELATIVE_L2_MAX,
    bootstrap_ci,
    changed_fraction,
    identity_criterion,
    permutation,
    reference_logits,
    split_behavior,
)
from smollm_experiment import NumpySmolLM, SmolLMCheckpoint, tensor_digest
from smollm_jspace_proxy import analyze_jspace


INTERVENTION_HELDOUT_COUNT = 16


def _gated_signature(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    index: int,
) -> tuple[bytes, bytes, bytes]:
    return (
        np.ascontiguousarray(gate[:, index]).tobytes(),
        np.ascontiguousarray(up[:, index]).tobytes(),
        np.ascontiguousarray(down[index, :]).tobytes(),
    )


def canonicalize_gated_mlp(
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.asarray(sorted(
        range(gate.shape[1]),
        key=lambda index: _gated_signature(gate, up, down, index),
    ), dtype=np.int64)
    return [gate[:, order], up[:, order], down[order, :]], order


def infer_gated_mlp_permutation(
    before: list[np.ndarray],
    after: list[np.ndarray],
) -> np.ndarray:
    before_signatures = {
        _gated_signature(before[0], before[1], before[2], index): index
        for index in range(before[0].shape[1])
    }
    if len(before_signatures) != before[0].shape[1]:
        raise ValueError("gated MLP channel signatures are not unique")
    try:
        inferred = np.asarray([
            before_signatures[_gated_signature(after[0], after[1], after[2], index)]
            for index in range(after[0].shape[1])
        ], dtype=np.int64)
    except KeyError as error:
        raise ValueError("transformed gated MLP contains an unmatched channel") from error
    if len(set(inferred.tolist())) != len(inferred):
        raise ValueError("inferred gated MLP mapping is not a permutation")
    return inferred


def _gqa_signature(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    output: np.ndarray,
    group: int,
    head_dim: int,
    query_heads_per_group: int,
) -> tuple[tuple[bytes, ...], bytes, bytes, tuple[bytes, ...]]:
    query_heads = range(group * query_heads_per_group, (group + 1) * query_heads_per_group)
    query_indexes = [head * head_dim + offset for head in query_heads for offset in range(head_dim)]
    kv_indexes = range(group * head_dim, (group + 1) * head_dim)
    return (
        tuple(np.ascontiguousarray(query[head * head_dim:(head + 1) * head_dim, :]).tobytes() for head in query_heads),
        np.ascontiguousarray(key[list(kv_indexes), :]).tobytes(),
        np.ascontiguousarray(value[list(kv_indexes), :]).tobytes(),
        tuple(np.ascontiguousarray(output[:, index]).tobytes() for index in query_indexes),
    )


def canonicalize_gqa(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    output: np.ndarray,
    head_dim: int,
    kv_groups: int,
    query_heads_per_group: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.asarray(sorted(
        range(kv_groups),
        key=lambda group: _gqa_signature(
            query, key, value, output, group, head_dim, query_heads_per_group,
        ),
    ), dtype=np.int64)
    query_indexes = np.concatenate([
        np.arange(group * query_heads_per_group * head_dim, (group + 1) * query_heads_per_group * head_dim)
        for group in order
    ])
    kv_indexes = np.concatenate([
        np.arange(group * head_dim, (group + 1) * head_dim)
        for group in order
    ])
    return [query[query_indexes, :], key[kv_indexes, :], value[kv_indexes, :], output[:, query_indexes]], order


def infer_gqa_permutation(
    before: list[np.ndarray],
    after: list[np.ndarray],
    head_dim: int,
    kv_groups: int,
    query_heads_per_group: int,
) -> np.ndarray:
    before_signatures = {
        _gqa_signature(*before, group, head_dim, query_heads_per_group): group
        for group in range(kv_groups)
    }
    if len(before_signatures) != kv_groups:
        raise ValueError("GQA group signatures are not unique")
    try:
        inferred = np.asarray([
            before_signatures[_gqa_signature(*after, group, head_dim, query_heads_per_group)]
            for group in range(kv_groups)
        ], dtype=np.int64)
    except KeyError as error:
        raise ValueError("transformed GQA attention contains an unmatched group") from error
    if len(set(inferred.tolist())) != len(inferred):
        raise ValueError("inferred GQA mapping is not a permutation")
    return inferred


def select_intervention_channels(
    model: NumpySmolLM,
    visible_tokenized: list[list[int]],
) -> tuple[dict[int, int], dict[int, dict[str, object]]]:
    """Select active gated coordinates on visible probes; held-out probes remain untouched."""

    traces = [model.mlp_activation_trace(ids) for ids in visible_tokenized]
    selected: dict[int, int] = {}
    details: dict[int, dict[str, object]] = {}
    for layer in range(model.n_layer):
        matrix = np.stack([trace[layer] for trace in traces])
        active_count = np.count_nonzero(np.abs(matrix) > 1e-8, axis=0)
        magnitude_sum = np.sum(np.abs(matrix), axis=0)
        order = sorted(
            range(model.ffn_dim),
            key=lambda index: (-int(active_count[index]), -float(magnitude_sum[index]), index),
        )
        channel = int(order[0])
        if int(active_count[channel]) == 0:
            raise ValueError(f"no active SmolLM gated MLP channel found at layer {layer}")
        selected[layer] = channel
        details[layer] = {
            "channel": channel,
            "visible_active_probe_count": int(active_count[channel]),
            "visible_absolute_activation_sum": round(float(magnitude_sum[channel]), 8),
        }
    return selected, details


def aggregate_effect(model: NumpySmolLM, tokenized: list[list[int]], **kwargs: object) -> np.ndarray:
    return np.concatenate([
        model.forward(ids, precision="float64")
        - model.forward(ids, precision="float64", **kwargs)
        for ids in tokenized
    ])


def mlp_case(
    model: NumpySmolLM,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
    intervention_channel: int,
    intervention_tokenized: list[list[int]],
) -> dict[str, object]:
    selected = model.gated_mlp_components(layer)
    perm = permutation(selected[0].shape[1], shift)
    transformed = [selected[0][:, perm], selected[1][:, perm], selected[2][perm, :]]
    variant = [model.forward(ids, permutation=perm, permutation_layer=layer) for ids in tokenized]
    baseline_effect = aggregate_effect(
        model,
        intervention_tokenized,
        ablate_channel=intervention_channel,
        ablate_channel_layer=layer,
    )
    variant_normal = np.concatenate([
        model.forward(ids, permutation=perm, permutation_layer=layer, precision="float64")
        for ids in intervention_tokenized
    ])
    same_label_effect = variant_normal - np.concatenate([
        model.forward(
            ids,
            permutation=perm,
            permutation_layer=layer,
            ablate_channel=intervention_channel,
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    inverse = np.argsort(perm)
    aligned_effect = variant_normal - np.concatenate([
        model.forward(
            ids,
            permutation=perm,
            permutation_layer=layer,
            ablate_channel=int(inverse[intervention_channel]),
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    canonical_before, _ = canonicalize_gated_mlp(*selected)
    canonical_after, _ = canonicalize_gated_mlp(*transformed)
    inferred = infer_gated_mlp_permutation(selected, transformed)
    case = {
        "kind": "gated_mlp_channel_permutation",
        "layer": layer,
        "shift": shift,
        "behavior": split_behavior(baseline, variant),
        "mechanism": {
            "before_digest": tensor_digest({"gate": selected[0], "up": selected[1], "down": selected[2]}),
            "after_digest": tensor_digest({"gate": transformed[0], "up": transformed[1], "down": transformed[2]}),
            "canonical_before_digest": tensor_digest({"gate": canonical_before[0], "up": canonical_before[1], "down": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"gate": canonical_after[0], "up": canonical_after[1], "down": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_gated_component_sort",
            "inferred_permutation_digest": full_digest(inferred.tolist()),
            "blind_mapping_match": bool(np.array_equal(inferred, perm)),
            "changed_parameter_fraction": round(changed_fraction(selected, transformed), 8),
        },
        "intervention": {
            "probe_count": len(intervention_tokenized),
            "baseline_channel": intervention_channel,
            "aligned_variant_channel": int(inverse[intervention_channel]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }
    case["identity_criterion"] = identity_criterion(case)
    return case


def attention_case(
    model: NumpySmolLM,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
    intervention_tokenized: list[list[int]],
) -> dict[str, object]:
    selected = model.gqa_components(layer)
    perm = permutation(model.n_kv_head, shift)
    query_indexes, kv_indexes = model._group_indexes(perm)
    transformed = [selected[0][query_indexes, :], selected[1][kv_indexes, :], selected[2][kv_indexes, :], selected[3][:, query_indexes]]
    variant = [
        model.forward(ids, attention_permutation=perm, attention_permutation_layer=layer)
        for ids in tokenized
    ]
    baseline_effect = aggregate_effect(
        model,
        intervention_tokenized,
        ablate_head_group=0,
        ablate_head_group_layer=layer,
    )
    variant_normal = np.concatenate([
        model.forward(ids, attention_permutation=perm, attention_permutation_layer=layer, precision="float64")
        for ids in intervention_tokenized
    ])
    same_label_effect = variant_normal - np.concatenate([
        model.forward(
            ids,
            attention_permutation=perm,
            attention_permutation_layer=layer,
            ablate_head_group=0,
            ablate_head_group_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    inverse = np.argsort(perm)
    aligned_effect = variant_normal - np.concatenate([
        model.forward(
            ids,
            attention_permutation=perm,
            attention_permutation_layer=layer,
            ablate_head_group=int(inverse[0]),
            ablate_head_group_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    canonical_before, _ = canonicalize_gqa(
        *selected, model.head_dim, model.n_kv_head, model.query_heads_per_kv,
    )
    canonical_after, _ = canonicalize_gqa(
        *transformed, model.head_dim, model.n_kv_head, model.query_heads_per_kv,
    )
    inferred = infer_gqa_permutation(
        selected, transformed, model.head_dim, model.n_kv_head, model.query_heads_per_kv,
    )
    case = {
        "kind": "gqa_attention_group_permutation",
        "layer": layer,
        "shift": shift,
        "behavior": split_behavior(baseline, variant),
        "mechanism": {
            "before_digest": tensor_digest({"query": selected[0], "key": selected[1], "value": selected[2], "output": selected[3]}),
            "after_digest": tensor_digest({"query": transformed[0], "key": transformed[1], "value": transformed[2], "output": transformed[3]}),
            "canonical_before_digest": tensor_digest({"query": canonical_before[0], "key": canonical_before[1], "value": canonical_before[2], "output": canonical_before[3]}),
            "canonical_after_digest": tensor_digest({"query": canonical_after[0], "key": canonical_after[1], "value": canonical_after[2], "output": canonical_after[3]}),
            "canonicalization": "blind_exact_lexicographic_gqa_group_sort",
            "inferred_permutation_digest": full_digest(inferred.tolist()),
            "blind_mapping_match": bool(np.array_equal(inferred, perm)),
            "changed_parameter_fraction": round(changed_fraction(selected, transformed), 8),
        },
        "intervention": {
            "probe_count": len(intervention_tokenized),
            "baseline_group": 0,
            "aligned_variant_group": int(inverse[0]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }
    case["identity_criterion"] = identity_criterion(case)
    return case


def nonsymmetry_case(
    model: NumpySmolLM,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    kind: str,
    channels: np.ndarray | None = None,
    replacement: tuple[int, int] | None = None,
) -> dict[str, object]:
    selected = model.gated_mlp_components(layer)
    transformed = [array.copy() for array in selected]
    if channels is not None:
        transformed[0][:, channels] = 0.0
        transformed[1][:, channels] = 0.0
    if replacement is not None:
        target, source = replacement
        transformed[0][:, target] = transformed[0][:, source]
        transformed[1][:, target] = transformed[1][:, source]
    variant = [
        model.forward(
            ids,
            ablate_channel_layer=layer,
            ablate_channels=channels,
            replacement_channel=replacement,
        )
        for ids in tokenized
    ]
    behavior = split_behavior(baseline, variant)
    behavior_changed = (
        behavior["visible"]["top1_agreement"] < 1.0
        or behavior["heldout"]["top1_agreement"] < 1.0
        or behavior["visible"]["mean_relative_l2"] > BEHAVIOR_CHANGE_RELATIVE_L2_MIN
        or behavior["heldout"]["mean_relative_l2"] > BEHAVIOR_CHANGE_RELATIVE_L2_MIN
    )
    canonical_before, _ = canonicalize_gated_mlp(*selected)
    canonical_after, _ = canonicalize_gated_mlp(*transformed)
    return {
        "kind": kind,
        "layer": layer,
        "behavior": behavior,
        "mechanism": {
            "before_digest": tensor_digest({"gate": selected[0], "up": selected[1], "down": selected[2]}),
            "after_digest": tensor_digest({"gate": transformed[0], "up": transformed[1], "down": transformed[2]}),
            "canonical_before_digest": tensor_digest({"gate": canonical_before[0], "up": canonical_before[1], "down": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"gate": canonical_after[0], "up": canonical_after[1], "down": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_gated_component_sort",
            "changed_parameter_fraction": round(changed_fraction(selected, transformed), 8),
        },
        "behavior_changed": behavior_changed,
        "identity_criterion": {
            "raw_digest_diff": True,
            "canonical_digest_match": False,
            "canonical_digest_diff": True,
            "blind_mapping_match": False,
            "behavior_match": False,
            "aligned_intervention_match": False,
            "accepted": False,
        },
    }


def atlas_channels(model: NumpySmolLM) -> tuple[int, ...]:
    return tuple(index for index in ATLAS_CHANNEL_CANDIDATES if index < model.ffn_dim)


def atlas_groups(model: NumpySmolLM) -> tuple[int, ...]:
    return tuple(range(min(3, model.n_kv_head)))


def atlas_effect(
    model: NumpySmolLM,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    component: str,
    index: int,
) -> dict[str, object]:
    scores: list[float] = []
    top1_changes: list[float] = []
    for ids, original in zip(tokenized, baseline):
        altered = (
            model.forward(ids, ablate_channel=index, ablate_channel_layer=layer)
            if component == "gated_mlp_channel"
            else model.forward(ids, ablate_head_group=index, ablate_head_group_layer=layer)
        )
        scores.append(float(np.linalg.norm(original - altered) / max(np.linalg.norm(original), 1e-12)))
        top1_changes.append(float(np.argmax(original) != np.argmax(altered)))
    return {
        "layer": layer,
        "component": component,
        "index": index,
        "mean_relative_logit_effect": round(float(np.mean(scores)), 8),
        "relative_logit_effect_ci95": bootstrap_ci(scores, layer * 1000 + index + (0 if component == "gated_mlp_channel" else 10000)),
        "top1_change_rate": round(float(np.mean(top1_changes)), 8),
        "top1_change_rate_ci95": bootstrap_ci(top1_changes, layer * 1000 + index + (5000 if component == "gated_mlp_channel" else 15000)),
    }


def run(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    checkpoint = SmolLMCheckpoint(model_dir, model_id)
    numpy_model = NumpySmolLM(checkpoint)
    reference_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.float32,
    ).eval()
    reference_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    prompts = VISIBLE_PROMPTS + HELDOUT_PROMPTS
    tokenized = [reference_tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    numpy_baseline = [numpy_model.forward(ids) for ids in tokenized]
    intervention_channels, intervention_selection = select_intervention_channels(
        numpy_model,
        tokenized[:len(VISIBLE_PROMPTS)],
    )
    intervention_tokenized = tokenized[
        len(VISIBLE_PROMPTS):len(VISIBLE_PROMPTS) + INTERVENTION_HELDOUT_COUNT
    ]
    reference_baseline = [reference_logits(reference_model, reference_tokenizer, prompt) for prompt in prompts]
    reference_metrics = compare_logits(numpy_baseline, reference_baseline)
    reference_pass = (
        reference_metrics["top1_agreement"] == 1.0
        and reference_metrics["mean_relative_l2"] < REFERENCE_RELATIVE_L2_MAX
    )
    cases = [
        mlp_case(
            numpy_model,
            tokenized,
            numpy_baseline,
            layer,
            shift,
            intervention_channels[layer],
            intervention_tokenized,
        )
        for layer in range(numpy_model.n_layer)
        for shift in MLP_SHIFTS
    ]
    cases.extend(
        attention_case(numpy_model, tokenized, numpy_baseline, layer, shift, intervention_tokenized)
        for layer in range(numpy_model.n_layer)
        for shift in ATTENTION_SHIFTS
    )
    nonsymmetry_cases = [
        nonsymmetry_case(
            numpy_model,
            tokenized,
            numpy_baseline,
            layer,
            "gated_mlp_single_channel_ablation",
            channels=np.asarray([intervention_channels[layer]]),
        )
        for layer in range(numpy_model.n_layer)
    ]
    control_layer = min(14, numpy_model.n_layer - 1)
    nonsymmetry_cases.extend((
        nonsymmetry_case(
            numpy_model, tokenized, numpy_baseline, control_layer,
            "gated_mlp_block_pruning", channels=np.arange(PRUNE_COUNT),
        ),
        nonsymmetry_case(
            numpy_model, tokenized, numpy_baseline, control_layer,
            "gated_mlp_channel_replacement", replacement=REPLACEMENT_CHANNEL,
        ),
    ))
    causal_atlas = {
        "gated_mlp_channels": [
            atlas_effect(numpy_model, tokenized, numpy_baseline, layer, "gated_mlp_channel", channel)
            for layer in range(numpy_model.n_layer)
            for channel in atlas_channels(numpy_model)
        ],
        "gqa_groups": [
            atlas_effect(numpy_model, tokenized, numpy_baseline, layer, "gqa_group", group)
            for layer in range(numpy_model.n_layer)
            for group in atlas_groups(numpy_model)
        ],
    }
    jspace_proxy = analyze_jspace(model_dir, checkpoint.model_id)
    overall_pass = (
        reference_pass
        and all(bool(case["identity_criterion"]["accepted"]) for case in cases)
        and all(bool(case["identity_criterion"]["canonical_digest_diff"]) for case in nonsymmetry_cases)
        and len(causal_atlas["gated_mlp_channels"]) == numpy_model.n_layer * len(atlas_channels(numpy_model))
        and len(causal_atlas["gqa_groups"]) == numpy_model.n_layer * len(atlas_groups(numpy_model))
        and jspace_proxy["status"] == "LOCAL_PROXY_PASS"
    )
    return {
        "status": "REFERENCE_SYMMETRY_ATLAS_AND_JSPACE_PROXY_PASS" if overall_pass else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_SMOLLM_TRANSFORMATION_OBSERVATION_ONLY",
        "checkpoint": {
            "weights_loaded": True,
            "loader": "independent_numpy_llama_from_safetensors",
            "checkpoint_digest": checkpoint.checkpoint_digest,
            "model_id": checkpoint.model_id,
            "revision": checkpoint.revision,
            "tensor_count": len(checkpoint.tensors),
            "runtime_id": checkpoint.runtime_id,
            "reference_runtime": ":".join((
                f"python-{platform.python_version()}",
                f"torch-{torch.__version__}",
                f"transformers-{transformers.__version__}",
            )),
        },
        "probe_set_digest": full_digest({"visible": VISIBLE_PROMPTS, "heldout": HELDOUT_PROMPTS}),
        "intervention_selection": {
            "selection_probe_set": "visible",
            "selection_probe_count": len(VISIBLE_PROMPTS),
            "selection_probe_digest": full_digest(VISIBLE_PROMPTS),
            "evaluation_probe_set": "heldout_prefix",
            "evaluation_probe_count": len(intervention_tokenized),
            "evaluation_probe_digest": full_digest(HELDOUT_PROMPTS[:INTERVENTION_HELDOUT_COUNT]),
            "channels_by_layer": intervention_selection,
        },
        "architecture": {
            "family": "llama",
            "layers": numpy_model.n_layer,
            "query_heads": numpy_model.n_head,
            "key_value_heads": numpy_model.n_kv_head,
            "query_heads_per_kv_group": numpy_model.query_heads_per_kv,
            "hidden_size": numpy_model.n_embd,
            "ffn_dim": numpy_model.ffn_dim,
            "activation": "silu_gated_mlp",
            "normalization": "rms_norm",
            "positional_encoding": "rotary",
        },
        "equivalence_relation": {
            "name": "function_preserving_reparameterization_equivalence",
            "generators": [
                "gated MLP channel permutations",
                "grouped-query attention group permutations",
            ],
            "canonicalization": "blind exact lexicographic sort of composite component signatures",
            "gqa_constraint": "permute complete K/V groups and their associated query-head blocks together",
            "acceptance_criterion": "raw digest differs + blind canonical digest matches + blind mapping matches + behavior matches + aligned interventions match",
            "mlp_shifts": list(MLP_SHIFTS),
            "gqa_group_shifts": list(ATTENTION_SHIFTS),
        },
        "reference": {"metrics": reference_metrics, "pass": reference_pass},
        "symmetry_cases": [{**case, "pass": bool(case["identity_criterion"]["accepted"])} for case in cases],
        "non_symmetry_cases": nonsymmetry_cases,
        "non_symmetry_summary": {
            "case_count": len(nonsymmetry_cases),
            "cases_with_behavior_change": sum(int(case["behavior_changed"]) for case in nonsymmetry_cases),
            "all_cases_changed_behavior": all(bool(case["behavior_changed"]) for case in nonsymmetry_cases),
        },
        "causal_atlas": causal_atlas,
        "jspace_proxy": jspace_proxy,
        "thresholds": {
            "reference_relative_l2_max": REFERENCE_RELATIVE_L2_MAX,
            "symmetry_relative_l2_max": SYMMETRY_RELATIVE_L2_MAX,
            "aligned_intervention_cosine_min": ALIGNED_INTERVENTION_COSINE_MIN,
            "aligned_intervention_max_abs": ALIGNED_INTERVENTION_MAX_ABS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
