"""Independent reference and symmetry sweep for the local OPT adapter."""

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
from opt_experiment import NumpyOPT, OPTCheckpoint, tensor_digest
from opt_jspace_proxy import analyze_jspace
from reference_check import (
    ALIGNED_INTERVENTION_COSINE_MIN,
    ALIGNED_INTERVENTION_MAX_ABS,
    ATLAS_CHANNEL_CANDIDATES,
    ATLAS_HEAD_CANDIDATES,
    ATTENTION_SHIFTS,
    BEHAVIOR_CHANGE_RELATIVE_L2_MIN,
    MLP_SHIFTS,
    PRUNE_COUNT,
    REPLACEMENT_CHANNEL,
    REFERENCE_RELATIVE_L2_MAX,
    SYMMETRY_RELATIVE_L2_MAX,
    bootstrap_ci,
    canonicalize_attention,
    canonicalize_mlp,
    changed_fraction,
    identity_criterion,
    infer_attention_permutation,
    infer_mlp_permutation,
    permutation,
    reference_logits,
    split_behavior,
)


INTERVENTION_HELDOUT_COUNT = 16


def select_intervention_channels(
    model: NumpyOPT,
    visible_tokenized: list[list[int]],
) -> tuple[dict[int, int], dict[int, dict[str, object]]]:
    """Select active coordinates on visible probes; held-out probes remain untouched."""

    traces = [model.mlp_activation_trace(ids) for ids in visible_tokenized]
    selected: dict[int, int] = {}
    details: dict[int, dict[str, object]] = {}
    for layer in range(model.n_layer):
        matrix = np.stack([trace[layer] for trace in traces])
        active_count = np.count_nonzero(matrix > 0.0, axis=0)
        activation_sum = np.sum(matrix, axis=0)
        order = sorted(
            range(model.ffn_dim),
            key=lambda index: (-int(active_count[index]), -float(activation_sum[index]), index),
        )
        channel = int(order[0])
        if int(active_count[channel]) == 0:
            raise ValueError(f"no active OPT MLP channel found at layer {layer}")
        selected[layer] = channel
        details[layer] = {
            "channel": channel,
            "visible_active_probe_count": int(active_count[channel]),
            "visible_activation_sum": round(float(activation_sum[channel]), 8),
        }
    return selected, details


def aggregate_effect(
    model: NumpyOPT,
    tokenized: list[list[int]],
    **kwargs: object,
) -> np.ndarray:
    return np.concatenate([
        model.forward(ids, precision="float64")
        - model.forward(ids, precision="float64", **kwargs)
        for ids in tokenized
    ])


def mlp_case(
    model: NumpyOPT,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
    intervention_channel: int,
    intervention_tokenized: list[list[int]],
) -> dict[str, object]:
    selected = model.mlp_components(layer)
    perm = permutation(selected[1].shape[0], shift)
    transformed = [selected[0][:, perm], selected[1][perm], selected[2][perm, :]]
    variant = [model.forward(ids, permutation=perm, permutation_layer=layer) for ids in tokenized]
    baseline_effect = aggregate_effect(
        model,
        intervention_tokenized,
        ablate_channel=intervention_channel,
        ablate_channel_layer=layer,
    )
    variant_normal = np.concatenate([
        model.forward(
            ids,
            permutation=perm,
            permutation_layer=layer,
            precision="float64",
        )
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
    canonical_before, _ = canonicalize_mlp(*selected)
    canonical_after, _ = canonicalize_mlp(*transformed)
    inferred = infer_mlp_permutation(selected, transformed)
    case = {
        "kind": "mlp_channel_permutation",
        "layer": layer,
        "shift": shift,
        "behavior": split_behavior(baseline, variant),
        "mechanism": {
            "before_digest": tensor_digest({"weight": selected[0], "bias": selected[1], "projection": selected[2]}),
            "after_digest": tensor_digest({"weight": transformed[0], "bias": transformed[1], "projection": transformed[2]}),
            "canonical_before_digest": tensor_digest({"weight": canonical_before[0], "bias": canonical_before[1], "projection": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"weight": canonical_after[0], "bias": canonical_after[1], "projection": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_component_sort",
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
    model: NumpyOPT,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
    intervention_tokenized: list[list[int]],
) -> dict[str, object]:
    selected = model.attention_components(layer)
    perm = permutation(model.n_head, shift)
    head_indexes = np.arange(model.n_head)[:, None] * model.head_dim + np.arange(model.head_dim)
    flat = head_indexes[perm].reshape(-1)
    qkv_order = np.concatenate((flat, model.n_embd + flat, 2 * model.n_embd + flat))
    transformed = [selected[0][:, qkv_order], selected[1][qkv_order], selected[2][flat, :]]
    variant = [
        model.forward(ids, attention_permutation=perm, attention_permutation_layer=layer)
        for ids in tokenized
    ]
    baseline_effect = aggregate_effect(
        model,
        intervention_tokenized,
        ablate_head=0,
        ablate_head_layer=layer,
    )
    variant_normal = np.concatenate([
        model.forward(
            ids,
            attention_permutation=perm,
            attention_permutation_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    same_label_effect = variant_normal - np.concatenate([
        model.forward(
            ids,
            attention_permutation=perm,
            attention_permutation_layer=layer,
            ablate_head=0,
            ablate_head_layer=layer,
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
            ablate_head=int(inverse[0]),
            ablate_head_layer=layer,
            precision="float64",
        )
        for ids in intervention_tokenized
    ])
    canonical_before, _ = canonicalize_attention(
        *selected, model.head_dim, model.n_embd, model.n_head,
    )
    canonical_after, _ = canonicalize_attention(
        *transformed, model.head_dim, model.n_embd, model.n_head,
    )
    inferred = infer_attention_permutation(
        selected, transformed, model.head_dim, model.n_embd, model.n_head,
    )
    case = {
        "kind": "attention_head_permutation",
        "layer": layer,
        "shift": shift,
        "behavior": split_behavior(baseline, variant),
        "mechanism": {
            "before_digest": tensor_digest({"qkv_weight": selected[0], "qkv_bias": selected[1], "projection": selected[2]}),
            "after_digest": tensor_digest({"qkv_weight": transformed[0], "qkv_bias": transformed[1], "projection": transformed[2]}),
            "canonical_before_digest": tensor_digest({"qkv_weight": canonical_before[0], "qkv_bias": canonical_before[1], "projection": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"qkv_weight": canonical_after[0], "qkv_bias": canonical_after[1], "projection": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_head_sort",
            "inferred_permutation_digest": full_digest(inferred.tolist()),
            "blind_mapping_match": bool(np.array_equal(inferred, perm)),
            "changed_parameter_fraction": round(changed_fraction(selected, transformed), 8),
        },
        "intervention": {
            "probe_count": len(intervention_tokenized),
            "baseline_head": 0,
            "aligned_variant_head": int(inverse[0]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }
    case["identity_criterion"] = identity_criterion(case)
    return case


def nonsymmetry_case(
    model: NumpyOPT,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    kind: str,
    channels: np.ndarray | None = None,
    replacement: tuple[int, int] | None = None,
) -> dict[str, object]:
    selected = model.mlp_components(layer)
    transformed = [array.copy() for array in selected]
    if channels is not None:
        transformed[0][:, channels] = 0.0
        transformed[1][channels] = 0.0
    if replacement is not None:
        target, source = replacement
        transformed[0][:, target] = transformed[0][:, source]
        transformed[1][target] = transformed[1][source]
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
    canonical_before, _ = canonicalize_mlp(*selected)
    canonical_after, _ = canonicalize_mlp(*transformed)
    return {
        "kind": kind,
        "layer": layer,
        "behavior": behavior,
        "mechanism": {
            "before_digest": tensor_digest({"weight": selected[0], "bias": selected[1], "projection": selected[2]}),
            "after_digest": tensor_digest({"weight": transformed[0], "bias": transformed[1], "projection": transformed[2]}),
            "canonical_before_digest": tensor_digest({"weight": canonical_before[0], "bias": canonical_before[1], "projection": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"weight": canonical_after[0], "bias": canonical_after[1], "projection": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_component_sort",
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


def atlas_channels(model: NumpyOPT) -> tuple[int, ...]:
    return tuple(index for index in ATLAS_CHANNEL_CANDIDATES if index < model.ffn_dim)


def atlas_heads(model: NumpyOPT) -> tuple[int, ...]:
    return tuple(index for index in ATLAS_HEAD_CANDIDATES if index < model.n_head)


def atlas_effect(
    model: NumpyOPT,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    component: str,
    index: int,
) -> dict[str, object]:
    scores: list[float] = []
    top1_changes: list[float] = []
    for ids, original in zip(tokenized, baseline):
        if component == "mlp_channel":
            altered = model.forward(ids, ablate_channel=index, ablate_channel_layer=layer)
        else:
            altered = model.forward(ids, ablate_head=index, ablate_head_layer=layer)
        scores.append(float(np.linalg.norm(original - altered) / max(np.linalg.norm(original), 1e-12)))
        top1_changes.append(float(np.argmax(original) != np.argmax(altered)))
    return {
        "layer": layer,
        "component": component,
        "index": index,
        "mean_relative_logit_effect": round(float(np.mean(scores)), 8),
        "relative_logit_effect_ci95": bootstrap_ci(scores, layer * 1000 + index + (0 if component == "mlp_channel" else 10000)),
        "top1_change_rate": round(float(np.mean(top1_changes)), 8),
        "top1_change_rate_ci95": bootstrap_ci(top1_changes, layer * 1000 + index + (5000 if component == "mlp_channel" else 15000)),
    }


def run(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    checkpoint = OPTCheckpoint(model_dir, model_id)
    numpy_model = NumpyOPT(checkpoint)
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
        attention_case(
            numpy_model,
            tokenized,
            numpy_baseline,
            layer,
            shift,
            intervention_tokenized,
        )
        for layer in range(numpy_model.n_layer)
        for shift in ATTENTION_SHIFTS
    )
    nonsymmetry_cases = [
        nonsymmetry_case(
            numpy_model,
            tokenized,
            numpy_baseline,
            layer,
            "mlp_single_channel_ablation",
            channels=np.asarray([intervention_channels[layer]]),
        )
        for layer in range(numpy_model.n_layer)
    ]
    control_layer = min(5, numpy_model.n_layer - 1)
    nonsymmetry_cases.extend((
        nonsymmetry_case(
            numpy_model, tokenized, numpy_baseline, control_layer,
            "mlp_block_pruning", channels=np.arange(PRUNE_COUNT),
        ),
        nonsymmetry_case(
            numpy_model, tokenized, numpy_baseline, control_layer,
            "mlp_channel_replacement", replacement=REPLACEMENT_CHANNEL,
        ),
    ))
    causal_atlas = {
        "mlp_channels": [
            atlas_effect(numpy_model, tokenized, numpy_baseline, layer, "mlp_channel", channel)
            for layer in range(numpy_model.n_layer)
            for channel in atlas_channels(numpy_model)
        ],
        "attention_heads": [
            atlas_effect(numpy_model, tokenized, numpy_baseline, layer, "attention_head", head)
            for layer in range(numpy_model.n_layer)
            for head in atlas_heads(numpy_model)
        ],
    }
    jspace_proxy = analyze_jspace(model_dir, checkpoint.model_id)
    overall_pass = (
        reference_pass
        and all(bool(case["identity_criterion"]["accepted"]) for case in cases)
        and all(bool(case["identity_criterion"]["canonical_digest_diff"]) for case in nonsymmetry_cases)
        and len(causal_atlas["mlp_channels"]) == numpy_model.n_layer * len(atlas_channels(numpy_model))
        and len(causal_atlas["attention_heads"]) == numpy_model.n_layer * len(atlas_heads(numpy_model))
        and jspace_proxy["status"] == "LOCAL_PROXY_PASS"
    )
    return {
        "status": "REFERENCE_SYMMETRY_ATLAS_AND_JSPACE_PROXY_PASS" if overall_pass else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_OPT_TRANSFORMATION_OBSERVATION_ONLY",
        "checkpoint": {
            "weights_loaded": True,
            "loader": "independent_numpy_opt_from_pytorch_state_dict",
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
            "family": "opt",
            "layers": numpy_model.n_layer,
            "heads": numpy_model.n_head,
            "hidden_size": numpy_model.n_embd,
            "ffn_dim": numpy_model.ffn_dim,
            "activation": "relu",
            "learned_position_offset": numpy_model.position_offset,
        },
        "equivalence_relation": {
            "name": "function_preserving_reparameterization_equivalence",
            "generators": ["MLP-channel permutations", "attention-head permutations"],
            "canonicalization": "blind exact lexicographic sort of composite component signatures",
            "acceptance_criterion": "raw digest differs + blind canonical digest matches + blind mapping matches + behavior matches + aligned interventions match",
            "mlp_shifts": list(MLP_SHIFTS),
            "attention_shifts": list(ATTENTION_SHIFTS),
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
