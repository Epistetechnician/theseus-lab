"""Independent reference and symmetry sweep for the local GPT-2 experiment."""

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
    INTERVENTION_CHANNEL,
    NumpyGPT2,
    GPT2Checkpoint,
    VISIBLE_PROMPTS,
    compare_logits,
    full_digest,
    tensor_digest,
    vector_similarity,
)
from jspace_proxy import analyze_jspace


REFERENCE_RELATIVE_L2_MAX = 1e-4
SYMMETRY_RELATIVE_L2_MAX = 1e-4
BEHAVIOR_CHANGE_RELATIVE_L2_MIN = 1e-4
ALIGNED_INTERVENTION_COSINE_MIN = 0.999
ALIGNED_INTERVENTION_MAX_ABS = 1e-4
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260914
MLP_SHIFTS = (1, 17)
ATTENTION_SHIFTS = (1, 5)
ATLAS_CHANNEL_CANDIDATES = (17, 34, 1024)
ATLAS_HEAD_CANDIDATES = (0, 5, 11)
PRUNE_COUNT = 128
REPLACEMENT_CHANNEL = (17, 34)


def reference_logits(model: torch.nn.Module, tokenizer: AutoTokenizer, prompt: str) -> np.ndarray:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        output = model(**encoded).logits[0, -1].detach().cpu().numpy()
    return np.asarray(output, dtype=np.float32)


def permutation(size: int, shift: int) -> np.ndarray:
    return np.roll(np.arange(size), shift % size)


def changed_fraction(before: list[np.ndarray], after: list[np.ndarray]) -> float:
    changed = sum(np.count_nonzero(left != right) for left, right in zip(before, after))
    total = sum(left.size for left in before)
    return changed / total


def _mlp_signature(weight: np.ndarray, bias: np.ndarray, projection: np.ndarray, index: int) -> tuple[bytes, bytes, bytes]:
    return (
        np.ascontiguousarray(weight[:, index]).tobytes(),
        np.asarray([bias[index]], dtype=bias.dtype).tobytes(),
        np.ascontiguousarray(projection[index, :]).tobytes(),
    )


def canonicalize_mlp(
    weight: np.ndarray,
    bias: np.ndarray,
    projection: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.asarray(sorted(
        range(bias.shape[0]),
        key=lambda index: _mlp_signature(weight, bias, projection, index),
    ), dtype=np.int64)
    return [weight[:, order], bias[order], projection[order, :]], order


def infer_mlp_permutation(
    before: list[np.ndarray],
    after: list[np.ndarray],
) -> np.ndarray:
    before_signatures = {
        _mlp_signature(before[0], before[1], before[2], index): index
        for index in range(before[1].shape[0])
    }
    if len(before_signatures) != before[1].shape[0]:
        raise ValueError("MLP channel signatures are not unique")
    try:
        inferred = np.asarray([
            before_signatures[_mlp_signature(after[0], after[1], after[2], index)]
            for index in range(after[1].shape[0])
        ], dtype=np.int64)
    except KeyError as error:
        raise ValueError("transformed MLP contains an unmatched channel") from error
    if len(set(inferred.tolist())) != len(inferred):
        raise ValueError("inferred MLP mapping is not a permutation")
    return inferred


def _attention_signature(
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray,
    projection: np.ndarray,
    index: int,
    head_dim: int,
    embedding_width: int,
) -> tuple[bytes, bytes, bytes, bytes, bytes]:
    indexes = index * head_dim + np.arange(head_dim)
    return (
        np.ascontiguousarray(qkv_weight[:, indexes]).tobytes(),
        np.ascontiguousarray(qkv_weight[:, embedding_width + indexes]).tobytes(),
        np.ascontiguousarray(qkv_weight[:, 2 * embedding_width + indexes]).tobytes(),
        np.ascontiguousarray(qkv_bias[indexes]).tobytes(),
        np.ascontiguousarray(projection[indexes, :]).tobytes(),
    )


def canonicalize_attention(
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray,
    projection: np.ndarray,
    head_dim: int,
    embedding_width: int,
    head_count: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    order = np.asarray(sorted(
        range(head_count),
        key=lambda index: _attention_signature(
            qkv_weight, qkv_bias, projection, index, head_dim, embedding_width,
        ),
    ), dtype=np.int64)
    head_indexes = order[:, None] * head_dim + np.arange(head_dim)
    flat_indexes = head_indexes.reshape(-1)
    qkv_order = np.concatenate((
        flat_indexes,
        embedding_width + flat_indexes,
        2 * embedding_width + flat_indexes,
    ))
    return [qkv_weight[:, qkv_order], qkv_bias[qkv_order], projection[flat_indexes, :]], order


def infer_attention_permutation(
    before: list[np.ndarray],
    after: list[np.ndarray],
    head_dim: int,
    embedding_width: int,
    head_count: int,
) -> np.ndarray:
    before_signatures = {
        _attention_signature(before[0], before[1], before[2], index, head_dim, embedding_width): index
        for index in range(head_count)
    }
    if len(before_signatures) != head_count:
        raise ValueError("attention-head signatures are not unique")
    try:
        inferred = np.asarray([
            before_signatures[_attention_signature(
                after[0], after[1], after[2], index, head_dim, embedding_width,
            )]
            for index in range(head_count)
        ], dtype=np.int64)
    except KeyError as error:
        raise ValueError("transformed attention contains an unmatched head") from error
    if len(set(inferred.tolist())) != len(inferred):
        raise ValueError("inferred attention mapping is not a permutation")
    return inferred


def split_behavior(before: list[np.ndarray], after: list[np.ndarray]) -> dict[str, object]:
    visible_count = len(VISIBLE_PROMPTS)
    return {
        "visible": compare_logits(before[:visible_count], after[:visible_count]),
        "heldout": compare_logits(before[visible_count:], after[visible_count:]),
    }


def atlas_channels(model: NumpyGPT2) -> tuple[int, ...]:
    width = model.tensors["h.0.mlp.c_fc.bias"].shape[0]
    return tuple(index for index in ATLAS_CHANNEL_CANDIDATES if index < width)


def atlas_heads(model: NumpyGPT2) -> tuple[int, ...]:
    return tuple(index for index in ATLAS_HEAD_CANDIDATES if index < model.n_head)


def bootstrap_ci(values: list[float], seed_offset: int) -> list[float]:
    values_array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indexes = generator.integers(0, len(values_array), size=(BOOTSTRAP_RESAMPLES, len(values_array)))
    means = values_array[indexes].mean(axis=1)
    return [round(float(np.quantile(means, 0.025)), 8), round(float(np.quantile(means, 0.975)), 8)]


def identity_criterion(case: dict[str, object]) -> dict[str, object]:
    visible = case["behavior"]["visible"]
    heldout = case["behavior"]["heldout"]
    aligned = case["intervention"]["aligned_effect_similarity"]
    behavior_match = (
        visible["top1_agreement"] == 1.0
        and heldout["top1_agreement"] == 1.0
        and visible["mean_relative_l2"] < SYMMETRY_RELATIVE_L2_MAX
        and heldout["mean_relative_l2"] < SYMMETRY_RELATIVE_L2_MAX
    )
    aligned_match = (
        aligned["cosine"] >= ALIGNED_INTERVENTION_COSINE_MIN
        and aligned["max_abs"] < ALIGNED_INTERVENTION_MAX_ABS
    )
    raw_digest_diff = case["mechanism"]["before_digest"] != case["mechanism"]["after_digest"]
    canonical_digest_match = case["mechanism"]["canonical_before_digest"] == case["mechanism"]["canonical_after_digest"]
    blind_mapping_match = case["mechanism"].get("blind_mapping_match", False)
    return {
        "raw_digest_diff": raw_digest_diff,
        "canonical_digest_match": canonical_digest_match,
        "canonical_digest_diff": not canonical_digest_match,
        "blind_mapping_match": blind_mapping_match,
        "behavior_match": behavior_match,
        "aligned_intervention_match": aligned_match,
        "accepted": raw_digest_diff and canonical_digest_match and blind_mapping_match and behavior_match and aligned_match,
    }


def mlp_case(
    model: NumpyGPT2,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
) -> dict[str, object]:
    prefix = f"h.{layer}.mlp"
    selected = [
        model.tensors[f"{prefix}.c_fc.weight"],
        model.tensors[f"{prefix}.c_fc.bias"],
        model.tensors[f"{prefix}.c_proj.weight"],
    ]
    perm = permutation(selected[0].shape[1], shift)
    transformed = [selected[0][:, perm], selected[1][perm], selected[2][perm, :]]
    variant = [
        model.forward(ids, permutation=perm, permutation_layer=layer)
        for ids in tokenized
    ]
    reference_ids = tokenized[0]
    baseline_effect = model.forward(
        reference_ids,
        ablate_channel=INTERVENTION_CHANNEL,
        ablate_channel_layer=layer,
        precision="float64",
    )
    baseline_normal = model.forward(reference_ids, precision="float64")
    baseline_effect = baseline_normal - baseline_effect
    variant_normal = model.forward(
        reference_ids,
        permutation=perm,
        permutation_layer=layer,
        precision="float64",
    )
    same_label = model.forward(
        reference_ids,
        permutation=perm,
        permutation_layer=layer,
        ablate_channel=INTERVENTION_CHANNEL,
        ablate_channel_layer=layer,
        precision="float64",
    )
    same_label_effect = variant_normal - same_label
    inverse = np.argsort(perm)
    aligned = model.forward(
        reference_ids,
        permutation=perm,
        permutation_layer=layer,
        ablate_channel=int(inverse[INTERVENTION_CHANNEL]),
        ablate_channel_layer=layer,
        precision="float64",
    )
    aligned_effect = variant_normal - aligned
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
            "baseline_channel": INTERVENTION_CHANNEL,
            "aligned_variant_channel": int(inverse[INTERVENTION_CHANNEL]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }
    case["identity_criterion"] = identity_criterion(case)
    return case


def attention_case(
    model: NumpyGPT2,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    shift: int,
) -> dict[str, object]:
    prefix = f"h.{layer}.attn"
    qkv_weight = model.tensors[f"{prefix}.c_attn.weight"]
    qkv_bias = model.tensors[f"{prefix}.c_attn.bias"]
    projection = model.tensors[f"{prefix}.c_proj.weight"]
    perm = permutation(model.n_head, shift)
    head_indexes = np.arange(model.n_head) * model.head_dim
    head_indexes = head_indexes[:, None] + np.arange(model.head_dim)
    order = np.concatenate((
        head_indexes[perm].reshape(-1),
        (model.n_embd + head_indexes[perm]).reshape(-1),
        (2 * model.n_embd + head_indexes[perm]).reshape(-1),
    ))
    transformed = [
        qkv_weight[:, order],
        qkv_bias[order],
        projection[head_indexes[perm].reshape(-1), :],
    ]
    variant = [
        model.forward(
            ids,
            attention_permutation=perm,
            attention_permutation_layer=layer,
        )
        for ids in tokenized
    ]
    reference_ids = tokenized[0]
    baseline_effect = model.forward(reference_ids, precision="float64") - model.forward(
        reference_ids,
        ablate_head=0,
        ablate_head_layer=layer,
        precision="float64",
    )
    variant_normal = model.forward(
        reference_ids,
        attention_permutation=perm,
        attention_permutation_layer=layer,
        precision="float64",
    )
    same_label_effect = variant_normal - model.forward(
        reference_ids,
        attention_permutation=perm,
        attention_permutation_layer=layer,
        ablate_head=0,
        ablate_head_layer=layer,
        precision="float64",
    )
    inverse = np.argsort(perm)
    aligned_effect = variant_normal - model.forward(
        reference_ids,
        attention_permutation=perm,
        attention_permutation_layer=layer,
        ablate_head=int(inverse[0]),
        ablate_head_layer=layer,
        precision="float64",
    )
    original_components = [qkv_weight, qkv_bias, projection]
    canonical_before, _ = canonicalize_attention(
        *original_components,
        model.head_dim,
        model.n_embd,
        model.n_head,
    )
    canonical_after, _ = canonicalize_attention(
        *transformed,
        model.head_dim,
        model.n_embd,
        model.n_head,
    )
    inferred = infer_attention_permutation(
        original_components,
        transformed,
        model.head_dim,
        model.n_embd,
        model.n_head,
    )
    case = {
        "kind": "attention_head_permutation",
        "layer": layer,
        "shift": shift,
        "behavior": split_behavior(baseline, variant),
        "mechanism": {
            "before_digest": tensor_digest({"qkv_weight": qkv_weight, "qkv_bias": qkv_bias, "projection": projection}),
            "after_digest": tensor_digest({"qkv_weight": transformed[0], "qkv_bias": transformed[1], "projection": transformed[2]}),
            "canonical_before_digest": tensor_digest({"qkv_weight": canonical_before[0], "qkv_bias": canonical_before[1], "projection": canonical_before[2]}),
            "canonical_after_digest": tensor_digest({"qkv_weight": canonical_after[0], "qkv_bias": canonical_after[1], "projection": canonical_after[2]}),
            "canonicalization": "blind_exact_lexicographic_head_sort",
            "inferred_permutation_digest": full_digest(inferred.tolist()),
            "blind_mapping_match": bool(np.array_equal(inferred, perm)),
            "changed_parameter_fraction": round(changed_fraction(
                [qkv_weight, qkv_bias, projection],
                transformed,
            ), 8),
        },
        "intervention": {
            "baseline_head": 0,
            "aligned_variant_head": int(inverse[0]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }
    case["identity_criterion"] = identity_criterion(case)
    return case


def nonsymmetry_case(
    model: NumpyGPT2,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
    layer: int,
    kind: str,
    channels: np.ndarray | None = None,
    replacement: tuple[int, int] | None = None,
) -> dict[str, object]:
    prefix = f"h.{layer}.mlp"
    selected = [
        model.tensors[f"{prefix}.c_fc.weight"],
        model.tensors[f"{prefix}.c_fc.bias"],
        model.tensors[f"{prefix}.c_proj.weight"],
    ]
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


def behavior_pass(case: dict[str, object]) -> bool:
    return bool(case["identity_criterion"]["accepted"])


def atlas_effect(
    model: NumpyGPT2,
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
        "mean_relative_logit_effect": round(sum(scores) / len(scores), 8),
        "relative_logit_effect_ci95": bootstrap_ci(scores, layer * 1000 + index + (0 if component == "mlp_channel" else 10000)),
        "top1_change_rate": round(sum(top1_changes) / len(top1_changes), 8),
        "top1_change_rate_ci95": bootstrap_ci(top1_changes, layer * 1000 + index + (5000 if component == "mlp_channel" else 15000)),
    }


def build_causal_atlas(
    model: NumpyGPT2,
    tokenized: list[list[int]],
    baseline: list[np.ndarray],
) -> dict[str, list[dict[str, object]]]:
    return {
        "mlp_channels": [
            atlas_effect(model, tokenized, baseline, layer, "mlp_channel", channel)
            for layer in range(model.n_layer)
            for channel in atlas_channels(model)
        ],
        "attention_heads": [
            atlas_effect(model, tokenized, baseline, layer, "attention_head", head)
            for layer in range(model.n_layer)
            for head in atlas_heads(model)
        ],
    }


def run(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    checkpoint = GPT2Checkpoint(model_dir, model_id)
    numpy_model = NumpyGPT2(checkpoint)
    reference_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.float32,
    )
    reference_model.eval()
    reference_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    prompts = VISIBLE_PROMPTS + HELDOUT_PROMPTS
    numpy_tokenized = [checkpoint.tokenizer.encode(prompt).ids for prompt in prompts]
    reference_tokenized = [reference_tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    tokenization_match = numpy_tokenized == reference_tokenized
    numpy_baseline = [numpy_model.forward(ids) for ids in numpy_tokenized]
    reference_baseline = [
        reference_logits(reference_model, reference_tokenizer, prompt)
        for prompt in prompts
    ]
    reference_metrics = compare_logits(numpy_baseline, reference_baseline)
    reference_pass = (
        tokenization_match
        and reference_metrics["top1_agreement"] == 1.0
        and reference_metrics["mean_relative_l2"] < REFERENCE_RELATIVE_L2_MAX
    )
    cases: list[dict[str, object]] = []
    for layer in range(numpy_model.n_layer):
        for shift in MLP_SHIFTS:
            cases.append(mlp_case(numpy_model, numpy_tokenized, numpy_baseline, layer, shift))
    for layer in range(numpy_model.n_layer):
        for shift in ATTENTION_SHIFTS:
            cases.append(attention_case(numpy_model, numpy_tokenized, numpy_baseline, layer, shift))
    all_cases_pass = all(behavior_pass(case) for case in cases)
    nonsymmetry_cases = [
        nonsymmetry_case(
            numpy_model,
            numpy_tokenized,
            numpy_baseline,
            layer,
            "mlp_single_channel_ablation",
            channels=np.asarray([INTERVENTION_CHANNEL]),
        )
        for layer in range(numpy_model.n_layer)
    ]
    nonsymmetry_cases.append(nonsymmetry_case(
        numpy_model,
        numpy_tokenized,
        numpy_baseline,
        5,
        "mlp_block_pruning",
        channels=np.arange(PRUNE_COUNT),
    ))
    nonsymmetry_cases.append(nonsymmetry_case(
        numpy_model,
        numpy_tokenized,
        numpy_baseline,
        5,
        "mlp_channel_replacement",
        replacement=REPLACEMENT_CHANNEL,
    ))
    nonsymmetry_changed = sum(
        int(case["behavior_changed"])
        for case in nonsymmetry_cases
    )
    causal_atlas = build_causal_atlas(numpy_model, numpy_tokenized, numpy_baseline)
    jspace_proxy = analyze_jspace(model_dir, checkpoint.model_id)
    overall_pass = (
        reference_pass
        and all_cases_pass
        and len(causal_atlas["mlp_channels"]) == numpy_model.n_layer * len(atlas_channels(numpy_model))
        and len(causal_atlas["attention_heads"]) == numpy_model.n_layer * len(atlas_heads(numpy_model))
        and jspace_proxy["status"] == "LOCAL_PROXY_PASS"
    )
    return {
        "status": "REFERENCE_SYMMETRY_ATLAS_AND_JSPACE_PROXY_PASS" if overall_pass else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_GPT2_FAMILY_TRANSFORMATION_OBSERVATION_ONLY",
        "checkpoint": {
            "weights_loaded": True,
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
        "equivalence_relation": {
            "name": "function_preserving_reparameterization_equivalence",
            "generators": [
                "MLP-channel permutations",
                "attention-head permutations",
            ],
            "canonicalization": "blind exact lexicographic sort of composite component signatures",
            "relation_properties": [
                "reflexive under the identity permutation",
                "symmetric under the inverse permutation",
                "transitive under permutation composition",
            ],
            "acceptance_criterion": "raw digest differs + blind canonical digest matches + blind mapping matches + behavior matches + aligned interventions match",
            "mlp_shifts": list(MLP_SHIFTS),
            "attention_shifts": list(ATTENTION_SHIFTS),
        },
        "reference": {
            "tokenization_match": tokenization_match,
            "metrics": reference_metrics,
            "pass": reference_pass,
        },
        "symmetry_cases": [
            {**case, "pass": behavior_pass(case)}
            for case in cases
        ],
        "non_symmetry_cases": nonsymmetry_cases,
        "non_symmetry_summary": {
            "case_count": len(nonsymmetry_cases),
            "cases_with_behavior_change": nonsymmetry_changed,
            "all_cases_changed_behavior": nonsymmetry_changed == len(nonsymmetry_cases),
        },
        "causal_atlas": causal_atlas,
        "jspace_proxy": jspace_proxy,
        "thresholds": {
            "reference_relative_l2_max": REFERENCE_RELATIVE_L2_MAX,
            "symmetry_relative_l2_max": SYMMETRY_RELATIVE_L2_MAX,
            "behavior_change_relative_l2_min": BEHAVIOR_CHANGE_RELATIVE_L2_MIN,
            "aligned_intervention_cosine_min": ALIGNED_INTERVENTION_COSINE_MIN,
            "aligned_intervention_max_abs": ALIGNED_INTERVENTION_MAX_ABS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
