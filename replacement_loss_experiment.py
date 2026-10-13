"""Fresh held-out trained-model replacement/loss experiment.

This protocol tests a structural replacement rather than a permutation.  A
visible-selected MLP channel is replaced by two newly materialized child
channels.  The original channel's incoming computation is duplicated and its
outgoing projection is split in half, so the treatment preserves the model
function while changing the component graph and the causal ablation
fingerprint.  Wrong-source and seeded-random replacements are held-out
negative controls.

The result is deliberately bounded: this is a local observation about a
trained checkpoint and a structural replacement.  It is not independent
retraining, mechanistic understanding, architecture-general proof, alignment
validation, or production authorization.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from transformers import AutoTokenizer

from gpt2_experiment import full_digest
from opt_experiment import NumpyOPT, OPTCheckpoint, tensor_digest as opt_tensor_digest
from smollm_experiment import NumpySmolLM, SmolLMCheckpoint, tensor_digest as smollm_tensor_digest


STATE_SLICE = "theseus-lab-sandbox-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = "theseus-replacement-loss-v2"
SEEDS = (20260924, 20260925, 20260926)

# These probes are intentionally distinct from the legacy GPT-2/OPT/SmolLM
# contracts.  Selection is performed only on VISIBLE_PROMPTS.  HELDOUT_PROMPTS
# is not read until all transformation and threshold metadata are locked.
VISIBLE_PROMPTS = (
    "An apple is a",
    "The opposite of noisy is",
    "Three times four equals",
    "A triangle has",
    "Water freezes at",
    "Birds can usually",
    "A library contains",
    "The color of grass is",
)

HELDOUT_PROMPTS = (
    "A standard deck has four",
    "A week contains seven",
    "The boiling point of water is about",
    "A dozen contains",
    "The planet closest to the sun is",
    "A mammal feeds its young with",
    "A compass points toward",
    "A square has four equal",
    "The season after autumn is",
    "A violin has strings made of",
    "A kilometer is longer than a",
    "The chemical symbol for oxygen is",
    "A calendar has twelve",
    "A caterpillar becomes a",
    "The largest land animal is the",
    "A triangle with three equal sides is",
    "A thermometer measures",
    "The opposite of victory is",
    "A rectangle has four",
    "The number after ninety-nine is",
    "A boat travels on",
    "The organ used to hear is the",
    "A map shows a representation of",
    "A candle produces light and",
    "A chessboard has sixty-four",
    "The fastest land animal is the",
    "A farmer plants seeds in the",
    "A museum displays objects of",
    "The Earth travels around the",
    "A key opens a",
    "A clock measures",
    "A rainbow contains several",
    "The opposite of narrow is",
    "A baker makes bread from",
    "A telescope helps observe",
    "A fish breathes using",
    "A shadow is made when light is",
    "A bicycle normally has two",
    "A doctor works in a",
    "A month is divided into",
    "A magnet attracts some kinds of",
    "A library is a place to borrow",
    "A seed can grow into a",
    "A mountain is higher than a",
    "A piano is played with",
    "The opposite of ancient is",
    "A boat needs water to",
    "A camera records an",
    "A blanket helps keep someone",
    "A calendar year has about three hundred",
    "A person who writes books is an",
    "A tunnel passes under or through",
    "A raincoat protects someone from",
    "The color made by mixing blue and yellow is",
    "A ruler measures length in units such as",
    "A baby dog is called a",
    "A garden grows plants in",
    "A train travels on",
    "A refrigerator keeps food",
    "A person who studies stars is an",
    "The opposite of empty is",
    "A house has rooms for people to",
    "A flashlight uses a source of",
    "A wheel turns around an",
)

# Locked before any held-out outputs are computed.
TREATMENT_RELATIVE_L2_MAX = 1e-8
TREATMENT_TOP1_MIN = 1.0
CONTROL_RELATIVE_L2_MIN = 1e-4
CONTROL_TOP1_MAX = 1.0
CAUSAL_SUM_RELATIVE_L2_MAX = 1e-8


def _tokenize(tokenizer: AutoTokenizer, prompts: Iterable[str]) -> list[list[int]]:
    encoded = [tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts]
    if any(not ids for ids in encoded):
        raise ValueError("all replacement probes must tokenize to non-empty sequences")
    return [[int(token) for token in ids] for ids in encoded]


def _behavior_summary(
    baseline: list[np.ndarray],
    variant: list[np.ndarray],
) -> dict[str, object]:
    if len(baseline) != len(variant) or not baseline:
        raise ValueError("behavior vectors must be non-empty and paired")
    relative_l2 = [
        float(np.linalg.norm(before - after) / max(np.linalg.norm(before), 1e-12))
        for before, after in zip(baseline, variant)
    ]
    top1 = [int(np.argmax(before) == np.argmax(after)) for before, after in zip(baseline, variant)]
    return {
        "observation_count": len(relative_l2),
        "mean_relative_l2": round(float(np.mean(relative_l2)), 12),
        "max_relative_l2": round(float(np.max(relative_l2)), 12),
        "top1_agreement": round(float(np.mean(top1)), 12),
        "top1_matches": int(sum(top1)),
    }


def _effect_summary(
    baseline: list[np.ndarray],
    variant: list[np.ndarray],
) -> dict[str, object]:
    effects = [
        before - after
        for before, after in zip(baseline, variant)
    ]
    norms = [float(np.linalg.norm(effect)) for effect in effects]
    return {
        "observation_count": len(effects),
        "mean_effect_l2": round(float(np.mean(norms)), 12),
    }


def _component_digest(model: object, kind: str, layer: int, index: int) -> str:
    if kind == "opt":
        assert isinstance(model, NumpyOPT)
        prefix = f"model.decoder.layers.{layer}"
        tensors = {
            "fc1": model.tensors[f"{prefix}.fc1.weight"][index],
            "bias": model.tensors[f"{prefix}.fc1.bias"][index:index + 1],
            "fc2": model.tensors[f"{prefix}.fc2.weight"][:, index],
        }
        return opt_tensor_digest(tensors)
    if kind == "smollm":
        assert isinstance(model, NumpySmolLM)
        prefix = f"model.layers.{layer}.mlp"
        tensors = {
            "gate": model.tensors[f"{prefix}.gate_proj.weight"][index],
            "up": model.tensors[f"{prefix}.up_proj.weight"][index],
            "down": model.tensors[f"{prefix}.down_proj.weight"][:, index],
        }
        return smollm_tensor_digest(tensors)
    raise ValueError(f"unknown model kind: {kind}")


def _layer_component_digests(model: object, kind: str, layer: int) -> list[str]:
    width = _layer_width(model, kind, layer)
    return [_component_digest(model, kind, layer, index) for index in range(width)]


def _layer_width(model: object, kind: str, layer: int) -> int:
    if kind == "opt":
        assert isinstance(model, NumpyOPT)
        return int(model.tensors[f"model.decoder.layers.{layer}.fc1.weight"].shape[0])
    if kind == "smollm":
        assert isinstance(model, NumpySmolLM)
        return int(model.tensors[f"model.layers.{layer}.mlp.gate_proj.weight"].shape[0])
    raise ValueError(f"unknown model kind: {kind}")


def _canonical_layer_digest(model: object, kind: str, layer: int) -> str:
    return full_digest(sorted(_layer_component_digests(model, kind, layer)))


def _copy_model(model: object) -> object:
    transformed = copy.copy(model)
    transformed.tensors = dict(model.tensors)  # type: ignore[attr-defined]
    return transformed


def _replace_layer_arrays(
    model: object,
    kind: str,
    layer: int,
    target: int,
    mode: str,
    seed: int,
    source: int,
) -> tuple[object, int]:
    """Build a width-expanded treatment or control without a permutation."""

    transformed = _copy_model(model)
    rng = np.random.default_rng(seed)
    if kind == "opt":
        assert isinstance(transformed, NumpyOPT)
        prefix = f"model.decoder.layers.{layer}"
        fc1_name = f"{prefix}.fc1.weight"
        bias_name = f"{prefix}.fc1.bias"
        fc2_name = f"{prefix}.fc2.weight"
        fc1 = model.tensors[fc1_name]
        bias = model.tensors[bias_name]
        fc2 = model.tensors[fc2_name]
        if mode == "treatment":
            incoming = fc1[target].copy()
            incoming_bias = np.asarray(bias[target]).copy()
            outgoing = fc2[:, target].copy()
        elif mode == "wrong_source":
            incoming = fc1[source].copy()
            incoming_bias = np.asarray(bias[source]).copy()
            outgoing = fc2[:, source].copy()
        elif mode == "random":
            incoming = rng.normal(0.0, max(float(np.std(fc1[target])), 1e-6), size=fc1.shape[1]).astype(fc1.dtype)
            incoming_bias = np.asarray(rng.normal(0.0, max(float(np.std(bias)), 1e-6)), dtype=bias.dtype)
            outgoing = rng.normal(0.0, max(float(np.std(fc2[:, target])), 1e-6), size=fc2.shape[0]).astype(fc2.dtype)
        else:
            raise ValueError(f"unknown replacement mode: {mode}")
        transformed.tensors[fc1_name] = np.concatenate((fc1.copy(), incoming[None, :]), axis=0)
        transformed.tensors[bias_name] = np.concatenate((bias.copy(), np.asarray([incoming_bias], dtype=bias.dtype)))
        updated_fc2 = fc2.copy()
        updated_fc2[:, target] = outgoing * 0.5
        transformed.tensors[fc2_name] = np.concatenate(
            (updated_fc2, (outgoing * 0.5)[:, None]),
            axis=1,
        )
        return transformed, int(fc1.shape[0])

    if kind == "smollm":
        assert isinstance(transformed, NumpySmolLM)
        prefix = f"model.layers.{layer}.mlp"
        gate_name = f"{prefix}.gate_proj.weight"
        up_name = f"{prefix}.up_proj.weight"
        down_name = f"{prefix}.down_proj.weight"
        gate = model.tensors[gate_name]
        up = model.tensors[up_name]
        down = model.tensors[down_name]
        if mode == "treatment":
            gate_row = gate[target].copy()
            up_row = up[target].copy()
            outgoing = down[:, target].copy()
        elif mode == "wrong_source":
            gate_row = gate[source].copy()
            up_row = up[source].copy()
            outgoing = down[:, source].copy()
        elif mode == "random":
            gate_row = rng.normal(0.0, max(float(np.std(gate[target])), 1e-6), size=gate.shape[1]).astype(gate.dtype)
            up_row = rng.normal(0.0, max(float(np.std(up[target])), 1e-6), size=up.shape[1]).astype(up.dtype)
            outgoing = rng.normal(0.0, max(float(np.std(down[:, target])), 1e-6), size=down.shape[0]).astype(down.dtype)
        else:
            raise ValueError(f"unknown replacement mode: {mode}")
        transformed.tensors[gate_name] = np.concatenate((gate.copy(), gate_row[None, :]), axis=0)
        transformed.tensors[up_name] = np.concatenate((up.copy(), up_row[None, :]), axis=0)
        updated_down = down.copy()
        updated_down[:, target] = outgoing * 0.5
        transformed.tensors[down_name] = np.concatenate(
            (updated_down, (outgoing * 0.5)[:, None]),
            axis=1,
        )
        return transformed, int(gate.shape[0])
    raise ValueError(f"unknown model kind: {kind}")


def _select_target(
    model: object,
    kind: str,
    visible_ids: list[list[int]],
    seed: int,
) -> tuple[int, int, int, dict[str, object]]:
    if kind == "opt":
        assert isinstance(model, NumpyOPT)
        candidate_layers = (0, model.n_layer // 2, model.n_layer - 1)
    else:
        assert isinstance(model, NumpySmolLM)
        candidate_layers = (0, model.n_layer // 2, model.n_layer - 1)
    layer = int(candidate_layers[seed % len(candidate_layers)])
    traces = [model.mlp_activation_trace(ids)[layer] for ids in visible_ids]
    matrix = np.stack(traces)
    active_count = np.count_nonzero(np.abs(matrix) > 1e-8, axis=0)
    magnitude_sum = np.sum(np.abs(matrix), axis=0)
    tie_break = np.random.default_rng(seed).random(len(active_count))
    order = sorted(
        range(len(active_count)),
        key=lambda index: (-int(active_count[index]), -float(magnitude_sum[index]), float(tie_break[index]), index),
    )
    active = [index for index in order if int(active_count[index]) > 0]
    if len(active) < 2:
        raise ValueError(f"fewer than two active channels at selected layer {layer}")
    target, source = int(active[0]), int(active[1])
    return layer, target, source, {
        "layer": layer,
        "target": target,
        "source_control": source,
        "visible_active_probe_count": int(active_count[target]),
        "visible_activation_sum": round(float(magnitude_sum[target]), 12),
        "selection_source": "visible_mlp_activation_trace_only",
    }


def _model_outputs(model: object, tokenized: list[list[int]]) -> list[np.ndarray]:
    return [model.forward(ids, precision="float64") for ids in tokenized]  # type: ignore[attr-defined]


def _causal_split_summary(
    base_model: object,
    treatment_model: object,
    kind: str,
    layer: int,
    target: int,
    child: int,
    heldout_ids: list[list[int]],
    baseline: list[np.ndarray],
    treatment: list[np.ndarray],
) -> dict[str, object]:
    baseline_ablation = _model_outputs(
        base_model,
        heldout_ids,
    )
    baseline_ablated = [
        base_model.forward(  # type: ignore[attr-defined]
            ids,
            ablate_channel=target,
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in heldout_ids
    ]
    treatment_old_ablated = [
        treatment_model.forward(  # type: ignore[attr-defined]
            ids,
            ablate_channel=target,
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in heldout_ids
    ]
    treatment_child_ablated = [
        treatment_model.forward(  # type: ignore[attr-defined]
            ids,
            ablate_channel=child,
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in heldout_ids
    ]
    treatment_both_ablated = [
        treatment_model.forward(  # type: ignore[attr-defined]
            ids,
            ablate_channels=np.asarray([target, child], dtype=np.int64),
            ablate_channel_layer=layer,
            precision="float64",
        )
        for ids in heldout_ids
    ]
    original_effects = [before - after for before, after in zip(baseline_ablation, baseline_ablated)]
    old_effects = [before - after for before, after in zip(treatment, treatment_old_ablated)]
    child_effects = [before - after for before, after in zip(treatment, treatment_child_ablated)]
    both_effects = [before - after for before, after in zip(treatment, treatment_both_ablated)]
    return {
        "original_target_effect": _effect_summary(baseline_ablation, baseline_ablated),
        "old_slot_effect": _effect_summary(treatment, treatment_old_ablated),
        "new_child_effect": _effect_summary(treatment, treatment_child_ablated),
        "both_children_vs_original": _behavior_summary(original_effects, both_effects),
        "old_slot_vs_new_child": _behavior_summary(old_effects, child_effects),
        "old_slot_vs_original": _behavior_summary(original_effects, old_effects),
        "child_vs_original": _behavior_summary(original_effects, child_effects),
        "kind": kind,
    }


def _run_case(
    model: object,
    kind: str,
    model_id: str,
    tokenizer: AutoTokenizer,
    checkpoint_digest: str,
    seed: int,
) -> dict[str, object]:
    visible_ids = _tokenize(tokenizer, VISIBLE_PROMPTS)
    heldout_ids = _tokenize(tokenizer, HELDOUT_PROMPTS)
    layer, target, source, selection = _select_target(model, kind, visible_ids, seed)
    base_visible = _model_outputs(model, visible_ids)
    base_heldout = _model_outputs(model, heldout_ids)

    locked = {
        "protocol_id": PROTOCOL_ID,
        "behavior_thresholds": {
            "treatment_relative_l2_max": TREATMENT_RELATIVE_L2_MAX,
            "treatment_top1_min": TREATMENT_TOP1_MIN,
            "control_relative_l2_min": CONTROL_RELATIVE_L2_MIN,
            "control_top1_max": CONTROL_TOP1_MAX,
        },
        "mechanism_thresholds": {
            "causal_sum_relative_l2_max": CAUSAL_SUM_RELATIVE_L2_MAX,
            "width_delta": 1,
        },
        "probe_contract_digest": full_digest({"visible": VISIBLE_PROMPTS, "heldout": HELDOUT_PROMPTS}),
        "selection": selection,
        "selection_locked_before_heldout": True,
    }

    treatment, child = _replace_layer_arrays(model, kind, layer, target, "treatment", seed, source)
    wrong_source, _ = _replace_layer_arrays(model, kind, layer, target, "wrong_source", seed, source)
    random_replacement, _ = _replace_layer_arrays(model, kind, layer, target, "random", seed, source)

    treatment_visible = _model_outputs(treatment, visible_ids)
    treatment_heldout = _model_outputs(treatment, heldout_ids)
    wrong_visible = _model_outputs(wrong_source, visible_ids)
    wrong_heldout = _model_outputs(wrong_source, heldout_ids)
    random_visible = _model_outputs(random_replacement, visible_ids)
    random_heldout = _model_outputs(random_replacement, heldout_ids)

    before_components = _layer_component_digests(model, kind, layer)
    treatment_components = _layer_component_digests(treatment, kind, layer)
    mechanism = {
        "base_width": len(before_components),
        "replacement_width": len(treatment_components),
        "width_delta": len(treatment_components) - len(before_components),
        "base_layer_canonical_digest": _canonical_layer_digest(model, kind, layer),
        "replacement_layer_canonical_digest": _canonical_layer_digest(treatment, kind, layer),
        "canonical_digest_changed": _canonical_layer_digest(model, kind, layer) != _canonical_layer_digest(treatment, kind, layer),
        "target_component_digest": before_components[target],
        "target_component_absent_after_replacement": before_components[target] not in treatment_components,
        "new_child_index": child,
        "new_child_component_digest": treatment_components[child],
    }
    causal = _causal_split_summary(
        model,
        treatment,
        kind,
        layer,
        target,
        child,
        heldout_ids,
        base_heldout,
        treatment_heldout,
    )
    treatment_behavior = {
        "visible": _behavior_summary(base_visible, treatment_visible),
        "heldout": _behavior_summary(base_heldout, treatment_heldout),
    }
    wrong_behavior = {
        "visible": _behavior_summary(base_visible, wrong_visible),
        "heldout": _behavior_summary(base_heldout, wrong_heldout),
    }
    random_behavior = {
        "visible": _behavior_summary(base_visible, random_visible),
        "heldout": _behavior_summary(base_heldout, random_heldout),
    }
    treatment_pass = bool(
        treatment_behavior["heldout"]["top1_agreement"] >= TREATMENT_TOP1_MIN
        and treatment_behavior["heldout"]["mean_relative_l2"] <= TREATMENT_RELATIVE_L2_MAX
        and mechanism["width_delta"] == 1
        and mechanism["canonical_digest_changed"]
        and mechanism["target_component_absent_after_replacement"]
        and causal["both_children_vs_original"]["mean_relative_l2"] <= CAUSAL_SUM_RELATIVE_L2_MAX
        and causal["old_slot_vs_new_child"]["mean_relative_l2"] <= CAUSAL_SUM_RELATIVE_L2_MAX
    )

    def control_failed(behavior: dict[str, object]) -> bool:
        heldout = behavior["heldout"]
        return bool(
            heldout["mean_relative_l2"] >= CONTROL_RELATIVE_L2_MIN
            or heldout["top1_agreement"] < CONTROL_TOP1_MAX
        )

    wrong_failed = control_failed(wrong_behavior)
    random_failed = control_failed(random_behavior)
    return {
        "model_id": model_id,
        "model_kind": kind,
        "checkpoint_digest": checkpoint_digest,
        "seed": seed,
        "locked": locked,
        "treatment": {
            "behavior": treatment_behavior,
            "mechanism": mechanism,
            "causal": causal,
            "pass": treatment_pass,
        },
        "controls": {
            "wrong_source": {"behavior": wrong_behavior, "failed_as_expected": wrong_failed},
            "seeded_random": {"behavior": random_behavior, "failed_as_expected": random_failed},
        },
        "status": "PASS" if treatment_pass and wrong_failed and random_failed else "INCONCLUSIVE",
    }


def _load_model(model_dir: Path, model_id: str) -> tuple[object, AutoTokenizer, str, str]:
    if model_id == "opt-125m":
        checkpoint = OPTCheckpoint(model_dir, model_id)
        return NumpyOPT(checkpoint), AutoTokenizer.from_pretrained(model_dir, local_files_only=True), checkpoint.checkpoint_digest, "opt"
    if model_id == "smollm2-135m":
        checkpoint = SmolLMCheckpoint(model_dir, model_id)
        return NumpySmolLM(checkpoint), AutoTokenizer.from_pretrained(model_dir, local_files_only=True), checkpoint.checkpoint_digest, "smollm"
    raise ValueError(f"unsupported replacement checkpoint: {model_id}")


def run(model_specs: dict[str, Path]) -> dict[str, object]:
    protocol_digest = full_digest({
        "protocol_id": PROTOCOL_ID,
        "visible": VISIBLE_PROMPTS,
        "heldout": HELDOUT_PROMPTS,
        "seeds": SEEDS,
        "thresholds": {
            "treatment_relative_l2_max": TREATMENT_RELATIVE_L2_MAX,
            "treatment_top1_min": TREATMENT_TOP1_MIN,
            "control_relative_l2_min": CONTROL_RELATIVE_L2_MIN,
            "control_top1_max": CONTROL_TOP1_MAX,
            "causal_sum_relative_l2_max": CAUSAL_SUM_RELATIVE_L2_MAX,
        },
    })
    reports: list[dict[str, object]] = []
    for model_id, model_dir in model_specs.items():
        model, tokenizer, checkpoint_digest, kind = _load_model(model_dir, model_id)
        for seed in SEEDS:
            case = _run_case(model, kind, model_id, tokenizer, checkpoint_digest, seed)
            case["protocol_digest"] = protocol_digest
            reports.append(case)
    status = "LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_PASS" if all(
        report["status"] == "PASS" for report in reports
    ) else "LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_INCONCLUSIVE"
    return {
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol_digest,
        "status": status,
        "claim_ceiling": "LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_OBSERVATION_ONLY",
        "independent_retraining": False,
        "provider_calls": False,
        "real_assessment_data": False,
        "independent_review": False,
        "architecture_general_validity": False,
        "reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt-dir", type=Path, required=True)
    parser.add_argument("--smollm2-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    result = run({"opt-125m": args.opt_dir, "smollm2-135m": args.smollm2_dir})
    encoded = json.dumps(result, sort_keys=True)
    print(encoded)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    return 0 if result["status"] == "LOCAL_TRAINED_MODEL_STRUCTURAL_REPLACEMENT_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
