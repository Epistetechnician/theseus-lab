"""float64 diagnostic for the r6 equivalence-diagnostic misses (replacement-loss-v8-scale-r7).

State slice: theseus-lab-sandbox-v1. Diagnostic only: never reclassifies r6.
Model and prompt data stay in external owner custody.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import replacement_loss_v3_runner as v3
import replacement_loss_v6_runner as v6_core
import replacement_loss_v8s6_runner as r6

STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-r7-equivalence-float64-diagnostic"
ARMS = ("NN256", "NN512")
DIAGNOSTIC_FAMILY = "llama_silu_gated"
THRESHOLD = v3.EQUIVALENCE_MAX_ABS_ERROR
FP64_REDUCTION = 100.0
RELATIVE_FP32_MAX = 1e-5
R6_ARM_OUTCOMES = {"NN256": "FAIL", "NN512": "FAIL"}
V8S7_PACKET_FILES = tuple(dict.fromkeys((
    "docs/replacement-loss-v8-scale-r7.md",
    "replacement_loss_v8s7_runner.py",
    "tests/test_replacement_loss_v8s7.py",
    "docs/replacement-loss-v8s6-freeze.json",
    *r6.V8S6_PACKET_FILES,
)))
FREEZE_PATH = "docs/replacement-loss-v8s7-freeze.json"


def _root() -> Path:
    return Path(__file__).resolve().parent


def freeze_packet() -> dict[str, Any]:
    target = _root() / FREEZE_PATH
    if target.exists():
        raise FileExistsError("r7 freeze is immutable")
    rows = [
        {"path": rel, "sha256": r6._sha(_root() / rel), "bytes": (_root() / rel).stat().st_size}
        for rel in V8S7_PACKET_FILES
    ]
    r6_freeze = r6._read_json(_root() / "docs" / "replacement-loss-v8s6-freeze.json")
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V8S7_RUN",
        "created_at_utc": r6._now(),
        "packet_files": rows,
        "pinned_v8s6_packet_sha256": r6_freeze["packet_sha256"],
        "diagnostic_only": True,
        "r6_arm_outcomes_unchanged": R6_ARM_OUTCOMES,
    }
    payload["packet_sha256"] = r6._canonical_digest(payload)
    r6._write_json(target, payload)
    return {"packet_sha256": payload["packet_sha256"], "packet_file_count": len(rows)}


def verify_packet() -> dict[str, Any]:
    manifest = r6._read_json(_root() / FREEZE_PATH)
    expected = r6._canonical_digest({k: v for k, v in manifest.items() if k != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("r7 freeze digest is invalid")
    for row in manifest["packet_files"]:
        if r6._sha(_root() / row["path"]) != row["sha256"]:
            raise ValueError(f"frozen r7 source changed: {row['path']}")
    r6_freeze = r6._read_json(_root() / "docs" / "replacement-loss-v8s6-freeze.json")
    if manifest["pinned_v8s6_packet_sha256"] != r6_freeze["packet_sha256"]:
        raise ValueError("r6 pin mismatch")
    return {"packet_sha256": manifest["packet_sha256"], "packet_file_count": len(manifest["packet_files"]), "verified": True}


def classify(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered P1/P2/P3 rules to per-case diagnostic rows."""

    def err32(c: dict[str, Any]) -> float:
        return max(c["float32"]["module_effect_max_abs_error"], c["float32"]["logit_effect_max_abs_error"])

    def err64(c: dict[str, Any]) -> float:
        return max(c["float64"]["module_effect_max_abs_error"], c["float64"]["logit_effect_max_abs_error"])

    over = [c for c in cases if err32(c) > THRESHOLD]
    p1 = all(err64(c) <= THRESHOLD for c in cases)
    p2 = all(err64(c) <= err32(c) / FP64_REDUCTION for c in over)
    p3 = all(c["relative_fp32_module_error"] <= RELATIVE_FP32_MAX for c in cases)
    if not over:
        outcome = "INCONCLUSIVE_NOT_REPRODUCED"
    elif not p1:
        outcome = "REJECTED"
    elif p2:
        outcome = "SUPPORTED_FLOAT32_NOISE"
    else:
        outcome = "PARTIAL"
    return {
        "outcome": outcome,
        "P1_float64_all_within_1e-4": p1,
        "P2_float64_reduction_ge_100x_where_fp32_over": p2,
        "P3_scale": "SCALE_CONSISTENT" if p3 else "SCALE_NOT_CONSISTENT",
        "cases_float32_over_threshold": len(over),
        "case_count": len(cases),
        "r6_arm_outcomes_unchanged": R6_ARM_OUTCOMES,
    }


def _ranks(values: list[float]) -> np.ndarray:
    return np.argsort(np.argsort(np.asarray(values))).astype(float)


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _ranks(a), _ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _predictions(model, family, layer, rows, modules, case, deltas, dtype) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {k: [] for k in (
        "t_mod", "a_mod", "t_log", "a_log", "t_out", "in_norm")}
    cursor = 0
    for group, inputs in v3._batches(rows, v3_pad(model)):
        count = len(group)
        d = torch.from_numpy(deltas[cursor: cursor + count]).to(dtype)
        res = {}
        for name, perm in (("t", None), ("a", case["raw_index_permutation"])):
            module = modules["treatment" if name == "t" else "mapping_aligned_permutation"]
            base_l, base_o = v3._model_forward_variant(model, family, layer, module, group, inputs, None, input_permutation=perm)
            chg_l, chg_o = v3._model_forward_variant(model, family, layer, module, group, inputs, d, input_permutation=perm)
            res[name] = (
                (chg_o.double().numpy() - base_o.double().numpy()),
                (v3._next_logits(chg_l, group).double().numpy() - v3._next_logits(base_l, group).double().numpy()),
                base_o.double().numpy(),
            )
        out["t_mod"].append(res["t"][0]); out["a_mod"].append(res["a"][0])
        out["t_log"].append(res["t"][1]); out["a_log"].append(res["a"][1])
        out["t_out"].append(res["t"][2])
        _l, act_in, _o = v3._model_forward_original(model, family, layer, group, inputs)
        out["in_norm"].append(np.linalg.norm(act_in.double().numpy(), axis=-1))
        cursor += count
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


_PAD: dict[int, int] = {}


def v3_pad(model) -> int:
    return _PAD["pad"]


def _measure(pred: dict[str, np.ndarray]) -> dict[str, float]:
    return {
        "module_effect_max_abs_error": float(np.max(np.abs(pred["t_mod"] - pred["a_mod"]))),
        "logit_effect_max_abs_error": float(np.max(np.abs(pred["t_log"] - pred["a_log"]))),
    }


def diagnose_case(model, tokenizer, checkpoint, case, custody: Path, challenge_row: dict[str, Any]) -> dict[str, Any]:
    family, seed_index, layer = checkpoint.family, int(case["seed_index"]), int(case["selected_layer"])
    cdir = custody / "challenges" / family / f"seed-{seed_index}"
    fixture = v3._load_json(cdir / "prompts.json")
    if v3._sha256((cdir / "prompts.json").read_bytes()) != challenge_row["prompt_file_sha256"]:
        raise ValueError("challenge prompt digest mismatch")
    rows = v3._token_rows(tokenizer, fixture["records"], with_continuation=True)
    deltas = np.load(cdir / "interventions.npz")["deltas"].astype(np.float32)
    modules = v3._module_map(custody, case)
    _PAD["pad"] = tokenizer.pad_token_id
    result: dict[str, Any] = {"family": family, "model_id": checkpoint.model_id, "seed_index": seed_index, "selected_layer": layer}
    preds = {}
    for label, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        model.to(dtype)
        cast = {k: m.to(dtype) for k, m in modules.items()}
        with torch.no_grad():
            preds[label] = _predictions(model, family, layer, rows, cast, case, deltas, dtype)
        result[label] = _measure(preds[label])
    p = preds["float64"]
    result["max_abs_boundary_output"] = float(np.max(np.abs(p["t_out"])))
    result["median_boundary_output_l2"] = float(np.median(np.linalg.norm(p["t_out"], axis=-1)))
    result["median_module_input_l2"] = float(np.median(p["in_norm"]))
    result["relative_fp32_module_error"] = result["float32"]["module_effect_max_abs_error"] / max(result["max_abs_boundary_output"], 1e-30)
    return result


def diagnose_existing(custody_root: str, out_path: str) -> dict[str, Any]:
    """Read-only: re-measure an existing r6 custody (exact fitted modules and challenge inputs)."""

    freeze = verify_packet()
    root = Path(custody_root).expanduser().resolve()
    manifest_path = root / "checkpoints.json"
    cm = v3._load_json(root / "challenge-manifest.json")
    checkpoint = {c.family: c for c in v3._load_checkpoints(manifest_path, root)}[DIAGNOSTIC_FAMILY]
    prep = v3._load_json(root / "candidate-preparation.json")
    with r6._activate_v8s6():
        model, tokenizer = v3._load_local_model(checkpoint)
    rows = []
    for crow in cm["cases"]:
        if crow["family"] != DIAGNOSTIC_FAMILY:
            continue
        case = next(c for c in prep["cases"] if c["family"] == DIAGNOSTIC_FAMILY and c["seed_index"] == crow["seed_index"])
        rows.append(diagnose_case(model, tokenizer, checkpoint, case, root, crow))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    summary = classify(rows)
    summary["spearman_fp32_error_vs_max_abs_boundary_output"] = spearman(
        [r["float32"]["module_effect_max_abs_error"] for r in rows], [r["max_abs_boundary_output"] for r in rows])
    r6_report = r6._read_json(root / "replacement-loss-v8s6-report.json")
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "arm": r6_report.get("arm"),
        "inputs": "original r6 custody (same fitted modules, same challenge prompts and interventions)",
        "r7_packet_sha256": freeze["packet_sha256"], "r6_report_packet_sha256": r6_report.get("packet_sha256"),
        "diagnostic_only": True, "cases": rows, "classification": summary, "created_at_utc": r6._now(),
    }
    r6._write_json(Path(out_path), report)
    return {"arm": report["arm"], "classification": summary}


def run(custody_root: str, checkpoint_manifest: str, arm: str, prior_custody_roots: list[str] | None = None) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    freeze = verify_packet()
    root = r6._custody_root(custody_root)
    manifest_path = Path(checkpoint_manifest).expanduser().resolve()
    prior = prior_custody_roots or []
    r6.attach_custody(root)
    with r6._arm_profile(arm):
        packet = r6.verify_packet(root)
        with r6._activate_v8s6():
            r6._validate_checkpoint_manifest(root, manifest_path, packet["packet_sha256"])
            disj = v6_core._validate_all_disjoint(prior, r6._fixture_payloads_for_preflight())
            r6._write_json(root / "v8s6-disjointness-preflight.json", disj)
            with v6_core._v6_profile():
                v3.archive_frozen_packet(root)
                v3.prepare_candidates(root, manifest_path)
                prep = r6._read_json(root / "candidate-preparation.json")
                prep["v8s6_packet_sha256"] = packet["packet_sha256"]
                r6._write_json(root / "candidate-preparation.json", prep)
                v3.freeze_candidates(root)
                v3.make_challenges(root, manifest_path)
                cm = r6._add_v8s6_binding_to_challenge_manifest(root, packet["packet_sha256"])
                checkpoints = {c.family: c for c in v3._load_checkpoints(v3._custodied_manifest_path(manifest_path, root), root)}
                checkpoint = checkpoints[DIAGNOSTIC_FAMILY]
                model, tokenizer = v3._load_local_model(checkpoint)
                rows = []
                case_rows = {(c["family"], c["seed_index"]): c for c in r6._read_json(root / "candidate-preparation.json")["cases"]}
                for crow in cm["cases"]:
                    if crow["family"] != DIAGNOSTIC_FAMILY:
                        continue
                    case = case_rows[(crow["family"], crow["seed_index"])]
                    rows.append(diagnose_case(model, tokenizer, checkpoint, case, root, crow))
                    print(json.dumps(rows[-1], sort_keys=True), flush=True)
    summary = classify(rows)
    summary["spearman_fp32_error_vs_max_abs_boundary_output"] = spearman(
        [r["float32"]["module_effect_max_abs_error"] for r in rows], [r["max_abs_boundary_output"] for r in rows])
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "arm": arm,
        "r7_packet_sha256": freeze["packet_sha256"], "r6_packet_sha256": packet["packet_sha256"],
        "diagnostic_only": True, "cases": rows, "classification": summary,
        "challenge_entropy_note": "challenge prompts regenerated this run unless stated otherwise in the results doc",
        "created_at_utc": r6._now(),
    }
    r6._write_json(root / "replacement-loss-v8s7-report.json", report)
    return {"arm": arm, "classification": summary, "report_sha256": r6._sha(root / "replacement-loss-v8s7-report.json")}


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    r = sub.add_parser("run")
    r.add_argument("--custody-root", required=True)
    r.add_argument("--checkpoint-manifest", required=True)
    r.add_argument("--arm", required=True, choices=ARMS)
    r.add_argument("--prior-custody-root", action="append", default=[])
    d = sub.add_parser("diagnose")
    d.add_argument("--custody-root", required=True)
    d.add_argument("--out", required=True)
    a = p.parse_args()
    if a.command == "freeze":
        out = freeze_packet()
    elif a.command == "verify":
        out = verify_packet()
    elif a.command == "diagnose":
        out = diagnose_existing(a.custody_root, a.out)
    else:
        out = run(a.custody_root, a.checkpoint_manifest, a.arm, a.prior_custody_root)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
