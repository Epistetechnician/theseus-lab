"""float64 downstream check of the open_llama_7b_v2 logit-effect cosines (replacement-loss-v8-scale-r8).

State slice: theseus-lab-sandbox-v1. Diagnostic only: never reclassifies Band B.
Model and prompt data stay in external owner custody (read-only). Standard output
carries the full per-case state of the measurement; nothing is persisted by default
except the report file passed with --out.
"""

from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch

import replacement_loss_v3_runner as v3
import replacement_loss_v8s7_runner as r7
import replacement_loss_v8sb_runner as bandb

STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-r8-float64-downstream-logit-cosine-diagnostic"
DIAGNOSTIC_FAMILY = "llama_silu_gated"
STRATA = ("compositional", "ordinary", "adversarial")
PRIMARY_STRATA = ("compositional", "ordinary")
GATE = 0.90
H1_FLOOR = 0.99
REPRODUCTION_TOLERANCE = 0.02
FLOAT32_EPS = float(np.finfo(np.float32).eps)
BANDB_OUTCOMES = {
    "open_llama_7b_v2": {"N256": "FAIL", "N512": "FAIL"},
    "opt-6.7b": {"N256": "FAIL", "N512": "PASS"},
    "arm": {"N256": "FAIL", "N512": "FAIL"},
}
BANDB_N512_REPORT = "docs/replacement-loss-v8-scale-bandb-N512-report.json"
FREEZE_PATH = "docs/replacement-loss-v8s8-freeze.json"
V8S8_OWN_FILES = (
    "docs/replacement-loss-v8-scale-r8.md",
    "replacement_loss_v8s8_runner.py",
    "tests/test_replacement_loss_v8s8.py",
)
PINNED_RESULT_FILES = (
    "docs/replacement-loss-v8sb-freeze.json",
    "docs/replacement-loss-v8-scale-bandb.md",
    "docs/replacement-loss-v8-scale-bandb-results.md",
    "docs/replacement-loss-v8-scale-bandb-N256-report.json",
    BANDB_N512_REPORT,
    "docs/bandb-open-llama-diagnosis.md",
    "docs/replacement-loss-v8s7-freeze.json",
    "docs/replacement-loss-v8-scale-r7-results.md",
    "docs/replacement-loss-v8-scale-r7-NN256-report.json",
    "docs/replacement-loss-v8-scale-r7-NN512-report.json",
)
V8S8_PACKET_FILES = tuple(dict.fromkeys((
    *V8S8_OWN_FILES,
    *PINNED_RESULT_FILES,
    *bandb.V8SB_PACKET_FILES,
    *r7.V8S7_PACKET_FILES,
)))


def _root() -> Path:
    return Path(__file__).resolve().parent


def freeze_packet() -> dict[str, Any]:
    target = _root() / FREEZE_PATH
    if target.exists():
        raise FileExistsError("r8 freeze is immutable")
    rows = [
        {"path": rel, "sha256": bandb._sha(_root() / rel), "bytes": (_root() / rel).stat().st_size}
        for rel in V8S8_PACKET_FILES
    ]
    pins = {
        "pinned_v8sb_packet_sha256": bandb._read_json(_root() / "docs/replacement-loss-v8sb-freeze.json")["packet_sha256"],
        "pinned_v8s7_packet_sha256": bandb._read_json(_root() / r7.FREEZE_PATH)["packet_sha256"],
    }
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V8S8_RUN",
        "created_at_utc": bandb._now(),
        "packet_files": rows,
        **pins,
        "diagnostic_only": True,
        "bandb_outcomes_unchanged": BANDB_OUTCOMES,
    }
    payload["packet_sha256"] = bandb._canonical_digest(payload)
    bandb._write_json(target, payload)
    return {"packet_sha256": payload["packet_sha256"], "packet_file_count": len(rows)}


def verify_packet() -> dict[str, Any]:
    manifest = bandb._read_json(_root() / FREEZE_PATH)
    expected = bandb._canonical_digest({k: v for k, v in manifest.items() if k != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("r8 freeze digest is invalid")
    for row in manifest["packet_files"]:
        if bandb._sha(_root() / row["path"]) != row["sha256"]:
            raise ValueError(f"frozen r8 source changed: {row['path']}")
    for key, rel in (("pinned_v8sb_packet_sha256", "docs/replacement-loss-v8sb-freeze.json"),
                     ("pinned_v8s7_packet_sha256", r7.FREEZE_PATH)):
        if manifest[key] != bandb._read_json(_root() / rel)["packet_sha256"]:
            raise ValueError(f"pin mismatch: {key}")
    return {"packet_sha256": manifest["packet_sha256"], "packet_file_count": len(manifest["packet_files"]), "verified": True}


# ---------------------------------------------------------------- classification

def classify(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the preregistered validity gate and H1/H3 decision rules.

    Each cell: {"seed_index", "stratum", "c32", "c64", "bandb"} (median logit cosines).
    """

    reproduced = all(abs(c["c32"] - c["bandb"]) <= REPRODUCTION_TOLERANCE for c in cells)
    primary = [c["c64"] for c in cells if c["stratum"] in PRIMARY_STRATA]
    result: dict[str, Any] = {
        "validity_gate_float32_reproduces_bandb_within_0.02": reproduced,
        "max_abs_float32_reproduction_gap": max(abs(c["c32"] - c["bandb"]) for c in cells),
        "primary_float64_cosines": primary,
        "bandb_outcomes_unchanged": BANDB_OUTCOMES,
    }
    if not reproduced:
        result["outcome"] = "INCONCLUSIVE_NOT_REPRODUCED"
    elif all(v >= H1_FLOOR for v in primary):
        result["outcome"] = "H1_SUPPORTED_ROUNDING_FLOOR"
    elif all(v < GATE for v in primary):
        result["outcome"] = "H3_SUPPORTED_ARCHITECTURE_OR_FIT"
    elif all(v >= GATE for v in primary):
        result["outcome"] = "PARTIAL_RISE"
    else:
        result["outcome"] = "MIXED"
    return result


# ---------------------------------------------------------------- measurement core

@contextmanager
def _device_projection(device: str) -> Iterator[None]:
    """Make the pinned v3 projection helpers device- and dtype-aware without editing v3."""

    run_projection, set_projection = v3._run_projection, v3._set_projection

    def to_device(inputs: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

    def run_dev(model, family, layer, inputs, *a, **kw):
        res = run_projection(model, family, layer, to_device(inputs), *a, **kw)
        return tuple(x.cpu() if isinstance(x, torch.Tensor) else x for x in res)

    def set_dev(model, family, layer, module):
        p = next(model.parameters())
        set_projection(model, family, layer, copy.deepcopy(module).to(device=p.device, dtype=p.dtype))

    try:
        v3._run_projection, v3._set_projection = run_dev, set_dev
        yield
    finally:
        v3._run_projection, v3._set_projection = run_projection, set_projection


def arm_effects(model, family, layer, rows, deltas: np.ndarray, module) -> dict[str, np.ndarray]:
    """Observed (original model) and predicted (module-replaced) effects in the model's current dtype."""

    keys = ("obs_logit", "pred_logit", "obs_mod", "pred_mod", "base_next", "repeat_next")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    cursor = 0
    for group, inputs in v3._batches(rows, PAD["pad"]):
        d = torch.from_numpy(deltas[cursor: cursor + len(group)])
        cursor += len(group)
        base_l, _in, base_o = v3._model_forward_original(model, family, layer, group, inputs)
        rep_l, _i2, _o2 = v3._model_forward_original(model, family, layer, group, inputs)
        chg_l, _i3, chg_o = v3._model_forward_original(model, family, layer, group, inputs, deltas=d)
        tb_l, tb_o = v3._model_forward_variant(model, family, layer, module, group, inputs, None)
        tc_l, tc_o = v3._model_forward_variant(model, family, layer, module, group, inputs, d)
        nxt = lambda lg: v3._next_logits(lg, group).numpy()  # noqa: E731
        out["base_next"].append(nxt(base_l))
        out["repeat_next"].append(nxt(rep_l))
        out["obs_logit"].append(nxt(chg_l) - nxt(base_l))
        out["pred_logit"].append(nxt(tc_l) - nxt(tb_l))
        out["obs_mod"].append(chg_o.numpy() - base_o.numpy())
        out["pred_mod"].append(tc_o.numpy() - tb_o.numpy())
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


PAD: dict[str, int] = {"pad": 0}


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)


def score_arm(rows, eff: dict[str, np.ndarray]) -> dict[str, Any]:
    """Band B scoring (unchanged v3 metrics, floors and scale) for one precision arm."""

    base, rep = eff["base_next"].astype(np.float64), eff["repeat_next"].astype(np.float64)
    floors = np.maximum(1e-12, 1e-8 * np.linalg.norm(base - rep, axis=1))
    scales = np.maximum(np.linalg.norm(base, axis=1), 1e-12)
    logit_m = v3._metrics_for_effect_rows(eff["pred_logit"], eff["obs_logit"], floors, scales)
    mod_m = v3._metrics_for_effect_rows(eff["pred_mod"], eff["obs_mod"], np.full(len(rows), 1e-12), scales)
    result: dict[str, Any] = {"logits": v3._aggregate_effect(logit_m), "module_boundary": v3._aggregate_effect(mod_m), "per_stratum": {}}
    for s in STRATA:
        idx = [i for i, r in enumerate(rows) if r["stratum"] == s]
        result["per_stratum"][s] = {
            "logits": v3._aggregate_effect([logit_m[i] for i in idx]),
            "module_boundary": v3._aggregate_effect([mod_m[i] for i in idx]),
        }
    return result


def noise_and_scale(rows, e32: dict[str, np.ndarray], e64: dict[str, np.ndarray]) -> dict[str, Any]:
    """Per-stratum effect size relative to logit scale and float32 epsilon, and fp32-vs-fp64 noise ratios."""

    obs64, pred64 = e64["obs_logit"], e64["pred_logit"]
    obs32, pred32 = e32["obs_logit"].astype(np.float64), e32["pred_logit"].astype(np.float64)
    lognorm = np.linalg.norm(e64["base_next"], axis=1)
    rel_eff = np.linalg.norm(obs64, axis=1) / np.maximum(lognorm, 1e-300)
    nr_obs = np.linalg.norm(obs32 - obs64, axis=1) / np.maximum(np.linalg.norm(obs64, axis=1), 1e-300)
    nr_pred = np.linalg.norm(pred32 - pred64, axis=1) / np.maximum(np.linalg.norm(pred64, axis=1), 1e-300)
    cos_obs, cos_pred = _cos(obs32, obs64), _cos(pred32, pred64)
    out: dict[str, Any] = {}
    for s in STRATA:
        idx = np.asarray([i for i, r in enumerate(rows) if r["stratum"] == s])
        med = lambda x: float(np.median(x[idx]))  # noqa: E731
        out[s] = {
            "median_relative_effect_norm_over_logit_norm": med(rel_eff),
            "median_relative_effect_in_float32_eps": med(rel_eff) / FLOAT32_EPS,
            "median_fp32_vs_fp64_noise_ratio_observed": med(nr_obs),
            "median_fp32_vs_fp64_noise_ratio_predicted": med(nr_pred),
            "median_cos_fp32_vs_fp64_observed": med(cos_obs),
            "median_cos_fp32_vs_fp64_predicted": med(cos_pred),
        }
    return out


def diagnose_case(model, rows, deltas, module, family, layer, device) -> dict[str, Any]:
    """Run the float32 arm, then the float64 arm (exact cast of the same weights), on one case."""

    with _device_projection(device), torch.no_grad():
        model.to(device=device, dtype=torch.float32)
        e32 = arm_effects(model, family, layer, rows, deltas, module.float())
        model.to(dtype=torch.float64)
        e64 = arm_effects(model, family, layer, rows, deltas, module.double())
        model.to(dtype=torch.float32)
    return {"float32": score_arm(rows, e32), "float64": score_arm(rows, e64), "scale_and_noise": noise_and_scale(rows, e32, e64)}


def diagnose_existing(custody_root: str, out_path: str, device: str = "cuda") -> dict[str, Any]:
    """Read-only: re-measure the Band B N512 custody (same fitted modules, challenge prompts, deltas)."""

    freeze = verify_packet()
    root = Path(custody_root).expanduser().resolve()
    cm = v3._load_json(root / "challenge-manifest.json")
    prep = v3._load_json(root / "candidate-preparation.json")
    checkpoint = {c.family: c for c in v3._load_checkpoints(root / "checkpoints.json", root)}[DIAGNOSTIC_FAMILY]
    bandb_report = bandb._read_json(_root() / BANDB_N512_REPORT)
    bandb_cases = {c["seed_index"]: c for c in bandb_report["cases"] if c["family"] == DIAGNOSTIC_FAMILY}
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    saved_cap = v3.MAX_PARAMETER_COUNT
    v3.MAX_PARAMETER_COUNT = bandb.MAX_PARAMETER_COUNT
    try:
        model, tokenizer = v3._load_local_model(checkpoint)
    finally:
        v3.MAX_PARAMETER_COUNT = saved_cap
    PAD["pad"] = tokenizer.pad_token_id
    cases, cells, lineage = [], [], {}
    for crow in cm["cases"]:
        if crow["family"] != DIAGNOSTIC_FAMILY:
            continue
        seed_index = int(crow["seed_index"])
        case = next(c for c in prep["cases"] if c["family"] == DIAGNOSTIC_FAMILY and c["seed_index"] == seed_index)
        cdir = root / "challenges" / DIAGNOSTIC_FAMILY / f"seed-{seed_index}"
        if v3._sha256((cdir / "prompts.json").read_bytes()) != crow["prompt_file_sha256"]:
            raise ValueError("challenge prompt digest mismatch")
        if v3._sha256((cdir / "interventions.npz").read_bytes()) != crow["interventions_file_sha256"]:
            raise ValueError("challenge intervention digest mismatch")
        fixture = v3._load_json(cdir / "prompts.json")
        rows = v3._token_rows(tokenizer, fixture["records"], with_continuation=True)
        deltas = np.load(cdir / "interventions.npz")["deltas"].astype(np.float32)
        module = v3._module_map(root, case)["treatment"]
        row = {"family": DIAGNOSTIC_FAMILY, "seed_index": seed_index, "selected_layer": int(case["selected_layer"]),
               "prompt_file_sha256": crow["prompt_file_sha256"], "interventions_file_sha256": crow["interventions_file_sha256"]}
        row.update(diagnose_case(model, rows, deltas, module, DIAGNOSTIC_FAMILY, int(case["selected_layer"]), device))
        bb = bandb_cases[seed_index]["treatment_effects"]["per_stratum"]
        row["bandb_recorded_logit_median_cosine"] = {s: bb[s]["logits"]["median_cosine"] for s in STRATA}
        for s in STRATA:
            cells.append({"seed_index": seed_index, "stratum": s,
                          "c32": row["float32"]["per_stratum"][s]["logits"]["median_cosine"],
                          "c64": row["float64"]["per_stratum"][s]["logits"]["median_cosine"],
                          "bandb": bb[s]["logits"]["median_cosine"]})
        cases.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    classification = classify(cells)
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "arm": "N512 (Band B custody, read-only)",
        "r8_packet_sha256": freeze["packet_sha256"], "diagnostic_only": True,
        "lineage": {"bandb_report": BANDB_N512_REPORT, "bandb_packet_sha256": bandb_report.get("packet_sha256"),
                    "custody_challenge_manifest_sha256": v3._sha256((root / "challenge-manifest.json").read_bytes()),
                    "custody_checkpoints_sha256": v3._sha256((root / "checkpoints.json").read_bytes())},
        "cases": cases, "cells": cells, "classification": classification,
        "float32_eps": FLOAT32_EPS, "created_at_utc": bandb._now(),
    }
    bandb._write_json(Path(out_path), report)
    return {"classification": classification}


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    d = sub.add_parser("diagnose")
    d.add_argument("--custody-root", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.command == "freeze":
        out = freeze_packet()
    elif a.command == "verify":
        out = verify_packet()
    else:
        out = diagnose_existing(a.custody_root, a.out, a.device)
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
