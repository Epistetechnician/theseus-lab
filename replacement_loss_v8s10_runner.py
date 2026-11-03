"""float64 measurement gate over already-fitted modules (replacement-loss-v8-scale-r10).

State slice: theseus-lab-sandbox-v1. Measurement only: never refits, never
reclassifies an earlier arm (r6, r9, Band B outcomes stay as recorded). Model and
prompt data stay in external owner custody (read-only). Standard output carries the
full per-case state; nothing is persisted except files passed with --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import replacement_loss_v3_runner as v3
import replacement_loss_v8s7_runner as r7
import replacement_loss_v8s8_runner as r8
import replacement_loss_v8s9_runner as r9
import replacement_loss_v8sb_runner as bandb

STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_ID = "theseus-replacement-loss-v8-scale-r10-float64-measurement-gate"
FREEZE_PATH = "docs/replacement-loss-v8s10-freeze.json"

# Preregistered float64 gate (docs/replacement-loss-v8-scale-r10.md section 4).
MODULE_MAX_ABS = 1e-10
LOGIT_MAX_ABS = 1e-4
COSINE_MIN = 0.90
# Preregistered decision constants (section 5).
FALSIFY_MODULE_ABS = 1e-8
FALSIFY_MIN_BAND_A_FAILS = 2
INCONCLUSIVE_BELOW_FRACTION = 0.60
OPEN_LLAMA_COSINE_FLOOR = 0.99
R8_REPRODUCTION_TOLERANCE = 0.005
# Informational only (never gates): float32 re-measurement should reproduce the recorded cosines.
FLOAT32_REPRODUCTION_TOLERANCE = 1e-3

SEEDS = (0, 1, 2)
FAMILIES = ("opt_relu", "gpt2_gelu", "llama_silu_gated")
BAND_A, BAND_B_POS, BAND_B_NEG = "band_a_positive", "band_b_positive", "band_b_negative_control"

R6 = "docs/replacement-loss-v8-scale-r6-{arm}-report.json"
R9 = "docs/replacement-loss-v8-scale-r9-{arm}-report.json"
BB = "docs/replacement-loss-v8-scale-bandb-{arm}-report.json"
R8_REPORT = "docs/replacement-loss-v8-scale-r8-report.json"

# arm id -> (group, recorded report, families planned)
ARMS: dict[str, dict[str, Any]] = {
    "r6-NN256": {"group": BAND_A, "report": R6.format(arm="NN256"), "families": FAMILIES},
    "r6-NN512": {"group": BAND_A, "report": R6.format(arm="NN512"), "families": FAMILIES},
    "r9-NN256": {"group": BAND_A, "report": R9.format(arm="NN256"), "families": FAMILIES},
    "r9-NN512": {"group": BAND_A, "report": R9.format(arm="NN512"), "families": FAMILIES},
    "bandb-N512": {"group": BAND_B_POS, "report": BB.format(arm="N512"), "families": FAMILIES},
    "bandb-N256": {"group": BAND_B_NEG, "report": BB.format(arm="N256"), "families": ("opt_relu",)},
}
PLANNED_CASES = sum(len(a["families"]) * len(SEEDS) for a in ARMS.values())

EARLIER_OUTCOMES_UNCHANGED = {
    "r6": {"NN256": "FAIL", "NN512": "FAIL"},
    "r9": {"NN256": "FAIL", "NN512": "FAIL"},
    "bandb": {"N256": "FAIL", "N512": "FAIL"},
    "bandb_opt-6.7b": {"N256": "FAIL", "N512": "PASS"},
}

V8S10_OWN_FILES = (
    "docs/replacement-loss-v8-scale-r10.md",
    "replacement_loss_v8s10_runner.py",
    "tests/test_replacement_loss_v8s10.py",
)
PINNED_RESULT_FILES = (
    "docs/replacement-loss-v8-scale-r5-results.md",
    "docs/replacement-loss-v8-scale-r5-N128-report.json",
    "docs/replacement-loss-v8-scale-r5-N256-report.json",
    "docs/replacement-loss-v8-scale-r5-N512-report.json",
    "docs/replacement-loss-v8-scale-r5-T256-report.json",
    "docs/replacement-loss-v8-scale-r6.md",
    "docs/replacement-loss-v8-scale-r6-results.md",
    "docs/replacement-loss-v8-scale-r6-NN128-report.json",
    "docs/replacement-loss-v8-scale-r6-NN256-report.json",
    "docs/replacement-loss-v8-scale-r6-NN512-report.json",
    "docs/replacement-loss-v8-scale-r6-T128-report.json",
    "docs/replacement-loss-v8-scale-r7.md",
    "docs/replacement-loss-v8-scale-r7-results.md",
    "docs/replacement-loss-v8-scale-r7-NN256-report.json",
    "docs/replacement-loss-v8-scale-r7-NN512-report.json",
    "docs/replacement-loss-v8-scale-r8.md",
    "docs/replacement-loss-v8-scale-r8-results.md",
    R8_REPORT,
    "docs/replacement-loss-v8-scale-r9.md",
    "docs/replacement-loss-v8-scale-r9-results.md",
    "docs/replacement-loss-v8-scale-r9-NN256-report.json",
    "docs/replacement-loss-v8-scale-r9-NN512-report.json",
    "docs/replacement-loss-v8-scale-bandb.md",
    "docs/replacement-loss-v8-scale-bandb-results.md",
    "docs/replacement-loss-v8-scale-bandb-N256-report.json",
    "docs/replacement-loss-v8-scale-bandb-N512-report.json",
    "docs/bandb-open-llama-diagnosis.md",
)
PINNED_FREEZES = {
    "pinned_v8s5_packet_sha256": "docs/replacement-loss-v8s5-freeze.json",
    "pinned_v8s6_packet_sha256": "docs/replacement-loss-v8s6-freeze.json",
    "pinned_v8s7_packet_sha256": "docs/replacement-loss-v8s7-freeze.json",
    "pinned_v8s8_packet_sha256": "docs/replacement-loss-v8s8-freeze.json",
    "pinned_v8s9_packet_sha256": "docs/replacement-loss-v8s9-freeze.json",
    "pinned_v8sb_packet_sha256": "docs/replacement-loss-v8sb-freeze.json",
}
V8S10_PACKET_FILES = tuple(dict.fromkeys((
    *V8S10_OWN_FILES,
    *PINNED_RESULT_FILES,
    *PINNED_FREEZES.values(),
    *r8.V8S8_PACKET_FILES,
    *r9.V8S9_PACKET_FILES,
)))


def _root() -> Path:
    return Path(__file__).resolve().parent


def freeze_packet() -> dict[str, Any]:
    target = _root() / FREEZE_PATH
    if target.exists():
        raise FileExistsError("r10 freeze is immutable")
    rows = [
        {"path": rel, "sha256": bandb._sha(_root() / rel), "bytes": (_root() / rel).stat().st_size}
        for rel in V8S10_PACKET_FILES
    ]
    pins = {key: bandb._read_json(_root() / rel)["packet_sha256"] for key, rel in PINNED_FREEZES.items()}
    payload: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "freeze_state": "FROZEN_BEFORE_V8S10_RUN",
        "created_at_utc": bandb._now(),
        "packet_files": rows,
        **pins,
        "measurement_only": True,
        "earlier_outcomes_unchanged": EARLIER_OUTCOMES_UNCHANGED,
        "gate": {"module_max_abs": MODULE_MAX_ABS, "logit_max_abs": LOGIT_MAX_ABS, "cosine_min": COSINE_MIN},
    }
    payload["packet_sha256"] = bandb._canonical_digest(payload)
    bandb._write_json(target, payload)
    return {"packet_sha256": payload["packet_sha256"], "packet_file_count": len(rows)}


def verify_packet() -> dict[str, Any]:
    manifest = bandb._read_json(_root() / FREEZE_PATH)
    expected = bandb._canonical_digest({k: v for k, v in manifest.items() if k != "packet_sha256"})
    if manifest.get("packet_sha256") != expected:
        raise ValueError("r10 freeze digest is invalid")
    for row in manifest["packet_files"]:
        if bandb._sha(_root() / row["path"]) != row["sha256"]:
            raise ValueError(f"frozen r10 source changed: {row['path']}")
    for key, rel in PINNED_FREEZES.items():
        if manifest[key] != bandb._read_json(_root() / rel)["packet_sha256"]:
            raise ValueError(f"pin mismatch: {key}")
    return {"packet_sha256": manifest["packet_sha256"], "packet_file_count": len(manifest["packet_files"]), "verified": True}


# ---------------------------------------------------------------- gate and classification

def case_gate(module_err: float, logit_err: float, strata_cosines: dict[str, float | None]) -> dict[str, Any]:
    """The preregistered float64 gate for one case: conditions 1-3 of section 4."""

    cos_ok = bool(strata_cosines) and all(c is not None and c >= COSINE_MIN for c in strata_cosines.values())
    out = {
        "module_ok": bool(module_err <= MODULE_MAX_ABS),
        "logit_ok": bool(logit_err <= LOGIT_MAX_ABS),
        "cosine_ok": cos_ok,
    }
    out["passes"] = out["module_ok"] and out["logit_ok"] and out["cosine_ok"]
    return out


def _fails_cosine(c: dict[str, Any]) -> bool:
    return not c["gate"]["cosine_ok"]


def classify(cases: list[dict[str, Any]], r8_reference: dict[int, dict[str, float]] | None = None) -> dict[str, Any]:
    """Apply the preregistered P1-P4 and classification rules.

    Each case: {"group", "arm", "family", "seed_index", "float64": {"module_effect_max_abs_error",
    "logit_effect_max_abs_error", "strata_median_cosine": {stratum: cosine}}, "gate": {...}}.
    `r8_reference` maps seed_index -> {stratum: r8 float64 median cosine} for open_llama N512 (P2).
    """

    runnable = len(cases)
    band_a = [c for c in cases if c["group"] == BAND_A]
    band_b = [c for c in cases if c["group"] == BAND_B_POS]
    neg = [c for c in cases if c["group"] == BAND_B_NEG]
    positives = band_a + band_b

    p1 = all(c["gate"]["passes"] for c in band_a)
    p2 = all(c["gate"]["passes"] for c in band_b)
    open_llama_checks: list[dict[str, Any]] = []
    for c in band_b:
        if c["family"] != "llama_silu_gated":
            continue
        cos = c["float64"]["strata_median_cosine"]
        floor_ok = all(v is not None and v >= OPEN_LLAMA_COSINE_FLOOR for v in cos.values())
        ref = (r8_reference or {}).get(c["seed_index"])
        repro_ok = ref is not None and all(abs(cos[s] - ref[s]) <= R8_REPRODUCTION_TOLERANCE for s in ref if s in cos)
        open_llama_checks.append({"seed_index": c["seed_index"], "floor_ok": floor_ok, "reproduces_r8_ok": repro_ok})
        p2 = p2 and floor_ok and repro_ok
    p3 = bool(neg) and all(_fails_cosine(c) for c in neg)
    p4 = all(c["float64"]["module_effect_max_abs_error"] <= MODULE_MAX_ABS for c in positives)

    neg_passes = [c for c in neg if c["gate"]["passes"]]
    band_a_hard_fails = [
        c for c in band_a
        if c["float64"]["module_effect_max_abs_error"] > FALSIFY_MODULE_ABS or _fails_cosine(c)
    ]
    result: dict[str, Any] = {
        "planned_cases": PLANNED_CASES,
        "runnable_cases": runnable,
        "runnable_fraction": runnable / PLANNED_CASES,
        "P1_band_a_positives_pass_float64_gate": p1,
        "P2_band_b_positives_pass_float64_gate": p2,
        "P2_open_llama_checks": open_llama_checks,
        "P3_negative_control_fails_cosine_in_every_run_case": p3,
        "P4_no_positive_module_error_above_1e-10": p4,
        "negative_control_cases_run": len(neg),
        "negative_control_cases_passing_float64_gate": len(neg_passes),
        "band_a_cases_failing_beyond_rounding": len(band_a_hard_fails),
        "positive_cases_failing_gate": [
            {"arm": c["arm"], "family": c["family"], "seed_index": c["seed_index"], "gate": c["gate"]}
            for c in positives if not c["gate"]["passes"]
        ],
        "earlier_outcomes_unchanged": EARLIER_OUTCOMES_UNCHANGED,
    }
    if runnable / PLANNED_CASES < INCONCLUSIVE_BELOW_FRACTION:
        result["outcome"] = "INCONCLUSIVE"
        result["reason"] = "fewer than 60% of planned cases runnable"
    elif neg_passes or len(band_a_hard_fails) >= FALSIFY_MIN_BAND_A_FAILS:
        result["outcome"] = "FALSIFIED"
        result["reason"] = (
            "negative control case(s) passed the float64 gate" if neg_passes
            else "two or more Band A positives fail beyond float64 rounding"
        )
    elif p1 and p2 and p3 and p4:
        result["outcome"] = "FLOAT32_WAS_LIMITING"
    elif p3:
        result["outcome"] = "PARTIAL"
    else:
        result["outcome"] = "UNCLASSIFIED_NEGATIVE_CONTROL_NOT_EVALUABLE"
        result["reason"] = "P3 not satisfied but no negative-control case passed the gate (or none ran)"
    return result


# ---------------------------------------------------------------- measurement core

def score_strata(rows: list[dict[str, Any]], eff: dict[str, np.ndarray]) -> dict[str, Any]:
    """Unchanged v3 effect metrics and floors; strata taken from the rows themselves."""

    base, rep = eff["base_next"].astype(np.float64), eff["repeat_next"].astype(np.float64)
    floors = np.maximum(1e-12, 1e-8 * np.linalg.norm(base - rep, axis=1))
    scales = np.maximum(np.linalg.norm(base, axis=1), 1e-12)
    logit_m = v3._metrics_for_effect_rows(eff["pred_logit"], eff["obs_logit"], floors, scales)
    mod_m = v3._metrics_for_effect_rows(eff["pred_mod"], eff["obs_mod"], np.full(len(rows), 1e-12), scales)
    result: dict[str, Any] = {"logits": v3._aggregate_effect(logit_m), "module_boundary": v3._aggregate_effect(mod_m), "per_stratum": {}}
    for s in sorted({r["stratum"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["stratum"] == s]
        result["per_stratum"][s] = {
            "logits": v3._aggregate_effect([logit_m[i] for i in idx]),
            "module_boundary": v3._aggregate_effect([mod_m[i] for i in idx]),
        }
    return result


def measure_precision(model, family, layer, rows, deltas, modules, case, dtype) -> dict[str, Any]:
    """One precision arm: permutation-diagnostic errors (r7 definition) plus cosine scoring (r8 definition)."""

    cast = {k: m.to(dtype) for k, m in modules.items()}
    with torch.no_grad():
        pred = r7._predictions(model, family, layer, rows, cast, case, deltas, dtype)
        eff = r8.arm_effects(model, family, layer, rows, deltas, cast["treatment"])
    out = r7._measure(pred)
    scored = score_strata(rows, eff)
    out["score"] = scored
    out["strata_median_cosine"] = {s: v["logits"].get("median_cosine") for s, v in scored["per_stratum"].items()}
    out["pooled_logit_median_cosine"] = scored["logits"].get("median_cosine")
    out["module_boundary_median_cosine"] = scored["module_boundary"].get("median_cosine")
    return out


def diagnose_case(model, rows, deltas, modules, case, family, layer, device) -> dict[str, Any]:
    """Informational float32 arm, then the gating float64 arm (exact cast of the same weights)."""

    r7._PAD["pad"] = r8.PAD["pad"]
    with r8._device_projection(device):
        model.to(device=device, dtype=torch.float32)
        f32 = measure_precision(model, family, layer, rows, deltas, modules, case, torch.float32)
        model.to(dtype=torch.float64)
        f64 = measure_precision(model, family, layer, rows, deltas, modules, case, torch.float64)
        model.to(dtype=torch.float32)
    return {"float32": f32, "float64": f64,
            "gate": case_gate(f64["module_effect_max_abs_error"], f64["logit_effect_max_abs_error"], f64["strata_median_cosine"])}


# ---------------------------------------------------------------- custody

def _custody_recorded(arm: str) -> dict[str, Any]:
    return bandb._read_json(_root() / ARMS[arm]["report"])


def verify_custody(custody_root: str, arm: str) -> dict[str, Any]:
    """Read-only custody check for one arm. Never regenerates anything; reports per-family status."""

    root = Path(custody_root).expanduser().resolve()
    spec = ARMS[arm]
    recorded = _custody_recorded(arm)
    receipt: dict[str, Any] = {"arm": arm, "custody_root": str(root), "families": {}, "files": {}}
    needed = {
        "challenge-manifest.json": "challenge_manifest_sha256",
        "candidate-preparation.json": "candidate_preparation_sha256",
        "checkpoints.json": "checkpoint_manifest_sha256",
    }
    ok_files = True
    for name, field in needed.items():
        path = root / name
        if not path.is_file():
            receipt["files"][name] = "MISSING"
            ok_files = False
            continue
        actual = v3._sha256(path.read_bytes())
        receipt["files"][name] = {"sha256": actual, "recorded": recorded.get(field), "matches_recorded": actual == recorded.get(field)}
    if not ok_files:
        for family in spec["families"]:
            receipt["families"][family] = {"status": "NOT_RUN", "reason": "custody metadata files missing"}
        return receipt
    cm = v3._load_json(root / "challenge-manifest.json")
    prep = v3._load_json(root / "candidate-preparation.json")
    checkpoints = {c.family: c for c in v3._load_checkpoints(root / "checkpoints.json", root)}
    for family in spec["families"]:
        problems: list[str] = []
        ckpt = checkpoints.get(family)
        if ckpt is None or not ckpt.local_path.is_dir() or not any(ckpt.local_path.glob("*.safetensors")) and not any(ckpt.local_path.glob("*.bin")):
            problems.append("model weights directory missing or empty")
        for seed in SEEDS:
            crow = next((c for c in cm["cases"] if c["family"] == family and int(c["seed_index"]) == seed), None)
            case = next((c for c in prep["cases"] if c["family"] == family and int(c["seed_index"]) == seed), None)
            if crow is None or case is None:
                problems.append(f"seed {seed}: case missing from manifests")
                continue
            cdir = root / "challenges" / family / f"seed-{seed}"
            for fname, field in (("prompts.json", "prompt_file_sha256"), ("interventions.npz", "interventions_file_sha256")):
                f = cdir / fname
                if not f.is_file():
                    problems.append(f"seed {seed}: {fname} missing")
                elif v3._sha256(f.read_bytes()) != crow[field]:
                    problems.append(f"seed {seed}: {fname} digest mismatch")
            for item in case["modules"]:
                if not (root / item["file"]).is_file():
                    problems.append(f"seed {seed}: module file {item['file']} missing")
        receipt["families"][family] = (
            {"status": "OK"} if not problems else {"status": "NOT_RUN", "reason": "custody unavailable: " + "; ".join(problems)}
        )
    return receipt


def _recorded_cosines(recorded: dict[str, Any]) -> dict[tuple[str, int], dict[str, float]]:
    return {
        (c["family"], int(c["seed_index"])): {s: v["logits"]["median_cosine"] for s, v in c["treatment_effects"]["per_stratum"].items()}
        for c in recorded["cases"]
    }


def diagnose_existing(custody_root: str, arm: str, families: list[str], out_path: str, device: str) -> dict[str, Any]:
    """Read-only: re-measure existing custody (same fitted modules, challenge prompts and deltas)."""

    freeze = verify_packet()
    spec = ARMS[arm]
    unplanned = [f for f in families if f not in spec["families"]]
    if unplanned:
        raise ValueError(f"families not planned for {arm}: {unplanned}")
    root = Path(custody_root).expanduser().resolve()
    cm = v3._load_json(root / "challenge-manifest.json")
    prep = v3._load_json(root / "candidate-preparation.json")
    checkpoints = {c.family: c for c in v3._load_checkpoints(root / "checkpoints.json", root)}
    recorded = _custody_recorded(arm)
    recorded_cos = _recorded_cosines(recorded)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    cases: list[dict[str, Any]] = []
    for family in families:
        checkpoint = checkpoints[family]
        saved_cap = v3.MAX_PARAMETER_COUNT
        v3.MAX_PARAMETER_COUNT = bandb.MAX_PARAMETER_COUNT
        try:
            model, tokenizer = v3._load_local_model(checkpoint)
        finally:
            v3.MAX_PARAMETER_COUNT = saved_cap
        r8.PAD["pad"] = tokenizer.pad_token_id
        for crow in cm["cases"]:
            if crow["family"] != family:
                continue
            seed_index = int(crow["seed_index"])
            case = next(c for c in prep["cases"] if c["family"] == family and int(c["seed_index"]) == seed_index)
            cdir = root / "challenges" / family / f"seed-{seed_index}"
            if v3._sha256((cdir / "prompts.json").read_bytes()) != crow["prompt_file_sha256"]:
                raise ValueError("challenge prompt digest mismatch")
            if v3._sha256((cdir / "interventions.npz").read_bytes()) != crow["interventions_file_sha256"]:
                raise ValueError("challenge intervention digest mismatch")
            rows = v3._token_rows(tokenizer, v3._load_json(cdir / "prompts.json")["records"], with_continuation=True)
            deltas = np.load(cdir / "interventions.npz")["deltas"].astype(np.float32)
            modules = v3._module_map(root, case)
            layer = int(case["selected_layer"])
            row: dict[str, Any] = {
                "arm": arm, "group": spec["group"], "family": family, "model_id": checkpoint.model_id,
                "seed_index": seed_index, "selected_layer": layer, "device": device,
                "prompt_file_sha256": crow["prompt_file_sha256"], "interventions_file_sha256": crow["interventions_file_sha256"],
            }
            row.update(diagnose_case(model, rows, deltas, modules, case, family, layer, device))
            rec = recorded_cos[(family, seed_index)]
            row["recorded_strata_median_cosine"] = rec
            row["float32_reproduces_recorded"] = {
                "max_abs_gap": max(abs(row["float32"]["strata_median_cosine"][s] - rec[s]) for s in rec),
                "within_tolerance": all(abs(row["float32"]["strata_median_cosine"][s] - rec[s]) <= FLOAT32_REPRODUCTION_TOLERANCE for s in rec),
                "informational_only": True,
            }
            cases.append(row)
            print(json.dumps(row, sort_keys=True, default=float), flush=True)
        del model
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "arm": arm, "families": families,
        "r10_packet_sha256": freeze["packet_sha256"], "measurement_only": True,
        "lineage": {
            "recorded_report": spec["report"], "recorded_packet_sha256": recorded.get("packet_sha256"),
            "custody_challenge_manifest_sha256": v3._sha256((root / "challenge-manifest.json").read_bytes()),
            "custody_candidate_preparation_sha256": v3._sha256((root / "candidate-preparation.json").read_bytes()),
            "custody_checkpoints_sha256": v3._sha256((root / "checkpoints.json").read_bytes()),
        },
        "cases": cases, "created_at_utc": bandb._now(),
    }
    bandb._write_json(Path(out_path), report)
    return {"arm": arm, "families": families, "cases": len(cases), "gate_passes": sum(c["gate"]["passes"] for c in cases)}


def r8_reference() -> dict[int, dict[str, float]]:
    rep = bandb._read_json(_root() / R8_REPORT)
    return {c["seed_index"]: {s: v["logits"]["median_cosine"] for s, v in c["float64"]["per_stratum"].items()} for c in rep["cases"]}


def assemble(result_files: list[str], custody_receipts: list[str], out_path: str) -> dict[str, Any]:
    """Merge partial per-arm reports and custody receipts into the final classified report."""

    freeze = verify_packet()
    cases: list[dict[str, Any]] = []
    for f in result_files:
        part = bandb._read_json(Path(f))
        if part["r10_packet_sha256"] != freeze["packet_sha256"]:
            raise ValueError(f"packet mismatch in {f}")
        cases.extend(part["cases"])
    seen = {(c["arm"], c["family"], c["seed_index"]) for c in cases}
    if len(seen) != len(cases):
        raise ValueError("duplicate case rows")
    receipts = [bandb._read_json(Path(f)) for f in custody_receipts]
    not_run: list[dict[str, Any]] = []
    for arm, spec in ARMS.items():
        for family in spec["families"]:
            if all((arm, family, s) in seen for s in SEEDS):
                continue
            reasons = [
                r["families"][family].get("reason", "custody OK but no result file")
                for r in receipts if r["arm"] == arm and family in r["families"] and r["families"][family]["status"] != "OK"
            ]
            not_run.append({"arm": arm, "family": family, "status": "NOT_RUN",
                            "reason": "; ".join(reasons) or "no result and no custody receipt"})
    report = {
        "protocol_id": PROTOCOL_ID, "state_slice": STATE_SLICE, "r10_packet_sha256": freeze["packet_sha256"],
        "measurement_only": True, "classification": classify(cases, r8_reference()),
        "not_run": not_run, "custody_receipts": receipts, "cases": cases, "created_at_utc": bandb._now(),
    }
    bandb._write_json(Path(out_path), report)
    return {"classification": report["classification"], "not_run": not_run}


def _cli() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze")
    sub.add_parser("verify")
    c = sub.add_parser("custody")
    c.add_argument("--custody-root", required=True)
    c.add_argument("--arm", required=True, choices=sorted(ARMS))
    c.add_argument("--out", required=True)
    d = sub.add_parser("diagnose")
    d.add_argument("--custody-root", required=True)
    d.add_argument("--arm", required=True, choices=sorted(ARMS))
    d.add_argument("--family", action="append", required=True, choices=FAMILIES)
    d.add_argument("--out", required=True)
    d.add_argument("--device", default="cuda")
    a = sub.add_parser("assemble")
    a.add_argument("--results", nargs="+", required=True)
    a.add_argument("--custody-receipts", nargs="*", default=[])
    a.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "freeze":
        out = freeze_packet()
    elif args.command == "verify":
        out = verify_packet()
    elif args.command == "custody":
        verify_packet()
        out = verify_custody(args.custody_root, args.arm)
        bandb._write_json(Path(args.out), out)
    elif args.command == "diagnose":
        out = diagnose_existing(args.custody_root, args.arm, args.family, args.out, args.device)
    else:
        out = assemble(args.results, args.custody_receipts, args.out)
    print(json.dumps(out, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
