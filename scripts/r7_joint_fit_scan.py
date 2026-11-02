"""Joint six-case feasibility scan for the r7 C1 two-level mechanism (objective 1, rsi/problem.md).

Iteration 20 (ledger, `scripts/r7_two_level_toy.py`) demonstrated the
single-mechanism two-level account for ONE toy cell (seed 8, max |logit| ~160).
It left three open items:

  1. whether the same mechanism can jointly fit ALL SIX frozen r7 case pairs
     simultaneously;
  2. the exact logit-magnitude window in which each case, and all six together,
     are fit;
  3. a rejection band: logit magnitudes at which at least one case is
     unmatchable by the mechanism.

This scan answers all three, read-only. It runs the existing two-level toy
DIRECTLY on a (seed, magnitude) grid — no analytic law is assumed: a
pre-declared law check (err_f64 = k(seed) * ulp_f32(L) with per-seed constant
k) FAILS empirically across binades (e.g. seed 0 gives 1.9e-6 at L~100 but
3e-8 at L~128), so the grid itself is the measurement.

Predeclared reading (fixed before running; a declared analysis parameter, not
a tuned constant): a frozen case with float64 floor e64 and float32 floor e32
is "matched" at (seed, L) when

    |err_f64(seed,L) - e64| <= case_tol * e64   and
    |err_f32(seed,L) - e32| <= case_tol * e32

with case_tol = 0.25 (the toy's sample count differs from the real cases'; the
test suite independently hard-asserts the frozen case values taken from the
frozen reports). A common logit-magnitude window is the set of grid L for
which EVERY case has at least one matching seed. Grid magnitudes below the
lower edge of that window form the rejection band for the mechanism.

No model is run, no torch is installed, no frozen file is modified. This is a
hypothesis probe: the real r7 max |logit| remains unrecorded in any frozen
report, so this constrains, and does not confirm, C1.
Run: python3 scripts/r7_joint_fit_scan.py            (default grid, ~1 min)
     python3 scripts/r7_joint_fit_scan.py --json
     python3 scripts/r7_joint_fit_scan.py --seeds 0,1,2 --logits 128,160,192 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r7_two_level_toy as toy  # noqa: E402


REPORTS = {
    "NN256": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "docs", "replacement-loss-v8-scale-r7-NN256-report.json"),
    "NN512": os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        os.pardir, "docs", "replacement-loss-v8-scale-r7-NN512-report.json"),
}

CASE_TOL = 0.25  # predeclared per-case match tolerance
DEFAULT_LOGITS = tuple(float(l) for l in np.arange(64.0, 328.0, 8.0))
DEFAULT_SEEDS = tuple(range(64))
TOL_SENSITIVITY = (0.25, 0.5)  # primary tol first; second values are sensitivity only


def frozen_cases() -> list[dict]:
    """Read the six frozen r7 case pairs (read-only)."""
    cases = []
    for name, path in sorted(REPORTS.items()):
        with open(path) as f:
            data = json.load(f)
        for i, case in enumerate(data["cases"]):
            cases.append({
                "case": f"{name}-case{i}",
                "e64": case["float64"]["logit_effect_max_abs_error"],
                "e32": case["float32"]["logit_effect_max_abs_error"],
                "max_abs_boundary_output": case["max_abs_boundary_output"],
            })
    return cases


def measure_cell(seed: int, logit_mag: float) -> dict:
    """Run the two-level toy at one (seed, magnitude) cell."""
    r = toy.run(seed=seed, target_logit=logit_mag)
    return {
        "err_f64": r["f64_logit32"]["logit_effect_max_abs_error"],
        "err_f32": r["f32"]["logit_effect_max_abs_error"],
        "f64_control": r["f64_control"]["logit_effect_max_abs_error"],
    }


def cell_matches(cell: dict, case: dict, tol: float = CASE_TOL) -> bool:
    return (abs(cell["err_f64"] - case["e64"]) <= tol * case["e64"]
            and abs(cell["err_f32"] - case["e32"]) <= tol * case["e32"])


def build_cells(logits: tuple[float, ...], seeds: tuple[int, ...],
                verbose: bool = False) -> dict:
    """Measure every (seed, magnitude) cell ONCE; keys are (seed, mag)."""
    table = {}
    n = len(seeds) * len(logits)
    done = 0
    for mag in logits:
        for seed in seeds:
            table[(seed, float(mag))] = measure_cell(seed, mag)
            done += 1
            if verbose and done % 64 == 0:
                print(f"  cells {done}/{n}", file=sys.stderr)
    return table


def scan(logits: tuple[float, ...] = DEFAULT_LOGITS,
         seeds: tuple[int, ...] = DEFAULT_SEEDS,
         tols: tuple[float, ...] = TOL_SENSITIVITY,
         verbose: bool = False) -> dict:
    """Measure every (seed, magnitude) cell once and evaluate the joint fit.

    The first tolerance in `tols` is the primary analysis; the rest are
    sensitivity values reported alongside, never instead of, the primary.
    """
    cases = frozen_cases()
    table = build_cells(logits, seeds, verbose=verbose)
    cells64 = [c["err_f64"] for c in table.values()]
    cells32 = [c["err_f32"] for c in table.values()]

    per_tol = {}
    for tol in tols:
        per_case = []
        for case in cases:
            fits = []
            for mag in logits:
                seeds_ok = [seed for seed in seeds
                            if cell_matches(table[(seed, float(mag))], case, tol)]
                if seeds_ok:
                    fits.append({"mag": float(mag), "n_seeds": len(seeds_ok),
                                 "seeds": seeds_ok})
            per_case.append({
                "case": case["case"],
                "e64": case["e64"],
                "e32": case["e32"],
                "n_cells": len(fits),
                "min_mag": min((f["mag"] for f in fits), default=None),
                "max_mag": max((f["mag"] for f in fits), default=None),
                "seeds_union": sorted({s for f in fits for s in f["seeds"]}),
            })
        common = []
        for mag in logits:
            ok = all(any(cell_matches(table[(seed, float(mag))], case, tol)
                         for seed in seeds)
                     for case in cases)
            if ok:
                common.append(float(mag))
        per_tol[tol] = {
            "per_case": per_case,
            "common_window": {
                "feasible": bool(common),
                "min_mag": min(common) if common else None,
                "max_mag": max(common) if common else None,
                "n_mags": len(common),
            },
            "n_cases_matched": sum(1 for pc in per_case if pc["n_cells"]),
        }

    return {
        "case_tol_primary": tols[0],
        "tols": list(tols),
        "n_seeds": len(seeds),
        "cell_stats": {
            "err_f64_max": max(cells64),
            "err_f64_zero_cells": sum(1 for v in cells64 if v == 0.0),
            "err_f32_min": min(cells32),
            "err_f32_max": max(cells32),
        },
        "per_tol": per_tol,
        "grid": {
            "min": float(min(logits)), "max": float(max(logits)),
            "n": len(logits),
            "step": float(logits[1] - logits[0]) if len(logits) > 1 else 0.0,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--logits", type=str, default=",".join(f"{l:g}" for l in DEFAULT_LOGITS))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    logits = tuple(float(s) for s in args.logits.split(",") if s.strip())
    res = scan(logits=logits, seeds=seeds, verbose=args.verbose)
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
        return
    print(f"predeclared: seeds 0..{res['n_seeds']-1}, "
          f"grid [{res['grid']['min']:g}, {res['grid']['max']:g}] step "
          f"{res['grid']['step']:g} ({res['grid']['n']} points); cell stats "
          f"err_f64 max {res['cell_stats']['err_f64_max']:.3e} "
          f"(zero in {res['cell_stats']['err_f64_zero_cells']} cells), "
          f"err_f32 [{res['cell_stats']['err_f32_min']:.2e}, "
          f"{res['cell_stats']['err_f32_max']:.2e}]")
    for tol, rt in res["per_tol"].items():
        label = "primary" if tol == res["case_tol_primary"] else "sensitivity"
        print(f"--- tol {tol:g} ({label}) ---")
        for pc in rt["per_case"]:
            if pc["n_cells"]:
                print(f"{pc['case']:>11}: e64={pc['e64']:.3e} e32={pc['e32']:.3e} "
                      f"feasible cells {pc['n_cells']}, "
                      f"L in [{pc['min_mag']:g}, {pc['max_mag']:g}], "
                      f"{len(pc['seeds_union'])}/{res['n_seeds']} seeds usable")
            else:
                print(f"{pc['case']:>11}: e64={pc['e64']:.3e} e32={pc['e32']:.3e} "
                      f"NO match on grid")
        w = rt["common_window"]
        if w["feasible"]:
            print(f"common window (all six cases simultaneously): {w['n_mags']} grid "
                  f"magnitudes, L in [{w['min_mag']:g}, {w['max_mag']:g}]")
        else:
            print("common window: EMPTY on this grid - the six cases are not "
                  "jointly fit by this mechanism at these magnitudes")


if __name__ == "__main__":
    main()
