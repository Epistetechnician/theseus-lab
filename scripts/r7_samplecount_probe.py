"""Sample-count probe for the r7 two-level mechanism's per-case fit (objective 1, rsi/problem.md).

Iteration 21 (`scripts/r7_joint_fit_scan.py`) measured the two-level toy
(float32 logit path) on 2,112 (seed 0-63, L = 64..320 step 8) cells at the
toy's default n = 256 samples and found: 3/6 frozen r7 case pairs matched at
the predeclared +-25% per-case tolerance, and NO common logit-magnitude window
for all six (only L in [88, 288] when the tolerance was doubled to +-50%).
The binding constraint is the discreteness of rounding-boundary draws: the
err_f64 statistic is a max over samples of |cast-difference differences|, and a
max over more samples can reach larger lattice values. The recorded open
prediction: with more samples per case than the toy's 256, the per-case ratio
gaps can close.

This scan tests exactly that half of the prediction, in pure numpy, at
fixed predeclared parameters (fixed before the recorded run):

  * n values: 256 (the iteration-21 anchor, re-measured for comparability),
    1024, 4096;
  * seeds 0..15 (the full 0..63 pool would be ~4x slower at n=4096 and this
    is the recorded declared model);
  * the SAME grid as iteration 21: L = 64..320 step 8, so cell (seed, L) values
    at n=256 can be compared one-to-one against the recorded scan;
  * the SAME predeclared match rule and tolerances, imported from
    `r7_joint_fit_scan`: a frozen case pair is matched when
    |toy - case| / case <= tol on BOTH the float64 and float32 floors, with
    primary tol 0.25 and sensitivity 0.5.

Predeclared reading: the halved-gap prediction is supported iff, as n grows,
(1) the occupied err/ulp lattice levels become denser, and (2) at some measured
n level the three cases unmatched at n=256 become matchable at the primary
tolerance, or the +-25% common window becomes non-empty. n does not AND does
not need to equal the cases' true (unrecorded) sample count; this is a
hypothesis probe, not a confirmation. No model, no torch, no weights, and no
frozen file is touched.
Run: python3 scripts/r7_samplecount_probe.py            (full predeclared config, ~4 min)
     python3 scripts/r7_samplecount_probe.py --json
     python3 scripts/r7_samplecount_probe.py --seeds 0,1 --logits 128,160 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r7_two_level_toy as toy  # noqa: E402
import r7_joint_fit_scan as scan_mod  # noqa: E402


N_VALUES = (256, 1024, 4096)          # predeclared sample counts
SEEDS = tuple(range(16))              # predeclared seed pool for this scan
LOGITS = tuple(float(l) for l in np.arange(64.0, 328.0, 8.0))  # iteration-21 grid
TOL_SENSITIVITY = (0.25, 0.5)
N_UNMATCHED_AT_N256 = ("NN256-case0", "NN256-case1", "NN512-case0")  # iteration-21 result


def measure_cell_n(seed: int, logit_mag: float, n: int) -> dict:
    """Run the two-level toy at one (seed, magnitude, n) cell."""
    r = toy.run(seed=seed, target_logit=logit_mag, n=n)
    f64 = r["f64_logit32"]
    return {
        "err_f64": f64["logit_effect_max_abs_error"],
        "err_f32": r["f32"]["logit_effect_max_abs_error"],
        "f64_control": r["f64_control"]["logit_effect_max_abs_error"],
        "err_per_ulp": f64["logit_effect_max_abs_error"] / f64["ulp_of_max_logit"],
    }


def evaluate(table: dict, cases: list[dict], tols: tuple[float, ...] = TOL_SENSITIVITY) -> dict:
    """Apply the imported iteration-21 match rule at every tolerance.

    `table` maps (seed, mag) -> cell dict for ONE n level.
    """
    seeds = sorted({s for s, _ in table})
    mags = sorted({m for _, m in table})
    per_tol = {}
    for tol in tols:
        per_case = []
        for case in cases:
            fits = []
            for mag in mags:
                seeds_ok = [seed for seed in seeds
                            if scan_mod.cell_matches(table[(seed, float(mag))], case, tol)]
                if seeds_ok:
                    fits.append({"mag": float(mag), "n_seeds": len(seeds_ok)})
            per_case.append({
                "case": case["case"],
                "e64": case["e64"],
                "e32": case["e32"],
                "n_cells": len(fits),
                "min_mag": min((f["mag"] for f in fits), default=None),
                "max_mag": max((f["mag"] for f in fits), default=None),
                "matched": len(fits) > 0,
            })
        common = [float(mag) for mag in mags
                  if all(any(scan_mod.cell_matches(table[(seed, float(mag))], case, tol)
                             for seed in seeds) for case in cases)]
        per_tol[tol] = {
            "n_cases_matched": sum(1 for pc in per_case if pc["matched"]),
            "matched_names": sorted(pc["case"] for pc in per_case if pc["matched"]),
            "common_window": {
                "feasible": bool(common),
                "min_mag": min(common) if common else None,
                "max_mag": max(common) if common else None,
                "n_mags": len(common),
            },
            "per_case": per_case,
        }
    return per_tol


def cell_stats(table: dict) -> dict:
    errs = [c["err_per_ulp"] for c in table.values()]
    candidates64 = sorted({c for c in errs})
    return {
        "err_f64_max": max(c["err_f64"] for c in table.values()),
        "err_per_ulp_max": max(errs),
        "err_per_ulp_distinct_levels": len(candidates64),
        "err_f32_min": min(c["err_f32"] for c in table.values()),
        "err_f32_max": max(c["err_f32"] for c in table.values()),
        "f64_control_max": max(c["f64_control"] for c in table.values()),
    }


def run_probe(n_values: tuple[int, ...] = N_VALUES,
              seeds: tuple[int, ...] = SEEDS,
              logits: tuple[float, ...] = LOGITS,
              tols: tuple[float, ...] = TOL_SENSITIVITY,
              verbose: bool = False) -> dict:
    """Measure every (seed, magnitude) cell at each n level ONCE and evaluate."""
    cases = scan_mod.frozen_cases()
    per_n = {}
    for n in n_values:
        table = {}
        total = len(seeds) * len(logits)
        done = 0
        for mag in logits:
            for seed in seeds:
                table[(seed, float(mag))] = measure_cell_n(seed, mag, n)
                done += 1
                if verbose and done % 128 == 0:
                    print(f"  n={n}: cells {done}/{total}", file=sys.stderr)
        stats = cell_stats(table)
        per_n[n] = {"cell_stats": stats, "per_tol": evaluate(table, cases, tols)}
    return {
        "predeclared": {
            "n_values": list(n_values),
            "n_seeds": len(seeds),
            "grid": {"min": float(min(logits)), "max": float(max(logits)),
                     "n": len(logits),
                     "step": float(logits[1] - logits[0]) if len(logits) > 1 else 0.0},
            "case_tol_primary": tols[0],
            "match_rule_imported_from": "r7_joint_fit_scan",
            "unmatched_at_n256_iteration21": list(N_UNMATCHED_AT_N256),
        },
        "per_n": per_n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str,
                    default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--logits", type=str,
                    default=",".join(f"{l:g}" for l in LOGITS))
    ap.add_argument("--n-python", type=str,
                    default=",".join(str(n) for n in N_VALUES))
    ap.add_argument("--quantile", type=float, default=0.0, help="unused; kept for CLI symmetry")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    logits = tuple(float(s) for s in args.logits.split(",") if s.strip())
    n_values = tuple(int(x) for x in args.n_python.split(",") if x.strip())
    res = run_probe(n_values, seeds, logits, verbose=args.verbose)
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
        return
    p = res["predeclared"]
    print(f"predeclared: n in {p['n_values']}, seeds 0..{p['n_seeds']-1}, "
          f"grid [{p['grid']['min']:g}, {p['grid']['max']:g}] step "
          f"{p['grid']['step']:g} ({p['grid']['n']} mags); match rule from "
          f"r7_joint_fit_scan (tol 0.25 primary, 0.5 sensitivity)")
    print(f"iteration-21 baseline: 3/6 matched at ±25%, common window EMPTY")
    for n, entry in res["per_n"].items():
        cs = entry["cell_stats"]
        print(f"--- n={n} ---")
        print(f"   cells: err/ulp max {cs['err_per_ulp_max']:.4f} over "
              f"{cs['err_per_ulp_distinct_levels']} distinct levels; "
              f"err_f32 [{cs['err_f32_min']:.2e}, {cs['err_f32_max']:.2e}]; "
              f"f64_control max {cs['f64_control_max']:.1e}")
        for tol, rt in entry["per_tol"].items():
            label = "primary" if tol == TOL_SENSITIVITY[0] else "sensitivity"
            print(f"   tol {tol:g} ({label}): "
                  f"{rt['n_cases_matched']}/6 matched -> {', '.join(rt['matched_names'])}")
            w = rt["common_window"]
            if w["feasible"]:
                print(f"      common window: {w['n_mags']} mags, "
                      f"L in [{w['min_mag']:g}, {w['max_mag']:g}]")
            else:
                print("      common window: EMPTY")
    rescued = sorted(n for n, e in res["per_n"].items()
                     if any(pc["matched"] for pc in e["per_tol"][TOL_SENSITIVITY[0]]["per_case"]
                            if pc["case"] in N_UNMATCHED_AT_N256))
    print(f"predeclared reading (n=256-unmatched cases matchable at the primary "
          f"tolerance): n in {rescued}")


if __name__ == "__main__":
    main()
