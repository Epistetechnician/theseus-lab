"""Summation-order probe: an f32-accumulated readout entering the f64 comparison.

Iteration 23 (ledger, `scripts/r7_two_site_toy.py`) rejected the two-site
hidden-cast variant and left open candidate (b): "summation-order effects in
the f32 pass entering the f64 comparison, not the logit cast".

This probe measures that remaining family directly, with NO logit cast in the
float64 pass at all (candidate C1 and its variants make err_f64 a multiple of
one f32 ulp; iteration 22 measured that pure power-of-two lattice cannot
close the 26.6-29.7% mantissa gaps of the three unmatched cases):

  Mechanism M4: in the float64 pass, the readout matmul's reduction over the
  d channel terms is performed with FLOAT32 ACCUMULATION - every partial sum
  rounds to float32. The treatment arm adds the terms h[c] * w[c] in
  canonical channel order; the mapping-aligned arm adds the SAME term values
  in PERMUTED channel order (mathematically the identical sum, evaluated in
  the permuted reduction order of the real aligned arm's permuted readout
  copy). The hidden state and module matmuls stay exact float64, and the
  module arm difference remains the r7 module-effect scale (eps ~1e-13,
  `toy._build` reused unchanged), so the ONLY new sites are reduction-order
  f32 partial-sum roundings.

Assumption stated up front (hypothesis, not verified from torch internals):
this realizes the aligned arm's f32 reduction in permuted channel order. If
the real pipeline's aligned reduction order were identical to the treatment
arm's, this probe overstates the family.

Predeclared reading (fixed before the recorded run):
  * The summation-order family is LIVE as a mantissa densifier for the three
    unmatched cases iff all three hold on the predeclared grid:
      P1: some probed nonzero err_f64 has err_f64/ulp(max|logit|) whose
          relative distance to the nearest power-of-two level exceeds 2%
          (dense mantissa, i.e. NOT a clean 2**k multiple);
      P2: at least one of {NN256-case0, NN256-case1, NN512-case0} is matched
          at the primary +-25% case tolerance on the f64 side (the f64-side
          match is the primary reading here because the densification
          question lives on the f64 side; the unchanged f32 pass IS reported
          and the both-sides joint window is reported as sensitivity via the
          unchanged iteration-21 rule);
      P3: the exact-f64 readout control (accumulate=False, same orders) stays
          at the r7 module-effect scale (<= 1e-9), i.e. the error measured
          above comes from the f32 accumulation sites, not from the module
          construction.
  * Otherwise the family is REJECTED for the unmatched cases by the
    iteration-22 criterion (a max statistic stuck on the power-of-two
    lattice, or one that misses the cases, cannot close their mantissa gaps).

Grid (predeclared): seeds 0..15, max|logit| L in [8, 320] step 8. The grid
starts well BELOW the prior scans' [64, 320] because M4's expected err_f64
tracks the logit magnitude like the f32 pass does, so matching the ~5e-6 e64
band may require a smaller L than the cast-based mechanism.
Run: python3 scripts/r7_summ_order_toy.py            (~2-3 min)
     python3 scripts/r7_summ_order_toy.py --json
     python3 scripts/r7_summ_order_toy.py --seeds 0,1 --logits 8,16 --json
No model is run, no torch is installed, no frozen file is modified.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r7_two_level_toy as toy  # noqa: E402
from r7_joint_fit_scan import CASE_TOL, frozen_cases  # noqa: E402


DEFAULT_MAGNITUDES = tuple(float(l) for l in np.arange(8.0, 328.0, 8.0))
DEFAULT_SEEDS = tuple(range(16))
UNMATCHED_CASES = ("NN256-case0", "NN256-case1", "NN512-case0")
DENSE_THRESHOLD = 0.02   # rel. distance above this is "not a power-of-two level"
CONTROL_MAX = 1e-9       # P3 bound for the exact-f64 readout control


def seq_f32_accumulate(h64: np.ndarray, wr64: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Sequential f32 accumulation of the readout terms h[:, j] * wr64[j, :].

    Every partial sum is rounded to float32 after each addition (models one
    f32 accumulator); the addition order is `order`. Products are formed in
    float64 and rounded once when added, so the ONLY rounding sites are the
    partial sums.
    """
    acc = (h64[:, order[0], None] * wr64[order[0], None, :]).astype(np.float32)
    for j in order[1:]:
        acc = (acc + (h64[:, j, None] * wr64[j, None, :]).astype(np.float32)).astype(np.float32)
    return acc.astype(np.float64)


def readout_pass(seed: int, target_logit: float, accumulate: bool) -> tuple[float, float]:
    """f64-pass logit effect max abs error (t vs aligned arm), no logit cast.

    accumulate=False is the exact-f64 readout control (P3). With accumulate
    the reduction over channels is f32-accumulated, in canonical order for
    the treatment arm and in the aligned permutation order for the aligned
    arm.
    """
    x, deltas, w_t, w_a, w_read, perm = toy._build(seed, target_logit=target_logit)
    x64 = x.astype(np.float64)
    d64 = deltas.astype(np.float64)
    wr64 = w_read.astype(np.float64)
    identity = np.arange(wr64.shape[0])
    effects = []
    lmax = 0.0
    for name, w in (("t", w_t), ("a", w_a)):
        w64 = w.astype(np.float64)
        hb = x64 + np.tanh(x64 @ w64)
        hc = x64 + d64 + np.tanh((x64 + d64) @ w64)
        order = identity if name == "t" else perm
        if accumulate:
            lb = seq_f32_accumulate(hb, wr64, order)
            lc = seq_f32_accumulate(hc, wr64, order)
        else:
            lb = hb @ wr64
            lc = hc @ wr64
        effects.append(lc - lb)
        lmax = max(lmax, float(np.max(np.abs(lb))))
    return float(np.max(np.abs(effects[0] - effects[1]))), lmax


def probe_cell(seed: int, mag: float) -> dict:
    """One (seed, magnitude) cell: M4 error + control + unchanged f32 pass."""
    err, lmax = readout_pass(seed, mag, accumulate=True)
    control, _ = readout_pass(seed, mag, accumulate=False)
    toyr = toy.run(seed=seed, target_logit=mag)
    err32 = toyr["f32"]["logit_effect_max_abs_error"]
    err_c1 = toyr["f64_logit32"]["logit_effect_max_abs_error"]
    ulp = float(np.spacing(np.float32(lmax)))
    per_ulp = err / ulp if err > 0 else 0.0
    rel = _rel_dist_pow2(per_ulp) if err > 0 else 0.0
    return {
        "seed": seed,
        "mag": float(mag),
        "err_f64": err,
        "err_f32": err32,
        "err_c1": err_c1,
        "control": control,
        "max_abs_logit": lmax,
        "ulp": ulp,
        "err_per_ulp": per_ulp,
        "rel_dist_pow2": rel,
        "dense": bool(err > 0 and rel > DENSE_THRESHOLD),
    }


def _rel_dist_pow2(q: float) -> float:
    """Relative distance of q > 0 to the nearest power-of-two value 2**k."""
    k = int(round(math.log2(q)))
    cands = (2.0 ** (k - 1), float(2.0 ** k), 2.0 ** (k + 1))
    return min(abs(q - c) / c for c in cands)


def probe(seeds: tuple[int, ...] = DEFAULT_SEEDS,
          magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDES,
          tol: float = CASE_TOL) -> dict:
    """Measure every (seed, magnitude) cell once; evaluate the predeclared reading."""
    if len(magnitudes) > 1 and float(magnitudes[1] - magnitudes[0]) <= 0:
        raise ValueError("magnitudes must be increasing")
    cases = frozen_cases()
    cells = []
    for mag in magnitudes:
        for seed in seeds:
            cells.append(probe_cell(seed, mag))
    nonzero = [c for c in cells if c["err_f64"] > 0]
    dense = [c for c in nonzero if c["dense"]]

    per_case = []
    for case in cases:
        fit64 = [(c["mag"], c["seed"]) for c in nonzero
                 if abs(c["err_f64"] - case["e64"]) <= tol * case["e64"]]
        joint = [(c["mag"], c["seed"]) for c in cells
                 if abs(c["err_f64"] - case["e64"]) <= tol * case["e64"]
                 and abs(c["err_f32"] - case["e32"]) <= tol * case["e32"]]
        best = min((abs(c["err_f64"] - case["e64"]) / case["e64"] for c in nonzero),
                   default=None)
        per_case.append({
            "case": case["case"],
            "e64": case["e64"],
            "e32": case["e32"],
            "n_fit64": len(fit64),
            "fit64_L": [min(m for m, _ in fit64), max(m for m, _ in fit64)] if fit64 else None,
            "n_joint": len(joint),
            "joint_L": [min(m for m, _ in joint), max(m for m, _ in joint)] if joint else None,
            "best_rel_dev_f64": best,
        })
    common64, common_joint = [], []
    for mag in magnitudes:
        c64 = [c for c in cells if c["mag"] == mag and c["err_f64"] > 0]
        if all(any(abs(c["err_f64"] - case["e64"]) <= tol * case["e64"] for c in c64)
               for case in cases):
            common64.append(float(mag))
        cj = [c for c in cells if c["mag"] == mag and c["err_f64"] > 0]
        if all(any(abs(c["err_f64"] - case["e64"]) <= tol * case["e64"]
                   and abs(c["err_f32"] - case["e32"]) <= tol * case["e32"]
                   for c in cj) for case in cases):
            common_joint.append(float(mag))
    windows = lambda arr: ({"feasible": bool(arr),
                            "min_L": min(arr) if arr else None,
                            "max_L": max(arr) if arr else None})
    unmatched_matched = [pc["case"] for pc in per_case
                         if pc["case"] in UNMATCHED_CASES and pc["n_fit64"]]
    p1 = bool(dense)
    p2 = bool(unmatched_matched)
    max_control = max((c["control"] for c in cells), default=0.0)
    p3 = max_control <= CONTROL_MAX
    return {
        "config": {"seeds": list(seeds), "magnitudes": list(magnitudes),
                   "tol": tol, "n_cells": len(cells)},
        "live": bool(p1 and p2 and p3),
        "P1_dense": {"n_nonzero_cells": len(nonzero),
                     "n_dense_cells": len(dense),
                     "dense_frac_of_nonzero": len(dense) / len(nonzero) if nonzero else None,
                     "dense_rel_dist_range": ([min(c["rel_dist_pow2"] for c in dense),
                                               max(c["rel_dist_pow2"] for c in dense)]
                                              if dense else None)},
        "P2_unmatched_f64_matched_cases": unmatched_matched,
        "P3_control_max": max_control,
        "per_case": per_case,
        "common_window_f64_only": windows(common64),
        "common_window_joint": windows(common_joint),
        "err_f64_range": ([min(c["err_f64"] for c in nonzero), max(c["err_f64"] for c in nonzero)]
                          if nonzero else None),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--logits", type=str, default=",".join(f"{l:g}" for l in DEFAULT_MAGNITUDES))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    mags = tuple(float(s) for s in args.logits.split(",") if s.strip())
    res = probe(seeds=seeds, magnitudes=mags)
    if args.json:
        print(json.dumps(res, indent=1, sort_keys=True))
        return
    print(f"predeclared: {res['config']['n_cells']} cells, seeds 0..{len(seeds)-1}, "
          f"L {min(mags):g}..{max(mags):g} step {mags[1]-mags[0]:g}, tol {res['config']['tol']}")
    p1, p2, p3 = res["P1_dense"], res["P2_unmatched_f64_matched_cases"], res["P3_control_max"]
    print(f"P1 dense mantissa: {p1['n_dense_cells']}/{p1['n_nonzero_cells']} nonzero cells dense"
          f" (rel dist {p1['dense_rel_dist_range']})")
    print(f"P2 unmatched matched f64-only {res['config']['tol']:.0%}: {p2}")
    print(f"P3 control max: {p3:.3e} (bound {CONTROL_MAX:g})")
    print(f"LIVE: {res['live']}")
    print(f"err_f64 range (nonzero): {res['err_f64_range']}")
    for pc in res["per_case"]:
        f64s = (f"cells {pc['n_fit64']}, L {pc['fit64_L'][0]:g}..{pc['fit64_L'][1]:g}"
                if pc["n_fit64"] else "NO f64 match")
        joints = (f"joint cells {pc['n_joint']}, L {pc['joint_L'][0]:g}..{pc['joint_L'][1]:g}"
                  if pc["n_joint"] else "no joint match")
        print(f"{pc['case']:>11}: e64={pc['e64']:.3e} best dev {pc['best_rel_dev_f64']:.3f} "
              f"{f64s} | {joints}")
    for label, key in (("f64-only", "common_window_f64_only"), ("joint", "common_window_joint")):
        w = res[key]
        if w["feasible"]:
            print(f"common window {label}: L {w['min_L']:g}..{w['max_L']:g}")
        else:
            print(f"common window {label}: EMPTY")


if __name__ == "__main__":
    main()
