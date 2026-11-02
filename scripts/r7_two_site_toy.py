"""Two-float32-site probe for the r7 f64 floor (objective 1, rsi/problem.md).

Iteration 22 (ledger, `scripts/r7_samplecount_probe.py`) measured that the
two-level toy's err_f64/ulp(max logit) is power-of-two-valued, so three of the
six frozen r7 cases (mantissa distances 26.6%, 29.1%, 29.7% from the nearest
power-of-two level) are unreachable at +-25% for ANY (seed, magnitude, n).
Its open question proposed the natural next variant: a SECOND float32 rounding
site (e.g. the hidden-state path) contributing at a different ulp scale, which
could fill those mantissa gaps.

This probe measures that variant directly (read-only over the frozen reports;
imports `_build` from the two-level toy so the arm construction cannot drift).
Mechanism variant M2 = C1 + one extra f32 cast of the hidden state h (per arm,
before the readout matmul). The arm differences are the recorded-scale f64
module differences (toy eps ~1e-13; r7 f64 module-effect 1e-13..4e-12), so the
hidden-state site can only contribute if it produces arm-DEPENDENT f32 flips.

Predeclared reading (fixed before the recorded run), measured per cell:
  * err_single: the single-site (C1) toy err_f64 at the same (seed, L) cell;
  * err_two_site: the M2 err_f64;
  * hidden_flips: positions where the f32 cast of the hidden state differs
    between the two arms (the only channel for a second contribution).

Decision rule: M2 is a live gap-filler iff some (seed, L) cell has
err_two_site inside the +-25% window of one of the three unmatched cases
(NN256-case0, NN256-case1, NN512-case0). M2 is falsified as a gap-filler AND
as an r7-compatible f64 pass if it QUENCHES the C1 f64 floor (err_two_site = 0
where err_single > 0) in most cells, because the six observed r7 f64 floors
(3.7e-6..6.6e-6) are all nonzero and no site downstream of a quenched hidden
cast can revive them. If the hidden site does flip, the resulting
err/ulp(max logit) must leave the single-site power-of-two lattice for the
mantissa gaps to be fillable; that is measured too.

No model is run, no torch is installed, no frozen file is modified. This
constrains, and does not confirm, C1: the real max |logit| and logit dtype
remain unrecorded in any frozen report.
Run: python3 scripts/r7_two_site_toy.py            (default probe, ~1 min)
     python3 scripts/r7_two_site_toy.py --json
     python3 scripts/r7_two_site_toy.py --seeds 0,1,2 --logits 128,160 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r7_two_level_toy as toy  # noqa: E402
from r7_joint_fit_scan import CASE_TOL, frozen_cases  # noqa: E402


DEFAULT_MAGNITUDES = tuple(float(l) for l in np.arange(64.0, 328.0, 16.0))
DEFAULT_SEEDS = tuple(range(32))
# The three cases iteration 22 found unreachable (mantissa distances 26.6-29.7%)
UNMATCHED_CASES = ("NN256-case0", "NN256-case1", "NN512-case0")


def two_site_f64_err(seed: int, target_logit: float, n: int = 256) -> dict:
    """M2 f64-logit pass: C1 plus one f32 cast of the hidden state, per arm."""
    x, deltas, w_t, w_a, w_read, _perm = toy._build(seed, n=n, target_logit=target_logit)
    effects = []
    hb_acc, hc_acc = [], []
    lmax64 = 0.0
    for w in (w_t, w_a):
        x64 = x.astype(np.float64)
        d64 = deltas.astype(np.float64)
        w64 = w.astype(np.float64)
        wr64 = w_read.astype(np.float64)
        hb = (x64 + np.tanh(x64 @ w64)).astype(np.float32)
        hc = (x64 + d64 + np.tanh((x64 + d64) @ w64)).astype(np.float32)
        hb_acc.append(hb)
        hc_acc.append(hc)
        lb = (hb.astype(np.float64) @ wr64).astype(np.float32).astype(np.float64)  # C1 site
        lc = (hc.astype(np.float64) @ wr64).astype(np.float32).astype(np.float64)  # C1 site
        effects.append(lc - lb)
        lmax64 = max(lmax64, float(np.max(np.abs(lb))))
    hidden_flips = int(np.sum(hb_acc[0] != hb_acc[1])) + int(np.sum(hc_acc[0] != hc_acc[1]))
    return {
        "err_f64": float(np.max(np.abs(effects[0] - effects[1]))),
        "max_abs_logit": lmax64,
        "hidden_flips": hidden_flips,
    }


def measure_cell(seed: int, logit_mag: float) -> dict:
    """One (seed, magnitude) cell: single-site toy vs the M2 variant."""
    single = toy.run(seed=seed, target_logit=logit_mag)
    two = two_site_f64_err(seed, logit_mag)
    return {
        "err_single": single["f64_logit32"]["logit_effect_max_abs_error"],
        "err_two_site": two["err_f64"],
        "hidden_flips": two["hidden_flips"],
        "f64_control": single["f64_control"]["logit_effect_max_abs_error"],
        "err_single_per_ulp": single["f64_logit32"]["logit_effect_max_abs_error"]
        / single["f64_logit32"]["ulp_of_max_logit"],
    }


def probe(seeds: tuple[int, ...] = DEFAULT_SEEDS,
          magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDES,
          tol: float = CASE_TOL) -> dict:
    """Measure every (seed, magnitude) cell once; the reading is predeclared."""
    cases = frozen_cases()
    unmatched = [c for c in cases if c["case"] in UNMATCHED_CASES]
    cells = {}
    n_single_nonzero = 0
    n_two_nonzero = 0
    n_quenched = 0      # single > 0 but two = 0: the hidden cast erased the floor
    two_pos_single_zero = 0
    single_unmatched_match_cells = 0
    two_matched_cases = set()
    flip_histogram = {}
    two_errs = []
    for mag in magnitudes:
        for seed in seeds:
            cell = measure_cell(seed, mag)
            cells[f"{seed},{mag:g}"] = cell
            flip_histogram[cell["hidden_flips"]] = \
                flip_histogram.get(cell["hidden_flips"], 0) + 1
            if cell["err_single"] > 0:
                n_single_nonzero += 1
                if cell["err_two_site"] == 0:
                    n_quenched += 1
            elif cell["err_two_site"] > 0:
                two_pos_single_zero += 1
            if cell["err_two_site"] > 0:
                n_two_nonzero += 1
                two_errs.append(cell["err_two_site"])
            for case in unmatched:
                if abs(cell["err_single"] - case["e64"]) <= tol * case["e64"]:
                    single_unmatched_match_cells += 1
            for case in cases:
                if abs(cell["err_two_site"] - case["e64"]) <= tol * case["e64"]:
                    two_matched_cases.add(case["case"])
    return {
        "config": {"seeds": list(seeds), "magnitudes": list(magnitudes), "tol": tol},
        "n_cells": len(cells),
        "single_site": {
            "n_nonzero": n_single_nonzero,
            "n_matched_unmatched_case": single_unmatched_match_cells,
        },
        "two_site": {
            "n_nonzero": n_two_nonzero,
            "n_quench_of_single_nonzero": n_quenched,
            "n_nonzero_while_single_zero": two_pos_single_zero,
            "matched_cases": sorted(two_matched_cases),
            "n_matched_unmatched_case": sum(1 for c in two_matched_cases
                                            if c in UNMATCHED_CASES),
            "err_min": min(two_errs) if two_errs else 0.0,
            "err_max": max(two_errs) if two_errs else 0.0,
            "hidden_flip_histogram": {str(k): v for k, v in sorted(flip_histogram.items())},
        },
        "f64_control_max": max(c["f64_control"] for c in cells.values()),
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
    cfg = res["config"]
    print(f"probe grid: seeds {cfg['seeds'][0]}..{cfg['seeds'][-1]}, "
          f"L {min(cfg['magnitudes']):g}..{max(cfg['magnitudes']):g} "
          f"({res['n_cells']} cells), tol {cfg['tol']}")
    s, t = res["single_site"], res["two_site"]
    print(f"single-site (C1):       nonzero errs {s['n_nonzero']}/{res['n_cells']}, "
          f"matches to the 3 unmatched cases within tol: "
          f"{s['n_matched_unmatched_case']} cells")
    print(f"two-site (M2 = C1 + hidden f32 cast):")
    print(f"  nonzero errs {t['n_nonzero']}/{res['n_cells']} "
          f"[{t['err_min']:.3e} .. {t['err_max']:.3e}]")
    print(f"  quenched (single>0 -> two=0): {t['n_quench_of_single_nonzero']} cells; "
          f"newly nonzero (single=0, two>0): {t['n_nonzero_while_single_zero']} cells")
    print(f"  hidden flips per cell: {t['hidden_flip_histogram']}")
    print(f"  cases matched at f64 side within tol: {t['matched_cases']} "
          f"(of the 3 unmatched: {t['n_matched_unmatched_case']})")
    print(f"f64 control max (no f32 site): {res['f64_control_max']:.2e}")


if __name__ == "__main__":
    main()
