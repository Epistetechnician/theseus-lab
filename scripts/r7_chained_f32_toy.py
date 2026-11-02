"""Chained-f32 probe: a depth-faithful f32 leg at the L~8 regime (objective 1, rsi/problem.md).

Iteration 24 (`scripts/r7_summ_order_toy.py`) left the recorded open question:
"construct an f32-pass variant that reaches e32 ~4e-5 at the SAME L~8 regime
(the joint-empty-window measurement half); or probe whether the f64 and f32
legs in r7 use DIFFERENT effective aggregation scales, which the 1/8 ratio
would require." There, the 64-wide single-readout f32 pass erred 1.7e-6..2.4e-6
at L~8 (15-30x below the e32 band) while M4's f64-leg (f32-accumulated
readout, permuted order, no logit cast) matched all six e64 values.

Mechanism M5 (no tuned constants):
  * f64 leg: M4 unchanged, reused by importing `readout_pass` from
    `r7_summ_order_toy.py` (arm construction, eps ~1e-13 module arm-diff,
    sequential f32-accumulated readout in permuted aligned-arm order).
  * f32 leg: a CHAINED float32 pipeline - K sequential tanh-MLP layers of
    width d (both matmuls per layer, residuals omitted: bounded activations),
    the replaced module at chain layer 0 built exactly like the real
    construction (alignment = permuted module INPUT and weight rows, exact
    index copies; residual/readout canonical), and both arms recomputed
    INDEPENDENTLY through the whole chain (each arm's rerounding is its own
    realization), as in the real two-pass diagnostic. Depth K and width d
    come from the recorded energetics of the diagnosis (num_hidden_layers=24;
    hidden scaled down 2048->512 for runtime), NOT fitted to the match.

Predeclared reading (fixed BEFORE the recorded run):
  Q1 (per-case, f32 leg only): the chained f32 leg matches each frozen case's
      e32 at +-25% (CASE_TOL imported unchanged from r7_joint_fit_scan).
  Q2 (primary): a common JOINT window (both legs at the same (seed, L), all
      six cases, +-25%) EXISTS on the predeclared grid - the iteration-24
      empty joint window was possibly an artifact of the 64-wide single-
      matmul f32 leg. Non-existence is also a clean recorded result.
  Q3 (flatness analog): the chained leg's RELATIVE logit noise err32/L is
      near-magnitude-independent across the grid (max/min < 2.2x grid-round).
  Q4 (control): M4's exact-f64 readout control stays <= 1e-9 (P3 of iter 24).
  Q5 (lattice, secondary): at grid magnitudes where involved logits are small,
      whether chained e32 values sit on the exact 2^-21 lattice of the frozen
      e32 column is REPORTED, not asserted either way: the code-derived
      construction-level rule is that an f32-pass statistic t_log-a_log
      (f64 differences of f32-valued logits, `_measure` + `_next_logits` in
      `replacement_loss_v3_runner.py`) is an EXACT multiple of 2^-21 iff all
      involved |logit| >= 4, regardless of how large the logits are
      (any binade >= [4,8): ulp >= 2^-21, and every f32 value ~>=4 is a
      multiple of 2^-21; `2**-21 = 4.768e-7`). Magnitudes below 4 involve
      finer spacings (lattice law densifies downward).

Grid (predeclared): seeds 0..7, max|logit| L in {2, 4, 6, 8, 9, 10, 12, 14}
(64 cells). Relative noise rel ~ 4e-6 (pilot) puts the e32 band at L ~ 8-14.
Run: python3 scripts/r7_chained_f32_toy.py            (~1-2 min)
     python3 scripts/r7_chained_f32_toy.py --json
     python3 scripts/r7_chained_f32_toy.py --seeds 0 --logits 8 --json
No model is run, no torch is installed, no frozen file is modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r7_summ_order_toy as m4  # noqa: E402
from r7_joint_fit_scan import CASE_TOL, frozen_cases  # noqa: E402


DEFAULT_MAGNITUDES = (2.0, 4.0, 6.0, 8.0, 9.0, 10.0, 12.0, 14.0)
DEFAULT_SEEDS = tuple(range(8))
CHAIN_WIDTH = 512          # recorded hidden_size 2048, scaled for runtime
CHAIN_DEPTH = 24           # recorded num_hidden_layers
CHAIN_SAMPLES = 256        # as _build
CHAIN_VOCAB = 512
CONTROL_MAX = m4.CONTROL_MAX
TWO_NEG_21 = 2.0 ** -21


def build_chain(seed: int, d: int = CHAIN_WIDTH, K: int = CHAIN_DEPTH,
                n: int = CHAIN_SAMPLES, vocab: int = CHAIN_VOCAB,
                target_logit: float = 8.0):
    """Chain weights + inputs, sized so max|f32 chain logit| ~ target_logit."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32) * np.float32(3.0)
    deltas = rng.standard_normal((n, d)).astype(np.float32) * np.float32(2.0)
    w_final = rng.standard_normal((d, vocab)).astype(np.float32)
    w1s = [(rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
           for _ in range(K)]
    w2s = [(rng.standard_normal((d, d)) / np.sqrt(d)).astype(np.float32)
           for _ in range(K)]
    base = chain_forward(x, w1s, w2s, None, w_final)
    scale = float(np.abs(base.astype(np.float64)).max())
    w_final = (w_final.astype(np.float64) * (target_logit / scale)).astype(np.float32)
    perm = rng.permutation(d)
    return x, deltas, w1s, w2s, w_final, perm


def chain_forward(h: np.ndarray, w1s, w2s, perm, w_final) -> np.ndarray:
    """One f32 arm: replaced tanh-MLP module at chain layer 0 (permuted input
    and weight-row index copies, alignment identical to the recorded
    construction), remaining layers recomputed independently per arm."""
    for i, (w1, w2) in enumerate(zip(w1s, w2s)):
        if i == 0 and perm is not None:
            h = h[:, perm]
            w1 = w1[perm, :]
        h = h @ w1
        h = np.tanh(h)
        h = h @ w2
    return h @ w_final


def chained_pass(seed: int, target_logit: float) -> tuple[float, float]:
    """f32-leg (chained) logit effect max abs error (t vs aligned arm)."""
    x, deltas, w1s, w2s, w_final, perm = build_chain(seed, target_logit=target_logit)
    effects = []
    for name, perm_arm in (("t", None), ("a", perm)):
        base = chain_forward(x, w1s, w2s, perm_arm, w_final)
        chg = chain_forward(x + deltas, w1s, w2s, perm_arm, w_final)
        effects.append(chg.astype(np.float64) - base.astype(np.float64))
    err = float(np.max(np.abs(effects[0] - effects[1])))
    lmax = float(np.max(np.abs((x @ w_final).astype(np.float64))))
    return err, lmax


def probe_cell(seed: int, mag: float) -> dict:
    """One (seed, magnitude) cell: M5 joint measurement + controls."""
    err64, lmax = m4.readout_pass(seed, mag, accumulate=True)
    control, _ = m4.readout_pass(seed, mag, accumulate=False)
    err32, lmax32 = chained_pass(seed, mag)
    return {
        "seed": seed,
        "mag": float(mag),
        "err_f64": err64,
        "err_f32": err32,
        "control": control,
        "max_abs_logit_f64leg": lmax,
        "max_abs_logit_f32leg": lmax32,
        "rel_f32": err32 / mag if mag > 0 else None,
    }


def probe(seeds: tuple[int, ...] = DEFAULT_SEEDS,
          magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDES,
          tol: float = CASE_TOL) -> dict:
    """Measure every (seed, magnitude) cell once; evaluate the predeclared reading."""
    cases = frozen_cases()
    cells = []
    for mag in magnitudes:
        for seed in seeds:
            cells.append(probe_cell(seed, mag))

    def matches(c: dict, case: dict) -> bool:
        return (abs(c["err_f64"] - case["e64"]) <= tol * case["e64"]
                and abs(c["err_f32"] - case["e32"]) <= tol * case["e32"])

    def common(key_pred):
        out = []
        for mag in magnitudes:
            here = [c for c in cells if c["mag"] == mag]
            if all(any(key_pred(c, case) for c in here) for case in cases):
                out.append(float(mag))
        return out

    per_case = []
    for case in cases:
        fit64 = [(c["mag"], c["seed"]) for c in cells
                 if abs(c["err_f64"] - case["e64"]) <= tol * case["e64"]]
        fit32 = [(c["mag"], c["seed"]) for c in cells
                 if abs(c["err_f32"] - case["e32"]) <= tol * case["e32"]]
        joint = [(c["mag"], c["seed"]) for c in cells if matches(c, case)]
        per_case.append({
            "case": case["case"],
            "e64": case["e64"],
            "e32": case["e32"],
            "n_fit64": len(fit64),
            "fit64_L": [min(m for m, _ in fit64), max(m for m, _ in fit64)] if fit64 else None,
            "n_fit32": len(fit32),
            "fit32_L": [min(m for m, _ in fit32), max(m for m, _ in fit32)] if fit32 else None,
            "n_joint": len(joint),
            "joint_L": [min(m for m, _ in joint), max(m for m, _ in joint)] if joint else None,
            "best_rel_dev_f32": (min(abs(c["err_f32"] - case["e32"]) / case["e32"]
                                     for c in cells) if cells else None),
            "best_L_f32": (min(((abs(c["err_f32"] - case["e32"]) / case["e32"], c["mag"])
                                for c in cells))[1] if cells else None),
        })
    common_f64 = common(lambda c, case: abs(c["err_f64"] - case["e64"]) <= tol * case["e64"])
    common_f32 = common(lambda c, case: abs(c["err_f32"] - case["e32"]) <= tol * case["e32"])
    common_joint = common(matches)
    max_control = max((c["control"] for c in cells), default=0.0)
    rels = [c["rel_f32"] for c in cells if c["rel_f32"]]
    return {
        "config": {"seeds": list(seeds), "magnitudes": list(magnitudes),
                   "tol": tol, "n_cells": len(cells),
                   "chain": {"d": CHAIN_WIDTH, "K": CHAIN_DEPTH,
                             "n": CHAIN_SAMPLES, "vocab": CHAIN_VOCAB}},
        "per_case": per_case,
        "common_window_f64_only": _win(common_f64),
        "common_window_f32_only": _win(common_f32),
        "common_window_joint": _win(common_joint),
        "Q3_flatness": {"rel_min": min(rels), "rel_max": max(rels),
                        "ratio": max(rels) / min(rels) if rels else None},
        "Q4_control_max": max_control,
        "Q5_lattice_f32": _lattice_reading(cells),
        "err_f32_range": [min(c["err_f32"] for c in cells),
                          max(c["err_f32"] for c in cells)],
    }


def _win(arr: list[float]) -> dict:
    return {"feasible": bool(arr),
            "min_L": min(arr) if arr else None,
            "max_L": max(arr) if arr else None,
            "n_L": len(arr)}


def _lattice_reading(cells: list[dict]) -> dict:
    """Q5: exact 2^-21-lattice membership of chained e32 values, per magnitude."""
    per_mag = []
    for mag in sorted({c["mag"] for c in cells}):
        vals = [c["err_f32"] for c in cells if c["mag"] == mag and c["err_f32"] > 0]
        counts = [v / TWO_NEG_21 for v in vals]
        per_mag.append({
            "mag": float(mag),
            "n_cells_nonzero": len(vals),
            "integer_count_of_2e-21": sum(1.0 for q in counts if abs(q - round(q)) < 1e-9),
            "nearest_count_dev_range": ([min(abs(q - round(q)) for q in counts),
                                         max(abs(q - round(q)) for q in counts)]
                                        if counts else None),
        })
    return {"two_neg_21": TWO_NEG_21, "per_magnitude": per_mag}


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
    print(f"predeclared: {cfg['n_cells']} cells, seeds 0..{len(seeds)-1} "
          f"(d={cfg['chain']['d']}, K={cfg['chain']['K']}), "
          f"L {min(mags):g}..{max(mags):g} step varies, tol {cfg['tol']:.0%}")
    q3 = res["Q3_flatness"]
    print(f"Q3 chained-leg rel noise: {q3['rel_min']:.2e}..{q3['rel_max']:.2e} "
          f"(ratio {q3['ratio']:.2f})")
    print(f"Q4 M4 f64 control max: {res['Q4_control_max']:.3e} (bound {CONTROL_MAX:g})")
    print(f"err_f32 chained-leg range: "
          f"[{res['err_f32_range'][0]:.2e}, {res['err_f32_range'][1]:.2e}]")
    for pc in res["per_case"]:
        def fmt(n, rng):
            return (f"cells {n}, L {rng[0]:g}..{rng[1]:g}" if n and rng else "NO match")
        print(f"{pc['case']:>11}: e64={pc['e64']:.3e} e32={pc['e32']:.3e} | "
              f"f64 {fmt(pc['n_fit64'], pc['fit64_L'])} | "
              f"f32 {fmt(pc['n_fit32'], pc['fit32_L'])} best_dev "
              f"{pc['best_rel_dev_f32']:.3f} at L {pc['best_L_f32']:g} | "
              f"joint {fmt(pc['n_joint'], pc['joint_L'])}")
    for label, key in (("f64-only", "common_window_f64_only"),
                       ("f32-only", "common_window_f32_only"),
                       ("JOINT", "common_window_joint")):
        w = res[key]
        if w["feasible"]:
            print(f"common window {label}: {w['n_L']} grid mags, L "
                  f"{w['min_L']:g}..{w['max_L']:g}")
        else:
            print(f"common window {label}: EMPTY")
    print("Q5 lattice (chained e32 vs exact 2^-21 counts), per magnitude:")
    for row in res["Q5_lattice_f32"]["per_magnitude"]:
        lo, hi = row["nearest_count_dev_range"]
        print(f"  L={row['mag']:5g}: nonzero {row['n_cells_nonzero']}, "
              f"integer-count cells {row['integer_count_of_2e-21']:.0f}, "
              f"nearest-count dev {lo:.2e}..{hi:.2e}")


if __name__ == "__main__":
    main()
