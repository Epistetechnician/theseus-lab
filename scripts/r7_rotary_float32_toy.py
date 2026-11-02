"""Toy test of candidate C2 for the r7 logit-effect floor (rsi/problem.md objective 1).

C2 (docs/r7-logit-floor-analysis.md): rotary cos/sin tables built in float32 and then
cast to float64 carry a fixed float32 rounding error. The error is shared by both arms
when both arms read the same table row for the same position, so it should cancel in
t - a. It can only leave a t-vs-a residue if the two arms read different table rows.

Two scenarios, both float64 otherwise:
  same_index : both arms read table row p for position p (the expected code path).
  perm_index : the mapping-aligned arm reads row sigma(p) (hypothetical; tests whether a
               permuted table lookup could carry the rounding error into t - a).

For each scenario we report:
  logit_effect_err : max |effect_t - effect_a|, where effect = logits(patched) - logits(base)
                     and logits use float32 tables (f32) vs exact float64 tables (f64).
                     This is the t-vs-a error attributable to the table precision.
  shift_err_*      : max |shift_t - shift_a| where shift = effect(f32 tables) - effect(f64 tables)
                     per arm; equivalent to the table-rounding effect entering t - a.

Hypothesis probe only. It does not identify the source of the real r7 floor.
Run: python3 scripts/r7_rotary_float32_toy.py [--seed 0] [--json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _tables(n: int, half: int, base: float = 10000.0):
    inv_freq = base ** (-np.arange(half) / half)
    ang = np.arange(n)[:, None] * inv_freq[None, :]
    return np.cos(ang), np.sin(ang)  # float64 exact


def _rot(x, idx, cos_t, sin_t):
    h = x.shape[1] // 2
    c = cos_t[idx]
    s = sin_t[idx]
    x1, x2 = x[:, :h], x[:, h:]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=1)


def _module(x, w):
    return np.tanh(x @ w)


def _build(seed: int, d: int = 64, vocab: int = 512, n: int = 256, target_logit: float = 25.0):
    rng = np.random.default_rng(seed)
    w_mod = rng.standard_normal((d, d)) / np.sqrt(d)
    x = rng.standard_normal((n, d)) * 3.0
    deltas = rng.standard_normal((n, d)) * 2.0
    w_read = rng.standard_normal((d, vocab))
    cos64, sin64 = _tables(n, d // 2)
    base = x + _module(_rot(x, np.arange(n), cos64, sin64), w_mod)
    w_read *= target_logit / np.abs(base @ w_read).max()
    perm = rng.permutation(n)
    return x, deltas, w_mod, w_read, perm


def _effect(x, deltas, w, w_read, idx, cos_t, sin_t):
    """Logit effect per position: logits(x + d + module(rot(x+d))) - logits(x + module(rot(x)))."""
    base_h = x + _module(_rot(x, idx, cos_t, sin_t), w)
    chg_in = x + deltas
    chg_h = chg_in + _module(_rot(chg_in, idx, cos_t, sin_t), w)
    return (chg_h - base_h) @ w_read


def run(seed: int = 0) -> dict:
    x, deltas, w_mod, w_read, perm = _build(seed)
    n, d = x.shape
    cos64, sin64 = _tables(n, d // 2)
    cos32 = cos64.astype(np.float32).astype(np.float64)
    sin32 = sin64.astype(np.float32).astype(np.float64)
    # tiny float64 module difference between arms (as in the base toy)
    rng = np.random.default_rng(seed + 1000)
    w_arm = w_mod + rng.standard_normal((d, d)) * 1e-13

    ident = np.arange(n)
    scenarios = {"same_index": (ident, ident), "perm_index": (ident, perm)}
    out: dict[str, dict[str, float]] = {}
    for name, (idx_t, idx_a) in scenarios.items():
        eff64_t = _effect(x, deltas, w_mod, w_read, idx_t, cos64, sin64)
        eff64_a = _effect(x, deltas, w_arm, w_read, idx_a, cos64, sin64)
        eff32_t = _effect(x, deltas, w_mod, w_read, idx_t, cos32, sin32)
        eff32_a = _effect(x, deltas, w_arm, w_read, idx_a, cos32, sin32)
        shift_t = eff32_t - eff64_t
        shift_a = eff32_a - eff64_a
        logits_max = float(np.max(np.abs(eff64_t)))
        out[name] = {
            "logit_effect_err_f32_tables": float(np.max(np.abs(eff32_t - eff32_a))),
            "logit_effect_err_f64_tables": float(np.max(np.abs(eff64_t - eff64_a))),
            "shift_err_t_vs_a": float(np.max(np.abs(shift_t - shift_a))),
            "table_rounding_max_abs_cos_sin": float(max(np.max(np.abs(cos32 - cos64)), np.max(np.abs(sin32 - sin64)))),
            "max_abs_effect": logits_max,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    res = run(args.seed)
    if args.json:
        print(json.dumps(res, indent=1))
        return
    print(f"seed={args.seed}")
    for k, v in res.items():
        print(k)
        for m, val in v.items():
            print(f"  {m:<34}{val:.3e}")


if __name__ == "__main__":
    main()
