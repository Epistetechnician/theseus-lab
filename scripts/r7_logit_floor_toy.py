"""Toy reproduction of the r7 logit-effect floor (objective 1, rsi/problem.md).

Question: does a float32 step in the logit path produce a float64 logit-effect
error floor of the same order as the ~5e-6 seen in replacement-loss-v8-scale-r7,
while module-level float64 error is ~1e-13?

Structure mirrored from replacement_loss_v8s7_runner._predictions:
  * treatment arm and mapping-aligned arm compute the same boundary module up to a
    tiny float64 perturbation (module error ~1e-13);
  * a downstream readout maps the patched activation to next-token logits;
  * the logit effect is logits(patched) - logits(base), compared across arms.

Configs:
  f64          : everything float64 (control; expect logit error ~ module error).
  f64_logit32  : final logits rounded to float32 before the effect is formed.
  f64_delta32  : intervention deltas rounded to float32 once, identically for both arms
                 (mirrors the .astype(np.float32) on load in the runner).
  f32          : everything float32 (reference for the module-level float32 error).

This is a hypothesis probe. It does not establish the source in the real checkpoint.
Run: python3 scripts/r7_logit_floor_toy.py [--seed 0] [--json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


def _build(seed: int, d: int = 64, vocab: int = 512, n: int = 256, target_logit: float = 25.0):
    rng = np.random.default_rng(seed)
    w_mod = rng.standard_normal((d, d)) / np.sqrt(d)
    # boundary output scale ~ activation scale (r7 median boundary output L2 ~ 41)
    x = rng.standard_normal((n, d)) * 3.0
    deltas = rng.standard_normal((n, d)) * 2.0
    w_read = rng.standard_normal((d, vocab))
    # rescale the readout so the largest base logit is about target_logit
    base_max = np.abs((x + _module(x, w_mod)) @ w_read).max()
    w_read *= target_logit / base_max
    # tiny float64 module difference between arms (r7 float64 module error ~1e-13..4e-12)
    eps = rng.standard_normal((d, d)) * 1e-13
    return x, deltas, w_mod, w_mod + eps, w_read


def _module(x, w):
    return np.tanh(x @ w)


def _logits(h, w_read, dtype_logits):
    z = (h @ w_read)
    return z.astype(dtype_logits)


def run(seed: int = 0, target_logit: float = 25.0) -> dict:
    x, deltas, w_t, w_a, w_read = _build(seed, target_logit=target_logit)
    out: dict[str, dict[str, float]] = {}

    def effect(dtype, round_logits32: bool, round_delta32: bool, module_dtype):
        xs = x.astype(dtype)
        d = deltas.astype(np.float32).astype(dtype) if round_delta32 else deltas.astype(dtype)
        wr = w_read.astype(dtype)
        res = {}
        for name, w in (("t", w_t), ("a", w_a)):
            wm = w.astype(module_dtype)
            base_o = _module(xs.astype(module_dtype), wm).astype(dtype)
            chg_o = _module((xs + d).astype(module_dtype), wm).astype(dtype)
            # simple readout of the patched boundary, plus a residual path, to mimic next-token logits
            base_h = xs + base_o
            chg_h = xs + d + chg_o
            dt = np.float32 if round_logits32 else dtype
            lb = _logits(base_h, wr, dt).astype(np.float64)
            lc = _logits(chg_h, wr, dt).astype(np.float64)
            res[name] = {"effect": lc - lb, "logit_max_abs": float(np.max(np.abs(lc)))}
        err = float(np.max(np.abs(res["t"]["effect"] - res["a"]["effect"])))
        mod_err = float(np.max(np.abs(
            (_module((x + deltas).astype(dtype), w_t.astype(dtype)).astype(np.float64))
            - (_module((x + deltas).astype(dtype), w_a.astype(dtype)).astype(np.float64))
        )))
        return err, mod_err, max(res["t"]["logit_max_abs"], res["a"]["logit_max_abs"])

    for label, args in (
        ("f64", (np.float64, False, False, np.float64)),
        ("f64_logit32", (np.float64, True, False, np.float64)),
        ("f64_delta32", (np.float64, False, True, np.float64)),
        ("f32", (np.float32, False, False, np.float32)),
    ):
        err, mod_err, lmax = effect(*args)
        out[label] = {
            "logit_effect_max_abs_error": err,
            "module_effect_max_abs_error": mod_err,
            "max_abs_logit": lmax,
            "float32_ulp_at_max_logit": float(np.spacing(np.float32(lmax))),
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
    print(f"{'config':<13}{'logit_effect_err':>18}{'module_err':>14}{'max|logit|':>12}{'fp32 ulp':>11}")
    for k, v in res.items():
        print(f"{k:<13}{v['logit_effect_max_abs_error']:>18.3e}{v['module_effect_max_abs_error']:>14.3e}"
              f"{v['max_abs_logit']:>12.2f}{v['float32_ulp_at_max_logit']:>11.2e}")


if __name__ == "__main__":
    main()
