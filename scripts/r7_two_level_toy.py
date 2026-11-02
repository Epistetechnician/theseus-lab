"""Two-level toy: can ONE mechanism reproduce both r7 floors? (objective 1, rsi/problem.md)

Frozen r7 reports show two logit-effect floors per case:
  * float64 pass: 3.7e-6..6.6e-6 (six cases, spread 1.8x) with module error 1e-13..4e-12.
  * float32 pass: 3.6e-5..4.8e-5 (six cases, flat within 1.4x while the boundary-output
    magnitude spans 70x), f64/f32 ratio 1/8..1/11.5.

Open question (ledger iteration 19): can a single mechanism produce BOTH levels?

Tested mechanism M (one idea, no tuned constants): the logit path is float32 in
both passes.
  * f64 pass: everything float64; the treatments' tiny module difference (relative
    ~ eps 1e-13 on the weights, mirroring r7's f64 module error) reaches the logits,
    which are then cast to float32 once (candidate C1). The t-vs-a error is a
    rounding-boundary artifact at the ulp scale of the logit magnitude.
  * f32 pass: the whole module + readout runs in float32, and the ONLY difference
    between the treatment and mapping-aligned arms is summation order in the
    permuted-channel float32 matmul (exactly what the real aligned arm has: a
    mathematically identical sum evaluated in permuted order). No noise is injected;
    the module error therefore emerges from float32 accumulation, at the
    ~sqrt(d) * 2^-24 relative scale.

Predeclared reading (stated before running): M reproduces the two-level structure
iff, at one common float64 logit magnitude, err_f64 lands within ~2x of the
observed f64 band (3.7e-6..6.6e-6) and err_f32 within ~2x of the observed f32 band
(3.6e-5..4.8e-5). This is a hypothesis probe; the real max |logit| is unrecorded in
any frozen r7 report and no model was run.
Run: python3 scripts/r7_two_level_toy.py [--seeds 0,1,2] [--targets 32,64,128,256] [--json]
"""

from __future__ import annotations

import argparse
import json

import numpy as np


F32_BAND = (3.6e-5, 4.8e-5)   # observed r7 float32 logit floor band (six cases)
F64_BAND = (3.7e-6, 6.6e-6)   # observed r7 float64 logit floor band (six cases)


def _build(seed: int, d: int = 64, vocab: int = 512, n: int = 256, target_logit: float = 100.0):
    rng = np.random.default_rng(seed)
    w_mod = rng.standard_normal((d, d)) / np.sqrt(d)
    x = rng.standard_normal((n, d)) * 3.0
    deltas = rng.standard_normal((n, d)) * 2.0
    w_read = rng.standard_normal((d, vocab))
    base_max = np.abs((x + np.tanh(x @ w_mod)) @ w_read).max()
    w_read *= target_logit / base_max
    eps = rng.standard_normal((d, d)) * 1e-13  # r7 f64 module-difference scale
    perm = rng.permutation(d)                  # fixed aligned-arm channel permutation
    return x, deltas, w_mod, w_mod + eps, w_read, perm


def run(seed: int = 0, target_logit: float = 100.0, d: int = 64, vocab: int = 512, n: int = 256) -> dict:
    x, deltas, w_t, w_a, w_read, perm = _build(seed, d=d, vocab=vocab, n=n, target_logit=target_logit)
    out: dict[str, dict[str, float]] = {}

    # f64 pass, no float32 anywhere (control: the floor must vanish here).
    def f64_control_arm(w):
        x64 = x.astype(np.float64); d64 = deltas.astype(np.float64)
        w64 = w.astype(np.float64); wr64 = w_read.astype(np.float64)
        lb = x64 + np.tanh(x64 @ w64)
        lc = x64 + d64 + np.tanh((x64 + d64) @ w64)
        return (lb @ wr64), (lc @ wr64)
    ct_lb, ct_lc = f64_control_arm(w_t)
    ca_lb, ca_lc = f64_control_arm(w_a)
    out["f64_control"] = {
        "logit_effect_max_abs_error": float(np.max(np.abs((ct_lc - ct_lb) - (ca_lc - ca_lb)))),
    }

    # f64 pass with the single float32 logit cast (candidate C1).
    effects, lmax64 = [], 0.0
    mods = []
    for w in (w_t, w_a):
        x64 = x.astype(np.float64); d64 = deltas.astype(np.float64)
        w64 = w.astype(np.float64); wr64 = w_read.astype(np.float64)
        lb = (x64 + np.tanh(x64 @ w64)) @ wr64
        lc = (x64 + d64 + np.tanh((x64 + d64) @ w64)) @ wr64
        mods.append(np.tanh((x64 + d64) @ w64))
        lb = lb.astype(np.float32).astype(np.float64)
        lc = lc.astype(np.float32).astype(np.float64)
        effects.append(lc - lb)
        lmax64 = max(lmax64, float(np.max(np.abs(lb))))
    out["f64_logit32"] = {
        "logit_effect_max_abs_error": float(np.max(np.abs(effects[0] - effects[1]))),
        "module_effect_max_abs_error": float(np.max(np.abs(mods[0] - mods[1]))),
        "max_abs_logit": lmax64,
        "ulp_of_max_logit": float(np.spacing(np.float32(lmax64))),
    }

    # f32 pass: full float32 arithmetic; the arm difference is ONLY summation order
    # in the permuted-channel matmul (mathematically identical sums).
    x32 = x.astype(np.float32)
    d32 = deltas.astype(np.float32)
    w32 = w_t.astype(np.float32)
    wr32 = w_read.astype(np.float32)

    def module32(xx, w):
        return np.tanh(xx @ w.astype(np.float32))

    eff32_c = []
    mods32 = []
    lmax32 = 0.0
    # treatment arm: canonical order. aligned arm: only the module input is
    # channel-permuted (mathematically identical sum in permuted order, output
    # back in canonical order), exactly as the real construction works: the
    # permuted input is consumed by the exact weight-row copy and the residual
    # path and readout stay in canonical channel order.
    eff32_c = []
    mods32 = []
    lmax32 = 0.0
    base_o_t = module32(x32, w32)
    chg_o_t = module32(x32 + d32, w32)
    base_o_a = module32(x32[:, perm], w32[perm, :])
    chg_o_a = module32((x32 + d32)[:, perm], w32[perm, :])
    mods32 = [chg_o_t, chg_o_a]
    for base_h, chg_h, base_o, chg_o in (
        (x32 + base_o_t, x32 + d32 + chg_o_t, base_o_t, chg_o_t),
        (x32 + base_o_a, x32 + d32 + chg_o_a, base_o_a, chg_o_a),
    ):
        lb = base_h @ wr32
        lc = chg_h @ wr32
        eff32_c.append((lc.astype(np.float64) - lb.astype(np.float64)))
        lmax32 = max(lmax32, float(np.max(np.abs(lb.astype(np.float64)))))
    err32 = float(np.max(np.abs(eff32_c[0] - eff32_c[1])))
    out["f32"] = {
        "logit_effect_max_abs_error": err32,
        "module_effect_max_abs_error": float(np.max(np.abs(mods32[0] - mods32[1]))),
        "max_abs_logit": lmax32,
        "ulp_of_max_logit": float(np.spacing(np.float32(lmax32))),
    }
    if out["f64_logit32"]["logit_effect_max_abs_error"] > 0:
        out["ratio_f32_over_f64"] = {
            "value": err32 / out["f64_logit32"]["logit_effect_max_abs_error"]
        }
    return out


def sweep(seeds: tuple[int, ...] = (0, 1, 2), targets: tuple[float, ...] = (32.0, 64.0, 128.0, 256.0)) -> list[dict]:
    rows = []
    for tl in targets:
        for seed in seeds:
            r = run(seed, target_logit=tl)
            rows.append({
                "target_logit": tl,
                "seed": seed,
                "err_f64": r["f64_logit32"]["logit_effect_max_abs_error"],
                "err_f32": r["f32"]["logit_effect_max_abs_error"],
                "ratio": r["f32"]["logit_effect_max_abs_error"] / r["f64_logit32"]["logit_effect_max_abs_error"],
                "max_abs_logit": r["f64_logit32"]["max_abs_logit"],
                "ulp": r["f64_logit32"]["ulp_of_max_logit"],
                "err_per_ulp_f64": r["f64_logit32"]["logit_effect_max_abs_error"] / r["f64_logit32"]["ulp_of_max_logit"],
                "err_per_logit_f32": r["f32"]["logit_effect_max_abs_error"] / r["f32"]["max_abs_logit"],
                "f64_control_err": r["f64_control"]["logit_effect_max_abs_error"],
                "module_err_f64": r["f64_logit32"]["module_effect_max_abs_error"],
                "module_err_f32": r["f32"]["module_effect_max_abs_error"],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--targets", type=str, default="32,64,128,256")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    targets = tuple(float(s) for s in args.targets.split(",") if s.strip())
    rows = sweep(seeds=seeds, targets=targets)
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print(f"observed r7 bands: f64 [{F64_BAND[0]:.2e}, {F64_BAND[1]:.2e}]  "
          f"f32 [{F32_BAND[0]:.2e}, {F32_BAND[1]:.2e}]  ratio ~1/8..1/11.5")
    print(f"{'target':>8}{'seed':>6}{'max|logit|':>12}{'ulp':>10}{'err_f64':>11}{'err/ulp':>9}"
          f"{'err_f32':>11}{'err/L':>10}{'ratio':>9}{'ctrl':>10}")
    for r in rows:
        print(f"{r['target_logit']:>8.0f}{r['seed']:>6}{r['max_abs_logit']:>12.1f}{r['ulp']:>10.2e}"
              f"{r['err_f64']:>11.3e}{r['err_per_ulp_f64']:>9.3f}"
              f"{r['err_f32']:>11.3e}{r['err_per_logit_f32']:>10.2e}{r['ratio']:>9.1f}{r['f64_control_err']:>10.1e}")


if __name__ == "__main__":
    main()
