"""Logit-magnitude sweep for the r7 logit-floor candidate C1 (objective 1, rsi/problem.md).

Iterates the toy in scripts/r7_logit_floor_toy.py (config f64_logit32: everything
float64, final logits rounded to float32 before the effect is formed) over
increasing logit magnitudes, and compares the resulting t-vs-a logit-effect
error to one float32 ulp at the realized max |logit|.

Purpose: the iteration-3 toy ran only at |logit| ~ 30 and gave ~4.8e-7, one
order below the r7 floor (3.7e-6..6.6e-6). This sweep answers whether that gap
was a magnitude artifact: C1 error is a rounding-boundary artifact whose size
depends on where the logit values sit on the float32 grid, so the ratio
err/ulp(max|logit|) and the required max |logit| are what matter.

This is a hypothesis probe. It does not establish the source in the real
checkpoint, and it does not record the real max |logit| (which no r7 report
contains).
Run: python3 scripts/r7_logit_floor_sweep.py [--json] [--scales 2,8,25]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r7_logit_floor_toy as toy  # noqa: E402


DEFAULT_SCALES = (2.0, 8.0, 25.0, 64.0, 200.0, 500.0, 1500.0)


def sweep(seeds: tuple[int, ...] = (0, 1, 2), scales: tuple[float, ...] = DEFAULT_SCALES) -> dict:
    out: dict[str, dict] = {}
    for tl in scales:
        cells = []
        for seed in seeds:
            r = toy.run(seed, target_logit=tl)
            cfg = r["f64_logit32"]
            ctrl = r["f64"]
            cells.append({
                "seed": seed,
                "logit32_effect_max_abs_error": cfg["logit_effect_max_abs_error"],
                "f64_control_effect_max_abs_error": ctrl["logit_effect_max_abs_error"],
                "max_abs_logit": cfg["max_abs_logit"],
                "float32_ulp_at_max_logit": cfg["float32_ulp_at_max_logit"],
                "err_per_ulp": cfg["logit_effect_max_abs_error"] / cfg["float32_ulp_at_max_logit"],
            })
        errs = [c["logit32_effect_max_abs_error"] for c in cells]
        ulps = [c["float32_ulp_at_max_logit"] for c in cells]
        out[f"scale_{tl:g}"] = {
            "target_logit": tl,
            "cells": cells,
            "max_err_over_seeds": max(errs),
            "max_f64_control_err": max(c["f64_control_effect_max_abs_error"] for c in cells),
            "ulp_at_max_logit": max(ulps),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--scales", type=str, default=",".join(str(s) for s in DEFAULT_SCALES))
    args = ap.parse_args()
    scales = tuple(float(s) for s in args.scales.split(",") if s.strip())
    res = sweep(scales=scales)
    if args.json:
        print(json.dumps(res, indent=1))
        return
    print(f"{'target_logit':>13}{'max|logit|':>11}{'fp32 ulp':>10}{'max err':>11}{'err/ulp':>9}  {''}f64 ctrl")
    for key, cell_group in res.items():
        big = max(cell_group["cells"], key=lambda c: c["logit32_effect_max_abs_error"])
        print(f"{cell_group['target_logit']:>13.0f}{big['max_abs_logit']:>11.1f}"
              f"{cell_group['ulp_at_max_logit']:>10.2e}{cell_group['max_err_over_seeds']:>11.3e}"
              f"{big['err_per_ulp']:>9.3f}  {cell_group['max_f64_control_err']:>10.2e}")


if __name__ == "__main__":
    main()
