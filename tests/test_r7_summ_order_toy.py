"""Tests for the summation-order probe (`scripts/r7_summ_order_toy.py`).

Iteration 23 left open candidate (b): "summation-order effects in the f32
pass entering the f64 comparison". The probe measures one realization of
that family directly: an f32-ACCUMULATED readout reduction (every partial
sum rounds to f32; no logit cast at all), with the aligned arm adding the
same term values in permuted channel order. The checks here assert: the
frozen case pairs are read unchanged, the match rule/tolerance has not
drifted, the sequential-f32-accumulation primitive actually rounds partial
sums (hand case) and is order-sensitive, the exact-f64 readout control stays
at the r7 module-effect scale, and the probe's predeclared LIVE fields exist
with the measured small-grid behavior. No model is run and no weights are
loaded.
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_summ_order_toy as probe_mod  # noqa: E402
import r7_two_level_toy as toy  # noqa: E402
import r7_joint_fit_scan as scan_mod  # noqa: E402


# Frozen values transcribed from docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json
FROZEN_E64 = {
    "NN256-case0": 5.194386229590009e-06,
    "NN256-case1": 5.882346931329607e-06,
    "NN256-case2": 6.6190921677034e-06,
    "NN512-case0": 5.909945519633197e-06,
    "NN512-case1": 3.7179262317721395e-06,
    "NN512-case2": 4.934110389598345e-06,
}
FROZEN_E32 = {
    "NN256-case0": 4.76837158203125e-05,
    "NN256-case1": 4.38690185546875e-05,
    "NN256-case2": 4.00543212890625e-05,
    "NN512-case0": 4.291534423828125e-05,
    "NN512-case1": 3.62396240234375e-05,
    "NN512-case2": 4.482269287109375e-05,
}
UNMATCHED = ("NN256-case0", "NN256-case1", "NN512-case0")


class FrozenCaseIntegrity(unittest.TestCase):
    def test_frozen_cases_unchanged(self):
        cases = scan_mod.frozen_cases()
        self.assertEqual(len(cases), 6)
        for case in cases:
            self.assertEqual(case["e64"], FROZEN_E64[case["case"]])
            self.assertEqual(case["e32"], FROZEN_E32[case["case"]])

    def test_shared_values_and_match_rule_not_loosened(self):
        self.assertEqual(probe_mod.UNMATCHED_CASES, UNMATCHED)
        self.assertEqual(probe_mod.CASE_TOL, 0.25)
        self.assertEqual(scan_mod.CASE_TOL, 0.25)
        case = {"e64": 1.0, "e32": 1.0}
        self.assertTrue(scan_mod.cell_matches({"err_f64": 1.0, "err_f32": 1.0}, case))
        self.assertFalse(scan_mod.cell_matches({"err_f64": 1.4, "err_f32": 1.0}, case))
        self.assertFalse(scan_mod.cell_matches({"err_f64": 1.0, "err_f32": 1.4}, case))


class AccumulationPrimitive(unittest.TestCase):
    def test_partial_sums_actually_round_to_f32(self):
        # h = ones(3 channels); terms 2**24, 1, 1: f32(2**24 + 1) rounds to
        # 2**24 (ties to even), so the sequential f32 sum is 2**24, while the
        # exact f64 sum is 2**24 + 2.
        h = np.ones((1, 3))
        terms = np.array([[float(2 ** 24)], [1.0], [1.0]])
        order = np.array([0, 1, 2])
        got = probe_mod.seq_f32_accumulate(h, terms, order)[0, 0]
        self.assertEqual(got, float(2 ** 24))
        self.assertNotEqual(got, float(2 ** 24) + 2.0)

    def test_addition_order_changes_the_accumulated_value(self):
        # terms 2**25, 1, 1, 1: canonical order loses all three ones (spacing 4),
        # reversed order sums 1+1+1=3 first and 2**25+3 rounds UP to 2**25+4.
        h = np.ones((1, 4))
        terms = np.array([[float(2 ** 25)], [1.0], [1.0], [1.0]])
        canonical = probe_mod.seq_f32_accumulate(h, terms, np.array([0, 1, 2, 3]))[0, 0]
        reversed_order = probe_mod.seq_f32_accumulate(h, terms, np.array([3, 2, 1, 0]))[0, 0]
        self.assertEqual(canonical, float(2 ** 25))
        self.assertEqual(reversed_order, float(2 ** 25) + 4.0)
        self.assertNotEqual(canonical, reversed_order)


class MantissaHelper(unittest.TestCase):
    def test_rel_dist_pow2(self):
        for q in (1.0, 0.5, 4.0, 0.125):
            self.assertEqual(probe_mod._rel_dist_pow2(q), 0.0)
        self.assertGreater(probe_mod._rel_dist_pow2(1.5), 0.02)
        self.assertAlmostEqual(probe_mod._rel_dist_pow2(1.5), 0.25)
        self.assertAlmostEqual(probe_mod._rel_dist_pow2(0.75), 0.25)


class ProbeBehavior(unittest.TestCase):
    def test_exact_f64_readout_control_stays_floor_free(self):
        for mag in (8.0, 128.0):
            control, _ = probe_mod.readout_pass(0, mag, accumulate=False)
            self.assertLessEqual(control, probe_mod.CONTROL_MAX)

    def test_readout_pass_deterministic(self):
        a = probe_mod.readout_pass(1, 8.0, accumulate=True)
        b = probe_mod.readout_pass(1, 8.0, accumulate=True)
        self.assertEqual(a, b)

    def test_single_magnitude_probe_is_allowed(self):
        # Regression: probe() used to index magnitudes[1] and crash on len==1.
        res = probe_mod.probe(seeds=(0,), magnitudes=(8.0,))
        self.assertEqual(res["config"]["n_cells"], 1)

    def test_probe_measured_small_grid_structure(self):
        res = probe_mod.probe(seeds=(0, 1, 2, 3), magnitudes=(8.0,))
        self.assertEqual(set(UNMATCHED) <= {pc["case"] for pc in res["per_case"]}, True)
        self.assertEqual(len(res["per_case"]), 6)
        cells_nonzero = res["P1_dense"]["n_nonzero_cells"]
        self.assertEqual(cells_nonzero, 4)
        # Measured on L=8, seeds 0..3: mantissas are dense, not power-of-two multiples.
        self.assertEqual(res["P1_dense"]["n_dense_cells"], 4)
        self.assertEqual(res["P1_dense"]["dense_frac_of_nonzero"], 1.0)
        # Measured: all four cells sit inside the +-25% window of >= 1 unmatched case.
        self.assertEqual(res["P2_unmatched_f64_matched_cases"], list(UNMATCHED))
        self.assertLessEqual(res["P3_control_max"], probe_mod.CONTROL_MAX)
        self.assertTrue(res["live"])
        # Measured at L=8: the unchanged f32 pass errs are ~15-30x below the e32 band,
        # so the both-sides joint window is empty even where the f64 side matches.
        self.assertFalse(res["common_window_joint"]["feasible"])
        self.assertTrue(res["common_window_f64_only"]["feasible"])
        for c in res["per_case"]:
            self.assertGreater(c["n_fit64"], 0)
            self.assertLessEqual(c["best_rel_dev_f64"], 0.25)


if __name__ == "__main__":
    unittest.main()
