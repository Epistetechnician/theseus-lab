"""Tests for the chained-f32 probe (`scripts/r7_chained_f32_toy.py`).

Iteration 24 left the recorded open item: an f32-pass variant reaching
e32 ~4e-5 at the SAME L~8 regime where the M4 f64 leg matches e64, i.e. the
joint-empty-window measurement half. The probe builds a depth-faithful
chained f32 leg (K=24 tanh-MLP layers, width 512 64-> recorded 2048 scaled
for runtime; no tuned constants) and reuses M4's f64 leg unchanged. The
checks here assert: the frozen case pairs are read unchanged and the match
rule has not drifted, the chained construction is deterministic and, in
exact f64, arm-equivalent up to summation order, the measured relative-noise
flatness (Q3), the M4 control bound (Q4), the measured joint-window cells
(Q2, values transcribed from the recorded run), and the measured exact
2^-21-lattice membership at large magnitudes (Q5). No model is run and no
weights are loaded.
"""

import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_chained_f32_toy as probe_mod  # noqa: E402
import r7_summ_order_toy as m4  # noqa: E402
import r7_joint_fit_scan as scan_mod  # noqa: E402


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
TWO_NEG_21 = 2.0 ** -21
# Measured cells from the recorded primary run (seeds 0..7, L in {2..14}):
# no single cell matches all six at once; the common joint-window convention
# (every case has SOME jointly-matching seed at that magnitude) holds at
# L in {8, 9}. Seed 0 at L=8 jointly matches five cases (all but
# NN512-case1, which is f32-only at that cell), and seed 2 jointly matches
# NN512-case1 at L=9.
SEED0_L8_JOINT = {"NN256-case0", "NN256-case1", "NN256-case2", "NN512-case0", "NN512-case2"}


class FrozenCaseIntegrity(unittest.TestCase):
    def test_frozen_cases_unchanged(self):
        cases = scan_mod.frozen_cases()
        self.assertEqual(len(cases), 6)
        for case in cases:
            self.assertEqual(case["e64"], FROZEN_E64[case["case"]])
            self.assertEqual(case["e32"], FROZEN_E32[case["case"]])

    def test_match_rule_tolerance_not_loosened(self):
        self.assertEqual(scan_mod.CASE_TOL, 0.25)
        case = {"e64": 1.0, "e32": 1.0}
        self.assertTrue(scan_mod.cell_matches({"err_f64": 1.0, "err_f32": 1.0}, case))
        self.assertFalse(scan_mod.cell_matches({"err_f64": 1.4, "err_f32": 1.0}, case))
        self.assertFalse(scan_mod.cell_matches({"err_f64": 1.0, "err_f32": 1.4}, case))

    def test_chain_constants_unfitted(self):
        self.assertEqual(probe_mod.CHAIN_DEPTH, 24)   # recorded num_hidden_layers
        self.assertEqual(probe_mod.CHAIN_WIDTH, 512)  # recorded 2048, scaled
        self.assertEqual(probe_mod.CONTROL_MAX, m4.CONTROL_MAX)


class ChainConstruction(unittest.TestCase):
    def test_build_and_pass_deterministic(self):
        a = probe_mod.chained_pass(3, 8.0)
        b = probe_mod.chained_pass(3, 8.0)
        self.assertEqual(a, b)
        x1, *rest1 = probe_mod.build_chain(5, target_logit=8.0)
        x2, *rest2 = probe_mod.build_chain(5, target_logit=8.0)
        self.assertTrue(np.array_equal(x1, x2))
        for r1, r2 in zip(rest1, rest2):
            self.assertTrue(np.array_equal(r1, r2))

    def test_permuted_arm_is_exact_f64_match(self):
        # In exact f64 the aligned arm computes the IDENTICAL sum; the arm
        # difference may only be f64 summation-order noise (<= 1e-9),
        # mirroring the recorded r7 f64 module-effect scale.
        xc, deltas, w1s, w2s, w_final, perm = probe_mod.build_chain(0, target_logit=8.0)
        xf = xc.astype(np.float64)
        wf = w_final.astype(np.float64)
        c_t = probe_mod.chain_forward(xf, [w.astype(np.float64) for w in w1s],
                                      [w.astype(np.float64) for w in w2s], None, wf)
        c_a = probe_mod.chain_forward(xf, [w.astype(np.float64) for w in w1s],
                                      [w.astype(np.float64) for w in w2s], perm, wf)
        self.assertLessEqual(float(np.max(np.abs(c_t - c_a))), 1e-9)

    def test_chain_forward_reaches_target_logit(self):
        xc, deltas, w1s, w2s, w_final, perm = probe_mod.build_chain(0, target_logit=12.0)
        base = probe_mod.chain_forward(xc, w1s, w2s, None, w_final)
        self.assertAlmostEqual(float(np.abs(base.astype(np.float64)).max()), 12.0, places=4)


class MeasuredBehavior(unittest.TestCase):
    def test_Q4_m4_control_stays_floor_free(self):
        for mag in (8.0, 14.0):
            control, _ = m4.readout_pass(0, mag, accumulate=False)
            self.assertLessEqual(control, probe_mod.CONTROL_MAX)

    def test_Q3_relative_noise_flat_across_magnitudes(self):
        rels = []
        for mag in (4.0, 8.0, 12.0, 14.0):
            err32, _ = probe_mod.chained_pass(0, mag)
            rels.append(err32 / mag)
        self.assertLess(max(rels) / min(rels), 2.2)

    def test_Q2_joint_window_membership_measured(self):
        cases = scan_mod.frozen_cases()
        c = probe_mod.probe_cell(0, 8.0)
        for case in cases:
            self.assertEqual(scan_mod.cell_matches(c, case),
                             case["case"] in SEED0_L8_JOINT,
                             f"seed 0 L=8 joint membership on {case['case']}")

    def test_Q5_lattice_membership_measured(self):
        # Large magnitudes: e32 lands exactly on the 2^-21 lattice (integer
        # counts), as the construction rule predicts when involved
        # |logit| >= 4. The (0, 32.0) cell measured half-integer because its
        # argmax element involved a [2,4)-binade logit (finer spacing 2^-22).
        for seed, mag, kind in ((0, 128.0, "on"), (6, 128.0, "on"),
                                (0, 256.0, "on"), (0, 32.0, "half")):
            err, _ = probe_mod.chained_pass(seed, mag)
            q = err / TWO_NEG_21
            if kind == "on":
                self.assertLess(abs(q - round(q)), 1e-6)
            else:
                self.assertLess(abs(abs(q - round(q)) - 0.5), 1e-6)


class ProbeAPI(unittest.TestCase):
    def test_probe_small_grid_shape_and_joint_window(self):
        # Seed 0 jointly matches all six cases at L=8 (and five at L=9);
        # NN512-case1's joint coverage at L=9 needs seeds 2 or 6.
        res = probe_mod.probe(seeds=(0, 2, 6), magnitudes=(8.0, 9.0))
        self.assertEqual(res["config"]["n_cells"], 6)
        self.assertEqual(len(res["per_case"]), 6)
        for pc in res["per_case"]:
            self.assertIn("n_joint", pc)
            self.assertIn("best_rel_dev_f32", pc)
        self.assertTrue(res["common_window_joint"]["feasible"])
        self.assertEqual(res["common_window_joint"]["min_L"], 8.0)
        self.assertEqual(res["common_window_joint"]["max_L"], 9.0)

    def test_lattice_reading_fields(self):
        res = probe_mod.probe(seeds=(0,), magnitudes=(32.0,))
        row = res["Q5_lattice_f32"]["per_magnitude"][0]
        self.assertEqual(row["mag"], 32.0)
        self.assertEqual(row["n_cells_nonzero"], 1)
        self.assertEqual(row["integer_count_of_2e-21"], 0.0)


if __name__ == "__main__":
    unittest.main()
