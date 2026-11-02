"""Tests for the joint six-case r7 floor feasibility scan (`scripts/r7_joint_fit_scan.py`).

The scan checks whether the iteration-20 two-level mechanism (float32 logit
path) can jointly fit ALL SIX frozen r7 case pairs, not just the one cell
(seed 8, |logit|~160) demonstrated there. The handwritten case pairs are
hard-asserted against the frozen reports (read-only). No model is run and no
weights are loaded; the tests use a small grid to stay fast while exercising
the same code path as the recorded full-grid run.
"""

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_joint_fit_scan as scan_mod  # noqa: E402


# e64/e32 values frozen in docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json
FROZEN = {
    "NN256-case0": (5.194386229590009e-06, 4.76837158203125e-05),
    "NN256-case1": (5.882346931329607e-06, 4.38690185546875e-05),
    "NN256-case2": (6.6190921677034e-06, 4.00543212890625e-05),
    "NN512-case0": (5.909945519633197e-06, 4.291534423828125e-05),
    "NN512-case1": (3.7179262317721395e-06, 3.62396240234375e-05),
    "NN512-case2": (4.934110389598345e-06, 4.482269287109375e-05),
}

SMALL_GRID = (128.0, 160.0, 256.0)
SMALL_SEEDS = (8, 9, 17, 34)  # nonzero-k seeds; 8 is iteration 20's in-band cell


class FrozenCaseParseTest(unittest.TestCase):
    def test_six_cases_parsed_with_exact_frozen_values(self):
        cases = scan_mod.frozen_cases()
        self.assertEqual({c["case"] for c in cases}, set(FROZEN))
        for c in cases:
            e64, e32 = FROZEN[c["case"]]
            self.assertEqual(c["e64"], e64, c["case"])
            self.assertEqual(c["e32"], e32, c["case"])


class CellMatchTest(unittest.TestCase):
    def test_match_is_within_declared_relative_tolerance_both_levels(self):
        case = {"e64": 1.0, "e32": 2.0}
        self.assertTrue(scan_mod.cell_matches(
            {"err_f64": 1.20, "err_f32": 2.40}, case))
        self.assertFalse(scan_mod.cell_matches(
            {"err_f64": 1.20, "err_f32": 2.60}, case))
        self.assertFalse(scan_mod.cell_matches(
            {"err_f64": 1.30, "err_f32": 2.40}, case))


class ScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = scan_mod.scan(logits=SMALL_GRID, seeds=SMALL_SEEDS)

    def test_deterministic(self):
        again = scan_mod.scan(logits=SMALL_GRID, seeds=SMALL_SEEDS)
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(self.res, sort_keys=True))

    def test_result_is_json_serializable_with_both_tolerances(self):
        self.assertEqual(self.res["case_tol_primary"], 0.25)
        self.assertEqual(self.res["tols"], [0.25, 0.5])
        for tol, rt in self.res["per_tol"].items():
            self.assertEqual(len(rt["per_case"]), 6, tol)
            self.assertEqual(
                {pc["case"] for pc in rt["per_case"]}, set(FROZEN), tol)
        json.dumps(self.res, sort_keys=True)

    def test_iteration20_cell_matches_one_frozen_pair(self):
        # iteration 20's seed-8 cell at |logit|~160 sits in the collective
        # bands; at the case level it matches NN512-case1 (e64 +2.6%, e32
        # +5.3% relative), not NN256-case2 (-42% in e64).
        r = scan_mod.measure_cell(seed=8, logit_mag=160.0)
        case = {"e64": FROZEN["NN512-case1"][0], "e32": FROZEN["NN512-case1"][1]}
        self.assertTrue(scan_mod.cell_matches(r, case, 0.25))
        case_far = {"e64": FROZEN["NN256-case2"][0],
                    "e32": FROZEN["NN256-case2"][1]}
        self.assertFalse(scan_mod.cell_matches(r, case_far, 0.25))

    def test_more_tolerant_scan_never_matches_fewer_cases(self):
        n_primary = self.res["per_tol"][0.25]["n_cases_matched"]
        n_sens = self.res["per_tol"][0.5]["n_cases_matched"]
        self.assertGreaterEqual(n_sens, n_primary)

    def test_matched_cells_stay_on_the_declared_grid(self):
        for tol, rt in self.res["per_tol"].items():
            for pc in rt["per_case"]:
                if pc["n_cells"]:
                    self.assertGreaterEqual(pc["min_mag"], min(SMALL_GRID))
                    self.assertLessEqual(pc["max_mag"], max(SMALL_GRID))
                    for m in (pc["min_mag"], pc["max_mag"]):
                        self.assertIn(m, [float(g) for g in SMALL_GRID])


if __name__ == "__main__":
    unittest.main()
