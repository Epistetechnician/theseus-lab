"""Tests for the r7 sample-count probe (`scripts/r7_samplecount_probe.py`).

Iteration 21 left the open prediction "more samples than the toy's 256 can
close the per-case ratio gaps". The probe measures that directly. The
handwritten checks here assert: the frozen case pairs are read unchanged, the
probe is deterministic, its structure matches its predeclared config, the
measured err/ulp statistic is power-of-two-valued (the basis of the
mantissa-distance criterion), and the criterion reproduces the
iteration-21 3/6 matched / 3/6 unmatched split at the predeclared ±25%
tolerance. No model is run and no weights are loaded.
"""

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_samplecount_probe as probe  # noqa: E402
import r7_joint_fit_scan as scan_mod  # noqa: E402


# e64/e32 values frozen in docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json
FROZEN_E64 = {
    "NN256-case0": 5.194386229590009e-06,
    "NN256-case1": 5.882346931329607e-06,
    "NN256-case2": 6.6190921677034e-06,
    "NN512-case0": 5.909945519633197e-06,
    "NN512-case1": 3.7179262317721395e-06,
    "NN512-case2": 4.934110389598345e-06,
}
MATCHABLE = {"NN256-case2", "NN512-case1", "NN512-case2"}

SMALL_CONFIG = dict(seeds=(0, 1), logits=(128.0, 160.0),
                    n_values=(256, 1024, 4096))


class FrozenCasesTest(unittest.TestCase):
    def test_probe_reads_the_six_frozen_cases_unchanged(self):
        cases = scan_mod.frozen_cases()
        self.assertEqual({c["case"] for c in cases}, set(FROZEN_E64))
        for c in cases:
            self.assertEqual(c["e64"], FROZEN_E64[c["case"]], c["case"])


class StructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = probe.run_probe(**SMALL_CONFIG)

    def test_deterministic(self):
        again = probe.run_probe(**SMALL_CONFIG)
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(self.res, sort_keys=True))

    def test_per_n_structure_matches_the_predeclared_config(self):
        self.assertEqual(list(self.res["per_n"]), list(SMALL_CONFIG["n_values"]))
        for n, entry in self.res["per_n"].items():
            self.assertEqual(entry["per_tol"][0.25]["n_cases_matched"],
                             len(entry["per_tol"][0.25]["matched_names"]))
            for tol in (0.25, 0.5):
                rt = entry["per_tol"][tol]
                self.assertEqual(len(rt["per_case"]), 6, (n, tol))
                for pc in rt["per_case"]:
                    self.assertEqual(pc["matched"], pc["n_cells"] > 0, pc["case"])
            w = entry["per_tol"][0.5]["common_window"]
            self.assertEqual(w["feasible"], w["n_mags"] > 0)

    def test_control_no_n_degrades_the_f64_calibration(self):
        # sample count must not move the f64 control out of its floor-free scale
        for n, entry in self.res["per_n"].items():
            self.assertLess(entry["cell_stats"]["f64_control_max"], 1e-8, n)


class PowerOfTwoLatticeTest(unittest.TestCase):
    CELLS = ((0, 160.0, 256), (3, 128.0, 1024), (7, 200.0, 4096),
             (12, 96.0, 1024), (101, 260.0, 4096))

    def test_err_f64_per_ulp_is_power_of_two_valued(self):
        for seed, mag, n in self.CELLS:
            cell = probe.measure_cell_n(seed, mag, n)
            q = cell["err_per_ulp"]
            if q == 0.0:
                continue
            mant, _ = math.frexp(q)
            self.assertEqual(mant, 0.5, (seed, mag, n, q))


class MantissaCriterionTest(unittest.TestCase):
    def test_rel_distance_to_nearest_power_of_two_reproduces_the_split(self):
        reached = {}
        for name, e64 in FROZEN_E64.items():
            lvl_lo = 2.0 ** math.floor(math.log2(e64))
            d = min((e64 - lvl_lo) / e64, (2.0 * lvl_lo - e64) / e64)
            reached[name] = d <= 0.25
        self.assertEqual({c for c, ok in reached.items() if ok}, MATCHABLE)

    def test_iteration20_cell_matches_its_case_at_n(self):
        case = next(c for c in probe.scan_mod.frozen_cases()
                    if c["case"] == "NN512-case1")
        self.assertTrue(scan_mod.cell_matches(
            probe.measure_cell_n(8, 160.0, 256), case, 0.25))


if __name__ == "__main__":
    unittest.main()
