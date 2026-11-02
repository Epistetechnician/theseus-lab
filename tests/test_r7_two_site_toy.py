"""Tests for the two-float32-site probe (`scripts/r7_two_site_toy.py`).

Iteration 22 left the open question "can a second float32 rounding site
(e.g. the hidden state) at a different ulp scale fill the err/ulp mantissa
gaps of the three unmatched r7 cases?". The probe measures that variant
directly on a predeclared (seed, magnitude) grid. The handwritten checks here
assert: the frozen case pairs are read unchanged, the probe is deterministic
and matches its predeclared structure, the hidden-state cast QUENCHES the
single-site C1 floor exactly (named cells), where it does flip the resulting
err/ulp stays power-of-two-valued, and no probed cell falls inside the +-25%
window of any of the three unmatched cases. No model is run and no weights
are loaded.
"""

import json
import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_two_site_toy as probe_mod  # noqa: E402
import r7_two_level_toy as toy  # noqa: E402
import r7_joint_fit_scan as scan_mod  # noqa: E402


# e64 values frozen in docs/replacement-loss-v8-scale-r7-{NN256,NN512}-report.json
FROZEN_E64 = {
    "NN256-case0": 5.194386229590009e-06,
    "NN256-case1": 5.882346931329607e-06,
    "NN256-case2": 6.6190921677034e-06,
    "NN512-case0": 5.909945519633197e-06,
    "NN512-case1": 3.7179262317721395e-06,
    "NN512-case2": 4.934110389598345e-06,
}
UNMATCHED = probe_mod.UNMATCHED_CASES

SMALL_CONFIG = dict(seeds=(0, 1, 2, 3, 4, 5, 6, 7),
                    magnitudes=(64.0, 96.0, 128.0, 160.0, 192.0))

# Named probes chosen from the recorded 544-cell run:
#  quench pair: single-site C1 err > 0, hidden cast reduces it to exactly 0
QUENCH_CELL = (0, 64.0)
#  flip cells: the hidden site produces 2 arm-dependent f32 flips here
FLIP_CELLS = ((6, 160.0), (6, 176.0))


class FrozenCasesTest(unittest.TestCase):
    def test_probe_reads_the_six_frozen_cases_unchanged(self):
        cases = scan_mod.frozen_cases()
        self.assertEqual({c["case"] for c in cases}, set(FROZEN_E64))
        for c in cases:
            self.assertEqual(c["e64"], FROZEN_E64[c["case"]], c["case"])


class StructureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = probe_mod.probe(**SMALL_CONFIG)

    def test_deterministic(self):
        again = probe_mod.probe(**SMALL_CONFIG)
        self.assertEqual(json.dumps(again, sort_keys=True),
                         json.dumps(self.res, sort_keys=True))

    def test_structure_matches_the_predeclared_config(self):
        self.assertEqual(self.res["n_cells"],
                         len(SMALL_CONFIG["seeds"]) * len(SMALL_CONFIG["magnitudes"]))
        t = self.res["two_site"]
        # every nonzero two-site cell is carried over from a nonzero single cell
        self.assertEqual(t["n_nonzero_while_single_zero"], 0)
        self.assertEqual(t["n_quench_of_single_nonzero"] + t["n_nonzero"],
                         self.res["single_site"]["n_nonzero"])
        self.assertEqual(sum(t["hidden_flip_histogram"].values()),
                         self.res["n_cells"])
        self.assertEqual(t["n_matched_unmatched_case"], 0)

    def test_nonvacuous(self):
        # the quenching claim needs single-site errors to exist and to vanish
        self.assertEqual(self.res["single_site"]["n_nonzero"], self.res["n_cells"])
        self.assertGreater(self.res["two_site"]["n_nonzero"], 0)
        self.assertGreater(self.res["two_site"]["n_quench_of_single_nonzero"], 0)

    def test_f64_control_stays_floor_free(self):
        self.assertLess(self.res["f64_control_max"], 1e-8)


class QuenchTest(unittest.TestCase):
    def test_hidden_cast_erases_the_c1_floor_exactly(self):
        seed, mag = QUENCH_CELL
        cell = probe_mod.measure_cell(seed, mag)
        self.assertGreater(cell["err_single"], 0.0, (seed, mag))
        self.assertEqual(cell["err_two_site"], 0.0, (seed, mag))
        self.assertEqual(cell["hidden_flips"], 0, (seed, mag))

    def test_flip_cells_retain_errors_on_the_power_of_two_lattice(self):
        for seed, mag in FLIP_CELLS:
            cell = probe_mod.measure_cell(seed, mag)
            self.assertGreater(cell["err_single"], 0.0, (seed, mag))
            self.assertGreater(cell["err_two_site"], 0.0, (seed, mag))
            self.assertGreater(cell["hidden_flips"], 0, (seed, mag))
            t = toy.run(seed=seed, target_logit=mag)["f64_logit32"]
            q = cell["err_two_site"] / t["ulp_of_max_logit"]
            self.assertEqual(math.frexp(q)[0], 0.5, (seed, mag, q))


class UnmatchedUnreachabilityTest(unittest.TestCase):
    def test_no_probed_cell_falls_inside_the_unmatched_window(self):
        for seed in SMALL_CONFIG["seeds"]:
            for mag in SMALL_CONFIG["magnitudes"]:
                cell = probe_mod.measure_cell(seed, mag)
                for case in scan_mod.frozen_cases():
                    if case["case"] not in UNMATCHED:
                        continue
                    e64 = FROZEN_E64[case["case"]]
                    self.assertGreater(
                        abs(cell["err_two_site"] - e64), 0.25 * e64,
                        (seed, mag, case["case"]))


if __name__ == "__main__":
    unittest.main()
