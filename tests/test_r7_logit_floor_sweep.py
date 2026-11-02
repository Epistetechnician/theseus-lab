import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_logit_floor_sweep as sweep_mod  # noqa: E402


class TestR7LogitFloorSweep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = sweep_mod.sweep()

    def test_deterministic(self):
        again = sweep_mod.sweep()
        self.assertEqual(again, self.res)

    def test_float64_control_stays_floor_free_at_every_scale(self):
        for key, group in self.res.items():
            self.assertLess(group["max_f64_control_err"], 1e-9, key)

    def test_logit32_error_is_sub_ulp_lattice_valued(self):
        # the C1 error is a rounding-boundary artifact: it stays strictly below
        # one float32 ulp at the realized max |logit| in this sample count, at
        # every magnitude tested.
        for key, group in self.res.items():
            for cell in group["cells"]:
                self.assertLess(cell["err_per_ulp"], 1.0, (key, cell["seed"]))
                self.assertGreaterEqual(cell["err_per_ulp"], 0.0, (key, cell["seed"]))

    def test_sweep_reaches_the_observed_floor_order(self):
        # at |logit| ~ 200-500 the C1 toy error enters the same order as the
        # r7 observed floor (3.7e-6..6.6e-6); this is why the iteration-3
        # estimate (|logit|~30) was a magnitude artifact, not a refutation.
        errs = [g["max_err_over_seeds"] for g in self.res.values()
                if 100.0 <= g["target_logit"] <= 1000.0]
        self.assertTrue(errs)
        self.assertGreaterEqual(max(errs), 1e-6)
        self.assertLessEqual(max(errs), 1e-4)

    def test_scale_expected_error_grows_by_at_least_one_order(self):
        # across the full sweep (2 .. 1500) the reachable C1 error must span
        # at least one order of magnitude, i.e. magnitude was a real knob.
        errs = [g["max_err_over_seeds"] for g in self.res.values()]
        self.assertGreater(max(errs) / min(errs), 10.0)


if __name__ == "__main__":
    unittest.main()
