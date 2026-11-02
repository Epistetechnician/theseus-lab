import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_logit_floor_toy as toy  # noqa: E402


class TestR7LogitFloorToy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = {seed: toy.run(seed) for seed in (0, 1, 2)}

    def test_deterministic(self):
        again = toy.run(0)
        self.assertEqual(again, self.res[0])

    def test_float64_control_has_no_floor(self):
        # float64 end to end: logit effect error stays at module scale (well below 1e-9)
        for seed, r in self.res.items():
            self.assertLess(r["f64"]["logit_effect_max_abs_error"], 1e-9, seed)

    def test_delta_float32_cast_cancels_between_arms(self):
        # both arms share the same rounded deltas, so the cast must not create a t-vs-a error
        for seed, r in self.res.items():
            self.assertLess(r["f64_delta32"]["logit_effect_max_abs_error"], 1e-9, seed)

    def test_float32_logit_step_is_ulp_scale_but_below_observed_floor(self):
        # a float32 rounding of logits gives an error of order a fraction of one ulp at |logit|~30
        # (~5e-7 here). This is one order below the r7 observed floor (3.7e-6..6.6e-6),
        # so the toy does not reproduce the r7 magnitude. Recorded as a finding, not a pass.
        for seed, r in self.res.items():
            e = r["f64_logit32"]["logit_effect_max_abs_error"]
            self.assertGreater(e, 1e-8, seed)
            self.assertLess(e, r["f64_logit32"]["float32_ulp_at_max_logit"], seed)


if __name__ == "__main__":
    unittest.main()
