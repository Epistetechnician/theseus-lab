"""Tests for the two-level r7 floor toy (`scripts/r7_two_level_toy.py`).

The toy tests one mechanism (float32 logit path) against the two floors observed
in the frozen r7 reports: a float64-pass floor of 3.7e-6..6.6e-6 and a float32-pass
floor of 3.6e-5..4.8e-5 with f64/f32 ratio 1/8..1/11.5. The toys are hypothesis
probes; no model is run and no weights are loaded.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "scripts"))
import r7_two_level_toy as toy  # noqa: E402


class TwoLevelToyTest(unittest.TestCase):
    def test_deterministic(self):
        a = toy.run(seed=3, target_logit=64.0)
        b = toy.run(seed=3, target_logit=64.0)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_f64_control_is_floor_free(self):
        r = toy.run(seed=0, target_logit=128.0)
        self.assertLess(r["f64_control"]["logit_effect_max_abs_error"], 1e-9)

    def test_errors_lie_on_the_predicted_grids(self):
        r = toy.run(seed=0, target_logit=96.0)
        f64 = r["f64_logit32"]
        f32 = r["f32"]
        self.assertGreater(f64["logit_effect_max_abs_error"], 0.0)
        self.assertLessEqual(f64["logit_effect_max_abs_error"] / f64["ulp_of_max_logit"], 1.0)
        self.assertGreater(f32["logit_effect_max_abs_error"], 0.0)
        # f32 accumulation is relative to the logit magnitude (~1e-7 scale).
        rel = f32["logit_effect_max_abs_error"] / f32["max_abs_logit"]
        self.assertGreater(rel, 1e-8)
        self.assertLess(rel, 1e-6)

    def test_both_observed_bands_reachable_at_one_magnitude(self):
        """Core two-level claim of the toy: within a small magnitude band, some
        seeds put the float64-pass error inside the observed r7 float64 band
        (3.7e-6..6.6e-6) AND the float32-pass error inside the observed float32
        band (3.6e-5..4.8e-5) for the same run."""
        hits = []
        for seed in range(16):
            r = toy.run(seed=seed, target_logit=160.0)
            e64 = r["f64_logit32"]["logit_effect_max_abs_error"]
            e32 = r["f32"]["logit_effect_max_abs_error"]
            if (toy.F64_BAND[0] <= e64 <= toy.F64_BAND[1]
                    and toy.F32_BAND[0] <= e32 <= toy.F32_BAND[1]):
                hits.append((seed, e64, e32, e32 / e64))
        self.assertTrue(hits, "no simultaneous in-band cell found at |logit| ~ 160")
        ratios = [h[3] for h in hits]
        self.assertTrue(all(2.0 <= r <= 32.0 for r in ratios), f"ratios out of family: {ratios}")

    def test_module_error_scales_are_separated(self):
        r = toy.run(seed=0, target_logit=128.0)
        self.assertLess(r["f64_logit32"]["module_effect_max_abs_error"], 1e-11)
        self.assertGreater(r["f32"]["module_effect_max_abs_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
