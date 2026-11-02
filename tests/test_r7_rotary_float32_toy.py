import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import r7_rotary_float32_toy as toy  # noqa: E402


class TestR7RotaryFloat32Toy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = {seed: toy.run(seed) for seed in (0, 1, 2)}

    def test_deterministic(self):
        self.assertEqual(toy.run(0), self.res[0])

    def test_table_rounding_is_float32_half_ulp_scale(self):
        # float32 rounding of values in [-1, 1] is at most 2**-25 ~ 3e-8 per entry
        for seed, r in self.res.items():
            e = r["same_index"]["table_rounding_max_abs_cos_sin"]
            self.assertGreater(e, 0.0, seed)
            self.assertLess(e, 6e-8, seed)

    def test_same_index_table_error_cancels_between_arms(self):
        # both arms read the same rows, so the f32 table error must not create a t-vs-a residue
        for seed, r in self.res.items():
            self.assertLess(r["same_index"]["shift_err_t_vs_a"], 1e-12, seed)

    def test_permuted_index_leaves_ulp_scale_residue_below_r7_floor(self):
        # a permuted lookup carries the rounding into t - a at ~3e-7..5e-7.
        # This is ~10x below the r7 observed floor (3.7e-6..6.6e-6): C2 is not confirmed.
        for seed, r in self.res.items():
            e = r["perm_index"]["shift_err_t_vs_a"]
            self.assertGreater(e, 1e-8, seed)
            self.assertLess(e, 3.7e-6, seed)

    def test_permuted_index_gross_mismatch_is_far_above_observed_floor(self):
        # a different row per arm also changes the positions themselves, which is far larger
        # than the r7 floor. A pipeline with this error would not show a 5e-6 floor.
        for seed, r in self.res.items():
            self.assertGreater(r["perm_index"]["logit_effect_err_f64_tables"], 1e-3, seed)


if __name__ == "__main__":
    unittest.main()
