from __future__ import annotations

import unittest

import numpy as np
import torch

import transcoder_baseline_core as core


def _params(d=6, f=12, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    return {"W_enc": r(d, f), "W_dec": r(f, d), "b_enc": r(f) * 0.1, "b_dec": r(d) * 0.1, "threshold": torch.rand(f, generator=g, dtype=dtype) * 0.5}


class CoreTests(unittest.TestCase):
    def test_jumprelu_matches_manual(self):
        p, x = _params(), torch.randn(5, 6, generator=torch.Generator().manual_seed(1))
        pre = x @ p["W_enc"] + p["b_enc"]
        manual = torch.where(pre > p["threshold"], pre, torch.zeros_like(pre)) @ p["W_dec"] + p["b_dec"]
        self.assertTrue(torch.allclose(core.jumprelu_transcoder(p, x), manual, atol=1e-6))

    def test_jumprelu_zeroes_below_threshold(self):
        p = _params()
        p["threshold"] = torch.full_like(p["threshold"], 1e9)
        out = core.jumprelu_transcoder(p, torch.randn(3, 6))
        self.assertTrue(torch.allclose(out, p["b_dec"].expand_as(out)))

    def test_jumprelu_missing_key(self):
        p = _params(); del p["threshold"]
        with self.assertRaises(KeyError):
            core.jumprelu_transcoder(p, torch.randn(2, 6))

    def test_float64_cast_is_exact_upcast(self):
        p, x = _params(), torch.randn(4, 6)
        out32 = core.jumprelu_transcoder(p, x)
        out64 = core.jumprelu_transcoder({k: v.double() for k, v in p.items()}, x.double())
        self.assertLess(float((out32.double() - out64).abs().max()), 1e-4)

    def test_ridge_recovers_affine_map(self):
        g = torch.Generator().manual_seed(2)
        x = torch.randn(400, 5, generator=g, dtype=torch.float64)
        w0, b0 = torch.randn(5, 3, generator=g, dtype=torch.float64), torch.randn(3, generator=g, dtype=torch.float64)
        w, b = core.ridge_affine(x, x @ w0 + b0)
        self.assertLess(float((w - w0).abs().max()), 1e-5)
        self.assertLess(float((b - b0).abs().max()), 1e-5)

    def test_cosine_and_eps(self):
        self.assertAlmostEqual(core.cosine([1, 0], [2, 0]), 1.0)
        self.assertTrue(np.isnan(core.cosine([0, 0], [1, 0])))
        self.assertAlmostEqual(core.effect_in_eps(np.array([core.FP32_EPS * 3.0]), 1.0), 3.0)

    def test_noise_floor_tiny_for_smooth_fn_and_deterministic(self):
        x = torch.randn(8, 6)
        f = lambda t: t @ torch.ones(6, 2)
        a, b = core.perturbation_noise_floor(f, x), core.perturbation_noise_floor(f, x)
        self.assertEqual(a, b)
        self.assertLess(a["relative"], 1e-5)

    def test_gate(self):
        self.assertTrue(core.case_gate({"a": 0.95, "b": 0.99})["passes"])
        self.assertFalse(core.case_gate({"a": 0.95, "b": 0.89})["passes"])
        self.assertFalse(core.case_gate({"a": float("nan")})["passes"])
        self.assertFalse(core.case_gate({})["passes"])

    def test_p1_p2(self):
        self.assertTrue(core.p1_transcoder_beats_affine([{"transcoder_min_cos": 0.99, "affine_min_cos": 0.95}]))
        self.assertFalse(core.p1_transcoder_beats_affine([]))
        self.assertTrue(core.p2_gap_follows_effect_size([{"effect_eps": 1e5, "gap": 0.001}, {"effect_eps": 10, "gap": 0.2}]))
        self.assertFalse(core.p2_gap_follows_effect_size([{"effect_eps": 1e5, "gap": 0.05}]))
        self.assertFalse(core.p2_gap_follows_effect_size([{"effect_eps": 1e3, "gap": 0.0}]))


if __name__ == "__main__":
    unittest.main()
