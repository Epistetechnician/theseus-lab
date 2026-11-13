from __future__ import annotations

import copy
import unittest
from unittest import mock

import torch

import transcoder_baseline_core as core
import transcoder_baseline_run as run
from tests.test_transcoder_baseline_model import _model


def _params(d=16, f=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)
    return {"W_enc": r(d, f) * 0.3, "W_dec": r(f, d) * 0.3, "b_enc": r(f) * 0.1, "b_dec": r(d) * 0.1, "threshold": torch.rand(f, generator=g) * 0.1}


def _ids(n, seed):
    return torch.randint(0, 64, (n, run.SEQ), generator=torch.Generator().manual_seed(seed))


class ConventionTests(unittest.TestCase):
    def test_convention_check(self):
        self.assertTrue(core.convention_check(0.6, {"a": -1.0, "b": 0.1}, 0.8)["passes"])
        self.assertFalse(core.convention_check(0.6, {"a": -1.0, "b": 0.7}, 0.8)["passes"])  # an alternative wins
        self.assertFalse(core.convention_check(0.6, {"a": -1.0}, 0.4)["passes"])  # cosine too low
        self.assertFalse(core.convention_check(0.6, {}, 0.9)["passes"])  # nothing to beat is not a pass
        self.assertFalse(core.convention_check(0.6, {"a": -1.0}, float("nan"))["passes"])

    def test_blend_keeps_biases_and_changes_weights(self):
        p = _params()
        q = core.blend_with_random(p)
        for k in ("b_enc", "b_dec", "threshold"):
            self.assertTrue(torch.equal(p[k], q[k]))
        self.assertFalse(torch.equal(p["W_enc"], q["W_enc"]))
        self.assertTrue(torch.equal(q["W_dec"], core.blend_with_random(p)["W_dec"]))  # deterministic


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.m32 = _model()
        self.m64 = copy.deepcopy(self.m32).double()
        self.fit, self.ev = _ids(6, 1), _ids(4, 2)

    def test_not_run_when_convention_fails(self):
        with mock.patch.object(core, "convention_check", return_value={"passes": False}):
            row = run.run_layer(self.m32, self.m64, 0, _params(), self.fit, self.ev)
        self.assertEqual(row["status"], "NOT_RUN: convention unverified")

    def test_machinery_runs_and_reports_all_fields(self):
        # convention gate bypassed (random params cannot pass it); checks structure only, not values
        with mock.patch.object(core, "convention_check", return_value={"passes": True}):
            row = run.run_layer(self.m32, self.m64, 1, _params(), self.fit, self.ev)
        self.assertEqual(row["status"], "RUN")
        for arm in ("transcoder", "affine", "control_blend"):
            for s in run.STRATA:
                self.assertIn("median_cos_float64", row["arms"][arm][s])
        for arm in ("transcoder", "affine"):
            self.assertIn("noise_floor", row["arms"][arm]["scale_1"])
        self.assertIn("passes", row["gate_float64"]["transcoder"])
        self.assertIn("delta", row["delta_lm_loss"]["affine"])
        summ = run.summarize({"1": row})
        self.assertEqual(summ["layers_run"], ["1"])

    def test_effect_of_original_block_matches_itself(self):
        orig = run.tm.original_block_fn(self.m64, 0)
        u = run._direction(16, 5)
        a = run.effect(self.m64, 0, self.ev[:1], orig, 3, 0.5 * u)
        b = run.effect(self.m64, 0, self.ev[:1], orig, 3, 0.5 * u)
        self.assertAlmostEqual(run._cos(a, b), 1.0, places=12)
        self.assertGreater(float(a.norm()), 0.0)


if __name__ == "__main__":
    unittest.main()
