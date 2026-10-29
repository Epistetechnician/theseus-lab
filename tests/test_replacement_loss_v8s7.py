from __future__ import annotations

import json
import unittest
from pathlib import Path

import replacement_loss_v8s6_runner as r6
import replacement_loss_v8s7_runner as runner


def _case(f32, f64, rel=1e-7):
    return {
        "float32": {"module_effect_max_abs_error": f32, "logit_effect_max_abs_error": 0.0},
        "float64": {"module_effect_max_abs_error": f64, "logit_effect_max_abs_error": 0.0},
        "relative_fp32_module_error": rel,
    }


class ReplacementLossV8ScaleR7Tests(unittest.TestCase):
    def test_pinned_chain_includes_every_r6_file_and_matches_the_freeze(self) -> None:
        root = Path(runner.__file__).resolve().parent
        r6_files = {row["path"] for row in json.loads((root / "docs/replacement-loss-v8s6-freeze.json").read_text())["packet_files"]}
        own = {
            "docs/replacement-loss-v8-scale-r7.md",
            "replacement_loss_v8s7_runner.py",
            "tests/test_replacement_loss_v8s7.py",
        }
        chain = set(runner.V8S7_PACKET_FILES)
        self.assertEqual(len(runner.V8S7_PACKET_FILES), len(chain))
        self.assertEqual(len(r6_files), 54)
        self.assertTrue(r6_files <= chain)
        self.assertEqual(chain, r6_files | own | {"docs/replacement-loss-v8s6-freeze.json"})
        self.assertEqual(len(chain), 58)
        freeze_path = root / runner.FREEZE_PATH
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            self.assertEqual([r["path"] for r in frozen["packet_files"]], list(runner.V8S7_PACKET_FILES))
            self.assertEqual(len(frozen["packet_files"]), 58)
            for row in frozen["packet_files"]:
                self.assertEqual(r6._sha(root / row["path"]), row["sha256"])
            self.assertTrue(runner.verify_packet()["verified"])

    def test_frozen_constants_are_unchanged(self) -> None:
        self.assertEqual(runner.THRESHOLD, 1e-4)
        self.assertEqual(runner.R6_ARM_OUTCOMES, {"NN256": "FAIL", "NN512": "FAIL"})

    def test_supported_when_float64_is_within_bound_and_100x_smaller(self) -> None:
        out = runner.classify([_case(6e-4, 1e-9), _case(1e-6, 1e-12)])
        self.assertEqual(out["outcome"], "SUPPORTED_FLOAT32_NOISE")
        self.assertEqual(out["r6_arm_outcomes_unchanged"], {"NN256": "FAIL", "NN512": "FAIL"})

    def test_rejected_when_float64_still_exceeds_bound(self) -> None:
        self.assertEqual(runner.classify([_case(6e-4, 3e-4)])["outcome"], "REJECTED")

    def test_inconclusive_when_misses_do_not_reproduce(self) -> None:
        self.assertEqual(runner.classify([_case(1e-6, 1e-12)])["outcome"], "INCONCLUSIVE_NOT_REPRODUCED")

    def test_partial_when_bound_met_but_reduction_under_100x(self) -> None:
        self.assertEqual(runner.classify([_case(6e-4, 9e-5)])["outcome"], "PARTIAL")

    def test_scale_claim_is_separate(self) -> None:
        self.assertEqual(runner.classify([_case(6e-4, 1e-9, rel=1e-3)])["P3_scale"], "SCALE_NOT_CONSISTENT")

    def test_spearman(self) -> None:
        self.assertAlmostEqual(runner.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertIsNone(runner.spearman([1, 2], [1, 2]))


if __name__ == "__main__":
    unittest.main()
