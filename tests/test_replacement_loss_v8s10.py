from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

import replacement_loss_v8s8_runner as r8
import replacement_loss_v8s9_runner as r9
import replacement_loss_v8sb_runner as bandb
import replacement_loss_v8s10_runner as runner

ROOT = Path(runner.__file__).resolve().parent
EXPECTED_CHAIN_FILE_COUNT = 118


def _case(group, arm, family, seed, module=1e-13, logit=5e-6, cos=0.999):
    strata = {"compositional": cos, "ordinary": cos, "adversarial": 0.9999}
    return {
        "group": group, "arm": arm, "family": family, "seed_index": seed,
        "float64": {"module_effect_max_abs_error": module, "logit_effect_max_abs_error": logit, "strata_median_cosine": strata},
        "gate": runner.case_gate(module, logit, strata),
    }


def _all_cases(neg_cos=0.8):
    cases = []
    for arm in ("r6-NN256", "r6-NN512", "r9-NN256", "r9-NN512"):
        for fam in runner.FAMILIES:
            cases += [_case(runner.BAND_A, arm, fam, s) for s in runner.SEEDS]
    for fam in runner.FAMILIES:
        cases += [_case(runner.BAND_B_POS, "bandb-N512", fam, s, cos=0.998) for s in runner.SEEDS]
    cases += [_case(runner.BAND_B_NEG, "bandb-N256", "opt_relu", s, cos=neg_cos) for s in runner.SEEDS]
    return cases


R8REF = {s: {"compositional": 0.998, "ordinary": 0.998, "adversarial": 0.9999} for s in runner.SEEDS}


class ReplacementLossV8ScaleR10Tests(unittest.TestCase):
    def test_pinned_chain_covers_r5_to_r9_and_band_b_and_matches_the_freeze(self) -> None:
        chain = runner.V8S10_PACKET_FILES
        self.assertEqual(len(chain), len(set(chain)))
        expected = (
            set(runner.V8S10_OWN_FILES) | set(runner.PINNED_RESULT_FILES) | set(runner.PINNED_FREEZES.values())
            | set(r8.V8S8_PACKET_FILES) | set(r9.V8S9_PACKET_FILES)
        )
        self.assertEqual(set(chain), expected)
        for freeze in runner.PINNED_FREEZES.values():
            frozen = {r["path"] for r in json.loads((ROOT / freeze).read_text())["packet_files"]}
            self.assertTrue(frozen <= set(chain), freeze)
        # r8 and Band B result files are in the chain (r9's own chain omitted r8 files)
        for rel in (runner.R8_REPORT, "docs/replacement-loss-v8-scale-r8-results.md",
                    "docs/replacement-loss-v8-scale-bandb-N256-report.json", "docs/replacement-loss-v8-scale-bandb-N512-report.json"):
            self.assertIn(rel, chain)
        for rel in chain:
            self.assertTrue((ROOT / rel).is_file(), rel)
        self.assertIsNotNone(EXPECTED_CHAIN_FILE_COUNT)
        self.assertEqual(len(chain), EXPECTED_CHAIN_FILE_COUNT)
        freeze_path = ROOT / runner.FREEZE_PATH
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            self.assertEqual([r["path"] for r in frozen["packet_files"]], list(chain))
            self.assertEqual(len(frozen["packet_files"]), EXPECTED_CHAIN_FILE_COUNT)
            for row in frozen["packet_files"]:
                self.assertEqual(bandb._sha(ROOT / row["path"]), row["sha256"])
            self.assertTrue(runner.verify_packet()["verified"])

    def test_preregistered_constants_are_as_written(self) -> None:
        self.assertEqual(runner.MODULE_MAX_ABS, 1e-10)
        self.assertEqual(runner.LOGIT_MAX_ABS, 1e-4)
        self.assertEqual(runner.COSINE_MIN, 0.90)
        self.assertEqual(runner.FALSIFY_MODULE_ABS, 1e-8)
        self.assertEqual(runner.FALSIFY_MIN_BAND_A_FAILS, 2)
        self.assertEqual(runner.INCONCLUSIVE_BELOW_FRACTION, 0.60)
        self.assertEqual(runner.OPEN_LLAMA_COSINE_FLOOR, 0.99)
        self.assertEqual(runner.R8_REPRODUCTION_TOLERANCE, 0.005)
        self.assertEqual(runner.PLANNED_CASES, 36 + 9 + 3)

    def test_plan_matches_recorded_reports_and_negative_control_premise(self) -> None:
        for arm, spec in runner.ARMS.items():
            rep = json.loads((ROOT / spec["report"]).read_text())
            have = {(c["family"], c["seed_index"]) for c in rep["cases"]}
            for fam in spec["families"]:
                for s in runner.SEEDS:
                    self.assertIn((fam, s), have, (arm, fam, s))
        rep = json.loads((ROOT / runner.ARMS["bandb-N256"]["report"]).read_text())
        opt = [c for c in rep["cases"] if c["family"] == "opt_relu"]
        self.assertEqual(len(opt), 3)
        for c in opt:
            self.assertLess(min(v["logits"]["median_cosine"] for v in c["treatment_effects"]["per_stratum"].values()), 0.90)

    def test_case_gate_conditions(self) -> None:
        ok = {"a": 0.95, "b": 0.99}
        self.assertTrue(runner.case_gate(1e-11, 5e-6, ok)["passes"])
        self.assertFalse(runner.case_gate(2e-10, 5e-6, ok)["passes"])
        self.assertFalse(runner.case_gate(1e-11, 2e-4, ok)["passes"])
        self.assertFalse(runner.case_gate(1e-11, 5e-6, {"a": 0.95, "b": 0.89})["passes"])
        self.assertFalse(runner.case_gate(1e-11, 5e-6, {"a": None})["passes"])
        self.assertFalse(runner.case_gate(1e-11, 5e-6, {})["passes"])

    def test_float32_was_limiting(self) -> None:
        out = runner.classify(_all_cases(), R8REF)
        self.assertEqual(out["outcome"], "FLOAT32_WAS_LIMITING")
        self.assertTrue(out["P1_band_a_positives_pass_float64_gate"] and out["P2_band_b_positives_pass_float64_gate"])

    def test_partial_when_a_positive_fails_but_control_fails(self) -> None:
        cases = _all_cases()
        cases[0] = _case(runner.BAND_A, "r6-NN256", "opt_relu", 0, logit=2e-4)
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "PARTIAL")

    def test_falsified_when_negative_control_passes(self) -> None:
        self.assertEqual(runner.classify(_all_cases(neg_cos=0.95), R8REF)["outcome"], "FALSIFIED")

    def test_falsified_when_any_single_negative_case_passes(self) -> None:
        cases = _all_cases()
        cases[-1] = _case(runner.BAND_B_NEG, "bandb-N256", "opt_relu", 2, cos=0.95)
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "FALSIFIED")

    def test_falsified_when_two_band_a_fail_beyond_rounding(self) -> None:
        cases = _all_cases()
        cases[0] = _case(runner.BAND_A, "r6-NN256", "opt_relu", 0, module=5e-8)
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "PARTIAL")
        cases[1] = _case(runner.BAND_A, "r6-NN256", "opt_relu", 1, cos=0.85)
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "FALSIFIED")

    def test_module_error_between_1e10_and_1e8_is_partial_not_falsified(self) -> None:
        cases = _all_cases()
        for i in range(3):
            cases[i] = _case(runner.BAND_A, "r6-NN256", "opt_relu", i, module=1e-9)
        out = runner.classify(cases, R8REF)
        self.assertEqual(out["outcome"], "PARTIAL")
        self.assertFalse(out["P4_no_positive_module_error_above_1e-10"])

    def test_inconclusive_below_sixty_percent_runnable(self) -> None:
        cases = _all_cases()[:28]  # 28/48 < 0.60
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "INCONCLUSIVE")
        self.assertEqual(runner.classify(_all_cases()[:29], R8REF)["outcome"] != "INCONCLUSIVE", True)

    def test_open_llama_must_reproduce_r8_and_clear_the_floor(self) -> None:
        cases = _all_cases()
        for i, c in enumerate(cases):
            if c["group"] == runner.BAND_B_POS and c["family"] == "llama_silu_gated" and c["seed_index"] == 0:
                cases[i] = _case(runner.BAND_B_POS, "bandb-N512", "llama_silu_gated", 0, cos=0.97)
        out = runner.classify(cases, R8REF)
        self.assertEqual(out["outcome"], "PARTIAL")
        self.assertFalse(out["P2_band_b_positives_pass_float64_gate"])

    def test_not_run_cases_are_excluded_not_substituted(self) -> None:
        cases = [c for c in _all_cases() if c["arm"] != "r9-NN512"]  # 39/48 runnable
        out = runner.classify(cases, R8REF)
        self.assertEqual(out["runnable_cases"], 39)
        self.assertEqual(out["outcome"], "FLOAT32_WAS_LIMITING")

    def test_no_negative_control_cannot_be_limiting(self) -> None:
        cases = [c for c in _all_cases() if c["group"] != runner.BAND_B_NEG]
        self.assertEqual(runner.classify(cases, R8REF)["outcome"], "UNCLASSIFIED_NEGATIVE_CONTROL_NOT_EVALUABLE")

    def test_toy_llama_float64_pipeline_exact_cast_and_exact_permutation(self) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                          num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
        model = LlamaForCausalLM(cfg).float().eval()
        layer = 1
        down = model.model.layers[layer].mlp.down_proj
        treat = torch.nn.Linear(32, 16, bias=False)
        with torch.no_grad():
            treat.weight.copy_(down.weight + 1e-3 * torch.randn_like(down.weight))
        perm = list(np.random.default_rng(1).permutation(32))
        aligned = torch.nn.Linear(32, 16, bias=False)
        with torch.no_grad():
            aligned.weight.copy_(treat.weight[:, perm])
        modules = {"treatment": treat, "mapping_aligned_permutation": aligned}
        case = {"raw_index_permutation": [int(i) for i in perm]}
        rng = np.random.default_rng(0)
        rows = []
        for i in range(12):
            n = 6 + i % 3
            ids = rng.integers(1, 64, n + 1).tolist()
            rows.append({"ids": ids[:n], "prompt_len": n, "continuation_ids": ids[n:],
                         "stratum": ("compositional", "ordinary", "adversarial")[i % 3]})
        deltas = rng.normal(size=(12, 32)).astype(np.float32)
        r8.PAD["pad"] = 0
        before = {k: v.clone() for k, v in model.state_dict().items()}
        res = runner.diagnose_case(model, rows, deltas, modules, case, "llama_silu_gated", layer, "cpu")
        self.assertLess(res["float64"]["module_effect_max_abs_error"], runner.MODULE_MAX_ABS)
        self.assertLess(res["float64"]["logit_effect_max_abs_error"], runner.LOGIT_MAX_ABS)
        self.assertEqual(set(res["float64"]["strata_median_cosine"]), {"compositional", "ordinary", "adversarial"})
        self.assertEqual(model.model.layers[layer].mlp.down_proj.weight.dtype, torch.float32)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    def test_custody_check_reports_not_run_and_never_regenerates(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            receipt = runner.verify_custody(d, "bandb-N256")
        self.assertEqual(receipt["families"]["opt_relu"]["status"], "NOT_RUN")
        self.assertIn("custody", receipt["families"]["opt_relu"]["reason"])


if __name__ == "__main__":
    unittest.main()
