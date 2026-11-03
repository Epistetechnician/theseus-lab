from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import torch

import replacement_loss_v8s7_runner as r7
import replacement_loss_v8s8_runner as runner
import replacement_loss_v8sb_runner as bandb

SEEDS = (0, 1, 2)


def _cells(c32=0.83, c64=0.995, bandb_value=0.83):
    return [
        {"seed_index": s, "stratum": st, "c32": c32, "c64": c64, "bandb": bandb_value}
        for s in SEEDS
        for st in runner.STRATA
    ]


class ReplacementLossV8ScaleR8Tests(unittest.TestCase):
    def test_pinned_chain_includes_bandb_and_r7_files_and_matches_the_freeze(self) -> None:
        root = Path(runner.__file__).resolve().parent
        bandb_files = {r["path"] for r in json.loads((root / "docs/replacement-loss-v8sb-freeze.json").read_text())["packet_files"]}
        r7_files = {r["path"] for r in json.loads((root / r7.FREEZE_PATH).read_text())["packet_files"]}
        self.assertEqual(len(bandb_files), 55)
        self.assertEqual(len(r7_files), 58)
        chain = set(runner.V8S8_PACKET_FILES)
        self.assertEqual(len(runner.V8S8_PACKET_FILES), len(chain))
        self.assertTrue(bandb_files <= chain)
        self.assertTrue(r7_files <= chain)
        self.assertEqual(chain, bandb_files | r7_files | set(runner.V8S8_OWN_FILES) | set(runner.PINNED_RESULT_FILES))
        self.assertEqual(len(chain), 75)
        for rel in chain:
            self.assertTrue((root / rel).is_file(), rel)
        freeze_path = root / runner.FREEZE_PATH
        if freeze_path.exists():
            frozen = json.loads(freeze_path.read_text())
            self.assertEqual([r["path"] for r in frozen["packet_files"]], list(runner.V8S8_PACKET_FILES))
            self.assertEqual(len(frozen["packet_files"]), 75)
            for row in frozen["packet_files"]:
                self.assertEqual(bandb._sha(root / row["path"]), row["sha256"])
            self.assertTrue(runner.verify_packet()["verified"])

    def test_frozen_constants_and_outcomes_are_unchanged(self) -> None:
        self.assertEqual(runner.GATE, 0.90)
        self.assertEqual(runner.H1_FLOOR, 0.99)
        self.assertEqual(runner.REPRODUCTION_TOLERANCE, 0.02)
        self.assertEqual(runner.BANDB_OUTCOMES["open_llama_7b_v2"], {"N256": "FAIL", "N512": "FAIL"})
        self.assertEqual(runner.BANDB_OUTCOMES["opt-6.7b"], {"N256": "FAIL", "N512": "PASS"})

    def test_recorded_bandb_values_support_the_premise(self) -> None:
        root = Path(runner.__file__).resolve().parent
        rep = json.loads((root / runner.BANDB_N512_REPORT).read_text())
        cases = [c for c in rep["cases"] if c["family"] == "llama_silu_gated"]
        self.assertEqual(sorted(c["seed_index"] for c in cases), [0, 1, 2])
        self.assertTrue(all(c["selected_layer"] == 7 for c in cases))
        comp = [c["treatment_effects"]["per_stratum"]["compositional"]["logits"]["median_cosine"] for c in cases]
        self.assertTrue(all(v < runner.GATE for v in comp))

    def test_h1_supported(self) -> None:
        self.assertEqual(runner.classify(_cells())["outcome"], "H1_SUPPORTED_ROUNDING_FLOOR")

    def test_h3_supported(self) -> None:
        self.assertEqual(runner.classify(_cells(c64=0.84))["outcome"], "H3_SUPPORTED_ARCHITECTURE_OR_FIT")

    def test_partial_rise_and_mixed(self) -> None:
        self.assertEqual(runner.classify(_cells(c64=0.95))["outcome"], "PARTIAL_RISE")
        cells = _cells()
        cells[0]["c64"] = 0.85
        self.assertEqual(runner.classify(cells)["outcome"], "MIXED")

    def test_adversarial_stratum_does_not_drive_the_outcome(self) -> None:
        cells = _cells()
        for c in cells:
            if c["stratum"] == "adversarial":
                c["c64"] = 0.5
        self.assertEqual(runner.classify(cells)["outcome"], "H1_SUPPORTED_ROUNDING_FLOOR")

    def test_validity_gate_blocks_attribution(self) -> None:
        out = runner.classify(_cells(c32=0.80, bandb_value=0.83))
        self.assertEqual(out["outcome"], "INCONCLUSIVE_NOT_REPRODUCED")
        self.assertEqual(out["bandb_outcomes_unchanged"]["arm"], {"N256": "FAIL", "N512": "FAIL"})

    def test_toy_llama_float64_pipeline_runs_and_is_exact_cast(self) -> None:
        from transformers import LlamaConfig, LlamaForCausalLM

        torch.manual_seed(0)
        cfg = LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                          num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
        model = LlamaForCausalLM(cfg).float().eval()
        layer = 1
        down = model.model.layers[layer].mlp.down_proj
        module = torch.nn.Linear(32, 16, bias=False)
        with torch.no_grad():
            module.weight.copy_(down.weight + 1e-3 * torch.randn_like(down.weight))
        rng = np.random.default_rng(0)
        rows = []
        for i in range(12):
            n = 6 + i % 3
            ids = rng.integers(1, 64, n + 1).tolist()
            rows.append({"ids": ids[:n], "prompt_len": n, "continuation_ids": ids[n:], "stratum": runner.STRATA[i % 3]})
        deltas = rng.normal(size=(12, 32)).astype(np.float32)
        runner.PAD["pad"] = 0
        before = {k: v.clone() for k, v in model.state_dict().items()}
        result = runner.diagnose_case(model, rows, deltas, module, "llama_silu_gated", layer, "cpu")
        for arm in ("float32", "float64"):
            self.assertEqual(result[arm]["logits"]["count"], 12)
            self.assertEqual(set(result[arm]["per_stratum"]), set(runner.STRATA))
        # fp32 weights are restored exactly (cast fp32 -> fp64 -> fp32 is lossless)
        self.assertEqual(model.model.layers[layer].mlp.down_proj.weight.dtype, torch.float32)
        for k, v in model.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)
        noise = result["scale_and_noise"]["compositional"]
        self.assertLess(noise["median_fp32_vs_fp64_noise_ratio_observed"], 1e-3)
        self.assertGreater(noise["median_cos_fp32_vs_fp64_observed"], 0.999)


if __name__ == "__main__":
    unittest.main()
