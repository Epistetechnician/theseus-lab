"""Regression tests for the fresh-probe SmolLM2 adapter boundary."""

from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


CUSTODY_ROOT = Path("/Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm2-135m")


@unittest.skipUnless(
    CUSTODY_ROOT.is_dir() and RUNTIME_AVAILABLE,
    "caller-owned SmolLM2 custody bundle and pinned runtime are required",
)
class TestSmolLM2Adapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from smollm_experiment import NumpySmolLM, SmolLMCheckpoint

        cls.checkpoint = SmolLMCheckpoint(CUSTODY_ROOT, "smollm2-135m")
        cls.numpy_model = NumpySmolLM(cls.checkpoint)
        cls.tokenizer = AutoTokenizer.from_pretrained(CUSTODY_ROOT, local_files_only=True)
        cls.reference = AutoModelForCausalLM.from_pretrained(
            CUSTODY_ROOT,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()

    def test_smollm2_adapter_matches_local_reference_on_fresh_probe(self) -> None:
        from smollm2_reference_check import FRESH_VISIBLE_PROMPTS

        ids = self.tokenizer(FRESH_VISIBLE_PROMPTS[0], add_special_tokens=False)["input_ids"]
        actual = self.numpy_model.forward(ids)
        with torch.no_grad():
            expected = self.reference(input_ids=torch.tensor([ids])).logits[0, -1].numpy()
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertEqual(np.argmax(actual), np.argmax(expected))
        self.assertLess(relative_l2, 1e-4)

    def test_smollm2_uses_a_distinct_fresh_probe_contract(self) -> None:
        from gpt2_experiment import HELDOUT_PROMPTS, VISIBLE_PROMPTS, full_digest
        from smollm2_reference_check import FRESH_HELDOUT_PROMPTS, FRESH_VISIBLE_PROMPTS

        self.assertEqual(len(FRESH_VISIBLE_PROMPTS), 8)
        self.assertEqual(len(FRESH_HELDOUT_PROMPTS), 64)
        self.assertNotEqual(
            full_digest({"visible": FRESH_VISIBLE_PROMPTS, "heldout": FRESH_HELDOUT_PROMPTS}),
            full_digest({"visible": VISIBLE_PROMPTS, "heldout": HELDOUT_PROMPTS}),
        )

    def test_smollm2_gated_mlp_permutation_preserves_behavior(self) -> None:
        ids = self.tokenizer("The color of the daytime sky is", add_special_tokens=False)["input_ids"]
        permutation = np.roll(np.arange(self.numpy_model.ffn_dim), 17)
        self.assertTrue(np.allclose(
            self.numpy_model.forward(ids),
            self.numpy_model.forward(ids, permutation=permutation, permutation_layer=14),
            rtol=1e-4,
            atol=1e-4,
        ))
