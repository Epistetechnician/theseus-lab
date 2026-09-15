"""Regression tests for the caller-owned OPT adapter boundary."""

from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


CUSTODY_ROOT = Path("/Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/opt-125m")

@unittest.skipUnless(
    CUSTODY_ROOT.is_dir() and RUNTIME_AVAILABLE,
    "caller-owned OPT custody bundle and pinned runtime are required",
)
class TestOPTAdapter(unittest.TestCase):
    @staticmethod
    def _loaded():
        from opt_experiment import NumpyOPT, OPTCheckpoint

        checkpoint = OPTCheckpoint(CUSTODY_ROOT, "opt-125m")
        numpy_model = NumpyOPT(checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(CUSTODY_ROOT, local_files_only=True)
        reference = AutoModelForCausalLM.from_pretrained(
            CUSTODY_ROOT,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()
        return checkpoint, numpy_model, tokenizer, reference

    def test_opt_adapter_matches_local_reference(self) -> None:
        _checkpoint, numpy_model, tokenizer, reference = self._loaded()
        ids = tokenizer("The capital of France is", add_special_tokens=False)["input_ids"]
        actual = numpy_model.forward(ids)
        with torch.no_grad():
            expected = reference(input_ids=torch.tensor([ids])).logits[0, -1].numpy()
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertEqual(np.argmax(actual), np.argmax(expected))
        self.assertLess(relative_l2, 1e-4)

    def test_opt_channel_permutation_preserves_forward_behavior(self) -> None:
        _checkpoint, numpy_model, tokenizer, _reference = self._loaded()
        ids = tokenizer("Water freezes at", add_special_tokens=False)["input_ids"]
        permutation = np.roll(np.arange(numpy_model.ffn_dim), 17)
        baseline = numpy_model.forward(ids)
        transformed = numpy_model.forward(ids, permutation=permutation, permutation_layer=5)
        self.assertTrue(np.allclose(baseline, transformed, rtol=1e-4, atol=1e-4))

    def test_opt_activation_patch_is_a_noop_for_the_source_activation(self) -> None:
        _checkpoint, numpy_model, tokenizer, _reference = self._loaded()
        ids = tokenizer("Water freezes at", add_special_tokens=False)["input_ids"]
        layer = 5
        activation = numpy_model.mlp_activation_trace(ids)[layer]
        baseline = numpy_model.forward(ids)
        patched = numpy_model.forward(
            ids,
            patch_mlp_activation=activation,
            patch_mlp_activation_layer=layer,
        )
        self.assertTrue(np.array_equal(baseline, patched))

    def test_opt_activation_patch_rejects_wrong_width(self) -> None:
        _checkpoint, numpy_model, tokenizer, _reference = self._loaded()
        ids = tokenizer("Water freezes at", add_special_tokens=False)["input_ids"]
        with self.assertRaisesRegex(ValueError, "patched OPT activation"):
            numpy_model.forward(
                ids,
                patch_mlp_activation=np.zeros(numpy_model.ffn_dim - 1),
                patch_mlp_activation_layer=5,
            )
