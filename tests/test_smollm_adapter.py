"""Regression tests for the caller-owned SmolLM adapter boundary."""

from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    RUNTIME_AVAILABLE = True
except ModuleNotFoundError:
    RUNTIME_AVAILABLE = False


CUSTODY_ROOT = Path("/Users/shaanp/.codex/custody/ahcl-integrated-stack-v2/model/smollm-135m")


@unittest.skipUnless(
    CUSTODY_ROOT.is_dir() and RUNTIME_AVAILABLE,
    "caller-owned SmolLM custody bundle and pinned runtime are required",
)
class TestSmolLMAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from smollm_experiment import NumpySmolLM, SmolLMCheckpoint

        cls.checkpoint = SmolLMCheckpoint(CUSTODY_ROOT, "smollm-135m")
        cls.numpy_model = NumpySmolLM(cls.checkpoint)
        cls.tokenizer = AutoTokenizer.from_pretrained(CUSTODY_ROOT, local_files_only=True)
        cls.reference = AutoModelForCausalLM.from_pretrained(
            CUSTODY_ROOT,
            local_files_only=True,
            dtype=torch.float32,
        ).eval()

    def test_smollm_adapter_matches_local_reference(self) -> None:
        ids = self.tokenizer("The capital of France is", add_special_tokens=False)["input_ids"]
        actual = self.numpy_model.forward(ids)
        with torch.no_grad():
            expected = self.reference(input_ids=torch.tensor([ids])).logits[0, -1].numpy()
        relative_l2 = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        self.assertEqual(np.argmax(actual), np.argmax(expected))
        self.assertLess(relative_l2, 1e-4)

    def test_gated_mlp_permutation_preserves_forward_behavior(self) -> None:
        from smollm_reference_check import canonicalize_gated_mlp, infer_gated_mlp_permutation

        ids = self.tokenizer("Water freezes at", add_special_tokens=False)["input_ids"]
        layer = 14
        components = self.numpy_model.gated_mlp_components(layer)
        permutation = np.roll(np.arange(self.numpy_model.ffn_dim), 17)
        transformed = [
            components[0][:, permutation],
            components[1][:, permutation],
            components[2][permutation, :],
        ]
        self.assertTrue(np.allclose(
            self.numpy_model.forward(ids),
            self.numpy_model.forward(ids, permutation=permutation, permutation_layer=layer),
            rtol=1e-4,
            atol=1e-4,
        ))
        before, _ = canonicalize_gated_mlp(*components)
        after, _ = canonicalize_gated_mlp(*transformed)
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(before, after)))
        self.assertTrue(np.array_equal(
            infer_gated_mlp_permutation(components, transformed), permutation,
        ))

    def test_gqa_group_permutation_preserves_forward_behavior(self) -> None:
        from smollm_reference_check import canonicalize_gqa, infer_gqa_permutation

        ids = self.tokenizer("A triangle has three", add_special_tokens=False)["input_ids"]
        layer = 14
        components = self.numpy_model.gqa_components(layer)
        permutation = np.roll(np.arange(self.numpy_model.n_kv_head), 1)
        query_indexes, kv_indexes = self.numpy_model._group_indexes(permutation)
        transformed = [
            components[0][query_indexes, :],
            components[1][kv_indexes, :],
            components[2][kv_indexes, :],
            components[3][:, query_indexes],
        ]
        self.assertTrue(np.allclose(
            self.numpy_model.forward(ids),
            self.numpy_model.forward(
                ids,
                attention_permutation=permutation,
                attention_permutation_layer=layer,
            ),
            rtol=1e-4,
            atol=1e-4,
        ))
        before, _ = canonicalize_gqa(
            *components,
            self.numpy_model.head_dim,
            self.numpy_model.n_kv_head,
            self.numpy_model.query_heads_per_kv,
        )
        after, _ = canonicalize_gqa(
            *transformed,
            self.numpy_model.head_dim,
            self.numpy_model.n_kv_head,
            self.numpy_model.query_heads_per_kv,
        )
        self.assertTrue(all(np.array_equal(left, right) for left, right in zip(before, after)))
        self.assertTrue(np.array_equal(
            infer_gqa_permutation(
                components,
                transformed,
                self.numpy_model.head_dim,
                self.numpy_model.n_kv_head,
                self.numpy_model.query_heads_per_kv,
            ),
            permutation,
        ))

    def test_causal_atlas_covers_all_smollm_gqa_groups(self) -> None:
        from smollm_reference_check import atlas_groups

        self.assertEqual(atlas_groups(self.numpy_model), (0, 1, 2))
