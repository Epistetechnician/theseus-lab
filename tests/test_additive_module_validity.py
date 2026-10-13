"""Hermetic unit checks for the additive-module split adapters."""

import unittest
from pathlib import Path

import numpy as np
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    LlamaConfig,
    LlamaForCausalLM,
    MambaConfig,
    MambaForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    T5Config,
    T5ForConditionalGeneration,
)

from additive_module_validity import (
    CheckpointSpec,
    FAMILIES,
    HELDOUT_PROMPTS,
    VISIBLE_PROMPTS,
    _config_blockers,
    _capture_inputs,
    compare_reproductions,
    _forward_logits,
    _model_inputs,
    _numpy_split_output,
    _projection,
    _projection_layout,
    _relative_error,
    _run_split_path,
    _select_channel,
    _torch_split_output,
)


class _FixtureTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, return_tensors: str, add_special_tokens: bool = True) -> dict[str, torch.Tensor]:
        del return_tensors, add_special_tokens
        tokens = [1] + [3 + ord(character) % 50 for character in text[:4]]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }


class TestSplitImplementations(unittest.TestCase):
    def test_numpy_split_is_exact_in_float64(self) -> None:
        inputs = np.asarray([[1.25, -0.75, 0.5]], dtype=np.float64)
        weight = np.asarray([[0.5, -0.25, 1.0], [1.5, 0.25, -0.5]], dtype=np.float64)
        bias = np.asarray([0.25, -0.5], dtype=np.float64)
        baseline = inputs @ weight.T + bias
        result = _numpy_split_output(inputs, weight, bias, 1, 0.5, "out_in")
        self.assertLessEqual(_relative_error(baseline, result), 1e-12)

    def test_torch_split_matches_independent_numpy_path(self) -> None:
        inputs = torch.tensor([[0.125, -1.5, 2.0]], dtype=torch.float64)
        weight = torch.tensor([[0.5, -0.25, 1.0], [1.5, 0.25, -0.5]], dtype=torch.float64)
        bias = torch.tensor([0.25, -0.5], dtype=torch.float64)
        baseline = torch.nn.functional.linear(inputs, weight, bias)
        result = _torch_split_output(inputs, weight, bias, 1, 0.5, "out_in")
        reference = _numpy_split_output(inputs.numpy(), weight.numpy(), bias.numpy(), 1, 0.5, "out_in")
        self.assertLessEqual(_relative_error(result.numpy(), reference), 1e-12)

    def test_gpt2_conv1d_layout_splits_input_row(self) -> None:
        inputs = torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64)
        weight = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
        baseline = inputs @ weight
        result = _torch_split_output(inputs, weight, None, 2, 0.25, "in_out")
        self.assertLessEqual(_relative_error(baseline.numpy(), result.numpy()), 1e-12)


class TestArchitectureContracts(unittest.TestCase):
    def test_family_config_contracts_fail_closed(self) -> None:
        class Config:
            model_type = "opt"
            activation_function = "relu"

        self.assertEqual(_config_blockers(Config(), "opt_relu"), [])
        self.assertTrue(_config_blockers(Config(), "gpt2_gelu"))

    def test_all_five_projection_adapters_resolve_on_tiny_configs(self) -> None:
        cases = (
            (
                "opt_relu",
                OPTForCausalLM(OPTConfig(
                    vocab_size=64, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
                    num_attention_heads=4, word_embed_proj_dim=16, max_position_embeddings=32,
                    bos_token_id=1, eos_token_id=2, pad_token_id=0,
                )),
                "out_in",
            ),
            (
                "gpt2_gelu",
                GPT2LMHeadModel(GPT2Config(
                    vocab_size=64, n_embd=16, n_layer=1, n_head=4, n_positions=32,
                    n_ctx=32, bos_token_id=1, eos_token_id=2,
                )),
                "in_out",
            ),
            (
                "llama_silu_gated",
                LlamaForCausalLM(LlamaConfig(
                    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
                )),
                "out_in",
            ),
            (
                "encoder_decoder_t5",
                T5ForConditionalGeneration(T5Config(
                    vocab_size=64, d_model=16, d_ff=32, num_layers=1,
                    num_decoder_layers=1, num_heads=4,
                )),
                "out_in",
            ),
            (
                "state_space_mamba",
                MambaForCausalLM(MambaConfig(
                    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    state_size=4, time_step_rank=4, conv_kernel=3, use_mambapy=False,
                )),
                "out_in",
            ),
        )
        for family, model, expected_layout in cases:
            with self.subTest(family=family):
                self.assertEqual(_config_blockers(model.config, family), [])
                self.assertEqual(_projection_layout(_projection(model, family)), expected_layout)

    def test_all_five_tiny_models_preserve_heldout_logits_in_both_paths(self) -> None:
        cases = (
            (
                "opt_relu",
                OPTForCausalLM(OPTConfig(
                    vocab_size=64, hidden_size=16, ffn_dim=32, num_hidden_layers=1,
                    num_attention_heads=4, word_embed_proj_dim=16, max_position_embeddings=32,
                    bos_token_id=1, eos_token_id=2, pad_token_id=0,
                )),
            ),
            (
                "gpt2_gelu",
                GPT2LMHeadModel(GPT2Config(
                    vocab_size=64, n_embd=16, n_layer=1, n_head=4, n_positions=32,
                    n_ctx=32, bos_token_id=1, eos_token_id=2,
                )),
            ),
            (
                "llama_silu_gated",
                LlamaForCausalLM(LlamaConfig(
                    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32,
                )),
            ),
            (
                "encoder_decoder_t5",
                T5ForConditionalGeneration(T5Config(
                    vocab_size=64, d_model=16, d_ff=32, num_layers=1,
                    num_decoder_layers=1, num_heads=4, decoder_start_token_id=0, pad_token_id=0,
                )),
            ),
            (
                "state_space_mamba",
                MambaForCausalLM(MambaConfig(
                    vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    state_size=4, time_step_rank=4, conv_kernel=3, use_mambapy=False,
                )),
            ),
        )
        tokenizer = _FixtureTokenizer()
        for family, model in cases:
            with self.subTest(family=family):
                model.eval()
                projection = _projection(model, family)
                visible = _capture_inputs(model, tokenizer, family, projection, VISIBLE_PROMPTS)
                channel, _selection = _select_channel(visible)
                spec = CheckpointSpec(family, family, "0" * 40, Path("/external/custody/model"), True)
                baseline = [
                    _forward_logits(model, _model_inputs(tokenizer, model, family, prompt))
                    for prompt in HELDOUT_PROMPTS
                ]
                torch_split = _run_split_path(model, tokenizer, spec, projection, channel, "torch")
                numpy_split = _run_split_path(model, tokenizer, spec, projection, channel, "numpy")
                for before, torch_after, numpy_after in zip(baseline, torch_split, numpy_split):
                    self.assertLessEqual(_relative_error(before, torch_after), 1e-5)
                    self.assertLessEqual(_relative_error(before, numpy_after), 1e-5)
                    self.assertLessEqual(_relative_error(torch_after, numpy_after), 1e-5)


class TestReproductionGate(unittest.TestCase):
    def _run_receipt(self, runtime_identity: str) -> dict[str, object]:
        cases = []
        for family in FAMILIES:
            cases.append({
                "family": family,
                "checkpoint": {"model_id": family, "revision": "a" * 40, "checkpoint_tree_sha256": "b" * 64},
                "selection": {"selected_index": 3},
                "visible_fixture_sha256": "c" * 64,
                "heldout_fixture_sha256": "d" * 64,
                "tokenized_visible_fixture_sha256": "e" * 64,
                "tokenized_heldout_fixture_sha256": "f" * 64,
                "thresholds": {"local_relative_error_max": 1e-12},
                "metrics": {
                    "local_split_relative_error_max": 0.0,
                    "torch_full_model_relative_error_max": 1e-7,
                    "numpy_full_model_relative_error_max": 0.0,
                    "torch_vs_numpy_relative_error_max": 1e-7,
                },
                "heldout_output_digests": {"baseline": "a", "torch_split": "b", "numpy_split": "c"},
                "status": "PASS",
            })
        return {
            "state_slice": "theseus-lab-sandbox-v1",
            "protocol_id": "theseus-additive-module-validity-v1",
            "packet_sha256": "1" * 64,
            "source": {"revision": "2" * 40, "worktree_status_sha256": "3" * 64},
            "execution_id": runtime_identity,
            "runtime": {
                "runtime_identity_sha256": runtime_identity,
                "dependency_manifest_sha256": "d" * 64,
            },
            "assurance_level": "INTERNAL_PASS",
            "reports": cases,
        }

    def test_reproduction_requires_a_distinct_runtime(self) -> None:
        first = self._run_receipt("runtime-a")
        same_runtime = compare_reproductions(first, self._run_receipt("runtime-a"))
        self.assertEqual(same_runtime["assurance_level"], "INCONCLUSIVE")
        second = compare_reproductions(first, self._run_receipt("runtime-b"))
        self.assertEqual(second["assurance_level"], "REPRODUCED_LOCAL")

    def test_reproduction_requires_matching_dependency_manifests(self) -> None:
        first = self._run_receipt("runtime-a")
        second = self._run_receipt("runtime-b")
        second["runtime"]["dependency_manifest_sha256"] = "e" * 64
        comparison = compare_reproductions(first, second)
        self.assertEqual(comparison["assurance_level"], "INCONCLUSIVE")
        self.assertIn("dependency_manifest_mismatch", comparison["blockers"])


if __name__ == "__main__":
    unittest.main()
