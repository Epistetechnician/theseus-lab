from __future__ import annotations

import unittest

import torch
from transformers import Gemma2Config, Gemma2ForCausalLM

import transcoder_baseline_model as tm


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    c = Gemma2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
                     num_key_value_heads=1, head_dim=8, max_position_embeddings=64)
    m = Gemma2ForCausalLM(c).to(dtype).eval()
    with torch.no_grad():  # non-trivial norm gains so the (1 + w) handling is exercised
        for blk in m.model.layers:
            blk.pre_feedforward_layernorm.weight.normal_(0, 0.5)
            blk.post_feedforward_layernorm.weight.normal_(0, 0.5)
    return m


IDS = torch.randint(0, 64, (2, 12), generator=torch.Generator().manual_seed(3))


class GlueTests(unittest.TestCase):
    def test_identity_replacement_reproduces_logits_float64(self):
        m = _model(torch.float64)
        base = tm.logits(m, IDS)
        with tm.replace_mlp_block(m, 1, tm.original_block_fn(m, 1)):
            rep = tm.logits(m, IDS)
        self.assertLess(float((base - rep).abs().max()), 1e-10)

    def test_identity_replacement_reproduces_logits_float32(self):
        m = _model()
        base = tm.logits(m, IDS)
        with tm.replace_mlp_block(m, 0, tm.original_block_fn(m, 0)):
            rep = tm.logits(m, IDS)
        self.assertLess(float((base - rep).abs().max()), 1e-4)

    def test_capture_shapes_and_target_consistency(self):
        m = _model(torch.float64)
        x, y = tm.capture_mlp_io(m, 0, IDS)
        self.assertEqual(tuple(x.shape), (2, 12, 16))
        self.assertLess(float((tm.original_block_fn(m, 0)(x) - y).abs().max()), 1e-10)
        self.assertAlmostEqual(tm.variance_explained(y, y), 1.0, places=9)

    def test_input_is_ungained_unit_rms(self):
        m = _model(torch.float64)
        x, _ = tm.capture_mlp_io(m, 0, IDS)
        rms = x.pow(2).mean(-1).sqrt()
        self.assertLess(float((rms - 1.0).abs().max()), 1e-3)  # RMSNorm eps only

    def test_zero_replacement_changes_logits_and_delta_loss(self):
        m = _model()
        zero = lambda x: torch.zeros_like(x)
        r = tm.delta_lm_loss(m, IDS, 0, zero)
        self.assertNotEqual(r["delta"], 0.0)
        ident = tm.delta_lm_loss(m, IDS, 0, tm.original_block_fn(m, 0))
        self.assertAlmostEqual(ident["delta"], 0.0, places=5)

    def test_hooks_removed_after_context(self):
        m = _model()
        base = tm.logits(m, IDS)
        with tm.replace_mlp_block(m, 0, lambda x: torch.zeros_like(x)):
            pass
        self.assertTrue(torch.equal(base, tm.logits(m, IDS)))


if __name__ == "__main__":
    unittest.main()
