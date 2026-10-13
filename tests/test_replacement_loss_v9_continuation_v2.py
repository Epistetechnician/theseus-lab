"""Tests for the v9 continuation-v2 activation-layout adapter.

State slice: theseus-lab-sandbox-v1.
"""

import unittest

import replacement_loss_v9_continuation_v2 as continuation
import replacement_loss_v9_runner as runner


class RestorePaddedLayoutTests(unittest.TestCase):
    def test_roundtrips_valid_tokens_and_keeps_padding_zero(self):
        flat = runner.torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mask = runner.torch.tensor([[1, 1, 0], [1, 0, 0]])

        padded = continuation._restore_padded_batch_layout(flat, mask)

        self.assertEqual(tuple(padded.shape), (2, 3, 2))
        self.assertTrue(runner.torch.equal(padded[mask.bool()], flat))
        self.assertTrue(runner.torch.equal(padded[0, 2], runner.torch.zeros(2)))
        self.assertTrue(runner.torch.equal(padded[1, 1], runner.torch.zeros(2)))

    def test_rejects_flattened_token_count_mismatch(self):
        flat = runner.torch.ones((2, 3))
        mask = runner.torch.tensor([[1, 0, 0]])

        with self.assertRaisesRegex(ValueError, "token count"):
            continuation._restore_padded_batch_layout(flat, mask)


if __name__ == "__main__":
    unittest.main()
