"""V3 capture-mask and inherited-exclusion contracts.

State slices: theseus-lab-sandbox-v1, theseus-v39-first-mlp-fit-factorial-v3,
global-experiment-assurance-lifecycle-v1.
"""
import unittest

import numpy as np
import torch

import theseus_v39_fit_factorial_v3 as pilot


class CaptureContracts(unittest.TestCase):
    def test_v39_mask_column_is_removed_without_changing_native_values(self):
        padded = np.zeros((96, 769), dtype=np.float32)
        padded[:5, :768] = np.arange(5 * 768, dtype=np.float32).reshape(5, 768)
        padded[:5, -1] = 1
        actual = pilot.unpad_capture(torch.from_numpy(padded))
        self.assertEqual(actual.shape, (5, 768))
        np.testing.assert_array_equal(actual, padded[:5, :768])

    def test_invalid_mask_or_nonzero_padding_is_rejected(self):
        padded = np.zeros((96, 769), dtype=np.float32)
        padded[:3, -1] = 1
        padded[4, -1] = 1
        with self.assertRaises(ValueError):
            pilot.unpad_capture(padded)
        padded[:3, -1] = 1
        padded[4, -1] = 0
        padded[5, 0] = 1
        with self.assertRaises(ValueError):
            pilot.unpad_capture(padded)

    def test_v2_prompt_identities_are_added_to_exclusions(self):
        seen_text, seen_tokens = set(), set()
        row = {"text_sha256": "b" * 64, "ids": [4, 5, 6]}
        pilot.add_encoded_hashes([row], seen_text, seen_tokens)
        self.assertEqual(seen_text, {"b" * 64})
        self.assertEqual(seen_tokens, {pilot.engine.c.digest([4, 5, 6])})

    def test_v3_uses_distinct_protocol_and_data_stream(self):
        self.assertEqual(pilot.PROTOCOL, "theseus-v39-first-mlp-fit-factorial-v3")
        self.assertEqual(pilot.engine.TRAINING_ENTROPY, "factorial-v3-adaptation")
        self.assertIn("v3", pilot.engine.BASE.name)


if __name__ == "__main__":
    unittest.main()
