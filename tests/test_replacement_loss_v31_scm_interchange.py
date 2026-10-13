import unittest

import torch

import replacement_loss_v31_runner as v31


class V31SCMTests(unittest.TestCase):
    def test_task_support_and_disjoint_prefix_split(self):
        all_contexts = v31.contexts()
        fit = {item for item in all_contexts if v31.ALPHABET[item[0]] in v31.ADAPT_PREFIX}
        heldout = {item for item in all_contexts if v31.ALPHABET[item[0]] in v31.CHALLENGE_PREFIX}
        self.assertEqual(len(all_contexts), 256)
        self.assertEqual(len(fit), 128)
        self.assertEqual(len(heldout), 128)
        self.assertTrue(fit.isdisjoint(heldout))

    def test_base_source_interchange_matches_scm(self):
        for base in v31.contexts():
            for position in v31.INTERVENED_POSITIONS:
                source = v31.source_for(base, position)
                counterfactual = v31.intervene(base, source, position)
                expected = list(v31.reverse_truth(base))
                expected[v31.INPUT_LENGTH - 1 - position] = source[position]
                self.assertEqual(v31.reverse_truth(counterfactual), tuple(expected))

    def test_challenge_fixture_is_deterministic_and_stays_held_out(self):
        entropy = bytes(range(32))
        pairs1, endpoints1 = v31.challenge_pairs(entropy)
        pairs2, endpoints2 = v31.challenge_pairs(entropy)
        self.assertEqual(pairs1, pairs2)
        self.assertEqual(endpoints1, endpoints2)
        self.assertEqual(len(pairs1), 32 * len(v31.CHALLENGE_PREFIX) * len(v31.INTERVENED_POSITIONS))
        self.assertTrue(all(v31.ALPHABET[item[0]] in v31.CHALLENGE_PREFIX for item in endpoints1))

    def test_transformer_shapes_and_oracle_activation_rescue(self):
        v31.set_determinism(909)
        model = v31.TinyDecoder().eval()
        items = v31.contexts()[:2]
        ids = v31.model_inputs(items)
        _, activation, captured_logits = v31.capture_module(model, items, layer=0)
        with torch.inference_mode():
            rescued_logits = v31.endpoint_logit_slice(
                model(ids, rescue_values=torch.from_numpy(activation))
            )
        self.assertEqual(tuple(captured_logits.shape), (2, 4, v31.VOCAB_SIZE))
        self.assertEqual(tuple(rescued_logits.shape), (2, 4, v31.VOCAB_SIZE))
        self.assertTrue(torch.equal(rescued_logits, torch.from_numpy(captured_logits)))
        self.assertEqual(
            v31.max_normalized_endpoint_error(
                rescued_logits.numpy(), captured_logits
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
