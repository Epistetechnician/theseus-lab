"""Contracts: theseus-v39-training-probe-replay-v1, theseus-lab-sandbox-v1,
global-experiment-assurance-lifecycle-v1."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
import theseus_training_probe_replay_v1 as v


class Replay(unittest.TestCase):
    def test_directions_match_actual_original_core_fit_rng(self):
        torch.set_num_threads(1)
        with torch.random.fork_rng():
            torch.manual_seed(410);teacher=torch.nn.Linear(4,4).eval().requires_grad_(False)
        x=torch.arange(28,dtype=torch.float32).reshape(7,4)/10;d=x.flip(0)
        with patch.object(v.c,'jvp',wraps=v.c.jvp) as spy:
            v.previous.parent.ORIGINAL_FIT(teacher,x,d,'gpt2_gelu',411,'jvp',updates=0)
        actual=spy.call_args_list[0].args[2]
        self.assertTrue(torch.equal(actual,v.directions(7,4,411)))

    def test_projection_matches_explicit_matrix_products(self):
        j=np.arange(24,dtype=np.float32).reshape(2,3,4);d=np.ones((2,4,4))
        p=v.project(j,d)
        for n in range(2):
            for i in range(4):np.testing.assert_array_equal(p[n,i],j[n]@d[n,i])
        with self.assertRaises(ValueError):v.project(j,d[:1])

    def test_replay_is_deterministic_and_seed_specific(self):
        self.assertTrue(torch.equal(v.directions(8,4,412),v.directions(8,4,412)))
        self.assertFalse(torch.equal(v.directions(8,4,412),v.directions(8,4,413)))
        with self.assertRaises(ValueError):v.directions(0,4,412)


if __name__=='__main__':unittest.main()
