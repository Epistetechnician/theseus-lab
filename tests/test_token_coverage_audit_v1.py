"""Hermetic coverage math; slices theseus-lab-sandbox-v1 and theseus-v38-token-coverage-audit-v1."""
import unittest
import numpy as np
from theseus_token_coverage_audit_v1 import rmse_sums,summary


class Coverage(unittest.TestCase):
    def test_ragged_prefix_weighting(self):
        rows=np.array([[4.,16.,4.,1.,16.,1.],[12.,48.,12.,1.,16.,1.]])
        value=summary(rows)
        self.assertEqual(value['prefix_nrmse'],.5);self.assertEqual(value['final_nrmse'],.25)
        self.assertTrue(value['twofold_coverage_gap'])

    def test_zero_reference_floor_and_invalid_count(self):
        self.assertEqual(rmse_sums(0.,0.,2),0.)
        with self.assertRaises(ValueError):rmse_sums(1.,1.,0)
        with self.assertRaises(ValueError):rmse_sums(float('nan'),1.,1)


if __name__=='__main__':unittest.main()
