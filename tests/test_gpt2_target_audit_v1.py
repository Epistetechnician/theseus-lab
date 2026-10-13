"""Selection audit checks, slices theseus-lab-sandbox-v1 and theseus-gpt2-visible-target-audit-v1."""
import unittest
from theseus_gpt2_target_audit_v1 import choose


class Selection(unittest.TestCase):
    def test_energy_cannot_select_unqualified_layer(self):
        metrics={str(s):{'removal_logit_nrmse':.3 if s in (0,2) else .08,
                         'removal_top1_change':.4 if s in (0,2) else .06} for s in range(4)}
        self.assertEqual(choose(metrics,4),([[0],[2]],[0,2]))

    def test_missing_critical_half_is_not_filled_by_weak_layer(self):
        metrics={str(s):{'removal_logit_nrmse':.2,'removal_top1_change':.3 if s==0 else .1} for s in range(4)}
        self.assertEqual(choose(metrics,4),([[0],[]],None))


if __name__=='__main__':unittest.main()
