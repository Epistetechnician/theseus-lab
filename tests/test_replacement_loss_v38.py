"""V38 retained development gates; no trained guide or held-out outcomes.

State slices: theseus-lab-sandbox-v1, theseus-reused-gpt2-early-ladder-v38.
"""
import unittest
import replacement_loss_v38_runner as v


class Pilot(unittest.TestCase):
    def test_fixed_early_pair_explicitly_reused(self):
        self.assertEqual(v.SITES,(0,2));self.assertEqual(v.c.FAMILIES,('gpt2_gelu',))
        self.assertIn('REUSED',v.c.CLAIM);self.assertIn('EARLY',v.c.CLAIM)

    def test_thresholds_not_weakened(self):
        good={'0':{'removal_logit_nrmse':.10,'removal_top1_change':.20},'2':{'removal_logit_nrmse':.11,'removal_top1_change':.21}}
        self.assertTrue(v.qualification(good))
        for key in ('removal_logit_nrmse','removal_top1_change'):
            changed={s:dict(x) for s,x in good.items()};changed['2'][key]=.099 if key.endswith('nrmse') else .199
            self.assertFalse(v.qualification(changed))

    def test_loss_optimizer_and_repeats_unchanged(self):
        self.assertEqual(v.c.UPDATES,1024);self.assertEqual(v.c.METHODS,('jvp','endpoints'))
        self.assertEqual(v.c.SEEDS,(36101,36102,36103));self.assertEqual(v.r.MAX_SECONDS,28800)


if __name__=='__main__':unittest.main()
