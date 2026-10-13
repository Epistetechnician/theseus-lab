"""V37 scope and eligibility; no public requests or pretrained outputs.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-development-transfer-v37,
theseus-public-checkpoint-acquisition-v37.
"""
import unittest
import replacement_loss_v37_runner as v


class Development(unittest.TestCase):
    def test_metadata_only_eligible_with_honest_scope(self):
        audit={'identities':{'owner/base':{'status':'METADATA_ONLY_IN_RETAINED_CUSTODY'}}}
        self.assertTrue(v.metadata_eligible('Owner/Base',audit))
        self.assertFalse(v.metadata_eligible('new/unclassified',audit))
        for status in ('UNKNOWN','EXECUTION_RECORDED','WEIGHTS_RECORDED'):
            self.assertFalse(v.metadata_eligible('owner/base',{'identities':{'owner/base':{'status':status}}}))

    def test_selection_metadata_order_not_input_order(self):
        audit={'identities':{x:{'status':'METADATA_ONLY_IN_RETAINED_CUSTODY'} for x in ('owner/a','owner/b')}}
        catalog={'family_screens':{'opt_relu':{'queries':[{'result_projection':[
            {'model_id':'owner/b','downloads':3,'model_type':'opt'},
            {'model_id':'owner/a','downloads':3,'model_type':'opt'},
            {'model_id':'owner/a','downloads':3,'model_type':'opt'}]}]}}}
        self.assertEqual([row['model_id'] for row in v.catalog_candidates(catalog,'opt_relu',audit)],['owner/a','owner/b'])

    def test_unknown_wrong_architecture_and_random_excluded(self):
        ids=('owner/a','owner/random');audit={'identities':{x:{'status':'METADATA_ONLY_IN_RETAINED_CUSTODY'} for x in ids}}
        catalog={'family_screens':{'opt_relu':{'queries':[{'result_projection':[
            {'model_id':'owner/a','model_type':'llama'}, {'model_id':'owner/random','model_type':'opt'},
            {'model_id':'unrecorded/model','model_type':'opt'}]}]}}}
        self.assertEqual(v.catalog_candidates(catalog,'opt_relu',audit),[])

    def test_capped_catalog(self):
        audit={'identities':{'o/a':{'status':'METADATA_ONLY_IN_RETAINED_CUSTODY'}}}
        catalog={'family_screens':{'opt_relu':{'queries':[{'result_projection':[{'model_id':'o/a','model_type':'opt'}]}]}}}
        self.assertEqual(v.catalog_candidates(catalog,'opt_relu',audit,limit=0),[])

    def test_family_activation_restores_scope(self):
        try:
            v.activate('gpt2_gelu');self.assertEqual(v.c.FAMILIES,('gpt2_gelu',))
            self.assertEqual(v.r.BASE,v.BASE/'gpt2_gelu')
        finally:v.activate(None)
        self.assertEqual(v.c.FAMILIES,v.FAMILIES)

    def test_no_implicit_public_acquisition(self):
        with self.assertRaises(PermissionError):v.acquire(False)

    def test_old_base_provenance_and_shape_gates_preserved(self):
        self.assertIsNotNone(v.adapter.public_base_reason('owner/base',{'license':'mit','base_model':'other/model'},'trained from scratch'))
        self.assertIsNotNone(v.adapter.config_reason({'model_type':'opt','hidden_size':2048},'opt_relu'))

    def test_loss_and_seed_contract_preserved(self):
        self.assertEqual(v.c.METHODS,('jvp','endpoints'))
        self.assertEqual(v.c.SEEDS,(36101,36102,36103))
        self.assertEqual(v.r.MAX_SECONDS,8*3600)


if __name__=='__main__':unittest.main()
