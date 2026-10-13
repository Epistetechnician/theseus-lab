"""Hermetic exposure classification checks; no model assets or network.

State slices: theseus-lab-sandbox-v1, theseus-checkpoint-exposure-feasibility-v1.
"""
import unittest
import theseus_checkpoint_exposure_v1 as a


class Exposure(unittest.TestCase):
    def test_catalog_download_popularity_not_weight_execution(self):
        row={'model_id':'Owner/Model','downloads':90000,'revision':'a'*40}
        self.assertEqual(list(a.identity_mentions({'queries':[row]},'v27-hub-screen.json'))[0][1],'METADATA')

    def test_metadata_weight_manifest_not_actual_download(self):
        row={'model_id':'Owner/Model','weight_files_metadata':[{'bytes':500,'sha256':'b'*64}]}
        self.assertEqual(a.context_class('checkpoint-screen-report.json','/reviews/0/model_id',row),'METADATA')

    def test_positive_download_requires_receipt(self):
        row={'model_id':'Owner/Model','downloaded_bytes':100,'downloaded_files':[{'path':'model.safetensors'}]}
        self.assertEqual(a.context_class('v25-download-manifest.json','/downloaded/0/model_id',row),'WEIGHT_RECEIPT')

    def test_prediction_and_outcome_execution(self):
        row={'model_id':'Owner/Model','outcome_array_sha256':'a'*64}
        self.assertEqual(a.context_class('report.json','/cases/0/model_id',row),'EXECUTION_RECEIPT')

    def test_unknown_is_not_relabelled_metadata(self):
        self.assertEqual(a.exposure_status({'METADATA':2,'UNKNOWN':1}),'UNKNOWN')
        self.assertEqual(a.exposure_status({}),'UNKNOWN')

    def test_exposure_monotone(self):
        self.assertEqual(a.exposure_status({'METADATA':1,'WEIGHT_FILE':1}),'WEIGHTS_RECORDED')
        self.assertEqual(a.exposure_status({'UNKNOWN':1,'EXECUTION_RECEIPT':1}),'EXECUTION_RECORDED')

    def test_source_copy_is_not_new_event(self):
        self.assertEqual(list(a.identity_mentions({'model_id':'Owner/Model'},'run/frozen-packet/checkpoints.json')),[])

    def test_catalog_ids_are_metadata_but_token_ids_are_not_models(self):
        self.assertEqual(list(a.identity_mentions({'ids':['Owner/Model']},'gpt2_gelu-catalog.json')),
                         [('owner/model','METADATA','/ids/0')])
        self.assertEqual(list(a.identity_mentions({'input_ids':[3,4]},'inputs.json')),[])


if __name__=='__main__':unittest.main()
