"""V36R1 custody/acquisition engineering; no public requests or trained assets.

State slices: theseus-lab-sandbox-v1, theseus-pretrained-whole-block-transfer-v36r1,
theseus-public-checkpoint-acquisition-v36r1, global-experiment-assurance-lifecycle-v1.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import replacement_loss_v36r1_runner as repair


class Continuation(unittest.TestCase):
    def test_known_opaque_exact_digest_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'legacy.json';path.write_bytes(b'')
            approved={'legacy.json':(0,hashlib.sha256(b'').hexdigest())}
            value,row=repair.classify(path,'legacy.json',approved)
            self.assertIsNone(value)
            self.assertEqual(row['kind'],'QUARANTINED_OPAQUE_LEGACY_EXPORT')
            path.write_text('changed')
            with self.assertRaisesRegex(ValueError,'changed'):repair.classify(path,'legacy.json',approved)

    def test_unknown_malformed_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'new.json';path.write_text('not json')
            with self.assertRaises(json.JSONDecodeError):repair.classify(path,'new.json',{})

    def test_valid_identity_record_preserves_exclusion(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'valid.json';path.write_text(json.dumps({'model_id':'Owner/Model','ids':[1,2]}))
            value,row=repair.classify(path,'valid.json',{})
            self.assertEqual(row['kind'],'JSON_IDENTITY_RECORD')
            sets={key:set() for key in ('model_ids','text_hashes','token_hashes','prior_digests')}
            repair.r.collect_history(value,sets)
            self.assertEqual(sets['model_ids'],{'owner/model'})
            self.assertEqual(sets['token_hashes'],{repair.c.digest([1,2])})

    def test_public_acquisition_is_explicit_opt_in(self):
        with self.assertRaises(PermissionError):repair.acquire(False)

    def test_from_scratch_and_license_provenance_required(self):
        self.assertIsNone(repair.public_base_reason('owner/base',{'license':'mit'},'This model was trained from scratch.'))
        for identity,card,text in (('owner/base',{},'trained from scratch'),
                ('owner/base',{'license':'mit','base_model':'prior/model'},'trained from scratch'),
                ('owner/random',{'license':'mit'},'trained from scratch'),
                ('owner/model_sft',{'license':'mit'},'trained from scratch'),
                ('owner/base',{'license':['mit']},'trained from scratch'),
                ('owner/base',{'license':'mit'},'a fine-tuned model')):
            self.assertIsNotNone(repair.public_base_reason(identity,card,text))

    def test_model_shape_and_custom_code_rejected(self):
        valid={'model_type':'opt','hidden_size':16,'num_hidden_layers':4,'vocab_size':29,'activation_function':'relu'}
        self.assertIsNone(repair.config_reason(valid,'opt_relu'))
        for key,value in (('hidden_size',4096),('hidden_size',None),('vocab_size',131072),('auto_map',{'x':'custom'}),('quantization_config',{'x':True}),('activation_function','gelu')):
            changed={**valid,key:value}
            self.assertIsNotNone(repair.config_reason(changed,'opt_relu'))

    def test_quarantine_receipt_cannot_drop_an_entry(self):
        with patch.object(repair,'ORIGINAL_VERIFY_INTAKE',return_value={'quarantined_legacy_exports':[]}):
            with self.assertRaises(ValueError):repair.verify_intake()

    def test_empty_catalog_opens_no_weight_download_and_cannot_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder)
            history={'exclusion':{'model_ids':[],'prior_digests':[]}}
            (base/'intake.json').write_text(json.dumps(history))
            with patch.object(repair.r,'BASE',base),patch.object(repair,'verify_intake',return_value=history),\
                 patch.object(repair.r,'mount_guard'),patch.object(repair.r,'verify_source',return_value='fixture'),\
                 patch.object(repair.r,'dependency_fingerprint',return_value={'dependency_files_sha256':'fixture'}),\
                 patch('huggingface_hub.HfApi') as api,patch('huggingface_hub.snapshot_download') as download:
                api.return_value.list_models.return_value=[]
                outcome=repair.acquire(True)
                self.assertEqual(outcome['outcome_status'],'NO_ELIGIBLE_CHECKPOINT_COHORT')
                self.assertFalse(outcome['weight_downloads_opened'])
                download.assert_not_called()
                self.assertEqual(api.return_value.list_models.call_count,3)
                for call in api.return_value.list_models.call_args_list:self.assertIs(call.kwargs['token'],False)
                with self.assertRaises(FileExistsError):repair.acquire(True)


if __name__=='__main__':unittest.main()
