"""Cache-first checks in state slices theseus-lab-sandbox-v1,
theseus-pretrained-development-transfer-v37r1 and theseus-public-checkpoint-acquisition-v37r1.
"""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import replacement_loss_v37r1_runner as v


class Cache(unittest.TestCase):
    def test_no_network_from_empty_cached_population(self):
        with patch.object(v,'cache_verified'),patch.object(v,'ORIGINAL_CANDIDATES',return_value=[]):
            self.assertEqual(v.cache_candidates({},'opt_relu',{}),[])

    def test_missing_public_metadata_excluded(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(v,'STOPPED_BASE',Path(folder)),\
             patch.object(v,'cache_verified'),patch.object(v,'ORIGINAL_CANDIDATES',return_value=[{'model_id':'owner/base'}]):
            self.assertEqual(v.cache_candidates({},'opt_relu',{}),[])

    def test_changed_metadata_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);p=base/'acquisition/error.json';p.parent.mkdir();p.write_text('{}')
            data=base/'config.json';data.write_text('{}')
            manifest={'terminal_error_sha256':v.r.sha(p),'files':[{'path':'config.json','bytes':2,'sha256':'a'*64}]}
            with patch.object(v,'STOPPED_BASE',base),patch.object(v.r,'read',return_value=manifest):
                with self.assertRaisesRegex(ValueError,'metadata differs'):v.cache_verified()

    def test_prior_selection_blocks_continuation(self):
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);p=base/'acquisition/error.json';p.parent.mkdir();p.write_text('{}')
            (p.parent/'selection-freeze.json').write_text('{}')
            with patch.object(v,'STOPPED_BASE',base),patch.object(v.r,'read',return_value={'terminal_error_sha256':v.r.sha(p),'files':[]}):
                with self.assertRaisesRegex(ValueError,'progressed'):v.cache_verified()


if __name__=='__main__':unittest.main()
