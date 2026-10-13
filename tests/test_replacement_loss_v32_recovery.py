"""Post-run fault-injection checks of frozen V32 code; separate from science.

State slice: theseus-replacement-loss-v32-causal-ladder-v1. All mutations are
in automatically cleaned constructed fixtures in external owner custody.
The successful scientific primary/replica roots are never changed or rerun.
"""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import replacement_loss_v32_runner as v


def inventory(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class Recovery(unittest.TestCase):
    def test_injected_failure_retained_and_root_cannot_be_reused(self):
        with tempfile.TemporaryDirectory(prefix='theseus-v32-engineering-', dir='/Volumes/PrimaryED/ResearchArtifacts') as directory:
            base = Path(directory)
            with patch.object(v, 'BASE', base), patch.object(v, 'TRAIN_ROWS', 2), patch.object(v, 'VISIBLE_ROWS', 2), patch.object(v, 'ADAPT_ROWS', 2), patch.object(v, 'train_guide', side_effect=RuntimeError('CONSTRUCTED_PRE_GUIDE_CRASH')):
                with self.assertRaisesRegex(RuntimeError, 'CONSTRUCTED_PRE_GUIDE_CRASH'):
                    v.run('primary')
                root = base / 'primary'
                receipt = json.loads((root / 'failure-receipt.json').read_text())
                self.assertEqual(receipt['outcome_status'], 'INCONCLUSIVE')
                self.assertFalse((root / 'candidate-freeze.json').exists())
                self.assertFalse((root / 'challenge-binding.json').exists())
                before = inventory(root)
                with self.assertRaises(FileExistsError):
                    v.run('primary')
                self.assertEqual(before, inventory(root))

    def test_bad_packet_digest_stops_before_fixture_generation(self):
        with tempfile.TemporaryDirectory(prefix='theseus-v32-engineering-', dir='/Volumes/PrimaryED/ResearchArtifacts') as directory:
            root = Path(directory)
            for member in (*v.MEMBERS, 'docs/replacement-loss-v32-freeze.json'):
                target = root / member
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(v.REPO / member, target)
            manifest = root / 'docs/replacement-loss-v32-freeze.json'
            value = json.loads(manifest.read_text())
            value['packet_sha256'] = '0' * 64
            v.write_json(manifest, value)
            with patch.object(v, 'REPO', root), patch.object(v, 'draw') as draw:
                with self.assertRaisesRegex(ValueError, 'frozen V32 packet mismatch'):
                    v.verify_freeze()
                draw.assert_not_called()


if __name__ == '__main__':
    unittest.main()
