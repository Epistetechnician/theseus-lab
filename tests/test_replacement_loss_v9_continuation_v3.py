"""Tests for the v9 continuation-v3 candidate digest adapter.

State slice: theseus-lab-sandbox-v1.
"""

import tempfile
import unittest
from pathlib import Path

import replacement_loss_v9_continuation_v3 as continuation
import replacement_loss_v9_runner as runner


class CandidateStateDigestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "stage-0.safetensors"
        self.module = runner.torch.nn.Linear(3, 2, bias=True, dtype=runner.torch.float32)
        self.state_digest = runner._save_module(self.path, self.module)
        self.file_digest = continuation._sha_file(self.path)
        self.stage = {
            "candidate_path": "stage-0.safetensors",
            "candidate_sha256": self.state_digest,
            "in_features": 3,
            "out_features": 2,
            "use_bias": True,
        }
        self.freeze = {"candidate_files": [{"path": "stage-0.safetensors", "sha256": self.file_digest}]}

    def tearDown(self):
        self.temp.cleanup()

    def test_checks_file_bytes_and_tensor_state_as_separate_digests(self):
        self.assertNotEqual(self.file_digest, self.state_digest)
        loaded = continuation._load_candidate_with_state_digest(self.root, self.stage, self.freeze)
        loaded_state = {name: tensor.detach().cpu().contiguous() for name, tensor in loaded.state_dict().items()}
        original_state = {name: tensor.detach().cpu().contiguous() for name, tensor in self.module.state_dict().items()}
        self.assertEqual(runner.v3._state_digest(loaded_state), runner.v3._state_digest(original_state))

    def test_rejects_changed_candidate_file_bytes(self):
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "file bytes changed"):
            continuation._load_candidate_with_state_digest(self.root, self.stage, self.freeze)


if __name__ == "__main__":
    unittest.main()
