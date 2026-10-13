"""Tests for v9 post-run replication report digest validation.

State slice: theseus-lab-sandbox-v1.
"""

import json
import tempfile
import unittest
from pathlib import Path

import replacement_loss_v9_replication_binding_v4 as binding


class VerifiedReportTests(unittest.TestCase):
    def test_accepts_a_valid_digest_bound_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            payload = {"outcome_status": "FAIL", "reproducibility_status": "REPRODUCED_LOCAL"}
            payload["report_sha256"] = binding._canonical_digest(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(binding._verified_report(path)["outcome_status"], "FAIL")

    def test_rejects_report_content_changed_after_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            payload = {"outcome_status": "FAIL"}
            payload["report_sha256"] = binding._canonical_digest(payload)
            payload["outcome_status"] = "PASS"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "digest is invalid"):
                binding._verified_report(path)

    def test_reconstructs_canonical_and_file_byte_digests_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            original = {
                "outcome_status": "FAIL",
                "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
            }
            original["report_sha256"] = binding._canonical_digest(original)
            binding._write_json(path, original)
            final = dict(original)
            final["reproducibility_status"] = "REPRODUCED_LOCAL"
            final["replication_manifest_sha256"] = "ab" * 32
            final.pop("report_sha256")
            final["report_sha256"] = binding._canonical_digest(final)

            identity = binding._pre_replication_report_identity(
                final,
                {
                    "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
                    "outcome_report_sha256": original["report_sha256"],
                },
            )

            self.assertEqual(identity["canonical_report_sha256"], original["report_sha256"])
            self.assertEqual(identity["report_file_sha256"], binding._sha_file(path))


if __name__ == "__main__":
    unittest.main()
