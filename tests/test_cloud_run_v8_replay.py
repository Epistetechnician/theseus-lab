from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cloud_run_v8_replay.cloud_run_entrypoint import _replica_metric_evidence
from cloud_run_v8_replay.freeze_run_packet import (
    _canonical_sha256,
    _continuation_lineage,
    _verify_input_object_preflight,
)


class CloudRunV8ReplayAdapterTests(unittest.TestCase):
    def test_preserves_early_reconstruction_blocker_without_evaluation_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replica = Path(temporary)
            replication = {
                "assurance_level": "INCONCLUSIVE",
                "blockers": ["family-a:candidate_weight_mismatch"],
            }
            (replica / "replication-report.json").write_text(json.dumps(replication))

            evaluation, captured = _replica_metric_evidence(replica)

        self.assertIsNone(evaluation)
        self.assertEqual(captured["assurance_level"], "INCONCLUSIVE")
        self.assertEqual(captured["blockers"], ["family-a:candidate_weight_mismatch"])

    def test_reads_evaluation_and_replication_reports_after_held_out_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            replica = Path(temporary)
            evaluation = {"outcome_status": "PASS", "cases": [{"case_status": "PASS"}]}
            replication = {"assurance_level": "REPRODUCED_LOCAL", "blockers": []}
            (replica / "replacement-loss-v3-report.json").write_text(json.dumps(evaluation))
            (replica / "replication-report.json").write_text(json.dumps(replication))

            captured_evaluation, captured_replication = _replica_metric_evidence(replica)

        self.assertEqual(captured_evaluation, evaluation)
        self.assertEqual(captured_replication, replication)

    def test_freezes_v2_failure_receipts_as_continuation_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            packet_path = root / "cloud-run-run-packet-v2.json"
            packet = {
                "schema": "theseus-v8-cloud-run-run-packet-v2",
                "target_digest": "87bcaba185c882f8f19201dfe3f8fe42a7e3e856d8ad6f4141d4683f5fb88f61",
                "v8_packet_sha256": "b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2",
            }
            packet["run_packet_sha256"] = _canonical_sha256(packet)
            packet_path.write_text(json.dumps(packet))

            receipt_names = (
                "cloud-run-execution-dispatch-v2.json",
                "cloud-run-job-update-v2.json",
                "cloud-run-job-describe-preexecution-v2.json",
                "cloud-run-execution-start-v2.json",
                "cloud-run-execution-final-v2.json",
                "cloud-run-task-final-v2.json",
                "cloud-run-logs-final-v2.json",
                "cloud-run-logs-direct-v2.json",
                "input-manifest-object-v2.json",
                "run-packet-object-v2.json",
                "pinned-v7-object-after-v2.json",
            )
            for name in receipt_names:
                value = {}
                if name == "cloud-run-execution-final-v2.json":
                    value = {
                        "metadata": {"name": "theseus-v8replay-1f9f1ac6-m4kcf"},
                        "status": {"conditions": [{"type": "Completed", "status": "False"}]},
                    }
                elif name == "cloud-run-task-final-v2.json":
                    value = [{"status": {"lastAttemptResult": {"exitCode": 1}}}]
                content = "retained log receipt" if name.startswith("cloud-run-logs-") else json.dumps(value)
                (receipts / name).write_text(content)

            lineage = _continuation_lineage(
                packet_path,
                receipts,
                "Use the clean pinned archive and preserve the v2 failure receipts.",
            )

        self.assertEqual(lineage["prior_run_packet_schema"], "theseus-v8-cloud-run-run-packet-v2")
        self.assertEqual(lineage["prior_completion_status"], "EXECUTION_FAILED")
        self.assertEqual(lineage["prior_task_exit_code"], 1)
        self.assertEqual(len(lineage["prior_receipts"]), 11)

    def test_preflight_must_bind_each_exact_frozen_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            inputs = [{"path": "phase.tar", "bytes": 12, "sha256": "a" * 64}]
            body = {
                "schema": "theseus-v8-cloud-run-input-object-preflight-v1",
                "prior_run_packet_sha256": "b" * 64,
                "prior_execution_name": "execution-v2",
                "verified_objects": [
                    {
                        "path": "phase.tar",
                        "bytes": 12,
                        "sha256": "a" * 64,
                        "status": "VERIFIED",
                    }
                ],
            }
            value = dict(body)
            value["preflight_sha256"] = _canonical_sha256(body)
            path.write_text(json.dumps(value))

            verified = _verify_input_object_preflight(
                path,
                inputs,
                {
                    "prior_run_packet_sha256": "b" * 64,
                    "prior_execution_name": "execution-v2",
                },
            )

        self.assertEqual(verified["verified_objects"][0]["status"], "VERIFIED")

    def test_preflight_rejects_an_input_digest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "preflight.json"
            body = {
                "schema": "theseus-v8-cloud-run-input-object-preflight-v1",
                "prior_run_packet_sha256": "b" * 64,
                "prior_execution_name": "execution-v2",
                "verified_objects": [
                    {
                        "path": "phase.tar",
                        "bytes": 12,
                        "sha256": "c" * 64,
                        "status": "VERIFIED",
                    }
                ],
            }
            value = dict(body)
            value["preflight_sha256"] = _canonical_sha256(body)
            path.write_text(json.dumps(value))

            with self.assertRaisesRegex(ValueError, "differs from frozen bytes"):
                _verify_input_object_preflight(
                    path,
                    [{"path": "phase.tar", "bytes": 12, "sha256": "a" * 64}],
                    {
                        "prior_run_packet_sha256": "b" * 64,
                        "prior_execution_name": "execution-v2",
                    },
                )


if __name__ == "__main__":
    unittest.main()
