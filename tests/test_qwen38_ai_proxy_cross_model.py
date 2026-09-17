import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from qwen38_ai_proxy_cross_model import (
    BLOCKER_RECEIPT_RELATIVE,
    EXPECTED_CUSTODY_ROOT,
    _assert_aggregate_safe,
    _candidate_search,
    _config_inventory_matches_locked,
    _owner_only_directory,
    _prepare_output_path,
    _tree_inventory,
    _unmanifested_model_files,
    _write_private,
    run,
)


def _model_manifest(
    path: str,
    artifact_type: str,
    repo_id: str,
    revision: str,
    weight_sha256: str,
) -> dict[str, object]:
    return {
        "path": path,
        "sha256": "a" * 64,
        "artifact_type": artifact_type,
        "repo_id": repo_id,
        "revision": revision,
        "declared_model_files": [
            {"path": "weights.bin", "bytes": 4, "sha256": weight_sha256}
        ],
    }


class TestQwen38AIProxyCrossModelContract(unittest.TestCase):
    def test_existing_assessor_and_same_weights_are_not_second_evaluator(self) -> None:
        manifests = [
            _model_manifest(
                "model/qwen/manifest.json",
                "model_weights",
                "mlx-community/Qwen3.8-27B-4bit",
                "qwen-revision",
                "b" * 64,
            ),
            _model_manifest(
                "assessor/manifest.json",
                "independent_safety_assessor",
                "LibrAI/maliciousinstruct-evaluator",
                "bert-revision",
                "c" * 64,
            ),
            _model_manifest(
                "assessor-copy/manifest.json",
                "independent_safety_assessor",
                "other/assessor",
                "other-revision",
                "c" * 64,
            ),
        ]
        result = _candidate_search(manifests, {})
        self.assertEqual(len(result["model_bearing_artifacts"]), 3)
        self.assertEqual(len(result["evaluator_candidates"]), 2)
        self.assertEqual(result["independent_evaluator_candidates"], [])
        self.assertFalse(result["same_assessor_weights_counted_as_independent"])

    def test_unknown_model_artifact_is_unclassified_not_independent(self) -> None:
        result = _candidate_search(
            [
                _model_manifest(
                    "future/manifest.json",
                    "future_model",
                    "future/repo",
                    "future-revision",
                    "d" * 64,
                )
            ],
            {},
        )
        self.assertEqual(len(result["unclassified_model_artifacts"]), 1)
        self.assertEqual(result["independent_evaluator_candidates"], [])
        self.assertFalse(result["independence_proven"])

    def test_unmanifested_model_file_is_reported_as_unclassified(self) -> None:
        unmanifested = _unmanifested_model_files(
            {
                "entries": [
                    {
                        "path": "unknown/model.safetensors",
                        "kind": "file",
                        "mode": "0o600",
                        "bytes": 12,
                    }
                ]
            },
            [],
        )
        self.assertEqual(len(unmanifested), 1)
        self.assertEqual(unmanifested[0]["path"], "unknown/model.safetensors")

    def test_blocker_receipt_rejects_raw_content_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw-content fields"):
            _assert_aggregate_safe({"nested": {"completion": "secret"}})
        with self.assertRaisesRegex(ValueError, "raw-content fields"):
            _assert_aggregate_safe({"untrusted_text": "secret"})

    def test_tree_inventory_binds_permissions_and_entries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "custody"
            root.mkdir(mode=0o700)
            file_path = root / "manifest.json"
            file_path.write_text("{}\n", encoding="utf-8")
            file_path.chmod(0o600)
            first = _tree_inventory(root)
            public_dir = root / "public-grouping"
            public_dir.mkdir(mode=0o755)
            second = _tree_inventory(root)
            self.assertNotEqual(first["entries_digest"], second["entries_digest"])
            self.assertIn(
                {
                    "path": "public-grouping",
                    "kind": "directory",
                    "actual_mode": "0o755",
                    "expected": "0o700",
                    "reason": "custody_directories_must_be_owner_only",
                },
                second["permission_findings"],
            )

    def test_config_inventory_requires_every_locked_config(self) -> None:
        self.assertFalse(_config_inventory_matches_locked([]))
        self.assertFalse(
            _config_inventory_matches_locked(
                [{"present": True, "digest_matches_locked": True}]
            )
        )
        self.assertTrue(
            _config_inventory_matches_locked(
                [
                    {"present": True, "digest_matches_locked": True},
                    {"present": True, "digest_matches_locked": True},
                    {"present": True, "digest_matches_locked": True},
                ]
            )
        )

    def test_symlink_custody_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir(mode=0o700)
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not contain a symlink"):
                _owner_only_directory(link)

    def test_report_output_is_private_and_non_overwriting(self) -> None:
        with TemporaryDirectory() as directory:
            output = _prepare_output_path(
                Path(directory).resolve() / "reports" / "receipt.json"
            )
            self.assertEqual(stat.S_IMODE(output.parent.stat().st_mode), 0o700)
            _write_private(output, "{}\n")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                _prepare_output_path(output)

    def test_canonical_receipt_exclusion_is_stable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "custody"
            root.mkdir(mode=0o700)
            receipt = root / BLOCKER_RECEIPT_RELATIVE
            receipt.parent.mkdir(parents=True, mode=0o700)
            receipt.parent.chmod(0o700)
            receipt.write_text("{}\n", encoding="utf-8")
            receipt.chmod(0o600)
            first = _tree_inventory(
                root,
                excluded_relative_paths=(BLOCKER_RECEIPT_RELATIVE,),
            )
            second = _tree_inventory(
                root,
                excluded_relative_paths=(BLOCKER_RECEIPT_RELATIVE,),
            )
            self.assertEqual(first, second)
            self.assertEqual(first["excluded_paths"], [BLOCKER_RECEIPT_RELATIVE])
            self.assertNotIn(
                {"path": BLOCKER_RECEIPT_RELATIVE, "kind": "file"},
                first["entries"],
            )

    @unittest.skipUnless(
        EXPECTED_CUSTODY_ROOT.joinpath(BLOCKER_RECEIPT_RELATIVE).is_file(),
        "persisted owner receipt is unavailable",
    )
    def test_persisted_receipt_round_trip_matches_rerun(self) -> None:
        persisted_path = EXPECTED_CUSTODY_ROOT / BLOCKER_RECEIPT_RELATIVE
        persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
        recomputed = run()
        self.assertEqual(persisted, recomputed)
        with TemporaryDirectory() as directory:
            round_trip = _prepare_output_path(Path(directory).resolve() / "receipt.json")
            _write_private(
                round_trip,
                json.dumps(recomputed, indent=2, sort_keys=True) + "\n",
            )
            loaded = json.loads(round_trip.read_text(encoding="utf-8"))
        self.assertEqual(loaded, recomputed)

    @unittest.skipUnless(EXPECTED_CUSTODY_ROOT.is_dir(), "owner custody is unavailable")
    def test_owner_receipt_is_blocked_without_opening_raw_or_label_files(self) -> None:
        opened: list[str] = []
        original_open = Path.open
        original_read_text = Path.read_text

        def tracked_open(path: Path, *args: object, **kwargs: object):
            opened.append(str(path))
            return original_open(path, *args, **kwargs)

        def tracked_read_text(path: Path, *args: object, **kwargs: object) -> str:
            opened.append(str(path))
            return original_read_text(path, *args, **kwargs)

        with patch.object(Path, "open", tracked_open), patch.object(
            Path, "read_text", tracked_read_text
        ):
            report = run()
        self.assertEqual(report["state_slice"], "theseus-lab-sandbox-v1")
        self.assertEqual(report["status"], "AI_PROXY_CROSS_MODEL_BLOCKED_MISSING_INDEPENDENT_EVALUATOR")
        self.assertEqual(report["claim_ceiling"], "LOCAL_AI_PROXY_CROSS_MODEL_CUSTODY_BLOCKED")
        self.assertIsNone(report["blocker"]["cross_model_rows_considered"])
        self.assertEqual(report["blocker"]["rows_available_in_existing_generation_receipt"], 1139)
        self.assertEqual(report["evaluator_search"]["independent_evaluator_candidates"], [])
        self.assertTrue(report["existing_same_model_result"]["preserved"])
        self.assertEqual(
            report["human_calibration"]["status"],
            "BLINDED_CALIBRATION_PACKET_READY",
        )
        self.assertFalse(report["evidence"]["raw_transcript_read"])
        self.assertFalse(report["evidence"]["human_packet_read"])
        self.assertFalse(report["evidence"]["labels_invented"])
        self.assertTrue(report["artifacts"]["primary_score_receipt"]["verified"])
        self.assertTrue(report["artifacts"]["same_model_alternate_score_receipts"]["all_verified"])
        self.assertFalse(report["future_cross_model_protocol"]["metrics_available"])
        self.assertIsNone(report["future_cross_model_protocol"]["threshold_selected"])
        self.assertFalse(report["behavior_mechanism_divergence"]["reported"])
        self.assertTrue(all(not path.endswith("responses.jsonl") for path in opened))
        self.assertTrue(all(not path.endswith("packet.jsonl") for path in opened))
        self.assertTrue(all("/labels/" not in path for path in opened))


if __name__ == "__main__":
    unittest.main()
