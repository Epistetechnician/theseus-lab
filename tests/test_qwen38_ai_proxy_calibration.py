import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.qwen38_mlx_adapter import full_digest
from qwen38_ai_proxy_calibration import (
    ALTERNATE_SCORE_FIELDS,
    PARTITION_NAMES,
    PRIMARY_SCORE_FIELDS,
    ProxyObservation,
    _adjudication_summary,
    _assert_aggregate_safe,
    _assign_partitions,
    _binary_metrics,
    _dependence_contract,
    _fit_threshold_candidates,
    _owner_only_directory,
    _prepare_output_path,
    _prepare_private_output_dir,
    _tune_threshold,
    _validate_score_records,
    _write_private,
)


def _primary_record(dataset: str, row_id: int, score: float) -> dict[str, object]:
    return {
        "dataset": dataset,
        "row_id": row_id,
        "harmful_score": score,
        "lexical_label": "refusal_style" if score <= 0.5 else "non_refusal_style",
        "short_response": False,
        "score_above_threshold": score > 0.5,
        "effective_unsafe": score > 0.5,
    }


def _observation(split: str, primary: float, alternate: float) -> ProxyObservation:
    return ProxyObservation._from_verified(
        dataset="harmful",
        row_id=PARTITION_NAMES.index(split),
        primary_score=primary,
        alternate_score=alternate,
        lexical_label="non_refusal_style",
        split=split,
    )


class TestQwen38AIProxyCalibrationContract(unittest.TestCase):
    def test_direct_constructor_rejects_invalid_observations(self) -> None:
        with self.assertRaisesRegex(TypeError, "verified loader"):
            ProxyObservation("forged", 0, 0.1, 0.2, "empty", "fit")
        with self.assertRaisesRegex(ValueError, "out-of-range primary"):
            ProxyObservation._from_verified("harmful", 0, float("nan"), 0.2, "empty", "fit")
        with self.assertRaisesRegex(ValueError, "invalid partition"):
            ProxyObservation._from_verified("harmful", 0, 0.1, 0.2, "empty", "heldout")

    def test_partition_identity_sets_are_pairwise_disjoint(self) -> None:
        primary = [
            _primary_record(dataset, row_id, 0.1 + 0.8 * ((row_id + index) % 2))
            for index, dataset in enumerate(("harmful", "benign_contrast"))
            for row_id in range(8)
        ]
        alternate = [
            {"dataset": row["dataset"], "row_id": row["row_id"], "harmful_score": 0.2}
            for row in primary
        ]
        observations, partition = _assign_partitions(primary, alternate)
        sets = {
            split: {(row.dataset, row.row_id) for row in observations if row.split == split}
            for split in PARTITION_NAMES
        }
        self.assertTrue(partition["pairwise_disjoint"])
        self.assertEqual(set(sets), set(PARTITION_NAMES))
        for left_index, left in enumerate(PARTITION_NAMES):
            for right in PARTITION_NAMES[left_index + 1 :]:
                self.assertTrue(sets[left].isdisjoint(sets[right]))

    def test_fit_and_threshold_tuning_reject_split_leakage(self) -> None:
        fit_row = _observation("fit", 0.2, 0.2)
        tune_row = _observation("threshold_tune", 0.8, 0.8)
        with self.assertRaisesRegex(ValueError, "only the fit partition"):
            _fit_threshold_candidates([tune_row])
        with self.assertRaisesRegex(ValueError, "only the threshold_tune partition"):
            _tune_threshold([fit_row], (0.5,))

    def test_threshold_tuning_uses_locked_candidate_support(self) -> None:
        rows = [
            ProxyObservation._from_verified("harmful", 0, 0.1, 0.1, "refusal_style", "fit"),
            ProxyObservation._from_verified("harmful", 1, 0.9, 0.9, "non_refusal_style", "fit"),
        ]
        candidates, fit = _fit_threshold_candidates(rows)
        self.assertIn(0.5, candidates)
        self.assertTrue(fit["heldout_rows_used"] is False)
        tuning_rows = [
            ProxyObservation._from_verified(
                "harmful", 2, 0.1, 0.1, "refusal_style", "threshold_tune"
            ),
            ProxyObservation._from_verified(
                "harmful", 3, 0.9, 0.9, "non_refusal_style", "threshold_tune"
            ),
        ]
        threshold, tuning = _tune_threshold(tuning_rows, candidates)
        self.assertIn(threshold, candidates)
        self.assertEqual(threshold, tuning["locked_primary_threshold"])
        self.assertTrue(tuning["heldout_rows_used"] is False)

    def test_digest_tampering_and_row_swaps_fail_closed(self) -> None:
        rows = [_primary_record("harmful", 0, 0.1), _primary_record("harmful", 1, 0.9)]
        expected_keys = {("harmful", 0), ("harmful", 1)}
        expected_digest = full_digest(rows)
        tampered = [dict(row) for row in rows]
        tampered[0]["harmful_score"] = 0.8
        with self.assertRaisesRegex(ValueError, "canonical digest mismatch"):
            _validate_score_records(
                tampered,
                expected_keys,
                allowed_fields=PRIMARY_SCORE_FIELDS,
                expected_digest=expected_digest,
                label="primary score",
            )
        swapped = [dict(rows[1]), dict(rows[0])]
        with self.assertRaisesRegex(ValueError, "canonical source-key order"):
            _validate_score_records(
                swapped,
                expected_keys,
                allowed_fields=PRIMARY_SCORE_FIELDS,
                expected_digest=expected_digest,
                label="primary score",
            )

    def test_raw_content_and_unknown_fields_are_rejected(self) -> None:
        row = {"dataset": "harmful", "row_id": 0, "harmful_score": 0.1}
        raw_leak = dict(row)
        raw_leak["completion"] = "secret completion"
        with self.assertRaisesRegex(ValueError, "prohibited or missing fields"):
            _validate_score_records(
                [raw_leak],
                {("harmful", 0)},
                allowed_fields=ALTERNATE_SCORE_FIELDS,
                expected_digest=full_digest([raw_leak]),
                label="alternate score",
            )
        with self.assertRaisesRegex(ValueError, "raw-content fields"):
            _assert_aggregate_safe(
                {"completion": "secret completion"},
                [{"completion": "secret completion"}],
            )

    def test_proxy_adjudication_quarantines_disagreements(self) -> None:
        rows = [
            ProxyObservation._from_verified(
                "harmful", 0, 0.9, 0.9, "non_refusal_style", "proxy_adjudication"
            ),
            ProxyObservation._from_verified(
                "harmful", 1, 0.1, 0.9, "non_refusal_style", "proxy_adjudication"
            ),
        ]
        summary = _adjudication_summary(rows, 0.5)
        self.assertEqual(summary["consensus_rows"], 1)
        self.assertEqual(summary["unresolved_rows"], 1)
        self.assertEqual(summary["policy"], "exact_proxy_agreement_is_consensus; disagreement_is_unresolved")

    def test_same_model_dependence_is_not_independence(self) -> None:
        rows = [
            ProxyObservation._from_verified(
                "harmful", 0, 0.9, 0.9, "non_refusal_style", "heldout_assessment"
            ),
            ProxyObservation._from_verified(
                "harmful", 1, 0.1, 0.1, "refusal_style", "heldout_assessment"
            ),
        ]
        contract = _dependence_contract(_binary_metrics(rows, 0.5), 1.0)
        self.assertTrue(contract["same_assessor_model_sha256"])
        self.assertFalse(contract["independent_of_primary_proxy"])
        self.assertFalse(contract["independent_model_claim"])

    def test_output_custody_is_0700_and_0600(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            score_dir = _prepare_private_output_dir(root / "scores")
            self.assertEqual(stat.S_IMODE(score_dir.stat().st_mode), 0o700)
            score_file = score_dir / "scores.jsonl"
            _write_private(score_file, json.dumps({"rows": 1}) + "\n")
            self.assertEqual(stat.S_IMODE(score_file.stat().st_mode), 0o600)
            report_path = _prepare_output_path(root / "reports" / "report.json")
            _write_private(report_path, "{}\n")
            self.assertEqual(stat.S_IMODE(report_path.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(report_path.stat().st_mode), 0o600)

    def test_output_parent_symlink_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir(mode=0o700)
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _prepare_output_path(link / "report.json")

    def test_symlink_custody_path_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir(mode=0o700)
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                _owner_only_directory(link)


if __name__ == "__main__":
    unittest.main()
