import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen38_calibration import (
    CALIBRATION_SEED,
    _fit_threshold,
    _load_labels,
    _packet_digest,
    _packet_rows,
    _stratified_sample,
    _owner_only_directory,
)


def _source_record(dataset: str, row_id: int, stratum: str) -> dict[str, object]:
    question = f"question-{dataset}-{row_id}"
    completion = f"completion-{dataset}-{row_id}"
    return {
        "dataset": dataset,
        "row_id": row_id,
        "risk_area": stratum if dataset == "harmful" else None,
        "task_type": stratum if dataset != "harmful" else None,
        "question": question,
        "completion": completion,
        "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
        "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
    }


class TestQwen38CalibrationContract(unittest.TestCase):
    def test_stratified_sample_is_stable_and_bounded(self) -> None:
        records = [_source_record("harmful", index, f"risk-{index % 2}") for index in range(10)]
        first = _stratified_sample(records, 6)
        second = _stratified_sample(records, 6)
        self.assertEqual(first, second)
        self.assertEqual(len({item["row_id"] for item in first}), 6)
        self.assertEqual(CALIBRATION_SEED.startswith("theseus-lab-sandbox-v1:"), True)

    def test_packet_splits_are_disjoint_and_digest_bound(self) -> None:
        records = [
            *[_source_record("harmful", index, f"risk-{index % 2}") for index in range(8)],
            *[_source_record("benign_contrast", index, f"task-{index % 2}") for index in range(8)],
        ]
        packet = _packet_rows(records, harmful_count=4, contrast_count=4)
        tune = {row["calibration_id"] for row in packet if row["split"] == "tune"}
        assessment = {row["calibration_id"] for row in packet if row["split"] == "assessment"}
        self.assertTrue(tune)
        self.assertTrue(assessment)
        self.assertTrue(tune.isdisjoint(assessment))
        self.assertEqual(_packet_digest(packet), _packet_digest(packet))

    def test_threshold_is_fitted_on_scores_and_labels(self) -> None:
        rows = [
            {"harmful_score": 0.1, "label": "safe"},
            {"harmful_score": 0.2, "label": "safe"},
            {"harmful_score": 0.8, "label": "unsafe"},
            {"harmful_score": 0.9, "label": "unsafe"},
        ]
        threshold, metrics = _fit_threshold(rows)
        self.assertEqual(threshold, 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)

    def test_label_loader_rejects_blank_or_extra_rows(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {"calibration_id": "one", "rater_id": "r1", "label": None}
                )
                + "\n"
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "blank or invalid label"):
                _load_labels(path, {"one"})

    def test_calibration_custody_rejects_repository_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            _owner_only_directory(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
