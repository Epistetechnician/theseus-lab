import json
import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen38_subagent_calibration import (
    ALLOWED_LABEL_FIELDS,
    _load_subagent_labels,
    _resolve_consensus,
)


class TestQwen38SubagentCalibrationContract(unittest.TestCase):
    def test_subagent_loader_requires_private_exact_schema(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "calibration_id": "one",
                        "split": "tune",
                        "rater_id": "subagent-rater-a",
                        "label": "safe",
                        "confidence": "high",
                    }
                )
                + "\n"
            )
            path.chmod(0o600)
            self.assertEqual(
                set(json.loads(path.read_text(encoding="utf-8"))),
                ALLOWED_LABEL_FIELDS,
            )
            rater_id, labels = _load_subagent_labels(path, {"one"})
            self.assertEqual(rater_id, "subagent-rater-a")
            self.assertEqual(labels, {"one": "safe"})
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_subagent_loader_rejects_raw_content_fields(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "calibration_id": "one",
                        "split": "tune",
                        "rater_id": "subagent-rater-a",
                        "label": "safe",
                        "confidence": "high",
                        "completion": "raw completion",
                    }
                )
                + "\n"
            )
            path.chmod(0o600)
            with self.assertRaisesRegex(ValueError, "prohibited or missing fields"):
                _load_subagent_labels(path, {"one"})

    def test_consensus_quarantines_disagreement_and_ambiguity(self) -> None:
        rows = [
            {"calibration_id": "one"},
            {"calibration_id": "two"},
            {"calibration_id": "three"},
        ]
        consensus, disagreements, ambiguous = _resolve_consensus(
            rows,
            [
                {"one": "safe", "two": "safe", "three": "ambiguous"},
                {"one": "safe", "two": "unsafe", "three": "ambiguous"},
            ],
        )
        self.assertEqual(consensus, {"one": "safe"})
        self.assertEqual(disagreements, ["two"])
        self.assertEqual(ambiguous, ["three"])


if __name__ == "__main__":
    unittest.main()
