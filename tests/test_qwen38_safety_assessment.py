import unittest
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from qwen38_safety_assessment import (
    CORPUS_SHA256,
    RAW_SCHEMA_VERSION,
    STATE_SLICE,
    _assessment_text,
    _load_transcript,
    _prepare_output_path,
    _prepare_scored_output_dir,
    _wilson_interval,
)


class TestQwen38SafetyAssessmentContract(unittest.TestCase):
    def test_wilson_interval_is_bounded_and_deterministic(self) -> None:
        lower, upper = _wilson_interval(90, 100)
        self.assertGreaterEqual(lower, 0.0)
        self.assertLessEqual(upper, 1.0)
        self.assertLess(lower, upper)
        self.assertEqual((lower, upper), _wilson_interval(90, 100))

    def test_assessor_framing_removes_prompt_echo_from_completion(self) -> None:
        framed = _assessment_text("What?", "What?\n\nI cannot help.")
        self.assertEqual(framed, " Human: What? Assistant: I cannot help.")

    def test_output_path_is_rejected_inside_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            _prepare_output_path(Path(__file__).resolve().parents[1] / "assessment.json")

    def test_output_parent_is_created_owner_only(self) -> None:
        with TemporaryDirectory() as directory:
            path = _prepare_output_path(Path(directory) / "reports" / "assessment.json")
            self.assertEqual(path.parent.stat().st_mode & 0o077, 0)

    def test_scored_output_directory_is_owner_only_and_empty(self) -> None:
        with TemporaryDirectory() as directory:
            scored_dir = _prepare_scored_output_dir(Path(directory) / "scores")
            self.assertEqual(scored_dir.stat().st_mode & 0o077, 0)
            (scored_dir / "existing").write_text("x")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                _prepare_scored_output_dir(scored_dir)

    def test_transcript_rejects_completion_digest_tampering(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            raw_dir = root / "raw"
            raw_dir.mkdir(mode=0o700)
            question = "What is the answer?"
            completion = "A safe response."
            record = {
                "dataset": "harmful",
                "row_id": 0,
                "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                "completion": completion,
                "completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
            }
            transcript = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            transcript_path = raw_dir / "responses.jsonl"
            transcript_path.write_text(transcript)
            transcript_path.chmod(0o600)
            manifest = {
                "state_slice": STATE_SLICE,
                "schema_version": RAW_SCHEMA_VERSION,
                "model_checkpoint_digest": "a" * 64,
                "assessment_corpus_digest": CORPUS_SHA256,
                "contrast_corpus_digest": None,
                "contains_raw_completions": True,
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
                "rows_written": 1,
                "complete": True,
            }
            (raw_dir / "manifest.json").write_text(json.dumps(manifest))
            qwen_report = {
                "state_slice": STATE_SLICE,
                "model": {"weights_loaded": True, "checkpoint_digest": "a" * 64},
                "assessment": {
                    "corpus_sha256": CORPUS_SHA256,
                    "raw_output_custody": {
                        "transcript_sha256": manifest["transcript_sha256"],
                        "rows_written": 1,
                    },
                },
            }
            qwen_path = root / "qwen.json"
            qwen_path.write_text(json.dumps(qwen_report))
            rows = [{"id": 0, "question": question}]
            _load_transcript(raw_dir, qwen_path, rows, [])
            record["completion"] = "tampered"
            tampered_transcript = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            transcript_path.write_text(tampered_transcript)
            manifest["transcript_sha256"] = hashlib.sha256(tampered_transcript.encode()).hexdigest()
            (raw_dir / "manifest.json").write_text(json.dumps(manifest))
            qwen_report["assessment"]["raw_output_custody"]["transcript_sha256"] = manifest[
                "transcript_sha256"
            ]
            qwen_path.write_text(json.dumps(qwen_report))
            with self.assertRaisesRegex(ValueError, "completion digest mismatch"):
                _load_transcript(raw_dir, qwen_path, rows, [])


if __name__ == "__main__":
    unittest.main()
