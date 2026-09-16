import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.qwen38_mlx_adapter import (
    MODEL_BASE_REPO_ID,
    MODEL_REPO_ID,
    MODEL_REVISION,
    STATE_SLICE,
    Qwen38MLXAdapter,
)
from qwen38_assessment import (
    CORPUS_ROWS,
    _prepare_output_path,
    _prepare_raw_output_dir,
    _raw_record,
    classify_refusal,
    load_rows,
)


class TestQwen38AssessmentContract(unittest.TestCase):
    @staticmethod
    def _fake_bundle(directory: Path) -> Qwen38MLXAdapter:
        (directory / "config.json").write_text(json.dumps({"model_type": "qwen3_5"}))
        (directory / "tokenizer.json").write_text("{}")
        (directory / "tokenizer_config.json").write_text("{}")
        (directory / "processor_config.json").write_text("{}")
        (directory / "model-00001-of-00003.safetensors").write_bytes(b"fake weights")
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"language_model.fake": "model-00001-of-00003.safetensors"}})
        )
        return Qwen38MLXAdapter(directory)

    def test_refusal_classifier_is_narrow_and_deterministic(self) -> None:
        self.assertEqual(classify_refusal("I can't assist with that request."), "refusal_style")
        self.assertEqual(classify_refusal("Here is a detailed answer."), "non_refusal_style")
        self.assertEqual(classify_refusal(""), "empty")

    def test_import_does_not_load_model(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = Qwen38MLXAdapter(Path(directory))
            self.assertFalse(adapter.weights_loaded)
            self.assertEqual(adapter.provider_calls, 0)

    def test_model_load_requires_custody_manifest(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = self._fake_bundle(Path(directory))
            with self.assertRaisesRegex(FileNotFoundError, "custody manifest"):
                adapter.load()

    def test_model_load_rejects_mismatched_custody_digest(self) -> None:
        with TemporaryDirectory() as directory:
            model_dir = Path(directory)
            adapter = self._fake_bundle(model_dir)
            inventory = adapter.inventory()
            manifest = {
                "state_slice": STATE_SLICE,
                "repo_id": MODEL_REPO_ID,
                "revision": MODEL_REVISION,
                "base_model": MODEL_BASE_REPO_ID,
                "checkpoint_digest": "0" * 64,
                "tensor_count": inventory.tensor_count,
                "files": list(inventory.files),
            }
            (model_dir / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "manifest mismatch for checkpoint_digest"):
                adapter.load()

    def test_corpus_loader_rejects_wrong_digest_before_parquet_read(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "not-the-corpus.parquet"
            path.write_bytes(b"not parquet")
            with self.assertRaisesRegex(ValueError, "corpus digest"):
                load_rows(path)

    def test_contract_constants_are_explicit(self) -> None:
        from qwen38_assessment import (
            CLASSIFIER_VERSION,
            CORPUS_REPO_ID,
            CORPUS_REVISION,
            MODEL_REVISION,
            TEMPERATURE,
        )

        self.assertEqual(CORPUS_REPO_ID, "LibrAI/do-not-answer")
        self.assertEqual(len(CORPUS_REVISION), 40)
        self.assertEqual(len(MODEL_REVISION), 40)
        self.assertEqual(TEMPERATURE, 0.0)
        self.assertTrue(CLASSIFIER_VERSION.startswith("refusal-lexicon-"))
        self.assertEqual(CORPUS_ROWS, 939)

    def test_output_path_is_rejected_inside_repository(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            _prepare_output_path(Path(__file__).resolve().parents[1] / "report.json")

    def test_raw_output_directory_is_owner_only_and_empty(self) -> None:
        with TemporaryDirectory() as directory:
            raw_dir = _prepare_raw_output_dir(Path(directory) / "raw")
            self.assertEqual(raw_dir.stat().st_mode & 0o077, 0)
            (raw_dir / "existing").write_text("x")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                _prepare_raw_output_dir(raw_dir)

    def test_raw_record_keeps_completion_outside_aggregate_observation(self) -> None:
        row = {"id": 3, "risk_area": "test", "question": "What?"}
        record = _raw_record(
            "harmful",
            row,
            {
                "text": "I cannot help.",
                "text_sha256": "hash",
                "prompt_tokens": 1,
                "generation_tokens": 2,
                "generation_tps": 3.0,
                "peak_memory": 4.0,
                "finish_reason": "stop",
            },
        )
        self.assertEqual(record["completion"], "I cannot help.")
        self.assertNotIn("risk_area", record)
        self.assertIn("question_sha256", record)


if __name__ == "__main__":
    unittest.main()
