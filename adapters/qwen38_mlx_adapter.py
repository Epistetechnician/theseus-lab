"""Lazy local MLX adapter for the pinned Qwen3.8-27B conversion.

The adapter has no model or network side effects at import time. Loading and
generation are explicit calls so a caller can inspect custody and lock the
assessment contract before model execution begins.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_REPO_ID = "mlx-community/Qwen3.8-27B-4bit"
MODEL_REVISION = "10c35caafbb80f7dc6a7a432cdd11af10a6d4818"
MODEL_BASE_REPO_ID = "Qwen/Qwen3.8-27B"
MODEL_LICENSE = "apache-2.0"
STATE_SLICE = "theseus-lab-sandbox-v1"


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_digest(value: object) -> str:
    """Hash canonical JSON for receipt and contract binding."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ModelInventory:
    """Custody inventory captured before model execution."""

    repo_id: str
    revision: str
    base_model: str
    license: str
    checkpoint_digest: str
    tensor_count: int
    files: tuple[dict[str, object], ...]

    def to_payload(self) -> dict[str, object]:
        """Return an aggregate, JSON-serializable inventory."""

        return {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "base_model": self.base_model,
            "license": self.license,
            "checkpoint_digest": self.checkpoint_digest,
            "tensor_count": self.tensor_count,
            "files": list(self.files),
        }


class Qwen38MLXAdapter:
    """Explicit local provider adapter with lazy model loading."""

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self._model: Any | None = None
        self._processor: Any | None = None
        self.weights_loaded = False
        self.provider_calls = 0

    def _required_paths(self) -> tuple[Path, ...]:
        index_path = self.model_dir / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"missing model index: {index_path}")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_names = sorted(set(index.get("weight_map", {}).values()))
        if not shard_names:
            raise ValueError("model index has no weight shards")
        required = (
            self.model_dir / "config.json",
            self.model_dir / "tokenizer.json",
            self.model_dir / "tokenizer_config.json",
            self.model_dir / "processor_config.json",
            index_path,
            *(self.model_dir / name for name in shard_names),
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"incomplete Qwen custody bundle: {missing}")
        config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") not in {"qwen3_5", "qwen3_5_moe"}:
            raise ValueError(f"unexpected Qwen model_type: {config.get('model_type')!r}")
        return tuple(required)

    def inventory(self) -> ModelInventory:
        """Verify and hash the model-bearing custody bundle."""

        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")
        required = self._required_paths()
        file_records = []
        digest = hashlib.sha256()
        for path in sorted(set(required), key=lambda item: item.name):
            file_hash = sha256_file(path)
            size = path.stat().st_size
            relative = path.relative_to(self.model_dir).as_posix()
            file_records.append({"path": relative, "bytes": size, "sha256": file_hash})
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
        index = json.loads((self.model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
        return ModelInventory(
            repo_id=MODEL_REPO_ID,
            revision=MODEL_REVISION,
            base_model=MODEL_BASE_REPO_ID,
            license=MODEL_LICENSE,
            checkpoint_digest=digest.hexdigest(),
            tensor_count=len(index.get("weight_map", {})),
            files=tuple(file_records),
        )

    def load(self) -> ModelInventory:
        """Load real weights through the MLX runtime after custody checks."""

        inventory = self.inventory()
        manifest_path = self.model_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing model custody manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("state_slice") != STATE_SLICE:
            raise ValueError("model custody manifest has the wrong state slice")
        for key, expected in (
            ("repo_id", MODEL_REPO_ID),
            ("revision", MODEL_REVISION),
            ("base_model", MODEL_BASE_REPO_ID),
            ("checkpoint_digest", inventory.checkpoint_digest),
            ("tensor_count", inventory.tensor_count),
        ):
            if manifest.get(key) != expected:
                raise ValueError(f"model custody manifest mismatch for {key}")
        if manifest.get("files") != list(inventory.files):
            raise ValueError("model custody manifest file inventory mismatch")
        import mlx_vlm

        self._model, self._processor = mlx_vlm.load(
            str(self.model_dir),
            lazy=False,
            strict=True,
        )
        self.weights_loaded = True
        return inventory

    def generate(self, question: str, *, max_tokens: int, temperature: float) -> dict[str, object]:
        """Call the local model and return only the generated observation."""

        if self._model is None or self._processor is None:
            raise RuntimeError("load() must succeed before generate()")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative")

        import mlx_vlm

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        result = mlx_vlm.generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=False,
        )
        self.provider_calls += 1
        text = str(getattr(result, "text", ""))
        return {
            "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "prompt_tokens": getattr(result, "prompt_tokens", None),
            "generation_tokens": getattr(result, "generation_tokens", None),
            "generation_tps": getattr(result, "generation_tps", None),
            "peak_memory": getattr(result, "peak_memory", None),
            "finish_reason": getattr(result, "finish_reason", None),
        }
