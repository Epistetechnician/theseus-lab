"""Independent NumPy SmolLM-135M adapter for the local Theseus experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from gpt2_experiment import MANIFEST_PATH, file_digest, full_digest


CHECKPOINT_MANIFESTS = json.loads(MANIFEST_PATH.read_text())


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    checksum = hashlib.sha256()
    for name in sorted(tensors):
        checksum.update(name.encode())
        checksum.update(np.ascontiguousarray(tensors[name]).tobytes())
    return checksum.hexdigest()


class SmolLMCheckpoint:
    """Verify and load owner-custodied SmolLM tensors without model code."""

    def __init__(self, model_dir: Path, model_id: str | None = None) -> None:
        self.model_dir = model_dir
        self.model_id = model_id or model_dir.name
        if self.model_id not in CHECKPOINT_MANIFESTS:
            raise ValueError(f"no checked-in manifest for checkpoint: {self.model_id}")
        self.manifest = CHECKPOINT_MANIFESTS[self.model_id]
        if self.manifest.get("loader") != "safetensors_llama":
            raise ValueError(f"manifest loader is not SmolLM-compatible: {self.model_id}")
        self.revision = self.manifest["revision"]
        self.artifact_digests = self._verify_artifacts()
        self.config = json.loads((model_dir / "config.json").read_text())
        for key, expected in self.manifest["config"].items():
            if self.config.get(key) != expected:
                raise ValueError(f"checkpoint config mismatch for {key}: {self.config.get(key)!r}")
        with safe_open(str(model_dir / "model.safetensors"), framework="pt") as handle:
            self.tensors = {
                name: handle.get_tensor(name).detach().cpu().to(torch.float32).numpy()
                for name in handle.keys()
            }
        self._verify_tensor_shapes()

    def _verify_artifacts(self) -> dict[str, str]:
        actual: dict[str, str] = {}
        for name, expected in self.manifest["artifacts"].items():
            path = self.model_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint artifact: {path}")
            observed = file_digest(path)
            if observed != expected:
                raise ValueError(f"checkpoint digest mismatch for {name}")
            actual[name] = observed
        return actual

    def _verify_tensor_shapes(self) -> None:
        required = {"model.embed_tokens.weight"}
        for layer in range(int(self.config["num_hidden_layers"])):
            prefix = f"model.layers.{layer}"
            required.update(
                f"{prefix}.{suffix}"
                for suffix in (
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "mlp.gate_proj.weight",
                    "mlp.up_proj.weight",
                    "mlp.down_proj.weight",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                )
            )
        required.add("model.norm.weight")
        if not required.issubset(self.tensors):
            missing = sorted(required - set(self.tensors))
            raise ValueError(f"checkpoint tensor set is incomplete: {missing[:3]}")
        embeddings = self.tensors["model.embed_tokens.weight"]
        output_embeddings = self.tensors.get("lm_head.weight")
        if output_embeddings is not None and not np.array_equal(embeddings, output_embeddings):
            raise ValueError("SmolLM checkpoint input and output embeddings are not tied")
        if output_embeddings is None and not bool(self.config["tie_word_embeddings"]):
            raise ValueError("SmolLM checkpoint omits an untied output embedding")
        if embeddings.shape != (int(self.config["vocab_size"]), int(self.config["hidden_size"])):
            raise ValueError("SmolLM embedding shape does not match config")

    @property
    def runtime_id(self) -> str:
        import tokenizers
        import transformers

        return ":".join((
            f"python-{platform.python_version()}",
            f"numpy-{np.__version__}",
            f"safetensors-{__import__('safetensors').__version__}",
            f"tokenizers-{tokenizers.__version__}",
            f"transformers-{transformers.__version__}",
        ))

    @property
    def checkpoint_digest(self) -> str:
        return full_digest({
            "model_id": self.model_id,
            "revision": self.revision,
            "runtime_id": self.runtime_id,
            "artifacts": self.artifact_digests,
            "tensor_digest": tensor_digest(self.tensors),
        })


class NumpySmolLM:
    """Independent forward path matching the pinned Llama-family contract."""

    def __init__(self, checkpoint: SmolLMCheckpoint) -> None:
        self.tensors = checkpoint.tensors
        self.n_layer = int(checkpoint.config["num_hidden_layers"])
        self.n_head = int(checkpoint.config["num_attention_heads"])
        self.n_kv_head = int(checkpoint.config["num_key_value_heads"])
        self.n_embd = int(checkpoint.config["hidden_size"])
        self.ffn_dim = int(checkpoint.config["intermediate_size"])
        self.head_dim = self.n_embd // self.n_head
        if self.n_head % self.n_kv_head:
            raise ValueError("SmolLM query heads must divide evenly into KV groups")
        self.query_heads_per_kv = self.n_head // self.n_kv_head
        self.rms_norm_eps = float(checkpoint.config["rms_norm_eps"])
        self.rope_theta = float(checkpoint.config["rope_theta"])
        self.max_position_embeddings = int(checkpoint.config["max_position_embeddings"])

    def forward(
        self,
        token_ids: list[int],
        permutation: np.ndarray | None = None,
        permutation_layer: int = 0,
        ablate_channel: int | None = None,
        ablate_channel_layer: int = 0,
        ablate_channels: np.ndarray | None = None,
        replacement_channel: tuple[int, int] | None = None,
        attention_permutation: np.ndarray | None = None,
        attention_permutation_layer: int = 0,
        ablate_head_group: int | None = None,
        ablate_head_group_layer: int = 0,
        precision: str = "float32",
    ) -> np.ndarray:
        if not token_ids or len(token_ids) > int(self._config_max_positions()):
            raise ValueError("token sequence length is outside the checkpoint bounds")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        compute_dtype = np.float64 if precision == "float64" else np.float32
        embeddings = self.tensors["model.embed_tokens.weight"].astype(compute_dtype, copy=False)
        positions = np.arange(len(token_ids), dtype=np.int64)
        x = (embeddings[token_ids]).copy()
        for layer in range(self.n_layer):
            x = x + self._attention(
                x,
                layer,
                attention_permutation,
                attention_permutation_layer,
                ablate_head_group,
                ablate_head_group_layer,
            )
            x = x + self._mlp(
                x,
                layer,
                permutation,
                permutation_layer,
                ablate_channel,
                ablate_channel_layer,
                ablate_channels,
                replacement_channel,
            )
        x = self._rms_norm(x, self.tensors["model.norm.weight"])
        return x[-1] @ embeddings.T

    def mlp_activation_trace(self, token_ids: list[int]) -> list[np.ndarray]:
        """Return each layer's last-token gated MLP activation for channel selection."""

        if not token_ids or len(token_ids) > int(self._config_max_positions()):
            raise ValueError("token sequence length is outside the checkpoint bounds")
        embeddings = self.tensors["model.embed_tokens.weight"]
        x = embeddings[token_ids].copy()
        activations: list[np.ndarray] = []
        for layer in range(self.n_layer):
            x = x + self._attention(x, layer, None, 0, None, 0)
            x = x + self._mlp(
                x,
                layer,
                None,
                0,
                None,
                0,
                None,
                None,
                capture=activations,
            )
        return activations

    def _config_max_positions(self) -> int:
        return self.max_position_embeddings

    def _attention(
        self,
        x: np.ndarray,
        layer: int,
        permutation: np.ndarray | None,
        permutation_layer: int,
        ablate_head_group: int | None,
        ablate_head_group_layer: int,
    ) -> np.ndarray:
        prefix = f"model.layers.{layer}"
        normalized = self._rms_norm(
            x,
            self.tensors[f"{prefix}.input_layernorm.weight"],
        )
        q_weight = self.tensors[f"{prefix}.self_attn.q_proj.weight"]
        k_weight = self.tensors[f"{prefix}.self_attn.k_proj.weight"]
        v_weight = self.tensors[f"{prefix}.self_attn.v_proj.weight"]
        output_weight = self.tensors[f"{prefix}.self_attn.o_proj.weight"]
        if layer == permutation_layer and permutation is not None:
            q_indexes, kv_indexes = self._group_indexes(permutation)
            q_weight = q_weight[q_indexes, :]
            k_weight = k_weight[kv_indexes, :]
            v_weight = v_weight[kv_indexes, :]
            output_weight = output_weight[:, q_indexes]
        query = normalized @ q_weight.T
        key = normalized @ k_weight.T
        value = normalized @ v_weight.T
        query = query.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        key = key.reshape(-1, self.n_kv_head, self.head_dim).transpose(1, 0, 2)
        value = value.reshape(-1, self.n_kv_head, self.head_dim).transpose(1, 0, 2)
        query, key = self._apply_rope(query, key)
        key = np.repeat(key, self.query_heads_per_kv, axis=0)
        value = np.repeat(value, self.query_heads_per_kv, axis=0)
        scores = (query * (1.0 / math.sqrt(self.head_dim))) @ key.transpose(0, 2, 1)
        length = scores.shape[-1]
        causal_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores = np.where(causal_mask[None, :, :], -np.inf, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        context = probabilities @ value
        if layer == ablate_head_group_layer and ablate_head_group is not None:
            start = ablate_head_group * self.query_heads_per_kv
            context[start:start + self.query_heads_per_kv, :, :] = 0.0
        context = context.transpose(1, 0, 2).reshape(-1, self.n_embd)
        return context @ output_weight.T

    def _mlp(
        self,
        x: np.ndarray,
        layer: int,
        permutation: np.ndarray | None,
        permutation_layer: int,
        ablate_channel: int | None,
        ablate_channel_layer: int,
        ablate_channels: np.ndarray | None,
        replacement_channel: tuple[int, int] | None,
        capture: list[np.ndarray] | None = None,
    ) -> np.ndarray:
        prefix = f"model.layers.{layer}"
        normalized = self._rms_norm(
            x,
            self.tensors[f"{prefix}.post_attention_layernorm.weight"],
        )
        gate_weight = self.tensors[f"{prefix}.mlp.gate_proj.weight"]
        up_weight = self.tensors[f"{prefix}.mlp.up_proj.weight"]
        down_weight = self.tensors[f"{prefix}.mlp.down_proj.weight"]
        if layer == permutation_layer and permutation is not None:
            gate_weight = gate_weight[permutation, :]
            up_weight = up_weight[permutation, :]
            down_weight = down_weight[:, permutation]
        channels = ablate_channels
        if layer == ablate_channel_layer and (ablate_channel is not None or channels is not None):
            gate_weight = gate_weight.copy()
            up_weight = up_weight.copy()
            if channels is None:
                channels = np.asarray([ablate_channel])
            gate_weight[channels, :] = 0.0
            up_weight[channels, :] = 0.0
        if layer == ablate_channel_layer and replacement_channel is not None:
            target, source = replacement_channel
            gate_weight = gate_weight.copy()
            up_weight = up_weight.copy()
            gate_weight[target, :] = gate_weight[source, :]
            up_weight[target, :] = up_weight[source, :]
        gate = normalized @ gate_weight.T
        up = normalized @ up_weight.T
        activated = self._silu(gate) * up
        if capture is not None:
            capture.append(activated[-1].copy())
        return activated @ down_weight.T

    def gated_mlp_components(self, layer: int) -> list[np.ndarray]:
        prefix = f"model.layers.{layer}.mlp"
        return [
            self.tensors[f"{prefix}.gate_proj.weight"].T,
            self.tensors[f"{prefix}.up_proj.weight"].T,
            self.tensors[f"{prefix}.down_proj.weight"].T,
        ]

    def gqa_components(self, layer: int) -> list[np.ndarray]:
        prefix = f"model.layers.{layer}.self_attn"
        return [
            self.tensors[f"{prefix}.q_proj.weight"],
            self.tensors[f"{prefix}.k_proj.weight"],
            self.tensors[f"{prefix}.v_proj.weight"],
            self.tensors[f"{prefix}.o_proj.weight"],
        ]

    def _group_indexes(self, permutation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(permutation) != self.n_kv_head or sorted(permutation.tolist()) != list(range(self.n_kv_head)):
            raise ValueError("GQA permutation must permute all KV groups exactly once")
        query_indexes = np.concatenate([
            np.arange(
                group * self.query_heads_per_kv * self.head_dim,
                (group + 1) * self.query_heads_per_kv * self.head_dim,
            )
            for group in permutation
        ])
        kv_indexes = np.concatenate([
            np.arange(group * self.head_dim, (group + 1) * self.head_dim)
            for group in permutation
        ])
        return query_indexes.astype(np.int64), kv_indexes.astype(np.int64)

    def _apply_rope(self, query: np.ndarray, key: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inverse_frequencies = 1.0 / (
            self.rope_theta ** (np.arange(0, self.head_dim, 2, dtype=np.float64) / self.head_dim)
        )
        positions = np.arange(query.shape[1], dtype=np.float64)
        frequencies = positions[:, None] * inverse_frequencies[None, :]
        embeddings = np.concatenate((frequencies, frequencies), axis=-1)
        cosine = np.cos(embeddings).astype(query.dtype, copy=False)
        sine = np.sin(embeddings).astype(query.dtype, copy=False)
        return (
            query * cosine[None, :, :] + self._rotate_half(query) * sine[None, :, :],
            key * cosine[None, :, :] + self._rotate_half(key) * sine[None, :, :],
        )

    @staticmethod
    def _rotate_half(x: np.ndarray) -> np.ndarray:
        midpoint = x.shape[-1] // 2
        return np.concatenate((-x[..., midpoint:], x[..., :midpoint]), axis=-1)

    @staticmethod
    def _silu(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.exp(-x))

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + self.rms_norm_eps) * weight
