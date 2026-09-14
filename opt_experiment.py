"""Independent NumPy OPT-125M adapter for the local Theseus experiment."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np
import torch

from gpt2_experiment import MANIFEST_PATH, full_digest, file_digest


CHECKPOINT_MANIFESTS = json.loads(MANIFEST_PATH.read_text())


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    checksum = hashlib.sha256()
    for name in sorted(tensors):
        checksum.update(name.encode())
        checksum.update(np.ascontiguousarray(tensors[name]).tobytes())
    return checksum.hexdigest()


class OPTCheckpoint:
    """Verify and load an owner-custodied OPT checkpoint without model code."""

    def __init__(self, model_dir: Path, model_id: str | None = None) -> None:
        self.model_dir = model_dir
        self.model_id = model_id or model_dir.name
        if self.model_id not in CHECKPOINT_MANIFESTS:
            raise ValueError(f"no checked-in manifest for checkpoint: {self.model_id}")
        self.manifest = CHECKPOINT_MANIFESTS[self.model_id]
        if self.manifest.get("loader") != "pytorch_state_dict":
            raise ValueError(f"manifest loader is not OPT-compatible: {self.model_id}")
        self.revision = self.manifest["revision"]
        self.artifact_digests = self._verify_artifacts()
        self.config = json.loads((model_dir / "config.json").read_text())
        for key, expected in self.manifest["config"].items():
            if self.config.get(key) != expected:
                raise ValueError(f"checkpoint config mismatch for {key}: {self.config.get(key)!r}")
        loaded = torch.load(
            model_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded, dict):
            raise ValueError("OPT checkpoint is not a state dictionary")
        self.tensors = {
            name: np.asarray(value.detach().cpu().numpy(), dtype=np.float32)
            for name, value in loaded.items()
            if isinstance(value, torch.Tensor)
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
        required = {
            "model.decoder.embed_tokens.weight",
            "model.decoder.embed_positions.weight",
            "model.decoder.final_layer_norm.weight",
            "model.decoder.final_layer_norm.bias",
            "lm_head.weight",
        }
        for layer in range(int(self.config["num_hidden_layers"])):
            prefix = f"model.decoder.layers.{layer}"
            required.update(
                f"{prefix}.{suffix}"
                for suffix in (
                    "self_attn.k_proj.weight",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.weight",
                    "self_attn.v_proj.bias",
                    "self_attn.q_proj.weight",
                    "self_attn.q_proj.bias",
                    "self_attn.out_proj.weight",
                    "self_attn.out_proj.bias",
                    "self_attn_layer_norm.weight",
                    "self_attn_layer_norm.bias",
                    "fc1.weight",
                    "fc1.bias",
                    "fc2.weight",
                    "fc2.bias",
                    "final_layer_norm.weight",
                    "final_layer_norm.bias",
                )
            )
        if not required.issubset(self.tensors):
            missing = sorted(required - set(self.tensors))
            raise ValueError(f"checkpoint tensor set is incomplete: {missing[:3]}")
        embeddings = self.tensors["model.decoder.embed_tokens.weight"]
        if not np.array_equal(embeddings, self.tensors["lm_head.weight"]):
            raise ValueError("OPT checkpoint input and output embeddings are not tied")
        if embeddings.shape != (int(self.config["vocab_size"]), int(self.config["word_embed_proj_dim"])):
            raise ValueError("OPT embedding shape does not match config")

    @property
    def runtime_id(self) -> str:
        import transformers
        import tokenizers

        return ":".join((
            f"python-{platform.python_version()}",
            f"numpy-{np.__version__}",
            f"torch-{torch.__version__}",
            f"transformers-{transformers.__version__}",
            f"tokenizers-{tokenizers.__version__}",
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


class NumpyOPT:
    """Independent forward path matching the pinned OPT decoder contract."""

    def __init__(self, checkpoint: OPTCheckpoint) -> None:
        self.tensors = checkpoint.tensors
        self.n_layer = int(checkpoint.config["num_hidden_layers"])
        self.n_head = int(checkpoint.config["num_attention_heads"])
        self.n_embd = int(checkpoint.config["hidden_size"])
        self.ffn_dim = int(checkpoint.config["ffn_dim"])
        self.head_dim = self.n_embd // self.n_head
        self.position_offset = 2
        self.layer_norm_eps = 1e-5

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
        ablate_head: int | None = None,
        ablate_head_layer: int = 0,
        precision: str = "float32",
    ) -> np.ndarray:
        position_count = self.tensors["model.decoder.embed_positions.weight"].shape[0]
        if not token_ids or len(token_ids) + self.position_offset > position_count:
            raise ValueError("token sequence length is outside the checkpoint bounds")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        compute_dtype = np.float64 if precision == "float64" else np.float32
        positions = np.arange(len(token_ids), dtype=np.int64) + self.position_offset
        embeddings = self.tensors["model.decoder.embed_tokens.weight"]
        position_embeddings = self.tensors["model.decoder.embed_positions.weight"]
        x = (embeddings[token_ids] + position_embeddings[positions]).astype(compute_dtype, copy=False)
        for layer in range(self.n_layer):
            x = x + self._attention(
                x,
                layer,
                attention_permutation,
                attention_permutation_layer,
                ablate_head,
                ablate_head_layer,
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
        x = self._layer_norm(
            x,
            self.tensors["model.decoder.final_layer_norm.weight"],
            self.tensors["model.decoder.final_layer_norm.bias"],
        )
        return x[-1] @ embeddings.astype(compute_dtype, copy=False).T

    def mlp_activation_trace(self, token_ids: list[int]) -> list[np.ndarray]:
        """Return each layer's last-token post-ReLU activation for channel selection."""

        position_count = self.tensors["model.decoder.embed_positions.weight"].shape[0]
        if not token_ids or len(token_ids) + self.position_offset > position_count:
            raise ValueError("token sequence length is outside the checkpoint bounds")
        positions = np.arange(len(token_ids), dtype=np.int64) + self.position_offset
        embeddings = self.tensors["model.decoder.embed_tokens.weight"]
        position_embeddings = self.tensors["model.decoder.embed_positions.weight"]
        x = embeddings[token_ids] + position_embeddings[positions]
        activations: list[np.ndarray] = []
        for layer in range(self.n_layer):
            x = x + self._attention(x, layer, None, 0, None, 0)
            x = x + self._mlp(
                x, layer, None, 0, None, 0, None, None, capture=activations,
            )
        return activations

    def _attention(
        self,
        x: np.ndarray,
        layer: int,
        permutation: np.ndarray | None,
        permutation_layer: int,
        ablate_head: int | None,
        ablate_head_layer: int,
    ) -> np.ndarray:
        prefix = f"model.decoder.layers.{layer}"
        normalized = self._layer_norm(
            x,
            self.tensors[f"{prefix}.self_attn_layer_norm.weight"],
            self.tensors[f"{prefix}.self_attn_layer_norm.bias"],
        )
        q_weight = self.tensors[f"{prefix}.self_attn.q_proj.weight"]
        q_bias = self.tensors[f"{prefix}.self_attn.q_proj.bias"]
        k_weight = self.tensors[f"{prefix}.self_attn.k_proj.weight"]
        k_bias = self.tensors[f"{prefix}.self_attn.k_proj.bias"]
        v_weight = self.tensors[f"{prefix}.self_attn.v_proj.weight"]
        v_bias = self.tensors[f"{prefix}.self_attn.v_proj.bias"]
        output_weight = self.tensors[f"{prefix}.self_attn.out_proj.weight"]
        if layer == permutation_layer and permutation is not None:
            head_indexes = np.arange(self.n_head)[:, None] * self.head_dim + np.arange(self.head_dim)
            flat = head_indexes[permutation].reshape(-1)
            q_weight, q_bias = q_weight[flat, :], q_bias[flat]
            k_weight, k_bias = k_weight[flat, :], k_bias[flat]
            v_weight, v_bias = v_weight[flat, :], v_bias[flat]
            output_weight = output_weight[:, flat]
        query = normalized @ q_weight.T + q_bias
        key = normalized @ k_weight.T + k_bias
        value = normalized @ v_weight.T + v_bias
        query = query.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        key = key.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        value = value.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        query = query * (1.0 / math.sqrt(self.head_dim))
        scores = query @ key.transpose(0, 2, 1)
        length = scores.shape[-1]
        causal_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores = np.where(causal_mask[None, :, :], -np.inf, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        context = probabilities @ value
        if layer == ablate_head_layer and ablate_head is not None:
            context[ablate_head, :, :] = 0.0
        context = context.transpose(1, 0, 2).reshape(-1, self.n_embd)
        return context @ output_weight.T + self.tensors[f"{prefix}.self_attn.out_proj.bias"]

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
        prefix = f"model.decoder.layers.{layer}"
        normalized = self._layer_norm(
            x,
            self.tensors[f"{prefix}.final_layer_norm.weight"],
            self.tensors[f"{prefix}.final_layer_norm.bias"],
        )
        fc1_weight = self.tensors[f"{prefix}.fc1.weight"]
        fc1_bias = self.tensors[f"{prefix}.fc1.bias"]
        fc2_weight = self.tensors[f"{prefix}.fc2.weight"]
        if layer == permutation_layer and permutation is not None:
            fc1_weight = fc1_weight[permutation, :]
            fc1_bias = fc1_bias[permutation]
            fc2_weight = fc2_weight[:, permutation]
        channels = ablate_channels
        if layer == ablate_channel_layer and (ablate_channel is not None or channels is not None):
            fc1_weight = fc1_weight.copy()
            fc1_bias = fc1_bias.copy()
            if channels is None:
                channels = np.asarray([ablate_channel])
            fc1_weight[channels, :] = 0.0
            fc1_bias[channels] = 0.0
        if layer == ablate_channel_layer and replacement_channel is not None:
            target, source = replacement_channel
            fc1_weight = fc1_weight.copy()
            fc1_bias = fc1_bias.copy()
            fc1_weight[target, :] = fc1_weight[source, :]
            fc1_bias[target] = fc1_bias[source]
        activated = np.maximum(normalized @ fc1_weight.T + fc1_bias, 0.0)
        if capture is not None:
            capture.append(activated[-1].copy())
        return activated @ fc2_weight.T + self.tensors[f"{prefix}.fc2.bias"]

    def mlp_components(self, layer: int) -> list[np.ndarray]:
        prefix = f"model.decoder.layers.{layer}"
        return [
            self.tensors[f"{prefix}.fc1.weight"].T,
            self.tensors[f"{prefix}.fc1.bias"],
            self.tensors[f"{prefix}.fc2.weight"].T,
        ]

    def attention_components(self, layer: int) -> list[np.ndarray]:
        prefix = f"model.decoder.layers.{layer}.self_attn"
        qkv_weight = np.concatenate((
            self.tensors[f"{prefix}.q_proj.weight"].T,
            self.tensors[f"{prefix}.k_proj.weight"].T,
            self.tensors[f"{prefix}.v_proj.weight"].T,
        ), axis=1)
        qkv_bias = np.concatenate((
            self.tensors[f"{prefix}.q_proj.bias"],
            self.tensors[f"{prefix}.k_proj.bias"],
            self.tensors[f"{prefix}.v_proj.bias"],
        ))
        return [qkv_weight, qkv_bias, self.tensors[f"{prefix}.out_proj.weight"].T]

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + 1e-5) * weight + bias
