"""Local GPT-2 Theseus experiment using only caller-owned checkpoint bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
from safetensors import safe_open
from tokenizers import Tokenizer


EXPECTED_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
EXPECTED_ARTIFACTS = {
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
    "generation_config.json": "ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "model.safetensors": "248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707",
    "tokenizer.json": "8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6",
    "tokenizer_config.json": "5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}
VISIBLE_PROMPTS = (
    "The capital of France is",
    "The opposite of hot is",
)
HELDOUT_PROMPTS = (
    "A triangle has three",
    "Water freezes at",
)
PERMUTATION_SHIFT = 17
INTERVENTION_CHANNEL = 17


def full_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def tensor_digest(tensors: dict[str, np.ndarray]) -> str:
    checksum = hashlib.sha256()
    for name in sorted(tensors):
        checksum.update(name.encode())
        checksum.update(np.ascontiguousarray(tensors[name]).tobytes())
    return checksum.hexdigest()


class GPT2Checkpoint:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.artifact_digests = self._verify_artifacts()
        self.config = json.loads((model_dir / "config.json").read_text())
        if self.config.get("model_type") != "gpt2":
            raise ValueError("checkpoint is not GPT-2")
        if self.config.get("n_layer") != 12 or self.config.get("n_head") != 12:
            raise ValueError("checkpoint dimensions differ from the pinned GPT-2 target")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        with safe_open(str(model_dir / "model.safetensors"), framework="np") as handle:
            self.tensors = {
                name: np.asarray(handle.get_tensor(name), dtype=np.float32)
                for name in handle.keys()
            }
        self._verify_tensor_shapes()

    def _verify_artifacts(self) -> dict[str, str]:
        actual: dict[str, str] = {}
        for name, expected in EXPECTED_ARTIFACTS.items():
            path = self.model_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"missing checkpoint artifact: {path}")
            observed = file_digest(path)
            if observed != expected:
                raise ValueError(f"checkpoint digest mismatch for {name}")
            actual[name] = observed
        return actual

    def _verify_tensor_shapes(self) -> None:
        required = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
        required.update(
            f"h.{layer}.{suffix}"
            for layer in range(12)
            for suffix in (
                "ln_1.weight",
                "ln_1.bias",
                "attn.c_attn.weight",
                "attn.c_attn.bias",
                "attn.c_proj.weight",
                "attn.c_proj.bias",
                "ln_2.weight",
                "ln_2.bias",
                "mlp.c_fc.weight",
                "mlp.c_fc.bias",
                "mlp.c_proj.weight",
                "mlp.c_proj.bias",
            )
        )
        if not required.issubset(self.tensors):
            raise ValueError("checkpoint tensor set is incomplete")

    @property
    def runtime_id(self) -> str:
        import safetensors
        import tokenizers

        return ":".join((
            f"python-{platform.python_version()}",
            f"numpy-{np.__version__}",
            f"safetensors-{safetensors.__version__}",
            f"tokenizers-{tokenizers.__version__}",
        ))

    @property
    def checkpoint_digest(self) -> str:
        return full_digest({
            "model_id": "gpt2",
            "revision": EXPECTED_REVISION,
            "runtime_id": self.runtime_id,
            "artifacts": self.artifact_digests,
            "tensor_digest": tensor_digest(self.tensors),
        })


class NumpyGPT2:
    def __init__(self, checkpoint: GPT2Checkpoint) -> None:
        self.tensors = checkpoint.tensors
        self.n_layer = int(checkpoint.config["n_layer"])
        self.n_head = int(checkpoint.config["n_head"])
        self.n_embd = int(checkpoint.config["n_embd"])
        self.head_dim = self.n_embd // self.n_head

    def forward(
        self,
        token_ids: list[int],
        permutation: np.ndarray | None = None,
        ablate_channel: int | None = None,
    ) -> np.ndarray:
        if not token_ids or len(token_ids) > self.tensors["wpe.weight"].shape[0]:
            raise ValueError("token sequence length is outside the checkpoint bounds")
        positions = np.arange(len(token_ids))
        x = self.tensors["wte.weight"][token_ids] + self.tensors["wpe.weight"][positions]
        for layer in range(self.n_layer):
            prefix = f"h.{layer}"
            x = x + self._attention(x, prefix)
            x = x + self._mlp(x, prefix, permutation, ablate_channel, layer)
        x = self._layer_norm(x, self.tensors["ln_f.weight"], self.tensors["ln_f.bias"])
        return x[-1] @ self.tensors["wte.weight"].T

    def _attention(self, x: np.ndarray, prefix: str) -> np.ndarray:
        normalized = self._layer_norm(
            x,
            self.tensors[f"{prefix}.ln_1.weight"],
            self.tensors[f"{prefix}.ln_1.bias"],
        )
        qkv = normalized @ self.tensors[f"{prefix}.attn.c_attn.weight"]
        qkv = qkv + self.tensors[f"{prefix}.attn.c_attn.bias"]
        query, key, value = np.split(qkv, 3, axis=-1)
        query = query.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        key = key.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        value = value.reshape(-1, self.n_head, self.head_dim).transpose(1, 0, 2)
        scores = query @ key.transpose(0, 2, 1) / math.sqrt(self.head_dim)
        length = scores.shape[-1]
        causal_mask = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores = np.where(causal_mask[None, :, :], -1e4, scores)
        scores = scores - np.max(scores, axis=-1, keepdims=True)
        probabilities = np.exp(scores)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        context = probabilities @ value
        context = context.transpose(1, 0, 2).reshape(-1, self.n_embd)
        return context @ self.tensors[f"{prefix}.attn.c_proj.weight"] + self.tensors[f"{prefix}.attn.c_proj.bias"]

    def _mlp(
        self,
        x: np.ndarray,
        prefix: str,
        permutation: np.ndarray | None,
        ablate_channel: int | None,
        layer: int,
    ) -> np.ndarray:
        normalized = self._layer_norm(
            x,
            self.tensors[f"{prefix}.ln_2.weight"],
            self.tensors[f"{prefix}.ln_2.bias"],
        )
        fc_weight = self.tensors[f"{prefix}.mlp.c_fc.weight"]
        fc_bias = self.tensors[f"{prefix}.mlp.c_fc.bias"]
        proj_weight = self.tensors[f"{prefix}.mlp.c_proj.weight"]
        if layer == 0 and permutation is not None:
            fc_weight = fc_weight[:, permutation]
            fc_bias = fc_bias[permutation]
            proj_weight = proj_weight[permutation, :]
        if layer == 0 and ablate_channel is not None:
            fc_weight = fc_weight.copy()
            fc_bias = fc_bias.copy()
            fc_weight[:, ablate_channel] = 0.0
            fc_bias[ablate_channel] = 0.0
        activated = self._gelu_new(normalized @ fc_weight + fc_bias)
        return activated @ proj_weight + self.tensors[f"{prefix}.mlp.c_proj.bias"]

    @staticmethod
    def _layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(variance + 1e-5) * weight + bias

    @staticmethod
    def _gelu_new(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1.0 + np.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        ))


def compare_logits(before: Iterable[np.ndarray], after: Iterable[np.ndarray]) -> dict[str, object]:
    pairs = list(zip(before, after))
    top1 = [int(np.argmax(left) == np.argmax(right)) for left, right in pairs]
    relative_l2 = [
        float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
        for left, right in pairs
    ]
    cosine = [
        float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12))
        for left, right in pairs
    ]
    return {
        "prompt_count": len(pairs),
        "top1_agreement": round(sum(top1) / len(top1), 6),
        "mean_relative_l2": round(sum(relative_l2) / len(relative_l2), 8),
        "mean_logit_cosine": round(sum(cosine) / len(cosine), 8),
        "baseline_top1_ids": [int(np.argmax(logits)) for logits in before],
        "variant_top1_ids": [int(np.argmax(logits)) for logits in after],
    }


def vector_similarity(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    return {
        "relative_l2": round(float(np.linalg.norm(before - after) / max(np.linalg.norm(before), 1e-12)), 8),
        "cosine": round(float(np.dot(before, after) / max(np.linalg.norm(before) * np.linalg.norm(after), 1e-12)), 8),
    }


def run(model_dir: Path) -> dict[str, object]:
    checkpoint = GPT2Checkpoint(model_dir)
    model = NumpyGPT2(checkpoint)
    permutation = np.roll(np.arange(model.tensors["h.0.mlp.c_fc.weight"].shape[1]), PERMUTATION_SHIFT)
    inverse_permutation = np.argsort(permutation)
    prompts = VISIBLE_PROMPTS + HELDOUT_PROMPTS
    tokenized = [checkpoint.tokenizer.encode(prompt).ids for prompt in prompts]
    baseline = [model.forward(ids) for ids in tokenized]
    permuted = [model.forward(ids, permutation=permutation) for ids in tokenized]
    visible_count = len(VISIBLE_PROMPTS)
    visible_metrics = compare_logits(baseline[:visible_count], permuted[:visible_count])
    heldout_metrics = compare_logits(baseline[visible_count:], permuted[visible_count:])

    selected = model.tensors["h.0.mlp.c_fc.weight"]
    selected_bias = model.tensors["h.0.mlp.c_fc.bias"]
    selected_proj = model.tensors["h.0.mlp.c_proj.weight"]
    permuted_selected = selected[:, permutation]
    permuted_bias = selected_bias[permutation]
    permuted_proj = selected_proj[permutation, :]
    original_mechanism_digest = tensor_digest({
        "c_fc.weight": selected,
        "c_fc.bias": selected_bias,
        "c_proj.weight": selected_proj,
    })
    permuted_mechanism_digest = tensor_digest({
        "c_fc.weight": permuted_selected,
        "c_fc.bias": permuted_bias,
        "c_proj.weight": permuted_proj,
    })
    changed = sum(np.count_nonzero(left != right) for left, right in (
        (selected, permuted_selected),
        (selected_bias, permuted_bias),
        (selected_proj, permuted_proj),
    ))
    total = selected.size + selected_bias.size + selected_proj.size

    reference_ids = tokenized[0]
    baseline_effect = model.forward(reference_ids) - model.forward(
        reference_ids,
        ablate_channel=INTERVENTION_CHANNEL,
    )
    same_label_effect = model.forward(reference_ids, permutation=permutation) - model.forward(
        reference_ids,
        permutation=permutation,
        ablate_channel=INTERVENTION_CHANNEL,
    )
    aligned_effect = model.forward(reference_ids, permutation=permutation) - model.forward(
        reference_ids,
        permutation=permutation,
        ablate_channel=int(inverse_permutation[INTERVENTION_CHANNEL]),
    )

    structurally_equivalent = (
        visible_metrics["top1_agreement"] == 1.0
        and heldout_metrics["top1_agreement"] == 1.0
        and visible_metrics["mean_relative_l2"] < 1e-4
        and heldout_metrics["mean_relative_l2"] < 1e-4
        and original_mechanism_digest != permuted_mechanism_digest
    )
    return {
        "status": "STRUCTURAL_FUNCTIONAL_SEPARATION_OBSERVED" if structurally_equivalent else "INCONCLUSIVE",
        "claim_ceiling": "LOCAL_GPT2_REPARAMETERIZATION_OBSERVATION_ONLY",
        "checkpoint": {
            "weights_loaded": True,
            "model_id": "gpt2",
            "revision": EXPECTED_REVISION,
            "model_dir": str(model_dir),
            "artifact_digests": checkpoint.artifact_digests,
            "checkpoint_digest": checkpoint.checkpoint_digest,
            "tensor_count": len(checkpoint.tensors),
            "runtime_id": checkpoint.runtime_id,
        },
        "transformation": {
            "kind": "layer_0_mlp_channel_permutation",
            "channel_width": int(selected.shape[1]),
            "shift": PERMUTATION_SHIFT,
            "permutation_digest": full_digest(permutation.tolist()),
            "selected_parameter_digest_before": original_mechanism_digest,
            "selected_parameter_digest_after": permuted_mechanism_digest,
            "changed_parameter_fraction": round(changed / total, 8),
        },
        "behavior": {
            "visible": visible_metrics,
            "heldout": heldout_metrics,
        },
        "intervention": {
            "layer": 0,
            "baseline_channel": INTERVENTION_CHANNEL,
            "aligned_variant_channel": int(inverse_permutation[INTERVENTION_CHANNEL]),
            "same_label_effect_similarity": vector_similarity(baseline_effect, same_label_effect),
            "aligned_effect_similarity": vector_similarity(baseline_effect, aligned_effect),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
