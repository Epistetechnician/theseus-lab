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


MANIFEST_PATH = Path(__file__).with_name("checkpoint_manifests.json")
CHECKPOINT_MANIFESTS = json.loads(MANIFEST_PATH.read_text())
VISIBLE_PROMPTS = (
    "The capital of France is",
    "The opposite of hot is",
    "The largest planet is",
    "A common color of grass is",
    "The first month of the year is",
    "Birds can usually",
    "Three times three equals",
    "The sun rises in the",
)
HELDOUT_PROMPTS = (
    "A triangle has three",
    "Water freezes at",
    "A year has twelve",
    "The chemical symbol for water is",
    "Before noon comes",
    "A mammal breathes",
    "The Pacific is an",
    "When rain falls, the ground gets",
    "A square has four",
    "A week has seven",
    "A day has twenty-four",
    "A dozen equals",
    "The opposite of up is",
    "The opposite of day is",
    "Ice is usually",
    "Fire is usually",
    "An octagon has eight",
    "A pentagon has five",
    "One hundred cents make a",
    "January follows",
    "Winter follows",
    "Red and blue make",
    "A bee makes",
    "A fish lives in",
    "A bird has two",
    "A human has two",
    "Normal human body temperature is",
    "The Earth orbits the",
    "The Moon orbits the",
    "The Sun is a",
    "Plants need light to",
    "A right angle has",
    "A minute has sixty",
    "An hour has sixty",
    "A kilogram is a unit of",
    "A meter has one hundred",
    "Primary colors include red, blue, and",
    "The alphabet begins with",
    "A rainbow has seven",
    "A spider has eight",
    "A cat says",
    "A dog says",
    "An infant is a young",
    "Liquid water becomes solid when it",
    "Solid water is called",
    "A compass points toward",
    "North is opposite",
    "East is opposite",
    "A map shows",
    "A library contains",
    "A recipe may list",
    "A clock measures",
    "A thermometer measures",
    "A keyboard has",
    "A book has",
    "A sentence ends with",
    "A question ends with",
    "A noun names",
    "A verb can express",
    "A planet is a",
    "An island is land surrounded by",
    "A desert receives little",
    "A forest contains many",
    "A cloud is made of",
)
PERMUTATION_SHIFT = 17
INTERVENTION_CHANNEL = 17
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260914


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
    def __init__(self, model_dir: Path, model_id: str | None = None) -> None:
        self.model_dir = model_dir
        self.model_id = model_id or model_dir.name
        if self.model_id not in CHECKPOINT_MANIFESTS:
            raise ValueError(f"no checked-in manifest for checkpoint: {self.model_id}")
        self.manifest = CHECKPOINT_MANIFESTS[self.model_id]
        self.revision = self.manifest["revision"]
        self.artifact_digests = self._verify_artifacts()
        self.config = json.loads((model_dir / "config.json").read_text())
        for key, expected in self.manifest["config"].items():
            if self.config.get(key) != expected:
                raise ValueError(f"checkpoint config mismatch for {key}: {self.config.get(key)!r}")
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        with safe_open(str(model_dir / "model.safetensors"), framework="np") as handle:
            raw_tensors = {
                name: np.asarray(handle.get_tensor(name), dtype=np.float32)
                for name in handle.keys()
            }
        prefix = "transformer." if "transformer.wte.weight" in raw_tensors else ""
        self.tensors = {
            name[len(prefix):] if prefix and name.startswith(prefix) else name: tensor
            for name, tensor in raw_tensors.items()
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
        required = {"wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"}
        required.update(
            f"h.{layer}.{suffix}"
            for layer in range(int(self.config["n_layer"]))
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
            "model_id": self.model_id,
            "revision": self.revision,
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
        if not token_ids or len(token_ids) > self.tensors["wpe.weight"].shape[0]:
            raise ValueError("token sequence length is outside the checkpoint bounds")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be float32 or float64")
        compute_dtype = np.float64 if precision == "float64" else np.float32
        positions = np.arange(len(token_ids))
        x = (
            self.tensors["wte.weight"][token_ids]
            + self.tensors["wpe.weight"][positions]
        ).astype(compute_dtype, copy=False)
        for layer in range(self.n_layer):
            prefix = f"h.{layer}"
            x = x + self._attention(
                x,
                prefix,
                attention_permutation,
                attention_permutation_layer,
                ablate_head,
                ablate_head_layer,
                layer,
            )
            x = x + self._mlp(
                x,
                prefix,
                permutation,
                permutation_layer,
                ablate_channel,
                ablate_channel_layer,
                ablate_channels,
                replacement_channel,
                layer,
            )
        x = self._layer_norm(x, self.tensors["ln_f.weight"], self.tensors["ln_f.bias"])
        return x[-1] @ self.tensors["wte.weight"].T

    def _attention(
        self,
        x: np.ndarray,
        prefix: str,
        permutation: np.ndarray | None,
        permutation_layer: int,
        ablate_head: int | None,
        ablate_head_layer: int,
        layer: int,
    ) -> np.ndarray:
        normalized = self._layer_norm(
            x,
            self.tensors[f"{prefix}.ln_1.weight"],
            self.tensors[f"{prefix}.ln_1.bias"],
        )
        qkv_weight = self.tensors[f"{prefix}.attn.c_attn.weight"]
        qkv_bias = self.tensors[f"{prefix}.attn.c_attn.bias"]
        output_weight = self.tensors[f"{prefix}.attn.c_proj.weight"]
        if layer == permutation_layer and permutation is not None:
            head_indexes = np.arange(self.n_head) * self.head_dim
            head_indexes = head_indexes[:, None] + np.arange(self.head_dim)
            order = np.concatenate((
                head_indexes[permutation].reshape(-1),
                (self.n_embd + head_indexes[permutation]).reshape(-1),
                (2 * self.n_embd + head_indexes[permutation]).reshape(-1),
            ))
            qkv_weight = qkv_weight[:, order]
            qkv_bias = qkv_bias[order]
            output_weight = output_weight[head_indexes[permutation].reshape(-1), :]
        qkv = normalized @ qkv_weight
        qkv = qkv + qkv_bias
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
        if layer == ablate_head_layer and ablate_head is not None:
            context[ablate_head, :, :] = 0.0
        context = context.transpose(1, 0, 2).reshape(-1, self.n_embd)
        return context @ output_weight + self.tensors[f"{prefix}.attn.c_proj.bias"]

    def _mlp(
        self,
        x: np.ndarray,
        prefix: str,
        permutation: np.ndarray | None,
        permutation_layer: int,
        ablate_channel: int | None,
        ablate_channel_layer: int,
        ablate_channels: np.ndarray | None,
        replacement_channel: tuple[int, int] | None,
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
        if layer == permutation_layer and permutation is not None:
            fc_weight = fc_weight[:, permutation]
            fc_bias = fc_bias[permutation]
            proj_weight = proj_weight[permutation, :]
        if layer == ablate_channel_layer and (ablate_channel is not None or ablate_channels is not None):
            fc_weight = fc_weight.copy()
            fc_bias = fc_bias.copy()
            channels = ablate_channels
            if channels is None:
                channels = np.asarray([ablate_channel])
            fc_weight[:, channels] = 0.0
            fc_bias[channels] = 0.0
        if layer == ablate_channel_layer and replacement_channel is not None:
            target, source = replacement_channel
            fc_weight = fc_weight.copy()
            fc_bias = fc_bias.copy()
            fc_weight[:, target] = fc_weight[:, source]
            fc_bias[target] = fc_bias[source]
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


def bootstrap_ci(values: Iterable[float], seed_offset: int) -> list[float]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if not len(values_array):
        raise ValueError("cannot bootstrap an empty metric")
    generator = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    indexes = generator.integers(0, len(values_array), size=(BOOTSTRAP_RESAMPLES, len(values_array)))
    means = values_array[indexes].mean(axis=1)
    return [
        round(float(np.quantile(means, 0.025)), 8),
        round(float(np.quantile(means, 0.975)), 8),
    ]


def compare_logits(before: Iterable[np.ndarray], after: Iterable[np.ndarray]) -> dict[str, object]:
    pairs = list(zip(before, after))
    top1 = [int(np.argmax(left) == np.argmax(right)) for left, right in pairs]
    relative_l2 = [
        float(np.linalg.norm(left - right) / max(np.linalg.norm(left), 1e-12))
        for left, right in pairs
    ]
    cosine = [
        float(np.clip(
            np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-12),
            -1.0,
            1.0,
        ))
        for left, right in pairs
    ]
    return {
        "prompt_count": len(pairs),
        "top1_agreement": round(sum(top1) / len(top1), 6),
        "top1_agreement_ci95": bootstrap_ci(top1, 1),
        "mean_relative_l2": round(sum(relative_l2) / len(relative_l2), 8),
        "mean_relative_l2_ci95": bootstrap_ci(relative_l2, 2),
        "mean_logit_cosine": round(sum(cosine) / len(cosine), 8),
        "mean_logit_cosine_ci95": bootstrap_ci(cosine, 3),
        "baseline_top1_ids": [int(np.argmax(logits)) for logits in before],
        "variant_top1_ids": [int(np.argmax(logits)) for logits in after],
    }


def vector_similarity(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    cosine = np.clip(
        np.dot(before, after) / max(np.linalg.norm(before) * np.linalg.norm(after), 1e-12),
        -1.0,
        1.0,
    )
    return {
        "relative_l2": round(float(np.linalg.norm(before - after) / max(np.linalg.norm(before), 1e-12)), 8),
        "absolute_l2": round(float(np.linalg.norm(before - after)), 8),
        "max_abs": round(float(np.max(np.abs(before - after))), 8),
        "cosine": round(float(cosine), 8),
    }


def run(model_dir: Path, model_id: str | None = None) -> dict[str, object]:
    checkpoint = GPT2Checkpoint(model_dir, model_id)
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
    baseline_effect = model.forward(reference_ids, precision="float64") - model.forward(
        reference_ids,
        ablate_channel=INTERVENTION_CHANNEL,
        precision="float64",
    )
    variant_normal = model.forward(
        reference_ids,
        permutation=permutation,
        precision="float64",
    )
    same_label_effect = variant_normal - model.forward(
        reference_ids,
        permutation=permutation,
        ablate_channel=INTERVENTION_CHANNEL,
        precision="float64",
    )
    aligned_effect = variant_normal - model.forward(
        reference_ids,
        permutation=permutation,
        ablate_channel=int(inverse_permutation[INTERVENTION_CHANNEL]),
        precision="float64",
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
        "claim_ceiling": "LOCAL_GPT2_FAMILY_TRANSFORMATION_OBSERVATION_ONLY",
        "checkpoint": {
            "weights_loaded": True,
            "model_id": checkpoint.model_id,
            "revision": checkpoint.revision,
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
    parser.add_argument("--model-id", type=str)
    args = parser.parse_args()
    print(json.dumps(run(args.model_dir, args.model_id), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
