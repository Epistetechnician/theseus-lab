"""Frozen local SCM input-intervention replacement qualification runner.

Generated prompts, weights, activations, predictions, and row-level metrics
are written outside the repository under owner-controlled custody.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import platform
import resource
import secrets
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

PROTOCOL_ID = "theseus-replacement-loss-v31-scm-interchange"
STATE_SLICE = "theseus-lab-sandbox-v1"
CONTINUATION_SLICE = "theseus-replacement-loss-v31-scm-interchange-v1"
ASSURANCE_SLICE = "global-experiment-assurance-lifecycle-v1"
REPO_ROOT = Path(__file__).resolve().parent
CUSTODY_BASE = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v31")
ALPHABET = ("A", "B", "C", "D")
CHAR_TO_ID = {char: index + 4 for index, char in enumerate(ALPHABET)}
VOCAB_SIZE = 8
BOS, SEP, EOS = 1, 2, 3
INPUT_LENGTH = 4
GUIDE_SEEDS = (3101, 3102, 3103)
ADAPT_PREFIX = ("A", "B")
CHALLENGE_PREFIX = ("C", "D")
INTERVENED_POSITIONS = (1, 2, 3)
GUIDE_STEPS = 1500
CANDIDATE_STEPS = 600
GUIDE_LR = 0.003
CANDIDATE_LR = 0.001
WEIGHT_DECAY = 0.0001
MAX_WALL_SECONDS = 1800
MAX_PEAK_RSS_BYTES = 4 * 1024**3
MAX_ARTIFACT_BYTES = 1024**3
EXPECTED_PYTHON = "3.12.13"
EXPECTED_NUMPY = "2.4.6"
EXPECTED_TORCH = "2.6.0"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def value_digest(value: Any) -> str:
    return sha256(canonical_json(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        if array.dtype.byteorder == ">" or (array.dtype.byteorder == "=" and sys.byteorder == "big"):
            array = array.byteswap().view(array.dtype.newbyteorder("<"))
        meta = {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        digest.update(canonical_json(meta))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def state_digest(module: nn.Module) -> str:
    arrays = {
        key: value.detach().cpu().numpy().astype("<f4", copy=False)
        for key, value in sorted(module.state_dict().items())
    }
    return array_digest(arrays)


def packet_files() -> list[dict[str, Any]]:
    paths = (
        Path("docs/replacement-loss-v31-scm-interchange.md"),
        Path("replacement_loss_v31_runner.py"),
        Path("tests/test_replacement_loss_v31_scm_interchange.py"),
        Path("requirements-v31.lock.txt"),
    )
    material = []
    for relative in paths:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen packet member missing: {relative}")
        material.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": file_digest(path)})
    return material


def verify_freeze() -> str:
    freeze_path = REPO_ROOT / "docs/replacement-loss-v31-freeze.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("V31 freeze manifest is missing")
    freeze = json.loads(freeze_path.read_text())
    material = packet_files()
    packet_sha = value_digest(material)
    if (
        freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("freeze_state") != "FROZEN_BEFORE_TASK_DATA_GENERATION"
        or freeze.get("packet_files") != material
        or freeze.get("packet_sha256") != packet_sha
    ):
        raise ValueError("V31 source packet does not match its frozen manifest")
    return packet_sha


def contexts() -> list[tuple[int, ...]]:
    return list(itertools.product(range(len(ALPHABET)), repeat=INPUT_LENGTH))


def reverse_truth(context: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(reversed(context))


def intervene(base: tuple[int, ...], source: tuple[int, ...], position: int) -> tuple[int, ...]:
    if len(base) != INPUT_LENGTH or len(source) != INPUT_LENGTH:
        raise ValueError("SCM contexts must have four variables")
    if position not in INTERVENED_POSITIONS:
        raise ValueError("V31 interventions are predeclared only for X_1, X_2, X_3")
    differences = [i for i, (left, right) in enumerate(zip(base, source)) if left != right]
    if differences != [position]:
        raise ValueError("source context must differ from base at exactly the intervened variable")
    counterfactual = list(base)
    counterfactual[position] = source[position]
    return tuple(counterfactual)


def source_for(base: tuple[int, ...], position: int) -> tuple[int, ...]:
    source = list(base)
    source[position] = (source[position] + 1) % len(ALPHABET)
    return tuple(source)


def encode_context(context: tuple[int, ...]) -> list[int]:
    text_ids = [CHAR_TO_ID[ALPHABET[index]] for index in context]
    output_ids = [CHAR_TO_ID[ALPHABET[index]] for index in reverse_truth(context)]
    full = [BOS, *text_ids, SEP, *output_ids, EOS]
    return full[:-1]


def task_arrays(items: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    encoded = np.asarray([encode_context(item) for item in items], dtype=np.int64)
    full = np.asarray(
        [
            [BOS, *[CHAR_TO_ID[ALPHABET[index]] for index in item], SEP,
             *[CHAR_TO_ID[ALPHABET[index]] for index in reverse_truth(item)], EOS]
            for item in items
        ],
        dtype=np.int64,
    )
    return encoded, full[:, 1:]


def endpoint_logit_slice(logits: torch.Tensor) -> torch.Tensor:
    return logits[:, 5:9, :]


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (part.transpose(1, 2) for part in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.output(attended.transpose(1, 2).contiguous().view(batch, length, width))


class MLP(nn.Module):
    def __init__(self, width: int = 32, hidden: int = 64) -> None:
        super().__init__()
        self.in_proj = nn.Linear(width, hidden)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(hidden, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(self.activation(self.in_proj(x)))


class ZeroMLP(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(32)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(32)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor, override_values: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        mlp_output = self.mlp(self.ln2(x))
        if override_values is not None:
            if override_values.shape != mlp_output.shape:
                raise ValueError("activation-rescue tensor has the wrong shape")
            mlp_output = override_values
        return x + mlp_output


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, 32)
        self.position_embedding = nn.Embedding(16, 32)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final_norm = nn.LayerNorm(32)
        self.lm_head = nn.Linear(32, VOCAB_SIZE, bias=False)

    def forward(self, token_ids: torch.Tensor, rescue_values: torch.Tensor | None = None) -> torch.Tensor:
        _, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None, :, :]
        for index, block in enumerate(self.blocks):
            x = block(x, rescue_values if index == 0 else None)
        return self.lm_head(self.final_norm(x))


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def environment_info() -> dict[str, Any]:
    packages = sorted(
        (dist.metadata["Name"], dist.version)
        for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")
    )
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__.split("+")[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "runtime_prefix": sys.prefix,
        "python_executable": sys.executable,
        "dependency_inventory": packages,
        "dependency_inventory_sha256": value_digest(packages),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
    }


def verify_runtime() -> dict[str, Any]:
    info = environment_info()
    expected = {
        "python": EXPECTED_PYTHON,
        "numpy": EXPECTED_NUMPY,
        "torch": EXPECTED_TORCH,
        "machine": "arm64",
        "device": "cpu",
        "torch_threads": 1,
    }
    mismatches = {key: {"expected": value, "actual": info.get(key)} for key, value in expected.items() if info.get(key) != value}
    if mismatches:
        raise RuntimeError(f"runtime contract mismatch: {mismatches}")
    return info


def model_inputs(items: list[tuple[int, ...]]) -> torch.Tensor:
    encoded, _ = task_arrays(items)
    return torch.as_tensor(encoded, dtype=torch.long)


def check_run_budget(started: float) -> None:
    elapsed = time.monotonic() - started
    rss = max_rss_bytes()
    if elapsed > MAX_WALL_SECONDS:
        raise TimeoutError(f"V31 wall-time budget exceeded: {elapsed:.1f}s")
    if rss > MAX_PEAK_RSS_BYTES:
        raise MemoryError(f"V31 peak-RSS budget exceeded: {rss} bytes")


def train_guide(seed: int, run_started: float) -> tuple[TinyDecoder, float, str]:
    set_determinism(seed)
    model = TinyDecoder().cpu()
    items = contexts()
    input_ids, targets = task_arrays(items)
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    labels = torch.as_tensor(targets, dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=GUIDE_LR, weight_decay=WEIGHT_DECAY)
    model.train()
    for step in range(GUIDE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(ids)[:, 5:10, :]
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), labels[:, 5:10].reshape(-1))
        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            check_run_budget(run_started)
    model.eval()
    accuracy = greedy_accuracy(model, items)
    return model, accuracy, state_digest(model)


def greedy_outputs(model: TinyDecoder, items: list[tuple[int, ...]]) -> np.ndarray:
    model.eval()
    prefixes = [[BOS, *[CHAR_TO_ID[ALPHABET[index]] for index in item], SEP] for item in items]
    tokens = torch.as_tensor(prefixes, dtype=torch.long)
    with torch.inference_mode():
        for _ in range(INPUT_LENGTH):
            next_ids = model(tokens)[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_ids), dim=1)
    return tokens[:, -INPUT_LENGTH:].cpu().numpy().astype(np.int64, copy=False)


def greedy_accuracy(model: TinyDecoder, items: list[tuple[int, ...]]) -> float:
    predicted = greedy_outputs(model, items)
    expected = np.asarray(
        [[CHAR_TO_ID[ALPHABET[index]] for index in reverse_truth(item)] for item in items], dtype=np.int64
    )
    return float(np.mean(np.all(predicted == expected, axis=1)))


def capture_module(
    model: TinyDecoder,
    items: list[tuple[int, ...]],
    layer: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ids = model_inputs(items)
    captured: dict[str, torch.Tensor] = {}

    def pre_hook(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        captured["input"] = args[0].detach()

    def post_hook(_module: nn.Module, _args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured["output"] = output.detach()

    module = model.blocks[layer].mlp
    pre = module.register_forward_pre_hook(pre_hook)
    post = module.register_forward_hook(post_hook)
    with torch.inference_mode():
        logits = model(ids)
    pre.remove()
    post.remove()
    return (
        captured["input"].cpu().numpy().astype("<f4", copy=False),
        captured["output"].cpu().numpy().astype("<f4", copy=False),
        endpoint_logit_slice(logits).cpu().numpy().astype("<f4", copy=False),
    )


def adaptation_pairs() -> list[dict[str, Any]]:
    fit = [item for item in contexts() if ALPHABET[item[0]] in ADAPT_PREFIX]
    pairs = []
    for base in fit:
        for position in INTERVENED_POSITIONS:
            source = source_for(base, position)
            counterfactual = intervene(base, source, position)
            pairs.append({"base": base, "source": source, "counterfactual": counterfactual, "position": position})
    return pairs


def challenge_pairs(entropy: bytes) -> tuple[list[dict[str, Any]], list[tuple[int, ...]]]:
    rng = np.random.default_rng(int.from_bytes(entropy, "big"))
    heldout = [item for item in contexts() if ALPHABET[item[0]] in CHALLENGE_PREFIX]
    chosen: list[tuple[int, ...]] = []
    for prefix in CHALLENGE_PREFIX:
        pool = [item for item in heldout if ALPHABET[item[0]] == prefix]
        indexes = rng.choice(len(pool), size=32, replace=False)
        chosen.extend(pool[int(index)] for index in indexes)
    pairs = []
    for base in chosen:
        for position in INTERVENED_POSITIONS:
            source = source_for(base, position)
            counterfactual = intervene(base, source, position)
            pairs.append({"base": base, "source": source, "counterfactual": counterfactual, "position": position})
    endpoints = sorted({item for pair in pairs for item in (pair["base"], pair["counterfactual"])})
    if any(ALPHABET[item[0]] not in CHALLENGE_PREFIX for item in endpoints):
        raise AssertionError("challenge endpoint escaped the frozen held-out split")
    return pairs, endpoints


def train_candidate(
    seed: int,
    run_started: float,
    inputs: np.ndarray,
    outputs: np.ndarray,
    pair_inputs_base: np.ndarray,
    pair_inputs_cf: np.ndarray,
    pair_outputs_base: np.ndarray,
    pair_outputs_cf: np.ndarray,
) -> tuple[MLP, MLP, dict[str, float]]:
    set_determinism(seed)
    pointwise = MLP()
    initial_state = copy.deepcopy(pointwise.state_dict())
    paired = MLP()
    paired.load_state_dict(initial_state)
    x = torch.from_numpy(inputs.reshape(-1, 32).copy())
    y = torch.from_numpy(outputs.reshape(-1, 32).copy())
    xb = torch.from_numpy(pair_inputs_base.astype("<f4", copy=False))
    xc = torch.from_numpy(pair_inputs_cf.astype("<f4", copy=False))
    yb = torch.from_numpy(pair_outputs_base.astype("<f4", copy=False))
    yc = torch.from_numpy(pair_outputs_cf.astype("<f4", copy=False))
    point_scale = torch.mean(y.square()).clamp_min(1e-8)
    effect_target = yc - yb
    effect_scale = torch.mean(effect_target.square()).clamp_min(1e-8)
    for model, use_causal in ((pointwise, False), (paired, True)):
        optimizer = torch.optim.AdamW(model.parameters(), lr=CANDIDATE_LR, weight_decay=WEIGHT_DECAY)
        model.train()
        for step in range(CANDIDATE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            point_loss = F.mse_loss(model(x), y) / point_scale
            loss = point_loss
            if use_causal:
                predicted_effect = model(xc) - model(xb)
                loss = loss + F.mse_loss(predicted_effect, effect_target) / effect_scale
            loss.backward()
            optimizer.step()
            if (step + 1) % 25 == 0:
                check_run_budget(run_started)
        model.eval()
    initial_digest = value_digest({
        key: value.detach().cpu().numpy().astype("<f4", copy=False).tobytes().hex()
        for key, value in sorted(initial_state.items())
    })
    return pointwise, paired, {
        "pointwise_normalizer": float(point_scale.item()),
        "paired_effect_normalizer": float(effect_scale.item()),
        "initial_state_sha256": initial_digest,
    }


def install_candidate(model: TinyDecoder, candidate: nn.Module, layer: int = 0) -> TinyDecoder:
    replaced = copy.deepcopy(model)
    replaced.blocks[layer].mlp = copy.deepcopy(candidate)
    replaced.eval()
    return replaced


def normalized_error(prediction: np.ndarray, target: np.ndarray) -> float:
    numerator = float(np.linalg.norm((prediction - target).astype(np.float64)))
    denominator = max(float(np.linalg.norm(target.astype(np.float64))), 1e-12)
    return numerator / denominator


def max_normalized_endpoint_error(prediction: np.ndarray, target: np.ndarray) -> float:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("endpoint prediction and target arrays must have matching batch dimensions")
    prediction_rows = prediction.astype(np.float64).reshape(prediction.shape[0], -1)
    target_rows = target.astype(np.float64).reshape(target.shape[0], -1)
    numerator = np.linalg.norm(prediction_rows - target_rows, axis=1)
    denominator = np.maximum(np.linalg.norm(target_rows, axis=1), 1e-12)
    return float(np.max(numerator / denominator))


def cosine(prediction: np.ndarray, target: np.ndarray) -> float:
    left, right = prediction.astype(np.float64).reshape(-1), target.astype(np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-24:
        return 0.0
    return float(np.dot(left, right) / denominator)


def task_cross_entropy(logits: np.ndarray, items: list[tuple[int, ...]]) -> float:
    labels = np.asarray(
        [[CHAR_TO_ID[ALPHABET[index]] for index in reverse_truth(item)] for item in items], dtype=np.int64
    )
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    selected = np.take_along_axis(log_probs, labels[:, :, None], axis=-1)[:, :, 0]
    return float(-selected.mean())


def jsonable_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        "base": "".join(ALPHABET[index] for index in pair["base"]),
        "source": "".join(ALPHABET[index] for index in pair["source"]),
        "counterfactual": "".join(ALPHABET[index] for index in pair["counterfactual"]),
        "variable": f"X_{pair['position']}",
        "target_output_variable": f"Y_{INPUT_LENGTH - 1 - pair['position']}",
    }


def make_prediction_arrays(
    models: dict[str, TinyDecoder],
    endpoints: list[tuple[int, ...]],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    ids = model_inputs(endpoints)
    for arm, model in models.items():
        with torch.inference_mode():
            logits = endpoint_logit_slice(model(ids)).cpu().numpy().astype("<f4", copy=False)
        arrays[f"{arm}_logits"] = logits
        arrays[f"{arm}_greedy"] = greedy_outputs(model, endpoints).astype("<i8", copy=False)
        if arm == "paired_causal":
            _, activation, _ = capture_module(model, endpoints, layer=0)
            arrays["paired_causal_target_activation"] = activation
    return arrays


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return array_digest(arrays)


def max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def custody_path(path_text: str) -> Path:
    path = Path(path_text).expanduser().resolve()
    base = CUSTODY_BASE.resolve()
    try:
        path.relative_to(base)
    except ValueError as error:
        raise ValueError("run root must remain under the frozen external custody root") from error
    try:
        path.relative_to(REPO_ROOT.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("model/data artifacts may not be written inside the repository")
    return path


def adaptation_captures(guide: TinyDecoder, pairs: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], list[tuple[int, ...]]]:
    endpoints = sorted({item for pair in pairs for item in (pair["base"], pair["counterfactual"])})
    x_all, y_all, _ = capture_module(guide, endpoints, layer=0)
    endpoint_index = {item: index for index, item in enumerate(endpoints)}
    pair_xb, pair_xc, pair_yb, pair_yc = [], [], [], []
    for pair in pairs:
        output_position = INPUT_LENGTH - 1 - pair["position"]
        token_position = 5 + output_position
        base_index = endpoint_index[pair["base"]]
        cf_index = endpoint_index[pair["counterfactual"]]
        pair_xb.append(x_all[base_index, token_position])
        pair_xc.append(x_all[cf_index, token_position])
        pair_yb.append(y_all[base_index, token_position])
        pair_yc.append(y_all[cf_index, token_position])
    return {
        "adaptation_inputs": x_all,
        "adaptation_targets": y_all,
        "pair_inputs_base": np.asarray(pair_xb, dtype="<f4"),
        "pair_inputs_counterfactual": np.asarray(pair_xc, dtype="<f4"),
        "pair_outputs_base": np.asarray(pair_yb, dtype="<f4"),
        "pair_outputs_counterfactual": np.asarray(pair_yc, dtype="<f4"),
    }, endpoints


def _score_seed(
    seed: int,
    challenge: list[dict[str, Any]],
    endpoints: list[tuple[int, ...]],
    predictions: dict[str, np.ndarray],
    guide_logits: np.ndarray,
    guide_activation: np.ndarray,
    guide_greedy: np.ndarray,
    rescue_error: float,
) -> dict[str, Any]:
    endpoint_index = {item: index for index, item in enumerate(endpoints)}
    expected_greedy = np.asarray(
        [[CHAR_TO_ID[ALPHABET[index]] for index in reverse_truth(item)] for item in endpoints], dtype=np.int64
    )
    guide_ce = task_cross_entropy(guide_logits, endpoints)
    guide_acc = float(np.mean(np.all(guide_greedy == expected_greedy, axis=1)))
    result: dict[str, Any] = {
        "seed": seed,
        "heldout_guide_greedy_accuracy_descriptive": guide_acc,
        "guide_cross_entropy": guide_ce,
        "effect_semantics": (
            "target-position next-token logit difference under teacher-forced correct prior outputs"
        ),
        "behavior_accuracy_semantics": "free-running greedy exact match",
        "arms": {},
    }
    effect_targets = []
    target_baseline_norms = []
    for pair in challenge:
        position = pair["position"]
        target_position = INPUT_LENGTH - 1 - position
        base_index = endpoint_index[pair["base"]]
        cf_index = endpoint_index[pair["counterfactual"]]
        effect_targets.append(guide_logits[cf_index, target_position] - guide_logits[base_index, target_position])
        target_baseline_norms.append(float(np.linalg.norm(guide_logits[base_index, target_position].astype(np.float64))))
    resolved = [
        index for index, target in enumerate(effect_targets)
        if float(np.linalg.norm(target.astype(np.float64))) > max(1e-8 * target_baseline_norms[index], 1e-12)
    ]
    for arm in ("paired_causal", "pointwise", "random", "no_op", "wrong_module"):
        logits = predictions[f"{arm}_logits"]
        greedy = predictions[f"{arm}_greedy"]
        task_accuracy = float(np.mean(np.all(greedy == expected_greedy, axis=1)))
        ce = task_cross_entropy(logits, endpoints)
        predicted_effects = []
        for pair in challenge:
            target_position = INPUT_LENGTH - 1 - pair["position"]
            base_index = endpoint_index[pair["base"]]
            cf_index = endpoint_index[pair["counterfactual"]]
            predicted_effects.append(logits[cf_index, target_position] - logits[base_index, target_position])
        rel_errors = [normalized_error(predicted_effects[i], effect_targets[i]) for i in resolved]
        effect_cosines = [cosine(predicted_effects[i], effect_targets[i]) for i in resolved]
        metrics: dict[str, Any] = {
            "task_accuracy": task_accuracy,
            "task_cross_entropy": ce,
            "task_cross_entropy_delta_vs_guide": ce - guide_ce,
            "median_normalized_output_logit_error": float(np.median([
                normalized_error(logits[i], guide_logits[i]) for i in range(len(endpoints))
            ])),
            "resolvable_effects": len(resolved),
            "median_effect_cosine": float(np.median(effect_cosines)) if effect_cosines else None,
            "median_effect_relative_error": float(np.median(rel_errors)) if rel_errors else None,
        }
        if arm == "paired_causal":
            activation = predictions["paired_causal_target_activation"]
            metrics.update({
                "median_activation_relative_error": float(np.median([
                    normalized_error(activation[i], guide_activation[i]) for i in range(len(endpoints))
                ])),
                "median_activation_cosine": float(np.median([
                    cosine(activation[i], guide_activation[i]) for i in range(len(endpoints))
                ])),
                "oracle_activation_rescue_max_normalized_logit_error": rescue_error,
            })
        result["arms"][arm] = metrics
    primary = result["arms"]["paired_causal"]
    pointwise = result["arms"]["pointwise"]
    treatment_error = primary["median_effect_relative_error"]
    pointwise_error = pointwise["median_effect_relative_error"]
    improvement = None
    if treatment_error is not None and pointwise_error is not None and pointwise_error > 0:
        improvement = 1 - treatment_error / pointwise_error
    control_margins = {}
    for arm in ("random", "no_op", "wrong_module"):
        control_error = result["arms"][arm]["median_effect_relative_error"]
        control_margins[arm] = None if control_error is None or treatment_error is None else control_error - treatment_error
    gates = {
        "task_accuracy": primary["task_accuracy"] >= 0.95,
        "task_cross_entropy_delta": primary["task_cross_entropy_delta_vs_guide"] <= 0.05,
        "output_logit_error": primary["median_normalized_output_logit_error"] <= 0.05,
        "activation_cosine": primary["median_activation_cosine"] >= 0.90,
        "activation_error": primary["median_activation_relative_error"] <= 0.20,
        "resolvable_effect_count": primary["resolvable_effects"] >= 140,
        "effect_cosine": primary["median_effect_cosine"] is not None and primary["median_effect_cosine"] >= 0.90,
        "effect_relative_error": treatment_error is not None and treatment_error <= 0.20,
        "paired_vs_pointwise_improvement": improvement is not None and improvement >= 0.10,
        "control_margins": all(control_margins[arm] is not None and control_margins[arm] >= 0.05 for arm in ("random", "no_op")),
        "oracle_activation_rescue": rescue_error <= 1e-5,
    }
    result["paired_causal_improvement_fraction_vs_pointwise"] = improvement
    result["control_effect_error_margins"] = control_margins
    result["gates"] = gates
    result["seed_outcome"] = "PASS" if all(gates.values()) else "FAIL"
    return result


def run_packet(root_text: str, replay_from_text: str | None = None) -> dict[str, Any]:
    started = time.monotonic()
    packet_sha = verify_freeze()
    runtime = verify_runtime()
    root = custody_path(root_text)
    if root.exists():
        raise FileExistsError("run root already exists; preserve it and choose a new named root")
    root.mkdir(parents=True)
    replay_from = custody_path(replay_from_text) if replay_from_text else None
    try:
        task = contexts()
        task_material = [[ALPHABET[index] for index in item] for item in task]
        task_sha = value_digest(task_material)
        task_array_sha = array_digest({"contexts": np.asarray(task, dtype="<i8")})
        write_json(root / "task-manifest.json", {
            "protocol_id": PROTOCOL_ID,
            "scm": "four independent uniform categorical variables; reverse-position deterministic outputs",
            "task_context_count": len(task),
            "task_support_sha256": task_sha,
            "task_array_sha256": task_array_sha,
            "adaptation_context_count": sum(ALPHABET[item[0]] in ADAPT_PREFIX for item in task),
            "heldout_context_count": sum(ALPHABET[item[0]] in CHALLENGE_PREFIX for item in task),
        })
        guides: dict[int, TinyDecoder] = {}
        pointwise_models: dict[int, MLP] = {}
        paired_models: dict[int, MLP] = {}
        random_models: dict[int, MLP] = {}
        guide_info: dict[int, dict[str, Any]] = {}
        candidate_info: dict[int, dict[str, Any]] = {}
        candidate_arrays: dict[str, np.ndarray] = {}
        adaptation_capture_digests: dict[int, str] = {}
        adaptation_pair_material = [jsonable_pair(pair) for pair in adaptation_pairs()]
        adaptation_fixture_sha = value_digest(adaptation_pair_material)
        write_json(root / "adaptation-fixtures.json", {
            "content_sha256": adaptation_fixture_sha,
            "pairs": adaptation_pair_material,
        })
        for seed in GUIDE_SEEDS:
            guide, accuracy, digest = train_guide(seed, started)
            guide_path = root / f"guide-seed-{seed}.npz"
            save_npz(guide_path, {
                key: value.detach().cpu().numpy().astype("<f4", copy=False)
                for key, value in sorted(guide.state_dict().items())
            })
            guide_info[seed] = {
                "weights_sha256": digest,
                "weight_file_sha256": file_digest(guide_path),
                "task_accuracy": accuracy,
            }
            if accuracy < 0.99:
                raise RuntimeError(f"GUIDE_QUALIFICATION_FAILED seed={seed} accuracy={accuracy}")
            guides[seed] = guide
            pairs = adaptation_pairs()
            fit_arrays, _ = adaptation_captures(guide, pairs)
            adaptation_capture_digests[seed] = save_npz(root / f"adaptation-captures-seed-{seed}.npz", fit_arrays)
            pointwise, paired, scales = train_candidate(
                9000 + seed,
                started,
                fit_arrays["adaptation_inputs"],
                fit_arrays["adaptation_targets"],
                fit_arrays["pair_inputs_base"],
                fit_arrays["pair_inputs_counterfactual"],
                fit_arrays["pair_outputs_base"],
                fit_arrays["pair_outputs_counterfactual"],
            )
            set_determinism(12000 + seed)
            random_candidate = MLP().eval()
            pointwise_models[seed] = pointwise
            paired_models[seed] = paired
            random_models[seed] = random_candidate
            for arm, module in (("pointwise", pointwise), ("paired", paired), ("random", random_candidate)):
                for key, value in sorted(module.state_dict().items()):
                    candidate_arrays[f"seed_{seed}_{arm}_{key}"] = value.detach().cpu().numpy().astype("<f4", copy=False)
            candidate_info[seed] = {
                "pointwise_weights_sha256": state_digest(pointwise),
                "paired_causal_weights_sha256": state_digest(paired),
                "random_weights_sha256": state_digest(random_candidate),
                "same_initialization_sha256": scales["initial_state_sha256"],
                "fit_pointwise_normalizer": scales["pointwise_normalizer"],
                "fit_effect_normalizer": scales["paired_effect_normalizer"],
                "adaptation_pair_count": len(pairs),
                "adaptation_capture_sha256": adaptation_capture_digests[seed],
            }
        candidate_array_sha = save_npz(root / "candidate-states.npz", candidate_arrays)
        write_json(root / "candidate-freeze.json", {
            "packet_sha256": packet_sha,
            "task_support_sha256": task_sha,
            "task_array_sha256": task_array_sha,
            "adaptation_fixture_sha256": adaptation_fixture_sha,
            "adaptation_capture_sha256_by_seed": adaptation_capture_digests,
            "guide_states": guide_info,
            "candidate_states": candidate_info,
            "candidate_array_content_sha256": candidate_array_sha,
        })
        if replay_from is None:
            entropy_by_seed = {str(seed): secrets.token_bytes(32).hex() for seed in GUIDE_SEEDS}
        else:
            entropy_by_seed = json.loads((replay_from / "challenge-entropy.json").read_text())["entropy_hex_by_seed"]
        write_json(root / "challenge-entropy.json", {
            "generated_after_candidate_freeze": replay_from is None,
            "entropy_hex_by_seed": entropy_by_seed,
        })
        challenge_by_seed: dict[int, tuple[list[dict[str, Any]], list[tuple[int, ...]]]] = {}
        challenge_material = {}
        for seed in GUIDE_SEEDS:
            entropy = bytes.fromhex(entropy_by_seed[str(seed)])
            if len(entropy) != 32:
                raise ValueError("challenge entropy must contain exactly 256 bits")
            pairs, endpoints = challenge_pairs(entropy)
            challenge_by_seed[seed] = (pairs, endpoints)
            challenge_material[str(seed)] = {
                "pairs": [jsonable_pair(pair) for pair in pairs],
                "endpoints": ["".join(ALPHABET[index] for index in item) for item in endpoints],
            }
        challenge_sha = value_digest(challenge_material)
        if replay_from is not None:
            expected_challenge = json.loads((replay_from / "challenge-fixtures.json").read_text())
            if expected_challenge.get("content_sha256") != challenge_sha:
                raise ValueError("replica-generated challenge fixtures differ from primary")
        write_json(root / "challenge-fixtures.json", {
            "content_sha256": challenge_sha,
            "generated_after_candidate_freeze": True,
            "by_seed": challenge_material,
        })
        prediction_digests: dict[int, str] = {}
        prediction_arrays_by_seed: dict[int, dict[str, np.ndarray]] = {}
        for seed in GUIDE_SEEDS:
            endpoints = challenge_by_seed[seed][1]
            models = {
                "paired_causal": install_candidate(guides[seed], paired_models[seed], 0),
                "pointwise": install_candidate(guides[seed], pointwise_models[seed], 0),
                "random": install_candidate(guides[seed], random_models[seed], 0),
                "no_op": install_candidate(guides[seed], ZeroMLP(), 0),
                "wrong_module": install_candidate(guides[seed], paired_models[seed], 1),
            }
            arrays = make_prediction_arrays(models, endpoints)
            prediction_arrays_by_seed[seed] = arrays
            prediction_digests[seed] = save_npz(root / f"candidate-predictions-seed-{seed}.npz", arrays)
        prediction_manifest = {
            "all_candidates_and_predictions_sealed": True,
            "challenge_fixture_sha256": challenge_sha,
            "prediction_content_sha256_by_seed": prediction_digests,
        }
        write_json(root / "prediction-freeze.json", prediction_manifest)

        # Held-out teacher outputs and activation rescue are opened only after
        # every seed's candidate/control prediction digest has been sealed.
        seed_results: dict[int, Any] = {}
        oracle_digest_map: dict[str, str] = {}
        oracle_file_digest_map: dict[str, str] = {}
        for seed in GUIDE_SEEDS:
            pairs, endpoints = challenge_by_seed[seed]
            guide = guides[seed]
            _, guide_activation, guide_logits = capture_module(guide, endpoints, layer=0)
            guide_greedy = greedy_outputs(guide, endpoints)
            rescue_model = install_candidate(guide, paired_models[seed], 0)
            rescue_ids = model_inputs(endpoints)
            with torch.inference_mode():
                rescued_logits = endpoint_logit_slice(
                    rescue_model(rescue_ids, rescue_values=torch.from_numpy(guide_activation))
                ).cpu().numpy().astype("<f4", copy=False)
            rescue_error = max_normalized_endpoint_error(rescued_logits, guide_logits)
            oracle_arrays = {
                "guide_logits": guide_logits,
                "guide_activation": guide_activation,
                "guide_greedy": guide_greedy,
                "rescue_logits": rescued_logits,
            }
            oracle_digest_map[str(seed)] = array_digest(oracle_arrays)
            oracle_file_digest_map[str(seed)] = save_npz(root / f"oracle-challenge-seed-{seed}.npz", oracle_arrays)
            seed_results[seed] = _score_seed(
                seed,
                pairs,
                endpoints,
                prediction_arrays_by_seed[seed],
                guide_logits,
                guide_activation,
                guide_greedy,
                rescue_error,
            )
        if all(item["seed_outcome"] == "PASS" for item in seed_results.values()):
            scientific_outcome = "PASS"
        else:
            scientific_outcome = "FAIL"
        wall_seconds = time.monotonic() - started
        rss = max_rss_bytes()
        artifact_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if wall_seconds > MAX_WALL_SECONDS or rss > MAX_PEAK_RSS_BYTES or artifact_bytes > MAX_ARTIFACT_BYTES:
            scientific_outcome = "INCONCLUSIVE"
        outcome_sha = value_digest({
            "scientific_outcome": scientific_outcome,
            "seed_results": seed_results,
            "oracle_array_digests": oracle_digest_map,
        })
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        git_status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True,
        ).stdout
        summary = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet_sha,
            "source_revision": git_revision,
            "worktree_status_sha256": sha256(git_status.encode()),
            "state_slices": [STATE_SLICE, CONTINUATION_SLICE, ASSURANCE_SLICE],
            "assurance": "INTERNAL_PASS" if scientific_outcome in {"PASS", "FAIL"} else "INCONCLUSIVE",
            "scientific_outcome": scientific_outcome,
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility": "INCONCLUSIVE" if replay_from is not None else "NOT_RUN",
            "runtime": runtime,
            "task_support_sha256": task_sha,
            "task_array_sha256": task_array_sha,
            "adaptation_fixture_sha256": adaptation_fixture_sha,
            "adaptation_capture_sha256_by_seed": adaptation_capture_digests,
            "challenge_fixture_sha256": challenge_sha,
            "runtime_dependency_inventory_sha256": runtime["dependency_inventory_sha256"],
            "guide_info": guide_info,
            "guide_state_digests": {str(seed): guide_info[seed]["weights_sha256"] for seed in GUIDE_SEEDS},
            "candidate_info": candidate_info,
            "candidate_array_content_sha256": candidate_array_sha,
            "candidate_prediction_digests": prediction_digests,
            "oracle_array_digests": oracle_digest_map,
            "oracle_file_content_digests": oracle_file_digest_map,
            "seed_results": seed_results,
            "outcome_sha256": outcome_sha,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": rss,
            "artifact_bytes_before_result_receipt": artifact_bytes,
            "custody": "EXTERNAL_OWNER_CONTROLLED",
            "claim_ceiling": "SYNTHETIC_TRAINED_TRANSFORMER_INPUT_INTERVENTION_REPLACEMENT_QUALIFICATION",
            "nonclaims": [
                "novelty", "IIT reproduction", "pretrained model evidence", "architecture generality",
                "consciousness", "alignment", "independent science", "production readiness",
            ],
        }
        write_json(root / "result.json", summary)
        if replay_from is not None:
            primary = json.loads((replay_from / "result.json").read_text())
            comparison_fields = (
                "packet_sha256", "source_revision", "worktree_status_sha256",
                "runtime_dependency_inventory_sha256", "task_support_sha256", "task_array_sha256",
                "adaptation_fixture_sha256", "adaptation_capture_sha256_by_seed",
                "challenge_fixture_sha256", "guide_state_digests", "candidate_array_content_sha256",
                "candidate_prediction_digests", "oracle_array_digests", "oracle_file_content_digests", "outcome_sha256",
            )
            mismatches = [field for field in comparison_fields if summary[field] != primary.get(field)]
            distinct_runtime = runtime["runtime_prefix"] != primary.get("runtime", {}).get("runtime_prefix")
            reproducibility = "REPRODUCED_LOCAL" if not mismatches and distinct_runtime else "INCONCLUSIVE"
            write_json(root / "reconciliation.json", {
                "reproducibility": reproducibility,
                "same_owner_second_environment": distinct_runtime,
                "comparison_fields": list(comparison_fields),
                "mismatches": mismatches,
                "primary_runtime_fingerprint": primary.get("runtime"),
                "replica_runtime_fingerprint": runtime,
            })
            summary["reproducibility"] = reproducibility
            write_json(root / "result.json", summary)
        print(json.dumps(summary, sort_keys=True, indent=2))
        return summary
    except Exception as error:
        write_json(root / "terminal.json", {
            "protocol_id": PROTOCOL_ID,
            "status": "INCONCLUSIVE",
            "state_slices": [STATE_SLICE, CONTINUATION_SLICE, ASSURANCE_SLICE],
            "assurance": "INCONCLUSIVE",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility": "INCONCLUSIVE" if replay_from_text else "NOT_RUN",
            "error_type": type(error).__name__,
            "error": str(error),
            "packet_sha256": packet_sha,
            "runtime": runtime,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": max_rss_bytes(),
            "custody": "EXTERNAL_OWNER_CONTROLLED",
            "claim_ceiling": "SYNTHETIC_TRAINED_TRANSFORMER_INPUT_INTERVENTION_REPLACEMENT_QUALIFICATION",
            "next_blocked_gate": "A completed valid local run followed by a separate blind replication and authorized external review for independent science",
            "preserve_partial_artifacts": True,
        })
        raise


def self_check() -> dict[str, Any]:
    all_contexts = contexts()
    assert len(all_contexts) == 256
    assert all(reverse_truth(item) == item[::-1] for item in all_contexts)
    fit = {item for item in all_contexts if ALPHABET[item[0]] in ADAPT_PREFIX}
    heldout = {item for item in all_contexts if ALPHABET[item[0]] in CHALLENGE_PREFIX}
    assert len(fit) == len(heldout) == 128 and fit.isdisjoint(heldout)
    checked_pairs = 0
    for base in all_contexts:
        for position in INTERVENED_POSITIONS:
            source = source_for(base, position)
            counterfactual = intervene(base, source, position)
            expected = list(reverse_truth(base))
            expected[INPUT_LENGTH - 1 - position] = source[position]
            assert reverse_truth(counterfactual) == tuple(expected)
            assert sum(a != b for a, b in zip(base, counterfactual)) == 1
            checked_pairs += 1
    set_determinism(901)
    model = TinyDecoder()
    sample = model_inputs(all_contexts[:3])
    with torch.inference_mode():
        logits = model(sample)
    assert logits.shape == (3, 10, VOCAB_SIZE)
    return {
        "protocol_id": PROTOCOL_ID,
        "status": "PASS",
        "task_contexts": len(all_contexts),
        "scm_intervention_pairs_checked": checked_pairs,
        "adaptation_contexts": len(fit),
        "heldout_contexts": len(heldout),
        "model_output_shape": list(logits.shape),
        "weights_loaded": False,
        "challenge_entropy_drawn": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-check", "run"))
    parser.add_argument("--root")
    parser.add_argument("--replay-from")
    args = parser.parse_args()
    if args.command == "self-check":
        verify_runtime()
        print(json.dumps(self_check(), sort_keys=True, indent=2))
        return
    if not args.root:
        parser.error("run requires --root under the V31 external custody root")
    run_packet(args.root, args.replay_from)


if __name__ == "__main__":
    main()
