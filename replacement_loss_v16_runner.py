"""Frozen local runner for an independently trained whole OPT MLP replacement.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v16-whole-mlp-independent-replacement-v1, and
global-experiment-assurance-lifecycle-v1.

All prompt-bearing and row-level artifacts are written only to owner custody
under /Volumes/PrimaryED/ResearchArtifacts.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import platform
import resource
import secrets
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import replacement_loss_v16_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v16-freeze.json"
CLAIM_CEILING = "LOCAL_WHOLE_MLP_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ON_ONE_REUSED_OPT_CHECKPOINT"

TARGET_MODEL_ID = "grushaprasad/opt125m-rootp-surprisal-curriculum"
TARGET_REVISION = "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
TARGET_TREE_SHA256 = "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466"
TARGET_WEIGHT_SHA256 = "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132"
TARGET_PATH = Path("/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v9/checkpoints/opt_relu")
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v16-whole-mlp-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v16-whole-mlp-replica"
RESEARCH_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
REPO_ROOT = Path(__file__).resolve().parent

TRAIN_UPDATES = 1000
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
MAX_PARAMETER_COUNT = 200_000_000
MAX_WEIGHT_BYTES = 800 * 1024**2
INFERENCE_BATCH_SIZE = 4
PACKET_FILES = (
    "AGENTS.md",
    "docs/replacement-loss-v16-whole-mlp.md",
    "replacement_loss_v16_fixtures.py",
    "replacement_loss_v16_runner.py",
    "replacement_loss_v3_fixtures.py",
    "requirements-v16.lock.txt",
    "docs/replacement-loss-v3-results.md",
    "docs/replacement-loss-v8-results.md",
    "docs/replacement-loss-v9-results.md",
    "docs/replacement-loss-v13-results.md",
    "docs/replacement-loss-v14-results.md",
    "docs/replacement-loss-v15c-results.md",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def _tree_digest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name.endswith(".lock") or ".cache" in path.parts:
            continue
        rows.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": _sha_file(path),
            "bytes": path.stat().st_size,
        })
    return _digest(rows), rows


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable artifact already exists with different bytes: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _save_npz_deterministic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in sorted(arrays.items()):
            item = io.BytesIO()
            np.lib.format.write_array(item, np.asarray(value), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, item.getvalue())
    path.write_bytes(buffer.getvalue())


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen V16 packet source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def freeze_packet() -> dict[str, Any]:
    path = REPO_ROOT / FREEZE_FILE
    if path.exists():
        return verify_packet()
    files = _packet_rows()
    packet = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": files,
        "packet_sha256": _digest(files),
        "target_identity": {
            "model_id": TARGET_MODEL_ID,
            "revision": TARGET_REVISION,
            "tree_sha256": TARGET_TREE_SHA256,
            "weight_sha256": TARGET_WEIGHT_SHA256,
            "target_known_before_protocol": True,
            "fresh_checkpoint_claim": False,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_packet() -> dict[str, Any]:
    path = REPO_ROOT / FREEZE_FILE
    if not path.is_file():
        raise FileNotFoundError("V16 packet is not frozen")
    frozen = _read_json(path)
    current = _packet_rows()
    if frozen.get("packet_files") != current or frozen.get("packet_sha256") != _digest(current):
        raise ValueError("V16 packet source bytes differ from the frozen packet")
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _digest(material):
        raise ValueError("V16 freeze digest is invalid")
    if (
        frozen.get("protocol_id") != PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("target_identity") != {
            "model_id": TARGET_MODEL_ID,
            "revision": TARGET_REVISION,
            "tree_sha256": TARGET_TREE_SHA256,
            "weight_sha256": TARGET_WEIGHT_SHA256,
            "target_known_before_protocol": True,
            "fresh_checkpoint_claim": False,
        }
    ):
        raise ValueError("V16 frozen protocol identity or claim ceiling differs")
    return frozen


def _custody_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.parent != RESEARCH_ROOT or root.name not in {PRIMARY_NAME, REPLICA_NAME}:
        raise ValueError("custody root must be one of the two frozen V16 PrimaryED roots")
    if root.is_symlink():
        raise ValueError("custody root cannot be a symlink")
    return root


def _max_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def _runtime_receipt() -> dict[str, Any]:
    import importlib.metadata as metadata

    names = ("huggingface-hub", "numpy", "safetensors", "tokenizers", "torch", "transformers")
    inventory = sorted(
        ({"name": distribution.metadata.get("Name", "unknown"), "version": distribution.version}
         for distribution in metadata.distributions()),
        key=lambda row: (row["name"].casefold(), row["version"]),
    )
    repository_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    receipt = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "repository_head": repository_head,
        "repository_worktree_dirty": bool(status),
        "repository_dirty_entry_count": len(status),
        "host_id_sha256": hashlib.sha256(platform.node().encode("utf-8")).hexdigest(),
        "uid": int(os.getuid()) if hasattr(os, "getuid") else None,
        "packages": {name: metadata.version(name) for name in names},
        "package_inventory": inventory,
        "package_inventory_sha256": _digest(inventory),
        "dependency_lock_sha256": _sha_file(REPO_ROOT / "requirements-v16.lock.txt"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_num_threads": int(torch.get_num_threads()),
    }
    if receipt["python"] != "3.12.13":
        raise ValueError("V16 requires the frozen Python 3.12.13 runtime")
    lock = {}
    for line in (REPO_ROOT / "requirements-v16.lock.txt").read_text(encoding="utf-8").splitlines():
        normalized = line.strip()
        if not normalized or normalized.startswith("#"):
            continue
        name, separator, version = normalized.partition("==")
        if not separator or not name or not version:
            raise ValueError("V16 dependency lock contains an invalid pin")
        lock[name.casefold().replace("_", "-")] = version
    installed = {row["name"].casefold().replace("_", "-"): row["version"] for row in inventory}
    if len(installed) != len(inventory) or installed != lock:
        raise ValueError("V16 installed package inventory differs from the frozen complete lock")
    return receipt


def _external_packet_archive(root: Path) -> dict[str, Any]:
    frozen_root = root / "frozen-packet"
    packet = verify_packet()
    if not (root / "frozen-packet-manifest.json").is_file():
        raise FileNotFoundError("external frozen packet archive is missing")
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("external frozen packet archive binds a different packet")
    for row in packet["packet_files"]:
        path = frozen_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"external frozen packet file differs: {row['path']}")
    if not (frozen_root / FREEZE_FILE).is_file() or _sha_file(frozen_root / FREEZE_FILE) != _sha_file(REPO_ROOT / FREEZE_FILE):
        raise ValueError("external frozen packet has a different freeze record")
    expected_rows = list(packet["packet_files"])
    freeze_copy = frozen_root / FREEZE_FILE
    expected_rows.append({"path": FREEZE_FILE, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected_rows or manifest.get("archive_sha256") != _digest(expected_rows):
        raise ValueError("external frozen packet archive manifest is invalid")
    return manifest


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _custody_root(custody_path)
    if root.exists():
        raise FileExistsError("V16 custody root already exists; preserve it and choose a fresh root")
    root.mkdir(parents=True, exist_ok=False)
    frozen_root = root / "frozen-packet"
    for row in packet["packet_files"]:
        source = REPO_ROOT / row["path"]
        destination = frozen_root / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    freeze_source = REPO_ROOT / FREEZE_FILE
    freeze_destination = frozen_root / FREEZE_FILE
    freeze_destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(freeze_source, freeze_destination)
    archive_rows = [{"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]} for row in packet["packet_files"]]
    archive_rows.append({
        "path": FREEZE_FILE,
        "sha256": _sha_file(freeze_destination),
        "bytes": freeze_destination.stat().st_size,
    })
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"],
        "files": archive_rows,
        "archive_sha256": _digest(archive_rows),
    }
    _write_json(root / "frozen-packet-manifest.json", manifest, immutable=True)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "archive_sha256": manifest["archive_sha256"], "file_count": len(archive_rows)}


def _ensure_root(custody_path: str | Path, *, require_archive: bool = True) -> Path:
    root = _custody_root(custody_path)
    if not root.is_dir():
        raise FileNotFoundError("V16 external custody root is missing")
    if (root / "run-stop.json").exists():
        raise RuntimeError("this V16 custody root has a terminal stop receipt; do not reuse it")
    if require_archive:
        _external_packet_archive(root)
    return root


def _target_tree_receipt() -> dict[str, Any]:
    if not TARGET_PATH.is_dir():
        raise FileNotFoundError("predeclared OPT checkpoint is not present in external owner custody")
    tree_digest, rows = _tree_digest(TARGET_PATH)
    weights = TARGET_PATH / "pytorch_model.bin"
    if tree_digest != TARGET_TREE_SHA256 or not weights.is_file() or _sha_file(weights) != TARGET_WEIGHT_SHA256:
        raise ValueError("predeclared OPT checkpoint bytes do not match the bound target digests")
    if weights.stat().st_size > MAX_WEIGHT_BYTES:
        raise ValueError("checkpoint weights exceed the frozen 800 MiB cap")
    return {"tree_sha256": tree_digest, "weight_sha256": _sha_file(weights), "weight_bytes": weights.stat().st_size, "file_count": len(rows)}


def _load_model() -> tuple[torch.nn.Module, Any]:
    _target_tree_receipt()
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_PATH, local_files_only=True, trust_remote_code=False, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("the frozen OPT tokenizer has no padding or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, trust_remote_code=False, dtype=torch.float32
    )
    config = model.config
    if (
        config.model_type != "opt"
        or int(config.num_hidden_layers) != 12
        or int(config.hidden_size) != 768
        or int(config.ffn_dim) != 3072
        or str(config.activation_function).lower() != "relu"
    ):
        raise ValueError("checkpoint architecture differs from the frozen OPT/ReLU whole-MLP contract")
    if sum(parameter.numel() for parameter in model.parameters()) > MAX_PARAMETER_COUNT:
        raise ValueError("checkpoint exceeds the frozen 200-million-parameter cap")
    model.to(device="cpu", dtype=torch.float32)
    model.eval()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return model, tokenizer


def _layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    result = model.model.decoder.layers
    if len(result) != 12:
        raise ValueError("OPT decoder layer count changed")
    return list(result)


def _encode(tokenizer: Any, record: dict[str, Any], *, with_continuation: bool) -> dict[str, Any]:
    prompt = [int(value) for value in tokenizer(record["input_text"], add_special_tokens=True, truncation=True, max_length=128)["input_ids"]]
    continuation = (
        [int(value) for value in tokenizer(record["continuation"], add_special_tokens=False, truncation=True, max_length=32)["input_ids"]]
        if with_continuation else []
    )
    if not prompt or (with_continuation and not continuation):
        raise ValueError("a frozen synthetic prompt or continuation tokenized to an empty sequence")
    return {**record, "ids": prompt + continuation, "prompt_len": len(prompt), "continuation_len": len(continuation)}


def _model_inputs(rows: list[dict[str, Any]], tokenizer: Any) -> tuple[dict[str, torch.Tensor], list[int]]:
    maximum = max(len(row["ids"]) for row in rows)
    ids = torch.full((len(rows), maximum), int(tokenizer.pad_token_id), dtype=torch.long)
    mask = torch.zeros((len(rows), maximum), dtype=torch.long)
    positions = []
    for index, row in enumerate(rows):
        length = len(row["ids"])
        ids[index, :length] = torch.tensor(row["ids"], dtype=torch.long)
        mask[index, :length] = 1
        positions.append(int(row["prompt_len"]) - 1)
    return {"input_ids": ids, "attention_mask": mask}, positions


def _continuation_ce(logits: torch.Tensor, inputs: dict[str, torch.Tensor], rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for index, row in enumerate(rows):
        prompt_len = int(row["prompt_len"])
        end = prompt_len + int(row["continuation_len"])
        if end <= prompt_len:
            raise ValueError("challenge continuation is empty")
        target_positions = torch.arange(prompt_len, end, dtype=torch.long)
        source_positions = target_positions - 1
        log_probs = F.log_softmax(logits[index, source_positions], dim=-1)
        labels = inputs["input_ids"][index, target_positions]
        values.append(float(-log_probs.gather(1, labels.unsqueeze(1)).mean().item()))
    return np.asarray(values, dtype=np.float64)


def _flat_positions(batch: int, sequence: int, positions: list[int]) -> torch.Tensor:
    return torch.tensor([index * sequence + position for index, position in enumerate(positions)], dtype=torch.long)


def _capture_visible_inputs(model: torch.nn.Module, tokenizer: Any, records: list[dict[str, Any]]) -> list[np.ndarray]:
    layers = _layers(model)
    collected: list[list[np.ndarray]] = [[] for _ in layers]
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module: torch.nn.Module, args: tuple[Any, ...], *, layer_index: int = index) -> None:
            activation = args[0]
            if activation.ndim != 2:
                raise ValueError("OPT fc1 input layout changed from flattened token rows")
            collected[layer_index].append(activation.detach().cpu().numpy())
        handles.append(layer.fc1.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for record in records:
                ids = torch.tensor([tokenizer(record["input_text"], add_special_tokens=True, truncation=True, max_length=128)["input_ids"]], dtype=torch.long)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
    finally:
        for handle in handles:
            handle.remove()
    return [np.concatenate(rows, axis=0) for rows in collected]


def _capture_adaptation(model: torch.nn.Module, tokenizer: Any, records: list[dict[str, Any]], target_layer: int, adjacent_layer: int) -> dict[str, np.ndarray]:
    captures: dict[int, dict[str, list[np.ndarray]]] = {
        target_layer: {"x": [], "y": []},
        adjacent_layer: {"x": [], "y": []},
    }
    layers = _layers(model)
    handles = []
    for index in (target_layer, adjacent_layer):
        layer = layers[index]
        def pre(_module: torch.nn.Module, args: tuple[Any, ...], *, layer_index: int = index) -> None:
            activation = args[0]
            if activation.ndim != 2:
                raise ValueError("OPT fc1 input layout changed from flattened token rows")
            captures[layer_index]["x"].append(activation.detach().cpu().numpy())
        def post(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any, *, layer_index: int = index) -> None:
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise ValueError("OPT fc2 output layout changed from flattened token rows")
            captures[layer_index]["y"].append(output.detach().cpu().numpy())
        handles.append(layer.fc1.register_forward_pre_hook(pre))
        handles.append(layer.fc2.register_forward_hook(post))
    prompt_indices = []
    try:
        with torch.inference_mode():
            for row_index, record in enumerate(records):
                ids = torch.tensor([tokenizer(record["input_text"], add_special_tokens=True, truncation=True, max_length=128)["input_ids"]], dtype=torch.long)
                model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, return_dict=True)
                prompt_indices.extend([row_index] * ids.shape[1])
    finally:
        for handle in handles:
            handle.remove()
    result: dict[str, np.ndarray] = {"prompt_index": np.asarray(prompt_indices, dtype=np.int32)}
    for index, label in ((target_layer, "target"), (adjacent_layer, "adjacent")):
        result[f"x_{label}"] = np.concatenate(captures[index]["x"], axis=0).astype(np.float32, copy=False)
        result[f"y_{label}"] = np.concatenate(captures[index]["y"], axis=0).astype(np.float32, copy=False)
    return result


class WholeMLPCandidate(torch.nn.Module):
    """Fresh OPT-width MLP; source parameters are never passed to this module."""

    def __init__(self, input_dim: int = 768, ffn_dim: int = 3072, output_dim: int = 768) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, ffn_dim, bias=True)
        self.fc2 = torch.nn.Linear(ffn_dim, output_dim, bias=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(inputs)))


class ZeroOutput(torch.nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*inputs.shape[:-1], self.output_dim), dtype=inputs.dtype, device=inputs.device)


class ZeroExpansion(torch.nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*inputs.shape[:-1], self.output_dim), dtype=inputs.dtype, device=inputs.device)


class NoOpCandidate(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = ZeroExpansion(3072)
        self.fc2 = ZeroOutput(768)


def _module_state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _new_candidate(seed: int) -> WholeMLPCandidate:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return WholeMLPCandidate()


def _train_candidate(
    inputs: np.ndarray,
    targets: np.ndarray,
    validation_inputs: np.ndarray,
    validation_targets: np.ndarray,
    *,
    seed: int,
    custody_root: Path | None = None,
) -> tuple[WholeMLPCandidate, WholeMLPCandidate, dict[str, Any]]:
    if inputs.ndim != 2 or targets.ndim != 2 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("candidate training requires paired rank-two adaptation arrays")
    if inputs.shape[1] != 768 or targets.shape[1] != 768:
        raise ValueError("candidate adaptation dimensions differ from the frozen OPT contract")
    x = torch.from_numpy(np.asarray(inputs, dtype=np.float32))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    xv = torch.from_numpy(np.asarray(validation_inputs, dtype=np.float32))
    yv = torch.from_numpy(np.asarray(validation_targets, dtype=np.float32))
    candidate = _new_candidate(seed)
    random_control = copy.deepcopy(candidate)
    initialization_sha = _module_state_digest(candidate)
    normalizer = max(float(y.square().mean().item()), 1e-12)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x16A5C3)
    trace = np.empty((TRAIN_UPDATES, TRAIN_BATCH_SIZE), dtype=np.int32)
    candidate.train()
    for update in range(TRAIN_UPDATES):
        indices = torch.randint(0, x.shape[0], (TRAIN_BATCH_SIZE,), generator=generator)
        trace[update] = indices.numpy()
        predicted = candidate(x.index_select(0, indices))
        loss = F.mse_loss(predicted, y.index_select(0, indices)) / normalizer
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V16 replacement training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if (update + 1) % 100 == 0:
            _check_caps(custody_root)
    candidate.eval()
    random_control.eval()
    with torch.inference_mode():
        fit_loss = float(F.mse_loss(candidate(x), y).item() / normalizer)
        validation_loss = float(F.mse_loss(candidate(xv), yv).item() / normalizer)
    receipt = {
        "initialization_sha256": initialization_sha,
        "candidate_state_sha256": _module_state_digest(candidate),
        "fit_token_rows": int(x.shape[0]),
        "validation_token_rows": int(xv.shape[0]),
        "updates": TRAIN_UPDATES,
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(trace.astype("<i4", copy=False).tobytes()).hexdigest(),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP,
        "normalized_fit_mse": fit_loss,
        "normalized_validation_mse": validation_loss,
        "normalizer": normalizer,
        "source_parameters_copied": False,
    }
    return candidate, random_control, receipt


def _save_candidate(path: Path, candidate: WholeMLPCandidate) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in candidate.state_dict().items()}
    save_file(state, str(path))
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": _module_state_digest(candidate),
        "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        "architecture": "Linear-ReLU-Linear",
    }


def _load_candidate(root: Path, row: dict[str, Any]) -> WholeMLPCandidate:
    path = root / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("V16 candidate bytes differ from their frozen manifest")
    candidate = WholeMLPCandidate()
    candidate.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if _module_state_digest(candidate) != row["state_sha256"]:
        raise ValueError("V16 candidate state digest differs")
    candidate.eval()
    return candidate


def _verify_candidate_freeze(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "candidate-freeze.json")
    if manifest.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("V16 candidate freeze names a different protocol")
    if manifest.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("V16 candidates bind a different frozen packet")
    material = {key: value for key, value in manifest.items() if key not in {"candidate_set_sha256", "candidate_freeze_sha256"}}
    if manifest.get("candidate_set_sha256") != _digest(material):
        raise ValueError("V16 candidate-set digest is invalid")
    freeze_material = {key: value for key, value in manifest.items() if key != "candidate_freeze_sha256"}
    if manifest.get("candidate_freeze_sha256") != _digest(freeze_material):
        raise ValueError("V16 candidate-freeze digest is invalid")
    candidates = manifest.get("candidates", [])
    expected = {(seed, method) for seed in range(len(fixtures.SEEDS)) for method in ("treatment", "wrong_layer", "random")}
    actual = {(row.get("seed_index"), row.get("method")) for row in candidates}
    if len(candidates) != len(expected) or actual != expected:
        raise ValueError("V16 candidate freeze has a missing or duplicate seed/method candidate")
    for artifact in manifest.get("preparation_artifacts", []):
        path = root / artifact["path"]
        if not path.is_file() or path.stat().st_size != artifact["bytes"] or _sha_file(path) != artifact["sha256"]:
            raise ValueError("V16 preparation artifact differs from candidate freeze")
    for row in candidates:
        path = root / row["path"]
        if not path.is_file() or _sha_file(path) != row["file_sha256"]:
            raise ValueError("V16 candidate tensor file differs from its frozen manifest")
    expected_artifact_paths = {
        "fixtures/visible.json", "block-selection.json",
        *(f"fixtures/adaptation-seed-{seed_index}.json" for seed_index in range(len(fixtures.SEEDS))),
        *(f"splits/seed-{seed_index}.json" for seed_index in range(len(fixtures.SEEDS))),
        *(f"adaptation/seed-{seed_index}.npz" for seed_index in range(len(fixtures.SEEDS))),
    }
    if {row.get("path") for row in manifest.get("preparation_artifacts", [])} != expected_artifact_paths:
        raise ValueError("V16 candidate freeze omits fixture, split, selection, or capture provenance")
    return manifest


def _verify_entropy_manifest(root: Path, candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(root / "challenge-entropy-manifest.json")
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidate_manifest["candidate_set_sha256"]
    ):
        raise ValueError("V16 challenge entropy is not bound to the frozen protocol and candidates")
    rows = manifest.get("entropy", [])
    if [row.get("seed_index") for row in rows] != list(range(len(fixtures.SEEDS))):
        raise ValueError("V16 challenge entropy manifest has missing or unordered seeds")
    if manifest.get("entropy_manifest_sha256") != _digest(rows):
        raise ValueError("V16 challenge entropy manifest digest is invalid")
    for row in rows:
        path = root / "challenge-entropy" / f"seed-{row['seed_index']}.bin"
        if not path.is_file() or path.stat().st_size != 32 or _sha_file(path) != row.get("sha256"):
            raise ValueError("V16 challenge entropy bytes differ from their manifest")
    return manifest


def _verify_fixture_payload(path: Path, *, expected_kind: str, expected_count: int) -> dict[str, Any]:
    payload = _read_json(path)
    records = payload.get("records")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("fixture_kind") != expected_kind
        or payload.get("prompt_rules_version") != fixtures.PROMPT_RULES_VERSION
        or not isinstance(records, list)
        or len(records) != expected_count
        or payload.get("count") != expected_count
    ):
        raise ValueError("V16 fixture identity or count differs from the frozen protocol")
    record_digests = [hashlib.sha256(_canonical_bytes(row)).hexdigest() for row in records]
    expected_digest = hashlib.sha256("\n".join(sorted(record_digests)).encode("ascii")).hexdigest()
    if len({row.get("example_id") for row in records}) != len(records) or payload.get("example_set_sha256") != expected_digest:
        raise ValueError("V16 fixture content digest or example IDs are invalid")
    if expected_kind == "visible":
        expected_strata = {"visible": fixtures.VISIBLE_COUNT}
    elif expected_kind.startswith("adaptation-"):
        expected_strata = {name: fixtures.PER_STRATUM_COUNT for name in fixtures.STRATA}
    else:
        expected_strata = {name: fixtures.CHALLENGES_PER_STRATUM for name in fixtures.STRATA}
    actual_strata = {name: sum(row.get("stratum") == name for row in records) for name in sorted(expected_strata)}
    if payload.get("stratum_counts") != actual_strata or actual_strata != dict(sorted(expected_strata.items())):
        raise ValueError("V16 fixture stratum counts differ from its records")
    return payload


def _verify_prediction_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "prediction-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "prediction_set_sha256"}
    if manifest.get("prediction_set_sha256") != _digest(material):
        raise ValueError("V16 prediction-set digest is invalid")
    candidate_manifest = _verify_candidate_freeze(root)
    entropy_manifest = _verify_entropy_manifest(root, candidate_manifest)
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidate_manifest["candidate_set_sha256"]
        or manifest.get("entropy_manifest_sha256") != entropy_manifest["entropy_manifest_sha256"]
    ):
        raise ValueError("V16 prediction manifest does not bind the frozen packet, candidates, and entropy")
    selection = _read_json(root / "block-selection.json")
    if manifest.get("target_layer") != selection.get("target_layer"):
        raise ValueError("V16 prediction target layer differs from visible-only selection")
    challenge_rows = manifest.get("challenge_rows", [])
    if [row.get("seed_index") for row in challenge_rows] != list(range(len(fixtures.SEEDS))):
        raise ValueError("V16 prediction manifest has missing or unordered challenge seeds")
    visible = _verify_fixture_payload(root / "fixtures" / "visible.json", expected_kind="visible", expected_count=fixtures.VISIBLE_COUNT)
    adaptation_payloads = []
    for seed_index in range(len(fixtures.SEEDS)):
        adaptation_payloads.append(_verify_fixture_payload(
            root / "fixtures" / f"adaptation-seed-{seed_index}.json",
            expected_kind=f"adaptation-seed-{seed_index}",
            expected_count=fixtures.PER_STRATUM_COUNT * len(fixtures.STRATA),
        ))
    for row in challenge_rows:
        seed_index = int(row["seed_index"])
        challenge = _verify_fixture_payload(
            root / "fixtures" / f"challenge-seed-{seed_index}.json",
            expected_kind=f"challenge-seed-{seed_index}",
            expected_count=fixtures.CHALLENGES_PER_STRATUM * len(fixtures.STRATA),
        )
        fixtures.validate_partition_disjointness(visible, *adaptation_payloads, challenge)
        if row.get("challenge_sha256") != challenge.get("example_set_sha256") or row.get("challenge_count") != challenge.get("count"):
            raise ValueError("V16 prediction challenge fixture differs from the sealed prediction row")
        path = root / "predictions" / f"seed-{int(row['seed_index'])}.npz"
        if not path.is_file() or _sha_file(path) != row["prediction_file_sha256"]:
            raise ValueError("V16 prediction bytes differ from the sealed prediction manifest")
    return manifest


def _phase_ledger(root: Path) -> dict[str, Any]:
    path = root / "phase-ledger.json"
    if path.exists():
        return _read_json(path)
    value = {"protocol_id": PROTOCOL_ID, "started_at_epoch": time.time(), "phases": []}
    _write_json(path, value, immutable=True)
    return value


def _check_caps(root: Path | None, phase_start: float | None = None) -> None:
    if phase_start is not None and time.monotonic() - phase_start > MAX_WALL_SECONDS:
        raise TimeoutError("V16 phase exceeded the frozen four-hour wall-clock cap")
    if root is not None and (root / "phase-ledger.json").exists():
        ledger = _read_json(root / "phase-ledger.json")
        elapsed = time.time() - float(ledger["started_at_epoch"])
        if elapsed > MAX_WALL_SECONDS:
            raise TimeoutError("V16 run exceeded the frozen four-hour wall-clock cap")
    rss = _max_rss_bytes()
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f"V16 process RSS {rss} exceeds the frozen cap {MAX_RSS_BYTES}")


def _record_phase(root: Path, phase: str, started: float, status: str, *, details: dict[str, Any] | None = None) -> None:
    ledger = _phase_ledger(root)
    phases = list(ledger.get("phases", []))
    phases.append({
        "phase": phase,
        "status": status,
        "elapsed_seconds": round(time.monotonic() - started, 6),
        "peak_rss_bytes": _max_rss_bytes(),
        "details": details or {},
    })
    ledger["phases"] = phases
    ledger["total_elapsed_seconds"] = round(time.time() - float(ledger["started_at_epoch"]), 6)
    _write_json(root / "phase-ledger.json", ledger)


def _split_adaptation(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed ^ 0x16D15A)
    fit: list[int] = []
    validation: list[int] = []
    for stratum in fixtures.STRATA:
        indices = np.asarray([index for index, row in enumerate(records) if row["stratum"] == stratum], dtype=np.int32)
        if indices.size != fixtures.PER_STRATUM_COUNT:
            raise ValueError("V16 adaptation fixture does not have the frozen per-stratum count")
        rng.shuffle(indices)
        fit.extend(int(item) for item in indices[:24])
        validation.extend(int(item) for item in indices[24:])
    return {"fit_prompt_indices": fit, "validation_prompt_indices": validation}


def prepare(custody_path: str | Path) -> dict[str, Any]:
    verify_packet()
    root = _ensure_root(custody_path)
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("V16 candidate freeze already exists; do not rerun preparation in place")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        visible_rows = fixtures.generate_visible_fixture()
        visible = fixtures.fixture_payload(visible_rows, fixture_kind="visible")
        adaptation_payloads = []
        adaptation_rows = []
        splits = []
        for seed_index, seed in enumerate(fixtures.SEEDS):
            rows = fixtures.generate_adaptation_fixture(seed_index)
            payload = fixtures.fixture_payload(rows, fixture_kind=f"adaptation-seed-{seed_index}")
            split = _split_adaptation(payload["records"], seed)
            adaptation_payloads.append(payload)
            adaptation_rows.append(payload["records"])
            splits.append(split)
        fixtures.validate_partition_disjointness(visible, *adaptation_payloads)
        _write_json(root / "fixtures" / "visible.json", visible, immutable=True)
        for seed_index, payload in enumerate(adaptation_payloads):
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
            _write_json(root / "splits" / f"seed-{seed_index}.json", splits[seed_index], immutable=True)

        model, tokenizer = _load_model()
        visible_inputs = _capture_visible_inputs(model, tokenizer, visible["records"])
        variances = [float(np.var(values.astype(np.float64), axis=0).mean()) for values in visible_inputs]
        target_layer = max(range(len(variances)), key=lambda index: (variances[index], -index))
        layer_count = len(_layers(model))
        adjacent_layer = target_layer + 1 if target_layer + 1 < layer_count else target_layer - 1
        if adjacent_layer < 0 or adjacent_layer == target_layer:
            raise ValueError("V16 target has no valid adjacent control layer")
        selection = {
            "protocol_id": PROTOCOL_ID,
            "target_model_id": TARGET_MODEL_ID,
            "target_revision": TARGET_REVISION,
            "target_tree_sha256": TARGET_TREE_SHA256,
            "target_layer": target_layer,
            "adjacent_control_layer": adjacent_layer,
            "selection_rule": "maximum visible-fixture mean per-feature variance of OPT fc1 inputs; ties choose lower layer",
            "visible_fixture_sha256": visible["example_set_sha256"],
            "layer_variances": variances,
        }
        _write_json(root / "block-selection.json", selection, immutable=True)
        all_candidate_rows = []
        runtime = _runtime_receipt()
        for seed_index, (records, split) in enumerate(zip(adaptation_rows, splits, strict=True)):
            _check_caps(root, started)
            captured_rows = _capture_adaptation(model, tokenizer, records, target_layer, adjacent_layer)
            _save_npz_deterministic(root / "adaptation" / f"seed-{seed_index}.npz", **captured_rows)
            fit_indices = set(split["fit_prompt_indices"])
            validation_indices = set(split["validation_prompt_indices"])
            prompt_indices = captured_rows["prompt_index"]
            fit_mask = np.isin(prompt_indices, np.asarray(sorted(fit_indices), dtype=np.int32))
            validation_mask = np.isin(prompt_indices, np.asarray(sorted(validation_indices), dtype=np.int32))
            if not fit_mask.any() or not validation_mask.any() or np.any(fit_mask & validation_mask):
                raise ValueError("V16 fit and validation token partitions are invalid")
            training_seed = fixtures.SEEDS[seed_index] ^ 0x16B10C
            treatment, random_control, treatment_receipt = _train_candidate(
                captured_rows["x_target"][fit_mask],
                captured_rows["y_target"][fit_mask],
                captured_rows["x_target"][validation_mask],
                captured_rows["y_target"][validation_mask],
                seed=training_seed,
                custody_root=root,
            )
            wrong, _wrong_random, wrong_receipt = _train_candidate(
                captured_rows["x_adjacent"][fit_mask],
                captured_rows["y_adjacent"][fit_mask],
                captured_rows["x_adjacent"][validation_mask],
                captured_rows["y_adjacent"][validation_mask],
                seed=training_seed,
                custody_root=root,
            )
            treatment_row = _save_candidate(root / "candidates" / f"seed-{seed_index}-treatment.safetensors", treatment)
            wrong_row = _save_candidate(root / "candidates" / f"seed-{seed_index}-wrong-layer.safetensors", wrong)
            random_row = _save_candidate(root / "candidates" / f"seed-{seed_index}-random.safetensors", random_control)
            for method, row, receipt in (
                ("treatment", treatment_row, treatment_receipt),
                ("wrong_layer", wrong_row, wrong_receipt),
                ("random", random_row, {"source_parameters_copied": False, "trained": False}),
            ):
                all_candidate_rows.append({
                    "seed_index": seed_index,
                    "method": method,
                    **row,
                    "fit_receipt": receipt,
                })
            del captured_rows, treatment, wrong, random_control
        preparation_paths = [
            "fixtures/visible.json",
            "block-selection.json",
            *(f"fixtures/adaptation-seed-{seed_index}.json" for seed_index in range(len(fixtures.SEEDS))),
            *(f"splits/seed-{seed_index}.json" for seed_index in range(len(fixtures.SEEDS))),
            *(f"adaptation/seed-{seed_index}.npz" for seed_index in range(len(fixtures.SEEDS))),
        ]
        preparation_artifacts = [
            {"path": relative, "bytes": (root / relative).stat().st_size, "sha256": _sha_file(root / relative)}
            for relative in sorted(preparation_paths)
        ]
        candidate_material = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": verify_packet()["packet_sha256"],
            "target_tree_sha256": TARGET_TREE_SHA256,
            "block_selection_sha256": _sha_file(root / "block-selection.json"),
            "preparation_artifacts": preparation_artifacts,
            "candidates": all_candidate_rows,
            "training": {
                "architecture": "Linear(768,3072)-ReLU-Linear(3072,768)",
                "updates": TRAIN_UPDATES,
                "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRADIENT_CLIP,
            },
            "runtime_versions": {key: runtime[key] for key in (
                "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256",
                "repository_head", "deterministic_algorithms", "torch_num_threads",
            )},
        }
        candidate_material["candidate_set_sha256"] = _digest(candidate_material)
        candidate_material["candidate_freeze_sha256"] = _digest(candidate_material)
        _write_json(root / "candidate-freeze.json", candidate_material, immutable=True)
        _write_json(root / "runtime.json", runtime, immutable=True)
        _record_phase(root, "prepare", started, "COMPLETED", details={
            "visible_count": visible["count"],
            "adaptation_counts": [payload["count"] for payload in adaptation_payloads],
            "target_layer": target_layer,
            "candidate_count": len(all_candidate_rows),
            "candidate_set_sha256": candidate_material["candidate_set_sha256"],
        })
        return {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "phase": "prepare",
            "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "packet_sha256": verify_packet()["packet_sha256"],
            "repository_head": runtime["repository_head"],
            "target_layer": target_layer,
            "adjacent_control_layer": adjacent_layer,
            "candidate_count": len(all_candidate_rows),
            "candidate_set_sha256": candidate_material["candidate_set_sha256"],
            "data_and_artifact_custody": "all prompts, captures, split IDs, and candidates remain in external owner custody",
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": PROTOCOL_ID,
            "phase": "prepare",
            "status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error": str(error),
            "packet_sha256": verify_packet()["packet_sha256"],
            "timestamp_epoch": time.time(),
        }, immutable=True)
        _record_phase(root, "prepare", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _batch_forward(
    model: torch.nn.Module,
    tokenizer: Any,
    target_layer: int,
    records: list[dict[str, Any]],
    *,
    candidate: WholeMLPCandidate | NoOpCandidate | None,
    positions: list[int],
    deltas: np.ndarray | None = None,
    capture_inputs: bool = False,
    capture_outputs: bool = True,
    capture_all_outputs: bool = False,
    restore_all_outputs: torch.Tensor | None = None,
) -> dict[str, Any]:
    layers = _layers(model)
    layer = layers[target_layer]
    source_fc1, source_fc2 = layer.fc1, layer.fc2
    if candidate is not None:
        layer.fc1, layer.fc2 = candidate.fc1, candidate.fc2
    inputs, checked_positions = _model_inputs(records, tokenizer)
    if checked_positions != positions:
        raise ValueError("V16 intervention positions differ from the prompt boundary")
    batch, sequence = inputs["input_ids"].shape
    flat_positions = _flat_positions(batch, sequence, positions)
    captured_input: list[torch.Tensor] = []
    captured_output: list[torch.Tensor] = []
    all_output: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        activation = args[0]
        if activation.ndim != 2 or activation.shape[0] != batch * sequence:
            raise ValueError("V16 fc1 hook shape differs from the frozen OPT path")
        if capture_inputs:
            captured_input.append(activation.index_select(0, flat_positions).detach().cpu())
        if deltas is None:
            return None
        changed = activation.clone()
        delta_tensor = torch.as_tensor(deltas, dtype=changed.dtype)
        changed[flat_positions] += delta_tensor
        return (changed, *args[1:])

    def post_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise ValueError("V16 fc2 hook shape differs from the frozen OPT path")
        if capture_outputs:
            captured_output.append(output.index_select(0, flat_positions).detach().cpu())
        if capture_all_outputs:
            all_output.append(output.detach().cpu())
        if restore_all_outputs is not None:
            if restore_all_outputs.shape != output.shape:
                raise ValueError("teacher activation rescue tensor has a mismatched shape")
            return restore_all_outputs.to(device=output.device, dtype=output.dtype)
        return output

    pre_handle = layer.fc1.register_forward_pre_hook(pre_hook)
    post_handle = layer.fc2.register_forward_hook(post_hook)
    try:
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, return_dict=True)
    finally:
        pre_handle.remove()
        post_handle.remove()
        layer.fc1, layer.fc2 = source_fc1, source_fc2
    logits = output.logits
    rows_index = torch.arange(batch, dtype=torch.long)
    position_tensor = torch.as_tensor(positions, dtype=torch.long)
    boundary_logits = logits[rows_index, position_tensor].detach().cpu().numpy()
    ce = _continuation_ce(logits, inputs, records)
    return {
        "boundary_logits": boundary_logits,
        "continuation_ce": ce,
        "module_input": captured_input[0].numpy() if captured_input else None,
        "module_output": captured_output[0].numpy() if captured_output else None,
        "all_module_output": all_output[0] if all_output else None,
        "all_logits": logits.detach().cpu() if capture_all_outputs or restore_all_outputs is not None else None,
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }


def _direction_rows(inputs: np.ndarray, records: list[dict[str, Any]], entropy: bytes) -> np.ndarray:
    rng_seed = int.from_bytes(hashlib.sha256(entropy + b"v16-directions").digest()[:8], "big")
    rng = np.random.default_rng(rng_seed)
    deltas = np.zeros_like(inputs, dtype=np.float32)
    for index, (activation, record) in enumerate(zip(inputs, records, strict=True)):
        rms = max(float(np.sqrt(np.mean(np.square(activation, dtype=np.float64)))), 1e-12)
        if record["stratum"] == "adversarial":
            vector = np.zeros_like(activation, dtype=np.float64)
            coordinate = int(np.argmax(np.abs(activation)))
            vector[coordinate] = 1.0 if rng.integers(0, 2) else -1.0
        else:
            vector = rng.normal(size=activation.shape).astype(np.float64)
            if record["stratum"] == "compositional":
                vector += rng.normal(size=activation.shape)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            raise ValueError("V16 challenge direction has zero norm")
        deltas[index] = (vector * (0.01 * rms / norm)).astype(np.float32)
    return deltas


def _candidate_for_method(root: Path, seed_index: int, method: str, manifest: dict[str, Any]) -> WholeMLPCandidate | NoOpCandidate | None:
    if method == "no_op":
        candidate = NoOpCandidate()
        candidate.eval()
        return candidate
    row = next((item for item in manifest["candidates"] if item["seed_index"] == seed_index and item["method"] == method), None)
    if row is None:
        raise ValueError(f"candidate freeze has no method {method} for seed {seed_index}")
    return _load_candidate(root, row)


def _predict_seed(root: Path, seed_index: int, entropy: bytes, target_layer: int, manifest: dict[str, Any]) -> dict[str, Any]:
    challenge_records = fixtures.generate_challenge_fixture(seed_index, entropy)
    payload = fixtures.fixture_payload(challenge_records, fixture_kind=f"challenge-seed-{seed_index}")
    visible = _read_json(root / "fixtures" / "visible.json")
    adaptation = _read_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json")
    fixtures.validate_partition_disjointness(visible, adaptation, payload)
    _write_json(root / "fixtures" / f"challenge-seed-{seed_index}.json", payload, immutable=True)
    model, tokenizer = _load_model()
    token_rows = [_encode(tokenizer, record, with_continuation=True) for record in payload["records"]]
    variants = ("treatment", "wrong_layer", "random", "no_op")
    accumulated: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"module_delta": [], "logit_delta": [], "continuation_ce": [], "module_base": []}
        for name in variants
    }
    treatment_base_logits = []
    direction_arrays = []
    for start in range(0, len(token_rows), INFERENCE_BATCH_SIZE):
        _check_caps(root)
        group = token_rows[start : start + INFERENCE_BATCH_SIZE]
        positions = [int(row["prompt_len"]) - 1 for row in group]
        treatment = _candidate_for_method(root, seed_index, "treatment", manifest)
        treatment_base = _batch_forward(
            model, tokenizer, target_layer, group, candidate=treatment,
            positions=positions, capture_inputs=True, capture_outputs=True,
        )
        directions = _direction_rows(treatment_base["module_input"], group, entropy)
        direction_arrays.append(directions)
        for method, variant in (("treatment", "treatment"), ("wrong_layer", "wrong_layer"), ("random", "random"), ("no_op", "no_op")):
            candidate = treatment if method == "treatment" else _candidate_for_method(root, seed_index, method, manifest)
            base = treatment_base if method == "treatment" else _batch_forward(
                model, tokenizer, target_layer, group, candidate=candidate,
                positions=positions, capture_inputs=False, capture_outputs=True,
            )
            changed = _batch_forward(
                model, tokenizer, target_layer, group, candidate=candidate,
                positions=positions, deltas=directions, capture_inputs=False, capture_outputs=True,
            )
            accumulated[variant]["module_delta"].append(
                (changed["module_output"] - base["module_output"]).astype(np.float32)
            )
            accumulated[variant]["logit_delta"].append(
                (changed["boundary_logits"] - base["boundary_logits"]).astype(np.float32)
            )
            accumulated[variant]["continuation_ce"].append(base["continuation_ce"])
            accumulated[variant]["module_base"].append(base["module_output"].astype(np.float32))
            if method == "treatment":
                treatment_base_logits.append(base["boundary_logits"].astype(np.float32))
    prediction_arrays: dict[str, np.ndarray] = {"directions": np.concatenate(direction_arrays, axis=0)}
    for method in variants:
        for metric, chunks in accumulated[method].items():
            prediction_arrays[f"{method}_{metric}"] = np.concatenate(chunks, axis=0)
    prediction_arrays["treatment_boundary_logits"] = np.concatenate(treatment_base_logits, axis=0)
    path = root / "predictions" / f"seed-{seed_index}.npz"
    _save_npz_deterministic(path, **prediction_arrays)
    return {
        "seed_index": seed_index,
        "challenge_count": len(token_rows),
        "challenge_sha256": payload["example_set_sha256"],
        "prediction_file_sha256": _sha_file(path),
        "prediction_file_bytes": path.stat().st_size,
    }


def predict(custody_path: str | Path, *, entropy_from: str | Path | None = None) -> dict[str, Any]:
    verify_packet()
    root = _ensure_root(custody_path)
    if not (root / "candidate-freeze.json").is_file():
        raise FileNotFoundError("V16 candidate freeze must precede hidden challenge generation")
    if (root / "prediction-manifest.json").exists():
        raise FileExistsError("V16 predictions already exist; do not overwrite a sealed prediction")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        selection = _read_json(root / "block-selection.json")
        candidate_manifest = _verify_candidate_freeze(root)
        entropy_root = root / "challenge-entropy"
        entropy_root.mkdir(parents=True, exist_ok=True)
        if entropy_from is not None:
            source = _custody_root(entropy_from)
            _external_packet_archive(source)
            source_candidates = _verify_candidate_freeze(source)
            _verify_entropy_manifest(source, source_candidates)
            for seed_index in range(len(fixtures.SEEDS)):
                source_file = source / "challenge-entropy" / f"seed-{seed_index}.bin"
                if not source_file.is_file():
                    raise FileNotFoundError("primary challenge entropy is missing for replica")
                destination = entropy_root / source_file.name
                if destination.exists():
                    raise FileExistsError("replica challenge entropy destination already exists")
                shutil.copyfile(source_file, destination)
        else:
            for seed_index in range(len(fixtures.SEEDS)):
                destination = entropy_root / f"seed-{seed_index}.bin"
                if destination.exists():
                    raise FileExistsError("challenge entropy already exists; do not redraw or overwrite it")
                destination.write_bytes(secrets.token_bytes(32))
        entropy_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            secret = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            if len(secret) != 32:
                raise ValueError("V16 challenge entropy file must contain 256 bits")
            entropy_rows.append({"seed_index": seed_index, "sha256": hashlib.sha256(secret).hexdigest(), "bytes": len(secret)})
        _write_json(root / "challenge-entropy-manifest.json", {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": candidate_manifest["candidate_set_sha256"],
            "entropy": entropy_rows,
            "entropy_manifest_sha256": _digest(entropy_rows),
        }, immutable=True)
        _verify_entropy_manifest(root, candidate_manifest)
        run_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            _check_caps(root, started)
            secret = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            run_rows.append(_predict_seed(root, seed_index, secret, int(selection["target_layer"]), candidate_manifest))
        prediction_manifest = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": candidate_manifest["candidate_set_sha256"],
            "entropy_manifest_sha256": _read_json(root / "challenge-entropy-manifest.json")["entropy_manifest_sha256"],
            "target_layer": selection["target_layer"],
            "challenge_rows": run_rows,
        }
        prediction_manifest["prediction_set_sha256"] = _digest(prediction_manifest)
        _write_json(root / "prediction-manifest.json", prediction_manifest, immutable=True)
        _verify_prediction_manifest(root)
        _record_phase(root, "predict", started, "COMPLETED", details={
            "challenge_count": sum(row["challenge_count"] for row in run_rows),
            "prediction_set_sha256": prediction_manifest["prediction_set_sha256"],
        })
        return {
            "protocol_id": PROTOCOL_ID,
            "phase": "predict",
            "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "challenge_count": sum(row["challenge_count"] for row in run_rows),
            "prediction_set_sha256": prediction_manifest["prediction_set_sha256"],
            "candidate_was_frozen_before_challenge": True,
            "challenge_entropy_persisted_externally": True,
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": PROTOCOL_ID, "phase": "predict", "status": "INCONCLUSIVE",
            "error_type": type(error).__name__, "error": str(error),
            "packet_sha256": verify_packet()["packet_sha256"], "timestamp_epoch": time.time(),
        }, immutable=True)
        _record_phase(root, "predict", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _metric_rows(predicted: np.ndarray, observed: np.ndarray, resolvability_scale: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for prediction, outcome, baseline_scale in zip(predicted, observed, resolvability_scale, strict=True):
        pred = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(outcome, dtype=np.float64)
        pred_norm = float(np.linalg.norm(pred))
        target_norm = float(np.linalg.norm(target))
        floor = max(float(baseline_scale) * 1e-8, 1e-12)
        resolvable = target_norm > floor
        dot = float(np.dot(pred, target))
        cosine = dot / max(pred_norm * target_norm, 1e-30) if pred_norm > 1e-30 and target_norm > 1e-30 else 0.0
        rows.append({
            "resolvable": resolvable,
            "direction_correct": bool(dot > 0.0) if resolvable else False,
            "cosine": float(np.clip(cosine, -1.0, 1.0)),
            "normalized_error": float(np.linalg.norm(pred - target) / max(target_norm, floor)),
        })
    return rows


def _summarize_effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
    resolvable = [row for row in rows if row["resolvable"]]
    if not resolvable:
        return {"total_count": len(rows), "resolvable_count": 0, "direction_accuracy": None, "median_cosine": None, "median_normalized_error": None}
    return {
        "total_count": len(rows),
        "resolvable_count": len(resolvable),
        "direction_accuracy": float(np.mean([row["direction_correct"] for row in resolvable])),
        "median_cosine": float(np.median([row["cosine"] for row in resolvable])),
        "median_normalized_error": float(np.median([row["normalized_error"] for row in resolvable])),
    }


def _passes_effect(summary: dict[str, Any]) -> bool:
    return (
        summary["resolvable_count"] >= 16
        and summary["direction_accuracy"] is not None
        and summary["direction_accuracy"] >= 0.80
        and summary["median_cosine"] >= 0.90
        and summary["median_normalized_error"] <= 0.20
    )


def _score_seed(root: Path, seed_index: int, target_layer: int) -> dict[str, Any]:
    payload = _read_json(root / "fixtures" / f"challenge-seed-{seed_index}.json")
    # Tokenize through the frozen local tokenizer after candidate predictions are sealed.
    model, tokenizer = _load_model()
    token_rows = [_encode(tokenizer, record, with_continuation=True) for record in payload["records"]]
    predictions_file = root / "predictions" / f"seed-{seed_index}.npz"
    with np.load(predictions_file, allow_pickle=False) as data:
        prediction = {name: data[name] for name in data.files}
    per_variant: dict[str, list[dict[str, Any]]] = {
        name: [] for name in ("treatment", "wrong_layer", "random", "no_op")
    }
    activation_rows = []
    behavior_rows = []
    oracle_errors = []
    for start in range(0, len(token_rows), INFERENCE_BATCH_SIZE):
        group = token_rows[start : start + INFERENCE_BATCH_SIZE]
        offset = start
        positions = [int(row["prompt_len"]) - 1 for row in group]
        original = _batch_forward(
            model, tokenizer, target_layer, group, candidate=None,
            positions=positions, capture_inputs=True, capture_outputs=True, capture_all_outputs=True,
        )
        directions = prediction["directions"][offset : offset + len(group)]
        observed = _batch_forward(
            model, tokenizer, target_layer, group, candidate=None,
            positions=positions, deltas=directions, capture_outputs=True,
        )
        original_module_delta = observed["module_output"] - original["module_output"]
        original_logit_delta = observed["boundary_logits"] - original["boundary_logits"]
        stratum_rows = payload["records"][offset : offset + len(group)]
        module_scale = np.maximum(np.linalg.norm(original["module_output"], axis=1), 1e-12)
        logit_scale = np.maximum(np.linalg.norm(original["boundary_logits"], axis=1), 1e-12)
        for variant in per_variant:
            predicted_module = prediction[f"{variant}_module_delta"][offset : offset + len(group)]
            predicted_logits = prediction[f"{variant}_logit_delta"][offset : offset + len(group)]
            module_metrics = _metric_rows(predicted_module, original_module_delta, module_scale)
            logit_metrics = _metric_rows(predicted_logits, original_logit_delta, logit_scale)
            for row, module_metric, logit_metric in zip(stratum_rows, module_metrics, logit_metrics, strict=True):
                per_variant[variant].append({
                    "stratum": row["stratum"],
                    "module": module_metric,
                    "logit": logit_metric,
                })
        candidate_module = prediction["treatment_module_base"][offset : offset + len(group)]
        activation_cos = []
        activation_error = []
        for candidate_output, original_output in zip(candidate_module, original["module_output"], strict=True):
            a = np.asarray(candidate_output, dtype=np.float64)
            b = np.asarray(original_output, dtype=np.float64)
            cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
            activation_cos.append(float(np.clip(cosine, -1.0, 1.0)))
            activation_error.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)))
        activation_rows.extend(zip(stratum_rows, activation_cos, activation_error, strict=True))

        # Teacher-output restoration is an oracle pipeline control, not a learned rescue.
        candidate = _candidate_for_method(root, seed_index, "treatment", _verify_candidate_freeze(root))
        oracle = _batch_forward(
            model, tokenizer, target_layer, group, candidate=candidate,
            positions=positions, capture_outputs=False,
            restore_all_outputs=original["all_module_output"],
        )
        active = original["attention_mask"].bool()
        teacher_logits = original["all_logits"]
        oracle_logits = oracle["all_logits"]
        if teacher_logits is None or oracle_logits is None:
            raise RuntimeError("V16 oracle control did not retain reference logits")
        oracle_errors.append(float((teacher_logits[active] - oracle_logits[active]).abs().max().item()))

        base_logits = prediction["treatment_boundary_logits"][offset : offset + len(group)]
        base_ce = prediction["treatment_continuation_ce"][offset : offset + len(group)]
        original_ce = original["continuation_ce"]
        for index, row in enumerate(stratum_rows):
            norm_error = float(
                np.linalg.norm(base_logits[index].astype(np.float64) - original["boundary_logits"][index].astype(np.float64))
                / max(float(np.linalg.norm(original["boundary_logits"][index])), 1e-12)
            )
            behavior_rows.append({
                "stratum": row["stratum"],
                "cross_entropy_increase": float(base_ce[index] - original_ce[index]),
                "normalized_next_logit_error": norm_error,
                "top1_agreement": bool(np.argmax(base_logits[index]) == np.argmax(original["boundary_logits"][index])),
            })
        _check_caps(root)

    result = {"seed_index": seed_index, "strata": {}}
    for stratum in fixtures.STRATA:
        selected_treatment = [row for row in per_variant["treatment"] if row["stratum"] == stratum]
        selected_activation = [(cosine, error) for row, cosine, error in activation_rows if row["stratum"] == stratum]
        selected_behavior = [row for row in behavior_rows if row["stratum"] == stratum]
        result["strata"][stratum] = {
            "challenge_count": len(selected_treatment),
            "treatment_module_effect": _summarize_effect([row["module"] for row in selected_treatment]),
            "treatment_next_logit_effect": _summarize_effect([row["logit"] for row in selected_treatment]),
            "activation": {
                "median_cosine": float(np.median([row[0] for row in selected_activation])),
                "median_normalized_error": float(np.median([row[1] for row in selected_activation])),
            },
            "behavior": {
                "mean_cross_entropy_increase": float(np.mean([row["cross_entropy_increase"] for row in selected_behavior])),
                "median_normalized_next_logit_error": float(np.median([row["normalized_next_logit_error"] for row in selected_behavior])),
                "top1_agreement": float(np.mean([row["top1_agreement"] for row in selected_behavior])),
            },
            "controls": {},
        }
        for variant in ("wrong_layer", "random", "no_op"):
            selected = [row for row in per_variant[variant] if row["stratum"] == stratum]
            result["strata"][stratum]["controls"][variant] = {
                "module_effect": _summarize_effect([row["module"] for row in selected]),
                "next_logit_effect": _summarize_effect([row["logit"] for row in selected]),
            }
    treatment_cos = []
    control_cos = {name: [] for name in ("wrong_layer", "random", "no_op")}
    for variant in per_variant:
        resolved = [row["logit"]["cosine"] for row in per_variant[variant] if row["logit"]["resolvable"]]
        if variant == "treatment":
            treatment_cos = resolved
        else:
            control_cos[variant] = resolved
    control_differences = {
        name: float(np.median(treatment_cos) - np.median(values)) if treatment_cos and values else None
        for name, values in control_cos.items()
    }
    result["control_median_cosine_differences"] = control_differences
    result["oracle_max_absolute_logit_error"] = float(max(oracle_errors)) if oracle_errors else None
    return result


def _scientific_outcome(results: list[dict[str, Any]]) -> str:
    for seed in results:
        if any(
            stratum["treatment_module_effect"]["resolvable_count"] < 16
            or stratum["treatment_next_logit_effect"]["resolvable_count"] < 16
            for stratum in seed["strata"].values()
        ):
            return "INCONCLUSIVE"
    treatment_pass = True
    for seed in results:
        for stratum in seed["strata"].values():
            treatment_pass &= _passes_effect(stratum["treatment_module_effect"])
            treatment_pass &= _passes_effect(stratum["treatment_next_logit_effect"])
            treatment_pass &= stratum["activation"]["median_cosine"] >= 0.90
            treatment_pass &= stratum["activation"]["median_normalized_error"] <= 0.20
            treatment_pass &= stratum["behavior"]["mean_cross_entropy_increase"] <= 0.05
            treatment_pass &= stratum["behavior"]["median_normalized_next_logit_error"] <= 0.05
        treatment_pass &= seed["oracle_max_absolute_logit_error"] <= 1e-5
        treatment_pass &= all(
            difference is not None and difference >= 0.05
            for difference in seed["control_median_cosine_differences"].values()
        )
    return "PASS" if treatment_pass else "FAIL"


def _behavior_mechanism_diverged(results: list[dict[str, Any]]) -> bool:
    return any(
        stratum["behavior"]["mean_cross_entropy_increase"] > 0.05
        or stratum["behavior"]["median_normalized_next_logit_error"] > 0.05
        or stratum["activation"]["median_normalized_error"] > 0.20
        or stratum["treatment_module_effect"]["median_normalized_error"] > 0.20
        or stratum["treatment_next_logit_effect"]["median_normalized_error"] > 0.20
        for seed in results for stratum in seed["strata"].values()
    )


def score(custody_path: str | Path) -> dict[str, Any]:
    verify_packet()
    root = _ensure_root(custody_path)
    if not (root / "prediction-manifest.json").is_file():
        raise FileNotFoundError("V16 sealed predictions must exist before scoring")
    if (root / "outcome-report.json").exists():
        raise FileExistsError("V16 outcomes already exist; do not rescore in place")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        _verify_candidate_freeze(root)
        _verify_prediction_manifest(root)
        selection = _read_json(root / "block-selection.json")
        results = [
            _score_seed(root, seed_index, int(selection["target_layer"]))
            for seed_index in range(len(fixtures.SEEDS))
        ]
        outcome = _scientific_outcome(results)
        manifest = _read_json(root / "candidate-freeze.json")
        prediction_manifest = _read_json(root / "prediction-manifest.json")
        report = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": manifest["candidate_set_sha256"],
            "prediction_set_sha256": prediction_manifest["prediction_set_sha256"],
            "target_identity": {
                "model_id": TARGET_MODEL_ID,
                "revision": TARGET_REVISION,
                "tree_sha256": TARGET_TREE_SHA256,
                "weight_sha256": TARGET_WEIGHT_SHA256,
                "target_layer": selection["target_layer"],
            },
            "scientific_outcome": outcome,
            "behavior_and_mechanism_diverged": _behavior_mechanism_diverged(results),
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "blind_status": "NOT_BLIND_LOCAL_CUSTODY_RUN",
            "results": results,
            "claim_ceiling": CLAIM_CEILING,
            "data_and_artifact_custody": "synthetic prompts, captures, candidates, challenge entropy, predictions, and row-level outcomes remain in external owner custody",
            "runtime": _runtime_receipt(),
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
            "limitations": [
                "one reused outcome-aware OPT/ReLU checkpoint",
                "one visible-selected complete feed-forward block and three synthetic seeds",
                "three seeds on one checkpoint are descriptive, not architecture or checkpoint replications",
                "teacher-output restoration is an oracle pipeline control, not learned activation rescue",
                "same-owner local replay is not blind replication or independent acceptance",
            ],
            "next_blocked_authority_gate": "blind external replication and an authorized independent reviewer are required for independent scientific acceptance",
        }
        scientific_material = {key: value for key, value in report.items() if key not in {"runtime", "assurance_level", "reviewer_status", "reproducibility_status", "blind_status"}}
        report["scientific_results_sha256"] = _digest(scientific_material)
        report["report_sha256"] = _digest(report)
        _write_json(root / "outcome-report.json", report, immutable=True)
        _record_phase(root, "score", started, "COMPLETED", details={
            "scientific_outcome": outcome,
            "scientific_results_sha256": report["scientific_results_sha256"],
        })
        return {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "phase": "score",
            "status": "COMPLETED",
            "scientific_outcome": outcome,
            "scientific_results_sha256": report["scientific_results_sha256"],
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "repository_head": report["runtime"]["repository_head"],
            "data_and_artifact_custody": report["data_and_artifact_custody"],
            "target_layer": selection["target_layer"],
            "behavior_and_mechanism_diverged": _behavior_mechanism_diverged(results),
            "claim_ceiling": CLAIM_CEILING,
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": PROTOCOL_ID, "phase": "score", "status": "INCONCLUSIVE",
            "error_type": type(error).__name__, "error": str(error),
            "packet_sha256": verify_packet()["packet_sha256"], "timestamp_epoch": time.time(),
        }, immutable=True)
        _record_phase(root, "score", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def preflight() -> dict[str, Any]:
    packet = verify_packet()
    target = _target_tree_receipt()
    model, _tokenizer = _load_model()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    layers = _layers(model)
    layer_contracts = []
    for layer in layers:
        if layer.fc1.in_features != 768 or layer.fc1.out_features != 3072 or layer.fc2.in_features != 3072 or layer.fc2.out_features != 768:
            raise ValueError("an OPT block differs from the frozen 768-3072-768 MLP contract")
        layer_contracts.append({"input_dim": 768, "ffn_dim": 3072, "output_dim": 768})
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "target_identity": {"model_id": TARGET_MODEL_ID, "revision": TARGET_REVISION, **target},
        "architecture": {"family": "OPT/ReLU", "layer_count": len(layers), "parameter_count": parameter_count, "blocks": layer_contracts},
        "runtime": _runtime_receipt(),
        "wall_cap_seconds": MAX_WALL_SECONDS,
        "rss_cap_bytes": MAX_RSS_BYTES,
    }


def _artifact_group(root: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"replication artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _relative_files(root: Path, prefix: str) -> list[str]:
    base = root / prefix
    if not base.exists():
        return []
    return [path.relative_to(root).as_posix() for path in sorted(base.rglob("*")) if path.is_file()]


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path)
    replica = _ensure_root(replica_path)
    primary_archive = _external_packet_archive(primary)
    replica_archive = _external_packet_archive(replica)
    if primary_archive["archive_sha256"] != replica_archive["archive_sha256"]:
        raise ValueError("primary and replica packet archives differ")
    primary_report = _read_json(primary / "outcome-report.json")
    replica_report = _read_json(replica / "outcome-report.json")
    groups_to_paths = {
        "fixtures": _relative_files(primary, "fixtures"),
        "splits": _relative_files(primary, "splits"),
        "captures": _relative_files(primary, "adaptation"),
        "candidates": _relative_files(primary, "candidates") + ["candidate-freeze.json"],
        "challenge_entropy": _relative_files(primary, "challenge-entropy") + ["challenge-entropy-manifest.json"],
        "predictions": _relative_files(primary, "predictions") + ["prediction-manifest.json"],
    }
    groups = {}
    for name, paths in groups_to_paths.items():
        left = _artifact_group(primary, paths)
        right = _artifact_group(replica, paths)
        groups[name] = {"primary_count": len(left), "replica_count": len(right), "primary_sha256": _digest(left), "replica_sha256": _digest(right), "byte_identical": left == right}
    result_match = primary_report["scientific_results_sha256"] == replica_report["scientific_results_sha256"]
    primary_runtime = _read_json(primary / "runtime.json")
    replica_runtime = _read_json(replica / "runtime.json")
    pin_fields = ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "deterministic_algorithms", "torch_num_threads")
    pins_match = all(primary_runtime.get(field) == replica_runtime.get(field) for field in pin_fields)
    separate_runtime = primary_runtime.get("python_executable") != replica_runtime.get("python_executable")
    same_host = primary_runtime.get("host_id_sha256") == replica_runtime.get("host_id_sha256")
    same_owner = primary_runtime.get("uid") == replica_runtime.get("uid")
    all_groups_match = all(row["byte_identical"] for row in groups.values())
    reproduced = all_groups_match and result_match and pins_match and separate_runtime and same_host and same_owner
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "packet_sha256": verify_packet()["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "scientific_outcome": primary_report["scientific_outcome"] if result_match else "INCONCLUSIVE",
        "scientific_results_match": result_match,
        "scientific_results_sha256": [primary_report["scientific_results_sha256"], replica_report["scientific_results_sha256"]],
        "artifact_groups": groups,
        "runtime_pins_match": pins_match,
        "separate_python_runtimes": separate_runtime,
        "same_host_and_owner": same_host and same_owner,
        "archive_sha256": [primary_archive["archive_sha256"], replica_archive["archive_sha256"]],
        "claim_ceiling": CLAIM_CEILING,
        "custody": "all prompts, captures, candidates, entropy, predictions, and row-level outcomes remain under external owner custody",
        "next_blocked_authority_gate": "blind external replication and authorized independent review",
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "replication-report.json", report, immutable=True)
    primary_report["reproducibility_status"] = report["assurance_level"]
    primary_report["replication_report_sha256"] = report["report_sha256"]
    primary_report.pop("report_sha256", None)
    primary_report["report_sha256"] = _digest(primary_report)
    _write_json(primary / "outcome-report.json", primary_report)
    return report


def replicate(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path)
    replica = _custody_root(replica_path)
    if replica.exists():
        raise FileExistsError("replica custody root already exists; preserve it and choose a fresh root")
    replica.mkdir(parents=True, exist_ok=False)
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(primary / "frozen-packet-manifest.json", replica / "frozen-packet-manifest.json")
    archive = _external_packet_archive(replica)
    prepare(replica)
    predict(replica, entropy_from=primary)
    score(replica)
    comparison = compare_replication(primary, replica)
    return {
        "protocol_id": PROTOCOL_ID,
        "replication_status": comparison["assurance_level"],
        "scientific_outcome": comparison["scientific_outcome"],
        "archive_sha256": archive["archive_sha256"],
        "replication_report_sha256": comparison["report_sha256"],
    }


def _phase_with_stop(command: str, action: Any, custody: str | None) -> Any:
    try:
        return action()
    except Exception as error:
        if custody is not None:
            try:
                root = _custody_root(custody)
                if root.exists() and not (root / "run-stop.json").exists():
                    _write_json(root / "run-stop.json", {
                        "protocol_id": PROTOCOL_ID,
                        "phase": command,
                        "status": "INCONCLUSIVE",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "timestamp_epoch": time.time(),
                    }, immutable=True)
            except Exception:
                pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "preflight"):
        commands.add_parser(name)
    archive = commands.add_parser("archive")
    archive.add_argument("--custody", required=True)
    prepare_command = commands.add_parser("prepare")
    prepare_command.add_argument("--custody", required=True)
    predict_command = commands.add_parser("predict")
    predict_command.add_argument("--custody", required=True)
    predict_command.add_argument("--entropy-from")
    score_command = commands.add_parser("score")
    score_command.add_argument("--custody", required=True)
    replicate_command = commands.add_parser("replicate")
    replicate_command.add_argument("--primary", required=True)
    replicate_command.add_argument("--replica", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "verify":
        packet = verify_packet()
        result = {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "freeze_sha256": packet["freeze_sha256"], "packet_file_count": len(packet["packet_files"])}
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "prepare":
        result = _phase_with_stop(args.command, lambda: prepare(args.custody), args.custody)
    elif args.command == "predict":
        result = _phase_with_stop(args.command, lambda: predict(args.custody, entropy_from=args.entropy_from), args.custody)
    elif args.command == "score":
        result = _phase_with_stop(args.command, lambda: score(args.custody), args.custody)
    elif args.command == "replicate":
        result = _phase_with_stop(args.command, lambda: replicate(args.primary, args.replica), args.replica)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
