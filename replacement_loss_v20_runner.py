"""Frozen V20 low-rank causal-residual experiment and local replay.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v20-low-rank-causal-residual-v1, and
global-experiment-assurance-lifecycle-v1.

All prompt-bearing artifacts remain in external owner custody.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import replacement_loss_v20_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
DEV_PROTOCOL_ID = fixtures.DEV_PROTOCOL_ID
CONFIRM_PROTOCOL_ID = fixtures.CONFIRM_PROTOCOL_ID
DEV_FREEZE_FILE = "docs/replacement-loss-v20-development-freeze.json"
CONFIRM_FREEZE_FILE = "docs/replacement-loss-v20-confirmation-freeze.json"
CONFIRM_CONFIG_FILE = "docs/replacement-loss-v20-confirmation-config.json"
DEV_RESULT_JSON = "docs/replacement-loss-v20-development-results.json"
DEV_RESULT_MD = "docs/replacement-loss-v20-development-results.md"
DEV_REPLICATION_JSON = "docs/replacement-loss-v20-development-replication.json"
CONFIRM_RESULT_JSON = "docs/replacement-loss-v20-confirmation-results.json"
CONFIRM_RESULT_MD = "docs/replacement-loss-v20-confirmation-results.md"
CONFIRM_REPLICATION_JSON = "docs/replacement-loss-v20-confirmation-replication.json"
CLAIM_CEILING = "LOCAL_LOW_RANK_CAUSAL_RESIDUAL_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT"
TARGET_LAYER = 8
ADJACENT_LAYER = 9
RANK = 16
TRAIN_UPDATES = 1000
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
ANCHOR_WEIGHT = 1.0
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
RESEARCH_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
DEV_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20-development-primary"
DEV_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20-development-replica"
CONFIRM_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20-confirmation-primary"
CONFIRM_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20-confirmation-replica"
RUNTIME_DIR = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v15-runtimes"
PRIMARY_PYTHON = RUNTIME_DIR / "primary/bin/python"
REPLICA_PYTHON = RUNTIME_DIR / "replica/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
PROTOCOL_FILE = "docs/replacement-loss-v20-low-rank-causal-residual.md"
FIXTURE_FILE = "replacement_loss_v20_fixtures.py"
RUNNER_FILE = "replacement_loss_v20_runner.py"
LOCK_FILE = "requirements-v20.lock.txt"

PARENT_FILES = (
    "docs/replacement-loss-v19-development-freeze.json",
    "docs/replacement-loss-v19-development-results.json",
    "docs/replacement-loss-v19-development-results.md",
    "docs/replacement-loss-v19-development-replication.json",
    "docs/replacement-loss-v19-development.md",
    "docs/replacement-loss-v19-confirmatory.md",
    "replacement_loss_v19_fixtures.py",
    "replacement_loss_v19_runner.py",
    "requirements-v19.lock.txt",
    "docs/replacement-loss-v18-freeze.json",
    "docs/replacement-loss-v18-results.md",
    "docs/replacement-loss-v18-projected-causal-refinement.md",
    "replacement_loss_v18_fixtures.py",
    "replacement_loss_v18_runner.py",
    "requirements-v18.lock.txt",
    "docs/replacement-loss-v17-freeze.json",
    "docs/replacement-loss-v17-results.md",
    "docs/replacement-loss-v17-causal-supervision.md",
    "replacement_loss_v17_fixtures.py",
    "replacement_loss_v17_runner.py",
    "requirements-v17.lock.txt",
    "docs/replacement-loss-v16-freeze.json",
    "docs/replacement-loss-v16-results.md",
    "docs/replacement-loss-v16-whole-mlp.md",
    "replacement_loss_v16_fixtures.py",
    "replacement_loss_v16_runner.py",
    "requirements-v16.lock.txt",
    "replacement_loss_v3_fixtures.py",
)
DEV_PACKET_FILES = (
    "AGENTS.md", PROTOCOL_FILE, FIXTURE_FILE, RUNNER_FILE, LOCK_FILE, *PARENT_FILES,
)
CONFIRM_PACKET_FILES = (
    "AGENTS.md", PROTOCOL_FILE, FIXTURE_FILE, RUNNER_FILE, LOCK_FILE,
    DEV_FREEZE_FILE, CONFIRM_CONFIG_FILE, DEV_RESULT_JSON, DEV_RESULT_MD,
    DEV_REPLICATION_JSON, *PARENT_FILES,
)

np: Any = None
torch: Any = None
F: Any = None
v16: Any = None
v17: Any = None
v19: Any = None
_DEPS_READY = False


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable V20 artifact already exists with different bytes: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _rows_for(files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in files:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V20 frozen source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def _verify_frozen_file(relative: str) -> dict[str, Any]:
    frozen = _read_json(REPO_ROOT / relative)
    rows = frozen.get("packet_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"V20 frozen packet has no source rows: {relative}")
    current = []
    for row in rows:
        path = REPO_ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {row['path']}")
        current.append({"path": row["path"], "sha256": _sha_file(path), "bytes": path.stat().st_size})
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if rows != current or frozen.get("packet_sha256") != _digest(current) or frozen.get("freeze_sha256") != _digest(material):
        raise ValueError(f"V20 frozen packet bytes or digest differ: {relative}")
    return frozen


def _verify_v19_parent() -> dict[str, Any]:
    import replacement_loss_v19_runner as v19_parent

    parent = v19_parent.verify_development_packet()
    result = _read_json(REPO_ROOT / "docs/replacement-loss-v19-development-results.json")
    replay = _read_json(REPO_ROOT / "docs/replacement-loss-v19-development-replication.json")
    if (
        result.get("packet_sha256") != parent.get("packet_sha256")
        or result.get("qualification_status") != "NO_QUALIFIED_DEVELOPMENT_CANDIDATE"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_results_sha256") != result.get("scientific_results_sha256")
    ):
        raise ValueError("V20 continuation does not match the frozen V19 failed-development/replay record")
    return parent


def freeze_development() -> dict[str, Any]:
    path = REPO_ROOT / DEV_FREEZE_FILE
    if path.exists():
        return verify_development_packet()
    parent = _verify_v19_parent()
    rows = _rows_for(DEV_PACKET_FILES)
    packet = {
        "protocol_id": DEV_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "parent_protocol": {
            "protocol_id": parent["protocol_id"],
            "packet_sha256": parent["packet_sha256"],
            "v19_qualification_status": "NO_QUALIFIED_DEVELOPMENT_CANDIDATE",
            "v19_reproducibility_status": "REPRODUCED_LOCAL",
            "v19_challenge_data_used_by_v20": False,
            "new_method": "rank-16 low-rank causal residual anchored to a separately trained pointwise base",
        },
        "target_identity": {
            "model_id": "grushaprasad/opt125m-rootp-surprisal-curriculum",
            "revision": "8af2a2926bc772599ccb3b4e7ad7b40b35259a63",
            "tree_sha256": "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466",
            "weight_sha256": "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132",
            "target_known_before_protocol": True,
            "fresh_checkpoint_claim": False,
            "selected_layer": TARGET_LAYER,
        },
        "qualification": {
            "seeds": list(fixtures.DEVELOPMENT_SEEDS),
            "strata": list(fixtures.STRATA),
            "cells": 9,
            "effect_ratio_max": 0.90,
            "value_ratio_max": 1.25,
            "continuation_ce_increase_max": 0.05,
            "normalized_next_logit_error_max": 0.05,
            "low_rank": RANK,
            "updates_base_and_residual": TRAIN_UPDATES,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_development_packet() -> dict[str, Any]:
    parent = _verify_v19_parent()
    frozen = _read_json(REPO_ROOT / DEV_FREEZE_FILE)
    current = _rows_for(DEV_PACKET_FILES)
    expected_parent = {
        "protocol_id": parent["protocol_id"],
        "packet_sha256": parent["packet_sha256"],
        "v19_qualification_status": "NO_QUALIFIED_DEVELOPMENT_CANDIDATE",
        "v19_reproducibility_status": "REPRODUCED_LOCAL",
        "v19_challenge_data_used_by_v20": False,
        "new_method": "rank-16 low-rank causal residual anchored to a separately trained pointwise base",
    }
    expected_qualification = {
        "seeds": list(fixtures.DEVELOPMENT_SEEDS), "strata": list(fixtures.STRATA), "cells": 9,
        "effect_ratio_max": 0.90, "value_ratio_max": 1.25,
        "continuation_ce_increase_max": 0.05, "normalized_next_logit_error_max": 0.05,
        "low_rank": RANK, "updates_base_and_residual": TRAIN_UPDATES,
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != DEV_PROTOCOL_ID or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current) or frozen.get("parent_protocol") != expected_parent
        or frozen.get("qualification") != expected_qualification or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V20 development packet identity or source bytes differ from its freeze")
    return frozen


def freeze_confirmation() -> dict[str, Any]:
    development = verify_development_packet()
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    if (
        result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_results_sha256") != result.get("scientific_results_sha256")
    ):
        raise ValueError("V20 confirmation requires a qualified development result and byte-identical local replay")
    config = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "development_packet_sha256": development["packet_sha256"],
        "development_freeze_sha256": development["freeze_sha256"],
        "development_result_sha256": result["scientific_results_sha256"],
        "development_replication_sha256": replay["report_sha256"],
        "seeds": list(fixtures.CONFIRMATION_SEEDS),
        "challenge_unlock_condition": "primary and replica preparations are byte-identical and all nine confirmation cells pass every prechallenge gate",
    }
    config["config_sha256"] = _digest(config)
    _write_json(REPO_ROOT / CONFIRM_CONFIG_FILE, config, immutable=True)
    path = REPO_ROOT / CONFIRM_FREEZE_FILE
    if path.exists():
        return verify_confirmation_packet()
    rows = _rows_for(CONFIRM_PACKET_FILES)
    packet = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "development_binding": {
            "packet_sha256": development["packet_sha256"],
            "result_sha256": result["scientific_results_sha256"],
            "replication_sha256": replay["report_sha256"],
            "qualification_status": result["qualification_status"],
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_confirmation_packet() -> dict[str, Any]:
    development = verify_development_packet()
    config = _read_json(REPO_ROOT / CONFIRM_CONFIG_FILE)
    config_material = {key: value for key, value in config.items() if key != "config_sha256"}
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    if (
        config.get("config_sha256") != _digest(config_material)
        or result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
    ):
        raise ValueError("V20 confirmation development entry gate is invalid")
    frozen = _read_json(REPO_ROOT / CONFIRM_FREEZE_FILE)
    current = _rows_for(CONFIRM_PACKET_FILES)
    expected_binding = {
        "packet_sha256": development["packet_sha256"],
        "result_sha256": result["scientific_results_sha256"],
        "replication_sha256": replay["report_sha256"],
        "qualification_status": "QUALIFIED",
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != CONFIRM_PROTOCOL_ID or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current) or frozen.get("development_binding") != expected_binding
        or config.get("development_packet_sha256") != development["packet_sha256"]
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V20 confirmation packet identity or source bytes differ from its freeze")
    return frozen


def _packet(stage: str) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    if stage == "development":
        return DEV_PROTOCOL_ID, DEV_FREEZE_FILE, DEV_PACKET_FILES, verify_development_packet()
    if stage == "confirmation":
        return CONFIRM_PROTOCOL_ID, CONFIRM_FREEZE_FILE, CONFIRM_PACKET_FILES, verify_confirmation_packet()
    raise ValueError("stage must be development or confirmation")


def _custody_root(value: str | Path, stage: str, *, replica: bool = False) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError("V20 external custody roots cannot be symlinks")
    root = candidate.resolve()
    expected = {
        ("development", False): DEV_PRIMARY_NAME, ("development", True): DEV_REPLICA_NAME,
        ("confirmation", False): CONFIRM_PRIMARY_NAME, ("confirmation", True): CONFIRM_REPLICA_NAME,
    }[(stage, replica)]
    if root.parent != RESEARCH_ROOT or root.name != expected or root.is_symlink():
        raise ValueError(f"V20 {stage} custody root differs from frozen external owner-custody path")
    return root


def _archive_packet(root: Path, stage: str) -> dict[str, Any]:
    protocol_id, freeze_file, _files, packet = _packet(stage)
    if root.exists():
        raise FileExistsError(f"V20 {stage} custody root already exists; preserve it and do not reuse")
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "frozen-packet"
    rows = []
    for row in packet["packet_files"]:
        source, target = REPO_ROOT / row["path"], archive / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append(dict(row))
    source, target = REPO_ROOT / freeze_file, archive / freeze_file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    rows.append({"path": freeze_file, "sha256": _sha_file(target), "bytes": target.stat().st_size})
    manifest = {
        "protocol_id": protocol_id, "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"], "files": rows, "archive_sha256": _digest(rows),
    }
    _write_json(root / "frozen-packet-manifest.json", manifest, immutable=True)
    return {"packet_sha256": packet["packet_sha256"], "archive_sha256": manifest["archive_sha256"]}


def _verify_archive(root: Path, stage: str) -> dict[str, Any]:
    protocol_id, freeze_file, _files, packet = _packet(stage)
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != packet["packet_sha256"] or manifest.get("protocol_id") != protocol_id:
        raise ValueError(f"V20 external {stage} archive binds another packet")
    expected = []
    for row in packet["packet_files"]:
        path = root / "frozen-packet" / row["path"]
        actual = {"path": row["path"], "sha256": _sha_file(path), "bytes": path.stat().st_size} if path.is_file() else None
        if actual != row:
            raise ValueError(f"V20 external packet source differs: {row['path']}")
        expected.append(actual)
    freeze_copy = root / "frozen-packet" / freeze_file
    if not freeze_copy.is_file() or _sha_file(freeze_copy) != _sha_file(REPO_ROOT / freeze_file):
        raise ValueError(f"V20 archived {stage} freeze bytes differ")
    expected.append({"path": freeze_file, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected or manifest.get("archive_sha256") != _digest(expected):
        raise ValueError(f"V20 external {stage} archive manifest is invalid")
    return manifest


def _ensure_root(value: str | Path, stage: str, *, replica: bool = False) -> Path:
    root = _custody_root(value, stage, replica=replica)
    if not root.is_dir():
        raise FileNotFoundError(f"V20 {stage} external custody root is missing")
    if (root / "run-stop.json").exists():
        raise RuntimeError(f"V20 {stage} root has a terminal stop receipt; do not reuse")
    _verify_archive(root, stage)
    return root


def _load_dependencies() -> None:
    global np, torch, F, v16, v17, v19, _DEPS_READY
    if _DEPS_READY:
        return
    import numpy as numpy_module
    import torch as torch_module
    import torch.nn.functional as functional_module
    import replacement_loss_v16_runner as v16_module
    import replacement_loss_v17_runner as v17_module
    import replacement_loss_v19_runner as v19_module

    np, torch, F, v16, v17, v19 = numpy_module, torch_module, functional_module, v16_module, v17_module, v19_module
    v19._load_dependencies()
    v19.fixtures = fixtures
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _DEPS_READY = True


def _parse_lock(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator:
            raise ValueError(f"invalid exact dependency pin in {path.name}")
        pins[name.casefold().replace("_", "-")] = version
    return pins


def _runtime_receipt() -> dict[str, Any]:
    _load_dependencies()
    if _parse_lock(REPO_ROOT / LOCK_FILE) != _parse_lock(REPO_ROOT / "requirements-v16.lock.txt"):
        raise ValueError("V20 lock differs from the reproduced V16/V17 runtime inventory")
    receipt = v19._runtime_receipt()
    receipt["dependency_lock_sha256"] = _sha_file(REPO_ROOT / LOCK_FILE)
    receipt["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    receipt["torch_num_threads"] = int(torch.get_num_threads())
    if receipt.get("python") != "3.12.13" or not receipt["deterministic_algorithms"] or receipt["torch_num_threads"] != 1:
        raise ValueError("V20 runtime differs from pinned Python 3.12.13 deterministic CPU contract")
    return receipt


def _check_caps(root: Path, started: float) -> None:
    if time.monotonic() - started > MAX_WALL_SECONDS:
        raise TimeoutError("V20 phase exceeded the frozen four-hour wall-clock cap")
    if v16._max_rss_bytes() > MAX_RSS_BYTES:
        raise MemoryError("V20 process exceeded the frozen 16-GiB RSS cap")


def _split(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed ^ 0x20D15A)
    fit: list[int] = []
    validation: list[int] = []
    for stratum in fixtures.STRATA:
        indices = np.asarray([i for i, row in enumerate(records) if row["stratum"] == stratum], dtype=np.int32)
        if len(indices) != fixtures.ADAPTATION_PER_STRATUM:
            raise ValueError("V20 adaptation fixture count differs from frozen stratum size")
        rng.shuffle(indices)
        fit.extend(int(index) for index in indices[: fixtures.FIT_PER_STRATUM])
        validation.extend(int(index) for index in indices[fixtures.FIT_PER_STRATUM :])
    if len(fit) != 144 or len(validation) != 48 or set(fit) & set(validation):
        raise ValueError("V20 fit and validation prompts are not a complete disjoint partition")
    return {"fit_prompt_indices": fit, "validation_prompt_indices": validation}


def _training_set(layer: Any, captures: dict[str, Any], records: list[dict[str, Any]], prompt_set: set[int], *, seed: int, label: str) -> dict[str, Any]:
    mask = np.isin(captures["prompt_index"], np.asarray(sorted(prompt_set), dtype=np.int32))
    data = v19._training_set(layer, captures, records, prompt_set, seed=seed, label=label)
    data["prompt_index"] = captures["prompt_index"][mask].copy()
    return data


def _factorized_candidate(base: Any, seed: int) -> Any:
    input_dim, hidden_dim, output_dim = base.fc1.in_features, base.fc1.out_features, base.fc2.out_features
    class FactorizedResidual(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = base
            self.a1 = torch.nn.Parameter(torch.empty(RANK, input_dim))
            self.b1 = torch.nn.Parameter(torch.zeros(hidden_dim, RANK))
            self.a2 = torch.nn.Parameter(torch.empty(RANK, hidden_dim))
            self.b2 = torch.nn.Parameter(torch.zeros(output_dim, RANK))
            torch.nn.init.kaiming_uniform_(self.a1, a=5 ** 0.5)
            torch.nn.init.kaiming_uniform_(self.a2, a=5 ** 0.5)
            for parameter in self.base.parameters():
                parameter.requires_grad_(False)
            self.base.eval()

        def forward(self, inputs: Any) -> Any:
            hidden = F.relu(self.base.fc1(inputs) + F.linear(F.linear(inputs, self.a1), self.b1))
            return self.base.fc2(hidden) + F.linear(F.linear(hidden, self.a2), self.b2)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed) ^ 0x20FACADE)
        return FactorizedResidual()


def _train_residual(base: Any, data: dict[str, Any], *, seed: int, root: Path, started: float) -> tuple[Any, dict[str, Any]]:
    x, delta, effect = (torch.from_numpy(data[key]) for key in ("x", "delta", "effect"))
    residual = _factorized_candidate(base, seed)
    parameters = [residual.a1, residual.b1, residual.a2, residual.b2]
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    effect_norm = max(float(effect.square().mean().item()), 1e-12)
    with torch.inference_mode():
        base_output = base(x).detach()
    value_norm = max(float(base_output.square().mean().item()), 1e-12)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x20A5C3)
    draws = np.empty((TRAIN_UPDATES, TRAIN_BATCH_SIZE), dtype=np.int32)
    residual.train()
    for update in range(TRAIN_UPDATES):
        indices = torch.randint(0, x.shape[0], (TRAIN_BATCH_SIZE,), generator=generator)
        draws[update] = indices.numpy()
        xb, db, eb = x.index_select(0, indices), delta.index_select(0, indices), effect.index_select(0, indices)
        with torch.no_grad():
            base_value = base(xb)
        prediction = residual(xb)
        effect_prediction = residual(xb + db) - prediction
        effect_loss = F.mse_loss(effect_prediction, eb) / effect_norm
        anchor_loss = F.mse_loss(prediction, base_value) / value_norm
        loss = effect_loss + ANCHOR_WEIGHT * anchor_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V20 low-rank residual objective")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP)
        optimizer.step()
        if (update + 1) % 100 == 0:
            _check_caps(root, started)
    residual.eval()
    dense = v16.WholeMLPCandidate()
    dense.load_state_dict(base.state_dict(), strict=True)
    with torch.no_grad():
        dense.fc1.weight.copy_(base.fc1.weight + residual.b1 @ residual.a1)
        dense.fc2.weight.copy_(base.fc2.weight + residual.b2 @ residual.a2)
        probe = x[: min(32, len(x))]
        max_merge_error = float((residual(probe) - dense(probe)).abs().max().item())
    if max_merge_error > 1e-5:
        raise ValueError("V20 merged dense candidate does not match its trained factorized residual")
    dense.eval()
    receipt = {
        "base_state_sha256": v16._module_state_digest(base),
        "residual_state_sha256": v16._module_state_digest(residual),
        "merged_state_sha256": v16._module_state_digest(dense),
        "factor_rank": RANK,
        "updates": TRAIN_UPDATES,
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(draws.astype("<i4", copy=False).tobytes()).hexdigest(),
        "objective": "normalized finite-effect MSE + 1.0 * normalized squared drift from frozen pointwise base output",
        "effect_normalizer": effect_norm,
        "base_output_normalizer": value_norm,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP,
        "anchor_weight": ANCHOR_WEIGHT,
        "factorized_to_merged_probe_max_abs_error": max_merge_error,
        "source_parameters_copied": False,
        "base_parameters_frozen": True,
        "fit_input_sha256": hashlib.sha256(data["x"].astype("<f4", copy=False).tobytes()).hexdigest(),
        "fit_effect_target_sha256": hashlib.sha256(data["effect"].astype("<f4", copy=False).tobytes()).hexdigest(),
    }
    return dense, receipt


def _train_pair(layer: Any, fit: dict[str, Any], validation: dict[str, Any], *, seed: int, root: Path, started: float) -> tuple[Any, Any, Any, dict[str, Any]]:
    base, random_control, base_receipt = v19._train(fit, validation, seed=seed, causal_weight=0.0, root=root, started=started)
    residual, residual_receipt = _train_residual(base, fit, seed=seed, root=root, started=started)
    return residual, base, random_control, {"pointwise_base": base_receipt, "causal_residual": residual_receipt}


def _artifact_rows(root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V20 preparation artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _selection(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells = [
        (row["seed_index"], stratum, row["candidates"]["causal"][stratum])
        for row in seed_rows for stratum in fixtures.STRATA
    ]
    gates = {
        "nine_seed_stratum_cells": len(cells) == 9,
        "effect_reduction_all_cells": len(cells) == 9 and all(cell[2]["effect_mse_ratio_zero"] <= 0.90 for cell in cells),
        "value_retention_all_cells": len(cells) == 9 and all(cell[2]["value_mse_ratio_pointwise"] <= 1.25 for cell in cells),
        "continuation_behavior_all_cells": len(cells) == 9 and all(cell[2]["validation_cross_entropy_increase"] <= 0.05 for cell in cells),
        "next_logit_behavior_all_cells": len(cells) == 9 and all(cell[2]["median_normalized_next_logit_error"] <= 0.05 for cell in cells),
    }
    ratios = [float(cell[2]["effect_mse_ratio_zero"]) for cell in cells]
    return {
        "qualified": all(gates.values()),
        "gates": gates,
        "equal_cell_mean_effect_mse_ratio": float(np.mean(ratios)) if ratios else None,
        "worst_cell_effect_mse_ratio": max(ratios) if ratios else None,
        "cell_count": len(cells),
        "cells": [
            {"seed_index": seed_index, "stratum": stratum, **metrics}
            for seed_index, stratum, metrics in cells
        ],
    }


def _runtime_versions(runtime: dict[str, Any]) -> dict[str, Any]:
    return {key: runtime.get(key) for key in (
        "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256",
        "repository_head", "deterministic_algorithms", "torch_num_threads",
    )}


def _run_preparation(stage: str, custody: str | Path, *, replica: bool = False) -> dict[str, Any]:
    protocol_id, _freeze, _files, packet = _packet(stage)
    root = _custody_root(custody, stage, replica=replica)
    if root.exists():
        _ensure_root(root, stage, replica=replica)
    else:
        _archive_packet(root, stage)
    if (root / "preparation-selection.json").exists():
        raise FileExistsError(f"V20 {stage} preparation already exists; preserve the root and do not rerun")
    _load_dependencies()
    started = time.monotonic()
    try:
        runtime = _runtime_receipt()
        target = v16._target_tree_receipt()
        if target.get("tree_sha256") != v16.TARGET_TREE_SHA256 or target.get("weight_sha256") != v16.TARGET_WEIGHT_SHA256:
            raise ValueError("V20 target checkpoint differs from frozen OPT identity")
        visible, old_adaptations = v19._reference_payloads()
        fixtures.activate_stage(stage)
        seeds = fixtures.seeds_for(stage)
        adaptations = []
        splits = []
        for seed_index, seed in enumerate(seeds):
            rows = fixtures.generate_adaptation_fixture(stage, seed_index)
            payload = fixtures.fixture_payload(rows, fixture_kind=f"adaptation-seed-{seed_index}", stage=stage)
            adaptations.append(payload)
            splits.append(_split(payload["records"], seed))
        fixtures.set_extra_disjoint_payloads(old_adaptations)
        fixtures.validate_partition_disjointness(visible, *adaptations)
        if stage == "confirmation":
            dev_primary = RESEARCH_ROOT / DEV_PRIMARY_NAME
            if not dev_primary.is_dir():
                raise FileNotFoundError("V20 development fixtures are missing from external custody")
            dev_adaptations = [
                _read_json(dev_primary / "fixtures" / f"adaptation-seed-{index}.json") for index in range(3)
            ]
            fixtures.validate_partition_disjointness(visible, *dev_adaptations, *adaptations)

        model, tokenizer = v16._load_model()
        layers = v16._layers(model)
        v16_root = RESEARCH_ROOT / v16.PRIMARY_NAME
        visible_v16 = _read_json(v16_root / "fixtures" / "visible.json")
        selection_path = v16_root / "block-selection.json"
        block_selection = _read_json(selection_path)
        if block_selection.get("target_layer") != TARGET_LAYER or block_selection.get("visible_fixture_sha256") != visible_v16.get("example_set_sha256"):
            raise ValueError("V20 target block differs from frozen V16 visible-only selection")
        _write_json(root / "fixtures" / "v16-visible-reference.json", {
            "protocol_id": protocol_id,
            "source_protocol_id": v16.PROTOCOL_ID,
            "source_example_set_sha256": visible_v16["example_set_sha256"],
            "source_count": visible_v16["count"],
            "source_selection_sha256": _sha_file(selection_path),
        }, immutable=True)
        for seed_index, (payload, split) in enumerate(zip(adaptations, splits, strict=True)):
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
            _write_json(root / "splits" / f"seed-{seed_index}.json", split, immutable=True)

        seed_results = []
        all_candidates = []
        for seed_index, (payload, split, seed) in enumerate(zip(adaptations, splits, seeds, strict=True)):
            _check_caps(root, started)
            records = payload["records"]
            captures = v16._capture_adaptation(model, tokenizer, records, TARGET_LAYER, ADJACENT_LAYER)
            v19._save_npz(root / "adaptation" / f"seed-{seed_index}.npz", **captures)
            fit_prompts, validation_prompts = set(split["fit_prompt_indices"]), set(split["validation_prompt_indices"])
            train_seed = int(seed) ^ (0x20B10C if stage == "development" else 0x20C10C)
            target_fit = _training_set(layers[TARGET_LAYER], captures, records, fit_prompts, seed=train_seed, label="target")
            target_val = _training_set(layers[TARGET_LAYER], captures, records, validation_prompts, seed=train_seed ^ 0x55A, label="target")
            target_residual, target_base, target_random, target_receipt = _train_pair(
                layers[TARGET_LAYER], target_fit, target_val, seed=train_seed, root=root, started=started,
            )
            candidate_modules = {
                "causal": (target_residual, target_receipt),
                "pointwise": (target_base, target_receipt["pointwise_base"]),
                "random": (target_random, {"trained": False, "initialization_sha256": v16._module_state_digest(target_random)}),
            }
            if stage == "confirmation":
                adjacent_fit = _training_set(layers[ADJACENT_LAYER], captures, records, fit_prompts, seed=train_seed ^ 0x9A1, label="adjacent")
                adjacent_val = _training_set(layers[ADJACENT_LAYER], captures, records, validation_prompts, seed=train_seed ^ 0x9A1 ^ 0x55A, label="adjacent")
                wrong_residual, wrong_base, wrong_random, wrong_receipt = _train_pair(
                    layers[ADJACENT_LAYER], adjacent_fit, adjacent_val, seed=train_seed ^ 0x9A1, root=root, started=started,
                )
                candidate_modules["wrong_layer"] = (wrong_residual, wrong_receipt)

            method_cells: dict[str, dict[str, Any]] = {}
            pointwise_cells = None
            for method, (candidate, fit_receipt) in candidate_modules.items():
                saved = v17._save_candidate(root / "candidates" / f"seed-{seed_index}-{method}.safetensors", candidate)
                cells = v19._candidate_cell_metrics(model, tokenizer, candidate, target_val, records, split["validation_prompt_indices"])
                all_candidates.append({
                    "seed_index": seed_index, "method": method, **saved,
                    "fit_receipt": fit_receipt, "validation_cells": cells,
                })
                method_cells[method] = cells
                if method == "pointwise":
                    pointwise_cells = cells
            if pointwise_cells is None:
                raise ValueError("V20 pointwise control was not trained")
            for method, cells in method_cells.items():
                for stratum in fixtures.STRATA:
                    cells[stratum]["value_mse_ratio_pointwise"] = (
                        cells[stratum]["validation_value_mse"] / max(pointwise_cells[stratum]["validation_value_mse"], 1e-30)
                    )
            no_op = v16.NoOpCandidate().eval()
            method_cells["no_op"] = v19._candidate_cell_metrics(
                model, tokenizer, no_op, target_val, records, split["validation_prompt_indices"],
            )
            for stratum in fixtures.STRATA:
                method_cells["no_op"][stratum]["value_mse_ratio_pointwise"] = (
                    method_cells["no_op"][stratum]["validation_value_mse"]
                    / max(pointwise_cells[stratum]["validation_value_mse"], 1e-30)
                )
            seed_results.append({"seed_index": seed_index, "candidates": method_cells})
            del captures, target_fit, target_val, target_residual, target_base, target_random, no_op
            if stage == "confirmation":
                del adjacent_fit, adjacent_val, wrong_residual, wrong_base, wrong_random

        selection = _selection(seed_results)
        artifacts = [
            "fixtures/v16-visible-reference.json",
            *(f"fixtures/adaptation-seed-{index}.json" for index in range(3)),
            *(f"splits/seed-{index}.json" for index in range(3)),
            *(f"adaptation/seed-{index}.npz" for index in range(3)),
        ]
        candidate_manifest = {
            "protocol_id": protocol_id,
            "packet_sha256": packet["packet_sha256"],
            "target_tree_sha256": v16.TARGET_TREE_SHA256,
            "target_layer": TARGET_LAYER,
            "adjacent_control_layer": ADJACENT_LAYER,
            "preparation_artifacts": _artifact_rows(root, artifacts),
            "candidates": all_candidates,
            "training": {
                "architecture": "Linear(768,3072)-ReLU-Linear(3072,768)",
                "updates_base_and_residual": TRAIN_UPDATES,
                "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "gradient_clip_norm": GRADIENT_CLIP,
                "low_rank": RANK,
                "residual_objective": "normalized finite-effect MSE + 1.0 * normalized squared drift from frozen pointwise base output",
                "pointwise_objective": "normalized pointwise MSE only",
                "confirmation_controls": ["pointwise", "wrong_layer", "random", "no_op", "zero_effect"],
            },
            "runtime_versions": _runtime_versions(runtime),
        }
        candidate_manifest["candidate_set_sha256"] = _digest(candidate_manifest)
        candidate_manifest["candidate_freeze_sha256"] = _digest(candidate_manifest)
        _write_json(root / "candidate-freeze.json", candidate_manifest, immutable=True)
        _write_json(root / "runtime.json", runtime, immutable=True)
        resources = {
            "wall_seconds": float(time.monotonic() - started),
            "peak_rss_bytes": int(v16._max_rss_bytes()),
            "wall_cap_seconds": MAX_WALL_SECONDS,
            "rss_cap_bytes": MAX_RSS_BYTES,
        }
        _write_json(root / "preparation-resources.json", resources, immutable=True)
        if resources["wall_seconds"] > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
            raise TimeoutError("V20 preparation exceeded a frozen resource cap")
        qualification_status = "QUALIFIED" if selection["qualified"] else "NO_QUALIFIED_DEVELOPMENT_CANDIDATE"
        if stage == "confirmation":
            qualification_status = "PRECHALLENGE_QUALIFIED" if selection["qualified"] else "CONFIRMATION_STOPPED_BEFORE_CHALLENGE"
        scientific = {
            "protocol_id": protocol_id,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidate_manifest["candidate_set_sha256"],
            "qualification_status": qualification_status,
            "selection": selection,
            "claim_ceiling": CLAIM_CEILING,
        }
        report = {
            **scientific,
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "blind_status": "NOT_BLIND_LOCAL_CUSTODY_RUN",
            "scientific_results_sha256": _digest(scientific),
            "custody": "prompts, continuations, captures, intervention targets, and candidates remain in external owner custody",
            "known_limitations": [
                "single reused OPT/ReLU checkpoint and preselected block 8",
                "synthetic prompts on one language-model family do not constitute architecture replication",
                "validation effects are held out by prompt and independently generated direction, but remain local same-owner data",
                "confirmation oracle restoration is a pipeline control, not learned activation rescue",
            ],
            "next_blocked_authority_gate": (
                "confirmation packet and fresh confirmation data" if stage == "development" and selection["qualified"]
                else "new method packet required; do not create confirmation fixtures" if stage == "development"
                else "held-out challenge unlock requires exact prechallenge replay and all nine cells to pass"
            ),
        }
        report["report_sha256"] = _digest(report)
        _write_json(root / "preparation-selection.json", report, immutable=True)
        _write_json(root / "preparation-report.json", {"seed_results": seed_results}, immutable=True)
        return {
            "protocol_id": protocol_id,
            "qualification_status": qualification_status,
            "candidate_set_sha256": candidate_manifest["candidate_set_sha256"],
            "scientific_results_sha256": report["scientific_results_sha256"],
            "packet_sha256": packet["packet_sha256"],
            "selection": selection,
        }
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": protocol_id,
            "phase": f"{stage}-preparation",
            "status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error": str(error),
            "packet_sha256": packet["packet_sha256"],
            "timestamp_epoch": time.time(),
        }, immutable=True)
        raise


def _relative_files(root: Path, prefix: str) -> list[str]:
    path = root / prefix
    return [item.relative_to(root).as_posix() for item in sorted(path.rglob("*")) if item.is_file()] if path.exists() else []


def _file_rows(root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V20 replay artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _compare_preparation(stage: str, primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path, stage)
    replica = _ensure_root(replica_path, stage, replica=True)
    left, right = _read_json(primary / "preparation-selection.json"), _read_json(replica / "preparation-selection.json")
    path_groups = {
        "fixtures": _relative_files(primary, "fixtures"),
        "splits": _relative_files(primary, "splits"),
        "captures": _relative_files(primary, "adaptation"),
        "candidates": _relative_files(primary, "candidates") + ["candidate-freeze.json"],
        "selection": ["preparation-selection.json", "preparation-report.json"],
    }
    groups = {}
    for name, paths in path_groups.items():
        left_rows, right_rows = _file_rows(primary, paths), _file_rows(replica, paths)
        if name == "fixtures":
            replica_paths = _relative_files(replica, "fixtures")
        elif name == "splits":
            replica_paths = _relative_files(replica, "splits")
        elif name == "captures":
            replica_paths = _relative_files(replica, "adaptation")
        elif name == "candidates":
            replica_paths = _relative_files(replica, "candidates") + ["candidate-freeze.json"]
        else:
            replica_paths = paths
        groups[name] = {
            "primary_count": len(left_rows), "replica_count": len(right_rows),
            "primary_sha256": _digest(left_rows), "replica_sha256": _digest(right_rows),
            "byte_identical": left_rows == right_rows and paths == replica_paths,
        }
    left_runtime, right_runtime = _read_json(primary / "runtime.json"), _read_json(replica / "runtime.json")
    pin_fields = (
        "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256",
        "deterministic_algorithms", "torch_num_threads",
    )
    pins_match = all(left_runtime.get(field) == right_runtime.get(field) for field in pin_fields)
    separate = left_runtime.get("python_executable") != right_runtime.get("python_executable")
    same_host = left_runtime.get("host_id_sha256") == right_runtime.get("host_id_sha256")
    same_owner = left_runtime.get("uid") == right_runtime.get("uid")
    scientific_match = left.get("scientific_results_sha256") == right.get("scientific_results_sha256")
    p_resources, r_resources = _read_json(primary / "preparation-resources.json"), _read_json(replica / "preparation-resources.json")
    caps_pass = all(
        item.get("wall_seconds", float("inf")) <= MAX_WALL_SECONDS
        and item.get("peak_rss_bytes", MAX_RSS_BYTES + 1) <= MAX_RSS_BYTES
        for item in (p_resources, r_resources)
    )
    reproduced = all(row["byte_identical"] for row in groups.values()) and pins_match and separate and same_host and same_owner and scientific_match and caps_pass
    report = {
        "protocol_id": DEV_PROTOCOL_ID if stage == "development" else CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": _packet(stage)[3]["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "qualification_status": left.get("qualification_status") if scientific_match else "INCONCLUSIVE",
        "pre_challenge_eligible": bool(left["selection"].get("qualified")) if stage == "confirmation" and scientific_match else None,
        "scientific_results_match": scientific_match,
        "scientific_results_sha256": left.get("scientific_results_sha256") if scientific_match else None,
        "artifact_groups": groups,
        "runtime_pins_match": pins_match,
        "separate_python_runtimes": separate,
        "same_host_and_owner": same_host and same_owner,
        "resource_caps_pass": caps_pass,
        "resource_measurements": {"primary": p_resources, "replica": r_resources},
        "claim_ceiling": CLAIM_CEILING,
        "custody": "prompt-bearing, row-level, capture, target, and candidate artifacts remain in external owner custody",
        "next_blocked_authority_gate": (
            "fresh confirmation data" if stage == "development" and left.get("qualification_status") == "QUALIFIED"
            else "held-out challenge remains locked unless all nine confirmation cells pass" if stage == "confirmation"
            else "a new method packet is required; no confirmation data will be created"
        ),
    }
    report["report_sha256"] = _digest(report)
    return report


def _write_development_results(replay: dict[str, Any], primary: Path) -> dict[str, Any]:
    outcome = _read_json(primary / "preparation-selection.json")
    selection = outcome["selection"]
    payload = {
        "protocol_id": DEV_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_level": replay["assurance_level"],
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": replay["assurance_level"],
        "blind_status": replay["blind_status"],
        "packet_sha256": outcome["packet_sha256"],
        "candidate_set_sha256": outcome["candidate_set_sha256"],
        "scientific_results_sha256": outcome["scientific_results_sha256"],
        "qualification_status": outcome["qualification_status"],
        "selection": selection,
        "resource_measurements": replay["resource_measurements"],
        "replication_report_sha256": replay["report_sha256"],
        "repository_revision": _read_json(primary / "runtime.json").get("repository_head"),
        "claim_ceiling": CLAIM_CEILING,
        "custody": "prompts, continuations, intervention targets, captures, candidates, and row-level metrics remain on external owner custody",
        "limitations": outcome["known_limitations"],
        "next_blocked_authority_gate": outcome["next_blocked_authority_gate"],
        "behavior_and_mechanism_diverged": any(
            cell["validation_cross_entropy_increase"] > 0.05 or cell["median_normalized_next_logit_error"] > 0.05
            for cell in selection["cells"]
        ),
    }
    payload["result_document_sha256"] = _digest(payload)
    _write_json(REPO_ROOT / DEV_RESULT_JSON, payload)
    _write_json(REPO_ROOT / DEV_REPLICATION_JSON, replay)
    table = "\n".join(
        f"| {cell['seed_index']} | {cell['stratum']} | {cell['effect_mse_ratio_zero']:.4f} | "
        f"{cell['value_mse_ratio_pointwise']:.4f} | {cell['validation_cross_entropy_increase']:.4f} | "
        f"{cell['median_normalized_next_logit_error']:.4f} | "
        f"{all((cell['effect_mse_ratio_zero'] <= 0.90, cell['value_mse_ratio_pointwise'] <= 1.25, cell['validation_cross_entropy_increase'] <= 0.05, cell['median_normalized_next_logit_error'] <= 0.05))} |"
        for cell in selection["cells"]
    )
    text = (
        "# Replacement-loss V20 development results\n\n"
        "State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v20-low-rank-causal-residual-v1`, and `global-experiment-assurance-lifecycle-v1`.\n\n"
        f"- Outcome: `{payload['qualification_status']}`.\n"
        f"- Assurance: `{payload['assurance_level']}`; reviewer: `NOT_SUBMITTED`; blind status: `{payload['blind_status']}`.\n"
        f"- Packet SHA-256: `{payload['packet_sha256']}`; result SHA-256: `{payload['scientific_results_sha256']}`.\n"
        f"- Claim ceiling: `{CLAIM_CEILING}`.\n"
        "- Custody: all prompt-bearing and row-level artifacts remain in external owner custody.\n\n"
        "## Nine predeclared development cells\n\n"
        "Each cell must pass the effect ratio <=0.90, pointwise value ratio <=1.25, continuation CE increase <=0.05, and normalized next-logit error <=0.05 gates.\n\n"
        "| Seed index | Stratum | Effect ratio | Value ratio | CE increase | Next-logit error | Cell passes |\n"
        "|---:|---|---:|---:|---:|---:|:---:|\n"
        f"{table}\n\n"
        f"Equal-cell mean effect ratio: `{selection['equal_cell_mean_effect_mse_ratio']:.6f}`; worst cell: `{selection['worst_cell_effect_mse_ratio']:.6f}`.\n\n"
        f"Next gate: {payload['next_blocked_authority_gate']}.\n"
    )
    (REPO_ROOT / DEV_RESULT_MD).write_text(text, encoding="utf-8")
    return payload


def _python_exec(executable: Path, arguments: list[str], *, log_path: Path) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(f"pinned V20 runtime executable is missing: {executable}")
    completed = subprocess.run(
        [str(executable), str(Path(__file__).resolve()), *arguments],
        check=False, capture_output=True, text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"V20 pinned runtime exited {completed.returncode}; see external-custody log {log_path.name}")
    decoder = json.JSONDecoder()
    for index, character in enumerate(completed.stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[index:])
        except json.JSONDecodeError:
            continue
        if not completed.stdout[index + end :].strip() and isinstance(value, dict):
            return value
    raise ValueError("V20 pinned runtime did not return one JSON receipt")


def run_development(primary_custody: str | Path, replica_custody: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    verify_development_packet()
    primary = _custody_root(primary_custody, "development")
    replica = _custody_root(replica_custody, "development", replica=True)
    if primary.exists() or replica.exists():
        raise FileExistsError("V20 development custody roots already exist; preserve them and choose a fresh identity")
    _python_exec(PRIMARY_PYTHON, ["run-preparation", "--stage", "development", "--custody", str(primary)], log_path=primary / "primary-runtime.log")
    _python_exec(REPLICA_PYTHON, ["run-preparation", "--stage", "development", "--custody", str(replica)], log_path=primary / "replica-runtime.log")
    replay = _compare_preparation("development", primary, replica)
    _write_json(primary / "preparation-replication.json", replay, immutable=True)
    result = _write_development_results(replay, primary)
    return result, replay


def _patch_v17_for_confirmation() -> None:
    _load_dependencies()
    fixtures.activate_stage("confirmation")
    v17.PROTOCOL_ID = CONFIRM_PROTOCOL_ID
    v17.STATE_SLICE = STATE_SLICE
    v17.PROTOCOL_SLICE = PROTOCOL_SLICE
    v17.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    v17.CLAIM_CEILING = CLAIM_CEILING
    v17.FREEZE_FILE = CONFIRM_FREEZE_FILE
    v17.PRIMARY_NAME = CONFIRM_PRIMARY_NAME
    v17.REPLICA_NAME = CONFIRM_REPLICA_NAME
    v17.RESEARCH_ROOT = RESEARCH_ROOT
    v17.REPO_ROOT = REPO_ROOT
    v17.fixtures = fixtures
    v17.verify_packet = verify_confirmation_packet
    v17._custody_root = lambda path: _custody_root(path, "confirmation", replica=Path(path).name == CONFIRM_REPLICA_NAME)
    v17._external_packet_archive = lambda root: _verify_archive(Path(root), "confirmation")
    v17._ensure_root = lambda path: _ensure_root(path, "confirmation", replica=Path(path).name == CONFIRM_REPLICA_NAME)
    v17._runtime_receipt = _runtime_receipt


def _set_confirmation_disjointness() -> None:
    _load_dependencies()
    _visible, old_adaptations = v19._reference_payloads()
    dev_root = RESEARCH_ROOT / DEV_PRIMARY_NAME
    confirm_root = RESEARCH_ROOT / CONFIRM_PRIMARY_NAME
    dev_payloads = [_read_json(dev_root / "fixtures" / f"adaptation-seed-{i}.json") for i in range(3)]
    confirm_payloads = [_read_json(confirm_root / "fixtures" / f"adaptation-seed-{i}.json") for i in range(3)]
    fixtures.set_extra_disjoint_payloads([*old_adaptations, *dev_payloads])
    fixtures.validate_partition_disjointness(*confirm_payloads)


def _run_confirmation_challenge(custody: str | Path, *, replica: bool = False, entropy_from: str | Path | None = None) -> dict[str, Any]:
    _patch_v17_for_confirmation()
    root = _ensure_root(custody, "confirmation", replica=replica)
    selection = _read_json(root / "preparation-selection.json")
    if not selection.get("selection", {}).get("qualified"):
        raise ValueError("V20 challenge entropy remains locked because prechallenge gates did not pass")
    primary = RESEARCH_ROOT / CONFIRM_PRIMARY_NAME
    replay = _read_json(primary / "preparation-replication.json")
    primary_selection = _read_json(primary / "preparation-selection.json")
    replica_selection = _read_json(RESEARCH_ROOT / CONFIRM_REPLICA_NAME / "preparation-selection.json")
    if (
        replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or not primary_selection.get("selection", {}).get("qualified")
        or not replica_selection.get("selection", {}).get("qualified")
        or not replay.get("scientific_results_match")
    ):
        raise ValueError("V20 challenge entropy requires a qualified exact primary/replica prechallenge replay")
    _set_confirmation_disjointness()
    if replica:
        if entropy_from is None:
            raise ValueError("V20 replica challenge must reuse primary challenge entropy")
        prediction = v17.predict(root, entropy_from=entropy_from)
    else:
        prediction = v17.predict(root)
    score = v17.score(root)
    return {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "prediction_set_sha256": prediction["prediction_set_sha256"],
        "scientific_outcome": score["scientific_outcome"],
        "scientific_results_sha256": score["scientific_results_sha256"],
        "assurance_level": score["assurance_level"],
    }


def _compare_confirmation(primary: Path, replica: Path) -> dict[str, Any]:
    _patch_v17_for_confirmation()
    replay = v17.compare_replication(primary, replica)
    prep_replay = _read_json(primary / "preparation-replication.json")
    replay["preparation_replay_report_sha256"] = prep_replay["report_sha256"]
    replay["preparation_reproduced_local"] = prep_replay["assurance_level"] == "REPRODUCED_LOCAL"
    replay["claim_ceiling"] = CLAIM_CEILING
    replay["next_blocked_authority_gate"] = "blind external replication and an authorized independent reviewer"
    replay["assurance_level"] = "REPRODUCED_LOCAL" if replay["assurance_level"] == "REPRODUCED_LOCAL" and replay["preparation_reproduced_local"] else "INCONCLUSIVE"
    replay.pop("report_sha256", None)
    replay["report_sha256"] = _digest(replay)
    _write_json(primary / "confirmation-replication.json", replay, immutable=True)
    return replay


def _confirmation_summary_cells(outcome: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for seed in outcome.get("results", []):
        for stratum, values in seed.get("strata", {}).items():
            rows.append({
                "seed_index": seed["seed_index"],
                "stratum": stratum,
                "causal_module_effect": values["causal"]["module_effect"],
                "causal_next_logit_effect": values["causal"]["next_logit_effect"],
                "pointwise_next_logit_effect": values["pointwise"]["next_logit_effect"],
                "wrong_layer_next_logit_effect": values["wrong_layer"]["next_logit_effect"],
                "random_next_logit_effect": values["random"]["next_logit_effect"],
                "no_op_next_logit_effect": values["no_op"]["next_logit_effect"],
                "activation": values["activation"],
                "behavior": values["behavior"],
                "causal_vs_pointwise": values["causal_vs_pointwise"],
            })
    return rows


def _write_confirmation_results(preparation_replay: dict[str, Any], *, challenge_replay: dict[str, Any] | None = None) -> dict[str, Any]:
    primary = RESEARCH_ROOT / CONFIRM_PRIMARY_NAME
    packet = verify_confirmation_packet()
    prep = _read_json(primary / "preparation-selection.json")
    challenge_outcome = _read_json(primary / "outcome-report.json") if challenge_replay is not None else None
    if challenge_replay is None:
        assurance = preparation_replay.get("assurance_level", "INCONCLUSIVE")
        scientific_outcome = "NOT_RUN_PRECHALLENGE_STOP"
        cells = prep["selection"]["cells"]
        scientific_sha = prep["scientific_results_sha256"]
        entropy_generated = False
        diverged = any(
            cell["validation_cross_entropy_increase"] > 0.05 or cell["median_normalized_next_logit_error"] > 0.05
            for cell in cells
        )
        next_gate = "revise the method under a new packet; confirmation challenge entropy was not generated"
    else:
        assurance = challenge_replay.get("assurance_level", "INCONCLUSIVE")
        scientific_outcome = challenge_outcome.get("scientific_outcome", "INCONCLUSIVE") if challenge_outcome else "INCONCLUSIVE"
        cells = _confirmation_summary_cells(challenge_outcome or {})
        scientific_sha = challenge_outcome.get("scientific_results_sha256") if challenge_outcome else None
        entropy_generated = True
        diverged = challenge_outcome.get("behavior_and_mechanism_diverged") if challenge_outcome else None
        next_gate = "blind external replication and an authorized independent reviewer remain required"
    result = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_level": assurance,
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": assurance,
        "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "packet_sha256": packet["packet_sha256"],
        "development_packet_sha256": _read_json(REPO_ROOT / DEV_FREEZE_FILE)["packet_sha256"],
        "development_result_sha256": _read_json(REPO_ROOT / DEV_RESULT_JSON)["scientific_results_sha256"],
        "preparation_replication_sha256": preparation_replay.get("report_sha256"),
        "challenge_replication_sha256": challenge_replay.get("report_sha256") if challenge_replay else None,
        "scientific_results_sha256": scientific_sha,
        "preparation_qualification_status": prep.get("qualification_status"),
        "pre_challenge_eligible": bool(prep.get("selection", {}).get("qualified")),
        "challenge_entropy_generated": entropy_generated,
        "scientific_outcome": scientific_outcome,
        "confirmation_cells": cells,
        "resource_measurements": preparation_replay.get("resource_measurements"),
        "claim_ceiling": CLAIM_CEILING,
        "custody": "prompts, continuations, captures, intervention targets, candidates, entropy, predictions, and detailed outputs remain in external owner custody",
        "limitations": [
            "one previously used OPT/ReLU checkpoint and selected block 8",
            "three synthetic seeds on one checkpoint do not establish architecture or fresh-checkpoint generality",
            "same-owner separate-runtime replay is not blind external replication or independent acceptance",
            "oracle restoration is a pipeline control, not evidence of learned activation rescue",
            "behavioral and mechanistic continuity does not establish subjective experience or alignment",
        ],
        "behavior_and_mechanism_diverged": diverged,
        "next_blocked_authority_gate": next_gate,
    }
    result["result_document_sha256"] = _digest(result)
    _write_json(REPO_ROOT / CONFIRM_RESULT_JSON, result)
    if challenge_replay is None:
        table = "\n".join(
            f"| {row['seed_index']} | {row['stratum']} | {row['effect_mse_ratio_zero']:.4f} | "
            f"{row['value_mse_ratio_pointwise']:.4f} | {row['validation_cross_entropy_increase']:.4f} | "
            f"{row['median_normalized_next_logit_error']:.4f} | "
            f"{all((row['effect_mse_ratio_zero'] <= 0.90, row['value_mse_ratio_pointwise'] <= 1.25, row['validation_cross_entropy_increase'] <= 0.05, row['median_normalized_next_logit_error'] <= 0.05))} |"
            for row in cells
        )
        table_header = (
            "| Seed index | Stratum | Effect ratio | Value ratio | CE increase | Next-logit error | Cell passes |\n"
            "|---:|---|---:|---:|---:|---:|:---:|\n"
        )
    else:
        table = "\n".join(
            f"| {row['seed_index']} | {row['stratum']} | {row['causal_next_logit_effect']['median_cosine']} | "
            f"{row['causal_next_logit_effect']['median_normalized_error']} | "
            f"{row['activation']['median_cosine']:.4f} | {row['activation']['median_relative_error']:.4f} | "
            f"{row['behavior']['mean_cross_entropy_increase']:.4f} | {row['behavior']['median_normalized_next_logit_error']:.4f} |"
            for row in cells
        )
        table_header = (
            "| Seed index | Stratum | Causal logit cosine | Causal logit error | Module cosine | Module relative error | CE increase | Next-logit error |\n"
            "|---:|---|---:|---:|---:|---:|---:|---:|\n"
        )
    text = (
        "# Replacement-loss V20 confirmation results\n\n"
        "State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v20-low-rank-causal-residual-v1`, and `global-experiment-assurance-lifecycle-v1`.\n\n"
        f"- Scientific outcome: `{scientific_outcome}`.\n"
        f"- Assurance: `{assurance}`; reviewer: `NOT_SUBMITTED`; blind status: `NOT_BLIND_SAME_OWNER_REPLAY`.\n"
        f"- Confirmation packet SHA-256: `{packet['packet_sha256']}`.\n"
        f"- Challenge entropy generated: `{entropy_generated}`.\n"
        f"- Claim ceiling: `{CLAIM_CEILING}`.\n"
        "- Data custody: prompt-bearing and row-level artifacts remain on external owner custody.\n\n"
        "## Confirmation cells\n\n"
        f"{table_header}{table}\n\n"
        f"Next gate: {next_gate}.\n"
    )
    (REPO_ROOT / CONFIRM_RESULT_MD).write_text(text, encoding="utf-8")
    _write_json(REPO_ROOT / CONFIRM_REPLICATION_JSON, challenge_replay or preparation_replay)
    return result


def _run_e2e(args: argparse.Namespace) -> dict[str, Any]:
    development_packet = freeze_development()
    primary = _custody_root(args.development_primary, "development")
    replica = _custody_root(args.development_replica, "development", replica=True)
    dev_result, dev_replay = run_development(primary, replica)
    if dev_replay["assurance_level"] != "REPRODUCED_LOCAL":
        return {"end_to_end_status": "STOPPED_DEVELOPMENT_REPLAY_MISMATCH", "development": dev_result}
    if dev_result["qualification_status"] != "QUALIFIED":
        return {
            "end_to_end_status": "STOPPED_NO_QUALIFIED_DEVELOPMENT_CANDIDATE",
            "development": dev_result,
            "development_packet_sha256": development_packet["packet_sha256"],
            "confirmation_packet_created": False,
            "confirmation_fixtures_created": False,
            "challenge_entropy_generated": False,
        }

    confirmation_packet = freeze_confirmation()
    confirm_primary = _custody_root(args.confirmation_primary, "confirmation")
    confirm_replica = _custody_root(args.confirmation_replica, "confirmation", replica=True)
    if confirm_primary.exists() or confirm_replica.exists():
        raise FileExistsError("V20 confirmation roots already exist; preserve them and choose a fresh packet identity")
    _python_exec(PRIMARY_PYTHON, ["run-preparation", "--stage", "confirmation", "--custody", str(confirm_primary)], log_path=confirm_primary / "primary-preparation-runtime.log")
    _python_exec(REPLICA_PYTHON, ["run-preparation", "--stage", "confirmation", "--custody", str(confirm_replica)], log_path=confirm_primary / "replica-preparation-runtime.log")
    prep_replay = _compare_preparation("confirmation", confirm_primary, confirm_replica)
    _write_json(confirm_primary / "preparation-replication.json", prep_replay, immutable=True)
    primary_eligible = bool(_read_json(confirm_primary / "preparation-selection.json")["selection"]["qualified"])
    replica_eligible = bool(_read_json(confirm_replica / "preparation-selection.json")["selection"]["qualified"])
    if prep_replay["assurance_level"] != "REPRODUCED_LOCAL" or not primary_eligible or not replica_eligible:
        confirmation_result = _write_confirmation_results(prep_replay)
        return {
            "end_to_end_status": "STOPPED_BEFORE_CHALLENGE",
            "development": dev_result,
            "confirmation": confirmation_result,
            "confirmation_packet_sha256": confirmation_packet["packet_sha256"],
            "confirmation_preparation_replay": prep_replay["assurance_level"],
            "challenge_entropy_generated": False,
        }

    _python_exec(PRIMARY_PYTHON, ["run-confirmation-challenge", "--custody", str(confirm_primary)], log_path=confirm_primary / "primary-challenge-runtime.log")
    _python_exec(REPLICA_PYTHON, [
        "run-confirmation-replica-challenge", "--custody", str(confirm_replica), "--entropy-from", str(confirm_primary),
    ], log_path=confirm_primary / "replica-challenge-runtime.log")
    compared = _python_exec(PRIMARY_PYTHON, [
        "compare-confirmation", "--primary", str(confirm_primary), "--replica", str(confirm_replica),
    ], log_path=confirm_primary / "confirmation-comparison-runtime.log")
    confirmation_result = _write_confirmation_results(prep_replay, challenge_replay=compared)
    return {
        "end_to_end_status": "CONFIRMATION_COMPLETED",
        "development": dev_result,
        "confirmation": confirmation_result,
        "confirmation_packet_sha256": confirmation_packet["packet_sha256"],
        "confirmation_replay_assurance": compared["assurance_level"],
        "confirmation_scientific_outcome": compared["scientific_outcome"],
        "confirmation_replay_sha256": compared["report_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze-development")
    commands.add_parser("verify-development")
    commands.add_parser("freeze-confirmation")
    commands.add_parser("verify-confirmation")
    run = commands.add_parser("run-e2e")
    run.add_argument("--development-primary", default=str(RESEARCH_ROOT / DEV_PRIMARY_NAME))
    run.add_argument("--development-replica", default=str(RESEARCH_ROOT / DEV_REPLICA_NAME))
    run.add_argument("--confirmation-primary", default=str(RESEARCH_ROOT / CONFIRM_PRIMARY_NAME))
    run.add_argument("--confirmation-replica", default=str(RESEARCH_ROOT / CONFIRM_REPLICA_NAME))
    preparation = commands.add_parser("run-preparation")
    preparation.add_argument("--stage", choices=("development", "confirmation"), required=True)
    preparation.add_argument("--custody", required=True)
    challenge = commands.add_parser("run-confirmation-challenge")
    challenge.add_argument("--custody", required=True)
    replica_challenge = commands.add_parser("run-confirmation-replica-challenge")
    replica_challenge.add_argument("--custody", required=True)
    replica_challenge.add_argument("--entropy-from", required=True)
    compare = commands.add_parser("compare-confirmation")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-development":
        result = freeze_development()
    elif args.command == "verify-development":
        result = verify_development_packet()
    elif args.command == "freeze-confirmation":
        result = freeze_confirmation()
    elif args.command == "verify-confirmation":
        result = verify_confirmation_packet()
    elif args.command == "run-preparation":
        result = _run_preparation(args.stage, args.custody, replica=Path(args.custody).name in {DEV_REPLICA_NAME, CONFIRM_REPLICA_NAME})
    elif args.command == "run-confirmation-challenge":
        result = _run_confirmation_challenge(args.custody)
    elif args.command == "run-confirmation-replica-challenge":
        result = _run_confirmation_challenge(args.custody, replica=True, entropy_from=args.entropy_from)
    elif args.command == "compare-confirmation":
        result = _compare_confirmation(Path(args.primary), Path(args.replica))
    else:
        result = _run_e2e(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
