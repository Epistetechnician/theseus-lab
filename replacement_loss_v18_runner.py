"""Frozen local runner for projected causal refinement of an OPT MLP.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v18-projected-causal-refinement-v1, and
global-experiment-assurance-lifecycle-v1.

Prompt-bearing and row-level artifacts are restricted to external owner
custody under /Volumes/PrimaryED/ResearchArtifacts.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import secrets
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import replacement_loss_v18_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FREEZE_FILE = "docs/replacement-loss-v18-freeze.json"
CLAIM_CEILING = "LOCAL_PROJECTED_CAUSAL_REFINEMENT_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT"
V16_PROTOCOL_ID = "theseus-replacement-loss-v16-whole-mlp-independent-replacement"
V17_PROTOCOL_ID = "theseus-replacement-loss-v17-causal-supervised-whole-mlp"
V16_CLAIM_CEILING = "LOCAL_WHOLE_MLP_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ON_ONE_REUSED_OPT_CHECKPOINT"
V17_CLAIM_CEILING = "LOCAL_CAUSAL_SUPERVISION_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT"
TARGET_MODEL_ID = "grushaprasad/opt125m-rootp-surprisal-curriculum"
TARGET_REVISION = "8af2a2926bc772599ccb3b4e7ad7b40b35259a63"
TARGET_TREE_SHA256 = "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466"
TARGET_WEIGHT_SHA256 = "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132"
TARGET_LAYER = 8
ADJACENT_LAYER = 9
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v18-projected-causal-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v18-projected-causal-replica"
V16_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v16-whole-mlp-primary"
V17_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v17-causal-supervision-primary"
RESEARCH_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
REPO_ROOT = Path(__file__).resolve().parent
BASE_UPDATES = 1000
REFINEMENT_UPDATES = 1000
TRAIN_BATCH_SIZE = 128
BASE_LEARNING_RATE = 1e-3
BASE_WEIGHT_DECAY = 1e-4
REFINEMENT_LEARNING_RATE = 1e-4
GRADIENT_CLIP = 1.0
FIT_RETENTION_RATIO = 1.25
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
INFERENCE_BATCH_SIZE = 4
PACKET_FILES = (
    "AGENTS.md",
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

np: Any = None
torch: Any = None
F: Any = None
load_file: Any = None
save_file: Any = None
v16: Any = None
v17: Any = None
_DEPENDENCIES_LOADED = False


def _load_dependencies() -> None:
    global np, torch, F, load_file, save_file, v16, v17, _DEPENDENCIES_LOADED
    if _DEPENDENCIES_LOADED:
        return
    import numpy as numpy_module
    import torch as torch_module
    import torch.nn.functional as functional_module
    from safetensors.torch import load_file as safe_load_file, save_file as safe_save_file

    import replacement_loss_v16_runner as v16_module
    import replacement_loss_v17_runner as v17_module

    np = numpy_module
    torch = torch_module
    F = functional_module
    load_file = safe_load_file
    save_file = safe_save_file
    v16 = v16_module
    v17 = v17_module
    _DEPENDENCIES_LOADED = True


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
            raise FileExistsError(f"immutable artifact already exists with different bytes: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _packet_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in PACKET_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V18 frozen packet source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def _verify_local_freeze(freeze_relative: str) -> dict[str, Any]:
    frozen = _read_json(REPO_ROOT / freeze_relative)
    current = []
    for row in frozen.get("packet_files", []):
        path = REPO_ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"parent frozen source is missing: {row['path']}")
        current.append({"path": row["path"], "sha256": _sha_file(path), "bytes": path.stat().st_size})
    if frozen.get("packet_files") != current or frozen.get("packet_sha256") != _digest(current):
        raise ValueError(f"parent packet source bytes differ from its freeze: {freeze_relative}")
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _digest(material):
        raise ValueError(f"parent freeze digest is invalid: {freeze_relative}")
    return frozen


def _verify_external_parent_archive(root_name: str, freeze_relative: str, frozen: dict[str, Any]) -> dict[str, Any]:
    root = RESEARCH_ROOT / root_name
    frozen_root = root / "frozen-packet"
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != frozen.get("packet_sha256"):
        raise ValueError(f"external parent archive binds a different packet: {root_name}")
    for row in frozen["packet_files"]:
        path = frozen_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"external parent packet file differs: {root_name}/{row['path']}")
    freeze_copy = frozen_root / freeze_relative
    if not freeze_copy.is_file() or _sha_file(freeze_copy) != _sha_file(REPO_ROOT / freeze_relative):
        raise ValueError(f"external parent archive has a different freeze record: {root_name}")
    expected = list(frozen["packet_files"])
    expected.append({"path": freeze_relative, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected or manifest.get("archive_sha256") != _digest(expected):
        raise ValueError(f"external parent archive manifest is invalid: {root_name}")
    return manifest


def _validate_parent_replay(root_name: str, protocol_id: str, packet_sha256: str, results_relative: str, claim_ceiling: str) -> None:
    root = RESEARCH_ROOT / root_name
    outcome = _read_json(root / "outcome-report.json")
    replay = _read_json(root / "replication-report.json")
    outcome_material = {key: value for key, value in outcome.items() if key != "report_sha256"}
    replay_material = {key: value for key, value in replay.items() if key != "report_sha256"}
    results_doc = (REPO_ROOT / results_relative).read_text(encoding="utf-8")
    groups = replay.get("artifact_groups", {})
    replay_fields = ("runtime_pins_match", "separate_python_runtimes", "same_host_and_owner", "scientific_results_match")
    if (
        outcome.get("protocol_id") != protocol_id
        or outcome.get("packet_sha256") != packet_sha256
        or outcome.get("scientific_outcome") != "FAIL"
        or outcome.get("claim_ceiling") != claim_ceiling
        or outcome.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or outcome.get("reviewer_status") != "NOT_SUBMITTED"
        or _digest(outcome_material) != outcome.get("report_sha256")
        or replay.get("protocol_id") != protocol_id
        or replay.get("packet_sha256") != packet_sha256
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_outcome") != "FAIL"
        or replay.get("reviewer_status") != "NOT_SUBMITTED"
        or not all(replay.get(field) is True for field in replay_fields)
        or _digest(replay_material) != replay.get("report_sha256")
        or not groups
        or not all(group.get("byte_identical") for group in groups.values())
        or outcome.get("report_sha256") not in results_doc
        or replay.get("report_sha256") not in results_doc
        or outcome.get("scientific_results_sha256") not in results_doc
    ):
        raise ValueError(f"{protocol_id} does not have a valid digest-bound FAIL and REPRODUCED_LOCAL receipt")


def _parent_binding() -> dict[str, Any]:
    v16_parent = _verify_local_freeze("docs/replacement-loss-v16-freeze.json")
    parent = _verify_local_freeze("docs/replacement-loss-v17-freeze.json")
    expected_v16_target = {
        "model_id": TARGET_MODEL_ID,
        "revision": TARGET_REVISION,
        "tree_sha256": TARGET_TREE_SHA256,
        "weight_sha256": TARGET_WEIGHT_SHA256,
        "target_known_before_protocol": True,
        "fresh_checkpoint_claim": False,
    }
    v17_parent = parent.get("parent_protocol", {})
    if (
        v16_parent.get("protocol_id") != V16_PROTOCOL_ID
        or v16_parent.get("state_slice") != STATE_SLICE
        or v16_parent.get("claim_ceiling") != V16_CLAIM_CEILING
        or v16_parent.get("target_identity") != expected_v16_target
        or parent.get("protocol_id") != V17_PROTOCOL_ID
        or parent.get("state_slice") != STATE_SLICE
        or parent.get("claim_ceiling") != V17_CLAIM_CEILING
        or parent.get("parent_protocol", {}).get("protocol_id") != V16_PROTOCOL_ID
        or v17_parent.get("packet_sha256") != v16_parent.get("packet_sha256")
        or v17_parent.get("scientific_outcome") != "FAIL"
        or v17_parent.get("selected_layer") != TARGET_LAYER
        or v17_parent.get("post_result_follow_up") is not True
        or parent.get("target_identity") != {**expected_v16_target, "fresh_checkpoint_claim": False}
    ):
        raise ValueError("V18 parent V17 packet is not the declared failed V16 follow-up")
    _verify_external_parent_archive(V16_PRIMARY_NAME, "docs/replacement-loss-v16-freeze.json", v16_parent)
    _verify_external_parent_archive(V17_PRIMARY_NAME, "docs/replacement-loss-v17-freeze.json", parent)
    _validate_parent_replay(V16_PRIMARY_NAME, V16_PROTOCOL_ID, v16_parent["packet_sha256"], "docs/replacement-loss-v16-results.md", V16_CLAIM_CEILING)
    _validate_parent_replay(V17_PRIMARY_NAME, V17_PROTOCOL_ID, parent["packet_sha256"], "docs/replacement-loss-v17-results.md", V17_CLAIM_CEILING)
    return {
        "protocol_id": V17_PROTOCOL_ID,
        "packet_sha256": parent["packet_sha256"],
        "scientific_outcome": "FAIL",
        "reproducibility_status": "REPRODUCED_LOCAL",
        "selected_layer": TARGET_LAYER,
        "post_result_follow_up": True,
    }


def _verify_parent_training_contract() -> None:
    tree = ast.parse((REPO_ROOT / "replacement_loss_v17_runner.py").read_text(encoding="utf-8"))
    names = {"TRAIN_UPDATES", "TRAIN_BATCH_SIZE", "LEARNING_RATE", "WEIGHT_DECAY", "GRADIENT_CLIP"}
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            matching = [target.id for target in node.targets if isinstance(target, ast.Name) and target.id in names]
            if matching:
                value = ast.literal_eval(node.value)
                for name in matching:
                    found[name] = value
    expected = {
        "TRAIN_UPDATES": BASE_UPDATES,
        "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
        "LEARNING_RATE": BASE_LEARNING_RATE,
        "WEIGHT_DECAY": BASE_WEIGHT_DECAY,
        "GRADIENT_CLIP": GRADIENT_CLIP,
    }
    if found != expected:
        raise ValueError("V17 pointwise baseline source differs from the V18 optimizer contract")


def freeze_packet() -> dict[str, Any]:
    path = REPO_ROOT / FREEZE_FILE
    if path.exists():
        return verify_packet()
    parent = _parent_binding()
    rows = _packet_rows()
    packet = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "parent_protocol": parent,
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
    parent = _parent_binding()
    _verify_parent_training_contract()
    path = REPO_ROOT / FREEZE_FILE
    if not path.is_file():
        raise FileNotFoundError("V18 packet is not frozen")
    frozen = _read_json(path)
    current = _packet_rows()
    if frozen.get("packet_files") != current or frozen.get("packet_sha256") != _digest(current):
        raise ValueError("V18 packet source bytes differ from the frozen packet")
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _digest(material):
        raise ValueError("V18 freeze digest is invalid")
    expected_target = {
        "model_id": TARGET_MODEL_ID,
        "revision": TARGET_REVISION,
        "tree_sha256": TARGET_TREE_SHA256,
        "weight_sha256": TARGET_WEIGHT_SHA256,
        "target_known_before_protocol": True,
        "fresh_checkpoint_claim": False,
    }
    if (
        frozen.get("protocol_id") != PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("parent_protocol") != parent
        or frozen.get("target_identity") != expected_target
    ):
        raise ValueError("V18 frozen identity, parent, or claim ceiling differs")
    return frozen


def _custody_root(value: str | Path) -> Path:
    root = Path(value).expanduser().resolve()
    if root.parent != RESEARCH_ROOT or root.name not in {PRIMARY_NAME, REPLICA_NAME} or root.is_symlink():
        raise ValueError("custody root must be one of the two frozen V18 PrimaryED roots")
    return root


def _external_packet_archive(root: Path) -> dict[str, Any]:
    packet = verify_packet()
    frozen_root = root / "frozen-packet"
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("external V18 archive binds a different packet")
    for row in packet["packet_files"]:
        path = frozen_root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"external V18 frozen packet file differs: {row['path']}")
    freeze_copy = frozen_root / FREEZE_FILE
    if not freeze_copy.is_file() or _sha_file(freeze_copy) != _sha_file(REPO_ROOT / FREEZE_FILE):
        raise ValueError("external V18 archive has a different freeze record")
    expected = list(packet["packet_files"])
    expected.append({"path": FREEZE_FILE, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected or manifest.get("archive_sha256") != _digest(expected):
        raise ValueError("external V18 archive manifest is invalid")
    return manifest


def archive_packet(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _custody_root(custody_path)
    if root.exists():
        raise FileExistsError("V18 custody root exists; preserve it and select a fresh root")
    root.mkdir(parents=True, exist_ok=False)
    frozen_root = root / "frozen-packet"
    for row in packet["packet_files"]:
        source = REPO_ROOT / row["path"]
        destination = frozen_root / row["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    freeze_source = REPO_ROOT / FREEZE_FILE
    freeze_copy = frozen_root / FREEZE_FILE
    freeze_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(freeze_source, freeze_copy)
    rows = list(packet["packet_files"])
    rows.append({"path": FREEZE_FILE, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"],
        "files": rows,
        "archive_sha256": _digest(rows),
    }
    _write_json(root / "frozen-packet-manifest.json", manifest, immutable=True)
    return {"protocol_id": PROTOCOL_ID, "packet_sha256": packet["packet_sha256"], "archive_sha256": manifest["archive_sha256"], "file_count": len(rows)}


def _ensure_root(value: str | Path) -> Path:
    root = _custody_root(value)
    if not root.is_dir():
        raise FileNotFoundError("V18 external custody root is missing")
    if (root / "run-stop.json").exists():
        raise RuntimeError("this V18 custody root has a terminal stop receipt; preserve it and do not reuse it")
    _external_packet_archive(root)
    return root


def _parse_lock(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise ValueError(f"invalid exact dependency pin in {path.name}")
        result[name.casefold().replace("_", "-")] = version
    return result


def _runtime_receipt() -> dict[str, Any]:
    _load_dependencies()
    parent = v17._runtime_receipt()
    v17_lock = _parse_lock(REPO_ROOT / "requirements-v17.lock.txt")
    v18_lock = _parse_lock(REPO_ROOT / "requirements-v18.lock.txt")
    if v17_lock != v18_lock:
        raise ValueError("V18 package pins differ from the reproduced V16/V17 runtime")
    if parent.get("deterministic_algorithms") is not True or parent.get("torch_num_threads") != 1:
        raise ValueError("V18 requires deterministic Torch algorithms and exactly one Torch thread")
    parent["dependency_lock_sha256"] = _sha_file(REPO_ROOT / "requirements-v18.lock.txt")
    return parent


def _phase_ledger(root: Path) -> dict[str, Any]:
    path = root / "phase-ledger.json"
    if path.exists():
        ledger = _read_json(path)
        if ledger.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("V18 phase ledger belongs to another protocol")
        return ledger
    ledger = {"protocol_id": PROTOCOL_ID, "started_at_epoch": time.time(), "phases": []}
    _write_json(path, ledger)
    return ledger


def _record_phase(root: Path, phase: str, started: float, status: str, *, details: dict[str, Any] | None = None) -> None:
    _load_dependencies()
    ledger = _phase_ledger(root)
    rows = list(ledger.get("phases", []))
    rows.append({"phase": phase, "status": status, "started_at_epoch": time.time() - (time.monotonic() - started), "duration_seconds": time.monotonic() - started, "max_rss_bytes": v16._max_rss_bytes(), "details": details or {}})
    ledger["phases"] = rows
    _write_json(root / "phase-ledger.json", ledger)


def _check_caps(root: Path, phase_start: float | None = None) -> None:
    _load_dependencies()
    if phase_start is not None and time.monotonic() - phase_start > MAX_WALL_SECONDS:
        raise TimeoutError("V18 phase exceeded the frozen four-hour wall-clock cap")
    if v16._max_rss_bytes() > MAX_RSS_BYTES:
        raise MemoryError("V18 process exceeded the frozen 16-GiB RSS cap")


def _stop(root: Path, phase: str, error: Exception) -> None:
    _write_json(root / "run-stop.json", {
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "status": "INCONCLUSIVE",
        "error_type": type(error).__name__,
        "error": str(error),
        "packet_sha256": verify_packet()["packet_sha256"],
        "timestamp_epoch": time.time(),
    }, immutable=True)


def _tree_receipt() -> dict[str, Any]:
    _load_dependencies()
    receipt = v16._target_tree_receipt()
    if receipt["tree_sha256"] != TARGET_TREE_SHA256 or receipt["weight_sha256"] != TARGET_WEIGHT_SHA256:
        raise ValueError("V18 checkpoint differs from the predeclared V16 target")
    return receipt


def _load_reference_fixtures() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v16_freeze = _verify_local_freeze("docs/replacement-loss-v16-freeze.json")
    v17_freeze = _verify_local_freeze("docs/replacement-loss-v17-freeze.json")
    v16_root = RESEARCH_ROOT / V16_PRIMARY_NAME
    v17_root = RESEARCH_ROOT / V17_PRIMARY_NAME
    _verify_external_parent_archive(V16_PRIMARY_NAME, "docs/replacement-loss-v16-freeze.json", v16_freeze)
    _verify_external_parent_archive(V17_PRIMARY_NAME, "docs/replacement-loss-v17-freeze.json", v17_freeze)
    visible = _read_json(v16_root / "fixtures" / "visible.json")
    old_adaptations = []
    for root in (v16_root, v17_root):
        for seed_index in range(3):
            old_adaptations.append(_read_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json"))
    return visible, old_adaptations


def _split(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed ^ 0x18D15A)
    fit: list[int] = []
    validation: list[int] = []
    for stratum in fixtures.STRATA:
        indices = np.asarray([i for i, row in enumerate(records) if row["stratum"] == stratum], dtype=np.int32)
        if len(indices) != fixtures.ADAPTATION_PER_STRATUM:
            raise ValueError("V18 adaptation fixture counts differ from frozen strata")
        rng.shuffle(indices)
        fit.extend(int(value) for value in indices[:24])
        validation.extend(int(value) for value in indices[24:])
    return {"fit_prompt_indices": fit, "validation_prompt_indices": validation}


def _artifact_rows(root: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for relative in relatives:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V18 preparation artifact is missing: {relative}")
        result.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return result


def _save_candidate(path: Path, candidate: torch.nn.Module) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in candidate.state_dict().items()}, str(path))
    return {
        "path": path.relative_to(path.parents[1]).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": v16._module_state_digest(candidate),
        "parameter_count": sum(parameter.numel() for parameter in candidate.parameters()),
        "architecture": "Linear-ReLU-Linear",
    }


def _projected_refinement(
    start_candidate: torch.nn.Module,
    data: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    seed: int,
    root: Path,
    phase_start: float,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    candidate = copy.deepcopy(start_candidate).train()
    parameters = tuple(candidate.parameters())
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    delta = torch.from_numpy(data["delta"])
    effect = torch.from_numpy(data["effect"])
    xv = torch.from_numpy(validation["x"])
    yv = torch.from_numpy(validation["y"])
    deltav = torch.from_numpy(validation["delta"])
    effectv = torch.from_numpy(validation["effect"])
    value_normalizer = max(float(y.square().mean().item()), 1e-12)
    effect_normalizer = max(float(effect.square().mean().item()), 1e-12)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x18A5C3)
    draws = np.empty((REFINEMENT_UPDATES, TRAIN_BATCH_SIZE), dtype=np.int32)
    conflict_count = 0
    pre_dots: list[float] = []
    post_dots: list[float] = []
    projection_coefficients: list[float] = []

    for update in range(REFINEMENT_UPDATES):
        indices = torch.randint(0, x.shape[0], (TRAIN_BATCH_SIZE,), generator=generator)
        draws[update] = indices.numpy()
        xb, yb = x.index_select(0, indices), y.index_select(0, indices)
        db, eb = delta.index_select(0, indices), effect.index_select(0, indices)
        predicted = candidate(xb)
        value_loss = F.mse_loss(predicted, yb) / value_normalizer
        predicted_effect = candidate(xb + db) - predicted
        causal_loss = F.mse_loss(predicted_effect, eb) / effect_normalizer
        if not torch.isfinite(value_loss) or not torch.isfinite(causal_loss):
            raise FloatingPointError("non-finite V18 projected-refinement loss")
        value_gradient = torch.autograd.grad(value_loss, parameters, retain_graph=True)
        causal_gradient = torch.autograd.grad(causal_loss, parameters)
        dot = sum(torch.sum(value * causal).item() for value, causal in zip(value_gradient, causal_gradient, strict=True))
        value_norm_squared = sum(torch.sum(value * value).item() for value in value_gradient)
        if not math.isfinite(dot) or not math.isfinite(value_norm_squared):
            raise FloatingPointError("V18 gradient conflict statistic is non-finite")
        coefficient = dot / value_norm_squared if dot < 0.0 and value_norm_squared > 1e-24 else 0.0
        if not math.isfinite(coefficient):
            raise FloatingPointError("V18 gradient projection coefficient is non-finite")
        if coefficient != 0.0:
            conflict_count += 1
        projected = tuple(causal - coefficient * value for value, causal in zip(value_gradient, causal_gradient, strict=True))
        post_dot = sum(torch.sum(value * causal).item() for value, causal in zip(value_gradient, projected, strict=True))
        pre_dots.append(float(dot))
        post_dots.append(float(post_dot))
        projection_coefficients.append(float(coefficient))
        for parameter, value, causal in zip(parameters, value_gradient, projected, strict=True):
            parameter.grad = value + causal
        torch.nn.utils.clip_grad_norm_(parameters, GRADIENT_CLIP, error_if_nonfinite=True)
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is None:
                    raise RuntimeError("V18 refinement produced a parameter without a gradient")
                parameter.add_(parameter.grad, alpha=-REFINEMENT_LEARNING_RATE)
                parameter.grad = None
        if (update + 1) % 100 == 0:
            if any(not torch.isfinite(parameter).all().item() for parameter in parameters):
                raise FloatingPointError("V18 candidate weights became non-finite")
            _check_caps(root, phase_start)

    candidate.eval()
    with torch.inference_mode():
        fit_value = float(F.mse_loss(candidate(x), y).item() / value_normalizer)
        fit_effect = float(F.mse_loss(candidate(x + delta) - candidate(x), effect).item() / effect_normalizer)
        validation_value = float(F.mse_loss(candidate(xv), yv).item() / value_normalizer)
        validation_effect = float(F.mse_loss(candidate(xv + deltav) - candidate(xv), effectv).item() / effect_normalizer)
    base_value = float(F.mse_loss(start_candidate(x), y).item() / value_normalizer)
    ratio = fit_value / max(base_value, 1e-12)
    receipt = {
        "start_state_sha256": v16._module_state_digest(start_candidate),
        "candidate_state_sha256": v16._module_state_digest(candidate),
        "updates": REFINEMENT_UPDATES,
        "batch_size": TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(draws.astype("<i4", copy=False).tobytes()).hexdigest(),
        "optimizer": "plain SGD after per-minibatch conflict projection",
        "learning_rate": REFINEMENT_LEARNING_RATE,
        "weight_decay": 0.0,
        "momentum": 0.0,
        "gradient_clip_norm": GRADIENT_CLIP,
        "pointwise_value_normalizer": value_normalizer,
        "causal_effect_normalizer": effect_normalizer,
        "conflict_updates": conflict_count,
        "conflict_fraction": conflict_count / REFINEMENT_UPDATES,
        "mean_preprojection_gradient_dot": float(np.mean(pre_dots)),
        "mean_postprojection_gradient_dot": float(np.mean(post_dots)),
        "mean_projection_coefficient": float(np.mean(projection_coefficients)),
        "normalized_fit_value_mse": fit_value,
        "normalized_fit_effect_mse": fit_effect,
        "normalized_validation_value_mse": validation_value,
        "normalized_validation_effect_mse": validation_effect,
        "fit_value_mse_ratio_to_matched_pointwise": ratio,
        "fit_retention_pass": ratio <= FIT_RETENTION_RATIO,
        "source_parameters_copied": False,
    }
    return candidate, receipt


def _candidate_manifest(root: Path) -> dict[str, Any]:
    _load_dependencies()
    path = root / "candidate-freeze.json"
    if not path.is_file():
        raise FileNotFoundError("V18 candidate freeze is missing")
    manifest = _read_json(path)
    material = {key: value for key, value in manifest.items() if key not in {"candidate_set_sha256", "candidate_freeze_sha256"}}
    digest = _digest(material)
    if manifest.get("candidate_set_sha256") != digest or manifest.get("candidate_freeze_sha256") != digest:
        raise ValueError("V18 candidate-freeze digest is invalid")
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("target_layer") != TARGET_LAYER
        or manifest.get("wrong_layer_control") != ADJACENT_LAYER
    ):
        raise ValueError("V18 candidates are bound to a different packet")
    for row in manifest.get("candidates", []):
        candidate_path = root / row["path"]
        if not candidate_path.is_file() or _sha_file(candidate_path) != row["file_sha256"]:
            raise ValueError("V18 candidate bytes differ from their frozen manifest")
        candidate = v16.WholeMLPCandidate()
        candidate.load_state_dict(load_file(str(candidate_path), device="cpu"), strict=True)
        if v16._module_state_digest(candidate) != row["state_sha256"]:
            raise ValueError("V18 candidate tensor digest differs from its manifest")
    expected = {(index, method) for index in range(len(fixtures.SEEDS)) for method in ("projected", "pointwise", "wrong_layer", "wrong_layer_base", "random")}
    actual = {(int(row["seed_index"]), row["method"]) for row in manifest.get("candidates", [])}
    if actual != expected or len(manifest.get("candidates", [])) != len(expected):
        raise ValueError("V18 candidate freeze has missing, extra, or duplicate method identities")
    projected_rows = [row for row in manifest.get("candidates", []) if row["method"] in {"projected", "wrong_layer"}]
    expected_retention = len(projected_rows) == 2 * len(fixtures.SEEDS) and all(
        isinstance(row.get("fit_receipt", {}).get("fit_retention_pass"), bool) for row in projected_rows
    ) and all(row["fit_receipt"]["fit_retention_pass"] for row in projected_rows)
    if manifest.get("fit_retention_pass_all") is not expected_retention:
        raise ValueError("V18 fit retention summary differs from the per-candidate receipts")
    for row in manifest.get("preparation_artifacts", []):
        artifact = root / row["path"]
        if not artifact.is_file() or artifact.stat().st_size != row["bytes"] or _sha_file(artifact) != row["sha256"]:
            raise ValueError(f"V18 preparation artifact differs from its candidate freeze: {row['path']}")
    return manifest


def prepare(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    _load_dependencies()
    root = _ensure_root(custody_path)
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("V18 candidates already frozen; do not regenerate in place")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        tree = _tree_receipt()
        model, tokenizer = v16._load_model()
        runtime = _runtime_receipt()
        layers = v16._layers(model)
        if len(layers) != 12 or TARGET_LAYER != 8 or ADJACENT_LAYER != 9:
            raise ValueError("V18 fixed block identities differ from the frozen OPT contract")
        for layer in (layers[TARGET_LAYER], layers[ADJACENT_LAYER]):
            if (layer.fc1.in_features, layer.fc1.out_features, layer.fc2.in_features, layer.fc2.out_features) != (768, 3072, 3072, 768):
                raise ValueError("V18 target/control MLP dimensions differ from the frozen OPT contract")
        visible, earlier_adaptations = _load_reference_fixtures()
        selection = _read_json(RESEARCH_ROOT / V16_PRIMARY_NAME / "block-selection.json")
        if selection.get("target_layer") != TARGET_LAYER or selection.get("visible_fixture_sha256") != visible.get("example_set_sha256"):
            raise ValueError("V18 target block is not bound to V16's visible-only selection")

        adaptations = []
        records_by_seed = []
        splits = []
        for seed_index, seed in enumerate(fixtures.SEEDS):
            payload = fixtures.fixture_payload(fixtures.generate_adaptation_fixture(seed_index), fixture_kind=f"adaptation-seed-{seed_index}")
            adaptations.append(payload)
            records_by_seed.append(payload["records"])
            splits.append(_split(payload["records"], seed))
        fixtures.validate_partition_disjointness(visible, *earlier_adaptations, *adaptations)
        _write_json(root / "fixtures" / "visible-reference.json", {
            "protocol_id": PROTOCOL_ID,
            "source_protocol_id": V16_PROTOCOL_ID,
            "source_example_set_sha256": visible["example_set_sha256"],
            "source_count": visible["count"],
            "source_selection_sha256": _sha_file(RESEARCH_ROOT / V16_PRIMARY_NAME / "block-selection.json"),
            "v17_parent_packet_sha256": _verify_local_freeze("docs/replacement-loss-v17-freeze.json")["packet_sha256"],
        }, immutable=True)
        for seed_index, payload in enumerate(adaptations):
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
            _write_json(root / "splits" / f"seed-{seed_index}.json", splits[seed_index], immutable=True)

        all_candidates = []
        all_retention_pass = True
        for seed_index, (records, split) in enumerate(zip(records_by_seed, splits, strict=True)):
            _check_caps(root, started)
            captures = v16._capture_adaptation(model, tokenizer, records, TARGET_LAYER, ADJACENT_LAYER)
            v16._save_npz_deterministic(root / "adaptation" / f"seed-{seed_index}.npz", **captures)
            fit_prompts = set(split["fit_prompt_indices"])
            validation_prompts = set(split["validation_prompt_indices"])
            prompt_ids = captures["prompt_index"]
            fit_mask = np.isin(prompt_ids, np.asarray(sorted(fit_prompts), dtype=np.int32))
            validation_mask = np.isin(prompt_ids, np.asarray(sorted(validation_prompts), dtype=np.int32))
            if not fit_mask.any() or not validation_mask.any() or np.any(fit_mask & validation_mask):
                raise ValueError("V18 prompt-level fit and validation partitions are invalid")
            fit_rows = {key: value[fit_mask] for key, value in captures.items()}
            validation_rows = {key: value[validation_mask] for key, value in captures.items()}
            train_seed = fixtures.SEEDS[seed_index] ^ 0x18B10C
            target_fit = v17._training_set(layers[TARGET_LAYER], fit_rows, records, fit_prompts, seed=train_seed, label="target")
            target_validation = v17._training_set(layers[TARGET_LAYER], validation_rows, records, validation_prompts, seed=train_seed ^ 0x55A, label="target")
            adjacent_fit = v17._training_set(layers[ADJACENT_LAYER], fit_rows, records, fit_prompts, seed=train_seed ^ 0x9A1, label="adjacent")
            adjacent_validation = v17._training_set(layers[ADJACENT_LAYER], validation_rows, records, validation_prompts, seed=train_seed ^ 0x9A1 ^ 0x55A, label="adjacent")
            if len(target_fit["x"]) != int(fit_mask.sum()) or len(target_validation["x"]) != int(validation_mask.sum()):
                raise ValueError("V18 causal-training rows do not match captured fit/validation token provenance")

            pointwise, random_control, pointwise_receipt = v17._train_candidate(
                target_fit, target_validation, seed=train_seed, causal_weight=0.0, root=root
            )
            adjacent_base, _adjacent_random, adjacent_base_receipt = v17._train_candidate(
                adjacent_fit, adjacent_validation, seed=train_seed ^ 0x9A1, causal_weight=0.0, root=root
            )
            projected, projected_receipt = _projected_refinement(
                pointwise, target_fit, target_validation, seed=train_seed ^ 0x18C, root=root, phase_start=started
            )
            wrong_layer, wrong_receipt = _projected_refinement(
                adjacent_base, adjacent_fit, adjacent_validation, seed=train_seed ^ 0x9A1 ^ 0x18C, root=root, phase_start=started
            )
            all_retention_pass &= bool(projected_receipt["fit_retention_pass"] and wrong_receipt["fit_retention_pass"])
            method_items = (
                ("projected", projected, {"pointwise_start": pointwise_receipt, **projected_receipt}),
                ("pointwise", pointwise, pointwise_receipt),
                ("wrong_layer", wrong_layer, {"pointwise_start": adjacent_base_receipt, **wrong_receipt}),
                ("wrong_layer_base", adjacent_base, adjacent_base_receipt),
                ("random", random_control, {"source_parameters_copied": False, "trained": False, "initialization_sha256": v16._module_state_digest(random_control)}),
            )
            for method, candidate, fit_receipt in method_items:
                saved = _save_candidate(root / "candidates" / f"seed-{seed_index}-{method}.safetensors", candidate)
                all_candidates.append({"seed_index": seed_index, "method": method, **saved, "fit_receipt": fit_receipt})
            del captures, pointwise, projected, wrong_layer, random_control, adjacent_base
            del target_fit, target_validation, adjacent_fit, adjacent_validation, fit_rows, validation_rows
            _check_caps(root, started)

        relatives = [
            "fixtures/visible-reference.json",
            *(f"fixtures/adaptation-seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
            *(f"splits/seed-{seed}.json" for seed in range(len(fixtures.SEEDS))),
            *(f"adaptation/seed-{seed}.npz" for seed in range(len(fixtures.SEEDS))),
        ]
        material = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "target_model_id": TARGET_MODEL_ID,
            "target_revision": TARGET_REVISION,
            "target_tree_sha256": tree["tree_sha256"],
            "target_weight_sha256": tree["weight_sha256"],
            "target_layer": TARGET_LAYER,
            "wrong_layer_control": ADJACENT_LAYER,
            "preparation_artifacts": _artifact_rows(root, relatives),
            "candidates": all_candidates,
            "fit_retention_pass_all": all_retention_pass,
            "training": {
                "base_updates": BASE_UPDATES,
                "refinement_updates": REFINEMENT_UPDATES,
                "batch_size": TRAIN_BATCH_SIZE,
                "base_optimizer": "AdamW",
                "base_learning_rate": BASE_LEARNING_RATE,
                "base_weight_decay": BASE_WEIGHT_DECAY,
                "refinement_optimizer": "plain SGD, causal gradient projected away from first-order pointwise conflict",
                "refinement_learning_rate": REFINEMENT_LEARNING_RATE,
                "gradient_clip_norm": GRADIENT_CLIP,
                "fit_retention_ratio_max": FIT_RETENTION_RATIO,
                "validation_used_for_selection": False,
            },
            "runtime_versions": {key: runtime[key] for key in ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "repository_head", "deterministic_algorithms", "torch_num_threads")},
        }
        digest = _digest(material)
        material["candidate_set_sha256"] = digest
        material["candidate_freeze_sha256"] = digest
        _write_json(root / "candidate-freeze.json", material, immutable=True)
        _write_json(root / "runtime.json", runtime, immutable=True)
        _candidate_manifest(root)
        _record_phase(root, "prepare", started, "COMPLETED", details={"seed_count": len(fixtures.SEEDS), "candidate_count": len(all_candidates), "fit_retention_pass_all": all_retention_pass, "candidate_set_sha256": digest})
        return {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "phase": "prepare",
            "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "candidate_set_sha256": digest,
            "fit_retention_pass_all": all_retention_pass,
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
            "data_and_artifact_custody": "adaptation prompts, captures, finite-effect targets, and candidate tensors remain in external owner custody",
        }
    except Exception as error:
        _stop(root, "prepare", error)
        _record_phase(root, "prepare", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _candidate_for(root: Path, seed_index: int, method: str, manifest: dict[str, Any]) -> torch.nn.Module:
    _load_dependencies()
    if method == "no_op":
        return v16.NoOpCandidate().eval()
    row = next((item for item in manifest["candidates"] if item["seed_index"] == seed_index and item["method"] == method), None)
    if row is None:
        raise ValueError(f"V18 candidate freeze lacks {method} for seed {seed_index}")
    candidate = v16.WholeMLPCandidate()
    path = root / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("V18 candidate bytes differ from the frozen candidate set")
    candidate.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if v16._module_state_digest(candidate) != row["state_sha256"]:
        raise ValueError("V18 candidate tensor state digest differs")
    return candidate.eval()


def _fixture_payload(root: Path, path: Path, kind: str, expected_count: int) -> dict[str, Any]:
    payload = _read_json(path)
    rows = payload.get("records")
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("fixture_kind") != kind
        or payload.get("prompt_rules_version") != fixtures.PROMPT_RULES_VERSION
        or not isinstance(rows, list)
        or len(rows) != expected_count
        or payload.get("count") != expected_count
    ):
        raise ValueError("V18 fixture identity or count differs from the frozen contract")
    digests = [hashlib.sha256(_canonical_bytes(row)).hexdigest() for row in rows]
    expected = hashlib.sha256("\n".join(sorted(digests)).encode("ascii")).hexdigest()
    expected_counts = {name: fixtures.ADAPTATION_PER_STRATUM for name in fixtures.STRATA} if kind.startswith("adaptation-") else {name: fixtures.CHALLENGE_PER_STRATUM for name in fixtures.STRATA}
    actual_counts = {name: sum(row.get("stratum") == name for row in rows) for name in fixtures.STRATA}
    if expected != payload.get("example_set_sha256") or len({row.get("example_id") for row in rows}) != expected_count or actual_counts != expected_counts or payload.get("stratum_counts") != dict(sorted(expected_counts.items())):
        raise ValueError("V18 fixture content digest or stratum balance differs")
    return payload


def _entropy_manifest(root: Path, candidate_set_sha: str) -> dict[str, Any]:
    manifest = _read_json(root / "challenge-entropy-manifest.json")
    rows = manifest.get("entropy", [])
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidate_set_sha
        or manifest.get("entropy_manifest_sha256") != _digest(rows)
        or [row.get("seed_index") for row in rows] != list(range(len(fixtures.SEEDS)))
    ):
        raise ValueError("V18 challenge entropy manifest is invalid or bound to different candidates")
    for row in rows:
        entropy_path = root / "challenge-entropy" / f"seed-{row['seed_index']}.bin"
        if not entropy_path.is_file() or entropy_path.stat().st_size != 32 or _sha_file(entropy_path) != row["sha256"]:
            raise ValueError("V18 challenge entropy bytes differ from their manifest")
    return manifest


def _previous_and_current_adaptation(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    visible, old_adaptations = _load_reference_fixtures()
    current = [
        _fixture_payload(root, root / "fixtures" / f"adaptation-seed-{index}.json", f"adaptation-seed-{index}", fixtures.ADAPTATION_PER_STRATUM * len(fixtures.STRATA))
        for index in range(len(fixtures.SEEDS))
    ]
    return visible, old_adaptations + current


def _heldout_directions(inputs: np.ndarray, records: list[dict[str, Any]], rng: np.random.Generator) -> np.ndarray:
    deltas = np.zeros_like(inputs, dtype=np.float32)
    for index, (activation, record) in enumerate(zip(inputs, records, strict=True)):
        rms = max(float(np.sqrt(np.mean(np.square(activation, dtype=np.float64)))), 1e-12)
        if record["stratum"] == "adversarial":
            vector = np.zeros_like(activation, dtype=np.float64)
            vector[int(np.argmax(np.abs(activation)))] = 1.0 if rng.integers(0, 2) else -1.0
        else:
            vector = rng.normal(size=activation.shape).astype(np.float64)
            if record["stratum"] == "compositional":
                vector += rng.normal(size=activation.shape)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-15:
            raise ValueError("V18 held-out direction has zero norm")
        deltas[index] = (vector * (0.01 * rms / norm)).astype(np.float32)
    return deltas


def _predict_seed(root: Path, seed_index: int, entropy: bytes, payload: dict[str, Any], candidates: dict[str, Any], model: torch.nn.Module, tokenizer: Any) -> dict[str, Any]:
    encoded = [v16._encode(tokenizer, row, with_continuation=True) for row in payload["records"]]
    methods = ("projected", "pointwise", "wrong_layer", "random", "no_op")
    loaded = {method: _candidate_for(root, seed_index, method, candidates) for method in methods}
    outputs: dict[str, dict[str, list[np.ndarray]]] = {
        method: {name: [] for name in ("module_delta", "logit_delta", "module_base", "boundary_logits", "continuation_ce")}
        for method in methods
    }
    direction_chunks = []
    direction_seed = int.from_bytes(hashlib.sha256(entropy + b"v18-heldout-directions").digest()[:8], "big")
    direction_rng = np.random.default_rng(direction_seed)
    for start in range(0, len(encoded), INFERENCE_BATCH_SIZE):
        _check_caps(root)
        group = encoded[start : start + INFERENCE_BATCH_SIZE]
        positions = [int(row["prompt_len"]) - 1 for row in group]
        projected_base = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=loaded["projected"], positions=positions, capture_inputs=True, capture_outputs=True)
        directions = _heldout_directions(projected_base["module_input"], payload["records"][start : start + len(group)], direction_rng)
        direction_chunks.append(directions)
        for method in methods:
            candidate = loaded[method]
            base = projected_base if method == "projected" else v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=candidate, positions=positions, capture_outputs=True)
            changed = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=candidate, positions=positions, deltas=directions, capture_outputs=True)
            outputs[method]["module_delta"].append((changed["module_output"] - base["module_output"]).astype(np.float32))
            outputs[method]["logit_delta"].append((changed["boundary_logits"] - base["boundary_logits"]).astype(np.float32))
            outputs[method]["module_base"].append(base["module_output"].astype(np.float32))
            outputs[method]["boundary_logits"].append(base["boundary_logits"].astype(np.float32))
            outputs[method]["continuation_ce"].append(base["continuation_ce"].astype(np.float32))
    arrays: dict[str, np.ndarray] = {"directions": np.concatenate(direction_chunks, axis=0)}
    for method, metrics in outputs.items():
        for metric, chunks in metrics.items():
            arrays[f"{method}_{metric}"] = np.concatenate(chunks, axis=0)
    path = root / "predictions" / f"seed-{seed_index}.npz"
    v16._save_npz_deterministic(path, **arrays)
    return {"seed_index": seed_index, "challenge_count": len(payload["records"]), "challenge_sha256": payload["example_set_sha256"], "prediction_file_sha256": _sha_file(path), "prediction_file_bytes": path.stat().st_size}


def predict(custody_path: str | Path, *, entropy_from: str | Path | None = None) -> dict[str, Any]:
    packet = verify_packet()
    root = _ensure_root(custody_path)
    _load_dependencies()
    if not (root / "candidate-freeze.json").is_file():
        raise FileNotFoundError("V18 candidates must be frozen before challenge generation")
    if (root / "prediction-manifest.json").exists():
        raise FileExistsError("V18 predictions already sealed; do not overwrite")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        candidates = _candidate_manifest(root)
        entropy_root = root / "challenge-entropy"
        entropy_root.mkdir(parents=True, exist_ok=True)
        if entropy_from is None:
            for seed_index in range(len(fixtures.SEEDS)):
                entropy_path = entropy_root / f"seed-{seed_index}.bin"
                if entropy_path.exists():
                    raise FileExistsError("V18 entropy exists; do not redraw or overwrite")
                entropy_path.write_bytes(secrets.token_bytes(32))
        else:
            source = _ensure_root(entropy_from)
            source_candidates = _candidate_manifest(source)
            _entropy_manifest(source, source_candidates["candidate_set_sha256"])
            for seed_index in range(len(fixtures.SEEDS)):
                source_path = source / "challenge-entropy" / f"seed-{seed_index}.bin"
                destination = entropy_root / source_path.name
                if destination.exists():
                    raise FileExistsError("V18 replica entropy destination exists")
                shutil.copyfile(source_path, destination)
        entropy_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            raw = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            if len(raw) != 32:
                raise ValueError("V18 entropy must contain exactly 256 bits per seed")
            entropy_rows.append({"seed_index": seed_index, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": 32})
        entropy_record = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "entropy": entropy_rows,
            "entropy_manifest_sha256": _digest(entropy_rows),
        }
        _write_json(root / "challenge-entropy-manifest.json", entropy_record, immutable=True)
        _entropy_manifest(root, candidates["candidate_set_sha256"])
        visible, known_adaptations = _previous_and_current_adaptation(root)
        challenge_payloads = []
        for seed_index in range(len(fixtures.SEEDS)):
            secret = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            challenge_payloads.append(fixtures.fixture_payload(
                fixtures.generate_challenge_fixture(seed_index, secret),
                fixture_kind=f"challenge-seed-{seed_index}",
            ))
        fixtures.validate_partition_disjointness(visible, *known_adaptations, *challenge_payloads)
        for seed_index, payload in enumerate(challenge_payloads):
            _write_json(root / "fixtures" / f"challenge-seed-{seed_index}.json", payload, immutable=True)
        model, tokenizer = v16._load_model()
        challenge_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            _check_caps(root, started)
            secret = (entropy_root / f"seed-{seed_index}.bin").read_bytes()
            challenge_rows.append(_predict_seed(root, seed_index, secret, challenge_payloads[seed_index], candidates, model, tokenizer))
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "entropy_manifest_sha256": entropy_record["entropy_manifest_sha256"],
            "target_layer": TARGET_LAYER,
            "challenge_rows": challenge_rows,
        }
        manifest["prediction_set_sha256"] = _digest(manifest)
        _write_json(root / "prediction-manifest.json", manifest, immutable=True)
        _prediction_manifest(root)
        _record_phase(root, "predict", started, "COMPLETED", details={"challenge_count": sum(row["challenge_count"] for row in challenge_rows), "prediction_set_sha256": manifest["prediction_set_sha256"]})
        return {
            "protocol_id": PROTOCOL_ID,
            "phase": "predict",
            "status": "COMPLETED",
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "challenge_count": sum(row["challenge_count"] for row in challenge_rows),
            "prediction_set_sha256": manifest["prediction_set_sha256"],
            "candidate_was_frozen_before_challenge": True,
            "behavior_and_mechanism_diverged": "NOT_YET_EVALUATED",
        }
    except Exception as error:
        _stop(root, "predict", error)
        _record_phase(root, "predict", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def _prediction_manifest(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "prediction-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "prediction_set_sha256"}
    candidates = _candidate_manifest(root)
    entropy = _entropy_manifest(root, candidates["candidate_set_sha256"])
    if (
        manifest.get("prediction_set_sha256") != _digest(material)
        or manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("packet_sha256") != verify_packet()["packet_sha256"]
        or manifest.get("candidate_set_sha256") != candidates["candidate_set_sha256"]
        or manifest.get("entropy_manifest_sha256") != entropy["entropy_manifest_sha256"]
        or manifest.get("target_layer") != TARGET_LAYER
    ):
        raise ValueError("V18 prediction manifest does not bind its packet, candidates, entropy, and target")
    rows = manifest.get("challenge_rows", [])
    if [row.get("seed_index") for row in rows] != list(range(len(fixtures.SEEDS))):
        raise ValueError("V18 prediction manifest has missing or unordered challenge seeds")
    visible, known_adaptations = _previous_and_current_adaptation(root)
    challenge_payloads = []
    for row in rows:
        seed_index = int(row["seed_index"])
        payload = _fixture_payload(root, root / "fixtures" / f"challenge-seed-{seed_index}.json", f"challenge-seed-{seed_index}", fixtures.CHALLENGE_PER_STRATUM * len(fixtures.STRATA))
        challenge_payloads.append(payload)
        prediction_path = root / "predictions" / f"seed-{seed_index}.npz"
        if row.get("challenge_sha256") != payload["example_set_sha256"] or row.get("challenge_count") != payload["count"] or not prediction_path.is_file() or _sha_file(prediction_path) != row["prediction_file_sha256"]:
            raise ValueError("V18 challenge fixture or prediction bytes differ from their sealed manifest")
    fixtures.validate_partition_disjointness(visible, *known_adaptations, *challenge_payloads)
    return manifest


def _score_seed(root: Path, seed_index: int) -> dict[str, Any]:
    _load_dependencies()
    payload = _fixture_payload(root, root / "fixtures" / f"challenge-seed-{seed_index}.json", f"challenge-seed-{seed_index}", fixtures.CHALLENGE_PER_STRATUM * len(fixtures.STRATA))
    model, tokenizer = v16._load_model()
    encoded = [v16._encode(tokenizer, row, with_continuation=True) for row in payload["records"]]
    with np.load(root / "predictions" / f"seed-{seed_index}.npz", allow_pickle=False) as stored:
        prediction = {name: stored[name] for name in stored.files}
    methods = ("projected", "pointwise", "wrong_layer", "random", "no_op")
    metrics: dict[str, list[dict[str, Any]]] = {name: [] for name in methods}
    activation_rows = []
    behavior_rows = []
    oracle_errors = []
    candidates = _candidate_manifest(root)
    treatment = _candidate_for(root, seed_index, "projected", candidates)
    for start in range(0, len(encoded), INFERENCE_BATCH_SIZE):
        _check_caps(root)
        group = encoded[start : start + INFERENCE_BATCH_SIZE]
        positions = [int(row["prompt_len"]) - 1 for row in group]
        original = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=None, positions=positions, capture_inputs=True, capture_outputs=True, capture_all_outputs=True)
        directions = prediction["directions"][start : start + len(group)]
        observed = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=None, positions=positions, deltas=directions, capture_outputs=True)
        teacher_module_effect = observed["module_output"] - original["module_output"]
        teacher_logit_effect = observed["boundary_logits"] - original["boundary_logits"]
        module_scale = np.maximum(np.linalg.norm(original["module_output"], axis=1), 1e-12)
        logit_scale = np.maximum(np.linalg.norm(original["boundary_logits"], axis=1), 1e-12)
        record_group = payload["records"][start : start + len(group)]
        for method in methods:
            predicted_module = prediction[f"{method}_module_delta"][start : start + len(group)]
            predicted_logits = prediction[f"{method}_logit_delta"][start : start + len(group)]
            module_metrics = v16._metric_rows(predicted_module, teacher_module_effect, module_scale)
            logit_metrics = v16._metric_rows(predicted_logits, teacher_logit_effect, logit_scale)
            metrics[method].extend({"stratum": row["stratum"], "module": module, "logit": logit} for row, module, logit in zip(record_group, module_metrics, logit_metrics, strict=True))
        treatment_base = prediction["projected_module_base"][start : start + len(group)]
        for row, candidate_output, teacher_output in zip(record_group, treatment_base, original["module_output"], strict=True):
            a = np.asarray(candidate_output, dtype=np.float64)
            b = np.asarray(teacher_output, dtype=np.float64)
            activation_rows.append({"stratum": row["stratum"], "cosine": float(np.clip(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30), -1.0, 1.0)), "relative_error": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))})
        treatment_logits = prediction["projected_boundary_logits"][start : start + len(group)]
        treatment_ce = prediction["projected_continuation_ce"][start : start + len(group)]
        for index, row in enumerate(record_group):
            behavior_rows.append({
                "stratum": row["stratum"],
                "cross_entropy_increase": float(treatment_ce[index] - original["continuation_ce"][index]),
                "normalized_next_logit_error": float(np.linalg.norm(treatment_logits[index].astype(np.float64) - original["boundary_logits"][index].astype(np.float64)) / max(np.linalg.norm(original["boundary_logits"][index]), 1e-12)),
            })
        oracle = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=treatment, positions=positions, capture_outputs=False, restore_all_outputs=original["all_module_output"])
        active = original["attention_mask"].bool()
        oracle_errors.append(float((original["all_logits"][active] - oracle["all_logits"][active]).abs().max().item()))

    result = {"seed_index": seed_index, "fit_retention_pass": next(row["fit_receipt"]["fit_retention_pass"] for row in candidates["candidates"] if row["seed_index"] == seed_index and row["method"] == "projected"), "strata": {}}
    for stratum in fixtures.STRATA:
        result["strata"][stratum] = {}
        for method in methods:
            selected = [row for row in metrics[method] if row["stratum"] == stratum]
            result["strata"][stratum][method] = {
                "module_effect": v16._summarize_effect([row["module"] for row in selected]),
                "next_logit_effect": v16._summarize_effect([row["logit"] for row in selected]),
            }
        selected_activation = [row for row in activation_rows if row["stratum"] == stratum]
        selected_behavior = [row for row in behavior_rows if row["stratum"] == stratum]
        result["strata"][stratum]["activation"] = {
            "median_cosine": float(np.median([row["cosine"] for row in selected_activation])),
            "median_relative_error": float(np.median([row["relative_error"] for row in selected_activation])),
        }
        result["strata"][stratum]["behavior"] = {
            "mean_cross_entropy_increase": float(np.mean([row["cross_entropy_increase"] for row in selected_behavior])),
            "median_normalized_next_logit_error": float(np.median([row["normalized_next_logit_error"] for row in selected_behavior])),
        }
        projected = result["strata"][stratum]["projected"]["next_logit_effect"]
        pointwise = result["strata"][stratum]["pointwise"]["next_logit_effect"]
        result["strata"][stratum]["projected_vs_pointwise"] = {
            "next_logit_median_cosine_gain": float(projected["median_cosine"] - pointwise["median_cosine"]) if projected["median_cosine"] is not None and pointwise["median_cosine"] is not None else None,
            "next_logit_relative_error_reduction": float(pointwise["median_normalized_error"] - projected["median_normalized_error"]) if projected["median_normalized_error"] is not None and pointwise["median_normalized_error"] is not None else None,
        }
        result["strata"][stratum]["control_cosine_margins"] = {
            name: float(projected["median_cosine"] - result["strata"][stratum][name]["next_logit_effect"]["median_cosine"])
            if projected["median_cosine"] is not None and result["strata"][stratum][name]["next_logit_effect"]["median_cosine"] is not None else None
            for name in ("wrong_layer", "random", "no_op")
        }
    result["oracle_max_absolute_logit_error"] = max(oracle_errors) if oracle_errors else None
    return result


def _scientific_outcome(results: list[dict[str, Any]]) -> str:
    incomplete = False
    passed = True
    for seed in results:
        if not seed.get("fit_retention_pass", False):
            passed = False
        for stratum in seed["strata"].values():
            treatment = stratum["projected"]
            for measure in (treatment["module_effect"], treatment["next_logit_effect"]):
                if measure["resolvable_count"] < 16:
                    incomplete = True
                passed &= v16._passes_effect(measure)
            passed &= stratum["activation"]["median_cosine"] >= 0.90
            passed &= stratum["activation"]["median_relative_error"] <= 0.20
            passed &= stratum["behavior"]["mean_cross_entropy_increase"] <= 0.05
            passed &= stratum["behavior"]["median_normalized_next_logit_error"] <= 0.05
            comparison = stratum["projected_vs_pointwise"]
            if comparison["next_logit_median_cosine_gain"] is None or comparison["next_logit_relative_error_reduction"] is None:
                incomplete = True
                passed = False
            else:
                passed &= comparison["next_logit_median_cosine_gain"] >= 0.05
                passed &= comparison["next_logit_relative_error_reduction"] >= 0.10
            for margin in stratum["control_cosine_margins"].values():
                if margin is None:
                    incomplete = True
                    passed = False
                else:
                    passed &= margin >= 0.05
        if seed["oracle_max_absolute_logit_error"] is None:
            incomplete = True
            passed = False
        else:
            passed &= seed["oracle_max_absolute_logit_error"] <= 1e-5
    return "INCONCLUSIVE" if incomplete else "PASS" if passed else "FAIL"


def _behavior_mechanism_diverged(results: list[dict[str, Any]]) -> bool | None:
    values = []
    for seed in results:
        for stratum in seed["strata"].values():
            values.extend((
                (stratum["behavior"]["mean_cross_entropy_increase"], 0.05),
                (stratum["behavior"]["median_normalized_next_logit_error"], 0.05),
                (stratum["activation"]["median_relative_error"], 0.20),
                (stratum["projected"]["module_effect"]["median_normalized_error"], 0.20),
                (stratum["projected"]["next_logit_effect"]["median_normalized_error"], 0.20),
            ))
    if any(value is None for value, _threshold in values):
        return None
    return any(float(value) > threshold for value, threshold in values)


def score(custody_path: str | Path) -> dict[str, Any]:
    packet = verify_packet()
    root = _ensure_root(custody_path)
    _load_dependencies()
    if not (root / "prediction-manifest.json").is_file():
        raise FileNotFoundError("V18 sealed predictions must exist before scoring")
    if (root / "outcome-report.json").exists():
        raise FileExistsError("V18 outcome is already frozen; do not rescore in place")
    started = time.monotonic()
    _phase_ledger(root)
    try:
        candidates = _candidate_manifest(root)
        predictions = _prediction_manifest(root)
        results = [_score_seed(root, seed_index) for seed_index in range(len(fixtures.SEEDS))]
        outcome = _scientific_outcome(results)
        report = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": candidates["candidate_set_sha256"],
            "prediction_set_sha256": predictions["prediction_set_sha256"],
            "target_identity": {
                "model_id": TARGET_MODEL_ID,
                "revision": TARGET_REVISION,
                "tree_sha256": TARGET_TREE_SHA256,
                "weight_sha256": TARGET_WEIGHT_SHA256,
                "target_layer": TARGET_LAYER,
                "control_layer": ADJACENT_LAYER,
                "selection_source": "V16 archived visible-only selection fixed before V17 and V18",
            },
            "scientific_outcome": outcome,
            "behavior_and_mechanism_diverged": _behavior_mechanism_diverged(results),
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "blind_status": "NOT_BLIND_LOCAL_CUSTODY_RUN",
            "results": results,
            "claim_ceiling": CLAIM_CEILING,
            "data_and_artifact_custody": "synthetic prompts, captures, fit arrays, candidates, challenge entropy, predictions, and row-level outcomes remain in external owner custody",
            "runtime": _runtime_receipt(),
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
            "limitations": [
                "post-result exploratory follow-up on one reused OPT/ReLU checkpoint and the previously selected block 8",
                "three synthetic seeds on one checkpoint do not constitute architecture or checkpoint replication",
                "activation similarity and oracle output restoration are not learned activation rescue",
                "finite-effect prediction measures local intervention response, not identity, consciousness, or a novel Theseus proof",
                "same-owner second-runtime replay is not blind replication or independent acceptance",
            ],
            "next_blocked_authority_gate": "blind external replication and an authorized independent reviewer are required for independent scientific acceptance",
        }
        scientific = {key: value for key, value in report.items() if key not in {"runtime", "assurance_level", "reviewer_status", "reproducibility_status", "blind_status"}}
        report["scientific_results_sha256"] = _digest(scientific)
        report["report_sha256"] = _digest(report)
        _write_json(root / "outcome-report.json", report, immutable=True)
        _record_phase(root, "score", started, "COMPLETED", details={"scientific_outcome": outcome, "scientific_results_sha256": report["scientific_results_sha256"]})
        return {
            "protocol_id": PROTOCOL_ID,
            "phase": "score",
            "status": "COMPLETED",
            "scientific_outcome": outcome,
            "scientific_results_sha256": report["scientific_results_sha256"],
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "behavior_and_mechanism_diverged": report["behavior_and_mechanism_diverged"],
            "claim_ceiling": CLAIM_CEILING,
        }
    except Exception as error:
        _stop(root, "score", error)
        _record_phase(root, "score", started, "STOPPED", details={"error_type": type(error).__name__})
        raise


def preflight() -> dict[str, Any]:
    packet = verify_packet()
    tree = _tree_receipt()
    model, _tokenizer = v16._load_model()
    runtime = _runtime_receipt()
    layers = v16._layers(model)
    if len(layers) != 12:
        raise ValueError("V18 target is not a 12-layer OPT decoder")
    for index in (TARGET_LAYER, ADJACENT_LAYER):
        layer = layers[index]
        if (layer.fc1.in_features, layer.fc1.out_features, layer.fc2.in_features, layer.fc2.out_features) != (768, 3072, 3072, 768):
            raise ValueError("V18 target/control block differs from the frozen MLP shape")
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "target_identity": {"model_id": TARGET_MODEL_ID, "revision": TARGET_REVISION, **tree},
        "layers": {"target": TARGET_LAYER, "wrong_layer_control": ADJACENT_LAYER},
        "runtime": runtime,
        "wall_cap_seconds": MAX_WALL_SECONDS,
        "rss_cap_bytes": MAX_RSS_BYTES,
    }


def run_primary(custody_path: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_path)
    if root.name != PRIMARY_NAME:
        raise ValueError("run-primary only accepts the frozen V18 primary custody root")
    packet = freeze_packet()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "freeze", "status": "COMPLETED", "packet_sha256": packet["packet_sha256"]}), file=sys.stderr, flush=True)
    preflight_receipt = preflight()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "preflight", "status": "COMPLETED", "target_tree_sha256": preflight_receipt["target_identity"]["tree_sha256"], "runtime_inventory_sha256": preflight_receipt["runtime"]["package_inventory_sha256"]}), file=sys.stderr, flush=True)
    archive_receipt = archive_packet(root)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "archive", "status": "COMPLETED", "archive_sha256": archive_receipt["archive_sha256"]}), file=sys.stderr, flush=True)
    prepared = prepare(root)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "prepare", "status": prepared["status"], "candidate_set_sha256": prepared.get("candidate_set_sha256"), "fit_retention_pass_all": prepared.get("fit_retention_pass_all")}), file=sys.stderr, flush=True)
    predicted = predict(root)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "predict", "status": predicted["status"], "prediction_set_sha256": predicted.get("prediction_set_sha256")}), file=sys.stderr, flush=True)
    scored = score(root)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "score", "status": scored["status"], "scientific_outcome": scored.get("scientific_outcome"), "scientific_results_sha256": scored.get("scientific_results_sha256")}), file=sys.stderr, flush=True)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "archive_sha256": archive_receipt["archive_sha256"],
        "preflight": preflight_receipt,
        "prepare": prepared,
        "predict": predicted,
        "score": scored,
    }


def _file_group(root: Path, relatives: Iterable[str]) -> list[dict[str, Any]]:
    result = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V18 replay artifact is missing: {relative}")
        result.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return result


def _relative_files(root: Path, prefix: str) -> list[str]:
    directory = root / prefix
    if not directory.is_dir():
        return []
    return [path.relative_to(root).as_posix() for path in sorted(directory.rglob("*")) if path.is_file()]


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _ensure_root(primary_path)
    replica = _ensure_root(replica_path)
    primary_archive = _external_packet_archive(primary)
    replica_archive = _external_packet_archive(replica)
    if primary_archive["archive_sha256"] != replica_archive["archive_sha256"]:
        raise ValueError("V18 primary and replica frozen packet archives differ")
    left_report = _read_json(primary / "outcome-report.json")
    right_report = _read_json(replica / "outcome-report.json")
    groups_spec = {
        "fixtures": (_relative_files(primary, "fixtures"), _relative_files(replica, "fixtures")),
        "splits": (_relative_files(primary, "splits"), _relative_files(replica, "splits")),
        "adaptation_captures": (_relative_files(primary, "adaptation"), _relative_files(replica, "adaptation")),
        "candidates": (_relative_files(primary, "candidates") + ["candidate-freeze.json"], _relative_files(replica, "candidates") + ["candidate-freeze.json"]),
        "entropy_and_predictions": (
            _relative_files(primary, "challenge-entropy") + ["challenge-entropy-manifest.json"] + _relative_files(primary, "predictions") + ["prediction-manifest.json"],
            _relative_files(replica, "challenge-entropy") + ["challenge-entropy-manifest.json"] + _relative_files(replica, "predictions") + ["prediction-manifest.json"],
        ),
    }
    groups = {}
    for name, (left_names, right_names) in groups_spec.items():
        left = _file_group(primary, left_names)
        right = _file_group(replica, right_names)
        groups[name] = {"primary": left, "replica": right, "byte_identical": left == right}
    scientific_match = left_report.get("scientific_results_sha256") == right_report.get("scientific_results_sha256") and left_report.get("scientific_outcome") == right_report.get("scientific_outcome")
    left_runtime = _read_json(primary / "runtime.json")
    right_runtime = _read_json(replica / "runtime.json")
    pins_match = all(left_runtime.get(field) == right_runtime.get(field) for field in ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "deterministic_algorithms", "torch_num_threads"))
    separate_runtime = left_runtime.get("python_executable") != right_runtime.get("python_executable")
    same_host = left_runtime.get("host_id_sha256") == right_runtime.get("host_id_sha256")
    same_owner = left_runtime.get("uid") == right_runtime.get("uid")
    groups_match = all(row["byte_identical"] for row in groups.values())
    reproduced = groups_match and scientific_match and pins_match and separate_runtime and same_host and same_owner
    report = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": verify_packet()["packet_sha256"],
        "archive_sha256": [primary_archive["archive_sha256"], replica_archive["archive_sha256"]],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reproducibility_status": "REPRODUCED_LOCAL" if reproduced else "REPLAY_MISMATCH_OR_INCOMPLETE",
        "scientific_outcome": left_report.get("scientific_outcome") if scientific_match else "INCONCLUSIVE",
        "scientific_results_match": scientific_match,
        "artifact_groups": groups,
        "runtime_match": pins_match,
        "separate_python_runtime": separate_runtime,
        "same_host": same_host,
        "same_owner": same_owner,
        "custody": "prompts, captures, candidates, entropy, predictions, and row-level outcomes remain in external owner custody",
        "reviewer_status": "NOT_SUBMITTED",
        "next_blocked_authority_gate": "blind external replication and authorized independent review",
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "replication-report.json", report, immutable=True)
    left_report["assurance_level"] = report["assurance_level"]
    left_report["reproducibility_status"] = report["reproducibility_status"]
    left_report["replication_report_sha256"] = report["report_sha256"]
    left_report.pop("report_sha256", None)
    left_report["report_sha256"] = _digest(left_report)
    _write_json(primary / "outcome-report.json", left_report)
    return {"protocol_id": PROTOCOL_ID, "assurance_level": report["assurance_level"], "scientific_outcome": report["scientific_outcome"], "scientific_results_match": scientific_match, "artifact_groups_match": groups_match, "runtime_match": pins_match, "separate_python_runtime": separate_runtime, "same_host": same_host, "same_owner": same_owner, "report_sha256": report["report_sha256"]}


def replicate(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    replica_runtime = preflight()
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "replica-preflight", "status": "COMPLETED", "runtime_inventory_sha256": replica_runtime["runtime"]["package_inventory_sha256"]}), file=sys.stderr, flush=True)
    primary = _ensure_root(primary_path)
    replica = _custody_root(replica_path)
    if replica.exists():
        raise FileExistsError("V18 replica custody root exists; preserve it and choose a fresh root")
    replica.mkdir(parents=True, exist_ok=False)
    shutil.copytree(primary / "frozen-packet", replica / "frozen-packet")
    shutil.copyfile(primary / "frozen-packet-manifest.json", replica / "frozen-packet-manifest.json")
    _external_packet_archive(replica)
    prepared = prepare(replica)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "replica-prepare", "status": prepared["status"], "candidate_set_sha256": prepared.get("candidate_set_sha256")}), file=sys.stderr, flush=True)
    predicted = predict(replica, entropy_from=primary)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "replica-predict", "status": predicted["status"], "prediction_set_sha256": predicted.get("prediction_set_sha256")}), file=sys.stderr, flush=True)
    scored = score(replica)
    print(json.dumps({"protocol_id": PROTOCOL_ID, "phase": "replica-score", "status": scored["status"], "scientific_outcome": scored.get("scientific_outcome")}), file=sys.stderr, flush=True)
    comparison = compare_replication(primary, replica)
    comparison["replica_preflight"] = replica_runtime
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "preflight"):
        commands.add_parser(name)
    run_command = commands.add_parser("run-primary")
    run_command.add_argument("--custody", required=True)
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
    compare_command = commands.add_parser("compare")
    compare_command.add_argument("--primary", required=True)
    compare_command.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "verify":
        result = verify_packet()
    elif args.command == "preflight":
        result = preflight()
    elif args.command == "run-primary":
        result = run_primary(args.custody)
    elif args.command == "archive":
        result = archive_packet(args.custody)
    elif args.command == "prepare":
        result = prepare(args.custody)
    elif args.command == "predict":
        result = predict(args.custody, entropy_from=args.entropy_from)
    elif args.command == "score":
        result = score(args.custody)
    elif args.command == "replicate":
        result = replicate(args.primary, args.replica)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
