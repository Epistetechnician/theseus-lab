"""Staged V19 local causal/value qualification and confirmation runner.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v19-causal-fit-qualification-v1, and
global-experiment-assurance-lifecycle-v1.

Prompt-bearing and row-level artifacts remain under external owner custody.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import replacement_loss_v19_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
DEV_PROTOCOL_ID = fixtures.DEV_PROTOCOL_ID
CONFIRM_PROTOCOL_ID = fixtures.CONFIRM_PROTOCOL_ID
DEV_FREEZE_FILE = "docs/replacement-loss-v19-development-freeze.json"
CONFIRM_FREEZE_FILE = "docs/replacement-loss-v19-confirmation-freeze.json"
CONFIRM_CONFIG_FILE = "docs/replacement-loss-v19-confirmation-config.json"
DEV_RESULT_JSON = "docs/replacement-loss-v19-development-results.json"
DEV_RESULT_MD = "docs/replacement-loss-v19-development-results.md"
DEV_REPLICATION_JSON = "docs/replacement-loss-v19-development-replication.json"
CONFIRM_RESULT_JSON = "docs/replacement-loss-v19-confirmation-results.json"
CONFIRM_RESULT_MD = "docs/replacement-loss-v19-confirmation-results.md"
CLAIM_CEILING = "LOCAL_CAUSAL_VALUE_PARETO_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT"
CONFIRM_CLAIM_CEILING = "LOCAL_CAUSAL_VALUE_PARETO_CONFIRMATION_ON_ONE_REUSED_OPT_CHECKPOINT"
TARGET_LAYER = 8
ADJACENT_LAYER = 9
LOSS_GRID = (0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
TRAIN_UPDATES = 1000
TRAIN_BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
MAX_WALL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
INFERENCE_BATCH_SIZE = 4
RESEARCH_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
DEV_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v19-development-primary"
DEV_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v19-development-replica"
CONFIRM_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v19-confirmation-primary"
CONFIRM_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v19-confirmation-replica"
PRIMARY_PYTHON = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v15-runtimes/primary/bin/python"
REPLICA_PYTHON = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v15-runtimes/replica/bin/python"
REPO_ROOT = Path(__file__).resolve().parent
V18_FREEZE_FILE = "docs/replacement-loss-v18-freeze.json"
V18_RESULT_FILE = "docs/replacement-loss-v18-results.md"
V18_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v18-projected-causal-primary"
V18_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v18-projected-causal-replica"
V18_PROTOCOL_FILE = "docs/replacement-loss-v18-projected-causal-refinement.md"
V18_FIXTURE_FILE = "replacement_loss_v18_fixtures.py"
V18_RUNNER_FILE = "replacement_loss_v18_runner.py"
V18_LOCK_FILE = "requirements-v18.lock.txt"
V18_PACKET_FILES = (
    "AGENTS.md", V18_PROTOCOL_FILE, V18_FIXTURE_FILE, V18_RUNNER_FILE, V18_LOCK_FILE,
    "docs/replacement-loss-v17-freeze.json", "docs/replacement-loss-v17-results.md",
    "docs/replacement-loss-v17-causal-supervision.md", "replacement_loss_v17_fixtures.py",
    "replacement_loss_v17_runner.py", "requirements-v17.lock.txt",
    "docs/replacement-loss-v16-freeze.json", "docs/replacement-loss-v16-results.md",
    "docs/replacement-loss-v16-whole-mlp.md", "replacement_loss_v16_fixtures.py",
    "replacement_loss_v16_runner.py", "requirements-v16.lock.txt", "replacement_loss_v3_fixtures.py",
)
DEV_PACKET_FILES = (
    "AGENTS.md", "docs/replacement-loss-v19-development.md", "docs/replacement-loss-v19-confirmatory.md",
    "replacement_loss_v19_fixtures.py", "replacement_loss_v19_runner.py", "requirements-v19.lock.txt",
    V18_FREEZE_FILE, V18_RESULT_FILE, V18_PROTOCOL_FILE, V18_FIXTURE_FILE, V18_RUNNER_FILE, V18_LOCK_FILE,
)
CONFIRM_PACKET_FILES = (
    "AGENTS.md", "docs/replacement-loss-v19-development.md", "docs/replacement-loss-v19-confirmatory.md",
    "replacement_loss_v19_fixtures.py", "replacement_loss_v19_runner.py", "requirements-v19.lock.txt",
    DEV_FREEZE_FILE, CONFIRM_CONFIG_FILE, DEV_RESULT_JSON, DEV_RESULT_MD, DEV_REPLICATION_JSON,
    V18_FREEZE_FILE, V18_RESULT_FILE,
)

np: Any = None
torch: Any = None
F: Any = None
v16: Any = None
v17: Any = None
v18: Any = None
_DEPENDENCIES_LOADED = False


def _load_dependencies() -> None:
    global np, torch, F, v16, v17, v18, _DEPENDENCIES_LOADED
    if _DEPENDENCIES_LOADED:
        return
    import numpy as numpy_module
    import torch as torch_module
    import torch.nn.functional as functional_module
    import replacement_loss_v16_runner as v16_module
    import replacement_loss_v17_runner as v17_module
    import replacement_loss_v18_runner as v18_module

    np, torch, F = numpy_module, torch_module, functional_module
    v16, v17, v18 = v16_module, v17_module, v18_module
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _DEPENDENCIES_LOADED = True


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
            raise FileExistsError(f"immutable artifact already exists with different bytes: {path.name}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _verify_frozen_file(relative: str) -> dict[str, Any]:
    frozen = _read_json(REPO_ROOT / relative)
    rows = frozen.get("packet_files")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"frozen packet has no source rows: {relative}")
    current = []
    for row in rows:
        path = REPO_ROOT / row["path"]
        if not path.is_file():
            raise FileNotFoundError(f"frozen source is missing: {row['path']}")
        current.append({"path": row["path"], "sha256": _sha_file(path), "bytes": path.stat().st_size})
    if rows != current or frozen.get("packet_sha256") != _digest(current):
        raise ValueError(f"frozen packet source bytes differ: {relative}")
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if frozen.get("freeze_sha256") != _digest(material):
        raise ValueError(f"freeze digest is invalid: {relative}")
    return frozen


def _rows_for(files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in files:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"packet source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def _verify_v18_parent() -> dict[str, Any]:
    global v18
    if v18 is None:
        import replacement_loss_v18_runner as v18_contract
        v18 = v18_contract
    parent = _verify_frozen_file(V18_FREEZE_FILE)
    parent_result = parent.get("parent_protocol", {})
    if parent.get("protocol_id") != "theseus-replacement-loss-v18-projected-causal-refinement":
        raise ValueError("V19 parent protocol identity differs from V18")
    if parent.get("packet_files") != _rows_for(V18_PACKET_FILES) or parent.get("packet_sha256") != _digest(_rows_for(V18_PACKET_FILES)):
        raise ValueError("V18 parent sources differ from the declared V18 packet files")
    result = REPO_ROOT / V18_RESULT_FILE
    result_text = result.read_text(encoding="utf-8") if result.is_file() else ""
    if (
        parent_result.get("scientific_outcome") != "FAIL"
        or parent_result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or parent.get("packet_sha256") not in result_text
        or "Scientific outcome: **`FAIL`**" not in result_text
        or "`REPRODUCED_LOCAL`" not in result_text
    ):
        raise ValueError("V18 frozen aggregate result does not support the declared post-result follow-up")
    root = RESEARCH_ROOT / V18_PRIMARY_NAME
    outcome = _read_json(root / "outcome-report.json")
    replay = _read_json(root / "replication-report.json")
    outcome_digest = _digest({key: value for key, value in outcome.items() if key != "report_sha256"})
    replay_digest = _digest({key: value for key, value in replay.items() if key != "report_sha256"})
    if (
        outcome.get("protocol_id") != parent["protocol_id"]
        or outcome.get("packet_sha256") != parent["packet_sha256"]
        or outcome.get("scientific_outcome") != "FAIL"
        or outcome.get("claim_ceiling") != "LOCAL_PROJECTED_CAUSAL_REFINEMENT_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT"
        or outcome.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or outcome.get("reviewer_status") != "NOT_SUBMITTED"
        or outcome_digest != outcome.get("report_sha256")
        or replay.get("protocol_id") != parent["protocol_id"]
        or replay.get("packet_sha256") != parent["packet_sha256"]
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_outcome") != "FAIL"
        or replay.get("reviewer_status") != "NOT_SUBMITTED"
        or not all(replay.get(field) is True for field in ("runtime_match", "separate_python_runtime", "same_host", "same_owner", "scientific_results_match"))
        or replay_digest != replay.get("report_sha256")
        or not replay.get("artifact_groups")
        or not all(group.get("byte_identical") for group in replay["artifact_groups"].values())
        or outcome.get("report_sha256") not in result_text
        or replay.get("report_sha256") not in result_text
        or outcome.get("scientific_results_sha256") not in result_text
    ):
        raise ValueError("V18 recorded outcome/replay receipts fail their bound integrity checks")
    return parent


def freeze_development() -> dict[str, Any]:
    path = REPO_ROOT / DEV_FREEZE_FILE
    if path.exists():
        return verify_development_packet()
    parent = _verify_v18_parent()
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
            "scientific_outcome": "FAIL",
            "reproducibility_status": "REPRODUCED_LOCAL",
            "v18_challenge_data_used_by_v19": False,
            "post_result_follow_up": True,
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
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_development_packet() -> dict[str, Any]:
    parent = _verify_v18_parent()
    frozen = _read_json(REPO_ROOT / DEV_FREEZE_FILE)
    current = _rows_for(DEV_PACKET_FILES)
    expected_parent = {
        "protocol_id": parent["protocol_id"], "packet_sha256": parent["packet_sha256"],
        "scientific_outcome": "FAIL", "reproducibility_status": "REPRODUCED_LOCAL",
        "v18_challenge_data_used_by_v19": False, "post_result_follow_up": True,
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != DEV_PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING
        or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current)
        or frozen.get("parent_protocol") != expected_parent
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V19 development packet identity or source bytes differ from its freeze")
    return frozen


def freeze_confirmation() -> dict[str, Any]:
    development = verify_development_packet()
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    selected = result.get("selected_lambda")
    if (
        result.get("qualification_status") != "QUALIFIED"
        or selected not in LOSS_GRID
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("selected_lambda") != selected
        or replay.get("scientific_results_sha256") != result.get("scientific_results_sha256")
    ):
        raise ValueError("confirmation requires one qualified global lambda and an exact local replay")
    config = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "development_packet_sha256": development["packet_sha256"],
        "development_freeze_sha256": development["freeze_sha256"],
        "development_result_sha256": result["scientific_results_sha256"],
        "development_replication_sha256": replay["report_sha256"],
        "selected_lambda": selected,
        "seeds": list(fixtures.CONFIRMATION_SEEDS),
        "challenge_unlock_condition": "all nine confirmation adaptation cells pass every frozen prechallenge gate",
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
        "claim_ceiling": CONFIRM_CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "development_binding": {
            "packet_sha256": development["packet_sha256"],
            "result_sha256": result["scientific_results_sha256"],
            "replication_sha256": replay["report_sha256"],
            "selected_lambda": selected,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_confirmation_packet() -> dict[str, Any]:
    development = verify_development_packet()
    config = _read_json(REPO_ROOT / CONFIRM_CONFIG_FILE)
    config_material = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != _digest(config_material):
        raise ValueError("V19 confirmation config digest is invalid")
    frozen = _read_json(REPO_ROOT / CONFIRM_FREEZE_FILE)
    current = _rows_for(CONFIRM_PACKET_FILES)
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    if (
        result.get("qualification_status") != "QUALIFIED"
        or result.get("selected_lambda") != config.get("selected_lambda")
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
    ):
        raise ValueError("confirmation packet no longer has its required reproduced development entry state")
    expected_binding = {
        "packet_sha256": development["packet_sha256"],
        "result_sha256": result["scientific_results_sha256"],
        "replication_sha256": replay["report_sha256"],
        "selected_lambda": config["selected_lambda"],
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != CONFIRM_PROTOCOL_ID
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE
        or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CONFIRM_CLAIM_CEILING
        or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current)
        or frozen.get("development_binding") != expected_binding
        or config.get("development_packet_sha256") != development["packet_sha256"]
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V19 confirmation packet identity or source bytes differ from its freeze")
    return frozen


def _freeze_for_stage(stage: str) -> tuple[str, str, tuple[str, ...], dict[str, Any]]:
    if stage == "development":
        return DEV_PROTOCOL_ID, DEV_FREEZE_FILE, DEV_PACKET_FILES, verify_development_packet()
    if stage == "confirmation":
        return CONFIRM_PROTOCOL_ID, CONFIRM_FREEZE_FILE, CONFIRM_PACKET_FILES, verify_confirmation_packet()
    raise ValueError("stage must be development or confirmation")


def _custody_root(value: str | Path, stage: str) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_symlink():
        raise ValueError("custody root cannot be a symlink")
    root = candidate.resolve()
    expected = {
        "development": {DEV_PRIMARY_NAME, DEV_REPLICA_NAME},
        "confirmation": {CONFIRM_PRIMARY_NAME, CONFIRM_REPLICA_NAME},
    }[stage]
    if root.parent != RESEARCH_ROOT or root.name not in expected or root.is_symlink():
        raise ValueError(f"custody root must be one of the frozen V19 {stage} roots on PrimaryED")
    return root


def _archive_packet(root: Path, stage: str) -> dict[str, Any]:
    protocol_id, freeze_file, _files, packet = _freeze_for_stage(stage)
    archive = root / "frozen-packet"
    if root.exists():
        raise FileExistsError(f"V19 {stage} custody root already exists; preserve it and choose a fresh identity")
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    for row in packet["packet_files"]:
        source = REPO_ROOT / row["path"]
        target = archive / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append(dict(row))
    source = REPO_ROOT / freeze_file
    target = archive / freeze_file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    rows.append({"path": freeze_file, "sha256": _sha_file(target), "bytes": target.stat().st_size})
    manifest = {
        "protocol_id": protocol_id,
        "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"],
        "files": rows,
        "archive_sha256": _digest(rows),
    }
    _write_json(root / "frozen-packet-manifest.json", manifest, immutable=True)
    return {"packet_sha256": packet["packet_sha256"], "archive_sha256": manifest["archive_sha256"]}


def _verify_external_archive(root: Path, stage: str) -> dict[str, Any]:
    protocol_id, freeze_file, _files, packet = _freeze_for_stage(stage)
    archive = root / "frozen-packet"
    manifest = _read_json(root / "frozen-packet-manifest.json")
    if manifest.get("packet_sha256") != packet["packet_sha256"] or manifest.get("protocol_id") != protocol_id:
        raise ValueError(f"external {stage} archive binds a different frozen packet")
    expected = []
    for row in packet["packet_files"]:
        path = archive / row["path"]
        actual = {"path": row["path"], "sha256": _sha_file(path), "bytes": path.stat().st_size} if path.is_file() else None
        if actual != row:
            raise ValueError(f"external {stage} packet source differs: {row['path']}")
        expected.append(actual)
    freeze_copy = archive / freeze_file
    if not freeze_copy.is_file() or _sha_file(freeze_copy) != _sha_file(REPO_ROOT / freeze_file):
        raise ValueError(f"external {stage} archive freeze bytes differ")
    expected.append({"path": freeze_file, "sha256": _sha_file(freeze_copy), "bytes": freeze_copy.stat().st_size})
    if manifest.get("files") != expected or manifest.get("archive_sha256") != _digest(expected):
        raise ValueError(f"external {stage} packet archive manifest is invalid")
    return manifest


def _ensure_root(value: str | Path, stage: str) -> Path:
    root = _custody_root(value, stage)
    if not root.is_dir():
        raise FileNotFoundError(f"V19 {stage} external custody root is missing")
    if (root / "run-stop.json").exists():
        raise RuntimeError(f"V19 {stage} root has a terminal stop receipt; do not reuse it")
    _verify_external_archive(root, stage)
    return root


def _runtime_receipt() -> dict[str, Any]:
    _load_dependencies()
    receipt = v16._runtime_receipt()
    v19_lock = _parse_lock(REPO_ROOT / "requirements-v19.lock.txt")
    v16_lock = _parse_lock(REPO_ROOT / "requirements-v16.lock.txt")
    if v19_lock != v16_lock:
        raise ValueError("V19 dependency inventory differs from reproduced V16/V17/V18 runtime pins")
    receipt["dependency_lock_sha256"] = _sha_file(REPO_ROOT / "requirements-v19.lock.txt")
    receipt["deterministic_algorithms"] = bool(torch.are_deterministic_algorithms_enabled())
    receipt["torch_num_threads"] = int(torch.get_num_threads())
    if receipt["python"] != "3.12.13" or not receipt["deterministic_algorithms"] or receipt["torch_num_threads"] != 1:
        raise ValueError("V19 runtime differs from the frozen Python/Torch determinism contract")
    return receipt


def _parse_lock(path: Path) -> dict[str, str]:
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if not separator:
            raise ValueError(f"invalid exact pin in {path.name}")
        pins[name.casefold().replace("_", "-")] = version
    return pins


def _check_caps(root: Path, started: float) -> None:
    if time.monotonic() - started > MAX_WALL_SECONDS:
        raise TimeoutError("V19 phase exceeded the frozen four-hour wall-clock cap")
    if v16._max_rss_bytes() > MAX_RSS_BYTES:
        raise MemoryError("V19 process exceeded the frozen 16-GiB RSS cap")


def _reference_payloads() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _load_dependencies()
    v18.verify_packet()
    v18._parent_binding()
    v18._external_packet_archive(RESEARCH_ROOT / V18_PRIMARY_NAME)
    visible, old_adaptations = v18._load_reference_fixtures()
    v18_root = RESEARCH_ROOT / V18_PRIMARY_NAME
    v18_adaptations = [
        v18._fixture_payload(
            v18_root,
            v18_root / "fixtures" / f"adaptation-seed-{index}.json",
            f"adaptation-seed-{index}",
            v18.fixtures.ADAPTATION_PER_STRATUM * len(v18.fixtures.STRATA),
        )
        for index in range(3)
    ]
    return visible, old_adaptations + v18_adaptations


def _split(records: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed ^ 0x19D15A)
    fit: list[int] = []
    validation: list[int] = []
    for stratum in fixtures.STRATA:
        indices = np.asarray([i for i, row in enumerate(records) if row["stratum"] == stratum], dtype=np.int32)
        if len(indices) != fixtures.ADAPTATION_PER_STRATUM:
            raise ValueError("V19 adaptation fixture count differs from the frozen strata")
        rng.shuffle(indices)
        fit.extend(int(index) for index in indices[: fixtures.FIT_PER_STRATUM])
        validation.extend(int(index) for index in indices[fixtures.FIT_PER_STRATUM :])
    return {"fit_prompt_indices": fit, "validation_prompt_indices": validation}


def _training_set(layer: Any, captures: dict[str, Any], records: list[dict[str, Any]], prompt_set: set[int], *, seed: int, label: str) -> dict[str, Any]:
    prompt_ids = captures["prompt_index"]
    mask = np.isin(prompt_ids, np.asarray(sorted(prompt_set), dtype=np.int32))
    data = v17._training_set(layer, captures, records, prompt_set, seed=seed, label=label)
    data["prompt_index"] = prompt_ids[mask].copy()
    return data


def _train(data: dict[str, Any], validation: dict[str, Any], *, seed: int, causal_weight: float, root: Path, started: float) -> tuple[Any, Any, dict[str, Any]]:
    x, y = torch.from_numpy(data["x"]), torch.from_numpy(data["y"])
    delta, effect = torch.from_numpy(data["delta"]), torch.from_numpy(data["effect"])
    xv, yv = torch.from_numpy(validation["x"]), torch.from_numpy(validation["y"])
    deltav, effectv = torch.from_numpy(validation["delta"]), torch.from_numpy(validation["effect"])
    candidate = v16._new_candidate(seed)
    random_control = copy.deepcopy(candidate)
    initial_sha = v16._module_state_digest(candidate)
    value_norm = max(float(y.square().mean().item()), 1e-12)
    effect_norm = max(float(effect.square().mean().item()), 1e-12)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) ^ 0x17A5C3)
    draws = np.empty((TRAIN_UPDATES, TRAIN_BATCH_SIZE), dtype=np.int32)
    candidate.train()
    for update in range(TRAIN_UPDATES):
        indices = torch.randint(0, x.shape[0], (TRAIN_BATCH_SIZE,), generator=generator)
        draws[update] = indices.numpy()
        x_batch, y_batch = x.index_select(0, indices), y.index_select(0, indices)
        delta_batch, effect_batch = delta.index_select(0, indices), effect.index_select(0, indices)
        predicted = candidate(x_batch)
        value_loss = F.mse_loss(predicted, y_batch) / value_norm
        if causal_weight:
            predicted_effect = candidate(x_batch + delta_batch) - predicted
            causal_loss = F.mse_loss(predicted_effect, effect_batch) / effect_norm
            loss = (1.0 - causal_weight) * value_loss + causal_weight * causal_loss
        else:
            loss = value_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite V19 replacement training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(candidate.parameters(), GRADIENT_CLIP)
        optimizer.step()
        if (update + 1) % 100 == 0:
            _check_caps(root, started)
    candidate.eval()
    random_control.eval()
    with torch.inference_mode():
        fit_value = float(F.mse_loss(candidate(x), y).item() / value_norm)
        fit_effect = float(F.mse_loss(candidate(x + delta) - candidate(x), effect).item() / effect_norm)
        val_value = float(F.mse_loss(candidate(xv), yv).item() / value_norm)
        val_effect = float(F.mse_loss(candidate(xv + deltav) - candidate(xv), effectv).item() / effect_norm)
    receipt = {
        "initialization_sha256": initial_sha,
        "candidate_state_sha256": v16._module_state_digest(candidate),
        "fit_token_rows": int(len(x)), "validation_token_rows": int(len(xv)),
        "updates": TRAIN_UPDATES, "batch_size": TRAIN_BATCH_SIZE,
        "batch_draws_sha256": hashlib.sha256(draws.astype("<i4", copy=False).tobytes()).hexdigest(),
        "optimizer": "AdamW", "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
        "gradient_clip_norm": GRADIENT_CLIP, "causal_loss_weight": causal_weight,
        "value_normalizer": value_norm, "effect_normalizer": effect_norm,
        "normalized_fit_value_mse": fit_value, "normalized_fit_effect_mse": fit_effect,
        "normalized_validation_value_mse": val_value, "normalized_validation_effect_mse": val_effect,
        "source_parameters_copied": False,
        "fit_input_sha256": hashlib.sha256(data["x"].astype("<f4", copy=False).tobytes()).hexdigest(),
        "fit_effect_target_sha256": hashlib.sha256(data["effect"].astype("<f4", copy=False).tobytes()).hexdigest(),
    }
    return candidate, random_control, receipt


def _save_npz(path: Path, **arrays: Any) -> None:
    v16._save_npz_deterministic(path, **arrays)


def _artifact_rows(root: Path, relatives: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(relatives):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V19 preparation artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _prompt_metrics(candidate: Any, data: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    x = torch.from_numpy(data["x"])
    y = torch.from_numpy(data["y"])
    delta = torch.from_numpy(data["delta"])
    effect = torch.from_numpy(data["effect"])
    with torch.inference_mode():
        predicted = candidate(x)
        predicted_effect = candidate(x + delta) - predicted
    value_rows = (predicted - y).square().mean(dim=1).cpu().numpy()
    effect_rows = (predicted_effect - effect).square().mean(dim=1).cpu().numpy()
    no_effect_rows = effect.square().mean(dim=1).cpu().numpy()
    grouped: dict[str, list[tuple[float, float, float]]] = {stratum: [] for stratum in fixtures.STRATA}
    for prompt_index in sorted(set(int(value) for value in data["prompt_index"])):
        row_mask = data["prompt_index"] == prompt_index
        stratum = records[prompt_index]["stratum"]
        grouped[stratum].append((float(value_rows[row_mask].mean()), float(effect_rows[row_mask].mean()), float(no_effect_rows[row_mask].mean())))
    result = {}
    for stratum, prompts in grouped.items():
        values = np.asarray(prompts, dtype=np.float64)
        if len(values) != fixtures.VALIDATION_PER_STRATUM:
            raise ValueError("V19 validation prompt count differs from frozen stratum size")
        result[stratum] = {
            "validation_value_mse": float(values[:, 0].mean()),
            "validation_effect_mse": float(values[:, 1].mean()),
            "zero_effect_validation_mse": float(values[:, 2].mean()),
            "effect_mse_ratio_zero": float(values[:, 1].mean() / max(values[:, 2].mean(), 1e-30)),
        }
    return result


def _behavior_metrics(model: Any, tokenizer: Any, candidate: Any, records: list[dict[str, Any]], validation_indices: list[int]) -> dict[str, dict[str, float]]:
    encoded = [v16._encode(tokenizer, records[index], with_continuation=True) for index in validation_indices]
    rows: dict[str, list[tuple[float, float]]] = {stratum: [] for stratum in fixtures.STRATA}
    for start in range(0, len(encoded), INFERENCE_BATCH_SIZE):
        group = encoded[start : start + INFERENCE_BATCH_SIZE]
        positions = [int(row["prompt_len"]) - 1 for row in group]
        base = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=None, positions=positions)
        replaced = v16._batch_forward(model, tokenizer, TARGET_LAYER, group, candidate=candidate, positions=positions)
        source_records = [records[index] for index in validation_indices[start : start + len(group)]]
        for index, (row, teacher_logit, candidate_logit, teacher_ce, candidate_ce) in enumerate(zip(
            source_records, base["boundary_logits"], replaced["boundary_logits"],
            base["continuation_ce"], replaced["continuation_ce"], strict=True,
        )):
            error = float(np.linalg.norm(candidate_logit.astype(np.float64) - teacher_logit.astype(np.float64)) / max(np.linalg.norm(teacher_logit), 1e-12))
            rows[row["stratum"]].append((float(candidate_ce - teacher_ce), error))
    return {
        stratum: {
            "validation_cross_entropy_increase": float(np.mean([row[0] for row in values])),
            "median_normalized_next_logit_error": float(np.median([row[1] for row in values])),
        }
        for stratum, values in rows.items()
    }


def _candidate_cell_metrics(model: Any, tokenizer: Any, candidate: Any, validation_data: dict[str, Any], records: list[dict[str, Any]], validation_indices: list[int]) -> dict[str, dict[str, float]]:
    metrics = _prompt_metrics(candidate, validation_data, records)
    behavior = _behavior_metrics(model, tokenizer, candidate, records, validation_indices)
    return {stratum: {**metrics[stratum], **behavior[stratum]} for stratum in fixtures.STRATA}


def _development_root(custody: str | Path, *, replica: bool = False) -> Path:
    root = _custody_root(custody, "development")
    expected = DEV_REPLICA_NAME if replica else DEV_PRIMARY_NAME
    if root.name != expected:
        raise ValueError("development custody role differs from the frozen primary/replica identity")
    return root


def _run_development(custody: str | Path, *, replica: bool = False) -> dict[str, Any]:
    packet = verify_development_packet()
    root = _development_root(custody, replica=replica)
    if not root.exists():
        _archive_packet(root, "development")
    else:
        _ensure_root(root, "development")
    if (root / "development-selection.json").exists():
        raise FileExistsError("V19 development outcome already exists; preserve this root and do not rerun")
    _load_dependencies()
    started = time.monotonic()
    try:
        runtime = _runtime_receipt()
        tree = v16._target_tree_receipt()
        if tree["tree_sha256"] != v16.TARGET_TREE_SHA256 or tree["weight_sha256"] != v16.TARGET_WEIGHT_SHA256:
            raise ValueError("V19 model checkpoint differs from the frozen OPT identity")
        visible, old_adaptations = _reference_payloads()
        fixtures.activate_stage("development")
        adaptations = []
        splits = []
        for seed_index, seed in enumerate(fixtures.DEVELOPMENT_SEEDS):
            rows = fixtures.generate_adaptation_fixture("development", seed_index)
            payload = fixtures.fixture_payload(rows, fixture_kind=f"adaptation-seed-{seed_index}", stage="development")
            adaptations.append(payload)
            splits.append(_split(payload["records"], seed))
        fixtures.set_extra_disjoint_payloads(old_adaptations)
        fixtures.validate_partition_disjointness(visible, *adaptations)
        model, tokenizer = v16._load_model()
        layers = v16._layers(model)
        v16_root = RESEARCH_ROOT / v16.PRIMARY_NAME
        visible_payload = _read_json(v16_root / "fixtures" / "visible.json")
        selection_path = v16_root / "block-selection.json"
        selection = _read_json(selection_path)
        if selection.get("target_layer") != TARGET_LAYER or selection.get("visible_fixture_sha256") != visible_payload.get("example_set_sha256"):
            raise ValueError("V19 block 8 differs from the frozen V16 visible-only selection")
        _write_json(root / "fixtures" / "v16-visible-reference.json", {
            "protocol_id": DEV_PROTOCOL_ID,
            "source_protocol_id": v16.PROTOCOL_ID,
            "source_example_set_sha256": visible_payload["example_set_sha256"],
            "source_count": visible_payload["count"],
            "source_selection_sha256": _sha_file(selection_path),
        }, immutable=True)
        for seed_index, (payload, split) in enumerate(zip(adaptations, splits, strict=True)):
            _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
            _write_json(root / "splits" / f"seed-{seed_index}.json", split, immutable=True)

        all_candidates = []
        all_metrics = []
        for seed_index, (payload, split) in enumerate(zip(adaptations, splits, strict=True)):
            _check_caps(root, started)
            records = payload["records"]
            captures = v16._capture_adaptation(model, tokenizer, records, TARGET_LAYER, ADJACENT_LAYER)
            _save_npz(root / "adaptation" / f"seed-{seed_index}.npz", **captures)
            fit_prompts = set(split["fit_prompt_indices"])
            val_prompts = set(split["validation_prompt_indices"])
            train_seed = fixtures.DEVELOPMENT_SEEDS[seed_index] ^ 0x19B10C
            fit = _training_set(layers[TARGET_LAYER], captures, records, fit_prompts, seed=train_seed, label="target")
            validation = _training_set(layers[TARGET_LAYER], captures, records, val_prompts, seed=train_seed ^ 0x55A, label="target")
            validation_indices = split["validation_prompt_indices"]
            seed_candidate_results = {}
            for loss_weight in (0.0, *LOSS_GRID):
                candidate, random_control, receipt = _train(fit, validation, seed=train_seed, causal_weight=loss_weight, root=root, started=started)
                method = "lambda-0.00" if loss_weight == 0.0 else f"lambda-{loss_weight:.2f}"
                row = v17._save_candidate(root / "candidates" / f"seed-{seed_index}-{method}.safetensors", candidate)
                cells = _candidate_cell_metrics(model, tokenizer, candidate, validation, records, validation_indices)
                item = {
                    "seed_index": seed_index, "method": method, "loss_weight": loss_weight,
                    **row, "fit_receipt": receipt, "validation_cells": cells,
                }
                all_candidates.append(item)
                seed_candidate_results[loss_weight] = cells
                all_metrics.append({"seed_index": seed_index, "loss_weight": loss_weight, "strata": cells})
                del candidate, random_control
            pointwise = seed_candidate_results[0.0]
            for candidate_row in [row for row in all_candidates if row["seed_index"] == seed_index]:
                cells = candidate_row["validation_cells"]
                for stratum in fixtures.STRATA:
                    cells[stratum]["value_mse_ratio_pointwise"] = cells[stratum]["validation_value_mse"] / max(pointwise[stratum]["validation_value_mse"], 1e-30)
            del captures, fit, validation, records

        artifacts = [
            "fixtures/v16-visible-reference.json",
            *(f"fixtures/adaptation-seed-{index}.json" for index in range(3)),
            *(f"splits/seed-{index}.json" for index in range(3)),
            *(f"adaptation/seed-{index}.npz" for index in range(3)),
        ]
        manifest = {
            "protocol_id": DEV_PROTOCOL_ID, "packet_sha256": packet["packet_sha256"],
            "target_tree_sha256": v16.TARGET_TREE_SHA256, "target_layer": TARGET_LAYER,
            "adjacent_control_layer": ADJACENT_LAYER, "preparation_artifacts": _artifact_rows(root, artifacts),
            "candidates": all_candidates,
            "training": {
                "architecture": "Linear(768,3072)-ReLU-Linear(3072,768)", "updates": TRAIN_UPDATES,
                "batch_size": TRAIN_BATCH_SIZE, "loss_weights": [0.0, *LOSS_GRID],
                "objective": "(1-lambda) normalized pointwise MSE + lambda normalized finite-effect MSE",
                "pointwise_control_objective": "normalized pointwise MSE only",
            },
            "runtime_versions": {key: runtime[key] for key in (
                "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256",
                "repository_head", "deterministic_algorithms", "torch_num_threads",
            )},
        }
        manifest["candidate_set_sha256"] = _digest(manifest)
        manifest["candidate_freeze_sha256"] = _digest(manifest)
        _write_json(root / "candidate-freeze.json", manifest, immutable=True)
        _write_json(root / "runtime.json", runtime, immutable=True)
        selection = _select_global_lambda(all_candidates)
        resources = {
            "wall_seconds": float(time.monotonic() - started),
            "peak_rss_bytes": int(v16._max_rss_bytes()),
            "wall_cap_seconds": MAX_WALL_SECONDS,
            "rss_cap_bytes": MAX_RSS_BYTES,
        }
        _write_json(root / "execution-resources.json", resources, immutable=True)
        if resources["wall_seconds"] > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
            raise TimeoutError("V19 development exceeded a frozen resource cap")
        outcome = {
            "protocol_id": DEV_PROTOCOL_ID, "state_slice": STATE_SLICE, "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY, "packet_sha256": packet["packet_sha256"],
            "candidate_set_sha256": manifest["candidate_set_sha256"],
            "qualification_status": "QUALIFIED" if selection["selected_lambda"] is not None else "NO_QUALIFIED_DEVELOPMENT_CANDIDATE",
            "selected_lambda": selection["selected_lambda"], "selection": selection,
            "claim_ceiling": CLAIM_CEILING, "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_YET_COMPARED",
            "custody": "prompts, captures, intervention targets, candidates, and row-level outcomes remain in external owner custody",
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
            "known_limitations": ["single reused OPT/ReLU checkpoint and preselected block 8", "post-result method selection on fresh adaptation data", "no held-out challenge opened at the development stage"],
            "next_blocked_authority_gate": "development replay before any confirmation packet; external blind replication and independent acceptance remain separate",
        }
        scientific = {key: value for key, value in outcome.items() if key not in {"assurance_overlay", "reviewer_status", "reproducibility_status"}}
        outcome["scientific_results_sha256"] = _digest(scientific)
        outcome["report_sha256"] = _digest(outcome)
        _write_json(root / "development-selection.json", outcome, immutable=True)
        _write_json(root / "development-report.json", {"per_seed_candidate_metrics": all_metrics, "candidate_count": len(all_candidates)}, immutable=True)
        return {key: outcome[key] for key in ("protocol_id", "qualification_status", "selected_lambda", "candidate_set_sha256", "scientific_results_sha256", "report_sha256")}
    except Exception as error:
        _write_json(root / "run-stop.json", {
            "protocol_id": DEV_PROTOCOL_ID, "phase": "development", "status": "INCONCLUSIVE",
            "error_type": type(error).__name__, "error": str(error), "packet_sha256": packet["packet_sha256"],
            "timestamp_epoch": time.time(),
        }, immutable=True)
        raise


def _select_global_lambda(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = []
    summaries = []
    for loss_weight in LOSS_GRID:
        rows = [row for row in candidates if row["loss_weight"] == loss_weight]
        cells = [
            (int(row["seed_index"]), stratum, row["validation_cells"][stratum])
            for row in rows for stratum in fixtures.STRATA
        ]
        gates = {
            "nine_seed_stratum_cells": len(cells) == 9,
            "effect_reduction_all_cells": bool(cells) and all(cell[2]["effect_mse_ratio_zero"] <= 0.90 for cell in cells),
            "value_retention_all_cells": bool(cells) and all(cell[2]["value_mse_ratio_pointwise"] <= 1.25 for cell in cells),
            "continuation_behavior_all_cells": bool(cells) and all(cell[2]["validation_cross_entropy_increase"] <= 0.05 for cell in cells),
            "next_logit_behavior_all_cells": bool(cells) and all(cell[2]["median_normalized_next_logit_error"] <= 0.05 for cell in cells),
        }
        mean_ratio = float(np.mean([cell[2]["effect_mse_ratio_zero"] for cell in cells])) if cells else None
        summary = {"loss_weight": loss_weight, "gates": gates, "equal_cell_mean_effect_mse_ratio": mean_ratio}
        summaries.append(summary)
        if all(gates.values()):
            eligible.append((mean_ratio, loss_weight))
    selected = min(eligible, key=lambda item: (item[0], item[1]))[1] if eligible else None
    return {
        "selection_rule": "one global weight must pass every predeclared gate in all nine seed-by-stratum cells; minimize equal-cell mean effect-MSE ratio; exact ties choose lower weight",
        "candidate_summaries": summaries,
        "selected_lambda": selected,
    }


def _file_group(root: Path, prefix_or_files: list[str]) -> list[dict[str, Any]]:
    rows = []
    for relative in sorted(prefix_or_files):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"replication artifact is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    return rows


def _relative_files(root: Path, prefix: str) -> list[str]:
    path = root / prefix
    return [item.relative_to(root).as_posix() for item in sorted(path.rglob("*")) if item.is_file()] if path.exists() else []


def _compare_development(primary: Path, replica: Path) -> dict[str, Any]:
    primary, replica = _ensure_root(primary, "development"), _ensure_root(replica, "development")
    p_report, r_report = _read_json(primary / "development-selection.json"), _read_json(replica / "development-selection.json")
    group_paths = {
        "fixtures": _relative_files(primary, "fixtures"),
        "splits": _relative_files(primary, "splits"),
        "captures": _relative_files(primary, "adaptation"),
        "candidates": _relative_files(primary, "candidates") + ["candidate-freeze.json"],
        "selection": ["development-selection.json", "development-report.json"],
    }
    groups = {}
    for name, paths in group_paths.items():
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
        left, right = _file_group(primary, paths), _file_group(replica, paths)
        groups[name] = {
            "primary_count": len(left), "replica_count": len(right),
            "primary_sha256": _digest(left), "replica_sha256": _digest(right),
            "byte_identical": left == right and paths == replica_paths,
        }
    left_runtime, right_runtime = _read_json(primary / "runtime.json"), _read_json(replica / "runtime.json")
    pin_fields = ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "deterministic_algorithms", "torch_num_threads")
    pins_match = all(left_runtime.get(field) == right_runtime.get(field) for field in pin_fields)
    separate = left_runtime.get("python_executable") != right_runtime.get("python_executable")
    same_host = left_runtime.get("host_id_sha256") == right_runtime.get("host_id_sha256")
    same_owner = left_runtime.get("uid") == right_runtime.get("uid")
    scientific_match = p_report.get("scientific_results_sha256") == r_report.get("scientific_results_sha256")
    p_resources = _read_json(primary / "execution-resources.json")
    r_resources = _read_json(replica / "execution-resources.json")
    resources_within_caps = all(
        row.get("wall_seconds", float("inf")) <= MAX_WALL_SECONDS
        and row.get("peak_rss_bytes", MAX_RSS_BYTES + 1) <= MAX_RSS_BYTES
        for row in (p_resources, r_resources)
    )
    reproduced = all(row["byte_identical"] for row in groups.values()) and pins_match and separate and same_host and same_owner and scientific_match and resources_within_caps
    report = {
        "protocol_id": DEV_PROTOCOL_ID, "state_slice": STATE_SLICE, "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": verify_development_packet()["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED", "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "qualification_status": p_report["qualification_status"] if scientific_match else "INCONCLUSIVE",
        "selected_lambda": p_report.get("selected_lambda") if scientific_match else None,
        "scientific_results_match": scientific_match,
        "scientific_results_sha256": p_report.get("scientific_results_sha256") if scientific_match else None,
        "artifact_groups": groups, "runtime_pins_match": pins_match,
        "separate_python_runtimes": separate, "same_host_and_owner": same_host and same_owner,
        "resource_caps_pass": resources_within_caps,
        "resource_measurements": {"primary": p_resources, "replica": r_resources},
        "claim_ceiling": CLAIM_CEILING,
        "custody": "all prompt-bearing, row-level, capture, target, and candidate artifacts remain in external owner custody",
        "next_blocked_authority_gate": "confirmation eligibility and then blind external replication and independent acceptance",
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "development-replication.json", report, immutable=True)
    p_report["reproducibility_status"] = report["assurance_level"]
    p_report["replication_report_sha256"] = report["report_sha256"]
    p_report.pop("report_sha256", None)
    p_report["report_sha256"] = _digest(p_report)
    _write_json(primary / "development-selection.json", p_report)
    return report


def _python_exec(path: Path, args: list[str], *, log_path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"pinned runtime executable is missing: {path}")
    process = subprocess.run([str(path), str(Path(__file__).resolve()), *args], check=False, capture_output=True, text=True)
    log_path.write_text(process.stdout + "\n--- stderr ---\n" + process.stderr, encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"pinned runtime process failed with exit code {process.returncode}; see its external-custody log")
    try:
        return json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("pinned runtime did not return a JSON receipt") from error


def _write_development_results(report: dict[str, Any], primary: Path) -> None:
    outcome = _read_json(primary / "development-selection.json")
    selection = outcome["selection"]
    payload = {
        "protocol_id": DEV_PROTOCOL_ID, "state_slice": STATE_SLICE, "protocol_slice": PROTOCOL_SLICE,
        "assurance_level": report["assurance_level"], "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": report["assurance_level"],
        "packet_sha256": outcome["packet_sha256"], "candidate_set_sha256": outcome["candidate_set_sha256"],
        "scientific_results_sha256": outcome["scientific_results_sha256"],
        "qualification_status": outcome["qualification_status"], "selected_lambda": outcome["selected_lambda"],
        "selection_summary": selection["candidate_summaries"],
        "resource_measurements": report["resource_measurements"],
        "replication_report_sha256": report["report_sha256"],
        "claim_ceiling": CLAIM_CEILING,
        "custody": "prompt-bearing and row-level artifacts remain on external PrimaryED custody",
        "limitations": outcome["known_limitations"],
        "next_blocked_authority_gate": outcome["next_blocked_authority_gate"],
    }
    payload["result_document_sha256"] = _digest(payload)
    _write_json(REPO_ROOT / DEV_RESULT_JSON, payload)
    rows = "\n".join(
        f"| {item['loss_weight']:.2f} | {item['equal_cell_mean_effect_mse_ratio']:.4f} | "
        f"{sum(item['gates'].values())}/5 | {all(item['gates'].values())} |"
        for item in selection["candidate_summaries"]
    )
    selected = "none" if outcome["selected_lambda"] is None else f"`{outcome['selected_lambda']:.2f}`"
    text = (
        "# Replacement-loss V19 development results\n\n"
        "State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v19-causal-fit-qualification-v1`, and `global-experiment-assurance-lifecycle-v1`.\n\n"
        f"- Outcome: `{outcome['qualification_status']}`; selected lambda: {selected}.\n"
        f"- Assurance: `{report['assurance_level']}`; reviewer: `NOT_SUBMITTED`; blind status: `NOT_BLIND_SAME_OWNER_REPLAY`.\n"
        f"- Packet SHA-256: `{outcome['packet_sha256']}`; scientific result SHA-256: `{outcome['scientific_results_sha256']}`.\n"
        f"- Claim ceiling: `{CLAIM_CEILING}`.\n"
        "- Custody: prompts, captures, intervention targets, candidate tensors, and row-level metrics remain in external owner custody.\n\n"
        "## Predeclared global selection screen\n\n"
        "A weight had to pass all five checks in every one of nine seed-by-stratum cells: effect-MSE ratio to same-row zero-effect baseline <=0.90, value-MSE ratio to paired pointwise <=1.25, continuation CE increase <=0.05, normalized next-logit error <=0.05, and complete cell coverage.\n\n"
        "| Lambda | Equal-cell mean effect ratio | Gates passed | Qualified |\n|---:|---:|---:|:---:|\n"
        f"{rows}\n\n"
        "This was development and global candidate selection on one reused OPT/ReLU checkpoint. It did not use or open V18 challenge data. A local replay is not independent acceptance.\n\n"
        f"Next gate: {outcome['next_blocked_authority_gate']}.\n"
    )
    (REPO_ROOT / DEV_RESULT_MD).write_text(text, encoding="utf-8")
    _write_json(REPO_ROOT / DEV_REPLICATION_JSON, report)


def run_development(primary_custody: str | Path, replica_custody: str | Path) -> dict[str, Any]:
    primary = _development_root(primary_custody)
    replica = _development_root(replica_custody, replica=True)
    if primary.exists() or replica.exists():
        raise FileExistsError("V19 development custody roots already exist; preserve them and do not reuse")
    primary_result = _run_development(primary)
    _python_exec(REPLICA_PYTHON, ["run-development-replica", "--custody", str(replica)], log_path=primary / "replica-runtime.log")
    report = _compare_development(primary, replica)
    _write_development_results(report, primary)
    outcome = _read_json(primary / "development-selection.json")
    return {
        "protocol_id": DEV_PROTOCOL_ID,
        "qualification_status": outcome["qualification_status"],
        "selected_lambda": outcome["selected_lambda"],
        "assurance_level": report["assurance_level"],
        "scientific_results_sha256": outcome["scientific_results_sha256"],
        "replication_report_sha256": report["report_sha256"],
        "packet_sha256": outcome["packet_sha256"],
    }


def _confirm_config() -> dict[str, Any]:
    config = _read_json(REPO_ROOT / CONFIRM_CONFIG_FILE)
    material = {key: value for key, value in config.items() if key != "config_sha256"}
    if config.get("config_sha256") != _digest(material):
        raise ValueError("confirmation config digest differs")
    if config.get("selected_lambda") not in LOSS_GRID:
        raise ValueError("confirmation config selected an undeclared loss weight")
    return config


def _patch_v17_for_confirmation() -> None:
    _load_dependencies()
    fixtures.activate_stage("confirmation")
    fixtures.PROTOCOL_ID = CONFIRM_PROTOCOL_ID
    fixtures.PROTOCOL_SLICE = PROTOCOL_SLICE
    fixtures.SEEDS = fixtures.CONFIRMATION_SEEDS
    fixtures.PER_STRATUM_COUNT = fixtures.ADAPTATION_PER_STRATUM
    fixtures.CHALLENGES_PER_STRATUM = 20
    fixtures.PROMPT_RULES_VERSION = "v19-causal-value-qualification-synthetic-v1"
    v17.STATE_SLICE = STATE_SLICE
    v17.PROTOCOL_SLICE = PROTOCOL_SLICE
    v17.PROTOCOL_ID = CONFIRM_PROTOCOL_ID
    v17.FREEZE_FILE = CONFIRM_FREEZE_FILE
    v17.CLAIM_CEILING = CONFIRM_CLAIM_CEILING
    v17.TARGET_LAYER = TARGET_LAYER
    v17.ADJACENT_LAYER = ADJACENT_LAYER
    v17.PRIMARY_NAME = CONFIRM_PRIMARY_NAME
    v17.REPLICA_NAME = CONFIRM_REPLICA_NAME
    v17.RESEARCH_ROOT = RESEARCH_ROOT
    v17.REPO_ROOT = REPO_ROOT
    v17.fixtures = fixtures
    v17.verify_packet = verify_confirmation_packet
    v17._custody_root = lambda value: _custody_root(value, "confirmation")
    v17._ensure_root = lambda value: _ensure_root(value, "confirmation")
    v17._external_packet_archive = lambda root: _verify_external_archive(root, "confirmation")
    v17._runtime_receipt = _runtime_receipt


def _run_confirmation_prepare(custody: str | Path, *, replica: bool = False) -> dict[str, Any]:
    packet = verify_confirmation_packet()
    root = _custody_root(custody, "confirmation")
    expected_name = CONFIRM_REPLICA_NAME if replica else CONFIRM_PRIMARY_NAME
    if root.name != expected_name:
        raise ValueError("confirmation custody role differs from its frozen primary/replica identity")
    if not root.exists():
        _archive_packet(root, "confirmation")
    else:
        _ensure_root(root, "confirmation")
    if (root / "candidate-freeze.json").exists():
        raise FileExistsError("V19 confirmation candidates already exist; preserve and do not retrain")
    _patch_v17_for_confirmation()
    runtime = _runtime_receipt()
    started = time.monotonic()
    visible, old_adaptations = _reference_payloads()
    # V16/V17/V18 adaptation partitions and V19 development fixtures are references; no parent challenge is loaded.
    dev_primary = RESEARCH_ROOT / DEV_PRIMARY_NAME
    dev_payloads = [
        _read_json(dev_primary / "fixtures" / f"adaptation-seed-{index}.json")
        for index in range(3)
    ]
    fixtures.set_extra_disjoint_payloads([*old_adaptations, *dev_payloads])
    adaptations, splits = [], []
    for seed_index, seed in enumerate(fixtures.CONFIRMATION_SEEDS):
        rows = fixtures.generate_adaptation_fixture("confirmation", seed_index)
        payload = fixtures.fixture_payload(rows, fixture_kind=f"adaptation-seed-{seed_index}", stage="confirmation")
        adaptations.append(payload)
        splits.append(_split(payload["records"], seed))
    fixtures.validate_partition_disjointness(visible, *dev_payloads, *adaptations)
    model, tokenizer = v16._load_model()
    layers = v16._layers(model)
    v16_root = RESEARCH_ROOT / v16.PRIMARY_NAME
    visible_v16 = _read_json(v16_root / "fixtures" / "visible.json")
    selection_path = v16_root / "block-selection.json"
    selection = _read_json(selection_path)
    if selection.get("target_layer") != TARGET_LAYER or selection.get("visible_fixture_sha256") != visible_v16.get("example_set_sha256"):
        raise ValueError("confirmation block differs from the V16 visible-only selection")
    _write_json(root / "fixtures" / "v16-visible-reference.json", {
        "protocol_id": CONFIRM_PROTOCOL_ID, "source_protocol_id": v16.PROTOCOL_ID,
        "source_example_set_sha256": visible_v16["example_set_sha256"], "source_count": visible_v16["count"],
        "source_selection_sha256": _sha_file(selection_path),
    }, immutable=True)
    for seed_index, (payload, split) in enumerate(zip(adaptations, splits, strict=True)):
        _write_json(root / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)
        _write_json(root / "splits" / f"seed-{seed_index}.json", split, immutable=True)
    candidates = []
    prechallenge = []
    selected_lambda = _confirm_config()["selected_lambda"]
    for seed_index, (payload, split) in enumerate(zip(adaptations, splits, strict=True)):
        _check_caps(root, started)
        records = payload["records"]
        captures = v16._capture_adaptation(model, tokenizer, records, TARGET_LAYER, ADJACENT_LAYER)
        _save_npz(root / "adaptation" / f"seed-{seed_index}.npz", **captures)
        fit_prompts, val_prompts = set(split["fit_prompt_indices"]), set(split["validation_prompt_indices"])
        train_seed = fixtures.CONFIRMATION_SEEDS[seed_index] ^ 0x19B10C
        fit = _training_set(layers[TARGET_LAYER], captures, records, fit_prompts, seed=train_seed, label="target")
        validation = _training_set(layers[TARGET_LAYER], captures, records, val_prompts, seed=train_seed ^ 0x55A, label="target")
        adjacent_fit = _training_set(layers[ADJACENT_LAYER], captures, records, fit_prompts, seed=train_seed ^ 0x9A1, label="adjacent")
        adjacent_val = _training_set(layers[ADJACENT_LAYER], captures, records, val_prompts, seed=train_seed ^ 0x9A1 ^ 0x55A, label="adjacent")
        validation_indices = split["validation_prompt_indices"]
        methods = []
        trained: dict[str, Any] = {}
        for method, weight, fit_data, val_data, method_seed in (
            ("causal", selected_lambda, fit, validation, train_seed),
            ("pointwise", 0.0, fit, validation, train_seed),
            ("wrong_layer", selected_lambda, adjacent_fit, adjacent_val, train_seed ^ 0x9A1),
        ):
            candidate, random_control, receipt = _train(fit_data, val_data, seed=method_seed, causal_weight=weight, root=root, started=started)
            trained[method] = (candidate, receipt)
            methods.append((method, candidate, receipt))
            if method == "causal":
                random_init = random_control
            else:
                del random_control
        if trained["causal"][1]["initialization_sha256"] != trained["pointwise"][1]["initialization_sha256"]:
            raise ValueError("confirmation causal and pointwise candidates did not share their frozen initialization")
        methods.append(("random", random_init, {"source_parameters_copied": False, "trained": False, "initialization_sha256": v16._module_state_digest(random_init)}))
        pointwise_cells = _candidate_cell_metrics(model, tokenizer, trained["pointwise"][0], validation, records, validation_indices)
        for method, candidate, receipt in methods:
            row = v17._save_candidate(root / "candidates" / f"seed-{seed_index}-{method}.safetensors", candidate)
            if method == "causal":
                cells = _candidate_cell_metrics(model, tokenizer, candidate, validation, records, validation_indices)
                for stratum in fixtures.STRATA:
                    cells[stratum]["value_mse_ratio_pointwise"] = cells[stratum]["validation_value_mse"] / max(pointwise_cells[stratum]["validation_value_mse"], 1e-30)
                    cells[stratum]["passes_prechallenge"] = (
                        cells[stratum]["effect_mse_ratio_zero"] <= 0.90
                        and cells[stratum]["value_mse_ratio_pointwise"] <= 1.25
                        and cells[stratum]["validation_cross_entropy_increase"] <= 0.05
                        and cells[stratum]["median_normalized_next_logit_error"] <= 0.05
                    )
                prechallenge.append({"seed_index": seed_index, "cells": cells})
                receipt = {**receipt, "validation_cells": cells}
            candidates.append({"seed_index": seed_index, "method": method, **row, "fit_receipt": receipt})
        del captures, fit, validation, adjacent_fit, adjacent_val, methods, trained, random_init
    relatives = [
        "fixtures/v16-visible-reference.json",
        *(f"fixtures/adaptation-seed-{index}.json" for index in range(3)),
        *(f"splits/seed-{index}.json" for index in range(3)),
        *(f"adaptation/seed-{index}.npz" for index in range(3)),
    ]
    material = {
        "protocol_id": CONFIRM_PROTOCOL_ID, "packet_sha256": packet["packet_sha256"],
        "target_tree_sha256": v16.TARGET_TREE_SHA256, "target_layer": TARGET_LAYER,
        "adjacent_control_layer": ADJACENT_LAYER, "preparation_artifacts": _artifact_rows(root, relatives),
        "candidates": candidates,
        "training": {"architecture": "Linear(768,3072)-ReLU-Linear(3072,768)", "updates": TRAIN_UPDATES,
                     "batch_size": TRAIN_BATCH_SIZE, "causal_loss_weight": selected_lambda,
                     "pointwise_control_objective": "normalized pointwise MSE only"},
        "runtime_versions": {key: runtime[key] for key in (
            "python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256",
            "repository_head", "deterministic_algorithms", "torch_num_threads",
        )},
    }
    material["candidate_set_sha256"] = _digest(material)
    material["candidate_freeze_sha256"] = _digest(material)
    _write_json(root / "candidate-freeze.json", material, immutable=True)
    _write_json(root / "runtime.json", runtime, immutable=True)
    _write_json(root / "preparation-resources.json", {
        "wall_seconds": float(time.monotonic() - started),
        "peak_rss_bytes": int(v16._max_rss_bytes()),
        "wall_cap_seconds": MAX_WALL_SECONDS,
        "rss_cap_bytes": MAX_RSS_BYTES,
    }, immutable=True)
    resources = _read_json(root / "preparation-resources.json")
    if resources["wall_seconds"] > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
        raise TimeoutError("V19 confirmation adaptation exceeded a frozen resource cap")
    all_pass = len(prechallenge) == 3 and all(cell["passes_prechallenge"] for row in prechallenge for cell in row["cells"].values())
    eligibility = {
        "protocol_id": CONFIRM_PROTOCOL_ID, "packet_sha256": packet["packet_sha256"],
        "candidate_set_sha256": material["candidate_set_sha256"], "selected_lambda": selected_lambda,
        "pre_challenge_eligible": all_pass,
        "status": "ELIGIBLE_TO_UNLOCK_CHALLENGE" if all_pass else "CONFIRMATION_STOPPED_BEFORE_CHALLENGE",
        "cell_results": prechallenge, "challenge_entropy_generated": False,
        "claim_ceiling": CONFIRM_CLAIM_CEILING,
    }
    eligibility["report_sha256"] = _digest(eligibility)
    _write_json(root / "eligibility-report.json", eligibility, immutable=True)
    return {"pre_challenge_eligible": all_pass, "status": eligibility["status"], "candidate_set_sha256": material["candidate_set_sha256"], "eligibility_report_sha256": eligibility["report_sha256"]}


def _set_reference_disjoint_for_predict(root: Path) -> None:
    _load_dependencies()
    _patch_v17_for_confirmation()
    _visible, old_adaptations = _reference_payloads()
    dev_primary = RESEARCH_ROOT / DEV_PRIMARY_NAME
    dev_payloads = [_read_json(dev_primary / "fixtures" / f"adaptation-seed-{i}.json") for i in range(3)]
    fixtures.set_extra_disjoint_payloads([*old_adaptations, *dev_payloads])


def _run_confirmation(primary: str | Path, *, replica: bool = False, entropy_from: str | Path | None = None) -> dict[str, Any]:
    root = _custody_root(primary, "confirmation")
    try:
        prepare_result = _run_confirmation_prepare(root, replica=replica)
    except Exception as error:
        if root.is_dir():
            _write_json(root / "run-stop.json", {
                "protocol_id": CONFIRM_PROTOCOL_ID, "phase": "prepare", "status": "INCONCLUSIVE",
                "error_type": type(error).__name__, "error": str(error),
                "packet_sha256": verify_confirmation_packet()["packet_sha256"],
                "timestamp_epoch": time.time(),
            }, immutable=True)
        raise
    if not prepare_result["pre_challenge_eligible"]:
        return prepare_result
    _set_reference_disjoint_for_predict(root)
    if replica:
        if entropy_from is None:
            raise ValueError("confirmation replay must reuse the primary's frozen challenge entropy")
        prediction = v17.predict(root, entropy_from=entropy_from)
    else:
        prediction = v17.predict(root)
    scored = v17.score(root)
    return {**prepare_result, "prediction_set_sha256": prediction["prediction_set_sha256"], "scientific_outcome": scored["scientific_outcome"], "scientific_results_sha256": scored["scientific_results_sha256"]}


def _compare_confirmation_prechallenge(primary: Path, replica: Path) -> dict[str, Any]:
    primary, replica = _ensure_root(primary, "confirmation"), _ensure_root(replica, "confirmation")
    left, right = _read_json(primary / "eligibility-report.json"), _read_json(replica / "eligibility-report.json")
    group_paths = {
        "fixtures": _relative_files(primary, "fixtures"), "splits": _relative_files(primary, "splits"),
        "captures": _relative_files(primary, "adaptation"),
        "candidates": _relative_files(primary, "candidates") + ["candidate-freeze.json"],
        "eligibility": ["eligibility-report.json"],
    }
    groups = {}
    for name, paths in group_paths.items():
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
        a, b = _file_group(primary, paths), _file_group(replica, paths)
        groups[name] = {"primary_sha256": _digest(a), "replica_sha256": _digest(b), "byte_identical": a == b and paths == replica_paths, "primary_count": len(a), "replica_count": len(b)}
    left_runtime, right_runtime = _read_json(primary / "runtime.json"), _read_json(replica / "runtime.json")
    left_resources = _read_json(primary / "preparation-resources.json")
    right_resources = _read_json(replica / "preparation-resources.json")
    pin_fields = ("python", "platform", "packages", "package_inventory_sha256", "dependency_lock_sha256", "deterministic_algorithms", "torch_num_threads")
    pins_match = all(left_runtime.get(key) == right_runtime.get(key) for key in pin_fields)
    separate = left_runtime.get("python_executable") != right_runtime.get("python_executable")
    same_host = left_runtime.get("host_id_sha256") == right_runtime.get("host_id_sha256")
    same_owner = left_runtime.get("uid") == right_runtime.get("uid")
    same_result = left.get("report_sha256") == right.get("report_sha256") and left.get("status") == right.get("status")
    resources_within_caps = all(
        row.get("wall_seconds", float("inf")) <= MAX_WALL_SECONDS
        and row.get("peak_rss_bytes", MAX_RSS_BYTES + 1) <= MAX_RSS_BYTES
        for row in (left_resources, right_resources)
    )
    reproduced = same_result and all(group["byte_identical"] for group in groups.values()) and pins_match and separate and same_host and same_owner and resources_within_caps
    report = {
        "protocol_id": CONFIRM_PROTOCOL_ID, "assurance_level": "REPRODUCED_LOCAL" if reproduced else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED", "blind_status": "NOT_BLIND_SAME_OWNER_REPLAY",
        "pre_challenge_status": left["status"] if same_result else "INCONCLUSIVE",
        "challenge_entropy_generated": False, "artifact_groups": groups,
        "runtime_pins_match": pins_match, "separate_python_runtimes": separate,
        "same_host_and_owner": same_host and same_owner,
        "resource_caps_pass": resources_within_caps,
        "resource_measurements": {"primary": left_resources, "replica": right_resources},
        "scientific_results_sha256": left.get("report_sha256") if same_result else None,
        "next_blocked_authority_gate": "held-out challenge remains locked because the prechallenge eligibility gate failed",
    }
    report["report_sha256"] = _digest(report)
    _write_json(primary / "confirmation-replication.json", report, immutable=True)
    return report


def _write_confirmation_results(primary: Path, replay: dict[str, Any] | None) -> None:
    if (primary / "outcome-report.json").is_file():
        outcome = _read_json(primary / "outcome-report.json")
        status = outcome["scientific_outcome"]
        scientific_digest = outcome["scientific_results_sha256"]
    else:
        eligibility = _read_json(primary / "eligibility-report.json")
        status = eligibility["status"]
        scientific_digest = eligibility["report_sha256"]
    payload = {
        "protocol_id": CONFIRM_PROTOCOL_ID, "state_slice": STATE_SLICE, "protocol_slice": PROTOCOL_SLICE,
        "assurance_level": (replay or {}).get("assurance_level", "INTERNAL_PASS"),
        "reviewer_status": "NOT_SUBMITTED", "reproducibility_status": (replay or {}).get("assurance_level", "NOT_YET_COMPARED"),
        "scientific_outcome": status, "scientific_results_sha256": scientific_digest,
        "packet_sha256": verify_confirmation_packet()["packet_sha256"],
        "selected_lambda": _confirm_config()["selected_lambda"],
        "replication_report_sha256": (replay or {}).get("report_sha256"),
        "claim_ceiling": CONFIRM_CLAIM_CEILING,
        "custody": "all prompt-bearing, row-level, candidate, entropy, and prediction artifacts remain in external owner custody",
        "next_blocked_authority_gate": "blind external replication and authorized independent review; production authorization remains separate",
    }
    payload["result_document_sha256"] = _digest(payload)
    _write_json(REPO_ROOT / CONFIRM_RESULT_JSON, payload)
    (REPO_ROOT / CONFIRM_RESULT_MD).write_text(
        "# Replacement-loss V19 confirmation results\n\n"
        f"- Outcome: `{status}`.\n- Assurance: `{payload['assurance_level']}`.\n"
        "- Reviewer status: `NOT_SUBMITTED`; blind status is same-owner local replay only.\n"
        f"- Packet SHA-256: `{payload['packet_sha256']}`.\n"
        f"- Selected development lambda: `{payload['selected_lambda']:.2f}`.\n"
        f"- Claim ceiling: `{CONFIRM_CLAIM_CEILING}`.\n"
        "- Prompt-bearing and row-level artifacts remain in external owner custody.\n\n"
        f"Next gate: {payload['next_blocked_authority_gate']}.\n",
        encoding="utf-8",
    )


def _run_end_to_end(args: argparse.Namespace) -> dict[str, Any]:
    verify_development_packet()
    dev = run_development(args.development_primary, args.development_replica)
    if dev["assurance_level"] != "REPRODUCED_LOCAL":
        return {**dev, "end_to_end_status": "STOPPED_DEVELOPMENT_REPLAY_MISMATCH", "confirmation_packet_created": False}
    if dev["qualification_status"] != "QUALIFIED":
        return {**dev, "end_to_end_status": "STOPPED_NO_QUALIFIED_DEVELOPMENT_CANDIDATE", "confirmation_packet_created": False}
    confirmation_packet = freeze_confirmation()
    primary = _custody_root(args.confirmation_primary, "confirmation")
    replica = _custody_root(args.confirmation_replica, "confirmation")
    if primary.exists() or replica.exists():
        raise FileExistsError("V19 confirmation custody roots already exist; preserve them and do not reuse")
    result = _run_confirmation(primary)
    if result["pre_challenge_eligible"]:
        replica_result = _python_exec(REPLICA_PYTHON, [
            "run-confirmation-replica", "--primary", str(primary), "--replica", str(replica),
        ], log_path=primary / "confirmation-replica-runtime.log")
        del replica_result
        _patch_v17_for_confirmation()
        _set_reference_disjoint_for_predict(primary)
        comparison = v17.compare_replication(primary, replica)
        comparison["claim_ceiling"] = CONFIRM_CLAIM_CEILING
        comparison["resource_measurements"] = {
            "primary": _read_json(primary / "preparation-resources.json"),
            "replica": _read_json(replica / "preparation-resources.json"),
        }
        comparison["resource_caps_pass"] = all(
            row["wall_seconds"] <= MAX_WALL_SECONDS and row["peak_rss_bytes"] <= MAX_RSS_BYTES
            for row in comparison["resource_measurements"].values()
        )
        if not comparison["resource_caps_pass"]:
            comparison["assurance_level"] = "INCONCLUSIVE"
        comparison["report_sha256"] = _digest({key: value for key, value in comparison.items() if key != "report_sha256"})
        _write_json(primary / "confirmation-replication.json", comparison)
    else:
        _python_exec(REPLICA_PYTHON, ["run-confirmation-replica-precheck", "--primary", str(primary), "--replica", str(replica)], log_path=primary / "confirmation-replica-runtime.log")
        comparison = _compare_confirmation_prechallenge(primary, replica)
    _write_confirmation_results(primary, comparison)
    return {
        **dev,
        "end_to_end_status": result["status"] if not result["pre_challenge_eligible"] else "CONFIRMATION_COMPLETED",
        "confirmation_packet_sha256": confirmation_packet["packet_sha256"],
        "confirmation_prechallenge_eligible": result["pre_challenge_eligible"],
        "confirmation_scientific_outcome": result.get("scientific_outcome"),
        "confirmation_assurance_level": comparison.get("assurance_level"),
        "confirmation_replication_report_sha256": comparison.get("report_sha256"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze-development", "verify-development", "freeze-confirmation", "verify-confirmation"):
        commands.add_parser(name)
    e2e = commands.add_parser("run-e2e")
    e2e.add_argument("--development-primary", default=str(RESEARCH_ROOT / DEV_PRIMARY_NAME))
    e2e.add_argument("--development-replica", default=str(RESEARCH_ROOT / DEV_REPLICA_NAME))
    e2e.add_argument("--confirmation-primary", default=str(RESEARCH_ROOT / CONFIRM_PRIMARY_NAME))
    e2e.add_argument("--confirmation-replica", default=str(RESEARCH_ROOT / CONFIRM_REPLICA_NAME))
    dev_replica = commands.add_parser("run-development-replica")
    dev_replica.add_argument("--custody", required=True)
    conf_replica = commands.add_parser("run-confirmation-replica")
    conf_replica.add_argument("--primary", required=True)
    conf_replica.add_argument("--replica", required=True)
    conf_precheck = commands.add_parser("run-confirmation-replica-precheck")
    conf_precheck.add_argument("--primary", required=True)
    conf_precheck.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze-development":
        result = freeze_development()
    elif args.command == "verify-development":
        result = verify_development_packet()
    elif args.command == "freeze-confirmation":
        result = freeze_confirmation()
    elif args.command == "verify-confirmation":
        result = verify_confirmation_packet()
    elif args.command == "run-development-replica":
        result = _run_development(args.custody, replica=True)
    elif args.command == "run-confirmation-replica":
        _patch_v17_for_confirmation()
        result = _run_confirmation(args.replica, replica=True, entropy_from=args.primary)
    elif args.command == "run-confirmation-replica-precheck":
        right = _run_confirmation_prepare(args.replica, replica=True)
        comparison = _compare_confirmation_prechallenge(Path(args.primary), Path(args.replica))
        result = {"replica": right, "comparison": comparison}
    else:
        result = _run_end_to_end(args)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
