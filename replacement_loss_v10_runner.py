"""Custody-bound local runner for replacement-loss-v10.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v10-opt-causal-fit-v1, and
global-experiment-assurance-lifecycle-v1. Prompts and model artifacts are
written only beneath an explicitly supplied external custody directory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import model_info, snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

import replacement_loss_v3 as v3_core
import replacement_loss_v3_runner as v3
import replacement_loss_v10_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = "theseus-replacement-loss-v10-opt-causal-fit-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FAMILY = "opt_relu"
METHODS = ("activation_ridge", "causal_fit")
CONTROLS = ("random_replacement", "wrong_layer_replacement", "zero_intervention")
DEPTH_BANDS = (0, 1)
TOKEN_BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 8
CAUSAL_FIT_UPDATES = 100
CAUSAL_FIT_LEARNING_RATE = 1e-3
CAUSAL_FIT_WEIGHT_DECAY = 0.0
RIDGE = 1e-6
MAX_PARAMETERS = 200_000_000
MAX_WEIGHT_BYTES = 800 * 1024**2
MAX_TOTAL_SECONDS = 4 * 60 * 60
MAX_RSS_BYTES = 16 * 1024**3
MAX_SEQUENCE_LENGTH = 128
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 202609299
PACKET_FREEZE = "docs/replacement-loss-v10-freeze.json"
PACKET_FILES = (
    "docs/replacement-loss-v10.md",
    "AGENTS.md",
    "replacement_loss_v10_fixtures.py",
    "replacement_loss_v10_runner.py",
    "tests/test_replacement_loss_v10.py",
    "requirements-v10.lock.txt",
    "replacement_loss_v3.py",
    "replacement_loss_v3_fixtures.py",
    "replacement_loss_v3_runner.py",
)
DEPENDENCY_PINS = {
    "huggingface-hub": "0.36.2",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}
KNOWN_PRIOR_CHECKPOINT_IDS = {
    "ai-forever/rugpt3small_based_on_gpt2",
    "amd/amd-llama-135m",
    "babylm-community/babylm-baseline-100m-gpt2",
    "babylm/babyllama-100m-2024",
    "babylm/opt-125m-strict-2023",
    "babylm/opt-125m-strict-small-2023",
    "facebook/opt-125m",
    "opt-125m",
    "facebook/opt-1.3b",
    "facebook/opt-350m",
    "huggingfacetb/smollm-360m",
    "kanishka/opt-babylm2-rewritten-clean-spacy_no-num-adj-earlystop-strictest-bpe_seed-211_1e-3",
    "openai-community/gpt2-medium",
    "pgryko/babyllama-10m",
    "erwanf/gpt2-mini",
    "distilbert/distilgpt2",
    "stentorlabs/stentor-30m",
    "grushaprasad/opt125m-rootp-surprisal-curriculum",
}
KNOWN_PRIOR_WEIGHT_FILE_DIGESTS = {
    "0d2a4a387b0430654bb1f6813436bd31f8e10a06e18ea0e5c41b33120b78458e",
    "16192f4149f7d6a7c754dbae4d0ec2d06a259b9b11b91445585ecb617f9b2bf0",
    "3fa416a22cd9f0f6126f2ecfec8ec619ba1b440eb849b3985097172ea4bd8124",
    "44c4152579055e34dc5ec45941f07908ba6328eb43a9851f23d71b427b97b242",
    "7130e95a8c216c699c159fe9656c286e524f8ba64b660cae25227ba8c272f670",
    "82ac2b419d0c9d358f691f10043eb4e79a6a692457b593970617c43795704147",
    "8be96341568c5ba5bae4f3bbf1b140af978e4ed5a5c0467ec0b72204a72e13d8",
    "a408ffb02a9256ff5408a5c02a97279d3708ec117fa7e997874e71ea867291dc",
    "a5223ae6f3c26c6d90003f96a6bcd9a4aaaef0d36fca6469112efeeb985f2842",
    "aaf7f28e0f5cff74ea168059b73f88d7913d39f8aab36bcf8b9e19c3bd96c873",
    "cc3e2ac25e6582453f9af5993cacece36eb9777c5045c590965b7703f5bf3e3c",
    "e91f05d8506ee5efbd8c0fbfc1799c49af2b2f2cce824bc2d801d5af2a716cc2",
    "fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd",
    "7771b175e62d68de32cd21bac7356ac2a1739042b1c630adaee49d0e8aeac49f",
    "a2624b31484c352497c44cabecc752d3b48e45e04bf2fe39ad79a35cc8072308",
    "5301f1fc21b2020d1b716c7f70cf5f956ee51b2dbb9580b3425c02f36b4db562",
}
PRIOR_IDENTITY_INVENTORY_SHA256 = "64b83b0f8458b234db87028b3495a62370e550dd0268e4d072b099d9d600633f"
V9_CHECKPOINT_MANIFEST_SHA256 = "b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277"
_ACTIVE_STARTED = 0.0
_PRIOR_SECONDS = 0.0
_CUSTODY: Path | None = None
_ACTIVE_PHASE = "idle"


def _root() -> Path:
    return Path(__file__).resolve().parent


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _canonical_digest(value: Any) -> str:
    return _sha_bytes(_canonical(value))


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        raise FileExistsError(f"immutable v10 artifact already exists: {path.name}")
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _custody_root(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(_root())
    except ValueError:
        return path
    raise ValueError("model, prompt, and run artifacts must stay outside the repository")


def _packet_rows(root: Path | None = None) -> list[dict[str, str]]:
    repo = (root or _root()).resolve()
    rows = []
    for relative in PACKET_FILES:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"v10 frozen packet file is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path)})
    return rows


def freeze_packet(root: Path | None = None) -> dict[str, Any]:
    repo = (root or _root()).resolve()
    destination = repo / PACKET_FREEZE
    if destination.exists():
        raise FileExistsError("the v10 packet freeze is immutable")
    rows = _packet_rows(repo)
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "freeze_state": "FROZEN_BEFORE_CHECKPOINT_SELECTION",
        "created_at_utc": _now(),
        "packet_files": rows,
        "packet_sha256": _canonical_digest(rows),
        "checkpoint_identities_selected": False,
        "challenge_fixtures_generated": False,
        "model_family_count": 1,
        "fixture_seed_count": len(fixtures.SEEDS),
        "selected_layer_count": len(DEPTH_BANDS),
        "methods": list(METHODS),
        "resource_limits": {
            "execution": "local_cpu_only",
            "max_total_seconds": MAX_TOTAL_SECONDS,
            "max_rss_bytes": MAX_RSS_BYTES,
            "max_parameters": MAX_PARAMETERS,
            "max_weight_bytes": MAX_WEIGHT_BYTES,
            "paid_or_remote_compute": False,
        },
    }
    _write_json(destination, payload, immutable=True)
    return payload


def verify_packet(root: Path | None = None) -> dict[str, Any]:
    repo = (root or _root()).resolve()
    payload = _read_json(repo / PACKET_FREEZE)
    rows = _packet_rows(repo)
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("state_slice") != STATE_SLICE
        or payload.get("protocol_slice") != PROTOCOL_SLICE
        or payload.get("freeze_state") != "FROZEN_BEFORE_CHECKPOINT_SELECTION"
        or payload.get("packet_files") != rows
        or payload.get("packet_sha256") != _canonical_digest(rows)
        or payload.get("checkpoint_identities_selected") is not False
        or payload.get("challenge_fixtures_generated") is not False
    ):
        raise ValueError("v10 packet freeze verification failed")
    return payload


def archive_packet(custody_path: str | Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Copy the exact frozen packet to external custody before checkpoint selection."""

    custody = _custody_root(custody_path)
    repo = (repo_root or _root()).resolve()
    packet = verify_packet(repo)
    archive = custody / "frozen-packet"
    if archive.exists():
        return _verify_packet_archive(custody, repo_root=repo)
    rows = []
    for relative in (*PACKET_FILES, PACKET_FREEZE):
        source = repo / relative
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        rows.append({"path": relative, "bytes": target.stat().st_size, "sha256": _sha_file(target)})
    material = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "archive_files": rows,
    }
    manifest = {**material, "archive_sha256": _canonical_digest(material)}
    _write_json(custody / "frozen-packet-manifest.json", manifest, immutable=True)
    return manifest


def _verify_packet_archive(custody: Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    repo = (repo_root or _root()).resolve()
    packet = verify_packet(repo)
    manifest = _read_json(custody / "frozen-packet-manifest.json")
    material = {key: value for key, value in manifest.items() if key != "archive_sha256"}
    if manifest.get("archive_sha256") != _canonical_digest(material) or manifest.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("custodied frozen packet archive digest verification failed")
    expected_paths = {*PACKET_FILES, PACKET_FREEZE}
    if {row.get("path") for row in manifest.get("archive_files", [])} != expected_paths:
        raise ValueError("custodied frozen packet archive file inventory differs")
    expected_digests = {row["path"]: row["sha256"] for row in packet["packet_files"]}
    expected_digests[PACKET_FREEZE] = _sha_file(repo / PACKET_FREEZE)
    for row in manifest["archive_files"]:
        path = custody / "frozen-packet" / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"] or row["sha256"] != expected_digests[row["path"]]:
            raise ValueError("custodied frozen packet archive bytes failed verification")
    return manifest


def _verify_runtime() -> dict[str, Any]:
    actual = {name: importlib.metadata.version(name) for name in DEPENDENCY_PINS}
    mismatch = {name: {"expected": expected, "actual": actual[name]} for name, expected in DEPENDENCY_PINS.items() if actual[name] != expected}
    if mismatch:
        raise RuntimeError(f"runtime dependency pins differ from the frozen v10 packet: {mismatch}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    return {
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "host_id_sha256": _sha_bytes(platform.node().encode("utf-8")),
        "packages": actual,
        "dependency_lock_sha256": _sha_file(_root() / "requirements-v10.lock.txt"),
        "torch_num_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _begin(custody: Path, phase: str) -> None:
    global _ACTIVE_STARTED, _PRIOR_SECONDS, _CUSTODY, _ACTIVE_PHASE
    _CUSTODY = custody
    _ACTIVE_PHASE = phase
    ledger_path = custody / "phase-ledger.json"
    if ledger_path.exists():
        ledger = _read_json(ledger_path)
        _PRIOR_SECONDS = float(ledger.get("total_elapsed_seconds", 0.0))
    else:
        _PRIOR_SECONDS = 0.0
    _ACTIVE_STARTED = time.monotonic()
    _check_budget()


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _check_budget() -> None:
    if _ACTIVE_STARTED <= 0.0:
        return
    elapsed = _PRIOR_SECONDS + (time.monotonic() - _ACTIVE_STARTED)
    if elapsed > MAX_TOTAL_SECONDS:
        raise TimeoutError("v10 exceeded its frozen four-hour total wall-time limit")
    if _peak_rss_bytes() > MAX_RSS_BYTES:
        raise MemoryError("v10 exceeded its frozen 16 GiB resident-memory limit")


def _finish_phase(custody: Path, phase: str, started: float, *, status: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    elapsed = time.monotonic() - started
    ledger_path = custody / "phase-ledger.json"
    previous = _read_json(ledger_path) if ledger_path.exists() else {"phases": [], "total_elapsed_seconds": 0.0}
    row = {
        "phase": phase,
        "status": status,
        "elapsed_seconds": elapsed,
        "peak_rss_bytes": _peak_rss_bytes(),
        "runtime": _verify_runtime(),
    }
    if details:
        row.update(details)
    previous.setdefault("phases", []).append(row)
    previous["total_elapsed_seconds"] = float(previous.get("total_elapsed_seconds", 0.0)) + elapsed
    _write_json(ledger_path, previous)
    return row


def _phase_started(custody: Path, phase: str) -> bool:
    if (custody / "run-stop.json").exists():
        return True
    ledger_path = custody / "phase-ledger.json"
    if not ledger_path.exists():
        return False
    ledger = _read_json(ledger_path)
    return any(row.get("phase") == phase for row in ledger.get("phases", []))


def _record_stop(custody: Path, phase: str, error: BaseException) -> None:
    _write_json(custody / "run-stop.json", {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "phase": phase,
        "status": "INCONCLUSIVE",
        "error_type": type(error).__name__,
        "error_message": str(error)[:500],
        "stopped_at_utc": _now(),
        "packet_sha256": verify_packet()["packet_sha256"],
    }, immutable=not (custody / "run-stop.json").exists())


def _checkpoint_manifest(custody: Path) -> dict[str, Any]:
    packet = verify_packet()
    path = custody / "checkpoint-selection.json"
    payload = _read_json(path)
    if payload.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("checkpoint selection is not bound to the frozen v10 packet")
    archive = _verify_packet_archive(custody)
    if payload.get("packet_archive_sha256") != archive["archive_sha256"]:
        raise ValueError("checkpoint selection is not bound to the custodied frozen packet bytes")
    if payload.get("selected_after_packet_freeze") is not True:
        raise ValueError("checkpoint selection did not occur after packet freeze")
    freeze_time = datetime.fromisoformat(packet["created_at_utc"])
    selected_time = datetime.fromisoformat(payload["selected_at_utc"])
    if selected_time.tzinfo is None or freeze_time.tzinfo is None:
        raise ValueError("checkpoint selection timestamps must include a timezone")
    if selected_time <= freeze_time:
        raise ValueError("checkpoint selection timestamp does not follow packet freeze")
    identity = payload.get("model_id", "").casefold()
    if identity in KNOWN_PRIOR_CHECKPOINT_IDS:
        raise ValueError("checkpoint identity appears in the frozen prior identity exclusion set")
    revision = payload.get("revision", "")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()) or payload.get("resolved_revision") != revision:
        raise ValueError("checkpoint manifest does not bind an exact resolved 40-character revision")
    freshness = payload.get("freshness_screen", {})
    if (
        freshness.get("prior_identity_inventory_sha256") != PRIOR_IDENTITY_INVENTORY_SHA256
        or freshness.get("v9_checkpoint_manifest_sha256") != V9_CHECKPOINT_MANIFEST_SHA256
        or freshness.get("identity_match_count") != 0
        or freshness.get("weight_file_digest_match_count") != 0
    ):
        raise ValueError("checkpoint freshness attestation is missing or reports a prior identity/weight match")
    card_metadata = payload.get("model_card_metadata")
    if (
        payload.get("model_visibility") != "public_ungated"
        or not isinstance(card_metadata, dict)
        or str(card_metadata.get("license", "")).casefold() != str(payload.get("license_id", "")).casefold()
        or card_metadata.get("base_model")
        or payload.get("model_card_metadata_sha256") != _canonical_digest(card_metadata)
    ):
        raise ValueError("checkpoint public metadata does not verify the recorded base-model and license screen")
    card_path = custody / payload.get("local_path", "") / "README.md"
    expected_card_url = f"https://huggingface.co/{payload['model_id']}/blob/{revision}/README.md"
    if (
        payload.get("model_card_url") != expected_card_url
        or not card_path.is_file()
        or payload.get("model_card_sha256") != _sha_file(card_path)
    ):
        raise ValueError("pinned model-card evidence is missing or does not match the selected revision")
    selection_material = {key: value for key, value in payload.items() if key != "selection_sha256"}
    if payload.get("selection_sha256") != _canonical_digest(selection_material):
        raise ValueError("checkpoint selection manifest digest verification failed")
    return payload


def select_checkpoint(
    custody_path: str | Path,
    model_id: str,
    revision: str,
    *,
    license_id: str,
    license_url: str,
    license_checked_at_utc: str,
) -> dict[str, Any]:
    packet = verify_packet()
    custody = _custody_root(custody_path)
    custody.mkdir(parents=True, exist_ok=True)
    destination = custody / "checkpoint-selection.json"
    if destination.exists():
        raise FileExistsError("checkpoint selection is immutable; use a new custody root")
    if model_id.casefold() in KNOWN_PRIOR_CHECKPOINT_IDS:
        raise ValueError("checkpoint identity is excluded because it was used by a prior Theseus packet")
    if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower()):
        raise ValueError("checkpoint revision must be a full 40-character commit SHA")
    if not license_id.strip() or not license_url.startswith("https://"):
        raise ValueError("record a verified license identifier and direct HTTPS source")
    license_checked = datetime.fromisoformat(license_checked_at_utc)
    if license_checked.tzinfo is None:
        raise ValueError("license verification timestamp must include a timezone")
    selected_at = _now()
    if datetime.fromisoformat(selected_at) <= datetime.fromisoformat(packet["created_at_utc"]):
        raise ValueError("checkpoint selection must be later than the protocol freeze")
    packet_archive = archive_packet(custody)
    info = model_info(model_id, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise ValueError("Hub resolved a different commit than the requested pinned revision")
    card_metadata = info.card_data.to_dict() if info.card_data is not None else None
    if info.private is not False or info.gated is not False:
        raise ValueError("checkpoint must be publicly accessible without a gate")
    if not isinstance(card_metadata, dict) or str(card_metadata.get("license", "")).casefold() != license_id.casefold():
        raise ValueError("checkpoint public model-card license metadata does not match the recorded license identifier")
    if card_metadata.get("base_model"):
        raise ValueError("checkpoint model card declares an upstream base model and is not eligible as a base checkpoint")
    model_dir = custody / "checkpoint" / "files"
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=model_dir,
        local_files_only=False,
        repo_type="model",
    )
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    if getattr(config, "model_type", None) != "opt" or getattr(config, "activation_function", None) != "relu":
        raise ValueError("selected checkpoint does not satisfy the OPT/ReLU architecture rule")
    if int(getattr(config, "num_hidden_layers", 0)) < 3:
        raise ValueError("selected OPT checkpoint must contain at least three layers")
    if int(getattr(config, "max_position_embeddings", 0)) < MAX_SEQUENCE_LENGTH:
        raise ValueError("selected OPT checkpoint context length is below the frozen 128-token limit")
    model_card_path = model_dir / "README.md"
    if not model_card_path.is_file():
        raise ValueError("checkpoint snapshot is missing its revision-pinned model card")
    weight_rows = []
    for path in sorted(model_dir.rglob("*")):
        if path.is_file() and path.suffix in {".safetensors", ".bin"} and ".cache" not in path.parts:
            weight_rows.append({"path": path.relative_to(model_dir).as_posix(), "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    total_weight_bytes = sum(row["bytes"] for row in weight_rows)
    if not weight_rows or total_weight_bytes > MAX_WEIGHT_BYTES:
        raise ValueError("checkpoint has no supported weight files or exceeds the frozen 800 MB limit")
    weight_matches = [row["sha256"] for row in weight_rows if row["sha256"] in KNOWN_PRIOR_WEIGHT_FILE_DIGESTS]
    if weight_matches:
        raise ValueError("selected checkpoint weight bytes match a prior Theseus model artifact")
    freshness_screen = {
        "prior_identity_inventory_sha256": PRIOR_IDENTITY_INVENTORY_SHA256,
        "v9_checkpoint_manifest_sha256": V9_CHECKPOINT_MANIFEST_SHA256,
        "known_prior_identity_count": len(KNOWN_PRIOR_CHECKPOINT_IDS),
        "known_prior_weight_file_digest_count": len(KNOWN_PRIOR_WEIGHT_FILE_DIGESTS),
        "identity_match_count": 0,
        "weight_file_digest_match_count": 0,
    }
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "packet_archive_sha256": packet_archive["archive_sha256"],
        "selected_after_packet_freeze": True,
        "selected_at_utc": selected_at,
        "model_id": model_id,
        "revision": revision,
        "resolved_revision": info.sha,
        "freshness_screen": freshness_screen,
        "license_id": license_id,
        "license_source_url": license_url,
        "license_checked_at_utc": license_checked_at_utc,
        "model_visibility": "public_ungated",
        "model_card_url": f"https://huggingface.co/{model_id}/blob/{revision}/README.md",
        "model_card_sha256": _sha_file(model_card_path),
        "model_card_metadata": card_metadata,
        "model_card_metadata_sha256": _canonical_digest(card_metadata),
        "config_sha256": _sha_file(model_dir / "config.json"),
        "config_architecture": {
            "model_type": config.model_type,
            "activation_function": config.activation_function,
            "num_hidden_layers": int(config.num_hidden_layers),
            "hidden_size": int(config.hidden_size),
            "ffn_dim": int(config.ffn_dim),
            "vocab_size": int(config.vocab_size),
        },
        "local_path": "checkpoint/files",
        "weight_files": weight_rows,
        "total_weight_bytes": total_weight_bytes,
        "weights_sha256": _canonical_digest(weight_rows),
    }
    payload["selection_sha256"] = _canonical_digest(payload)
    _write_json(destination, payload, immutable=True)
    return payload


def _load_checkpoint(custody: Path) -> tuple[v3.Checkpoint, dict[str, Any]]:
    manifest = _checkpoint_manifest(custody)
    model_path = (custody / manifest["local_path"]).resolve()
    for row in manifest["weight_files"]:
        path = model_path / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError("custodied checkpoint weight bytes failed digest verification")
        if row["sha256"] in KNOWN_PRIOR_WEIGHT_FILE_DIGESTS:
            raise ValueError("selected checkpoint weight bytes match a known prior model artifact")
    if sum(int(row["bytes"]) for row in manifest["weight_files"]) > MAX_WEIGHT_BYTES:
        raise ValueError("custodied checkpoint exceeds the frozen 800 MB weight limit")
    if _sha_file(model_path / "config.json") != manifest["config_sha256"]:
        raise ValueError("custodied checkpoint config failed digest verification")
    checkpoint = v3.Checkpoint(
        family=FAMILY,
        model_id=manifest["model_id"],
        revision=manifest["revision"],
        local_path=model_path,
        selected_after_protocol_freeze=True,
    )
    return checkpoint, manifest


def _token_rows(records: list[dict[str, Any]], tokenizer: Any, *, continuation: bool) -> list[dict[str, Any]]:
    rows = v3._token_rows(tokenizer, records, with_continuation=continuation)
    for row in rows:
        if int(row["prompt_len"]) > MAX_SEQUENCE_LENGTH or len(row["ids"]) > MAX_SEQUENCE_LENGTH:
            raise ValueError("a prompt exceeds the frozen 128-token limit")
        if continuation and not row["continuation_ids"]:
            raise ValueError("a challenge continuation is empty")
    return rows


def _selected_positions(group: list[dict[str, Any]]) -> list[int]:
    return [int(row["prompt_len"]) - 1 for row in group]


def _next_logits(logits: torch.Tensor, positions: list[int]) -> torch.Tensor:
    return torch.stack([logits[index, position].detach().cpu() for index, position in enumerate(positions)])


def _direction(input_vector: torch.Tensor, *, key: bytes, rms_scale: float | None = None) -> torch.Tensor:
    vector = input_vector.detach().float().cpu().numpy()
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("intervention base activation must be a non-empty vector")
    digest = hashlib.sha256(key).digest()
    coordinate = int.from_bytes(digest[:8], "big") % vector.size
    sign = 1.0 if digest[8] & 1 else -1.0
    scale = float(np.sqrt(np.mean(np.square(vector, dtype=np.float64)))) if rms_scale is None else float(rms_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("intervention RMS must be positive and finite")
    result = torch.zeros(vector.size, dtype=torch.float32)
    result[coordinate] = sign * 0.01 * scale
    return result


def _visible_selection(model: torch.nn.Module, tokenizer: Any, records: list[dict[str, Any]]) -> tuple[list[int], list[float], list[float]]:
    rows = _token_rows(records, tokenizer, continuation=False)
    layer_count = v3._layer_count(model, FAMILY)
    if layer_count < 3:
        raise ValueError("checkpoint must contain at least three OPT MLP projections")
    captured: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    for _group, inputs in v3._batches(rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
        _check_budget()
        activations = v3._capture_all_layer_inputs(model, FAMILY, inputs)
        for layer, values in enumerate(activations):
            captured[layer].append(values.float().cpu())
    all_x = [torch.cat(parts) for parts in captured]
    variances = [float(values.var(dim=0, unbiased=False).mean().item()) for values in all_x]
    rms_values = [float(values.square().mean().sqrt().item()) for values in all_x]
    selected = []
    for band in DEPTH_BANDS:
        start = (band * layer_count) // 3
        end = ((band + 1) * layer_count) // 3
        if end <= start:
            raise ValueError("one of the frozen depth bands has no layer")
        selected.append(max(range(start, end), key=lambda index: (variances[index], -index)))
    if len(set(selected)) != 2 or selected != sorted(selected):
        raise ValueError("visible-only selector did not return two distinct ordered layers")
    return selected, variances, rms_values


def _capture_adaptation_layer(
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    rows: list[dict[str, Any]],
    seed_index: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    tokenized = _token_rows(rows, tokenizer, continuation=False)
    for index, row in enumerate(tokenized):
        row["fixture_index"] = index
    x_all, y_all = [], []
    training_rows: list[dict[str, Any] | None] = [None] * len(rows)
    for group, inputs in v3._batches(tokenized, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
        _check_budget()
        positions = _selected_positions(group)
        base_logits, x, y = v3._run_projection(model, FAMILY, layer, inputs, positions=positions)
        if base_logits is None or y is None:
            raise RuntimeError("adaptation capture omitted source outputs")
        deltas = torch.stack([
            _direction(
                x[index],
                key=f"{PROTOCOL_ID}\0adapt\0{seed_index}\0{layer}\0{row['example_id']}".encode(),
            )
            for index, row in enumerate(group)
        ])
        changed_logits, _x_changed, changed_y = v3._run_projection(
            model, FAMILY, layer, inputs, positions=positions, deltas=deltas
        )
        if changed_logits is None or changed_y is None:
            raise RuntimeError("adaptation intervention capture omitted source outputs")
        base_next = _next_logits(base_logits, positions)
        changed_next = _next_logits(changed_logits, positions)
        x_all.append(x.float().cpu())
        y_all.append(y.float().cpu())
        for index, row in enumerate(group):
            fixture_index = int(row["fixture_index"])
            training_rows[fixture_index] = {
                "input_ids": list(row["ids"]),
                "position": int(row["prompt_len"]) - 1,
                "delta": deltas[index].cpu().numpy(),
                "target_y": y[index].float().cpu().numpy(),
                "teacher_effect": (changed_next[index] - base_next[index]).float().cpu().numpy(),
                "stratum": str(row["stratum"]),
                "example_id": str(row["example_id"]),
            }
    if any(row is None for row in training_rows):
        raise RuntimeError("adaptation capture has missing rows")
    return torch.cat(x_all), torch.cat(y_all), [row for row in training_rows if row is not None]


def _collate_training(rows: list[dict[str, Any]], pad_token_id: int) -> tuple[dict[str, torch.Tensor], list[int], torch.Tensor, torch.Tensor, torch.Tensor]:
    max_length = max(len(row["input_ids"]) for row in rows)
    input_ids = torch.full((len(rows), max_length), int(pad_token_id), dtype=torch.long)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.long)
    for index, row in enumerate(rows):
        token_ids = torch.as_tensor(row["input_ids"], dtype=torch.long)
        input_ids[index, : len(token_ids)] = token_ids
        attention_mask[index, : len(token_ids)] = 1
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    positions = [int(row["position"]) for row in rows]
    deltas = torch.as_tensor(np.stack([row["delta"] for row in rows]), dtype=torch.float32)
    targets = torch.as_tensor(np.stack([row["target_y"] for row in rows]), dtype=torch.float32)
    effects = torch.as_tensor(np.stack([row["teacher_effect"] for row in rows]), dtype=torch.float32)
    return inputs, positions, deltas, targets, effects


def _candidate_forward(
    model: torch.nn.Module,
    layer: int,
    inputs: dict[str, torch.Tensor],
    positions: list[int],
    deltas: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    module = v3._projection(model, FAMILY, layer)
    outputs: list[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if deltas is None:
            return None
        activation = args[0]
        batch, sequence = inputs["input_ids"].shape
        activation_bth = v3._activation_bth(activation, batch, sequence)
        changed = activation_bth.clone()
        for index, position in enumerate(positions):
            changed[index, position] += deltas[index].to(device=changed.device, dtype=changed.dtype)
        if activation.ndim == 2:
            changed = changed.reshape_as(activation)
        return (changed, *args[1:])

    def post_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        tensor = output[0] if isinstance(output, tuple) else output
        batch, sequence = inputs["input_ids"].shape
        values = v3._activation_bth(tensor, batch, sequence)
        outputs.append(torch.stack([values[index, position] for index, position in enumerate(positions)]))

    before = module.register_forward_pre_hook(pre_hook)
    after = module.register_forward_hook(post_hook)
    try:
        logits = model(**inputs, use_cache=False, return_dict=True).logits
    finally:
        before.remove()
        after.remove()
    if not outputs:
        raise RuntimeError("candidate output hook was not invoked")
    next_logits = torch.stack([logits[index, position] for index, position in enumerate(positions)])
    return next_logits, outputs[0]


def _new_linear(in_features: int, out_features: int, use_bias: bool, seed: int) -> torch.nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32, device="cpu")


def _fit_causal_candidate(
    model: torch.nn.Module,
    layer: int,
    in_features: int,
    out_features: int,
    use_bias: bool,
    training_rows: list[dict[str, Any]],
    pad_token_id: int,
    *,
    seed: int,
    updates: int = CAUSAL_FIT_UPDATES,
    learning_rate: float = CAUSAL_FIT_LEARNING_RATE,
    weight_decay: float = CAUSAL_FIT_WEIGHT_DECAY,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    if updates < 1 or len(training_rows) < TRAIN_BATCH_SIZE:
        raise ValueError("causal fitting requires a positive update count and at least one full batch")
    target_values = torch.as_tensor(np.stack([row["target_y"] for row in training_rows]), dtype=torch.float32)
    if target_values.shape[1] != out_features:
        raise ValueError("adaptation targets do not match candidate output width")
    source_module = v3._projection(model, FAMILY, layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidate = _new_linear(in_features, out_features, use_bias, seed)
    initialization_digest = v3_core._module_digest(candidate)
    for parameter in candidate.parameters():
        parameter.requires_grad_(True)
    v3._set_projection(model, FAMILY, layer, candidate)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=learning_rate, weight_decay=weight_decay)
    generator = np.random.default_rng(int(seed) ^ 0xCA05A1)
    order = generator.permutation(len(training_rows))
    activation_scale = float(target_values.square().mean().clamp_min(1e-6).item())
    final_activation_loss = None
    final_causal_loss = None
    try:
        model.eval()
        candidate.train()
        for update in range(updates):
            _check_budget()
            start = (update * TRAIN_BATCH_SIZE) % len(order)
            indexes = order[(start + np.arange(TRAIN_BATCH_SIZE)) % len(order)]
            batch_rows = [training_rows[int(index)] for index in indexes]
            inputs, positions, deltas, targets, teacher_effects = _collate_training(batch_rows, pad_token_id)
            base_logits, predicted_y = _candidate_forward(model, layer, inputs, positions, None)
            changed_logits, _changed_y = _candidate_forward(model, layer, inputs, positions, deltas)
            predicted_effect = changed_logits - base_logits
            effect_scales = teacher_effects.square().mean(dim=1).sqrt().clamp_min(1e-5)
            activation_loss = (predicted_y - targets).square().mean() / activation_scale
            causal_loss = ((predicted_effect - teacher_effects) / effect_scales[:, None]).square().mean()
            loss = activation_loss + causal_loss
            if not torch.isfinite(loss):
                raise ValueError("causal-fit optimization produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            final_activation_loss = float(activation_loss.detach().item())
            final_causal_loss = float(causal_loss.detach().item())
        candidate.eval()
        receipt = {
            "method": "causal_fit",
            "initialization": "new_affine_module_without_source_weights",
            "initialization_sha256": initialization_digest,
            "fitted_candidate_sha256": v3_core._module_digest(candidate),
            "adaptation_rows": len(training_rows),
            "updates": updates,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "activation_loss_weight": 1.0,
            "causal_loss_weight": 1.0,
            "activation_normalization_mean_square": activation_scale,
            "final_update_normalized_activation_mse": final_activation_loss,
            "final_update_normalized_causal_mse": final_causal_loss,
            "optimizer": "AdamW",
            "seed": int(seed),
        }
        return candidate, receipt
    finally:
        v3._set_projection(model, FAMILY, layer, source_module)
        gc.collect()


def _candidate_seed(seed_index: int, layer: int, method: str) -> int:
    digest = hashlib.sha256(f"{PROTOCOL_ID}\0{seed_index}\0{layer}\0{method}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _save_candidate(path: Path, module: torch.nn.Linear) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    save_file(state, str(path))
    return {
        "path": path.relative_to(_CUSTODY).as_posix(),
        "file_sha256": _sha_file(path),
        "state_sha256": v3_core._module_digest(module),
        "in_features": int(module.in_features),
        "out_features": int(module.out_features),
        "use_bias": module.bias is not None,
    }


def _load_candidate(custody: Path, row: dict[str, Any]) -> torch.nn.Linear:
    path = custody / row["path"]
    if not path.is_file() or _sha_file(path) != row["file_sha256"]:
        raise ValueError("frozen replacement candidate bytes do not match the candidate manifest")
    module = torch.nn.Linear(int(row["in_features"]), int(row["out_features"]), bias=bool(row["use_bias"]), dtype=torch.float32)
    module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    if v3_core._module_digest(module) != row["state_sha256"]:
        raise ValueError("candidate tensor-state digest differs from its manifest")
    module.eval()
    return module


def _candidate_freeze(custody: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_json(custody / "candidate-freeze.json")
    material = {key: value for key, value in payload.items() if key not in {"candidate_set_sha256", "created_at_utc"}}
    if payload.get("candidate_set_sha256") != _canonical_digest(material):
        raise ValueError("candidate freeze digest verification failed")
    packet = verify_packet()
    if payload.get("packet_sha256") != packet["packet_sha256"]:
        raise ValueError("candidate freeze is bound to a different packet")
    checkpoint = _checkpoint_manifest(custody)
    if payload.get("checkpoint_selection_sha256") != checkpoint.get("selection_sha256"):
        raise ValueError("candidate freeze binds a different checkpoint")
    return payload, checkpoint


def _verify_candidates(custody: Path, freeze: dict[str, Any]) -> None:
    rows = freeze.get("candidates", [])
    expected = {(seed, layer, method) for seed in range(len(fixtures.SEEDS)) for layer in freeze["selected_layers"] for method in METHODS}
    actual = {(int(row["seed_index"]), int(row["layer"]), row["method"]) for row in rows}
    if actual != expected:
        raise ValueError("candidate freeze does not cover every seed/layer/method cell")
    for row in rows:
        module = _load_candidate(custody, row)
        del module


def prepare(custody_path: str | Path) -> dict[str, Any]:
    custody = _custody_root(custody_path)
    if (custody / "candidate-freeze.json").exists() or _phase_started(custody, "prepare"):
        raise FileExistsError("v10 preparation is immutable once started; preserve it and use a new custody root")
    packet = verify_packet()
    runtime = _verify_runtime()
    started = time.monotonic()
    _begin(custody, "prepare")
    try:
        checkpoint, selection = _load_checkpoint(custody)
        model, tokenizer = v3._load_local_model(checkpoint)
        model.eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > MAX_PARAMETERS:
            raise ValueError("loaded checkpoint exceeds the frozen 200 million parameter limit")
        visible_records = [row for row in fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")["records"]]
        visible_payload = fixtures.fixture_payload(fixtures.generate_visible_fixture(), fixture_kind="visible")
        _write_json(custody / "fixtures" / "visible.json", visible_payload, immutable=True)
        selected_layers, variances, visible_rms = _visible_selection(model, tokenizer, visible_records)
        adaptation_payloads = []
        for seed_index in range(len(fixtures.SEEDS)):
            payload = fixtures.fixture_payload(fixtures.generate_adaptation_fixture(seed_index), fixture_kind=f"adaptation_seed_{seed_index}")
            fixtures.validate_partition_disjointness(visible_payload, *adaptation_payloads, payload)
            adaptation_payloads.append(payload)
            _write_json(custody / "fixtures" / f"adaptation-seed-{seed_index}.json", payload, immutable=True)

        candidate_rows = []
        adaptation_artifacts = []
        layer_training_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}
        layer_features: dict[int, tuple[int, int, bool]] = {}
        stress_coordinates: dict[str, int] = {}
        input_rms_by_layer = {str(layer): visible_rms[layer] for layer in selected_layers}
        for seed_index, payload in enumerate(adaptation_payloads):
            records = payload["records"]
            for layer in selected_layers:
                x, y, training_rows = _capture_adaptation_layer(model, tokenizer, layer, records, seed_index)
                layer_training_rows[(seed_index, layer)] = training_rows
                stress_coordinates[f"{seed_index}:{layer}"] = int(x.var(dim=0, unbiased=False).argmax().item())
                module_dimensions = v3._module_dimensions(v3._projection(model, FAMILY, layer))
                layer_features[layer] = module_dimensions
                adaptation_path = custody / "adaptation" / f"seed-{seed_index}" / f"layer-{layer}.npz"
                v3._save_npz_deterministic(
                    adaptation_path,
                    inputs=x.numpy(),
                    targets=y.numpy(),
                    deltas=np.stack([row["delta"] for row in training_rows]),
                    teacher_logit_effects=np.stack([row["teacher_effect"] for row in training_rows]),
                    example_ids=np.asarray([row["example_id"] for row in training_rows], dtype="U24"),
                    strata=np.asarray([row["stratum"] for row in training_rows], dtype="U24"),
                )
                adaptation_artifacts.append({
                    "seed_index": seed_index,
                    "layer": layer,
                    "path": adaptation_path.relative_to(custody).as_posix(),
                    "sha256": _sha_file(adaptation_path),
                    "row_count": len(training_rows),
                })
                in_features, out_features, use_bias = module_dimensions
                ridge_module, ridge_receipt = v3_core.fit_independent_linear_replacement(
                    in_features,
                    out_features,
                    x,
                    y,
                    use_bias=use_bias,
                    seed=_candidate_seed(seed_index, layer, "activation_ridge"),
                    ridge=RIDGE,
                )
                candidates = [("activation_ridge", ridge_module, ridge_receipt)]
                causal_module, causal_receipt = _fit_causal_candidate(
                    model,
                    layer,
                    in_features,
                    out_features,
                    use_bias,
                    training_rows,
                    tokenizer.pad_token_id,
                    seed=_candidate_seed(seed_index, layer, "causal_fit"),
                )
                candidates.append(("causal_fit", causal_module, causal_receipt))
                for method, module, receipt in candidates:
                    relative = f"candidates/seed-{seed_index}/layer-{layer}/{method}.safetensors"
                    saved = _save_candidate(custody / relative, module)
                    candidate_rows.append({
                        "seed_index": seed_index,
                        "seed": fixtures.SEEDS[seed_index],
                        "layer": layer,
                        "method": method,
                        **saved,
                        "fit_receipt": receipt,
                        "adaptation_sha256": adaptation_artifacts[-1]["sha256"],
                    })
                    del module
                del ridge_module, causal_module
                gc.collect()
                _check_budget()
        checkpoint_identity_digest = _canonical_digest({
            "model_id": selection["model_id"],
            "revision": selection["revision"],
            "weights_sha256": selection["weights_sha256"],
        })
        freeze_material = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": packet["packet_sha256"],
            "checkpoint_selection_sha256": selection["selection_sha256"],
            "checkpoint_identity_sha256": checkpoint_identity_digest,
            "visible_fixture_sha256": visible_payload["example_set_sha256"],
            "visible_selection_only": True,
            "selected_layers": selected_layers,
            "visible_layer_variance": variances,
            "visible_input_rms": input_rms_by_layer,
            "adaptation_stress_coordinate": stress_coordinates,
            "layer_dimensions": {str(layer): list(layer_features[layer]) for layer in selected_layers},
            "adaptation_fixture_sha256": [row["example_set_sha256"] for row in adaptation_payloads],
            "adaptation_artifacts": adaptation_artifacts,
            "candidate_count": len(candidate_rows),
            "candidates": candidate_rows,
            "causal_fit_updates": CAUSAL_FIT_UPDATES,
            "runtime": runtime,
            "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=_root(), check=True, capture_output=True, text=True).stdout.strip(),
        }
        freeze = {
            **freeze_material,
            "created_at_utc": _now(),
            "candidate_set_sha256": _canonical_digest(freeze_material),
        }
        _write_json(custody / "candidate-freeze.json", freeze, immutable=True)
        _verify_candidates(custody, freeze)
        _finish_phase(custody, "prepare", started, status="COMPLETED", details={
            "selected_layers": selected_layers,
            "candidate_count": len(candidate_rows),
            "checkpoint_identity_sha256": checkpoint_identity_digest,
        })
        del model, tokenizer
        gc.collect()
        return {"candidate_set_sha256": freeze["candidate_set_sha256"], "selected_layers": selected_layers, "candidate_count": len(candidate_rows)}
    except BaseException as error:
        _finish_phase(custody, "prepare", started, status="STOPPED", details={"error_type": type(error).__name__})
        _record_stop(custody, "prepare", error)
        raise


def _load_fixtures(custody: Path, relative: str) -> dict[str, Any]:
    payload = _read_json(custody / relative)
    if payload.get("protocol_id") != PROTOCOL_ID or payload.get("protocol_slice") != PROTOCOL_SLICE:
        raise ValueError("fixture belongs to another protocol")
    return payload


def _challenge_direction(
    entropy: bytes,
    seed_index: int,
    layer: int,
    example_id: str,
    in_features: int,
    rms: float,
    *,
    stratum: str,
    stress_coordinate: int,
) -> torch.Tensor:
    key = b"\0".join((entropy, PROTOCOL_ID.encode(), str(seed_index).encode(), str(layer).encode(), example_id.encode()))
    digest = hashlib.sha256(key).digest()
    if stratum == "adversarial":
        coordinate = int(stress_coordinate)
        if coordinate < 0 or coordinate >= in_features:
            raise ValueError("adaptation-selected stress coordinate is outside the target module input")
    else:
        coordinate = int.from_bytes(digest[:8], "big") % in_features
    sign = 1.0 if digest[8] & 1 else -1.0
    vector = torch.zeros(in_features, dtype=torch.float32)
    vector[coordinate] = sign * 0.01 * float(rms)
    return vector


def _candidate_row(freeze: dict[str, Any], seed_index: int, layer: int, method: str) -> dict[str, Any]:
    matches = [row for row in freeze["candidates"] if int(row["seed_index"]) == seed_index and int(row["layer"]) == layer and row["method"] == method]
    if len(matches) != 1:
        raise ValueError("candidate lookup is ambiguous or incomplete")
    return matches[0]


def _predict_arrays(
    model: torch.nn.Module,
    tokenizer: Any,
    layer: int,
    records: list[dict[str, Any]],
    deltas: np.ndarray,
) -> dict[str, np.ndarray]:
    tokenized = _token_rows(records, tokenizer, continuation=True)
    for index, row in enumerate(tokenized):
        row["fixture_index"] = index
    fields: dict[str, list[np.ndarray]] = {name: [] for name in ("boundary_output", "module_effect", "next_logits", "changed_next_logits", "logit_effect", "ce")}
    for group, inputs in v3._batches(tokenized, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
        _check_budget()
        positions = _selected_positions(group)
        indexes = [int(row["fixture_index"]) for row in group]
        local_deltas = torch.as_tensor(deltas[indexes], dtype=torch.float32)
        base_logits, _x, base_outputs = v3._run_projection(model, FAMILY, layer, inputs, positions=positions)
        changed_logits, _x_changed, changed_outputs = v3._run_projection(model, FAMILY, layer, inputs, positions=positions, deltas=local_deltas)
        if base_logits is None or changed_logits is None or base_outputs is None or changed_outputs is None:
            raise RuntimeError("challenge prediction omitted required module outputs")
        next_logits = _next_logits(base_logits, positions).numpy()
        changed_next_logits = _next_logits(changed_logits, positions).numpy()
        fields["boundary_output"].append(base_outputs.cpu().numpy())
        fields["module_effect"].append((changed_outputs - base_outputs).cpu().numpy())
        fields["next_logits"].append(next_logits)
        fields["changed_next_logits"].append(changed_next_logits)
        fields["logit_effect"].append(changed_next_logits - next_logits)
        fields["ce"].append(v3._continuation_cross_entropy(base_logits, group))
    arrays = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    arrays["example_ids"] = np.asarray([row["example_id"] for row in records], dtype="U24")
    arrays["strata"] = np.asarray([row["stratum"] for row in records], dtype="U24")
    return arrays


def _activation_rescue_forward(
    model: torch.nn.Module,
    layer: int,
    group: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    candidate: torch.nn.Linear,
    teacher_outputs_all: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a held-out forward pass after restoring teacher boundary activations."""

    original = v3._projection(model, FAMILY, layer)
    v3._set_projection(model, FAMILY, layer, candidate)
    module = v3._projection(model, FAMILY, layer)

    def rescue_hook(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        batch, sequence = inputs["input_ids"].shape
        source = v3._activation_bth(teacher_outputs_all, batch, sequence).to(device=tensor.device, dtype=tensor.dtype)
        replacement = source.reshape_as(tensor) if tensor.ndim == 2 else source
        if isinstance(output, tuple):
            return (replacement, *output[1:])
        return replacement

    handle = module.register_forward_hook(rescue_hook)
    try:
        with torch.no_grad():
            logits = model(**inputs, use_cache=False, return_dict=True).logits
        positions = _selected_positions(group)
        return _next_logits(logits, positions).numpy(), v3._continuation_cross_entropy(logits, group)
    finally:
        handle.remove()
        v3._set_projection(model, FAMILY, layer, original)


def _activation_rescue_metrics(
    rescued_logits: np.ndarray,
    rescued_ce: np.ndarray,
    teacher_logits: np.ndarray,
    teacher_ce: np.ndarray,
) -> dict[str, float]:
    relative = np.linalg.norm(rescued_logits - teacher_logits, axis=1) / np.maximum(np.linalg.norm(teacher_logits, axis=1), 1e-12)
    ce_delta = np.asarray(rescued_ce, dtype=np.float64) - np.asarray(teacher_ce, dtype=np.float64)
    return {
        "next_token_normalized_logit_error_max": float(np.max(relative)),
        "continuation_cross_entropy_delta_max_abs_nats_per_token": float(np.max(np.abs(ce_delta))),
    }


def predict(custody_path: str | Path, *, entropy_source: str | Path | None = None) -> dict[str, Any]:
    custody = _custody_root(custody_path)
    if (custody / "prediction-manifest.json").exists() or _phase_started(custody, "predict"):
        raise FileExistsError("v10 prediction phase is immutable once started; preserve it and use a new custody root")
    runtime = _verify_runtime()
    started = time.monotonic()
    _begin(custody, "predict")
    try:
        freeze, selection = _candidate_freeze(custody)
        _verify_candidates(custody, freeze)
        checkpoint, _ = _load_checkpoint(custody)
        entropy_root = Path(entropy_source).expanduser().resolve() if entropy_source else None
        entropy_rows = []
        challenge_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
        deltas_by_seed_layer: dict[tuple[int, int], np.ndarray] = {}
        for seed_index in range(len(fixtures.SEEDS)):
            entropy_path = (entropy_root / "challenge-entropy" / f"seed-{seed_index}.bin") if entropy_root else None
            if entropy_path is not None:
                entropy = entropy_path.read_bytes()
            else:
                entropy = os.urandom(32)
            if len(entropy) != 32:
                raise ValueError("challenge entropy must be exactly 256 bits")
            local_entropy = custody / "challenge-entropy" / f"seed-{seed_index}.bin"
            local_entropy.parent.mkdir(parents=True, exist_ok=True)
            if local_entropy.exists():
                raise FileExistsError("challenge entropy already exists; preserve it and use a new custody root")
            local_entropy.write_bytes(entropy)
            local_entropy.chmod(0o600)
            records = fixtures.generate_challenge_fixture(seed_index, entropy=entropy)[0]
            payload = fixtures.fixture_payload(records, fixture_kind=f"challenge_seed_{seed_index}")
            adaptation = _load_fixtures(custody, f"fixtures/adaptation-seed-{seed_index}.json")
            visible = _load_fixtures(custody, "fixtures/visible.json")
            fixtures.validate_partition_disjointness(visible, adaptation, payload)
            _write_json(custody / "challenges" / f"seed-{seed_index}" / "prompts.json", payload, immutable=True)
            challenge_rows_by_seed[seed_index] = payload["records"]
            entropy_rows.append({
                "seed_index": seed_index,
                "entropy_sha256": _sha_file(local_entropy),
                "fixture_sha256": payload["example_set_sha256"],
            })
            for layer in freeze["selected_layers"]:
                dimensions = freeze["layer_dimensions"][str(layer)]
                input_width = int(dimensions[0])
                rms = float(freeze["visible_input_rms"][str(layer)])
                deltas = np.stack([
                    _challenge_direction(
                        entropy,
                        seed_index,
                        int(layer),
                        row["example_id"],
                        input_width,
                        rms,
                        stratum=str(row["stratum"]),
                        stress_coordinate=int(freeze["adaptation_stress_coordinate"][f"{seed_index}:{layer}"]),
                    ).numpy()
                    for row in payload["records"]
                ])
                path = custody / "challenges" / f"seed-{seed_index}" / f"layer-{layer}-interventions.npz"
                v3._save_npz_deterministic(path, deltas=deltas, example_ids=np.asarray([row["example_id"] for row in payload["records"]], dtype="U24"))
                deltas_by_seed_layer[(seed_index, int(layer))] = deltas
        challenge_manifest_material = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": freeze["candidate_set_sha256"],
            "challenge_generation_started_after_candidate_freeze": True,
            "created_at_utc": _now(),
            "entropy": entropy_rows,
            "challenge_rows": [],
        }
        for seed_index in range(len(fixtures.SEEDS)):
            prompt_path = custody / "challenges" / f"seed-{seed_index}" / "prompts.json"
            challenge_manifest_material["challenge_rows"].append({
                "seed_index": seed_index,
                "prompt_sha256": _sha_file(prompt_path),
                "interventions": [
                    {"layer": int(layer), "sha256": _sha_file(custody / "challenges" / f"seed-{seed_index}" / f"layer-{layer}-interventions.npz")}
                    for layer in freeze["selected_layers"]
                ],
            })
        challenge_manifest_material["challenge_manifest_sha256"] = _canonical_digest({k: v for k, v in challenge_manifest_material.items() if k != "created_at_utc"})
        _write_json(custody / "challenge-manifest.json", challenge_manifest_material, immutable=True)

        prediction_rows = []
        for seed_index in range(len(fixtures.SEEDS)):
            records = challenge_rows_by_seed[seed_index]
            for layer in freeze["selected_layers"]:
                deltas = deltas_by_seed_layer[(seed_index, int(layer))]
                model, tokenizer = v3._load_local_model(checkpoint)
                model.eval()
                original_module = v3._projection(model, FAMILY, int(layer))
                for method in METHODS:
                    candidate_info = _candidate_row(freeze, seed_index, int(layer), method)
                    candidate = _load_candidate(custody, candidate_info)
                    v3._set_projection(model, FAMILY, int(layer), candidate)
                    arrays = _predict_arrays(model, tokenizer, int(layer), records, deltas)
                    relative = f"predictions/seed-{seed_index}/layer-{layer}/{method}.npz"
                    path = custody / relative
                    v3._save_npz_deterministic(path, **arrays)
                    prediction_rows.append({
                        "seed_index": seed_index,
                        "layer": int(layer),
                        "method": method,
                        "path": relative,
                        "sha256": _sha_file(path),
                        "original_model_outcomes_opened": False,
                    })
                    v3._set_projection(model, FAMILY, int(layer), original_module)
                    del candidate
                    gc.collect()

                random_seed = _candidate_seed(seed_index, int(layer), "random_control")
                in_features, out_features, use_bias = v3._module_dimensions(original_module)
                random_candidate = _new_linear(in_features, out_features, use_bias, random_seed).eval()
                v3._set_projection(model, FAMILY, int(layer), random_candidate)
                arrays = _predict_arrays(model, tokenizer, int(layer), records, deltas)
                relative = f"predictions/seed-{seed_index}/layer-{layer}/control-random.npz"
                path = custody / relative
                v3._save_npz_deterministic(path, **arrays)
                prediction_rows.append({"seed_index": seed_index, "layer": int(layer), "control": "random_replacement", "path": relative, "sha256": _sha_file(path), "original_model_outcomes_opened": False})
                v3._set_projection(model, FAMILY, int(layer), original_module)
                del random_candidate
                gc.collect()

                other_layer = next(int(item) for item in freeze["selected_layers"] if int(item) != int(layer))
                wrong_info = _candidate_row(freeze, seed_index, other_layer, "causal_fit")
                wrong_candidate = _load_candidate(custody, wrong_info)
                target_dimensions = v3._module_dimensions(original_module)
                if (wrong_candidate.in_features, wrong_candidate.out_features, wrong_candidate.bias is not None) != target_dimensions:
                    raise ValueError("wrong-layer control module dimensions do not match the selected layer")
                v3._set_projection(model, FAMILY, int(layer), wrong_candidate)
                arrays = _predict_arrays(model, tokenizer, int(layer), records, deltas)
                relative = f"predictions/seed-{seed_index}/layer-{layer}/control-wrong-layer.npz"
                path = custody / relative
                v3._save_npz_deterministic(path, **arrays)
                prediction_rows.append({"seed_index": seed_index, "layer": int(layer), "control": "wrong_layer_replacement", "path": relative, "sha256": _sha_file(path), "original_model_outcomes_opened": False})
                v3._set_projection(model, FAMILY, int(layer), original_module)
                del wrong_candidate
                gc.collect()

                zero_info = _candidate_row(freeze, seed_index, int(layer), "causal_fit")
                zero_candidate = _load_candidate(custody, zero_info)
                v3._set_projection(model, FAMILY, int(layer), zero_candidate)
                zeros = np.zeros_like(deltas)
                arrays = _predict_arrays(model, tokenizer, int(layer), records, zeros)
                relative = f"predictions/seed-{seed_index}/layer-{layer}/control-zero-intervention.npz"
                path = custody / relative
                v3._save_npz_deterministic(path, **arrays)
                prediction_rows.append({"seed_index": seed_index, "layer": int(layer), "control": "zero_intervention", "path": relative, "sha256": _sha_file(path), "original_model_outcomes_opened": False})
                v3._set_projection(model, FAMILY, int(layer), original_module)
                del model, tokenizer, zero_candidate, original_module
                gc.collect()
                _check_budget()

        prediction_material = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": freeze["candidate_set_sha256"],
            "challenge_manifest_sha256": challenge_manifest_material["challenge_manifest_sha256"],
            "predictions": prediction_rows,
            "created_at_utc": _now(),
            "original_model_outcomes_opened": False,
        }
        prediction_material["prediction_manifest_sha256"] = _canonical_digest({k: v for k, v in prediction_material.items() if k != "created_at_utc"})
        _write_json(custody / "prediction-manifest.json", prediction_material, immutable=True)
        _finish_phase(custody, "predict", started, status="COMPLETED", details={"prediction_count": len(prediction_rows), "challenge_manifest_sha256": challenge_manifest_material["challenge_manifest_sha256"]})
        return {"prediction_count": len(prediction_rows), "prediction_manifest_sha256": prediction_material["prediction_manifest_sha256"]}
    except BaseException as error:
        _finish_phase(custody, "predict", started, status="STOPPED", details={"error_type": type(error).__name__})
        _record_stop(custody, "predict", error)
        raise


def _effect_metrics(predicted: np.ndarray, observed: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    rows = [
        v3_core.causal_effect_metrics(p, o, resolution_floor=1e-12, scale=float(s))
        for p, o, s in zip(predicted, observed, scale)
    ]
    resolvable = [row for row in rows if row["resolvable"]]
    if not resolvable:
        return {"count": len(rows), "resolvable_count": 0, "direction_accuracy": None, "median_cosine": None, "median_normalized_error": None}
    return {
        "count": len(rows),
        "resolvable_count": len(resolvable),
        "direction_accuracy": float(np.mean([row["direction_correct"] for row in resolvable])),
        "median_cosine": float(np.median([row["cosine_similarity"] for row in resolvable])),
        "median_normalized_error": float(np.median([row["normalized_error"] for row in resolvable])),
    }


def _activation_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    cosines, errors = [], []
    for left, right in zip(predicted, observed):
        cosine = float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30))
        error = float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-12))
        cosines.append(cosine)
        errors.append(error)
    return {"median_cosine": float(np.median(cosines)), "median_normalized_error": float(np.median(errors))}


def _behavior_metrics(prediction: dict[str, np.ndarray], teacher: dict[str, np.ndarray]) -> dict[str, Any]:
    rel = np.linalg.norm(prediction["next_logits"] - teacher["next_logits"], axis=1) / np.maximum(np.linalg.norm(teacher["next_logits"], axis=1), 1e-12)
    weights = np.asarray([max(1, len(row["continuation_ids"])) for row in teacher["rows"]], dtype=np.float64)
    strata = np.asarray([row["stratum"] for row in teacher["rows"]])
    per_stratum = {}
    for stratum in fixtures.STRATA:
        indexes = np.flatnonzero(strata == stratum)
        per_stratum[stratum] = {
            "continuation_cross_entropy_delta_nats_per_token": float(np.average(prediction["ce"][indexes] - teacher["ce"][indexes], weights=weights[indexes])),
            "next_token_normalized_logit_error_max": float(np.max(rel[indexes])),
            "next_token_top1_agreement": float(np.mean(np.argmax(prediction["next_logits"][indexes], axis=1) == np.argmax(teacher["next_logits"][indexes], axis=1))),
        }
    return {
        "continuation_cross_entropy_delta_nats_per_token": float(np.average(prediction["ce"] - teacher["ce"], weights=weights)),
        "next_token_normalized_logit_error_max": float(np.max(rel)),
        "next_token_top1_agreement": float(np.mean(np.argmax(prediction["next_logits"], axis=1) == np.argmax(teacher["next_logits"], axis=1))),
        "per_stratum": per_stratum,
    }


def _passes_effect(row: dict[str, Any]) -> bool:
    return row.get("resolvable_count", 0) >= fixtures.CHALLENGES_PER_STRATUM and row.get("direction_accuracy", 0.0) >= 0.80 and row.get("median_cosine", -1.0) >= 0.90 and row.get("median_normalized_error", float("inf")) <= 0.20


def _passes_behavior(row: dict[str, Any]) -> bool:
    return row.get("continuation_cross_entropy_delta_nats_per_token", float("inf")) <= 0.05 and row.get("next_token_normalized_logit_error_max", float("inf")) <= 0.05


def score(custody_path: str | Path) -> dict[str, Any]:
    custody = _custody_root(custody_path)
    if (custody / "outcome-report.json").exists() or (custody / "outcomes").exists() or _phase_started(custody, "score"):
        raise FileExistsError("v10 scoring phase is immutable once started; preserve it and use a new custody root")
    runtime = _verify_runtime()
    started = time.monotonic()
    _begin(custody, "score")
    try:
        freeze, selection = _candidate_freeze(custody)
        _verify_candidates(custody, freeze)
        challenge_manifest = _read_json(custody / "challenge-manifest.json")
        challenge_material = {key: value for key, value in challenge_manifest.items() if key not in {"challenge_manifest_sha256", "created_at_utc"}}
        if challenge_manifest.get("challenge_manifest_sha256") != _canonical_digest(challenge_material):
            raise ValueError("challenge manifest digest verification failed")
        if challenge_manifest.get("candidate_set_sha256") != freeze["candidate_set_sha256"] or challenge_manifest.get("challenge_generation_started_after_candidate_freeze") is not True:
            raise ValueError("challenge set is not bound to the frozen candidate set")
        for item in challenge_manifest["challenge_rows"]:
            seed_index = int(item["seed_index"])
            prompt_path = custody / "challenges" / f"seed-{seed_index}" / "prompts.json"
            if _sha_file(prompt_path) != item["prompt_sha256"]:
                raise ValueError("challenge prompt bytes changed after prediction sealing")
            for intervention in item["interventions"]:
                intervention_path = custody / "challenges" / f"seed-{seed_index}" / f"layer-{intervention['layer']}-interventions.npz"
                if _sha_file(intervention_path) != intervention["sha256"]:
                    raise ValueError("challenge intervention bytes changed after prediction sealing")
        prediction_manifest = _read_json(custody / "prediction-manifest.json")
        prediction_material = {key: value for key, value in prediction_manifest.items() if key not in {"prediction_manifest_sha256", "created_at_utc"}}
        if prediction_manifest.get("prediction_manifest_sha256") != _canonical_digest(prediction_material):
            raise ValueError("prediction manifest digest verification failed")
        if prediction_manifest.get("original_model_outcomes_opened") is not False or prediction_manifest.get("candidate_set_sha256") != freeze["candidate_set_sha256"]:
            raise ValueError("predictions were not sealed before source outcomes")
        for row in prediction_manifest["predictions"]:
            path = custody / row["path"]
            if not path.is_file() or _sha_file(path) != row["sha256"]:
                raise ValueError("candidate prediction bytes failed digest verification")

        checkpoint, _ = _load_checkpoint(custody)
        outcome_rows = []
        control_results = []
        prediction_lookup = {(int(row["seed_index"]), int(row["layer"]), row.get("method", row.get("control"))): row for row in prediction_manifest["predictions"]}
        for seed_index in range(len(fixtures.SEEDS)):
            challenge_payload = _load_fixtures(custody, f"challenges/seed-{seed_index}/prompts.json")
            records = challenge_payload["records"]
            tokenizer_probe_model, tokenizer = v3._load_local_model(checkpoint)
            tokenized = _token_rows(records, tokenizer, continuation=True)
            for index, row in enumerate(tokenized):
                row["fixture_index"] = index
            del tokenizer_probe_model, tokenizer
            gc.collect()
            for layer in freeze["selected_layers"]:
                intervention_path = custody / "challenges" / f"seed-{seed_index}" / f"layer-{layer}-interventions.npz"
                with np.load(intervention_path, allow_pickle=False) as archive:
                    deltas = archive["deltas"].copy()
                    expected_ids = archive["example_ids"].copy()
                if not np.array_equal(expected_ids, np.asarray([row["example_id"] for row in records], dtype="U24")):
                    raise ValueError("challenge intervention rows do not match prompt order")
                model, tokenizer = v3._load_local_model(checkpoint)
                model.eval()
                prediction_arrays = {}
                rescue_fields = {method: {"next_logits": [], "ce": []} for method in METHODS}
                for method in METHODS:
                    info = prediction_lookup[(seed_index, int(layer), method)]
                    with np.load(custody / info["path"], allow_pickle=False) as archive:
                        prediction_arrays[method] = {key: archive[key].copy() for key in archive.files}
                source_fields: dict[str, list[np.ndarray]] = {name: [] for name in ("boundary_output", "module_effect", "next_logits", "changed_next_logits", "logit_effect", "ce")}
                for group, inputs in v3._batches(tokenized, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                    positions = _selected_positions(group)
                    indexes = [int(row["fixture_index"]) for row in group]
                    local_deltas = torch.as_tensor(deltas[indexes], dtype=torch.float32)
                    base_logits, _source_inputs_all, source_outputs_all = v3._run_projection(model, FAMILY, int(layer), inputs)
                    changed_logits, _x2, changed_outputs = v3._run_projection(model, FAMILY, int(layer), inputs, positions=positions, deltas=local_deltas)
                    if base_logits is None or changed_logits is None or source_outputs_all is None or changed_outputs is None:
                        raise RuntimeError("teacher outcome omitted required source outputs")
                    base_outputs = torch.stack([source_outputs_all[index, position] for index, position in enumerate(positions)])
                    next_logits = _next_logits(base_logits, positions).numpy()
                    changed_next = _next_logits(changed_logits, positions).numpy()
                    source_fields["boundary_output"].append(base_outputs.cpu().numpy())
                    source_fields["module_effect"].append((changed_outputs - base_outputs).cpu().numpy())
                    source_fields["next_logits"].append(next_logits)
                    source_fields["changed_next_logits"].append(changed_next)
                    source_fields["logit_effect"].append(changed_next - next_logits)
                    source_fields["ce"].append(v3._continuation_cross_entropy(base_logits, group))
                    for method in METHODS:
                        candidate = _load_candidate(custody, _candidate_row(freeze, seed_index, int(layer), method))
                        rescued_logits, rescued_ce = _activation_rescue_forward(
                            model, int(layer), group, inputs, candidate, source_outputs_all
                        )
                        rescue_fields[method]["next_logits"].append(rescued_logits)
                        rescue_fields[method]["ce"].append(rescued_ce)
                        del candidate
                teacher = {name: np.concatenate(parts, axis=0) for name, parts in source_fields.items()}
                teacher["rows"] = tokenized
                outcome_path = custody / "outcomes" / f"seed-{seed_index}" / f"layer-{layer}.npz"
                v3._save_npz_deterministic(
                    outcome_path,
                    **{key: value for key, value in teacher.items() if key != "rows"},
                    example_ids=np.asarray([row["example_id"] for row in records], dtype="U24"),
                    strata=np.asarray([row["stratum"] for row in records], dtype="U24"),
                )
                effect_scale = np.maximum(np.linalg.norm(teacher["next_logits"], axis=1), 1e-12)
                for method in METHODS:
                    pred_info = prediction_lookup[(seed_index, int(layer), method)]
                    prediction = prediction_arrays[method]
                    expected_example_ids = np.asarray([row["example_id"] for row in records], dtype="U24")
                    expected_strata = np.asarray([row["stratum"] for row in records], dtype="U24")
                    if not np.array_equal(prediction["example_ids"], expected_example_ids) or not np.array_equal(prediction["strata"], expected_strata):
                        raise ValueError("candidate prediction rows differ from challenge outcomes")
                    per_stratum = {}
                    rescued_logits_all = np.concatenate(rescue_fields[method]["next_logits"], axis=0)
                    rescued_ce_all = np.concatenate(rescue_fields[method]["ce"], axis=0)
                    behavior_all = _behavior_metrics(
                        {"next_logits": prediction["next_logits"], "ce": prediction["ce"]},
                        {"next_logits": teacher["next_logits"], "ce": teacher["ce"], "rows": tokenized},
                    )
                    for stratum in fixtures.STRATA:
                        indexes = np.flatnonzero(expected_strata == stratum)
                        effect = _effect_metrics(prediction["logit_effect"][indexes], teacher["logit_effect"][indexes], effect_scale[indexes])
                        activation = _activation_metrics(prediction["boundary_output"][indexes], teacher["boundary_output"][indexes])
                        behavior = behavior_all["per_stratum"][stratum]
                        rescue = _activation_rescue_metrics(
                            rescued_logits_all[indexes], rescued_ce_all[indexes], teacher["next_logits"][indexes], teacher["ce"][indexes]
                        )
                        per_stratum[stratum] = {
                            "next_logits_causal_effect": effect,
                            "activation_reconstruction": activation,
                            "behavior": behavior,
                            "activation_rescue": rescue,
                        }
                    rescue_pass = all(
                        per_stratum[stratum]["activation_rescue"]["next_token_normalized_logit_error_max"] <= 1e-6
                        and per_stratum[stratum]["activation_rescue"]["continuation_cross_entropy_delta_max_abs_nats_per_token"] <= 1e-6
                        for stratum in fixtures.STRATA
                    )
                    row_pass = all(
                        _passes_effect(per_stratum[stratum]["next_logits_causal_effect"])
                        and per_stratum[stratum]["activation_reconstruction"]["median_cosine"] >= 0.90
                        and per_stratum[stratum]["activation_reconstruction"]["median_normalized_error"] <= 0.20
                        and _passes_behavior(per_stratum[stratum]["behavior"])
                        for stratum in fixtures.STRATA
                    ) and (method != "causal_fit" or rescue_pass)
                    cosines = [
                        float(per_stratum[name]["next_logits_causal_effect"]["median_cosine"] or 0.0)
                        for name in fixtures.STRATA
                    ]
                    outcome_rows.append({
                        "seed_index": seed_index,
                        "seed": fixtures.SEEDS[seed_index],
                        "layer": int(layer),
                        "method": method,
                        "prediction_sha256": pred_info["sha256"],
                        "outcome_sha256": _sha_file(outcome_path),
                        "per_stratum": per_stratum,
                        "mean_stratum_effect_cosine": float(np.mean(cosines)),
                        "activation_rescue_pass": rescue_pass,
                        "retention_pass": row_pass,
                    })

                for control in ("random_replacement", "wrong_layer_replacement", "zero_intervention"):
                    pred_info = prediction_lookup[(seed_index, int(layer), control)]
                    with np.load(custody / pred_info["path"], allow_pickle=False) as archive:
                        control_logits = archive["logit_effect"].copy()
                        control_boundary = archive["boundary_output"].copy()
                    control_row = {
                        "seed_index": seed_index,
                        "layer": int(layer),
                        "control": control,
                        "prediction_sha256": pred_info["sha256"],
                    }
                    if control == "zero_intervention":
                        control_row["max_abs_logit_effect"] = float(np.max(np.abs(control_logits)))
                        control_row["no_op_pass"] = control_row["max_abs_logit_effect"] <= 1e-6
                    else:
                        control_row["median_next_logit_effect_cosine"] = _effect_metrics(control_logits, teacher["logit_effect"], effect_scale)["median_cosine"]
                        control_row["activation_median_cosine"] = _activation_metrics(control_boundary, teacher["boundary_output"])["median_cosine"]
                    control_results.append(control_row)
                del model, tokenizer
                gc.collect()
                _check_budget()

        by_seed = []
        for seed_index in range(len(fixtures.SEEDS)):
            treatment = [row["mean_stratum_effect_cosine"] for row in outcome_rows if row["seed_index"] == seed_index and row["method"] == "causal_fit"]
            comparator = [row["mean_stratum_effect_cosine"] for row in outcome_rows if row["seed_index"] == seed_index and row["method"] == "activation_ridge"]
            if len(treatment) != 2 or len(comparator) != 2:
                raise ValueError("primary comparison is missing a seed/layer result")
            by_seed.append({"seed_index": seed_index, "causal_fit": float(np.mean(treatment)), "activation_ridge": float(np.mean(comparator)), "paired_difference": float(np.mean(treatment) - np.mean(comparator))})
        lower95 = v3_core.bootstrap_mean_difference_lower_bound(
            np.asarray([row["causal_fit"] for row in by_seed]),
            np.asarray([row["activation_ridge"] for row in by_seed]),
            seed=BOOTSTRAP_SEED,
            draws=BOOTSTRAP_DRAWS,
        )
        causal_retention_pass = all(row["retention_pass"] for row in outcome_rows if row["method"] == "causal_fit")
        activation_rescue_pass = all(row["activation_rescue_pass"] for row in outcome_rows if row["method"] == "causal_fit")
        no_op_pass = all(row["no_op_pass"] for row in control_results if row["control"] == "zero_intervention")
        comparative_pass = lower95 > 0.05
        scientific_outcome = "PASS" if causal_retention_pass and activation_rescue_pass and no_op_pass and comparative_pass else "FAIL"
        source_revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_root(), check=True, capture_output=True, text=True).stdout.strip()
        _finish_phase(custody, "score", started, status="COMPLETED", details={"scientific_outcome": scientific_outcome})
        phase_ledger = _read_json(custody / "phase-ledger.json")
        scientific_summary = {
            "scientific_outcome": scientific_outcome,
            "primary_comparison": {
                "cluster_unit": "fixture_seed",
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "mean_paired_difference": float(np.mean([row["paired_difference"] for row in by_seed])),
                "lower_95_bound": lower95,
                "threshold": 0.05,
                "pass": comparative_pass,
                "per_seed": by_seed,
            },
            "causal_fit_retention_pass": causal_retention_pass,
            "activation_rescue_pass": activation_rescue_pass,
            "zero_intervention_control_pass": no_op_pass,
            "results": outcome_rows,
            "controls": control_results,
        }
        payload = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "protocol_slice": PROTOCOL_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "source_revision": source_revision,
            "packet_sha256": verify_packet()["packet_sha256"],
            "candidate_set_sha256": freeze["candidate_set_sha256"],
            "checkpoint": {
                "model_id": selection["model_id"],
                "revision": selection["revision"],
                "weights_sha256": selection["weights_sha256"],
                "weight_bytes": selection["total_weight_bytes"],
                "license_id": selection["license_id"],
                "license_source_url": selection["license_source_url"],
                "custody": "external_owner_custody",
            },
            "runtime": runtime,
            "phase_ledger": phase_ledger,
            "scientific_outcome": scientific_outcome,
            "assurance_level": "INTERNAL_PASS",
            "reviewer_status": "NOT_SUBMITTED",
            "reproducibility_status": "NOT_REPRODUCED",
            "data_and_artifact_custody": "synthetic prompts, checkpoint bytes, candidates, interventions, predictions, and row-level outcomes remain in external owner custody",
            **scientific_summary,
            "known_limitations": [
                "one OPT/ReLU checkpoint and three synthetic fixture seeds only",
                "two visible-selected projections tested independently, with no cumulative replacement chain",
                "three seed clusters provide limited uncertainty resolution",
                "same-owner local execution is not independent scientific acceptance",
            ],
            "claim_ceiling": "LOCAL_CAUSAL_LOSS_IMPROVEMENT_FOR_TWO_VISIBLE_SELECTED_OPT_PROJECTIONS_ON_ONE_CHECKPOINT_AND_THREE_SYNTHETIC_SEEDS",
            "next_blocked_authority_gate": "blind external replication is required for independent science; separate human-calibrated alignment, consciousness, and production gates remain closed",
            "resource_cost": {"monetary": "NOT_APPLICABLE", "energy": "NOT_MEASURED", "paid_or_remote_compute": False},
            "created_at_utc": _now(),
        }
        payload["scientific_results_sha256"] = _canonical_digest(scientific_summary)
        payload["outcome_report_sha256"] = _canonical_digest({key: value for key, value in payload.items() if key not in {"outcome_report_sha256", "created_at_utc"}})
        _write_json(custody / "outcome-report.json", payload, immutable=True)
        return {"scientific_outcome": scientific_outcome, "assurance_level": "INTERNAL_PASS", "outcome_report_sha256": payload["outcome_report_sha256"], "comparative_lower_95": lower95, "causal_fit_retention_pass": causal_retention_pass, "activation_rescue_pass": activation_rescue_pass}
    except BaseException as error:
        _finish_phase(custody, "score", started, status="STOPPED", details={"error_type": type(error).__name__})
        _record_stop(custody, "score", error)
        raise


def _verify_completed_artifacts(custody: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    challenge = _read_json(custody / "challenge-manifest.json")
    challenge_material = {key: value for key, value in challenge.items() if key not in {"challenge_manifest_sha256", "created_at_utc"}}
    if challenge.get("challenge_manifest_sha256") != _canonical_digest(challenge_material):
        raise ValueError("replication challenge manifest digest is invalid")
    if challenge.get("candidate_set_sha256") != freeze["candidate_set_sha256"] or challenge.get("challenge_generation_started_after_candidate_freeze") is not True:
        raise ValueError("replication challenge manifest has the wrong candidate binding")
    for item in challenge["entropy"]:
        path = custody / "challenge-entropy" / f"seed-{item['seed_index']}.bin"
        if not path.is_file() or _sha_file(path) != item["entropy_sha256"]:
            raise ValueError("replication challenge entropy digest is invalid")
    for item in challenge["challenge_rows"]:
        seed_index = int(item["seed_index"])
        prompt_path = custody / "challenges" / f"seed-{seed_index}" / "prompts.json"
        if not prompt_path.is_file() or _sha_file(prompt_path) != item["prompt_sha256"]:
            raise ValueError("replication challenge prompt digest is invalid")
        for intervention in item["interventions"]:
            path = custody / "challenges" / f"seed-{seed_index}" / f"layer-{intervention['layer']}-interventions.npz"
            if not path.is_file() or _sha_file(path) != intervention["sha256"]:
                raise ValueError("replication intervention digest is invalid")
    predictions = _read_json(custody / "prediction-manifest.json")
    prediction_material = {key: value for key, value in predictions.items() if key not in {"prediction_manifest_sha256", "created_at_utc"}}
    if predictions.get("prediction_manifest_sha256") != _canonical_digest(prediction_material):
        raise ValueError("replication prediction manifest digest is invalid")
    if predictions.get("original_model_outcomes_opened") is not False or predictions.get("candidate_set_sha256") != freeze["candidate_set_sha256"]:
        raise ValueError("replication predictions were not sealed before guide outcomes")
    for item in predictions["predictions"]:
        path = custody / item["path"]
        if not path.is_file() or _sha_file(path) != item["sha256"]:
            raise ValueError("replication prediction bytes are invalid")
    report = _read_json(custody / "outcome-report.json")
    scientific_keys = (
        "scientific_outcome",
        "primary_comparison",
        "causal_fit_retention_pass",
        "activation_rescue_pass",
        "zero_intervention_control_pass",
        "results",
        "controls",
    )
    scientific_summary = {key: report[key] for key in scientific_keys}
    if report.get("scientific_results_sha256") != _canonical_digest(scientific_summary):
        raise ValueError("replication scientific summary digest is invalid")
    report_material = {key: value for key, value in report.items() if key not in {"outcome_report_sha256", "created_at_utc"}}
    if report.get("outcome_report_sha256") != _canonical_digest(report_material):
        raise ValueError("replication outcome report digest is invalid")
    if report.get("candidate_set_sha256") != freeze["candidate_set_sha256"] or report.get("packet_sha256") != verify_packet()["packet_sha256"]:
        raise ValueError("replication outcome report is bound to a different frozen target")
    return {
        "challenge_manifest_sha256": challenge["challenge_manifest_sha256"],
        "prediction_manifest_sha256": predictions["prediction_manifest_sha256"],
        "scientific_results_sha256": report["scientific_results_sha256"],
        "scientific_outcome": report["scientific_outcome"],
        "runtime": report["runtime"],
    }


def compare_replication(primary_path: str | Path, replica_path: str | Path) -> dict[str, Any]:
    primary = _custody_root(primary_path)
    replica = _custody_root(replica_path)
    primary_freeze, primary_selection = _candidate_freeze(primary)
    replica_freeze, replica_selection = _candidate_freeze(replica)
    _verify_candidates(primary, primary_freeze)
    _verify_candidates(replica, replica_freeze)
    primary_artifacts = _verify_completed_artifacts(primary, primary_freeze)
    replica_artifacts = _verify_completed_artifacts(replica, replica_freeze)
    fields = {
        "candidate_set_sha256": (primary_freeze["candidate_set_sha256"], replica_freeze["candidate_set_sha256"]),
        "checkpoint_identity_sha256": (primary_freeze["checkpoint_identity_sha256"], replica_freeze["checkpoint_identity_sha256"]),
        "checkpoint_selection_sha256": (primary_selection["selection_sha256"], replica_selection["selection_sha256"]),
        "challenge_manifest_sha256": (primary_artifacts["challenge_manifest_sha256"], replica_artifacts["challenge_manifest_sha256"]),
        "prediction_manifest_sha256": (primary_artifacts["prediction_manifest_sha256"], replica_artifacts["prediction_manifest_sha256"]),
        "scientific_results_sha256": (primary_artifacts["scientific_results_sha256"], replica_artifacts["scientific_results_sha256"]),
    }
    pins_match = primary_artifacts["runtime"]["packages"] == replica_artifacts["runtime"]["packages"]
    toolchains_match = (
        primary_artifacts["runtime"]["python"] == replica_artifacts["runtime"]["python"]
        and primary_artifacts["runtime"]["platform"] == replica_artifacts["runtime"]["platform"]
        and primary_artifacts["runtime"]["dependency_lock_sha256"] == replica_artifacts["runtime"]["dependency_lock_sha256"]
    )
    separate_python_environment = primary_artifacts["runtime"]["python_executable"] != replica_artifacts["runtime"]["python_executable"]
    same_owner = primary.stat().st_uid == replica.stat().st_uid
    same_host = primary_artifacts["runtime"]["host_id_sha256"] == replica_artifacts["runtime"]["host_id_sha256"]
    identity_and_outputs_match = all(left == right for left, right in fields.values())
    same = identity_and_outputs_match and pins_match and toolchains_match and separate_python_environment and same_owner
    result = {
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_level": "REPRODUCED_LOCAL" if same else "INCONCLUSIVE",
        "same_owner": same_owner,
        "same_host": same_host,
        "separate_python_environment": separate_python_environment,
        "dependency_pins_match": pins_match,
        "runtime_platform_and_lock_match": toolchains_match,
        "exact_identity_and_output_digests_match": same,
        "digest_comparisons": fields,
        "claim_boundary": "same-owner local reproduction is not independent scientific acceptance",
        "created_at_utc": _now(),
    }
    _write_json(primary / "replication-report.json", result, immutable=True)
    return result


def run_local(custody_path: str | Path, *, entropy_source: str | Path | None = None) -> dict[str, Any]:
    prepare_result = prepare(custody_path)
    prediction_result = predict(custody_path, entropy_source=entropy_source)
    outcome_result = score(custody_path)
    return {"prepare": prepare_result, "predict": prediction_result, "score": outcome_result}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("freeze")
    select_parser = commands.add_parser("select-checkpoint")
    select_parser.add_argument("--custody", required=True)
    select_parser.add_argument("--model-id", required=True)
    select_parser.add_argument("--revision", required=True)
    select_parser.add_argument("--license-id", required=True)
    select_parser.add_argument("--license-url", required=True)
    select_parser.add_argument("--license-checked-at-utc", required=True)
    for name in ("prepare", "predict", "score", "run-local"):
        command = commands.add_parser(name)
        command.add_argument("--custody", required=True)
        if name in {"predict", "run-local"}:
            command.add_argument("--entropy-source")
    compare = commands.add_parser("compare-replication")
    compare.add_argument("--primary", required=True)
    compare.add_argument("--replica", required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_packet()
    elif args.command == "select-checkpoint":
        result = select_checkpoint(
            args.custody,
            args.model_id,
            args.revision,
            license_id=args.license_id,
            license_url=args.license_url,
            license_checked_at_utc=args.license_checked_at_utc,
        )
    elif args.command == "prepare":
        result = prepare(args.custody)
    elif args.command == "predict":
        result = predict(args.custody, entropy_source=args.entropy_source)
    elif args.command == "score":
        result = score(args.custody)
    elif args.command == "run-local":
        result = run_local(args.custody, entropy_source=args.entropy_source)
    else:
        result = compare_replication(args.primary, args.replica)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
