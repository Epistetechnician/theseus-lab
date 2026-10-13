"""Custody-bound runner for the replacement-loss-v9 causal ladder.

State slice: theseus-lab-sandbox-v1. Model and prompt data remain outside the
repository. Standard output is restricted to aggregate receipts and digests.
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
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

import replacement_loss_v3 as v3_core
import replacement_loss_v3_runner as v3
import replacement_loss_v8_runner as v8
import replacement_loss_v9_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
PROTOCOL_ID = fixtures.PROTOCOL_ID
FAMILIES = ("opt_relu", "gpt2_gelu", "llama_silu_gated")
METHODS = ("ridge", "cka", "logit_distill")
CONTROLS = ("random", "wrong_module", "no_op")
DEPTH_THIRDS = 3
MAX_PARAMETER_COUNT = 500_000_000
MAX_SEQUENCE_LENGTH = 128
TOKEN_BATCH_SIZE = 8
CKA_BATCH_SIZE = 128
CKA_UPDATES = 100
CKA_LEARNING_RATE = 1e-3
KD_UPDATES = 32
KD_LEARNING_RATE = 1e-3
RIDGE = 1e-6
MAX_CASE_SECONDS = 3_600
MAX_TOTAL_SECONDS = 18 * 60 * 60
MAX_RSS_BYTES = 24 * 1024**3
MIN_ADAPTATION_ROWS = 512
PRIOR_CHECKPOINT_IDS = set(v8.PRIOR_CHECKPOINT_IDS) | {
    # v8 selected these after its own protocol freeze; v9 excludes them too.
    "kanishka/opt-babylm2-rewritten-clean-spacy_no-num-adj-earlystop-strictest-bpe_seed-211_1e-3",
    "ai-forever/rugpt3small_based_on_gpt2",
    "amd/amd-llama-135m",
}
PACKET_FILES = tuple(dict.fromkeys((
    "docs/replacement-loss-v9.md",
    "docs/replacement-loss-v9-development.md",
    "replacement_loss_v9_fixtures.py",
    "replacement_loss_v9_runner.py",
    "replacement_loss_v9_replicator.py",
    "tests/test_replacement_loss_v9.py",
    "requirements.txt",
    "requirements-v9.lock.txt",
    *v3.PACKET_FILES,
    *v8.V8_PACKET_FILES,
    v3.PACKET_FREEZE,
    "docs/replacement-loss-v8-freeze.json",
)))
DEPENDENCY_PINS = {
    "huggingface-hub": "0.36.2",
    "numpy": "2.4.6",
    "safetensors": "0.8.0",
    "tokenizers": "0.22.2",
    "torch": "2.6.0",
    "transformers": "4.57.3",
}
_ACTIVE_RUN_STARTED: float | None = None
_ACTIVE_CASE_STARTED: float | None = None
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


def _weights_digest(file_rows: list[dict[str, Any]]) -> str:
    weight_rows = [
        row for row in file_rows
        if Path(str(row["path"])).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    if not weight_rows:
        raise ValueError("checkpoint tree contains no supported model weight files")
    return _canonical_digest(weight_rows)


def _canonical_digest(value: Any) -> str:
    return _sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _module_digest(module: torch.nn.Module) -> str:
    return v3_core._module_digest(module)


def _candidate_seed(family: str, seed: int, method: str, stage_index: int) -> int:
    material = f"{PROTOCOL_ID}\0{family}\0{seed}\0{method}\0{stage_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _custody_root(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if root == _root() or _root() in root.parents:
        raise ValueError("v9 custody root must be outside the repository")
    return root


def _source_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=_root(), check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _source_identity() -> dict[str, str]:
    rows = []
    for relative in PACKET_FILES:
        path = _root() / relative
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return {
        "git_head": _source_revision(),
        "packet_source_sha256": _canonical_digest(rows),
        "worktree_packet_file_count": str(len(rows)),
    }


def _runtime_receipt() -> dict[str, Any]:
    packages = {}
    for name in DEPENDENCY_PINS:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "MISSING"
    all_distributions = {
        str(distribution.metadata["Name"]).strip().lower().replace("_", "-"): str(distribution.version)
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    lock_path = _root() / "requirements-v9.lock.txt"
    lock_rows = {}
    if lock_path.is_file():
        for line in lock_path.read_text(encoding="utf-8").splitlines():
            value = line.strip()
            if value and not value.startswith("#") and "==" in value:
                name, version = value.split("==", 1)
                lock_rows[name.strip().lower().replace("_", "-")] = version.strip()
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "dependencies": packages,
        "all_installed_distributions": all_distributions,
        "locked_distributions": lock_rows,
        "dependency_manifest_sha256": _canonical_digest(all_distributions),
        "locked_environment_pass": all_distributions == lock_rows,
        "cpu_thread_count": torch.get_num_threads(),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)),
        "process_cpu_seconds": float(time.process_time()),
    }


def _check_runtime_budget() -> None:
    now = time.monotonic()
    if _ACTIVE_RUN_STARTED is not None and now - _ACTIVE_RUN_STARTED > MAX_TOTAL_SECONDS:
        raise TimeoutError(f"v9 total run exceeded {MAX_TOTAL_SECONDS}s during {_ACTIVE_PHASE}")
    if _ACTIVE_CASE_STARTED is not None and now - _ACTIVE_CASE_STARTED > MAX_CASE_SECONDS:
        raise TimeoutError(f"v9 family-seed trajectory exceeded {MAX_CASE_SECONDS}s during {_ACTIVE_PHASE}")
    receipt = _runtime_receipt()
    if receipt["peak_rss_bytes"] > MAX_RSS_BYTES:
        raise MemoryError(f"v9 process exceeded {MAX_RSS_BYTES} bytes peak RSS during {_ACTIVE_PHASE}")


def verify_runtime() -> dict[str, Any]:
    torch.set_num_threads(1)
    receipt = _runtime_receipt()
    if receipt["python"] != "3.12.13":
        raise ValueError(f"v9 requires Python 3.12.13, got {receipt['python']}")
    if receipt["dependencies"] != DEPENDENCY_PINS:
        raise ValueError(f"dependency versions differ from the frozen v9 toolchain: {receipt['dependencies']}")
    if not receipt["locked_environment_pass"]:
        raise ValueError("installed distributions differ from requirements-v9.lock.txt")
    return receipt


def freeze_packet(custody_root: str | Path) -> dict[str, Any]:
    """Freeze all v9 and pinned v3 helper bytes before target selection."""

    root = _custody_root(custody_root)
    v3_freeze = v3.verify_packet()
    local_manifest = _root() / "docs" / "replacement-loss-v9-freeze.json"
    archive_manifest = root / "v9-frozen-packet-manifest.json"
    archive = root / "v9-frozen-packet"
    staging = root / ".v9-frozen-packet-staging"
    external_manifest_staging = root / ".v9-frozen-packet-manifest-staging.json"
    local_manifest_staging = local_manifest.with_name(".replacement-loss-v9-freeze-staging.json")
    if any(path.exists() for path in (local_manifest, archive_manifest, archive, staging, external_manifest_staging, local_manifest_staging)):
        raise FileExistsError("v9 packet freeze already exists; frozen packets are immutable")
    missing = [relative for relative in PACKET_FILES if not (_root() / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"v9 packet files are missing: {', '.join(missing)}")
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    try:
        for relative in PACKET_FILES:
            source = _root() / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            rows.append({"path": relative, "sha256": _sha_file(source), "bytes": source.stat().st_size})
        payload: dict[str, Any] = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "freeze_state": "FROZEN_BEFORE_V9_CHECKPOINT_SELECTION",
            "created_at_utc": _now(),
            "packet_files": rows,
            "pinned_v3_packet_sha256": v3_freeze["packet_sha256"],
            "checkpoint_identities_selected": False,
            "challenge_fixtures_generated": False,
            "model_family_count": len(FAMILIES),
            "seeds": list(fixtures.SEEDS),
            "methods": list(METHODS),
            "replacement_stages": DEPTH_THIRDS,
            "runtime_pins": DEPENDENCY_PINS,
        }
        payload["packet_sha256"] = _canonical_digest(payload)
        _write_json(external_manifest_staging, payload)
        _write_json(local_manifest_staging, payload)
        os.replace(staging, archive)
        os.replace(external_manifest_staging, archive_manifest)
        os.replace(local_manifest_staging, local_manifest)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        for path in (external_manifest_staging, local_manifest_staging):
            if path.exists():
                path.unlink()
        raise
    return {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": payload["packet_sha256"],
        "packet_file_count": len(rows),
        "freeze_state": payload["freeze_state"],
    }


def verify_packet(custody_root: str | Path) -> dict[str, Any]:
    root = _custody_root(custody_root)
    v3_freeze = v3.verify_packet()
    manifest = _read_json(root / "v9-frozen-packet-manifest.json")
    local_manifest = _read_json(_root() / "docs" / "replacement-loss-v9-freeze.json")
    if manifest != local_manifest:
        raise ValueError("local and custodied v9 freeze manifests differ")
    if manifest.get("protocol_id") != PROTOCOL_ID or manifest.get("state_slice") != STATE_SLICE:
        raise ValueError("v9 packet identity mismatch")
    if manifest.get("pinned_v3_packet_sha256") != v3_freeze.get("packet_sha256"):
        raise ValueError("pinned v3 helper packet changed")
    material = {key: value for key, value in manifest.items() if key != "packet_sha256"}
    if manifest.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("v9 packet digest is invalid")
    for item in manifest.get("packet_files", []):
        relative = str(item["path"])
        source = _root() / relative
        archived = root / "v9-frozen-packet" / relative
        if not source.is_file() or not archived.is_file():
            raise FileNotFoundError(f"frozen v9 packet member missing: {relative}")
        if _sha_file(source) != item["sha256"] or _sha_file(archived) != item["sha256"]:
            raise ValueError(f"frozen v9 packet member changed: {relative}")
    return {"packet_sha256": manifest["packet_sha256"], "verified": True, "packet_file_count": len(manifest["packet_files"])}


def write_checkpoint_manifest(
    custody_root: str | Path,
    checkpoint_specs: list[dict[str, Any]],
    prior_checkpoint_identities: list[dict[str, str]],
) -> dict[str, Any]:
    """Attest post-freeze target identities and their custody digests."""

    root = _custody_root(custody_root)
    packet = verify_packet(root)
    if any((root / name).exists() for name in ("v9-checkpoint-manifest.json", "prior-checkpoint-identity-inventory.json")):
        raise FileExistsError("v9 checkpoint selection is immutable")
    if len(checkpoint_specs) != len(FAMILIES) or {str(row.get("family")) for row in checkpoint_specs} != set(FAMILIES):
        raise ValueError("select exactly one checkpoint for each frozen family")
    if not prior_checkpoint_identities:
        raise ValueError("freshness requires an inventory of prior custodied checkpoint identities")
    if any(not str(row.get("model_id", "")).strip() or not str(row.get("revision", "")).strip() for row in prior_checkpoint_identities):
        raise ValueError("prior checkpoint inventory entries require model IDs and revisions")
    prior = PRIOR_CHECKPOINT_IDS | {
        str(row.get("model_id", "")).strip().lower() for row in prior_checkpoint_identities
    }
    prior_revisions = {
        (str(row.get("model_id", "")).strip().lower(), str(row.get("revision", "")).lower())
        for row in prior_checkpoint_identities
    }
    prior_weight_digests = {
        str(row.get("weights_sha256", "")).lower()
        for row in prior_checkpoint_identities
        if row.get("weights_sha256")
    }
    entries = []
    for row in checkpoint_specs:
        family = str(row["family"])
        model_id = str(row["model_id"]).strip()
        revision = str(row["revision"]).lower()
        license_id = str(row["license"]).lower()
        relative = Path(str(row["local_path"]))
        if model_id.lower() in prior or (model_id.lower(), revision) in prior_revisions:
            raise ValueError(f"checkpoint was used in a prior Theseus-lab experiment: {family}")
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError("checkpoint revision must be a full 40-character commit SHA")
        if license_id not in {"mit", "apache-2.0"}:
            raise ValueError("checkpoint license must be MIT or Apache-2.0")
        if row.get("public_checkpoint") is not True or row.get("selected_after_protocol_freeze") is not True:
            raise ValueError("checkpoint must be public and selected after packet freeze")
        if row.get("not_used_for_protocol_design") is not True or row.get("instruction_tuned") is not False:
            raise ValueError("checkpoint must be unseen during design and not instruction-tuned")
        if not row.get("license_evidence_url") or not row.get("freshness_attestation"):
            raise ValueError("checkpoint license and freshness attestations are required")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("checkpoint path must be relative to the external custody root")
        local = (root / relative).resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"custodied checkpoint tree is missing for {family}")
        try:
            local.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("checkpoint path escapes v9 custody") from error
        try:
            local.relative_to(_root())
        except ValueError:
            pass
        else:
            raise ValueError("model assets may not be stored in this repository")
        if any(path.is_symlink() for path in local.rglob("*")):
            raise ValueError("checkpoint custody tree must not contain symlinks")
        tree_sha, file_rows = v3._tree_digest(local)
        weights_sha = _weights_digest(file_rows)
        if weights_sha in prior_weight_digests:
            raise ValueError(f"checkpoint weights exactly match a prior Theseus-lab target: {family}")
        entries.append({
            "family": family,
            "model_id": model_id,
            "revision": revision,
            "local_path": relative.as_posix(),
            "license": license_id,
            "license_evidence_url": str(row["license_evidence_url"]),
            "public_checkpoint": True,
            "instruction_tuned": False,
            "not_used_for_protocol_design": True,
            "selected_after_protocol_freeze": True,
            "freshness_attestation": str(row["freshness_attestation"]),
            "artifact_tree_sha256": tree_sha,
            "artifact_file_count": len(file_rows),
            "weights_sha256": weights_sha,
            "weight_file_count": sum(Path(str(item["path"])).suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"} for item in file_rows),
        })
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "selected_at_utc": _now(),
        "prior_checkpoint_identity_manifest_sha256": _canonical_digest(prior_checkpoint_identities),
        "prior_checkpoint_identity_count": len(prior_checkpoint_identities),
        "checkpoints": sorted(entries, key=lambda item: FAMILIES.index(item["family"])),
    }
    identity_inventory = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "known_prior_checkpoint_identities": sorted(
            [
                {"model_id": str(row.get("model_id", "")), "revision": str(row.get("revision", ""))}
                for row in prior_checkpoint_identities
            ],
            key=lambda row: (row["model_id"].lower(), row["revision"].lower()),
        ),
        "static_exclusion_count": len(PRIOR_CHECKPOINT_IDS),
        "created_at_utc": _now(),
    }
    identity_inventory["inventory_sha256"] = _canonical_digest(identity_inventory)
    _write_json(root / "prior-checkpoint-identity-inventory.json", identity_inventory)
    payload["prior_checkpoint_identity_inventory_sha256"] = _sha_file(root / "prior-checkpoint-identity-inventory.json")
    payload["checkpoint_manifest_sha256"] = _canonical_digest(payload)
    _write_json(root / "v9-checkpoint-manifest.json", payload)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
        "checkpoint_count": len(entries),
    }


def _load_checkpoint_specs(custody: Path, packet_sha: str) -> list[dict[str, Any]]:
    path = custody / "v9-checkpoint-manifest.json"
    payload = _read_json(path)
    if payload.get("packet_sha256") != packet_sha or payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("checkpoint manifest does not bind the frozen v9 packet")
    digest = payload.get("checkpoint_manifest_sha256")
    material = {key: value for key, value in payload.items() if key != "checkpoint_manifest_sha256"}
    if digest != _canonical_digest(material):
        raise ValueError("checkpoint manifest digest mismatch")
    inventory_path = custody / "prior-checkpoint-identity-inventory.json"
    inventory = _read_json(inventory_path)
    if _sha_file(inventory_path) != payload.get("prior_checkpoint_identity_inventory_sha256"):
        raise ValueError("checkpoint manifest does not bind the prior-identity inventory")
    inventory_material = {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    if inventory.get("inventory_sha256") != _canonical_digest(inventory_material):
        raise ValueError("prior-identity inventory digest is invalid")
    if len(inventory.get("known_prior_checkpoint_identities", [])) != payload.get("prior_checkpoint_identity_count"):
        raise ValueError("prior-identity inventory count differs from checkpoint manifest")
    selected = datetime.fromisoformat(str(payload["selected_at_utc"]))
    frozen = datetime.fromisoformat(str(_read_json(_root() / "docs/replacement-loss-v9-freeze.json")["created_at_utc"]))
    if selected.tzinfo is None or frozen.tzinfo is None or selected <= frozen:
        raise ValueError("v9 checkpoints must be selected after packet freeze")
    entries = payload.get("checkpoints", [])
    if len(entries) != len(FAMILIES) or {row.get("family") for row in entries} != set(FAMILIES):
        raise ValueError("checkpoint manifest must cover all three families exactly once")
    for row in entries:
        if str(row.get("model_id", "")).strip().lower() in PRIOR_CHECKPOINT_IDS:
            raise ValueError(f"checkpoint is in the frozen prior-use exclusion set: {row.get('family')}")
        revision = str(row.get("revision", "")).lower()
        if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
            raise ValueError("checkpoint manifest has a non-immutable revision")
        if str(row.get("license", "")).lower() not in {"mit", "apache-2.0"}:
            raise ValueError("checkpoint manifest has an ineligible license")
        if not row.get("license_evidence_url") or not row.get("freshness_attestation"):
            raise ValueError("checkpoint manifest is missing license or freshness evidence")
        path = Path(str(row["local_path"]))
        local = (custody / path).resolve()
        if not local.is_dir():
            raise FileNotFoundError(f"custodied checkpoint missing for {row['family']}")
        tree_sha, files = v3._tree_digest(local)
        if tree_sha != row.get("artifact_tree_sha256") or len(files) != row.get("artifact_file_count"):
            raise ValueError(f"checkpoint artifact tree changed for {row['family']}")
        if _weights_digest(files) != row.get("weights_sha256"):
            raise ValueError(f"checkpoint weight digest changed for {row['family']}")
        if any(item.is_symlink() for item in local.rglob("*")):
            raise ValueError("checkpoint custody tree must not contain symlinks")
    return sorted(entries, key=lambda item: FAMILIES.index(item["family"]))


def _record_payload(records: Iterable[Any], fixture_kind: str) -> dict[str, Any]:
    return fixtures.fixture_payload(records, fixture_kind=fixture_kind)


def _fixture_records(custody: Path, relative: str) -> list[dict[str, Any]]:
    payload = _read_json(custody / relative)
    if payload.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("fixture protocol identity mismatch")
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("fixture records missing")
    return records


def _tokenized(records: list[dict[str, Any]], tokenizer: Any, *, continuation: bool) -> list[dict[str, Any]]:
    rows = v3._token_rows(tokenizer, records, with_continuation=continuation)
    for row in rows:
        prompt_len = int(row["prompt_len"])
        if prompt_len > MAX_SEQUENCE_LENGTH:
            raise ValueError("v9 prompt exceeds the frozen 128-token input limit")
        if len(row["ids"]) > MAX_SEQUENCE_LENGTH:
            row["ids"] = row["ids"][:MAX_SEQUENCE_LENGTH]
            row["continuation_ids"] = row["continuation_ids"][: max(0, MAX_SEQUENCE_LENGTH - prompt_len)]
        if continuation and not row["continuation_ids"]:
            raise ValueError("challenge continuation truncated to empty")
    return rows


def _select_layers(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    records: list[dict[str, Any]],
) -> tuple[list[int], list[float]]:
    token_rows = _tokenized(records, tokenizer, continuation=False)
    layer_count = v3._layer_count(model, family)
    if layer_count < DEPTH_THIRDS:
        raise ValueError("checkpoint must contain at least three MLP projections")
    by_layer: list[list[torch.Tensor]] = [[] for _ in range(layer_count)]
    for _group, inputs in v3._batches(token_rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
        _check_runtime_budget()
        activations = v3._capture_all_layer_inputs(model, family, inputs)
        for layer, rows in enumerate(activations):
            by_layer[layer].append(rows.float())
    variances = [float(torch.cat(parts).var(dim=0, unbiased=False).mean().item()) for parts in by_layer]
    chosen = []
    for band in range(DEPTH_THIRDS):
        start = (band * layer_count) // DEPTH_THIRDS
        end = ((band + 1) * layer_count) // DEPTH_THIRDS
        if end <= start:
            raise ValueError("depth thirds do not each contain an MLP projection")
        selected = max(range(start, end), key=lambda layer: (variances[layer], -layer))
        chosen.append(selected)
    if len(set(chosen)) != DEPTH_THIRDS or chosen != sorted(chosen):
        raise ValueError("visible-only selection did not produce three ordered layers")
    return chosen, variances


def _capture_pairs(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    layer: int,
    records: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = _tokenized(records, tokenizer, continuation=False)
    for index, row in enumerate(rows):
        row["fixture_index"] = index
    inputs_all: list[torch.Tensor] = []
    outputs_all: list[torch.Tensor] = []
    next_logits: list[torch.Tensor | None] = [None] * len(rows)
    for group, batch in v3._batches(rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
        _check_runtime_budget()
        logits, inputs, outputs = v3._run_projection(model, family, layer, batch)
        if logits is None or outputs is None:
            raise RuntimeError("adaptation capture omitted logits or projection outputs")
        mask = batch["attention_mask"].bool()
        inputs_all.append(inputs[mask].float().cpu())
        outputs_all.append(outputs[mask].float().cpu())
        for local, row in enumerate(group):
            next_logits[int(row["fixture_index"])] = logits[local, int(row["prompt_len"] - 1)].detach().float().cpu()
    if any(value is None for value in next_logits):
        raise RuntimeError("adaptation logit capture is incomplete")
    x = torch.cat(inputs_all)
    y = torch.cat(outputs_all)
    if len(x) < MIN_ADAPTATION_ROWS:
        raise ValueError(f"adaptation yielded {len(x)} token rows; need {MIN_ADAPTATION_ROWS}")
    return x, y, torch.stack([value for value in next_logits if value is not None])


def _new_linear(in_features: int, out_features: int, use_bias: bool, seed: int) -> torch.nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32, device="cpu")


def _linear_cka(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.float() - left.float().mean(dim=0, keepdim=True)
    right = right.float() - right.float().mean(dim=0, keepdim=True)
    left_gram = left @ left.T
    right_gram = right @ right.T
    denominator = torch.linalg.vector_norm(left_gram) * torch.linalg.vector_norm(right_gram)
    if float(denominator.detach().item()) <= 1e-20:
        return torch.zeros((), dtype=left.dtype, device=left.device)
    return (left_gram * right_gram).sum() / denominator


def _fit_cka(
    in_features: int,
    out_features: int,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    use_bias: bool,
    seed: int,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < CKA_BATCH_SIZE:
        raise ValueError("CKA requires paired rank-2 adaptation matrices with at least 128 rows")
    module = _new_linear(in_features, out_features, use_bias, seed)
    init_digest = _module_digest(module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=CKA_LEARNING_RATE, weight_decay=0.0)
    rng = np.random.default_rng(seed ^ 0xC4A)
    permutation = rng.permutation(len(x))
    final_score = 0.0
    for update in range(CKA_UPDATES):
        _check_runtime_budget()
        start = (update * CKA_BATCH_SIZE) % len(permutation)
        indexes = permutation[(start + np.arange(CKA_BATCH_SIZE)) % len(permutation)]
        batch_index = torch.as_tensor(indexes, dtype=torch.long)
        prediction = module(x[batch_index])
        score = _linear_cka(prediction, y[batch_index])
        loss = 1.0 - score
        if not torch.isfinite(loss):
            raise ValueError("CKA optimization produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_score = float(score.detach().item())
    return module, {
        "method": "centered_linear_cka_adamw",
        "seed": int(seed),
        "updates": CKA_UPDATES,
        "batch_size": CKA_BATCH_SIZE,
        "learning_rate": CKA_LEARNING_RATE,
        "weight_decay": 0.0,
        "initialization_sha256": init_digest,
        "final_adaptation_cka": final_score,
        "fitted_replacement_sha256": _module_digest(module),
    }


def _fit_logit_distill(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    layer: int,
    records: list[dict[str, Any]],
    teacher_logits: torch.Tensor,
    in_features: int,
    out_features: int,
    use_bias: bool,
    seed: int,
) -> tuple[torch.nn.Linear, dict[str, Any]]:
    rows = _tokenized(records, tokenizer, continuation=False)
    for index, row in enumerate(rows):
        row["fixture_index"] = index
    batches = list(v3._batches(rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE))
    module = _new_linear(in_features, out_features, use_bias, seed)
    init_digest = _module_digest(module)
    original = v3._projection(model, family, layer)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    module.requires_grad_(True)
    v3._set_projection(model, family, layer, module)
    optimizer = torch.optim.AdamW(module.parameters(), lr=KD_LEARNING_RATE, weight_decay=0.0)
    try:
        for update in range(KD_UPDATES):
            _check_runtime_budget()
            group, inputs = batches[update % len(batches)]
            indexes = torch.as_tensor([int(row["fixture_index"]) for row in group], dtype=torch.long)
            targets = teacher_logits[indexes]
            logits = model(**inputs, use_cache=False, return_dict=True).logits
            student = torch.stack([logits[i, int(row["prompt_len"] - 1)] for i, row in enumerate(group)])
            loss = F.kl_div(
                F.log_softmax(student.float(), dim=-1),
                F.softmax(targets.float(), dim=-1),
                reduction="batchmean",
            )
            if not torch.isfinite(loss):
                raise ValueError("logit-distillation optimization produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    except Exception:
        v3._set_projection(model, family, layer, original)
        raise
    module.requires_grad_(False)
    return module, {
        "method": "next_token_logit_kl_adamw",
        "seed": int(seed),
        "updates": KD_UPDATES,
        "batch_size": TOKEN_BATCH_SIZE,
        "temperature": 1.0,
        "learning_rate": KD_LEARNING_RATE,
        "weight_decay": 0.0,
        "initialization_sha256": init_digest,
        "fitted_replacement_sha256": _module_digest(module),
    }


def _save_module(path: Path, module: torch.nn.Module) -> str:
    state = {name: tensor.detach().cpu().contiguous() for name, tensor in module.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(path))
    return v3._state_digest(state)


def _load_module(path: Path, in_features: int, out_features: int, use_bias: bool) -> torch.nn.Linear:
    module = torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32)
    module.load_state_dict(load_file(str(path), device="cpu"), strict=True)
    module.eval()
    return module


def _adjacent_layer(layer: int, selected: list[int], count: int) -> int:
    options = [index for index in range(count) if index != layer]
    if not options:
        raise ValueError("wrong-module control needs another projection")
    return min(options, key=lambda index: (abs(index - layer), index))


def _fit_controls(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    layer: int,
    selected: list[int],
    records: list[dict[str, Any]],
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    candidate_dir: Path,
) -> list[dict[str, Any]]:
    target_projection = v3._projection(model, family, layer)
    in_features, out_features, use_bias = v3._module_dimensions(target_projection)
    adjacent = _adjacent_layer(layer, selected, v3._layer_count(model, family))
    wrong_x, wrong_y, _ = _capture_pairs(model, tokenizer, family, adjacent, records)
    wrong, wrong_fit = v3.fit_independent_linear_replacement(
        in_features, out_features, wrong_x, wrong_y, use_bias=use_bias, seed=seed + 811, ridge=RIDGE
    )
    random_module = _new_linear(in_features, out_features, use_bias, seed + 919)
    noop = torch.nn.Linear(in_features, out_features, bias=use_bias, dtype=torch.float32)
    with torch.no_grad():
        noop.weight.zero_()
        if noop.bias is not None:
            noop.bias.zero_()
    modules = {"wrong_module": wrong, "random": random_module, "no_op": noop}
    receipts = []
    for name, module in modules.items():
        path = candidate_dir / f"{name}.safetensors"
        digest = _save_module(path, module)
        receipts.append({
            "method": "ridge_control" if name == "wrong_module" else name,
            "name": name,
            "path": path,
            "sha256": digest,
            "source_layer": adjacent if name == "wrong_module" else None,
            "fit": wrong_fit if name == "wrong_module" else None,
        })
    return receipts


def _existing_fixture_payloads(root: Path) -> list[dict[str, Any]]:
    payloads = []
    for subtree in ("fixtures", "challenges"):
        directory = root / subtree
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                payload = _read_json(path)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                continue
            if isinstance(payload.get("records"), list):
                payloads.append(payload)
    return payloads


def _assert_disjoint_with_prior(
    v9_payloads: list[dict[str, Any]],
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    prior_sets = []
    seen_roots = set()
    for root_value in prior_custody_roots:
        root = Path(root_value).expanduser().resolve()
        if root in seen_roots:
            continue
        seen_roots.add(root)
        prior_sets.extend(_existing_fixture_payloads(root))
    if not prior_sets:
        raise ValueError("v9 requires prior custodied fixture roots for exact-overlap checks")
    fixtures.validate_partition_disjointness(*v9_payloads)
    prior_ids: set[str] = set()
    prior_prompts: set[str] = set()
    prior_contents: set[str] = set()
    prior_unique_records: set[tuple[str, str, str]] = set()
    for payload in prior_sets:
        for row in payload["records"]:
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = prompt + "\0" + str(row["continuation"])
            prior_ids.add(example_id)
            prior_prompts.add(prompt)
            prior_contents.add(content)
            prior_unique_records.add((example_id, prompt, content))
    for payload in v9_payloads:
        for row in payload["records"]:
            example_id = str(row["example_id"])
            prompt = str(row["input_text"])
            content = prompt + "\0" + str(row["continuation"])
            if example_id in prior_ids or prompt in prior_prompts or content in prior_contents:
                raise ValueError("v9 fixtures overlap an earlier custodied fixture")
    def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "protocol_id": payload.get("protocol_id"),
                "fixture_kind": payload.get("fixture_kind"),
                "example_set_sha256": payload.get("example_set_sha256"),
                "count": payload.get("count"),
            }
            for payload in rows
        ]
    return {
        "prior_fixture_payload_count": len(prior_sets),
        "prior_unique_record_count": len(prior_unique_records),
        "v9_fixture_payload_count": len(v9_payloads),
        "prior_fixture_identities": summarize(prior_sets),
        "v9_fixture_identities": summarize(v9_payloads),
        "prior_custody_root_count": len(seen_roots),
        "exact_id_prompt_continuation_overlap": 0,
        "validated_at_utc": _now(),
    }


def prepare_candidates(
    custody_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    started = time.monotonic()
    global _ACTIVE_CASE_STARTED, _ACTIVE_PHASE
    _ACTIVE_PHASE = "candidate_preparation"
    custody = _custody_root(custody_root)
    packet = verify_packet(custody)
    runtime = verify_runtime()
    manifest_entries = _load_checkpoint_specs(custody, packet["packet_sha256"])
    if any((custody / name).exists() for name in ("candidate-preparation.json", "fixtures", "candidates")):
        raise FileExistsError("v9 candidate preparation is immutable; partial preparation is retained")
    custody.mkdir(parents=True, exist_ok=True)

    visible_records_obj = fixtures.generate_visible_fixture()
    visible_payload = _record_payload(visible_records_obj, "visible_selection")
    adaptation_payloads = [
        _record_payload(fixtures.generate_adaptation_fixture(index), f"adaptation_seed_{fixtures.SEEDS[index]}")
        for index in range(len(fixtures.SEEDS))
    ]
    fixtures.validate_partition_disjointness(visible_payload, *adaptation_payloads)
    visible_path = custody / "fixtures/visible-selection.json"
    _write_json(visible_path, visible_payload)
    for index, payload in enumerate(adaptation_payloads):
        _write_json(custody / "fixtures" / f"adaptation-seed-{index}.json", payload)
    disjointness = _assert_disjoint_with_prior([visible_payload, *adaptation_payloads], prior_custody_roots)
    _write_json(custody / "v9-disjointness-preflight.json", disjointness)

    model_rows = []
    selected_layers: dict[str, list[int]] = {}
    visible_records = visible_payload["records"]
    for entry in manifest_entries:
        checkpoint = v3.Checkpoint(
            family=entry["family"],
            model_id=entry["model_id"],
            revision=entry["revision"],
            local_path=(custody / entry["local_path"]).resolve(),
            selected_after_protocol_freeze=True,
        )
        model, tokenizer = v3._load_local_model(checkpoint)
        layers, variances = _select_layers(model, tokenizer, checkpoint.family, visible_records)
        if len(layers) != DEPTH_THIRDS:
            raise ValueError("visible selection failed to choose three projections")
        selected_layers[checkpoint.family] = layers
        model_rows.append({
            "family": checkpoint.family,
            "model_id": checkpoint.model_id,
            "revision": checkpoint.revision,
            "checkpoint_tree_sha256": entry["artifact_tree_sha256"],
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "layer_count": v3._layer_count(model, checkpoint.family),
            "selected_layers": layers,
            "visible_layer_input_variance": variances,
        })
        del model, tokenizer
        gc.collect()

    records_by_seed = [payload["records"] for payload in adaptation_payloads]
    candidate_rows = []
    for entry in manifest_entries:
        family = entry["family"]
        checkpoint = v3.Checkpoint(
            family=family,
            model_id=entry["model_id"],
            revision=entry["revision"],
            local_path=(custody / entry["local_path"]).resolve(),
            selected_after_protocol_freeze=True,
        )
        for seed_index, seed in enumerate(fixtures.SEEDS):
            adaptation_records = records_by_seed[seed_index]
            trajectory_started = time.monotonic()
            _ACTIVE_CASE_STARTED = trajectory_started
            for method_index, method in enumerate(METHODS):
                case_started = time.monotonic()
                model, tokenizer = v3._load_local_model(checkpoint)
                cases_dir = custody / "candidates" / family / f"seed-{seed_index}" / method
                cases_dir.mkdir(parents=True, exist_ok=True)
                method_stages = []
                for stage_index, layer in enumerate(selected_layers[family]):
                    _check_runtime_budget()
                    capture_start = time.monotonic()
                    x, y, teacher_logits = _capture_pairs(
                        model, tokenizer, family, layer, adaptation_records
                    )
                    target = v3._projection(model, family, layer)
                    in_features, out_features, use_bias = v3._module_dimensions(target)
                    method_seed = _candidate_seed(family, seed, method, stage_index)
                    if method == "ridge":
                        candidate, fit_receipt = v3.fit_independent_linear_replacement(
                            in_features, out_features, x, y, use_bias=use_bias,
                            seed=method_seed, ridge=RIDGE,
                        )
                    elif method == "cka":
                        candidate, fit_receipt = _fit_cka(
                            in_features, out_features, x, y, use_bias=use_bias, seed=method_seed
                        )
                    else:
                        candidate, fit_receipt = _fit_logit_distill(
                            model, tokenizer, family, layer, adaptation_records, teacher_logits,
                            in_features, out_features, use_bias, method_seed,
                        )
                    candidate_path = cases_dir / f"stage-{stage_index}.safetensors"
                    candidate_sha = _save_module(candidate_path, candidate)
                    method_stages.append({
                        "stage_index": stage_index,
                        "layer": layer,
                        "in_features": in_features,
                        "out_features": out_features,
                        "use_bias": use_bias,
                        "candidate_path": candidate_path.relative_to(custody).as_posix(),
                        "candidate_sha256": candidate_sha,
                        "candidate_fit": fit_receipt,
                        "adaptation_fixture_sha256": adaptation_payloads[seed_index]["example_set_sha256"],
                        "adaptation_token_row_count": int(len(x)),
                        "teacher_next_logits_sha256": _sha_bytes(teacher_logits.numpy().tobytes()),
                        "fit_and_capture_seconds": time.monotonic() - capture_start,
                    })
                    v3._set_projection(model, family, layer, candidate)
                    if method == "ridge" or stage_index == len(selected_layers[family]) - 1:
                        controls_dir = custody / "candidates" / family / f"seed-{seed_index}" / "controls" / f"stage-{stage_index}"
                        controls = _fit_controls(
                            model, tokenizer, family, layer, selected_layers[family], adaptation_records,
                            x, y, method_seed, controls_dir,
                        ) if method == "ridge" else []
                        method_stages[-1]["controls"] = [
                            {key: value for key, value in item.items() if key != "path"}
                            | {"candidate_path": item["path"].relative_to(custody).as_posix()}
                            for item in controls
                        ]
                candidate_rows.append({
                    "family": family,
                    "seed_index": seed_index,
                    "seed": seed,
                    "method": method,
                    "stages": method_stages,
                    "trajectory_elapsed_seconds": time.monotonic() - case_started,
                })
                del model, tokenizer
                gc.collect()
                if time.monotonic() - trajectory_started > MAX_CASE_SECONDS:
                    raise TimeoutError(f"family-seed trajectory exceeded {MAX_CASE_SECONDS}s")
            _ACTIVE_CASE_STARTED = None

    files = []
    for path in sorted((custody / "candidates").rglob("*.safetensors")):
        files.append({"path": path.relative_to(custody).as_posix(), "sha256": _sha_file(path), "bytes": path.stat().st_size})
    prep = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "checkpoint_manifest_sha256": _sha_file(custody / "v9-checkpoint-manifest.json"),
        "visible_fixture_sha256": visible_payload["example_set_sha256"],
        "adaptation_fixture_sha256": [row["example_set_sha256"] for row in adaptation_payloads],
        "disjointness_receipt_sha256": _sha_file(custody / "v9-disjointness-preflight.json"),
        "layers_by_family": selected_layers,
        "checkpoints": model_rows,
        "trajectories": candidate_rows,
        "candidate_files": files,
        "created_at_utc": _now(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "runtime": runtime,
    }
    prep["candidate_preparation_sha256"] = _canonical_digest(prep)
    _write_json(custody / "candidate-preparation.json", prep)
    return {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": packet["packet_sha256"],
        "trajectory_count": len(candidate_rows),
        "candidate_placement_count": sum(len(row["stages"]) for row in candidate_rows),
        "candidate_file_count": len(files),
        "candidate_preparation_sha256": prep["candidate_preparation_sha256"],
        "elapsed_wall_seconds": prep["elapsed_wall_seconds"],
    }


def freeze_candidates(custody_root: str | Path) -> dict[str, Any]:
    custody = _custody_root(custody_root)
    packet = verify_packet(custody)
    path = custody / "candidate-freeze.json"
    if path.exists():
        raise FileExistsError("v9 candidate freeze is immutable")
    prep_path = custody / "candidate-preparation.json"
    prep = _read_json(prep_path)
    prep_material = {key: value for key, value in prep.items() if key != "candidate_preparation_sha256"}
    if prep.get("candidate_preparation_sha256") != _canonical_digest(prep_material):
        raise ValueError("candidate preparation digest is invalid")
    expected_trajectories = {
        (family, seed_index, method)
        for family in FAMILIES
        for seed_index in range(len(fixtures.SEEDS))
        for method in METHODS
    }
    observed_trajectories = {
        (row.get("family"), int(row.get("seed_index", -1)), row.get("method"))
        for row in prep.get("trajectories", [])
    }
    if prep.get("packet_sha256") != packet["packet_sha256"] or observed_trajectories != expected_trajectories:
        raise ValueError("candidate preparation is incomplete or unbound")
    if prep.get("checkpoint_manifest_sha256") != _sha_file(custody / "v9-checkpoint-manifest.json"):
        raise ValueError("candidate preparation does not bind the selected checkpoint manifest")
    if sum(len(row.get("stages", [])) for row in prep["trajectories"]) != 81:
        raise ValueError("candidate preparation must bind all 81 placements")
    file_rows = prep.get("candidate_files", [])
    for row in file_rows:
        if _sha_file(custody / row["path"]) != row["sha256"]:
            raise ValueError("candidate file changed before freeze")
    prediction_rules = []
    for trajectory in prep["trajectories"]:
        checkpoint = next(row for row in prep["checkpoints"] if row["family"] == trajectory["family"])
        for stage in trajectory["stages"]:
            prediction_rules.append({
                "family": trajectory["family"],
                "seed_index": trajectory["seed_index"],
                "method": trajectory["method"],
                "stage_index": stage["stage_index"],
                "checkpoint_tree_sha256": checkpoint["checkpoint_tree_sha256"],
                "candidate_sha256": stage["candidate_sha256"],
                "predictor": "frozen_candidate_hybrid_forward_effect_map_v1",
                "predictor_code_sha256": packet["packet_sha256"],
                "outputs": ["module_boundary_effect", "next_logit_effect", "candidate_next_logits", "continuation_cross_entropy", "projection_output"],
            })
    if len(prediction_rules) != 81:
        raise ValueError("prediction rules must cover every frozen candidate placement")
    payload = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "candidate_preparation_sha256": _sha_file(prep_path),
        "candidate_files": file_rows,
        "candidate_files_sha256": _canonical_digest(file_rows),
        "candidate_placement_count": sum(len(row["stages"]) for row in prep["trajectories"]),
        "prediction_rules": prediction_rules,
        "prediction_rule_count": len(prediction_rules),
        "predictions_functionally_bound_before_challenge_entropy": True,
        "created_at_utc": _now(),
        "challenge_fixtures_generated": False,
        "original_outcomes_opened": False,
    }
    payload["candidate_freeze_sha256"] = _canonical_digest(payload)
    _write_json(path, payload)
    return {
        "candidate_freeze_sha256": payload["candidate_freeze_sha256"],
        "candidate_files_sha256": payload["candidate_files_sha256"],
        "candidate_placement_count": payload["candidate_placement_count"],
    }


def _verify_candidate_freeze(custody: Path, packet_sha: str) -> dict[str, Any]:
    freeze_path = custody / "candidate-freeze.json"
    freeze = _read_json(freeze_path)
    material = {key: value for key, value in freeze.items() if key != "candidate_freeze_sha256"}
    if freeze.get("candidate_freeze_sha256") != _canonical_digest(material):
        raise ValueError("candidate freeze digest is invalid")
    if freeze.get("packet_sha256") != packet_sha or freeze.get("candidate_placement_count") != 81:
        raise ValueError("candidate freeze is incomplete or not bound to this packet")
    if freeze.get("predictions_functionally_bound_before_challenge_entropy") is not True:
        raise ValueError("candidate prediction functions were not bound before challenge entropy")
    if freeze.get("prediction_rule_count") != 81:
        raise ValueError("candidate prediction rule coverage is incomplete")
    prep_path = custody / "candidate-preparation.json"
    prep = _read_json(prep_path)
    if _sha_file(prep_path) != freeze.get("candidate_preparation_sha256"):
        raise ValueError("candidate preparation changed after candidate freeze")
    if freeze.get("candidate_files") != prep.get("candidate_files"):
        raise ValueError("candidate freeze file inventory differs from candidate preparation")
    for row in freeze.get("candidate_files", []):
        if _sha_file(custody / row["path"]) != row["sha256"]:
            raise ValueError("a frozen candidate or control changed")
    if len(freeze.get("candidate_files", [])) != len(prep.get("candidate_files", [])):
        raise ValueError("candidate freeze file coverage is incomplete")
    return freeze


def _load_preparation(custody: Path, packet_sha: str) -> dict[str, Any]:
    prep = _read_json(custody / "candidate-preparation.json")
    if prep.get("packet_sha256") != packet_sha or len(prep.get("trajectories", [])) != 27:
        raise ValueError("v9 candidate preparation does not bind the frozen packet")
    return prep


def _load_candidate(custody: Path, stage: dict[str, Any]) -> torch.nn.Linear:
    path = custody / stage["candidate_path"]
    if _sha_file(path) != stage["candidate_sha256"]:
        raise ValueError("frozen candidate bytes changed")
    return _load_module(path, int(stage["in_features"]), int(stage["out_features"]), bool(stage["use_bias"]))


def _find_trajectory(prep: dict[str, Any], family: str, seed_index: int, method: str) -> dict[str, Any]:
    matches = [row for row in prep["trajectories"] if row["family"] == family and row["seed_index"] == seed_index and row["method"] == method]
    if len(matches) != 1:
        raise ValueError("v9 trajectory key is not unique")
    return matches[0]


def generate_challenges(
    custody_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    started = time.monotonic()
    global _ACTIVE_PHASE
    _ACTIVE_PHASE = "challenge_generation"
    custody = _custody_root(custody_root)
    packet = verify_packet(custody)
    prep = _load_preparation(custody, packet["packet_sha256"])
    freeze_path = custody / "candidate-freeze.json"
    freeze = _verify_candidate_freeze(custody, packet["packet_sha256"])
    if freeze.get("challenge_fixtures_generated") is not False:
        raise ValueError("candidate freeze is invalid or challenges were already generated")
    if freeze.get("prediction_rules_functionally_bound_before_challenge_entropy") is not True or freeze.get("prediction_rule_count") != 81:
        raise ValueError("candidate prediction functions were not bound before challenge generation")
    if any((custody / name).exists() for name in ("challenge-manifest.json", "challenges", "challenge-secrets")):
        raise FileExistsError("v9 challenge packet is immutable; partial challenge material is retained")

    visible = _read_json(custody / "fixtures/visible-selection.json")
    adaptation_payloads = [
        _read_json(custody / "fixtures" / f"adaptation-seed-{index}.json")
        for index in range(len(fixtures.SEEDS))
    ]
    challenge_payloads = []
    entropy_rows = []
    summary_rows = []
    for seed_index, seed in enumerate(fixtures.SEEDS):
        secret = os.urandom(32)
        secret_path = custody / "challenge-secrets" / f"seed-{seed_index}.bin"
        secret_path.parent.mkdir(parents=True, exist_ok=True)
        secret_path.write_bytes(secret)
        entropy_rows.append({"seed_index": seed_index, "entropy_sha256": _sha_bytes(secret)})
        records = fixtures.generate_challenge_fixture(seed_index, secret_seed=secret)
        payload = _record_payload(records, f"challenge_seed_{seed}")
        fixtures.validate_partition_disjointness(visible, *adaptation_payloads, payload)
        challenge_path = custody / "fixtures" / f"challenge-seed-{seed_index}.json"
        _write_json(challenge_path, payload)
        challenge_payloads.append(payload)

        for checkpoint in prep["checkpoints"]:
            family = checkpoint["family"]
            manifest_entry = next(row for row in _load_checkpoint_specs(custody, packet["packet_sha256"]) if row["family"] == family)
            model = v3.Checkpoint(
                family=family,
                model_id=checkpoint["model_id"],
                revision=checkpoint["revision"],
                local_path=(custody / manifest_entry["local_path"]).resolve(),
                selected_after_protocol_freeze=True,
            )
            loaded, tokenizer = v3._load_local_model(model)
            token_rows = _tokenized(payload["records"], tokenizer, continuation=True)
            layers = prep["layers_by_family"][family]
            stage_deltas = [[] for _ in layers]
            for group, inputs in v3._batches(token_rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                _check_runtime_budget()
                activations = v3._capture_all_layer_inputs(loaded, family, inputs)
                for stage_index, layer in enumerate(layers):
                    for local, row in enumerate(group):
                        position = int(row["prompt_len"] - 1)
                        direction_entropy = secret + family.encode("utf-8") + bytes([stage_index])
                        stage_deltas[stage_index].append(
                            v3._direction(row, activations[layer][local, position].cpu().numpy(), direction_entropy)
                        )
            relative = f"challenges/{family}/seed-{seed_index}"
            challenge_dir = custody / relative
            challenge_dir.mkdir(parents=True, exist_ok=True)
            intervention_arrays = {
                f"stage_{stage_index}": np.stack(rows).astype(np.float32)
                for stage_index, rows in enumerate(stage_deltas)
            }
            intervention_path = challenge_dir / "interventions.npz"
            v3._save_npz_deterministic(intervention_path, **intervention_arrays)
            prompt_path = challenge_dir / "prompts.json"
            _write_json(prompt_path, payload)
            summary_rows.append({
                "family": family,
                "seed_index": seed_index,
                "fixture_sha256": payload["example_set_sha256"],
                "prompt_file_sha256": _sha_file(prompt_path),
                "interventions_file_sha256": _sha_file(intervention_path),
                "record_count": len(payload["records"]),
                "stratum_counts": payload["stratum_counts"],
            })
            del loaded, tokenizer

    v9_disjointness = _assert_disjoint_with_prior([visible, *adaptation_payloads, *challenge_payloads], prior_custody_roots)
    _write_json(custody / "v9-final-disjointness-receipt.json", v9_disjointness)
    challenge_manifest = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "candidate_freeze_sha256": _sha_file(freeze_path),
        "challenge_generation_started_after_candidate_freeze": True,
        "entropy_commitments": entropy_rows,
        "cases": summary_rows,
        "fixture_count": len(challenge_payloads),
        "final_disjointness_receipt_sha256": _sha_file(custody / "v9-final-disjointness-receipt.json"),
        "created_at_utc": _now(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "runtime": _runtime_receipt(),
    }
    challenge_manifest["challenge_manifest_sha256"] = _canonical_digest(challenge_manifest)
    _write_json(custody / "challenge-manifest.json", challenge_manifest)
    return {
        "challenge_case_count": len(summary_rows),
        "prompt_fixture_count": len(challenge_payloads),
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "elapsed_wall_seconds": challenge_manifest["elapsed_wall_seconds"],
    }


def _next_logits(logits: torch.Tensor, group: list[dict[str, Any]]) -> torch.Tensor:
    return torch.stack([logits[index, int(row["prompt_len"] - 1)].detach().cpu() for index, row in enumerate(group)])


def _predict_stage(
    model: torch.nn.Module,
    tokenizer: Any,
    family: str,
    layer: int,
    group: list[dict[str, Any]],
    inputs: dict[str, torch.Tensor],
    deltas: torch.Tensor,
) -> dict[str, np.ndarray]:
    positions = [int(row["prompt_len"] - 1) for row in group]
    base_logits, _, base_outputs = v3._run_projection(model, family, layer, inputs, positions=positions)
    changed_logits, _, changed_outputs = v3._run_projection(
        model, family, layer, inputs, positions=positions, deltas=deltas
    )
    if base_logits is None or changed_logits is None or base_outputs is None or changed_outputs is None:
        raise RuntimeError("v9 candidate prediction omitted required outputs")
    next_logits = _next_logits(base_logits, group).numpy()
    changed_next = _next_logits(changed_logits, group).numpy()
    return {
        "module_effect": changed_outputs.cpu().numpy() - base_outputs.cpu().numpy(),
        "logit_effect": changed_next - next_logits,
        "boundary_output": base_outputs.cpu().numpy(),
        "next_logits": next_logits,
        "ce": v3._continuation_cross_entropy(base_logits, group),
        "changed_next_logits": changed_next,
    }


def seal_predictions(custody_root: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    global _ACTIVE_PHASE
    _ACTIVE_PHASE = "prediction_sealing"
    custody = _custody_root(custody_root)
    packet = verify_packet(custody)
    prep = _load_preparation(custody, packet["packet_sha256"])
    _verify_candidate_freeze(custody, packet["packet_sha256"])
    freeze_path = custody / "candidate-freeze.json"
    challenge_manifest = _read_json(custody / "challenge-manifest.json")
    challenge_material = {key: value for key, value in challenge_manifest.items() if key != "challenge_manifest_sha256"}
    if challenge_manifest.get("challenge_manifest_sha256") != _canonical_digest(challenge_material):
        raise ValueError("v9 challenge manifest digest mismatch")
    if challenge_manifest.get("candidate_freeze_sha256") != _sha_file(freeze_path):
        raise ValueError("v9 challenges are not bound to the frozen candidate set")
    if challenge_manifest.get("challenge_generation_started_after_candidate_freeze") is not True:
        raise ValueError("v9 challenge generation preceded candidate freeze")
    expected_challenge_keys = {
        (family, seed_index)
        for family in FAMILIES
        for seed_index in range(len(fixtures.SEEDS))
    }
    challenge_cases = challenge_manifest.get("cases", [])
    if {(row.get("family"), int(row.get("seed_index", -1))) for row in challenge_cases} != expected_challenge_keys:
        raise ValueError("v9 challenge manifest does not cover all nine family-seed fixtures")
    for row in challenge_cases:
        challenge_dir = custody / "challenges" / row["family"] / f"seed-{row['seed_index']}"
        if _sha_file(challenge_dir / "prompts.json") != row.get("prompt_file_sha256"):
            raise ValueError("challenge prompt fixture changed after its manifest was sealed")
        if _sha_file(challenge_dir / "interventions.npz") != row.get("interventions_file_sha256"):
            raise ValueError("challenge intervention fixture changed after its manifest was sealed")
    if any((custody / name).exists() for name in ("prediction-manifest.json", "predictions")):
        raise FileExistsError("v9 prediction packet is immutable; partial prediction material is retained")

    checkpoint_entries = _load_checkpoint_specs(custody, packet["packet_sha256"])
    prediction_rows = []
    for trajectory in prep["trajectories"]:
        family, seed_index, method = trajectory["family"], int(trajectory["seed_index"]), trajectory["method"]
        entry = next(row for row in checkpoint_entries if row["family"] == family)
        checkpoint = v3.Checkpoint(
            family=family, model_id=entry["model_id"], revision=entry["revision"],
            local_path=(custody / entry["local_path"]).resolve(), selected_after_protocol_freeze=True,
        )
        model, tokenizer = v3._load_local_model(checkpoint)
        challenge_path = custody / "challenges" / family / f"seed-{seed_index}"
        payload = _read_json(challenge_path / "prompts.json")
        records = _tokenized(payload["records"], tokenizer, continuation=True)
        with np.load(challenge_path / "interventions.npz", allow_pickle=False) as archive:
            stage_deltas = [archive[f"stage_{index}"].copy() for index in range(DEPTH_THIRDS)]
        for stage_index, stage in enumerate(trajectory["stages"]):
            for previous_index in range(stage_index):
                prior_stage = trajectory["stages"][previous_index]
                module = _load_candidate(custody, prior_stage)
                v3._set_projection(model, family, int(prior_stage["layer"]), module)
            candidate = _load_candidate(custody, stage)
            layer = int(stage["layer"])
            v3._set_projection(model, family, layer, candidate)
            fields: dict[str, list[np.ndarray]] = {
                key: [] for key in ("module_effect", "logit_effect", "boundary_output", "next_logits", "ce", "changed_next_logits")
            }
            cursor = 0
            for group, inputs in v3._batches(records, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                _check_runtime_budget()
                count = len(group)
                local_delta = torch.from_numpy(stage_deltas[stage_index][cursor : cursor + count])
                values = _predict_stage(model, tokenizer, family, layer, group, inputs, local_delta)
                for key, value in values.items():
                    fields[key].append(value)
                cursor += count
            arrays = {key: np.concatenate(values, axis=0) for key, values in fields.items()}
            arrays["example_ids"] = np.asarray([row["example_id"] for row in records], dtype="U24")
            arrays["strata"] = np.asarray([row["stratum"] for row in records], dtype="U24")
            relative = f"predictions/{family}/seed-{seed_index}/{method}/stage-{stage_index}.npz"
            prediction_path = custody / relative
            v3._save_npz_deterministic(prediction_path, **arrays)
            prediction_rows.append({
                "family": family,
                "seed_index": seed_index,
                "method": method,
                "stage_index": stage_index,
                "prediction_path": relative,
                "prediction_sha256": _sha_file(prediction_path),
                "candidate_sha256": stage["candidate_sha256"],
                "challenge_fixture_sha256": payload["example_set_sha256"],
                "challenge_count": len(records),
                "original_outcomes_opened": False,
                "sealed_at_utc": _now(),
            })
            # Restore every later original module by reloading before the next stage.
            if stage_index < DEPTH_THIRDS - 1:
                del model
                model, tokenizer = v3._load_local_model(checkpoint)
                for previous_index in range(stage_index + 1):
                    prior_stage = trajectory["stages"][previous_index]
                    module = _load_candidate(custody, prior_stage)
                    v3._set_projection(model, family, int(prior_stage["layer"]), module)
        del model, tokenizer

    control_rows = _seal_control_predictions(custody, prep, checkpoint_entries)
    prediction_manifest = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "candidate_freeze_sha256": _sha_file(freeze_path),
        "challenge_manifest_sha256": challenge_manifest["challenge_manifest_sha256"],
        "predictions": prediction_rows,
        "controls": control_rows,
        "prediction_count": len(prediction_rows),
        "control_prediction_count": len(control_rows),
        "all_prediction_receipts_precede_source_outcomes": True,
        "created_at_utc": _now(),
        "elapsed_wall_seconds": time.monotonic() - started,
        "runtime": _runtime_receipt(),
    }
    prediction_manifest["prediction_manifest_sha256"] = _canonical_digest(prediction_manifest)
    _write_json(custody / "prediction-manifest.json", prediction_manifest)
    return {
        "prediction_count": len(prediction_rows),
        "control_prediction_count": len(control_rows),
        "prediction_manifest_sha256": prediction_manifest["prediction_manifest_sha256"],
        "elapsed_wall_seconds": prediction_manifest["elapsed_wall_seconds"],
    }


def _seal_control_predictions(
    custody: Path,
    prep: dict[str, Any],
    checkpoint_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows_out = []
    for family in FAMILIES:
        entry = next(row for row in checkpoint_entries if row["family"] == family)
        for seed_index in range(len(fixtures.SEEDS)):
            trajectory = _find_trajectory(prep, family, seed_index, "ridge")
            checkpoint = v3.Checkpoint(
                family=family, model_id=entry["model_id"], revision=entry["revision"],
                local_path=(custody / entry["local_path"]).resolve(), selected_after_protocol_freeze=True,
            )
            payload = _read_json(custody / "challenges" / family / f"seed-{seed_index}" / "prompts.json")
            model, tokenizer = v3._load_local_model(checkpoint)
            records = _tokenized(payload["records"], tokenizer, continuation=True)
            with np.load(custody / "challenges" / family / f"seed-{seed_index}" / "interventions.npz", allow_pickle=False) as archive:
                stage_deltas = [archive[f"stage_{index}"].copy() for index in range(DEPTH_THIRDS)]
            for stage_index, stage in enumerate(trajectory["stages"]):
                for previous_index in range(stage_index):
                    previous = trajectory["stages"][previous_index]
                    v3._set_projection(model, family, int(previous["layer"]), _load_candidate(custody, previous))
                for control in stage.get("controls", []):
                    name = control["name"]
                    candidate_path = custody / control["candidate_path"]
                    candidate = _load_module(candidate_path, int(stage["in_features"]), int(stage["out_features"]), bool(stage["use_bias"]))
                    v3._set_projection(model, family, int(stage["layer"]), candidate)
                    fields = {key: [] for key in ("module_effect", "logit_effect", "boundary_output", "next_logits", "ce", "changed_next_logits")}
                    cursor = 0
                    for group, inputs in v3._batches(records, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                        _check_runtime_budget()
                        count = len(group)
                        values = _predict_stage(
                            model, tokenizer, family, int(stage["layer"]), group, inputs,
                            torch.from_numpy(stage_deltas[stage_index][cursor : cursor + count]),
                        )
                        for key, value in values.items():
                            fields[key].append(value)
                        cursor += count
                    arrays = {key: np.concatenate(values, axis=0) for key, values in fields.items()}
                    arrays["example_ids"] = np.asarray([row["example_id"] for row in records], dtype="U24")
                    arrays["strata"] = np.asarray([row["stratum"] for row in records], dtype="U24")
                    relative = f"predictions/{family}/seed-{seed_index}/controls/{name}/stage-{stage_index}.npz"
                    path = custody / relative
                    v3._save_npz_deterministic(path, **arrays)
                    rows_out.append({
                        "family": family, "seed_index": seed_index, "control": name,
                        "stage_index": stage_index, "prediction_path": relative,
                        "prediction_sha256": _sha_file(path), "original_outcomes_opened": False,
                    })
                    # A fresh model avoids leaving the control module in the path.
                    del model
                    model, tokenizer = v3._load_local_model(checkpoint)
                    for previous_index in range(stage_index):
                        previous = trajectory["stages"][previous_index]
                        v3._set_projection(model, family, int(previous["layer"]), _load_candidate(custody, previous))
            del model, tokenizer
    return rows_out


def _summary_effect(predicted: np.ndarray, observed: np.ndarray, scale: np.ndarray) -> dict[str, Any]:
    predicted = np.asarray(predicted)
    observed = np.asarray(observed)
    scale = np.asarray(scale, dtype=np.float64)
    if predicted.shape != observed.shape or predicted.ndim != 2 or scale.shape != (len(predicted),):
        raise ValueError("effect predictions, outcomes, and normalization scales must be paired")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("effect normalization scales must be finite and positive")
    scores = [
        v3.causal_effect_metrics(p, o, resolution_floor=1e-12, scale=float(s))
        for p, o, s in zip(predicted, observed, scale)
    ]
    resolvable = [row for row in scores if row["resolvable"]]
    if not resolvable:
        return {"count": len(scores), "resolvable_count": 0, "direction_accuracy": None, "median_cosine": None, "median_normalized_error": None}
    return {
        "count": len(scores),
        "resolvable_count": len(resolvable),
        "direction_accuracy": float(np.mean([row["direction_correct"] for row in resolvable])),
        "median_cosine": float(np.median([row["cosine_similarity"] for row in resolvable])),
        "median_normalized_error": float(np.median([row["normalized_error"] for row in resolvable])),
    }


def _activation_metrics(predicted: np.ndarray, observed: np.ndarray, strata: np.ndarray) -> dict[str, Any]:
    if predicted.shape != observed.shape or predicted.ndim != 2 or strata.shape != (len(predicted),):
        raise ValueError("activation predictions, outcomes, and strata must be paired")
    if not np.isfinite(predicted).all() or not np.isfinite(observed).all():
        raise ValueError("activation predictions and outcomes must be finite")
    cosines, errors = [], []
    for left, right in zip(predicted, observed):
        cosine = float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-30))
        error = float(np.linalg.norm(left - right) / max(float(np.linalg.norm(right)), 1e-12))
        cosines.append(cosine)
        errors.append(error)
    result = {"median_cosine": float(np.median(cosines)), "median_normalized_error": float(np.median(errors)), "per_stratum": {}}
    for stratum in fixtures.STRATA:
        indexes = np.flatnonzero(strata == stratum)
        result["per_stratum"][stratum] = {
            "median_cosine": float(np.median(np.asarray(cosines)[indexes])),
            "median_normalized_error": float(np.median(np.asarray(errors)[indexes])),
        }
    return result


def _behavior_metrics(candidate_logits: np.ndarray, candidate_ce: np.ndarray, original_logits: np.ndarray, original_ce: np.ndarray, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rel = np.linalg.norm(candidate_logits - original_logits, axis=1) / np.maximum(np.linalg.norm(original_logits, axis=1), 1e-12)
    weights = np.asarray([max(1, len(row["continuation_ids"])) for row in rows], dtype=np.float64)
    top_agreement = np.argmax(candidate_logits, axis=1) == np.argmax(original_logits, axis=1)
    per_stratum = {}
    for stratum in fixtures.STRATA:
        indexes = np.asarray([i for i, row in enumerate(rows) if row["stratum"] == stratum], dtype=int)
        per_stratum[stratum] = {
            "continuation_cross_entropy_delta_nats_per_token": float(np.average(candidate_ce[indexes] - original_ce[indexes], weights=weights[indexes])),
            "next_token_normalized_logit_error_max": float(np.max(rel[indexes])),
            "next_token_top1_agreement": float(np.mean(top_agreement[indexes])),
        }
    return {
        "continuation_cross_entropy_delta_nats_per_token": float(np.average(candidate_ce - original_ce, weights=weights)),
        "next_token_normalized_logit_error_max": float(np.max(rel)),
        "next_token_top1_agreement": float(np.mean(top_agreement)),
        "per_stratum": per_stratum,
    }


def _passes_effect(row: dict[str, Any]) -> bool:
    return (
        row.get("resolvable_count", 0) >= fixtures.CHALLENGES_PER_STRATUM
        and row.get("direction_accuracy", 0.0) >= 0.80
        and row.get("median_cosine", -1.0) >= 0.90
        and row.get("median_normalized_error", float("inf")) <= 0.20
    )


def _passes_behavior(row: dict[str, Any]) -> bool:
    return (
        row.get("continuation_cross_entropy_delta_nats_per_token", float("inf")) <= 0.05
        and row.get("next_token_normalized_logit_error_max", float("inf")) <= 0.05
    )


def evaluate_outcomes(custody_root: str | Path) -> dict[str, Any]:
    started = time.monotonic()
    global _ACTIVE_PHASE
    _ACTIVE_PHASE = "source_outcome_evaluation"
    custody = _custody_root(custody_root)
    packet = verify_packet(custody)
    prep = _load_preparation(custody, packet["packet_sha256"])
    candidate_freeze = _verify_candidate_freeze(custody, packet["packet_sha256"])
    if (custody / "outcomes").exists():
        raise FileExistsError("v9 outcome evaluation is immutable; partial outcome files are retained")
    prediction_path = custody / "prediction-manifest.json"
    prediction_manifest = _read_json(prediction_path)
    pred_material = {key: value for key, value in prediction_manifest.items() if key != "prediction_manifest_sha256"}
    if prediction_manifest.get("prediction_manifest_sha256") != _canonical_digest(pred_material):
        raise ValueError("v9 prediction manifest digest mismatch")
    if prediction_manifest.get("all_prediction_receipts_precede_source_outcomes") is not True:
        raise ValueError("v9 predictions were not sealed before source outcomes")
    if prediction_manifest.get("candidate_freeze_sha256") != _sha_file(custody / "candidate-freeze.json"):
        raise ValueError("v9 predictions do not bind the candidate freeze")
    challenge_manifest = _read_json(custody / "challenge-manifest.json")
    challenge_material = {key: value for key, value in challenge_manifest.items() if key != "challenge_manifest_sha256"}
    if challenge_manifest.get("challenge_manifest_sha256") != _canonical_digest(challenge_material):
        raise ValueError("v9 challenge manifest digest mismatch before outcomes")
    if prediction_manifest.get("challenge_manifest_sha256") != challenge_manifest.get("challenge_manifest_sha256"):
        raise ValueError("v9 predictions do not bind the held-out challenge manifest")
    if len(prediction_manifest.get("predictions", [])) != 81 or len(prediction_manifest.get("controls", [])) != 81:
        raise ValueError("v9 prediction packet must contain 81 method and 81 control predictions")
    for item in (*prediction_manifest["predictions"], *prediction_manifest["controls"]):
        if _sha_file(custody / item["prediction_path"]) != item["prediction_sha256"]:
            raise ValueError("a prediction file changed after its receipt was sealed")
    outcome_path = custody / "outcome-manifest.json"
    if outcome_path.exists():
        raise FileExistsError("v9 outcome evaluation is immutable")
    entries = _load_checkpoint_specs(custody, packet["packet_sha256"])
    results = []
    guide_baselines: dict[tuple[str, int], tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]] = {}

    for entry in entries:
        family = entry["family"]
        checkpoint = v3.Checkpoint(
            family=family, model_id=entry["model_id"], revision=entry["revision"],
            local_path=(custody / entry["local_path"]).resolve(), selected_after_protocol_freeze=True,
        )
        for seed_index in range(len(fixtures.SEEDS)):
            challenge_dir = custody / "challenges" / family / f"seed-{seed_index}"
            payload = _read_json(challenge_dir / "prompts.json")
            model, tokenizer = v3._load_local_model(checkpoint)
            rows = _tokenized(payload["records"], tokenizer, continuation=True)
            original_next_rows, original_ce_rows = [], []
            for group, inputs in v3._batches(rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                _check_runtime_budget()
                with torch.no_grad():
                    logits = model(**inputs, use_cache=False, return_dict=True).logits
                original_next_rows.append(_next_logits(logits, group).numpy())
                original_ce_rows.append(v3._continuation_cross_entropy(logits, group))
            guide_baselines[(family, seed_index)] = (np.concatenate(original_next_rows), np.concatenate(original_ce_rows), rows)
            del model, tokenizer

            for method in METHODS:
                trajectory = _find_trajectory(prep, family, seed_index, method)
                for stage_index, stage in enumerate(trajectory["stages"]):
                    model, tokenizer = v3._load_local_model(checkpoint)
                    for previous_index in range(stage_index):
                        previous = trajectory["stages"][previous_index]
                        v3._set_projection(model, family, int(previous["layer"]), _load_candidate(custody, previous))
                    layer = int(stage["layer"])
                    challenge_rows = _tokenized(payload["records"], tokenizer, continuation=True)
                    with np.load(challenge_dir / "interventions.npz", allow_pickle=False) as archive:
                        deltas = archive[f"stage_{stage_index}"].copy()
                    observed_module, observed_logits = [], []
                    source_outputs, source_next, source_ce = [], [], []
                    cursor = 0
                    for group, inputs in v3._batches(challenge_rows, tokenizer.pad_token_id, batch_size=TOKEN_BATCH_SIZE):
                        _check_runtime_budget()
                        count = len(group)
                        local_deltas = torch.from_numpy(deltas[cursor : cursor + count])
                        base_logits, _, outputs = v3._run_projection(
                            model, family, layer, inputs, positions=[int(row["prompt_len"] - 1) for row in group]
                        )
                        changed_logits, _, changed_outputs = v3._run_projection(
                            model, family, layer, inputs,
                            positions=[int(row["prompt_len"] - 1) for row in group], deltas=local_deltas,
                        )
                        if base_logits is None or changed_logits is None or outputs is None or changed_outputs is None:
                            raise RuntimeError("source outcome pass omitted outputs")
                        base_next = _next_logits(base_logits, group).numpy()
                        changed_next = _next_logits(changed_logits, group).numpy()
                        source_outputs.append(outputs.numpy())
                        observed_module.append(changed_outputs.numpy() - outputs.numpy())
                        source_next.append(base_next)
                        observed_logits.append(changed_next - base_next)
                        source_ce.append(v3._continuation_cross_entropy(base_logits, group))
                        cursor += count
                    source_output = np.concatenate(source_outputs)
                    source_module_effect = np.concatenate(observed_module)
                    source_logit_effect = np.concatenate(observed_logits)
                    source_next_logits = np.concatenate(source_next)
                    source_continuation_ce = np.concatenate(source_ce)
                    pred_info = next(row for row in prediction_manifest["predictions"] if row["family"] == family and row["seed_index"] == seed_index and row["method"] == method and row["stage_index"] == stage_index)
                    prediction = np.load(custody / pred_info["prediction_path"], allow_pickle=False)
                    expected_ids = np.asarray([row["example_id"] for row in challenge_rows], dtype="U24")
                    if not np.array_equal(prediction["example_ids"], expected_ids):
                        raise ValueError("prediction and source outcome example order differs")
                    expected_strata = np.asarray([row["stratum"] for row in challenge_rows], dtype="U24")
                    if not np.array_equal(prediction["strata"], expected_strata):
                        raise ValueError("prediction and source outcome strata order differs")
                    original_logits, original_ce, _baseline_rows = guide_baselines[(family, seed_index)]
                    module_scale = np.maximum(np.linalg.norm(source_output, axis=1), 1e-12)
                    logit_scale = np.maximum(np.linalg.norm(source_next_logits, axis=1), 1e-12)
                    metrics = {
                        "module_boundary": _summary_effect(prediction["module_effect"], source_module_effect, module_scale),
                        "next_logits": _summary_effect(prediction["logit_effect"], source_logit_effect, logit_scale),
                        "activation_reconstruction": _activation_metrics(prediction["boundary_output"], source_output, prediction["strata"]),
                        "behavior": _behavior_metrics(prediction["next_logits"], prediction["ce"], original_logits, original_ce, challenge_rows),
                    }
                    per_stratum = {}
                    strata = prediction["strata"]
                    for stratum in fixtures.STRATA:
                        indexes = np.flatnonzero(strata == stratum)
                        per_stratum[stratum] = {
                            "module_boundary": _summary_effect(prediction["module_effect"][indexes], source_module_effect[indexes], module_scale[indexes]),
                            "next_logits": _summary_effect(prediction["logit_effect"][indexes], source_logit_effect[indexes], logit_scale[indexes]),
                            "activation_reconstruction": {
                                "median_cosine": metrics["activation_reconstruction"]["per_stratum"][stratum]["median_cosine"],
                                "median_normalized_error": metrics["activation_reconstruction"]["per_stratum"][stratum]["median_normalized_error"],
                            },
                            "behavior": metrics["behavior"]["per_stratum"][stratum],
                        }
                    stage_pass = all(
                        _passes_effect(per_stratum[stratum][metric])
                        for stratum in fixtures.STRATA for metric in ("module_boundary", "next_logits")
                    ) and all(
                        _passes_behavior(per_stratum[stratum]["behavior"])
                        and per_stratum[stratum]["activation_reconstruction"]["median_cosine"] >= 0.90
                        and per_stratum[stratum]["activation_reconstruction"]["median_normalized_error"] <= 0.20
                        for stratum in fixtures.STRATA
                    )
                    out_relative = f"outcomes/{family}/seed-{seed_index}/{method}/stage-{stage_index}.npz"
                    out_path = custody / out_relative
                    v3._save_npz_deterministic(
                        out_path,
                        source_module_effect=source_module_effect,
                        source_logit_effect=source_logit_effect,
                        source_boundary_output=source_output,
                        source_next_logits=source_next_logits,
                        source_continuation_ce=source_continuation_ce,
                        example_ids=expected_ids,
                        strata=strata,
                    )
                    results.append({
                        "family": family,
                        "model_id": entry["model_id"],
                        "revision": entry["revision"],
                        "seed_index": seed_index,
                        "seed": fixtures.SEEDS[seed_index],
                        "method": method,
                        "stage_index": stage_index,
                        "layer": layer,
                        "prediction_sha256": pred_info["prediction_sha256"],
                        "outcome_sha256": _sha_file(out_path),
                        "metrics": metrics,
                        "per_stratum": per_stratum,
                        "stage_pass": stage_pass,
                    })
                    del model, tokenizer, prediction

    control_results = []
    for item in prediction_manifest["controls"]:
        family, seed_index = item["family"], int(item["seed_index"])
        stage_index, name = int(item["stage_index"]), item["control"]
        source_path = custody / "outcomes" / family / f"seed-{seed_index}" / "ridge" / f"stage-{stage_index}.npz"
        with np.load(source_path, allow_pickle=False) as source:
            source_module_effect = source["source_module_effect"].copy()
            source_logit_effect = source["source_logit_effect"].copy()
            source_output = source["source_boundary_output"].copy()
            source_next = source["source_next_logits"].copy()
            strata = source["strata"].copy()
            example_ids = source["example_ids"].copy()
        with np.load(custody / item["prediction_path"], allow_pickle=False) as predicted:
            if not np.array_equal(predicted["example_ids"], example_ids) or not np.array_equal(predicted["strata"], strata):
                raise ValueError("control prediction rows differ from the source outcome rows")
            predicted_module = predicted["module_effect"].copy()
            predicted_logits = predicted["logit_effect"].copy()
            boundary_output = predicted["boundary_output"].copy()
            next_logits = predicted["next_logits"].copy()
            continuation_ce = predicted["ce"].copy()
        original_logits, original_ce, rows = guide_baselines[(family, seed_index)]
        module_scale = np.maximum(np.linalg.norm(source_output, axis=1), 1e-12)
        logit_scale = np.maximum(np.linalg.norm(source_next, axis=1), 1e-12)
        activation = _activation_metrics(boundary_output, source_output, strata)
        behavior = _behavior_metrics(next_logits, continuation_ce, original_logits, original_ce, rows)
        per_stratum = {}
        for stratum in fixtures.STRATA:
            indexes = np.flatnonzero(strata == stratum)
            per_stratum[stratum] = {
                "module_boundary": _summary_effect(predicted_module[indexes], source_module_effect[indexes], module_scale[indexes]),
                "next_logits": _summary_effect(predicted_logits[indexes], source_logit_effect[indexes], logit_scale[indexes]),
                "activation_reconstruction": activation["per_stratum"][stratum],
                "behavior": behavior["per_stratum"][stratum],
            }
        control_results.append({
            "family": family,
            "seed_index": seed_index,
            "control": name,
            "stage_index": stage_index,
            "prediction_sha256": item["prediction_sha256"],
            "metrics": {
                "module_boundary": _summary_effect(predicted_module, source_module_effect, module_scale),
                "next_logits": _summary_effect(predicted_logits, source_logit_effect, logit_scale),
                "activation_reconstruction": activation,
                "behavior": behavior,
            },
            "per_stratum": per_stratum,
        })
    if len(control_results) != 81:
        raise ValueError("control outcome coverage must include all three controls at all nine stages")

    paired_differences = {"cka": [], "logit_distill": []}
    by_trajectory: dict[tuple[str, int], dict[str, list[float]]] = {}
    for row in results:
        if row["method"] not in {"ridge", *paired_differences.keys()}:
            continue
        key = (row["family"], row["seed_index"])
        by_trajectory.setdefault(key, {}).setdefault(row["method"], []).append(
            float(row["metrics"]["next_logits"]["median_cosine"])
        )
    for key, methods in by_trajectory.items():
        if len(methods.get("ridge", [])) != DEPTH_THIRDS:
            raise ValueError("ridge trajectory is missing stage scores")
        for comparator in paired_differences:
            if len(methods.get(comparator, [])) != DEPTH_THIRDS:
                raise ValueError("comparator trajectory is missing stage scores")
            paired_differences[comparator].append(float(np.mean(methods["ridge"]) - np.mean(methods[comparator])))
    paired_bootstrap = {
        name: _paired_bootstrap(values, seed=202609279 + index)
        for index, (name, values) in enumerate(paired_differences.items())
    }
    ridge_rows = [row for row in results if row["method"] == "ridge"]
    ridge_retention_pass = len(ridge_rows) == 27 and all(row["stage_pass"] for row in ridge_rows)
    comparative_pass = all(
        row["lower_95"] > 0.05 and row["mean_difference"] >= 0.05
        for row in paired_bootstrap.values()
    )
    outcome = "PASS" if ridge_retention_pass and comparative_pass else "FAIL"
    # Missing work and resource termination are caught before this function
    # returns; if either occurs, the caller records INCONCLUSIVE separately.
    elapsed = time.monotonic() - started
    runtime = _runtime_receipt()
    if elapsed > MAX_TOTAL_SECONDS or runtime["peak_rss_bytes"] > MAX_RSS_BYTES:
        outcome = "INCONCLUSIVE"
    report = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "outcome_status": outcome,
        "assurance_level": "INTERNAL_PASS" if len(results) == 81 and len(control_results) == 81 and outcome != "INCONCLUSIVE" else "INCONCLUSIVE",
        "reviewer_status": "NOT_SUBMITTED",
        "reproducibility_status": "PENDING_CLEAN_LOCAL_REPRODUCTION",
        "packet_sha256": packet["packet_sha256"],
        "source_identity": _source_identity(),
        "checkpoint_manifest_sha256": _sha_file(custody / "v9-checkpoint-manifest.json"),
        "candidate_freeze_sha256": _sha_file(custody / "candidate-freeze.json"),
        "challenge_manifest_sha256": _sha_file(custody / "challenge-manifest.json"),
        "prediction_manifest_sha256": _sha_file(prediction_path),
        "method_stage_result_count": len(results),
        "control_stage_result_count": len(control_results),
        "ridge_retention_pass": ridge_retention_pass,
        "paired_causal_effect_cosine_bootstrap": paired_bootstrap,
        "comparative_pass": comparative_pass,
        "claim_ceiling": "LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS",
        "known_limitations": [
            "one checkpoint per model family and three synthetic fixture seeds",
            "CKA comparator is a module-level NoT-style alignment baseline, not a full NoT reproduction",
            "same-owner local reproduction is not independent science",
            "no parameter permutation or symmetry sweep was run",
            "monetary cost and energy are not measured",
        ],
        "nonclaims": ["novelty priority", "all architectures", "whole-model identity", "general mechanistic understanding", "independent science", "consciousness", "alignment", "human equivalence", "production readiness"],
        "next_blocked_authority_gate": "blind external replication and independent signed review; no v9 packet has been submitted",
        "data_and_artifact_custody": "model assets, prompts, token IDs, captures, candidates, predictions, and row-level outcomes remain in external owner custody",
        "resource_limits": {"max_process_rss_bytes": MAX_RSS_BYTES, "max_case_seconds": MAX_CASE_SECONDS, "max_total_seconds": MAX_TOTAL_SECONDS},
        "elapsed_wall_seconds": elapsed,
        "custody_storage": v3._custody_storage_bytes(custody),
        "runtime": runtime,
        "monetary_cost": "NOT_MEASURED",
        "energy": "NOT_MEASURED",
        "results": results,
        "control_results": control_results,
        "created_at_utc": _now(),
    }
    report["report_sha256"] = _canonical_digest(report)
    _write_json(outcome_path, report)
    return {
        "outcome_status": outcome,
        "assurance_level": report["assurance_level"],
        "method_stage_result_count": len(results),
        "ridge_retention_pass": ridge_retention_pass,
        "comparative_pass": comparative_pass,
        "report_sha256": report["report_sha256"],
    }


def _paired_bootstrap(values: list[float], *, seed: int, draws: int = 10_000) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    if data.shape != (9,):
        raise ValueError("paired cluster bootstrap requires nine family-by-seed trajectories")
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(data), size=(draws, len(data)))
    means = data[indexes].mean(axis=1)
    return {
        "trajectory_count": int(len(data)),
        "mean_difference": float(data.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
        "minimum_meaningful_effect": 0.05,
    }


def _write_results_document(custody: Path, report_path: Path) -> None:
    report = _read_json(report_path)
    lines = [
        "# Replacement-loss v9 results: causal replacement ladder",
        "",
        "State slice: `theseus-lab-sandbox-v1`.",
        "",
        f"Packet SHA-256: `{report['packet_sha256']}`.",
        "",
        f"Outcome: **`{report['outcome_status']}`**. Assurance: `{report['assurance_level']}`.",
        "",
        "This aggregate record contains model identities and scalar summaries only. Prompts, weights, captures, candidate tensors, prediction arrays, and row-level outcomes remain in external owner custody.",
        "",
        f"- Checkpoint manifest SHA-256: `{report['checkpoint_manifest_sha256']}`",
        f"- Candidate freeze SHA-256: `{report['candidate_freeze_sha256']}`",
        f"- Challenge manifest SHA-256: `{report['challenge_manifest_sha256']}`",
        f"- Prediction manifest SHA-256: `{report['prediction_manifest_sha256']}`",
        f"- Report SHA-256: `{report['report_sha256']}`",
        f"- Reviewer status: `{report['reviewer_status']}`",
        f"- Reproducibility status: `{report['reproducibility_status']}`",
        "",
        "## Comparative causal-effect result",
        "",
        "| Comparator | Mean paired median-cosine difference | 95% bootstrap interval |",
        "| --- | ---: | ---: |",
    ]
    for name, row in report["paired_causal_effect_cosine_bootstrap"].items():
        lines.append(f"| {name} | {row['mean_difference']:.6f} | [{row['lower_95']:.6f}, {row['upper_95']:.6f}] |")
    lines.extend([
        "",
        "## Claim ceiling and nonclaims",
        "",
        f"`{report['claim_ceiling']}`",
        "",
        "This local result does not establish novelty priority, all-architecture validity, whole-model identity, subjective experience, independent science, alignment, or production readiness.",
        "",
    ])
    (_root() / "docs" / "replacement-loss-v9-results.md").write_text("\n".join(lines), encoding="utf-8")


def run_primary(custody_root: str | Path, prior_custody_roots: list[str | Path]) -> dict[str, Any]:
    global _ACTIVE_RUN_STARTED, _ACTIVE_CASE_STARTED, _ACTIVE_PHASE
    custody = _custody_root(custody_root)
    if (custody / "primary-run-stop.json").exists():
        raise FileExistsError("a stopped v9 primary run is immutable")
    started = time.monotonic()
    _ACTIVE_RUN_STARTED = started
    _ACTIVE_CASE_STARTED = None
    try:
        _ACTIVE_PHASE = "candidate_preparation"
        prepare_candidates(custody, prior_custody_roots)
        _check_runtime_budget()
        _ACTIVE_PHASE = "candidate_freeze"
        freeze_candidates(custody)
        _check_runtime_budget()
        _ACTIVE_PHASE = "challenge_generation"
        generate_challenges(custody, prior_custody_roots)
        _check_runtime_budget()
        _ACTIVE_PHASE = "prediction_sealing"
        seal_predictions(custody)
        _check_runtime_budget()
        _ACTIVE_PHASE = "source_outcome_evaluation"
        result = evaluate_outcomes(custody)
        total_elapsed = time.monotonic() - started
        report_path = custody / "outcome-manifest.json"
        report = _read_json(report_path)
        if total_elapsed > MAX_TOTAL_SECONDS or _runtime_receipt()["peak_rss_bytes"] > MAX_RSS_BYTES:
            report["outcome_status"] = "INCONCLUSIVE"
        report["primary_protocol_wall_seconds"] = total_elapsed
        report["primary_budget_pass"] = report["outcome_status"] != "INCONCLUSIVE"
        report.pop("report_sha256", None)
        report["report_sha256"] = _canonical_digest(report)
        _write_json(report_path, report)
        result = {
            **result,
            "outcome_status": report["outcome_status"],
            "report_sha256": report["report_sha256"],
            "primary_protocol_wall_seconds": total_elapsed,
        }
        _write_results_document(custody, report_path)
        return result
    except Exception as error:
        stop = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": verify_packet(custody)["packet_sha256"],
            "checkpoint_manifest_sha256": _sha_file(custody / "v9-checkpoint-manifest.json") if (custody / "v9-checkpoint-manifest.json").is_file() else None,
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": _ACTIVE_PHASE,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_wall_seconds": time.monotonic() - started,
            "runtime": _runtime_receipt(),
            "created_at_utc": _now(),
        }
        stop["stop_sha256"] = _canonical_digest(stop)
        _write_json(custody / "primary-run-stop.json", stop)
        raise
    finally:
        _ACTIVE_RUN_STARTED = None
        _ACTIVE_CASE_STARTED = None
        _ACTIVE_PHASE = "idle"


def select_checkpoint_set(
    custody_root: str | Path,
    selection_spec_path: str | Path,
    prior_identity_path: str | Path,
) -> dict[str, Any]:
    custody = _custody_root(custody_root)
    selection_path = Path(selection_spec_path).expanduser().resolve()
    prior_path = Path(prior_identity_path).expanduser().resolve()
    for source in (selection_path, prior_path):
        try:
            source.relative_to(_root())
        except ValueError:
            pass
        else:
            raise ValueError("checkpoint selection and prior inventory files must stay outside the repository")
    selection = _read_json(selection_path)
    prior = _read_json(prior_path)
    checkpoints = selection.get("checkpoints")
    identities = prior.get("known_prior_checkpoint_identities")
    if not isinstance(checkpoints, list) or not isinstance(identities, list):
        raise ValueError("selection spec and prior inventory require checkpoint/identity arrays")
    return write_checkpoint_manifest(custody, checkpoints, identities)


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "select", "prepare", "freeze-candidates", "challenges", "predict", "evaluate", "run"))
    parser.add_argument("--custody-root", required=True)
    parser.add_argument("--prior-custody-root", action="append", default=[])
    parser.add_argument("--selection-spec")
    parser.add_argument("--prior-identities")
    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_packet(args.custody_root)
    elif args.command == "verify":
        result = verify_packet(args.custody_root)
    elif args.command == "select":
        if not args.selection_spec or not args.prior_identities:
            parser.error("select requires --selection-spec and --prior-identities")
        result = select_checkpoint_set(args.custody_root, args.selection_spec, args.prior_identities)
    elif args.command == "prepare":
        result = prepare_candidates(args.custody_root, args.prior_custody_root)
    elif args.command == "freeze-candidates":
        result = freeze_candidates(args.custody_root)
    elif args.command == "challenges":
        result = generate_challenges(args.custody_root, args.prior_custody_root)
    elif args.command == "predict":
        result = seal_predictions(args.custody_root)
    elif args.command == "evaluate":
        result = evaluate_outcomes(args.custody_root)
        _write_results_document(_custody_root(args.custody_root), _custody_root(args.custody_root) / "outcome-manifest.json")
    else:
        result = run_primary(args.custody_root, args.prior_custody_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
