#!/usr/bin/env python3
"""V4f wrapper around the digest-bound V4 scientific replication implementation.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f"
STATE_SLICE = PROTOCOL_ID
PACKET_NAME = "v4f"
EXPECTED_ACTIVE_MANIFEST_SHA256 = "0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63"
V4E_PACKET_SHA256 = "7b24806d2a6d3c97ca9579e10c1c0136a5415c08d940b69583cfa0f73886e914"
V4E_FREEZE_FILE_SHA256 = "44cfcbe29e5fdd000278304da8199c7910c858501c3ae3fcbd1ed3118c2ceb93"
V4E_STOP_FILE_SHA256 = "265f7dc9f966267a431d009f073f9ca8de3f8e36813bf5e966e0535d685cff4d"
V4E_TRANSPORT_FILE_SHA256 = "3192dbaada484be1b02975ad7c1078f72e966066c2ae870551665aa2a0b34cab"
V4_SOURCE_SHA256 = "2ae1cd0f8b813c28d8815c1b1bfab01a46fcd65afa7d0da24aad81a78b22fc97"


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _read_lock(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and "==" in value:
            name, version = value.split("==", 1)
            normalized = name.strip().lower().replace("_", "-")
            if normalized in rows:
                raise ValueError(f"duplicate normalized package in V9 lock: {normalized}")
            rows[normalized] = version.strip()
    return rows


def _verify_local_packet(packet_root: Path) -> dict[str, Any]:
    freeze = _read_json(packet_root / "freeze.json")
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if (
        freeze.get("packet_sha256") != _canonical_digest(material)
        or freeze.get("protocol_id") != PROTOCOL_ID
        or freeze.get("state_slice") != STATE_SLICE
    ):
        raise ValueError("V4f packet identity or canonical digest mismatch")
    for key, filename in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
        ("importer", "importer.py"),
    ):
        row = freeze[key]
        path = packet_root / filename
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V4f frozen file mismatch: {key}")
    return freeze


def _verify_parent_v4e(workspace: Path, freeze: dict[str, Any]) -> None:
    parent_root = workspace / "v4e"
    parent = freeze["parent_v4e"]
    freeze_path = parent_root / "freeze.json"
    stop_path = parent_root / "v4-preflight-stop.json"
    transport_path = parent_root / "transport-preflight-v4e.json"
    for path, expected in (
        (freeze_path, V4E_FREEZE_FILE_SHA256),
        (stop_path, V4E_STOP_FILE_SHA256),
        (transport_path, V4E_TRANSPORT_FILE_SHA256),
    ):
        if not path.is_file() or _sha_file(path) != expected:
            raise ValueError(f"V4e parent evidence digest mismatch: {path.name}")
    parent_freeze = _read_json(freeze_path)
    parent_material = {key: value for key, value in parent_freeze.items() if key != "packet_sha256"}
    if (
        parent_freeze.get("packet_sha256") != V4E_PACKET_SHA256
        or parent_freeze.get("packet_sha256") != _canonical_digest(parent_material)
        or parent.get("packet_sha256") != V4E_PACKET_SHA256
        or parent_freeze.get("source_snapshot", {}).get("snapshot_id") != "snap-t3r2t"
    ):
        raise ValueError("V4e parent freeze identity mismatch")
    stop = _read_json(stop_path)
    stop_material = {key: value for key, value in stop.items() if key != "v4_stop_sha256"}
    if (
        stop.get("v4_stop_sha256") != "4c6ed337a6ccd555ac983afedc274c3aca9f33264165843b9024cceab3bfafb5"
        or stop.get("v4_stop_sha256") != _canonical_digest(stop_material)
        or stop.get("stopped_phase") != "preflight"
        or stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("packet_sha256") != V4E_PACKET_SHA256
        or (parent_root / "v4-run-receipt.json").exists()
    ):
        raise ValueError("V4e is not the bound pre-evaluation stop")
    transport = _read_json(transport_path)
    transport_material = {key: value for key, value in transport.items() if key != "receipt_sha256"}
    if (
        transport.get("receipt_sha256") != "5776d2639ddb835e79718f215cf01ae0f65fa52f8a2d2ddd88c322a9f3ca12eb"
        or transport.get("receipt_sha256") != _canonical_digest(transport_material)
        or transport.get("all_checks_pass") is not True
        or transport.get("held_out_predictions_or_outcomes_opened") is not False
        or transport.get("packet_sha256") != V4E_PACKET_SHA256
    ):
        raise ValueError("V4e transport receipt does not bind sealed held-out data")


def _active_distribution_map() -> tuple[dict[str, str], list[dict[str, str]]]:
    rows: dict[str, str] = {}
    vendored: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        normalized = str(name).strip().lower().replace("_", "-")
        version = str(distribution.version)
        metadata_path = getattr(distribution, "_path", None)
        if metadata_path is None:
            raise ValueError(f"distribution metadata path is unavailable: {normalized}")
        path = Path(str(metadata_path))
        if "_vendor" in path.parts:
            vendored.append({"name": normalized, "version": version, "metadata_path": str(path)})
            continue
        if normalized in rows:
            raise ValueError(f"duplicate active distribution metadata: {normalized}")
        rows[normalized] = version
    return rows, vendored


def _make_runtime_loader(workspace: Path, freeze: dict[str, Any]):
    source_root = workspace / "input" / "source-checkout"
    source_expected = freeze["source_identity"]
    for name, key in (
        ("replacement_loss_v9_runner.py", "runner_sha256"),
        ("replacement_loss_v9_replicator.py", "replicator_sha256"),
    ):
        path = source_root / name
        if not path.is_file() or _sha_file(path) != source_expected[key]:
            raise ValueError(f"frozen V9 source identity mismatch: {key}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    import torch  # noqa: PLC0415
    import replacement_loss_v9_runner as runner  # noqa: PLC0415

    torch.set_num_threads(1)
    runtime_spec = freeze["runtime"]
    reference_path = workspace / "receipts" / "runtime-preflight-v3.json"
    if (
        not reference_path.is_file()
        or _sha_file(reference_path) != runtime_spec["base_reference_receipt_sha256"]
    ):
        raise ValueError("V3 runtime reference receipt differs from the frozen V4f binding")
    reference = _read_json(reference_path)["runtime"]
    expected_distributions = dict(reference["all_installed_distributions"])
    if expected_distributions.get("packaging") != runtime_spec["packaging_base_version"]:
        raise ValueError("V3 packaging version differs from the V4f derivation")
    expected_distributions["packaging"] = runtime_spec["packaging_target_version"]
    expected_manifest_sha = _canonical_digest(expected_distributions)
    if (
        expected_manifest_sha != runtime_spec["dependency_manifest_sha256"]
        or expected_manifest_sha != EXPECTED_ACTIVE_MANIFEST_SHA256
        or len(expected_distributions) != runtime_spec["expected_distribution_count"]
    ):
        raise ValueError("derived V4f active distribution manifest differs from the frozen digest")

    lock_path = source_root / "requirements-v9.lock.txt"
    if not lock_path.is_file() or _sha_file(lock_path) != freeze["target_lock"]["sha256"]:
        raise ValueError("frozen V9 requirements lock differs from V4f binding")
    lock_rows = _read_lock(lock_path)
    if len(lock_rows) != freeze["target_lock"]["entry_count"]:
        raise ValueError("V9 lock row count differs from V4f freeze")

    original_runtime_receipt = runner._runtime_receipt
    excluded_vendored_metadata: list[dict[str, str]] = []

    def normalized_runtime_receipt() -> dict[str, Any]:
        nonlocal excluded_vendored_metadata
        observed = original_runtime_receipt()
        active, vendored = _active_distribution_map()
        if active != expected_distributions:
            mismatch = {
                name: {"expected": version, "actual": active.get(name)}
                for name, version in expected_distributions.items()
                if active.get(name) != version
            }
            extra = sorted(set(active) - set(expected_distributions))
            raise ValueError(
                "active V4f distribution manifest mismatch: "
                + json.dumps({"mismatched": mismatch, "extra": extra}, sort_keys=True)
            )
        expected_vendored = [
            row for row in vendored
            if row["name"] == "packaging" and row["version"] == runtime_spec["packaging_vendored_duplicate_version"]
        ]
        if not expected_vendored:
            raise ValueError("diagnosed vendored packaging metadata entry is absent")
        excluded_vendored_metadata = vendored
        observed["all_installed_distributions"] = active
        observed["dependency_manifest_sha256"] = _canonical_digest(active)
        observed["locked_environment_pass"] = active == observed.get("locked_distributions", {})
        observed["cpu_thread_count"] = torch.get_num_threads()
        return observed

    def validate_runtime() -> dict[str, Any]:
        torch.set_num_threads(1)
        observed = normalized_runtime_receipt()
        actual = observed["all_installed_distributions"]
        if (
            observed.get("python") != runtime_spec["python"]
            or observed.get("platform") != reference.get("platform")
            or observed.get("machine") != runtime_spec["machine"]
            or observed.get("processor") != reference.get("processor")
            or observed.get("dependencies") != runner.DEPENDENCY_PINS
            or observed.get("cpu_thread_count") != runtime_spec["cpu_thread_count"]
            or actual != expected_distributions
            or observed.get("dependency_manifest_sha256") != runtime_spec["dependency_manifest_sha256"]
        ):
            raise ValueError("V4f active runtime identity differs from the frozen runtime map")
        if observed.get("locked_distributions", {}) != lock_rows:
            raise ValueError("V9 runner lock parser differs from frozen lock rows")
        lock_mismatches = {
            name: {"locked": version, "active": actual.get(name)}
            for name, version in lock_rows.items()
            if actual.get(name) != version
        }
        if lock_mismatches:
            raise ValueError(
                "V9 lock entries are not exact active installed versions: "
                + json.dumps(lock_mismatches, sort_keys=True)
            )
        return observed

    runner._runtime_receipt = normalized_runtime_receipt
    runner.verify_runtime = validate_runtime
    runtime = validate_runtime()
    runtime_receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "runtime": runtime,
        "runtime_match": True,
        "mismatch_fields": [],
        "base_v3_runtime_receipt_sha256": _sha_file(reference_path),
        "base_v3_dependency_manifest_sha256": reference["dependency_manifest_sha256"],
        "derived_runtime_manifest_sha256": expected_manifest_sha,
        "filtered_vendored_metadata_count": len(excluded_vendored_metadata),
        "filtered_vendored_metadata_sha256": _canonical_digest(excluded_vendored_metadata),
        "filtered_packaging_versions": sorted(
            row["version"] for row in excluded_vendored_metadata if row["name"] == "packaging"
        ),
        "target_lock_sha256": _sha_file(lock_path),
        "target_lock_entry_count": len(lock_rows),
        "target_lock_subset_pass": True,
        "extra_active_distribution_count": len(expected_distributions) - len(lock_rows),
        "adapter_sha256": freeze["wrapper"]["sha256"],
    }

    def load_v9_runtime(_workspace: Path, _freeze: dict[str, Any]):
        return runner, torch, runtime_receipt

    return load_v9_runtime


def _load_implementation():
    packet_root = Path(__file__).resolve().parent
    freeze = _verify_local_packet(packet_root)
    workspace = packet_root.parent
    _verify_parent_v4e(workspace, freeze)
    implementation_path = workspace / freeze["implementation"]["path"]
    if not implementation_path.is_file() or _sha_file(implementation_path) != freeze["implementation"]["sha256"]:
        raise ValueError("digest-bound V4 implementation differs from V4f")
    spec = importlib.util.spec_from_file_location("theseus_v4f_frozen_implementation", implementation_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load digest-bound V4 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    module.STATE_SLICE = STATE_SLICE
    module._packet_root = lambda workspace_root: workspace_root / PACKET_NAME
    module._load_v9_runtime = _make_runtime_loader(workspace, freeze)

    original_verify = module._verify_v4_packet

    def verify_packet(workspace_root: Path):
        packet = original_verify(workspace_root)
        receipt_path = workspace_root / PACKET_NAME / "transport-preflight-v4f.json"
        receipt = _read_json(receipt_path)
        material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != _canonical_digest(material)
            or receipt.get("all_checks_pass") is not True
            or receipt.get("held_out_predictions_or_outcomes_opened") is not False
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("state_slice") != STATE_SLICE
            or receipt.get("packet_sha256") != packet.get("packet_sha256")
        ):
            raise ValueError("V4f transport receipt does not bind passing custody preflight")
        return packet

    module._verify_v4_packet = verify_packet
    return module


if __name__ == "__main__":
    _load_implementation().main()

