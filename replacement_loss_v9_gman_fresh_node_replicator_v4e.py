"""V4e runtime adapter around the digest-bound V4 scientific replicator.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e"
V4C_PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c"


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
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _packet_root(workspace: Path) -> Path:
    return workspace / "v4e"


def _read_lock(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#") and "==" in value:
            name, version = value.split("==", 1)
            rows[name.strip().lower().replace("_", "-")] = version.strip()
    return rows


def _verify_local_packet(packet_root: Path) -> dict[str, Any]:
    freeze = _read_json(packet_root / "freeze.json")
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V4e freeze canonical digest mismatch")
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("state_slice") != STATE_SLICE:
        raise ValueError("V4e protocol or state-slice identity mismatch")
    for key, local_name in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
        ("importer", "importer.py"),
    ):
        row = freeze[key]
        path = packet_root / local_name
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"V4e frozen file mismatch: {key}")
    return freeze


def _verify_parent_v4c(workspace: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    parent_root = workspace / "v4"
    parent = freeze["parent_v4c"]
    freeze_path = parent_root / "freeze.json"
    if not freeze_path.is_file() or _sha_file(freeze_path) != parent["freeze_file_sha256"]:
        raise ValueError("terminal V4c freeze file differs from the V4e binding")
    parent_freeze = _read_json(freeze_path)
    material = {key: value for key, value in parent_freeze.items() if key != "packet_sha256"}
    if (
        parent_freeze.get("packet_sha256") != parent["packet_sha256"]
        or parent_freeze.get("packet_sha256") != _canonical_digest(material)
        or parent_freeze.get("protocol_id") != V4C_PROTOCOL_ID
    ):
        raise ValueError("terminal V4c packet identity is invalid")
    implementation = parent_root / "implementation.py"
    if not implementation.is_file() or _sha_file(implementation) != parent["implementation_sha256"]:
        raise ValueError("digest-bound V4 implementation is missing or changed")
    for key, path_name in (
        ("transport_receipt", "transport-preflight-v4c.json"),
        ("runtime_receipt", "runtime-preflight-v4.json"),
        ("preflight_receipt", "v4-preflight.json"),
        ("stop_receipt", "v4-run-stop.json"),
    ):
        row = parent[key]
        path = parent_root / path_name
        if not path.is_file() or path.stat().st_size != row["bytes"] or _sha_file(path) != row["sha256"]:
            raise ValueError(f"terminal V4c {key} differs from the V4e binding")

    transport = _read_json(parent_root / "transport-preflight-v4c.json")
    transport_material = {key: value for key, value in transport.items() if key != "receipt_sha256"}
    if (
        transport.get("receipt_sha256") != parent["transport_receipt_sha256"]
        or transport.get("receipt_sha256") != _canonical_digest(transport_material)
        or transport.get("all_checks_pass") is not True
        or transport.get("packet_sha256") != parent["packet_sha256"]
    ):
        raise ValueError("V4c transport receipt is not a valid passing parent receipt")

    preflight = _read_json(parent_root / "v4-preflight.json")
    preflight_material = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if (
        preflight.get("preflight_sha256") != parent["preflight_receipt_sha256"]
        or preflight.get("preflight_sha256") != _canonical_digest(preflight_material)
        or preflight.get("all_checks_pass") is not True
        or preflight.get("held_out_predictions_or_outcomes_opened") is not False
        or preflight.get("packet_sha256") != parent["packet_sha256"]
    ):
        raise ValueError("V4c preflight is not a passing, unrevealed parent receipt")

    stop = _read_json(parent_root / "v4-run-stop.json")
    if (
        stop.get("v4_stop_sha256") != parent["stop_receipt_digest"]
        or stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("packet_sha256") != parent["packet_sha256"]
    ):
        raise ValueError("V4c stop receipt is not the bound terminal pre-evaluation stop")
    return parent_freeze


def _make_v4e_runtime_loader(module: Any, freeze: dict[str, Any]):
    def load_v9_runtime(workspace: Path, v4e_freeze: dict[str, Any]):
        source_root = workspace / "input" / "source-checkout"
        expected_source = v4e_freeze["source_identity"]
        source_checks = {
            "runner_sha256": _sha_file(source_root / "replacement_loss_v9_runner.py"),
            "replicator_sha256": _sha_file(source_root / "replacement_loss_v9_replicator.py"),
        }
        for key, actual in source_checks.items():
            if actual != expected_source[key]:
                raise ValueError(f"frozen V9 source identity mismatch: {key}")

        source_text = str(source_root)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        import torch  # noqa: PLC0415
        import replacement_loss_v9_runner as runner  # noqa: PLC0415

        torch.set_num_threads(1)
        reference_path = workspace / "receipts" / "runtime-preflight-v3.json"
        runtime_spec = v4e_freeze["runtime"]
        if not reference_path.is_file() or _sha_file(reference_path) != runtime_spec["base_reference_receipt_sha256"]:
            raise ValueError("V3 runtime reference receipt differs from the frozen V4e binding")
        reference = _read_json(reference_path)["runtime"]
        expected_distributions = dict(reference["all_installed_distributions"])
        if expected_distributions.get("packaging") != runtime_spec["packaging_base_version"]:
            raise ValueError("V3 packaging version differs from the frozen V4e derivation")
        expected_distributions["packaging"] = runtime_spec["packaging_target_version"]

        lock_path = Path(runner.__file__).resolve().parent / "requirements-v9.lock.txt"
        if not lock_path.is_file() or _sha_file(lock_path) != v4e_freeze["target_lock"]["sha256"]:
            raise ValueError("frozen V9 requirements lock differs from the V4e binding")
        lock_rows = _read_lock(lock_path)
        if len(lock_rows) != v4e_freeze["target_lock"]["entry_count"]:
            raise ValueError("frozen V9 requirements lock has an unexpected entry count")

        def validate_runtime() -> dict[str, Any]:
            torch.set_num_threads(1)
            observed = runner._runtime_receipt()
            actual_distributions = observed["all_installed_distributions"]
            if actual_distributions != expected_distributions:
                missing_or_different = {
                    name: {"expected": version, "actual": actual_distributions.get(name)}
                    for name, version in expected_distributions.items()
                    if actual_distributions.get(name) != version
                }
                extra = sorted(set(actual_distributions) - set(expected_distributions))
                raise ValueError(
                    "installed distributions differ from the V4e-derived full runtime manifest: "
                    + json.dumps({"mismatched": missing_or_different, "extra": extra}, sort_keys=True)
                )
            if (
                observed.get("python") != runtime_spec["python"]
                or observed.get("platform") != reference.get("platform")
                or observed.get("machine") != runtime_spec["machine"]
                or observed.get("processor") != reference.get("processor")
                or observed.get("dependencies") != runner.DEPENDENCY_PINS
                or observed.get("cpu_thread_count") != runtime_spec["cpu_thread_count"]
            ):
                raise ValueError("Python, platform, primary dependencies, or thread count differs from V4e")
            locked = observed.get("locked_distributions", {})
            if locked != lock_rows:
                raise ValueError("runner lock parser differs from the frozen V9 lock rows")
            lock_mismatches = {
                name: {"locked": version, "installed": actual_distributions.get(name)}
                for name, version in lock_rows.items()
                if actual_distributions.get(name) != version
            }
            if lock_mismatches:
                raise ValueError(
                    "V9 lock entries are not exact members of the V4e runtime: "
                    + json.dumps(lock_mismatches, sort_keys=True)
                )
            return observed

        # Adapt only the V9 environment gate. The complete V4e profile is
        # checked above; all frozen V9 lock rows remain exact-version checks.
        runner.verify_runtime = validate_runtime
        observed = runner.verify_runtime()
        runtime_receipt = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": v4e_freeze["packet_sha256"],
            "runtime": observed,
            "runtime_match": True,
            "mismatch_fields": [],
            "base_v3_runtime_receipt_sha256": _sha_file(reference_path),
            "base_v3_dependency_manifest_sha256": reference["dependency_manifest_sha256"],
            "derived_runtime_manifest_sha256": _canonical_digest(expected_distributions),
            "target_lock_sha256": _sha_file(lock_path),
            "target_lock_entry_count": len(lock_rows),
            "target_lock_subset_pass": True,
            "extra_distribution_count": len(expected_distributions) - len(lock_rows),
            "runtime_delta": {
                "packaging": {
                    "base": runtime_spec["packaging_base_version"],
                    "target": runtime_spec["packaging_target_version"],
                }
            },
            "adapter_sha256": v4e_freeze["wrapper"]["sha256"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        return runner, torch, runtime_receipt

    return load_v9_runtime


def _load_implementation():
    packet_root = Path(__file__).resolve().parent
    freeze = _verify_local_packet(packet_root)
    workspace = packet_root.parent
    parent_freeze = _verify_parent_v4c(workspace, freeze)
    implementation_path = workspace / freeze["implementation"]["path"]
    if (
        _sha_file(implementation_path) != freeze["implementation"]["sha256"]
        or parent_freeze["implementation"]["sha256"] != freeze["implementation"]["sha256"]
    ):
        raise ValueError("digest-bound V4 implementation differs from V4c")
    spec = importlib.util.spec_from_file_location("theseus_v4e_frozen_implementation", implementation_path)
    if spec is None or spec.loader is None:
        raise ImportError("could not load the digest-bound V4 implementation")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.PROTOCOL_ID = PROTOCOL_ID
    module.STATE_SLICE = STATE_SLICE
    module._packet_root = _packet_root

    original_verify = module._verify_v4_packet

    def verify_packet(workspace_root: Path):
        packet = original_verify(workspace_root)
        receipt_path = _packet_root(workspace_root) / "transport-preflight-v4e.json"
        receipt = _read_json(receipt_path)
        receipt_material = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != _canonical_digest(receipt_material)
            or receipt.get("all_checks_pass") is not True
            or receipt.get("protocol_id") != PROTOCOL_ID
            or receipt.get("state_slice") != STATE_SLICE
            or receipt.get("packet_sha256") != packet.get("packet_sha256")
            or receipt.get("artifacts") != packet.get("custody_transfer", {}).get("artifacts")
        ):
            raise ValueError("V4e transport receipt differs from the frozen custody manifest")
        return packet

    module._verify_v4_packet = verify_packet
    module._load_v9_runtime = _make_v4e_runtime_loader(module, freeze)
    return module


if __name__ == "__main__":
    _load_implementation().main()
