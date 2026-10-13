"""Digest-bound Linux-runtime reproduction wrapper for frozen v9.

State slice: theseus-replacement-loss-v9-gman-linux-runtime-reproduction-v2.
The v9 model, fixture, candidate, and outcome code is imported unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verify_freeze(freeze_path: Path, protocol_path: Path) -> dict[str, Any]:
    freeze = _read_json(freeze_path)
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V2 freeze manifest canonical digest mismatch")
    document = freeze["packet_document"]
    if _sha_file(protocol_path) != document["sha256"]:
        raise ValueError("V2 protocol document digest mismatch")
    wrapper = freeze["wrapper"]
    if _sha_file(Path(__file__).resolve()) != wrapper["sha256"]:
        raise ValueError("V2 wrapper digest mismatch")
    return freeze


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    primary = Path(args.primary_root).resolve()
    replica = Path(args.replica_root).resolve()
    freeze_path = Path(args.freeze).resolve()
    protocol_path = Path(args.protocol_document).resolve()
    runtime_path = Path(args.runtime_preflight).resolve()
    prior_roots = [Path(item).resolve() for item in args.prior_custody_root]

    freeze = _verify_freeze(freeze_path, protocol_path)
    if _sha_file(source_root / "replacement_loss_v9_runner.py") != freeze["source_identity"]["runner_sha256"]:
        raise ValueError("frozen v9 runner source digest mismatch")
    replicator_path = source_root / "replacement_loss_v9_replicator.py"
    if _sha_file(replicator_path) != freeze["source_identity"]["replicator_sha256"]:
        raise ValueError("frozen v9 replicator source digest mismatch")
    if _sha_file(runtime_path) != freeze["runtime"]["preflight_receipt_sha256"]:
        raise ValueError("V1 runtime-preflight receipt digest mismatch")
    if _sha_file(Path(args.input_archive)) != freeze["input"]["archive_sha256"]:
        raise ValueError("retained input archive digest mismatch")
    if _sha_file(Path(args.input_manifest)) != freeze["input"]["input_manifest_sha256"]:
        raise ValueError("retained input manifest digest mismatch")
    if _sha_file(primary / "outcome-manifest.json") != freeze["primary"]["outcome_manifest_sha256"]:
        raise ValueError("primary outcome manifest changed before V2 replication")

    expected_prior_roots = freeze["prior_fixture_roots"]
    if sorted(path.name for path in prior_roots) != expected_prior_roots:
        raise ValueError("prior fixture root set differs from the V2 freeze")

    sys.path.insert(0, str(source_root))
    import replacement_loss_v9_runner as runner  # noqa: E402
    import torch  # noqa: E402

    runtime_reference = _read_json(runtime_path)["runtime"]
    runtime_fields = (
        "python",
        "platform",
        "machine",
        "processor",
        "dependencies",
        "all_installed_distributions",
        "locked_distributions",
        "dependency_manifest_sha256",
        "locked_environment_pass",
    )

    def verify_v2_runtime() -> dict[str, Any]:
        torch.set_num_threads(1)
        current = runner._runtime_receipt()
        current["cpu_thread_count"] = torch.get_num_threads()
        for field in runtime_fields:
            if current[field] != runtime_reference[field]:
                raise ValueError(f"Linux runtime differs from the V2 frozen receipt: {field}")
        if current["python"] != "3.12.13" or current["cpu_thread_count"] != 1:
            raise ValueError("V2 requires Python 3.12.13 and one PyTorch CPU thread")
        if current["dependency_manifest_sha256"] != freeze["runtime"]["dependency_manifest_sha256"]:
            raise ValueError("V2 dependency manifest digest mismatch")
        return current

    runtime_receipt = verify_v2_runtime()
    runner.verify_runtime = verify_v2_runtime
    packet = runner.verify_packet(primary)
    if packet["packet_sha256"] != freeze["target"]["v9_packet_sha256"]:
        raise ValueError("v9 target packet digest mismatch")

    source_text = replicator_path.read_text(encoding="utf-8")
    old_check = (
        '        "dependency_manifest": primary_prep["runtime"]["dependency_manifest_sha256"] '
        '== replica_prep["runtime"]["dependency_manifest_sha256"],'
    )
    new_check = (
        '        "dependency_manifest": replica_prep["runtime"]["dependency_manifest_sha256"] '
        '== EXPECTED_DEPENDENCY_MANIFEST_SHA256,'
    )
    if source_text.count(old_check) != 1:
        raise ValueError("expected one frozen dependency comparison in v9 replicator")
    patched_source = source_text.replace(old_check, new_check, 1)
    if hashlib.sha256(patched_source.encode("utf-8")).hexdigest() != freeze["source_identity"]["patched_replicator_sha256"]:
        raise ValueError("predeclared V2 in-memory replicator patch digest mismatch")

    namespace: dict[str, Any] = {
        "__name__": "replacement_loss_v9_gman_runtime_replicator_v2",
        "__file__": str(replicator_path),
        "EXPECTED_DEPENDENCY_MANIFEST_SHA256": freeze["runtime"]["dependency_manifest_sha256"],
    }
    exec(compile(patched_source, str(replicator_path), "exec"), namespace)
    run_replica = namespace["run_replica"]

    try:
        result = run_replica(primary, replica, prior_roots)
        receipt_path = replica / "replication-manifest.json"
        receipt = _read_json(receipt_path)
        primary_prep = _read_json(primary / "candidate-preparation.json")
        replica_prep = _read_json(replica / "candidate-preparation.json")
        primary_runtime_digest = primary_prep["runtime"]["dependency_manifest_sha256"]
        replica_runtime_digest = replica_prep["runtime"]["dependency_manifest_sha256"]
        if replica_runtime_digest != freeze["runtime"]["dependency_manifest_sha256"]:
            raise ValueError("replica preparation did not bind the V2 runtime digest")
        receipt["runtime_reproduction_v2"] = {
            "protocol_id": freeze["protocol_id"],
            "packet_sha256": freeze["packet_sha256"],
            "runtime_preflight_receipt_sha256": freeze["runtime"]["preflight_receipt_sha256"],
            "expected_replica_dependency_manifest_sha256": freeze["runtime"]["dependency_manifest_sha256"],
            "primary_dependency_manifest_sha256": primary_runtime_digest,
            "replica_dependency_manifest_sha256": replica_runtime_digest,
            "primary_and_replica_dependency_manifests_equal": primary_runtime_digest == replica_runtime_digest,
            "runtime_identity_verified": True,
        }
        receipt["claim_ceiling_v2"] = freeze["claim_ceiling"]
        receipt.pop("replication_sha256", None)
        receipt["replication_sha256"] = _canonical_digest(receipt)
        _write_json(receipt_path, receipt)
        primary_receipt_path = primary / "replication-manifest.json"
        _write_json(primary_receipt_path, receipt)

        primary_report_path = primary / "outcome-manifest.json"
        primary_report = _read_json(primary_report_path)
        primary_report["reproducibility_status"] = receipt["reproducibility_status"]
        primary_report["replication_manifest_sha256"] = _sha_file(primary_receipt_path)
        primary_report.pop("report_sha256", None)
        primary_report["report_sha256"] = _canonical_digest(primary_report)
        _write_json(primary_report_path, primary_report)
        runner._write_results_document(primary, primary_report_path)

        result["replication_sha256"] = receipt["replication_sha256"]
        result["v2_packet_sha256"] = freeze["packet_sha256"]
        result["runtime_reproduction_v2"] = receipt["runtime_reproduction_v2"]
        return result
    except Exception as error:
        stop = {
            "protocol_id": freeze["protocol_id"],
            "packet_sha256": freeze["packet_sha256"],
            "state_slice": freeze["state_slice"],
            "stop_status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error": str(error),
            "runtime_dependency_manifest_sha256": runtime_receipt["dependency_manifest_sha256"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["stop_sha256"] = _canonical_digest(stop)
        _write_json(replica / "replication-v2-stop.json", stop)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--protocol-document", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--input-archive", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--runtime-preflight", required=True)
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(_run(args), sort_keys=True))
    except Exception as error:
        replica = Path(args.replica_root)
        stop_path = replica / "replication-v2-stop.json"
        if not stop_path.exists():
            freeze_path = Path(args.freeze)
            packet_sha256 = _read_json(freeze_path).get("packet_sha256") if freeze_path.is_file() else None
            stop = {
                "protocol_id": "theseus-replacement-loss-v9-linux-runtime-reproduction-v2",
                "state_slice": "theseus-replacement-loss-v9-gman-linux-runtime-reproduction-v2",
                "packet_sha256": packet_sha256,
                "stop_status": "INCONCLUSIVE",
                "error_type": type(error).__name__,
                "error": str(error),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            stop["stop_sha256"] = _canonical_digest(stop)
            _write_json(stop_path, stop)
        raise


if __name__ == "__main__":
    main()
