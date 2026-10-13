"""Fresh-node, digest-bound wrapper for the frozen replacement-loss-v9 target.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v3"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v3"


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
        raise ValueError("V3 freeze manifest canonical digest mismatch")
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("state_slice") != STATE_SLICE:
        raise ValueError("V3 protocol or state-slice identity mismatch")
    if _sha_file(protocol_path) != freeze["packet_document"]["sha256"]:
        raise ValueError("V3 protocol document digest mismatch")
    if _sha_file(Path(__file__).resolve()) != freeze["wrapper"]["sha256"]:
        raise ValueError("V3 wrapper digest mismatch")
    bootstrap_path = Path(__file__).resolve().parent / freeze["bootstrap"]["path"]
    if _sha_file(bootstrap_path) != freeze["bootstrap"]["sha256"]:
        raise ValueError("V3 bootstrap digest mismatch")
    delegate_path = Path(__file__).resolve().parent / freeze["delegate_wrapper"]["path"]
    if _sha_file(delegate_path) != freeze["delegate_wrapper"]["sha256"]:
        raise ValueError("digest-bound V2 helper wrapper mismatch")
    return freeze


def _observed_runtime(source_root: Path, output_path: Path, freeze: dict[str, Any]) -> str:
    sys.path.insert(0, str(source_root))
    import torch  # noqa: PLC0415
    import replacement_loss_v9_runner as runner  # noqa: PLC0415

    torch.set_num_threads(1)
    runtime = runner._runtime_receipt()
    runtime["cpu_thread_count"] = torch.get_num_threads()
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": freeze["packet_sha256"],
        "state_slice": STATE_SLICE,
        "runtime": runtime,
    }
    _write_json(output_path, receipt)
    return _sha_file(output_path)


def _bind_v3_receipts(
    primary: Path,
    replica: Path,
    freeze: dict[str, Any],
    observed_runtime_sha256: str,
    delegate_reproducibility: str,
) -> dict[str, Any]:
    report_path = primary / "outcome-manifest.json"
    report = _read_json(report_path)
    primary_outcome = report["outcome_status"]
    expected_outcome = freeze["acceptance"]["required_replicated_primary_outcome"]
    primary_outcome_gate_passed = primary_outcome == expected_outcome
    if delegate_reproducibility == "REPRODUCED_LOCAL" and primary_outcome_gate_passed:
        final_reproducibility = "REPRODUCED_LOCAL"
    elif delegate_reproducibility == "DIVERGED" or not primary_outcome_gate_passed:
        final_reproducibility = "DIVERGED"
    else:
        final_reproducibility = "INCONCLUSIVE"

    receipt_paths = [replica / "replication-manifest.json", primary / "replication-manifest.json"]
    for path in receipt_paths:
        if not path.is_file():
            raise FileNotFoundError(f"replication receipt missing: {path.name}")
        receipt = _read_json(path)
        detail = receipt.pop("runtime_reproduction_v2", None)
        if not isinstance(detail, dict):
            raise ValueError("delegate did not produce its expected runtime receipt")
        detail["protocol_id"] = PROTOCOL_ID
        detail["packet_sha256"] = freeze["packet_sha256"]
        detail["observed_runtime_receipt_sha256"] = observed_runtime_sha256
        detail["replicated_primary_outcome"] = primary_outcome
        detail["required_primary_outcome"] = expected_outcome
        detail["primary_outcome_gate_passed"] = primary_outcome_gate_passed
        detail["delegate_reproducibility_status"] = delegate_reproducibility
        receipt["runtime_reproduction_v3"] = detail
        receipt["reproducibility_status"] = final_reproducibility
        if final_reproducibility != "REPRODUCED_LOCAL":
            receipt["assurance_level"] = "INCONCLUSIVE"
        receipt.pop("claim_ceiling_v2", None)
        receipt["claim_ceiling_v3"] = freeze["claim_ceiling"]
        receipt.pop("replication_sha256", None)
        receipt["replication_sha256"] = _canonical_digest(receipt)
        _write_json(path, receipt)

    report["reproducibility_status"] = final_reproducibility
    report["replicated_primary_outcome_gate_passed"] = primary_outcome_gate_passed
    report["replication_manifest_sha256"] = _sha_file(primary / "replication-manifest.json")
    report.pop("report_sha256", None)
    report["report_sha256"] = _canonical_digest(report)
    _write_json(report_path, report)

    import replacement_loss_v9_runner as runner  # noqa: PLC0415

    runner._write_results_document(primary, report_path)
    return report


def _run(args: argparse.Namespace) -> dict[str, Any]:
    source_root = Path(args.source_root).resolve()
    primary = Path(args.primary_root).resolve()
    replica = Path(args.replica_root).resolve()
    freeze_path = Path(args.freeze).resolve()
    protocol_path = Path(args.protocol_document).resolve()
    runtime_path = Path(args.runtime_reference).resolve()
    observed_runtime_path = Path(args.observed_runtime_output).resolve()
    prior_roots = [Path(item).resolve() for item in args.prior_custody_root]

    freeze = _verify_freeze(freeze_path, protocol_path)
    if observed_runtime_path.exists():
        raise FileExistsError("V3 observed-runtime receipt already exists")
    observed_runtime_sha256 = _observed_runtime(source_root, observed_runtime_path, freeze)

    delegate_freeze = {key: value for key, value in freeze.items() if key != "delegate_wrapper"}
    delegate_freeze["wrapper"] = freeze["delegate_wrapper"]
    delegate_material = {key: value for key, value in delegate_freeze.items() if key != "packet_sha256"}
    delegate_freeze["packet_sha256"] = _canonical_digest(delegate_material)

    import replacement_loss_v9_gman_linux_runtime_replicator_v2 as delegate  # noqa: PLC0415

    with tempfile.TemporaryDirectory(prefix="theseus-v9-v3-freeze-") as temp_dir:
        delegate_freeze_path = Path(temp_dir) / "delegate-freeze.json"
        _write_json(delegate_freeze_path, delegate_freeze)
        delegate_args = SimpleNamespace(
            source_root=str(source_root),
            primary_root=str(primary),
            replica_root=str(replica),
            freeze=str(delegate_freeze_path),
            protocol_document=str(protocol_path),
            input_archive=args.input_archive,
            input_manifest=args.input_manifest,
            runtime_preflight=str(runtime_path),
            prior_custody_root=args.prior_custody_root,
        )
        result = delegate._run(delegate_args)

    report = _bind_v3_receipts(
        primary,
        replica,
        freeze,
        observed_runtime_sha256,
        str(result.get("reproducibility_status", "INCONCLUSIVE")),
    )
    result.pop("v2_packet_sha256", None)
    result.pop("runtime_reproduction_v2", None)
    result["v3_packet_sha256"] = freeze["packet_sha256"]
    result["observed_runtime_receipt_sha256"] = observed_runtime_sha256
    result["reproduced_primary_outcome"] = report["outcome_status"]
    result["reproducibility_status"] = report["reproducibility_status"]
    result["assurance_level"] = "REPRODUCED_LOCAL" if report["reproducibility_status"] == "REPRODUCED_LOCAL" else "INCONCLUSIVE"
    result["claim_reached"] = report["reproducibility_status"] == "REPRODUCED_LOCAL"
    result["replication_sha256"] = _sha_file(primary / "replication-manifest.json")
    result["runtime_reproduction_v3"] = _read_json(primary / "replication-manifest.json")["runtime_reproduction_v3"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--protocol-document", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--input-archive", required=True)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--runtime-reference", required=True)
    parser.add_argument("--observed-runtime-output", required=True)
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(_run(args), sort_keys=True))
    except Exception as error:
        replica = Path(args.replica_root)
        replica.mkdir(parents=True, exist_ok=True)
        freeze_path = Path(args.freeze)
        freeze = _read_json(freeze_path) if freeze_path.is_file() else {}
        old_stop = replica / "replication-v2-stop.json"
        if old_stop.is_file():
            old_stop.unlink()
        stop = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": freeze.get("packet_sha256"),
            "stop_status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error": str(error),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["stop_sha256"] = _canonical_digest(stop)
        _write_json(replica / "replication-v3-stop.json", stop)
        raise


if __name__ == "__main__":
    main()
