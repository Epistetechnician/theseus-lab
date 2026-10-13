"""Digest-bound V4 fresh-runtime reproduction of the frozen V9 experiment.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4.

V4 refits a clean replica from visible/adaptation data, checks candidate file
bytes and tensor state before challenge reveal, then seals predictions before
opening held-out outcomes. It corrects only the candidate digest-domain check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable


PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4"
STATE_SLICE = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4"
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
WORKSPACE_DEFAULT = Path("/home/dev/theseus-v9")
V3_POST_ERROR_REPORT_SHA256 = "0753bfbef92562940e3aca07ec5cdbe497d1b1659b82f8d2e06d97c65c645b08"
PRIMARY_REPORT_SHA256 = "12e0befa4c329d3a6a18b14d19144ab6296e5ac5b82f0cb2326eaeff7c776431"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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


def _packet_root(workspace: Path) -> Path:
    return workspace / "v4"


def _verify_v4_packet(workspace: Path) -> dict[str, Any]:
    packet_root = _packet_root(workspace)
    freeze = _read_json(packet_root / "freeze.json")
    material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
    if freeze.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V4 freeze canonical digest mismatch")
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("state_slice") != STATE_SLICE:
        raise ValueError("V4 protocol or state-slice identity mismatch")
    for key, local_name in (
        ("packet_document", "protocol.md"),
        ("wrapper", "runner.py"),
        ("bootstrap", "bootstrap.sh"),
    ):
        expected = freeze[key]
        path = packet_root / local_name
        if not path.is_file() or path.stat().st_size != expected["bytes"] or _sha_file(path) != expected["sha256"]:
            raise ValueError(f"V4 frozen file mismatch: {key}")
    return freeze


def _verify_v3_packet(workspace: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    path = workspace / "v3" / "freeze.json"
    if not path.is_file() or _sha_file(path) != freeze["v3_freeze_file_sha256"]:
        raise ValueError("V3 freeze file differs from the V4 binding")
    v3 = _read_json(path)
    material = {key: value for key, value in v3.items() if key != "packet_sha256"}
    if v3.get("packet_sha256") != _canonical_digest(material):
        raise ValueError("V3 freeze canonical digest mismatch")
    if v3.get("packet_sha256") != freeze["v3_packet_sha256"]:
        raise ValueError("V3 packet digest differs from the V4 binding")
    def find_bound_file(name: str, expected_sha256: str) -> Path:
        matches = [
            path
            for path in (workspace / "v3").rglob(name)
            if path.is_file() and _sha_file(path) == expected_sha256
        ]
        if len(matches) != 1:
            raise ValueError(f"V3 frozen file is missing or ambiguous: {name}")
        return matches[0]

    for key in ("wrapper", "bootstrap", "delegate_wrapper"):
        row = v3[key]
        find_bound_file(Path(row["path"]).name, row["sha256"])
    find_bound_file(Path(v3["packet_document"]["path"]).name, v3["packet_document"]["sha256"])
    return v3


def _find_v3_stop(workspace: Path, expected_sha256: str) -> Path:
    found: list[Path] = []
    excluded = {".git", "__pycache__", "venv", "checkpoints", "models", "candidates", "predictions", "outcomes"}
    for current, directories, files in os.walk(workspace):
        directories[:] = [name for name in directories if name not in excluded]
        if "replication-v3-stop.json" in files:
            candidate = Path(current) / "replication-v3-stop.json"
            if _sha_file(candidate) == expected_sha256:
                found.append(candidate)
    if len(found) != 1:
        raise ValueError("V3 stop receipt is missing or not uniquely bound")
    stop = _read_json(found[0])
    if (
        stop.get("stop_status") != "INCONCLUSIVE"
        or stop.get("error_type") != "ValueError"
        or stop.get("error") != "frozen candidate bytes changed"
    ):
        raise ValueError("V3 stop receipt does not record the bound digest-domain failure")
    return found[0]


def _find_pristine_primary(workspace: Path, freeze: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    input_root = workspace / "input"
    archive_path = input_root / "gman-v9-input.tar.gz"
    expected_archive = freeze["input"]["archive_sha256"]
    if not archive_path.is_file() or _sha_file(archive_path) != expected_archive:
        raise ValueError("V3 frozen input archive digest mismatch")
    manifest_path = input_root / "metadata" / "input-manifest.json"
    if not manifest_path.is_file() or _sha_file(manifest_path) != freeze["input"]["input_manifest_sha256"]:
        raise ValueError("V3 frozen input manifest digest mismatch")

    matches: list[tuple[PurePosixPath, bytes]] = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = PurePosixPath(member.name)
            if not member.isfile() or member_path.name != "outcome-manifest.json":
                continue
            stream = archive.extractfile(member)
            if stream is None:
                continue
            data = stream.read()
            if _sha_bytes(data) == freeze["target"]["primary_outcome_manifest_sha256"]:
                matches.append((member_path, data))
    if len(matches) != 1:
        raise ValueError("pristine V9 primary outcome manifest is not uniquely present in the frozen archive")

    member_path, data = matches[0]
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError("frozen archive primary path is unsafe")
    primary = input_root.joinpath(*member_path.parts[:-1])
    report_path = primary / "outcome-manifest.json"
    if not primary.is_dir() or not report_path.is_file():
        raise FileNotFoundError("extracted frozen primary custody is incomplete")
    current_sha = _sha_file(report_path)
    allowed_current = {
        freeze["target"]["primary_outcome_manifest_sha256"],
        V3_POST_ERROR_REPORT_SHA256,
    }
    if current_sha not in allowed_current:
        raise ValueError("V3 clone primary report differs from both frozen and known post-error bytes")
    if current_sha != freeze["target"]["primary_outcome_manifest_sha256"]:
        report_path.write_bytes(data)
    if _sha_file(report_path) != freeze["target"]["primary_outcome_manifest_sha256"]:
        raise ValueError("restored V9 primary outcome report failed its frozen byte digest")
    return primary, _read_json(report_path)


def _resolve_prior_roots(workspace: Path, v3: dict[str, Any]) -> list[Path]:
    input_root = workspace / "input"
    roots: list[Path] = []
    for name in v3["prior_fixture_roots"]:
        matches = [path for path in input_root.rglob(name) if path.is_dir() and path.name == name]
        if len(matches) != 1:
            raise ValueError(f"frozen prior fixture root is missing or ambiguous: {name}")
        roots.append(matches[0])
    if sorted(path.name for path in roots) != v3["prior_fixture_roots"]:
        raise ValueError("prior fixture root set differs from the V3 freeze")
    return roots


def _load_v9_runtime(workspace: Path, freeze: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    source_root = workspace / "input" / "source-checkout"
    if not source_root.is_dir():
        raise FileNotFoundError("frozen V9 source checkout is missing")
    expected_source = freeze["source_identity"]
    checks = {
        "runner_sha256": _sha_file(source_root / "replacement_loss_v9_runner.py"),
        "replicator_sha256": _sha_file(source_root / "replacement_loss_v9_replicator.py"),
    }
    for key, actual in checks.items():
        if actual != expected_source[key]:
            raise ValueError(f"V9 source identity mismatch: {key}")
    sys.path.insert(0, str(source_root))
    import torch  # noqa: PLC0415
    import replacement_loss_v9_runner as runner  # noqa: PLC0415

    torch.set_num_threads(1)
    observed = runner._runtime_receipt()
    observed["cpu_thread_count"] = torch.get_num_threads()
    reference_path = workspace / "receipts" / "runtime-preflight-v3.json"
    if not reference_path.is_file() or _sha_file(reference_path) != freeze["runtime"]["v3_runtime_receipt_sha256"]:
        raise ValueError("V3 observed runtime receipt differs from the V4 binding")
    reference = _read_json(reference_path)
    expected = reference["runtime"]
    fields = (
        "python",
        "platform",
        "machine",
        "processor",
        "dependencies",
        "all_installed_distributions",
        "locked_distributions",
        "dependency_manifest_sha256",
        "locked_environment_pass",
        "cpu_thread_count",
    )
    mismatches = [field for field in fields if observed.get(field) != expected.get(field)]
    if (
        mismatches
        or observed.get("python") != "3.12.13"
        or observed.get("cpu_thread_count") != 1
        or observed.get("dependency_manifest_sha256") != freeze["runtime"]["dependency_manifest_sha256"]
    ):
        raise ValueError("V4 rebuilt runtime does not match the V3-bound Linux runtime")
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": freeze["packet_sha256"],
        "state_slice": STATE_SLICE,
        "runtime": observed,
        "runtime_match": True,
        "mismatch_fields": [],
        "reference_runtime_receipt_sha256": _sha_file(reference_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return runner, torch, receipt


def _validate_primary(workspace: Path, freeze: dict[str, Any], runner: Any) -> tuple[Path, list[Path], dict[str, Any]]:
    v3 = _verify_v3_packet(workspace, freeze)
    _find_v3_stop(workspace, freeze["v3_stop_sha256"])
    primary, report = _find_pristine_primary(workspace, freeze)
    packet = runner.verify_packet(primary)
    if packet["packet_sha256"] != freeze["target"]["v9_packet_sha256"]:
        raise ValueError("frozen V9 packet digest mismatch")
    if _sha_file(primary / "candidate-freeze.json") != freeze["target"]["candidate_freeze_sha256"]:
        raise ValueError("frozen V9 candidate inventory digest mismatch")
    if _sha_file(primary / "challenge-manifest.json") != freeze["target"]["challenge_manifest_sha256"]:
        raise ValueError("frozen V9 challenge manifest digest mismatch")
    if report.get("outcome_status") != freeze["target"]["required_primary_outcome"]:
        raise ValueError("frozen V9 primary scientific outcome differs from the V4 requirement")
    if report.get("source_identity") != runner._source_identity():
        raise ValueError("V9 primary report source identity differs from the frozen checkout")
    prior_roots = _resolve_prior_roots(workspace, v3)
    return primary, prior_roots, report


def _preflight(workspace: Path) -> dict[str, Any]:
    freeze = _verify_v4_packet(workspace)
    v3 = _verify_v3_packet(workspace, freeze)
    _find_v3_stop(workspace, freeze["v3_stop_sha256"])
    runner, _, runtime_receipt = _load_v9_runtime(workspace, freeze)
    primary, prior_roots, report = _validate_primary(workspace, freeze, runner)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "v3_packet_sha256": v3["packet_sha256"],
        "v3_stop_sha256": freeze["v3_stop_sha256"],
        "primary_outcome_status": report["outcome_status"],
        "primary_outcome_manifest_sha256": _sha_file(primary / "outcome-manifest.json"),
        "candidate_freeze_sha256": _sha_file(primary / "candidate-freeze.json"),
        "challenge_manifest_sha256": _sha_file(primary / "challenge-manifest.json"),
        "prior_fixture_root_count": len(prior_roots),
        "runtime_preflight": runtime_receipt,
        "held_out_predictions_or_outcomes_opened": False,
        "all_checks_pass": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["preflight_sha256"] = _canonical_digest(receipt)
    _write_json(_packet_root(workspace) / "runtime-preflight-v4.json", runtime_receipt)
    _write_json(_packet_root(workspace) / "v4-preflight.json", receipt)
    return receipt


def _candidate_integrity_preflight(root: Path, runner: Any, packet_root: Path) -> dict[str, Any]:
    packet = runner.verify_packet(root)
    freeze = runner._verify_candidate_freeze(root, packet["packet_sha256"])
    preparation = _read_json(root / "candidate-preparation.json")
    checked = 0
    for trajectory in preparation.get("trajectories", []):
        for stage in trajectory.get("stages", []):
            module = runner._load_module(
                root / str(stage["candidate_path"]),
                int(stage["in_features"]),
                int(stage["out_features"]),
                bool(stage["use_bias"]),
            )
            state = {
                name: tensor.detach().cpu().contiguous()
                for name, tensor in module.state_dict().items()
            }
            if runner.v3._state_digest(state) != stage.get("candidate_sha256"):
                raise ValueError("candidate tensor-state digest differs from frozen stage")
            del module, state
            checked += 1
    if checked != 81 or len(freeze.get("candidate_files", [])) != 162:
        raise ValueError("candidate integrity preflight did not cover 81 stages and 162 files")
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "candidate_freeze_sha256": _sha_file(root / "candidate-freeze.json"),
        "candidate_stage_state_digest_checks": checked,
        "candidate_and_control_file_byte_checks": len(freeze["candidate_files"]),
        "visible_adaptation_only": True,
        "held_out_challenges_revealed": False,
        "prediction_or_outcome_data_used": False,
        "all_checks_pass": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt["preflight_sha256"] = _canonical_digest(receipt)
    _write_json(packet_root / "candidate-integrity-preflight-v4.json", receipt)
    return receipt


def _patched_replica_runner(
    freeze: dict[str, Any],
    source_root: Path,
    packet_root: Path,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    import replacement_loss_v9_runner as runner  # noqa: PLC0415

    replicator_path = source_root / "replacement_loss_v9_replicator.py"
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
        raise ValueError("frozen V9 replicator dependency comparison is not unique")
    patched_source = source_text.replace(old_check, new_check, 1)
    patched_sha = _sha_bytes(patched_source.encode("utf-8"))
    expected_sha = freeze["source_identity"]["patched_replicator_sha256"]
    if patched_sha != expected_sha:
        raise ValueError("in-memory V9 dependency adaptation differs from the frozen V3 patch")

    namespace: dict[str, Any] = {
        "__name__": "replacement_loss_v9_gman_runtime_replicator_v4",
        "__file__": str(replicator_path),
        "EXPECTED_DEPENDENCY_MANIFEST_SHA256": freeze["runtime"]["dependency_manifest_sha256"],
    }
    exec(compile(patched_source, str(replicator_path), "exec"), namespace)
    candidate_freezes: dict[Path, dict[str, Any]] = {}

    def load_candidate(custody: Path, stage: dict[str, Any]) -> Any:
        root = Path(custody).resolve()
        if root not in candidate_freezes:
            packet = runner.verify_packet(root)
            candidate_freezes[root] = runner._verify_candidate_freeze(root, packet["packet_sha256"])
        frozen = candidate_freezes[root]
        relative = str(stage["candidate_path"])
        rows = [row for row in frozen.get("candidate_files", []) if row.get("path") == relative]
        if len(rows) != 1:
            raise ValueError("candidate is not uniquely covered by the frozen file inventory")
        path = root / relative
        if _sha_file(path) != rows[0].get("sha256"):
            raise ValueError("frozen candidate file-byte digest changed")
        module = runner._load_module(
            path,
            int(stage["in_features"]),
            int(stage["out_features"]),
            bool(stage["use_bias"]),
        )
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in module.state_dict().items()
        }
        if runner.v3._state_digest(state) != stage.get("candidate_sha256"):
            raise ValueError("frozen candidate tensor-state digest changed")
        return module

    def wrap_candidate_loader(operation: Callable[..., Any]) -> Callable[..., Any]:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            original = runner._load_candidate
            runner._load_candidate = load_candidate
            try:
                return operation(*args, **kwargs)
            finally:
                runner._load_candidate = original
        return wrapped

    runner.seal_predictions = wrap_candidate_loader(runner.seal_predictions)
    runner.evaluate_outcomes = wrap_candidate_loader(runner.evaluate_outcomes)

    reveal = namespace["_reveal_challenges_after_candidate_freeze"]

    def reveal_after_preflight(primary: Path, replica: Path) -> dict[str, Any]:
        _candidate_integrity_preflight(replica, runner, packet_root)
        return reveal(primary, replica)

    namespace["_reveal_challenges_after_candidate_freeze"] = reveal_after_preflight
    return namespace["run_replica"], {
        "patched_replicator_sha256": patched_sha,
        "loader_contract": "candidate file bytes checked against freeze; loaded tensor state checked against stage digest",
    }


def _run(workspace: Path) -> dict[str, Any]:
    packet_root = _packet_root(workspace)
    freeze = _verify_v4_packet(workspace)
    preflight_path = packet_root / "v4-preflight.json"
    runtime_path = packet_root / "runtime-preflight-v4.json"
    if not preflight_path.is_file() or not runtime_path.is_file():
        raise FileNotFoundError("V4 runtime and input preflight receipts are required")
    preflight = _read_json(preflight_path)
    if preflight.get("all_checks_pass") is not True or preflight.get("packet_sha256") != freeze["packet_sha256"]:
        raise ValueError("V4 preflight receipt is not a passing receipt for this packet")
    preflight_material = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
    if preflight.get("preflight_sha256") != _canonical_digest(preflight_material):
        raise ValueError("V4 preflight receipt canonical digest mismatch")
    if _sha_file(runtime_path) != _sha_bytes(
        (json.dumps(preflight["runtime_preflight"], indent=2, sort_keys=True) + "\n").encode("utf-8")
    ):
        raise ValueError("V4 runtime receipt differs from the passing preflight")

    source_root = workspace / "input" / "source-checkout"
    runner, _, runtime_receipt = _load_v9_runtime(workspace, freeze)
    primary, prior_roots, primary_report = _validate_primary(workspace, freeze, runner)
    replica = packet_root / "replica"
    if replica.exists():
        if any(replica.iterdir()):
            raise FileExistsError("V4 replica root is not empty; this packet permits one attempt")
    else:
        replica.mkdir(parents=True)

    replicate, patch_receipt = _patched_replica_runner(freeze, source_root, packet_root)
    started = time.monotonic()
    try:
        result = replicate(primary, replica, prior_roots)
        report = _read_json(primary / "outcome-manifest.json")
        replication_path = primary / "replication-manifest.json"
        replication = _read_json(replication_path)
        checks = replication.get("checks", {})
        base_status = str(result.get("reproducibility_status", "INCONCLUSIVE"))
        if base_status == "REPRODUCED_LOCAL" and report.get("outcome_status") == freeze["target"]["required_primary_outcome"] and all(checks.values()):
            final_status = "REPRODUCED_LOCAL"
        elif base_status == "DIVERGED" or not all(checks.values()) or report.get("outcome_status") != freeze["target"]["required_primary_outcome"]:
            final_status = "DIVERGED"
        else:
            final_status = "INCONCLUSIVE"
        runtime_detail = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": freeze["packet_sha256"],
            "runtime_preflight_sha256": _sha_file(runtime_path),
            "dependency_manifest_sha256": runtime_receipt["runtime"]["dependency_manifest_sha256"],
            "runtime_identity_verified": runtime_receipt["runtime_match"],
            "primary_dependency_manifest_sha256": _read_json(primary / "candidate-preparation.json")["runtime"]["dependency_manifest_sha256"],
            "replica_dependency_manifest_sha256": _read_json(replica / "candidate-preparation.json")["runtime"]["dependency_manifest_sha256"],
            "primary_and_replica_dependency_manifests_equal": (
                _read_json(primary / "candidate-preparation.json")["runtime"]["dependency_manifest_sha256"]
                == _read_json(replica / "candidate-preparation.json")["runtime"]["dependency_manifest_sha256"]
            ),
            "patched_replicator_sha256": patch_receipt["patched_replicator_sha256"],
        }
        for root in (primary, replica):
            path = root / "replication-manifest.json"
            bound = _read_json(path)
            bound["runtime_reproduction_v4"] = runtime_detail
            bound["reproducibility_status"] = final_status
            bound["assurance_level"] = "REPRODUCED_LOCAL" if final_status == "REPRODUCED_LOCAL" else "INCONCLUSIVE"
            bound["claim_ceiling_v4"] = freeze["claim_ceiling"]
            bound.pop("replication_sha256", None)
            bound["replication_sha256"] = _canonical_digest(bound)
            _write_json(path, bound)
        report = _read_json(primary / "outcome-manifest.json")
        report["reproducibility_status"] = final_status
        report["assurance_level"] = "REPRODUCED_LOCAL" if final_status == "REPRODUCED_LOCAL" else "INCONCLUSIVE"
        report["replication_manifest_sha256"] = _sha_file(primary / "replication-manifest.json")
        report["v4_packet_sha256"] = freeze["packet_sha256"]
        report.pop("report_sha256", None)
        report["report_sha256"] = _canonical_digest(report)
        _write_json(primary / "outcome-manifest.json", report)
        runner._write_results_document(primary, primary / "outcome-manifest.json")

        receipt = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "assurance_overlay": ASSURANCE_OVERLAY,
            "packet_sha256": freeze["packet_sha256"],
            "target_v9_packet_sha256": freeze["target"]["v9_packet_sha256"],
            "v3_packet_sha256": freeze["v3_packet_sha256"],
            "v3_stop_sha256": freeze["v3_stop_sha256"],
            "assurance_level": "REPRODUCED_LOCAL" if final_status == "REPRODUCED_LOCAL" else "INCONCLUSIVE",
            "reproducibility_status": final_status,
            "outcome_status": report["outcome_status"],
            "primary_outcome_manifest_sha256": _sha_file(primary / "outcome-manifest.json"),
            "replica_outcome_manifest_sha256": _sha_file(replica / "outcome-manifest.json"),
            "primary_replication_manifest_sha256": _sha_file(primary / "replication-manifest.json"),
            "replica_replication_manifest_sha256": _sha_file(replica / "replication-manifest.json"),
            "candidate_integrity_preflight_sha256": _sha_file(packet_root / "candidate-integrity-preflight-v4.json"),
            "runtime_preflight_sha256": _sha_file(runtime_path),
            "runtime_identity_verified": runtime_receipt["runtime_match"],
            "checks": checks,
            "candidate_refit_in_v4_runtime": True,
            "challenge_entropy_regenerated": False,
            "held_out_outcomes_opened_after_prediction_seal": True,
            "elapsed_wall_seconds": time.monotonic() - started,
            "reviewer_status": "NOT_SUBMITTED",
            "claim_ceiling": freeze["claim_ceiling"],
            "nonclaims": freeze["nonclaims"],
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        receipt["v4_receipt_sha256"] = _canonical_digest(receipt)
        _write_json(packet_root / "v4-run-receipt.json", receipt)
        result_out = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": freeze["packet_sha256"],
            "assurance_level": receipt["assurance_level"],
            "reproducibility_status": final_status,
            "outcome_status": report["outcome_status"],
            "v4_receipt_sha256": receipt["v4_receipt_sha256"],
            "primary_replication_manifest_sha256": receipt["primary_replication_manifest_sha256"],
            "checks_passed": sum(bool(value) for value in checks.values()),
            "checks_total": len(checks),
            "elapsed_wall_seconds": receipt["elapsed_wall_seconds"],
            "reviewer_status": "NOT_SUBMITTED",
        }
        return result_out
    except Exception as error:
        stop = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": freeze["packet_sha256"],
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": getattr(runner, "_ACTIVE_PHASE", "preflight_or_replication"),
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_wall_seconds": time.monotonic() - started,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["v4_stop_sha256"] = _canonical_digest(stop)
        _write_json(packet_root / "v4-run-stop.json", stop)
        return {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": freeze["packet_sha256"],
            "assurance_level": "INCONCLUSIVE",
            "reproducibility_status": "INCONCLUSIVE",
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": stop["stopped_phase"],
            "error_type": stop["error_type"],
            "v4_stop_sha256": stop["v4_stop_sha256"],
            "reviewer_status": "NOT_SUBMITTED",
        }
    finally:
        runner._ACTIVE_RUN_STARTED = None
        runner._ACTIVE_CASE_STARTED = None
        runner._ACTIVE_PHASE = "idle"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--workspace-root", type=Path, default=WORKSPACE_DEFAULT)
    args = parser.parse_args()
    try:
        result = _preflight(args.workspace_root) if args.command == "preflight" else _run(args.workspace_root)
    except Exception as error:
        workspace = args.workspace_root
        packet_root = _packet_root(workspace)
        freeze_path = packet_root / "freeze.json"
        freeze = _read_json(freeze_path) if freeze_path.is_file() else {}
        stop = {
            "protocol_id": PROTOCOL_ID,
            "state_slice": STATE_SLICE,
            "packet_sha256": freeze.get("packet_sha256"),
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": args.command,
            "error_type": type(error).__name__,
            "error": str(error),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["v4_stop_sha256"] = _canonical_digest(stop)
        _write_json(packet_root / f"v4-{args.command}-stop.json", stop)
        result = {
            "protocol_id": PROTOCOL_ID,
            "packet_sha256": freeze.get("packet_sha256"),
            "assurance_level": "INCONCLUSIVE",
            "reproducibility_status": "INCONCLUSIVE",
            "stop_status": "INCONCLUSIVE",
            "stopped_phase": args.command,
            "error_type": type(error).__name__,
            "v4_stop_sha256": stop["v4_stop_sha256"],
            "reviewer_status": "NOT_SUBMITTED",
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
