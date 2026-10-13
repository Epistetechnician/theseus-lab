"""Run a digest-bound, same-owner v8 replay inside a private Cloud Run Job."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


V8_PACKET_SHA256 = "b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2"
V8_PROTOCOL_SHA256 = "a1568874241ba4ddd609355857675ebad8e49f8d427a90e9029ce52f4750bfed"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(raw)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _replica_metric_evidence(replica: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    evaluation_path = replica / "replacement-loss-v3-report.json"
    if evaluation_path.is_file():
        return _read_json(evaluation_path), _read_json(replica / "replication-report.json")

    replication_path = replica / "replication-report.json"
    if not replication_path.is_file():
        raise FileNotFoundError(
            "replica has neither a held-out v3 evaluation report nor an early-stop replication report"
        )
    return None, _read_json(replication_path)


def _verify_container_source(root: Path) -> dict[str, Any]:
    manifest_path = root / "container-source-manifest.json"
    manifest = _read_json(manifest_path)
    expected = str(manifest.pop("container_source_sha256"))
    if _canonical_sha256(manifest) != expected:
        raise ValueError("container source manifest digest mismatch")
    if manifest.get("v8_packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("container source manifest binds another v8 packet")
    for row in manifest.get("source_files", []):
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe source manifest path")
        path = root.joinpath(*relative.parts)
        data = path.read_bytes()
        if len(data) != row["bytes"] or _sha256_bytes(data) != row["sha256"]:
            raise ValueError(f"container source digest mismatch: {relative.name}")
    manifest["container_source_sha256"] = expected
    return manifest


def _verify_run_packet(input_root: Path) -> dict[str, Any]:
    packet = _read_json(input_root / "cloud-run-run-packet.json")
    expected = str(packet.pop("run_packet_sha256"))
    if _canonical_sha256(packet) != expected:
        raise ValueError("run packet digest mismatch")
    if packet.get("v8_packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("run packet binds another v8 packet")
    if packet.get("v8_protocol_sha256") != V8_PROTOCOL_SHA256:
        raise ValueError("run packet binds another v8 protocol")
    if packet.get("project_id") != "secops1219" or packet.get("region") != "us-east1":
        raise ValueError("run packet does not bind the configured Cloud Run target")
    if packet.get("schema") == "theseus-v8-cloud-run-run-packet-v2":
        continuation = packet.get("continuation_of")
        if (
            not isinstance(continuation, dict)
            or continuation.get("schema") != "theseus-v8-cloud-run-continuation-lineage-v1"
            or continuation.get("prior_completion_status") != "EXECUTION_FAILED"
            or continuation.get("prior_task_exit_code") != 1
        ):
            raise ValueError("v2 run packet is missing its validated failed-run lineage")
        if packet.get("repeat_count") != 1 or packet.get("resources", {}).get("max_retries") != 0:
            raise ValueError("continuation run must remain one-shot with zero retries")
    packet["run_packet_sha256"] = expected
    return packet


def _verify_inputs(input_root: Path, run_packet: dict[str, Any]) -> dict[str, Any]:
    manifest = _read_json(input_root / "cloud-run-input-manifest.json")
    expected = str(manifest.pop("input_manifest_sha256"))
    if _canonical_sha256(manifest) != expected:
        raise ValueError("input manifest digest mismatch")
    if manifest.get("v8_packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("input manifest binds another v8 packet")
    if manifest.get("target_digest") != run_packet.get("target_digest"):
        raise ValueError("input manifest and run packet bind different targets")
    if manifest.get("artifact_manifest_sha256") != run_packet.get("artifact_manifest_sha256"):
        raise ValueError("input manifest and run packet bind different artifact manifests")
    if manifest.get("container_source_sha256") != run_packet.get("container_source_sha256"):
        raise ValueError("input manifest and run packet bind different container sources")
    if expected != run_packet.get("input_manifest_sha256"):
        raise ValueError("run packet binds another input manifest")
    if manifest.get("continuation_of") != run_packet.get("continuation_of"):
        raise ValueError("input manifest and run packet bind different continuation lineage")
    for row in manifest.get("inputs", []):
        relative = PurePosixPath(str(row["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe input manifest path")
        path = input_root.joinpath(*relative.parts)
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"input file missing or size mismatch: {relative.name}")
        if _sha256_file(path) != row["sha256"]:
            raise ValueError(f"input file digest mismatch: {relative.name}")
    manifest["input_manifest_sha256"] = expected
    return manifest


def _extract_tar(archive_path: Path, destination: Path, *, omit_replica_tree: bool = False) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r") as archive:
        members = archive.getmembers()
        selected: list[tarfile.TarInfo] = []
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe path in frozen archive")
            if omit_replica_tree and relative.parts and relative.parts[0] == "replica":
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError("unexpected non-regular member in frozen archive")
            selected.append(member)
        archive.extractall(path=destination, members=selected, filter="data")
        return len(selected)


def _dependency_inventory() -> tuple[str, list[str]]:
    entries = sorted(
        f"{distribution.metadata.get('Name', distribution.name)}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
    )
    payload = "\n".join(entries) + "\n"
    return _sha256_bytes(payload.encode("utf-8")), entries


def _file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "output-manifest.json":
            relative = path.relative_to(root).as_posix()
            rows.append({
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            })
    return rows


def _runtime_details() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "safetensors", "tokenizers", "torch", "transformers"):
        packages[package] = importlib.metadata.version(package)
    try:
        import torch

        thread_count = int(torch.get_num_threads())
    except Exception:
        thread_count = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": thread_count,
        "packages": packages,
    }


def run() -> dict[str, Any]:
    started = time.monotonic()
    image_digest = os.environ.get("V8_REPLAY_IMAGE_DIGEST", "")
    input_root = Path(os.environ.get("V8_REPLAY_INPUT_ROOT", "/mnt/input"))
    output_root = Path(os.environ.get("V8_REPLAY_OUTPUT_ROOT", "/mnt/output"))
    if not image_digest.startswith("sha256:"):
        raise ValueError("missing immutable container image digest")

    source_manifest = _verify_container_source(Path(__file__).resolve().parent)
    run_packet = _verify_run_packet(input_root)
    if run_packet.get("container_image_digest") != image_digest:
        raise ValueError("deployed image digest does not match frozen run packet")
    if run_packet.get("container_image_reference", "").split("@")[-1] != image_digest:
        raise ValueError("frozen container reference does not bind the deployed image digest")
    runtime_job_name = os.environ.get("CLOUD_RUN_JOB")
    if runtime_job_name and run_packet.get("job_name") != runtime_job_name:
        raise ValueError("running Cloud Run Job does not match the frozen run packet")
    if run_packet.get("container_source_sha256") != source_manifest["container_source_sha256"]:
        raise ValueError("container source manifest does not match frozen run packet")
    input_manifest = _verify_inputs(input_root, run_packet)

    work = Path("/tmp/theseus-v8-cloud-run-replay")
    if work.exists():
        shutil.rmtree(work)
    primary = work / "primary"
    replica = work / "replica"
    pinned_v7 = work / "pinned-v7"
    extracted_members = 0
    for phase in (1, 2, 3):
        extracted_members += _extract_tar(
            input_root / f"blind-replication-phase{phase}.tar",
            primary,
            omit_replica_tree=(phase == 3),
        )
    _extract_tar(input_root / "pinned-v7-verification.tar", pinned_v7)

    # The frozen v7 verifier reads its pinned v6 archive from this container-local path.
    v6_fixed_root = Path("/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v6")
    v6_fixed_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pinned_v7 / "v6-frozen-packet-manifest.json", v6_fixed_root / "v6-frozen-packet-manifest.json")
    v6_archive = pinned_v7 / "v6-frozen-packet"
    v6_destination = v6_fixed_root / "v6-frozen-packet"
    if v6_destination.exists():
        shutil.rmtree(v6_destination)
    shutil.copytree(v6_archive, v6_destination)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import replacement_loss_v8_runner as v8_runner

    runner_result = v8_runner.replicate_packet(
        primary,
        replica,
        primary / "checkpoints.json",
        pinned_v7,
    )
    v3_report, replication_report = _replica_metric_evidence(replica)
    v8_report = _read_json(replica / "replacement-loss-v8-replication-report.json")
    cases = v3_report.get("cases", []) if v3_report is not None else []
    replication_blockers = replication_report.get("blockers", [])
    metric_pass = (
        v3_report is not None
        and v3_report.get("outcome_status") == "PASS"
        and len(cases) == 9
        and all(row.get("case_status") == "PASS" and not row.get("blockers") for row in cases)
        and not v3_report.get("resource_blockers")
    )
    if runner_result.get("packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("replication result binds another v8 packet")

    output_parent = output_root / "replacement-loss-v8-cloud-run-replay-v2"
    output_parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(replica, output_parent / "replica")
    shutil.copyfile(
        primary / "replacement-loss-v8-report.json",
        output_parent / "primary-replacement-loss-v8-report.json",
    )
    shutil.copyfile(input_root / "cloud-run-input-manifest.json", output_parent / "cloud-run-input-manifest.json")
    shutil.copyfile(input_root / "cloud-run-run-packet.json", output_parent / "cloud-run-run-packet.json")

    dependency_sha, dependency_lines = _dependency_inventory()
    (output_parent / "dependency-inventory.txt").write_text(
        "\n".join(dependency_lines) + "\n",
        encoding="utf-8",
    )
    runtime = _runtime_details()
    status = "INTERNAL_PASS" if metric_pass else "INCONCLUSIVE"
    result = {
        "schema": "theseus-v8-cloud-run-replay-result-v2",
        "state_slices": [
            "theseus-lab-sandbox-v1",
            "global-experiment-assurance-lifecycle-v1",
        ],
        "protocol_id": "theseus-replacement-loss-v8-ridge-primary-causal",
        "target_digest": run_packet["target_digest"],
        "v8_packet_sha256": V8_PACKET_SHA256,
        "v8_protocol_sha256": V8_PROTOCOL_SHA256,
        "artifact_manifest_sha256": run_packet["artifact_manifest_sha256"],
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "run_packet_sha256": run_packet["run_packet_sha256"],
        "run_packet_schema": run_packet.get("schema"),
        "continuation_of": run_packet.get("continuation_of"),
        "container_source_sha256": source_manifest["container_source_sha256"],
        "container_image_digest": image_digest,
        "cloud_run_job": os.environ.get("CLOUD_RUN_JOB"),
        "cloud_run_execution": os.environ.get("CLOUD_RUN_EXECUTION"),
        "cloud_run_task_index": os.environ.get("CLOUD_RUN_TASK_INDEX"),
        "assurance_level": status,
        "replication_stage_status": (
            "HELD_OUT_EVALUATION_COMPLETED"
            if v3_report is not None
            else "INCONCLUSIVE_BEFORE_HELD_OUT_EVALUATION"
        ),
        "experiment_outcome_status": (
            v3_report.get("outcome_status")
            if v3_report is not None
            else "INCOMPLETE_BEFORE_HELD_OUT_EVALUATION"
        ),
        "strict_v8_reproducibility_status": runner_result.get("assurance_level", "INCONCLUSIVE"),
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "same_owner_cloud_runtime": True,
        "independent_science_or_acceptance": False,
        "metric_acceptance_rule": "INTERNAL_PASS requires the frozen evaluator outcome PASS, exactly nine case_status PASS rows, no case blockers, and no resource blockers.",
        "metric_case_count": len(cases),
        "metric_case_pass_count": sum(row.get("case_status") == "PASS" for row in cases),
        "metric_blocker_count": (
            sum(len(row.get("blockers", [])) for row in cases)
            if v3_report is not None
            else len(replication_blockers)
        ),
        "metric_evaluation_report_available": v3_report is not None,
        "replication_blockers": replication_blockers,
        "replication_report_sha256": _sha256_file(replica / "replication-report.json"),
        "resource_blockers": (
            v3_report.get("resource_blockers", []) if v3_report is not None else None
        ),
        "strict_replication_checks": {
            key: v8_report.get(key)
            for key in (
                "candidate_reconstruction_match",
                "prediction_and_outcome_array_digests_match",
                "dependencies_match",
                "distinct_clean_runtime",
                "source_identity_match",
            )
        },
        "claim_ceiling": (
            v3_report.get("claim_ceiling") if v3_report is not None else run_packet["claim_ceiling"]
        ),
        "nonclaims": (
            v3_report.get("nonclaims", []) if v3_report is not None else run_packet["nonclaims"]
        ),
        "runtime": runtime,
        "dependency_inventory_sha256": dependency_sha,
        "extracted_primary_member_count": extracted_members,
        "elapsed_wall_seconds": round(time.monotonic() - started, 3),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cost_boundary": run_packet["cost_boundary"],
    }
    (output_parent / "cloud-run-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_manifest = {
        "schema": "theseus-v8-cloud-run-output-manifest-v2",
        "run_packet_sha256": run_packet["run_packet_sha256"],
        "files": _file_manifest(output_parent),
    }
    output_manifest["output_manifest_sha256"] = _canonical_sha256(output_manifest)
    (output_parent / "output-manifest.json").write_text(
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "assurance_level": status,
        "strict_v8_reproducibility_status": result["strict_v8_reproducibility_status"],
        "metric_case_count": result["metric_case_count"],
        "metric_case_pass_count": result["metric_case_pass_count"],
        "output_manifest_sha256": output_manifest["output_manifest_sha256"],
        "run_packet_sha256": run_packet["run_packet_sha256"],
    }
    print(json.dumps(summary, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
