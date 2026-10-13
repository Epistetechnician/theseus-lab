"""Freeze the Cloud Run input and execution manifests before job creation."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


V8_PACKET_SHA256 = "b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2"
V8_PROTOCOL_SHA256 = "a1568874241ba4ddd609355857675ebad8e49f8d427a90e9029ce52f4750bfed"
TARGET_DIGEST = "87bcaba185c882f8f19201dfe3f8fe42a7e3e856d8ad6f4141d4683f5fb88f61"
ARTIFACT_MANIFEST_SHA256 = "06bfab2c91535924f8673de8752fe063690f0c8e95927070505caa4c46180d6f"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256(raw)


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _continuation_lineage(
    packet_path: Path,
    receipts_root: Path,
    reason: str | None = None,
) -> dict[str, Any]:
    packet_bytes = packet_path.read_bytes()
    previous_packet = json.loads(packet_bytes)
    packet_body = dict(previous_packet)
    declared_packet_sha = str(packet_body.pop("run_packet_sha256", ""))
    if declared_packet_sha != _canonical_sha256(packet_body):
        raise ValueError("previous run packet digest mismatch")
    previous_schema = previous_packet.get("schema")
    receipt_names_by_schema = {
        "theseus-v8-cloud-run-run-packet-v1": (
            "cloud-run-execution-dispatch.json",
            "cloud-run-execution-start.json",
            "cloud-run-job-describe-preexecution.json",
            "cloud-run-execution-final.json",
            "cloud-run-task-final.json",
            "cloud-run-logs-final.json",
        ),
        "theseus-v8-cloud-run-run-packet-v2": (
            "cloud-run-execution-dispatch-v2.json",
            "cloud-run-job-update-v2.json",
            "cloud-run-job-describe-preexecution-v2.json",
            "cloud-run-execution-start-v2.json",
            "cloud-run-execution-final-v2.json",
            "cloud-run-task-final-v2.json",
            "cloud-run-logs-final-v2.json",
            "cloud-run-logs-direct-v2.json",
            "input-manifest-object-v2.json",
            "run-packet-object-v2.json",
            "pinned-v7-object-after-v2.json",
        ),
    }
    receipt_names = receipt_names_by_schema.get(str(previous_schema))
    if receipt_names is None:
        raise ValueError("continuation must point to a supported frozen v1 or v2 run packet")
    if previous_packet.get("v8_packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("previous packet binds another v8 experiment")
    if previous_packet.get("target_digest") != TARGET_DIGEST:
        raise ValueError("previous packet binds another scientific target")

    receipts = []
    loaded: dict[str, Any] = {}
    for name in receipt_names:
        path = receipts_root / name
        data = path.read_bytes()
        receipts.append({"path": name, "bytes": len(data), "sha256": _sha256(data)})
        if name.endswith(".json") and not name.startswith("cloud-run-logs-"):
            loaded[name] = json.loads(data)

    execution = next(
        value for name, value in loaded.items() if name.startswith("cloud-run-execution-final")
    )
    task_rows = next(
        value for name, value in loaded.items() if name.startswith("cloud-run-task-final")
    )
    completed = next(
        (row for row in execution.get("status", {}).get("conditions", []) if row.get("type") == "Completed"),
        {},
    )
    task = task_rows[0] if isinstance(task_rows, list) and len(task_rows) == 1 else {}
    exit_code = task.get("status", {}).get("lastAttemptResult", {}).get("exitCode")
    if completed.get("status") != "False" or exit_code != 1:
        raise ValueError("previous run receipts do not show the captured one-shot failure")

    return {
        "schema": "theseus-v8-cloud-run-continuation-lineage-v1",
        "prior_run_packet_schema": previous_schema,
        "prior_run_packet_sha256": declared_packet_sha,
        "prior_execution_name": execution.get("metadata", {}).get("name"),
        "prior_completion_status": "EXECUTION_FAILED",
        "prior_task_exit_code": exit_code,
        "continuation_reason": reason or "Versioned continuation after a retained one-shot failure.",
        "prior_receipts": receipts,
    }


def _verify_input_object_preflight(
    path: Path | None,
    inputs: list[dict[str, Any]],
    continuation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if path is None:
        return None
    preflight = json.loads(path.read_text(encoding="utf-8"))
    expected_digest = str(preflight.pop("preflight_sha256", ""))
    if _canonical_sha256(preflight) != expected_digest:
        raise ValueError("input object preflight digest mismatch")
    rows = preflight.get("verified_objects")
    if not isinstance(rows, list):
        raise ValueError("input object preflight is missing verified object rows")
    expected_inputs = {row["path"]: row for row in inputs}
    actual_inputs = {str(row.get("path")): row for row in rows}
    if set(actual_inputs) != set(expected_inputs):
        raise ValueError("input object preflight does not cover the exact frozen input set")
    for relative, frozen in expected_inputs.items():
        checked = actual_inputs[relative]
        if checked.get("status") != "VERIFIED":
            raise ValueError(f"input object preflight is not verified: {relative}")
        if checked.get("bytes") != frozen["bytes"] or checked.get("sha256") != frozen["sha256"]:
            raise ValueError(f"input object preflight differs from frozen bytes: {relative}")
    if continuation is not None and (
        preflight.get("prior_run_packet_sha256") != continuation["prior_run_packet_sha256"]
        or preflight.get("prior_execution_name") != continuation["prior_execution_name"]
    ):
        raise ValueError("input object preflight does not bind the continued execution")
    preflight["preflight_sha256"] = expected_digest
    return preflight


def _archive_projection(path: Path, *, omit_replica_tree: bool = False) -> dict[str, int]:
    regular_files = 0
    uncompressed_bytes = 0
    selected_members = 0
    with tarfile.open(path, mode="r") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe path in staged archive: {path.name}")
            if omit_replica_tree and relative.parts and relative.parts[0] == "replica":
                continue
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unexpected non-regular staged archive member: {path.name}")
            selected_members += 1
            if member.isfile():
                regular_files += 1
                uncompressed_bytes += member.size
    return {
        "selected_member_count": selected_members,
        "regular_file_count": regular_files,
        "uncompressed_bytes": uncompressed_bytes,
    }


def freeze(
    input_source: Path,
    pinned_v7_archive: Path,
    source_manifest_path: Path,
    artifact_manifest_path: Path,
    target_path: Path,
    build_receipt_path: Path,
    build_log_path: Path,
    run_output: Path,
    input_manifest_output: Path,
    *,
    image_digest: str,
    project_id: str,
    region: str,
    job_name: str,
    artifact_image: str,
    build_staging_bucket: str,
    input_bucket: str,
    output_bucket: str,
    service_account_email: str,
    build_service_account_email: str,
    continuation_of_packet: Path | None = None,
    continuation_receipts: Path | None = None,
    continuation_reason: str | None = None,
    input_object_preflight_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(image_digest) != 71 or not image_digest.startswith("sha256:") or any(
        character not in "0123456789abcdef" for character in image_digest[7:]
    ):
        raise ValueError("container image digest must use sha256")
    if project_id != "secops1219" or region != "us-east1":
        raise ValueError("only the predeclared project and region are authorized")
    if input_bucket == output_bucket or build_staging_bucket in (input_bucket, output_bucket):
        raise ValueError("build, input, and output buckets must be separate")
    if not artifact_image.startswith(f"{region}-docker.pkg.dev/{project_id}/"):
        raise ValueError("artifact repository must be in the predeclared project and region")
    if not service_account_email.endswith(f"@{project_id}.iam.gserviceaccount.com"):
        raise ValueError("job service account must belong to the predeclared project")
    if not build_service_account_email.endswith(f"@{project_id}.iam.gserviceaccount.com"):
        raise ValueError("build service account must belong to the predeclared project")
    if (continuation_of_packet is None) != (continuation_receipts is None):
        raise ValueError("both continuation packet and receipt directory are required together")
    continuation = (
        _continuation_lineage(continuation_of_packet, continuation_receipts, continuation_reason)
        if continuation_of_packet is not None and continuation_receipts is not None
        else None
    )

    target = json.loads(target_path.read_text(encoding="utf-8"))
    if target.get("protocol_digest") != V8_PROTOCOL_SHA256:
        raise ValueError("target protocol digest mismatch")
    if _canonical_sha256(target) != TARGET_DIGEST:
        raise ValueError("target digest mismatch")
    artifact_bytes = artifact_manifest_path.read_bytes()
    artifact_file_sha = _sha256(artifact_bytes)
    if artifact_file_sha != ARTIFACT_MANIFEST_SHA256:
        raise ValueError("external artifact manifest digest mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_digest = source_manifest.get("container_source_sha256")
    source_body = dict(source_manifest)
    source_body.pop("container_source_sha256", None)
    if source_digest != _canonical_sha256(source_body):
        raise ValueError("container source manifest digest mismatch")
    if source_manifest.get("v8_packet_sha256") != V8_PACKET_SHA256:
        raise ValueError("container source manifest binds another v8 packet")
    build_receipt_bytes = build_receipt_path.read_bytes()
    build_receipt = json.loads(build_receipt_bytes)
    successful_build = build_receipt.get("successful_build", {})
    if build_receipt.get("container_source_sha256") != source_digest:
        raise ValueError("image build receipt binds another container source")
    if successful_build.get("status") != "SUCCESS" or successful_build.get("image_digest") != image_digest:
        raise ValueError("image build receipt does not bind the successful image digest")
    if build_receipt.get("build_service_account_email") != build_service_account_email:
        raise ValueError("image build receipt binds another build service account")
    if _digest_file(build_log_path) != successful_build.get("log_sha256"):
        raise ValueError("saved Cloud Build log digest differs from its receipt")

    repo_root = Path(__file__).resolve().parents[1]
    operator_sources = []
    for relative in (
        "cloud_run_v8_replay/Dockerfile",
        "cloud_run_v8_replay/cloud_run_entrypoint.py",
        "cloud_run_v8_replay/prepare_context.py",
        "cloud_run_v8_replay/freeze_run_packet.py",
        "docs/replacement-loss-v8-cloud-run-replay-v1.md",
    ):
        source = repo_root / relative
        data = source.read_bytes()
        operator_sources.append({
            "path": relative,
            "bytes": len(data),
            "sha256": _sha256(data),
        })

    raw_rows = json.loads(artifact_bytes)
    rows = raw_rows if isinstance(raw_rows, list) else raw_rows.get("entries", [])
    by_name = {Path(str(row.get("path", ""))).name: row for row in rows}
    inputs: list[dict[str, Any]] = []
    phase_projections: list[dict[str, Any]] = []
    for phase in (1, 2, 3):
        name = f"blind-replication-phase{phase}.tar"
        path = input_source / name
        row = by_name.get(name)
        if row is None:
            raise ValueError(f"artifact manifest is missing {name}")
        expected = str(row["sha256"])
        actual = _digest_file(path)
        if path.stat().st_size != int(row.get("bytes", path.stat().st_size)):
            raise ValueError(f"archive size differs from artifact manifest: {name}")
        if actual != expected:
            raise ValueError(f"archive digest differs from artifact manifest: {name}")
        projection = _archive_projection(path, omit_replica_tree=(phase == 3))
        phase_projections.append({"path": name, **projection})
        inputs.append({
            "path": name,
            "source_path": f"replacement-loss-v8-review-v1/{name}",
            "bytes": path.stat().st_size,
            "sha256": expected,
            **projection,
        })

    pinned_bytes = pinned_v7_archive.stat().st_size
    inputs.append({
        "path": "pinned-v7-verification.tar",
        "source_path": "pinned-v7-verification.tar",
        "bytes": pinned_bytes,
        "sha256": _digest_file(pinned_v7_archive),
        **_archive_projection(pinned_v7_archive),
    })

    input_preflight = _verify_input_object_preflight(
        input_object_preflight_path,
        inputs,
        continuation,
    )

    input_manifest: dict[str, Any] = {
        "schema": (
            "theseus-v8-cloud-run-input-manifest-v2"
            if continuation is not None
            else "theseus-v8-cloud-run-input-manifest-v1"
        ),
        "state_slices": [
            "theseus-lab-sandbox-v1",
            "global-experiment-assurance-lifecycle-v1",
        ],
        "target_digest": TARGET_DIGEST,
        "v8_packet_sha256": V8_PACKET_SHA256,
        "v8_protocol_sha256": V8_PROTOCOL_SHA256,
        "artifact_manifest_sha256": artifact_file_sha,
        "container_source_sha256": source_digest,
        "inputs": inputs,
        "transfer_authorization": "explicit user authorization for this same-owner Cloud Run replay",
    }
    if continuation is not None:
        input_manifest["continuation_of"] = continuation
    if input_preflight is not None:
        input_manifest["input_object_preflight"] = input_preflight
    input_manifest["input_manifest_sha256"] = _canonical_sha256(input_manifest)

    cost_boundary = {
        "projected_cloud_run_cpu_ceiling_usd": 0.75,
        "cpu_vcpu": 4,
        "memory_gib": 16,
        "task_timeout_seconds": 7200,
        "task_count": 1,
        "parallelism": 1,
        "max_retries": 0,
        "estimate_excludes": [
            "container build",
            "artifact registry storage",
            "cloud storage storage and operations",
            "logging",
        ],
        "billing_basis": "Tier 1 published Cloud Run rates; excludes any free-tier credit",
    }
    primary_report_path = input_source / "primary-report.json"
    primary_report_bytes = primary_report_path.read_bytes()
    primary_report = json.loads(primary_report_bytes)
    baseline_peak_rss = int(primary_report["runtime"]["peak_rss_bytes"])
    primary_uncompressed_bytes = sum(row["uncompressed_bytes"] for row in phase_projections)
    scratch_plus_baseline_reference = primary_uncompressed_bytes + baseline_peak_rss
    run_packet: dict[str, Any] = {
        "schema": (
            "theseus-v8-cloud-run-run-packet-v2"
            if continuation is not None
            else "theseus-v8-cloud-run-run-packet-v1"
        ),
        "state_slices": [
            "theseus-lab-sandbox-v1",
            "global-experiment-assurance-lifecycle-v1",
        ],
        "target_digest": TARGET_DIGEST,
        "v8_packet_sha256": V8_PACKET_SHA256,
        "v8_protocol_sha256": V8_PROTOCOL_SHA256,
        "artifact_manifest_sha256": artifact_file_sha,
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "container_source_sha256": source_digest,
        "container_image_digest": image_digest,
        "container_image_reference": f"{artifact_image}@{image_digest}",
        "operator_source_files": operator_sources,
        "operator_source_manifest_sha256": _canonical_sha256(operator_sources),
        "cloud_build_receipt_sha256": _sha256(build_receipt_bytes),
        "cloud_build_receipt": build_receipt,
        "project_id": project_id,
        "region": region,
        "job_name": job_name,
        "cloud_resources": {
            "artifact_repository": artifact_image.rsplit("/", 1)[0],
            "build_staging_bucket": build_staging_bucket,
            "input_bucket": input_bucket,
            "output_bucket": output_bucket,
            "job_service_account_email": service_account_email,
            "build_service_account_email": build_service_account_email,
            "volume_mounts": {
                "input": {
                    "mount_path": "/mnt/input",
                    "bucket": input_bucket,
                    "read_only": True,
                    "uid": 10001,
                    "gid": 10001,
                },
                "output": {
                    "mount_path": "/mnt/output",
                    "bucket": output_bucket,
                    "read_only": False,
                    "uid": 10001,
                    "gid": 10001,
                },
            },
            "storage_controls": {
                "uniform_bucket_level_access": True,
                "public_access_prevention": "enforced",
                "soft_delete_duration": "0s",
                "object_lifecycle_age_days": 3,
            },
            "iam_scope": {
                "job_input_role": "roles/storage.objectViewer",
                "job_output_role": "roles/storage.objectUser",
                "build_staging_role": "roles/storage.objectUser scoped to the build staging bucket",
                "build_artifact_role": "roles/artifactregistry.writer scoped to the dedicated repository",
            },
        },
        "resources": {
            "cpu_vcpu": 4,
            "memory_gib": 16,
            "task_timeout_seconds": 7200,
            "task_count": 1,
            "parallelism": 1,
            "max_retries": 0,
            "public_endpoint": False,
            "scheduled": False,
        },
        "working_storage_projection": {
            "workspace_path": "/tmp/theseus-v8-cloud-run-replay",
            "cloud_run_writable_filesystem": "memory-backed",
            "primary_archive_projections": phase_projections,
            "primary_uncompressed_bytes": primary_uncompressed_bytes,
            "primary_report_sha256": _sha256(primary_report_bytes),
            "prior_local_peak_rss_bytes": baseline_peak_rss,
            "sum_of_separate_local_rss_and_scratch_measurements_bytes": scratch_plus_baseline_reference,
            "cloud_run_memory_limit_bytes": 16 * 1024 * 1024 * 1024,
            "combined_cloud_run_peak_is_measured": False,
            "interpretation": "The sum is a sizing reference only. Cloud Run memory, FUSE buffers, and filesystem usage are not jointly measured before this one-shot execution; any resource failure is retained without retry.",
        },
        "acceptance_rule": "INTERNAL_PASS iff frozen evaluator outcome is PASS, all nine case statuses are PASS, no case blockers exist, and no resource blockers exist; strict v8 REPRODUCED_LOCAL remains the runner's unmodified result.",
        "repeat_count": 1,
        "stop_rule": "retain a failed or inconclusive run; do not retry under this packet",
        "claim_ceiling": "LOCAL_INDEPENDENT_AFFINE_PROJECTION_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION",
        "nonclaims": [
            "independent science",
            "all architectures",
            "whole-model mechanistic identity",
            "consciousness",
            "alignment",
            "production readiness",
        ],
        "reviewer_status": "EXTERNAL_REVIEW_UNAVAILABLE",
        "operator_authorization": "explicit user instruction to launch this cloud run; no external reviewer submission requested",
        "model_rights_note": "exact OPT and GPT-2 revisions have no declared license in pinned model metadata; user authorized temporary transfer to the same-owner project; no legal review claimed",
        "cost_boundary": cost_boundary,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if continuation is not None:
        run_packet["continuation_of"] = continuation
        run_packet["operator_authorization"] = (
            "explicit user instruction authorizes one versioned same-owner continuation after preserving "
            "the failed execution referenced by continuation_of; no external reviewer submission requested"
        )
    if input_preflight is not None:
        run_packet["input_object_preflight_sha256"] = input_preflight["preflight_sha256"]
    run_packet["run_packet_sha256"] = _canonical_sha256(run_packet)
    input_manifest_output.write_text(
        json.dumps(input_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    run_output.write_text(
        json.dumps(run_packet, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return input_manifest, run_packet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-source", required=True, type=Path)
    parser.add_argument("--pinned-v7-archive", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--build-receipt", required=True, type=Path)
    parser.add_argument("--build-log", required=True, type=Path)
    parser.add_argument("--run-packet-output", required=True, type=Path)
    parser.add_argument("--input-manifest-output", required=True, type=Path)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--project-id", default="secops1219")
    parser.add_argument("--region", default="us-east1")
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--artifact-image", required=True)
    parser.add_argument("--build-staging-bucket", required=True)
    parser.add_argument("--input-bucket", required=True)
    parser.add_argument("--output-bucket", required=True)
    parser.add_argument("--service-account-email", required=True)
    parser.add_argument("--build-service-account-email", required=True)
    parser.add_argument("--continuation-of-packet", type=Path)
    parser.add_argument("--continuation-receipts", type=Path)
    parser.add_argument("--continuation-reason")
    parser.add_argument("--input-object-preflight", type=Path)
    args = parser.parse_args()
    input_manifest, run_packet = freeze(
        args.input_source.expanduser().resolve(),
        args.pinned_v7_archive.expanduser().resolve(),
        args.source_manifest.expanduser().resolve(),
        args.artifact_manifest.expanduser().resolve(),
        args.target.expanduser().resolve(),
        args.build_receipt.expanduser().resolve(),
        args.build_log.expanduser().resolve(),
        args.run_packet_output.expanduser().resolve(),
        args.input_manifest_output.expanduser().resolve(),
        image_digest=args.image_digest,
        project_id=args.project_id,
        region=args.region,
        job_name=args.job_name,
        artifact_image=args.artifact_image,
        build_staging_bucket=args.build_staging_bucket,
        input_bucket=args.input_bucket,
        output_bucket=args.output_bucket,
        service_account_email=args.service_account_email,
        build_service_account_email=args.build_service_account_email,
        continuation_of_packet=(
            args.continuation_of_packet.expanduser().resolve()
            if args.continuation_of_packet
            else None
        ),
        continuation_receipts=(
            args.continuation_receipts.expanduser().resolve()
            if args.continuation_receipts
            else None
        ),
        continuation_reason=args.continuation_reason,
        input_object_preflight_path=(
            args.input_object_preflight.expanduser().resolve()
            if args.input_object_preflight
            else None
        ),
    )
    print(json.dumps({
        "target_digest": run_packet["target_digest"],
        "input_manifest_sha256": input_manifest["input_manifest_sha256"],
        "run_packet_sha256": run_packet["run_packet_sha256"],
        "container_source_sha256": run_packet["container_source_sha256"],
        "container_image_digest": run_packet["container_image_digest"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
