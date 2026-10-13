"""Clean-runtime same-owner replication for replacement-loss-v9.

State slice: theseus-lab-sandbox-v1. Held-out fixtures are copied only after
the replica independently refits and freezes its candidate modules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import replacement_loss_v9_runner as runner


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path.name}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_rows(root: Path, relative_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"path": str(row["path"]), "sha256": _sha_file(root / str(row["path"]))}
        for row in relative_rows
    ]


def _copy_frozen_inputs(primary: Path, replica: Path) -> None:
    for relative in ("v9-frozen-packet",):
        source = primary / relative
        if not source.is_dir():
            raise FileNotFoundError(f"primary frozen packet archive missing: {relative}")
        shutil.copytree(source, replica / relative)
    for relative in (
        "v9-frozen-packet-manifest.json",
        "v9-checkpoint-manifest.json",
        "prior-checkpoint-identity-inventory.json",
    ):
        source = primary / relative
        if not source.is_file():
            raise FileNotFoundError(f"primary frozen input missing: {relative}")
        shutil.copy2(source, replica / relative)
    manifest = _read_json(replica / "v9-checkpoint-manifest.json")
    for entry in manifest.get("checkpoints", []):
        relative = Path(str(entry["local_path"]))
        source = primary / relative
        target = replica / relative
        if not source.is_dir():
            raise FileNotFoundError(f"primary custodied checkpoint missing: {entry['family']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target, copy_function=shutil.copy2)
    runner._load_checkpoint_specs(replica, runner.verify_packet(replica)["packet_sha256"])


def _reveal_challenges_after_candidate_freeze(primary: Path, replica: Path) -> dict[str, Any]:
    runner._verify_candidate_freeze(primary, runner.verify_packet(primary)["packet_sha256"])
    replica_freeze_path = replica / "candidate-freeze.json"
    if not replica_freeze_path.is_file():
        raise ValueError("replica candidate state must be frozen before held-out fixtures are revealed")
    primary_manifest_path = primary / "challenge-manifest.json"
    primary_manifest = _read_json(primary_manifest_path)
    primary_manifest_digest = primary_manifest.get("challenge_manifest_sha256")
    primary_material = {key: value for key, value in primary_manifest.items() if key != "challenge_manifest_sha256"}
    if primary_manifest_digest != _canonical_digest(primary_material):
        raise ValueError("primary challenge manifest digest is invalid")
    if primary_manifest.get("candidate_freeze_sha256") != _sha_file(primary / "candidate-freeze.json"):
        raise ValueError("primary challenges are not bound to the primary candidate freeze")
    for relative in ("challenges",):
        source = primary / relative
        if not source.is_dir():
            raise FileNotFoundError(f"primary held-out fixture directory missing: {relative}")
        shutil.copytree(source, replica / relative)
    replica_fixtures = replica / "fixtures"
    replica_fixtures.mkdir(parents=True, exist_ok=True)
    for source in sorted((primary / "fixtures").glob("challenge-seed-*.json")):
        shutil.copy2(source, replica_fixtures / source.name)
    for relative in ("v9-final-disjointness-receipt.json",):
        source = primary / relative
        if source.is_file():
            shutil.copy2(source, replica / relative)
    replica_manifest = dict(primary_manifest)
    replica_manifest["primary_challenge_manifest_sha256"] = primary_manifest_digest
    replica_manifest["primary_candidate_freeze_sha256"] = _sha_file(primary / "candidate-freeze.json")
    replica_manifest["candidate_freeze_sha256"] = _sha_file(replica_freeze_path)
    replica_manifest["revealed_after_replica_candidate_freeze"] = True
    replica_manifest.pop("challenge_manifest_sha256", None)
    replica_manifest["challenge_manifest_sha256"] = _canonical_digest(replica_manifest)
    _write_json(replica / "challenge-manifest.json", replica_manifest)
    return {
        "primary_challenge_manifest_sha256": primary_manifest_digest,
        "replica_challenge_manifest_sha256": replica_manifest["challenge_manifest_sha256"],
        "challenge_case_count": len(replica_manifest.get("cases", [])),
        "revealed_after_replica_candidate_freeze": True,
    }


def _run_replica_impl(
    primary_root: str | Path,
    replica_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    started = time.monotonic()
    primary = runner._custody_root(primary_root)
    replica = runner._custody_root(replica_root)
    if primary == replica:
        raise ValueError("primary and replica custody roots must be distinct")
    if not (primary / "outcome-manifest.json").is_file():
        raise FileNotFoundError("primary outcome report is not complete")
    if replica.exists() and any(replica.iterdir()):
        raise FileExistsError("replica custody must be empty before replication")
    replica.mkdir(parents=True, exist_ok=True)
    runner._ACTIVE_RUN_STARTED = started
    runner._ACTIVE_PHASE = "clean_local_replication"
    packet = runner.verify_packet(primary)
    _copy_frozen_inputs(primary, replica)

    # Refit from visible/adaptation data without reading primary candidates or outcomes.
    runner.prepare_candidates(replica, prior_custody_roots)
    runner.freeze_candidates(replica)

    primary_prep = _read_json(primary / "candidate-preparation.json")
    replica_prep = _read_json(replica / "candidate-preparation.json")
    candidate_match = _artifact_rows(primary, primary_prep["candidate_files"]) == _artifact_rows(
        replica, replica_prep["candidate_files"]
    )
    challenge_transfer = _reveal_challenges_after_candidate_freeze(primary, replica)
    runner.seal_predictions(replica)
    runner.evaluate_outcomes(replica)

    primary_prep = _read_json(primary / "candidate-preparation.json")
    replica_prep = _read_json(replica / "candidate-preparation.json")
    primary_predictions = _read_json(primary / "prediction-manifest.json")
    replica_predictions = _read_json(replica / "prediction-manifest.json")
    primary_report = _read_json(primary / "outcome-manifest.json")
    replica_report = _read_json(replica / "outcome-manifest.json")

    def prediction_identities(manifest: dict[str, Any]) -> list[dict[str, str]]:
        rows = [*manifest["predictions"], *manifest["controls"]]
        return sorted(
            ({"path": str(row["prediction_path"]), "sha256": str(row["prediction_sha256"])} for row in rows),
            key=lambda row: row["path"],
        )

    def outcome_identities(report: dict[str, Any]) -> list[dict[str, str]]:
        return sorted(
            ({"key": f"{row['family']}:{row['seed_index']}:{row['method']}:{row['stage_index']}", "sha256": str(row["outcome_sha256"])} for row in report["results"]),
            key=lambda row: row["key"],
        )

    primary_checkpoints = _read_json(primary / "v9-checkpoint-manifest.json")["checkpoints"]
    replica_checkpoints = _read_json(replica / "v9-checkpoint-manifest.json")["checkpoints"]
    checks = {
        "source_identity": runner._source_identity() == primary_report["source_identity"],
        "checkpoint_manifest": _sha_file(primary / "v9-checkpoint-manifest.json") == _sha_file(replica / "v9-checkpoint-manifest.json"),
        "checkpoint_identities": primary_checkpoints == replica_checkpoints,
        "dependency_manifest": primary_prep["runtime"]["dependency_manifest_sha256"] == replica_prep["runtime"]["dependency_manifest_sha256"],
        "visible_and_adaptation_fixtures": (
            primary_prep["visible_fixture_sha256"] == replica_prep["visible_fixture_sha256"]
            and primary_prep["adaptation_fixture_sha256"] == replica_prep["adaptation_fixture_sha256"]
        ),
        "candidate_states": candidate_match,
        "challenge_fixtures": False,
        "prediction_arrays": prediction_identities(primary_predictions) == prediction_identities(replica_predictions),
        "outcome_arrays": outcome_identities(primary_report) == outcome_identities(replica_report),
        "outcome_summaries": _canonical_digest(primary_report["results"]) == _canonical_digest(replica_report["results"]),
        "outcome_status": primary_report["outcome_status"] == replica_report["outcome_status"],
    }
    # The challenge fixture comparison uses the actual challenge manifests.
    primary_challenge_cases = _read_json(primary / "challenge-manifest.json").get("cases", [])
    replica_challenge_cases = _read_json(replica / "challenge-manifest.json").get("cases", [])
    checks["challenge_fixtures"] = sorted(
        (row["family"], row["seed_index"], row["fixture_sha256"], row["interventions_file_sha256"])
        for row in primary_challenge_cases
    ) == sorted(
        (row["family"], row["seed_index"], row["fixture_sha256"], row["interventions_file_sha256"])
        for row in replica_challenge_cases
    )
    checks["control_outcome_metrics"] = _canonical_digest(primary_report["control_results"]) == _canonical_digest(replica_report["control_results"])
    reproducibility = "REPRODUCED_LOCAL" if all(checks.values()) else "DIVERGED"
    receipt = {
        "protocol_id": runner.PROTOCOL_ID,
        "state_slice": runner.STATE_SLICE,
        "packet_sha256": packet["packet_sha256"],
        "assurance_level": "REPRODUCED_LOCAL" if reproducibility == "REPRODUCED_LOCAL" else "INCONCLUSIVE",
        "reproducibility_status": reproducibility,
        "primary_report_sha256_before_replication": _sha_file(primary / "outcome-manifest.json"),
        "replica_report_sha256": _sha_file(replica / "outcome-manifest.json"),
        "candidate_state_match": candidate_match,
        "challenge_transfer": challenge_transfer,
        "checks": checks,
        "elapsed_wall_seconds": time.monotonic() - started,
        "replica_custody_storage": runner.v3._custody_storage_bytes(replica),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "claim_ceiling": "LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS",
        "reviewer_status": "NOT_SUBMITTED",
        "known_limitations": ["same-owner clean-runtime replication is not independent scientific acceptance"],
    }
    receipt["replication_sha256"] = _canonical_digest(receipt)
    _write_json(replica / "replication-manifest.json", receipt)
    primary_replication_path = primary / "replication-manifest.json"
    _write_json(primary_replication_path, receipt)
    primary_report["reproducibility_status"] = reproducibility
    primary_report["replication_manifest_sha256"] = _sha_file(primary_replication_path)
    primary_report.pop("report_sha256", None)
    primary_report["report_sha256"] = _canonical_digest(primary_report)
    _write_json(primary / "outcome-manifest.json", primary_report)
    runner._write_results_document(primary, primary / "outcome-manifest.json")
    runner._ACTIVE_RUN_STARTED = None
    runner._ACTIVE_PHASE = "idle"
    return {
        "reproducibility_status": reproducibility,
        "assurance_level": receipt["assurance_level"],
        "checks": checks,
        "replication_sha256": receipt["replication_sha256"],
        "elapsed_wall_seconds": receipt["elapsed_wall_seconds"],
    }


def run_replica(
    primary_root: str | Path,
    replica_root: str | Path,
    prior_custody_roots: list[str | Path],
) -> dict[str, Any]:
    primary = runner._custody_root(primary_root)
    replica = runner._custody_root(replica_root)
    try:
        return _run_replica_impl(primary, replica, prior_custody_roots)
    except Exception as error:
        stop = {
            "protocol_id": runner.PROTOCOL_ID,
            "state_slice": runner.STATE_SLICE,
            "stop_status": "INCONCLUSIVE",
            "error_type": type(error).__name__,
            "error": str(error),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        stop["stop_sha256"] = _canonical_digest(stop)
        if replica.exists() and not (replica / "replication-manifest.json").exists():
            _write_json(replica / "replication-stop.json", stop)
        report_path = primary / "outcome-manifest.json"
        if report_path.is_file():
            report = _read_json(report_path)
            report["reproducibility_status"] = "INCONCLUSIVE"
            report["replication_stop_sha256"] = stop["stop_sha256"]
            report.pop("report_sha256", None)
            report["report_sha256"] = _canonical_digest(report)
            _write_json(report_path, report)
            runner._write_results_document(primary, report_path)
        runner._ACTIVE_RUN_STARTED = None
        runner._ACTIVE_CASE_STARTED = None
        runner._ACTIVE_PHASE = "idle"
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", required=True)
    parser.add_argument("--replica-root", required=True)
    parser.add_argument("--prior-custody-root", action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(run_replica(args.primary_root, args.replica_root, args.prior_custody_root), sort_keys=True))


if __name__ == "__main__":
    main()
