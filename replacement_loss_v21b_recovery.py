"""Digest-bound path recovery for the frozen V21 pre-data stop.

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v21b-run-recovery-v1.

This wrapper changes only external custody-root constants and the runner path
used to launch the unchanged V21 source in the pinned runtimes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import replacement_loss_v20_runner as v20
import replacement_loss_v20b_runner as v20b
import replacement_loss_v21_runner as v21


STATE_SLICE = "theseus-lab-sandbox-v1"
RECOVERY_SLICE = "theseus-replacement-loss-v21b-run-recovery-v1"
REPO_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = Path("/Volumes/PrimaryED/ResearchArtifacts")
V21_FAILED_ROOT = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v21-development-primary"
V20B_PRIMARY_ROOT = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v20b-development-primary"
RECOVERY_FREEZE_FILE = "docs/replacement-loss-v21b-recovery-freeze.json"
RECOVERY_DOC = "docs/replacement-loss-v21b-run-recovery.md"
RECOVERY_FILES = (
    "AGENTS.md",
    "docs/causal-effect-identifiability-boundary-v1.md",
    "docs/replacement-loss-v21-development-freeze.json",
    "docs/replacement-loss-v21-paired-endpoint-supervision.md",
    "replacement_loss_v21_fixtures.py",
    "replacement_loss_v21_runner.py",
    "requirements-v21.lock.txt",
    RECOVERY_DOC,
    "replacement_loss_v21b_recovery.py",
)
DEV_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21b-development-primary"
DEV_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21b-development-replica"
CONFIRM_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21b-confirmation-primary"
CONFIRM_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v21b-confirmation-replica"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path.name}")
    return value


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for relative in RECOVERY_FILES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"recovery packet source is missing: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    return rows


def _failed_attempt() -> dict[str, Any]:
    packet = v21.verify_development_packet()
    stop_path = V21_FAILED_ROOT / "run-stop.json"
    archive_path = V21_FAILED_ROOT / "frozen-packet-manifest.json"
    stop, archive = _read(stop_path), _read(archive_path)
    if (
        stop.get("protocol_id") != v21.DEV_PROTOCOL_ID
        or stop.get("packet_sha256") != packet["packet_sha256"]
        or stop.get("status") != "INCONCLUSIVE"
        or stop.get("error_type") != "AttributeError"
        or stop.get("error") != "module 'replacement_loss_v20b_runner' has no attribute 'DEV_PRIMARY_NAME'"
        or archive.get("packet_sha256") != packet["packet_sha256"]
    ):
        raise ValueError("V21 stop receipt or packet archive differs from the observed pre-data failure")
    inventory = []
    for path in sorted(V21_FAILED_ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(V21_FAILED_ROOT).as_posix()
        inventory.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha(path)})
    forbidden_prefixes = ("fixtures/", "splits/", "adaptation/", "candidates/", "predictions/", "outcomes/")
    if any(row["path"].startswith(forbidden_prefixes) for row in inventory):
        raise ValueError("V21 failed root contains data artifacts; recovery must preserve and audit them first")
    v20b_parent = v20b.verify_development_packet()
    return {
        "protocol_id": packet["protocol_id"],
        "packet_sha256": packet["packet_sha256"],
        "freeze_sha256": packet["freeze_sha256"],
        "failed_root_stop_sha256": _sha(stop_path),
        "failed_root_archive_manifest_sha256": _sha(archive_path),
        "failed_root_inventory": inventory,
        "failed_root_inventory_sha256": _digest(inventory),
        "no_v21_data_generated": True,
        "v20b_parent_packet_sha256": v20b_parent["packet_sha256"],
        "v20b_primary_root": V20B_PRIMARY_ROOT.name,
    }


def freeze_recovery() -> dict[str, Any]:
    path = REPO_ROOT / RECOVERY_FREEZE_FILE
    if path.exists():
        return verify_recovery()
    parent = _failed_attempt()
    rows = _source_rows()
    packet = {
        "protocol_id": "theseus-replacement-loss-v21b-run-recovery",
        "state_slice": STATE_SLICE,
        "protocol_slice": RECOVERY_SLICE,
        "assurance_overlay": "global-experiment-assurance-lifecycle-v1",
        "claim_ceiling": v21.CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "parent_failed_attempt": parent,
        "execution_identity": {
            "v20b_primary_root": V20B_PRIMARY_ROOT.name,
            "development_primary_root": DEV_PRIMARY_NAME,
            "development_replica_root": DEV_REPLICA_NAME,
            "confirmation_primary_root": CONFIRM_PRIMARY_NAME,
            "confirmation_replica_root": CONFIRM_REPLICA_NAME,
            "v21_scientific_packet_unchanged": True,
            "fresh_output_roots_required": True,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    payload = json.dumps(packet, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(payload)
    return packet


def verify_recovery() -> dict[str, Any]:
    frozen = _read(REPO_ROOT / RECOVERY_FREEZE_FILE)
    rows, parent = _source_rows(), _failed_attempt()
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != "theseus-replacement-loss-v21b-run-recovery"
        or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != RECOVERY_SLICE
        or frozen.get("packet_files") != rows
        or frozen.get("packet_sha256") != _digest(rows)
        or frozen.get("parent_failed_attempt") != parent
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V21B recovery freeze differs from its sources or failed parent")
    return frozen


def _configure_v21() -> None:
    v20b.DEV_PRIMARY_NAME = V20B_PRIMARY_ROOT.name
    v21.DEV_PRIMARY_NAME = DEV_PRIMARY_NAME
    v21.DEV_REPLICA_NAME = DEV_REPLICA_NAME
    v21.CONFIRM_PRIMARY_NAME = CONFIRM_PRIMARY_NAME
    v21.CONFIRM_REPLICA_NAME = CONFIRM_REPLICA_NAME
    original_install = v21._install

    def install_recovery() -> None:
        original_install()
        v20._python_exec = _python_exec

    v21._install = install_recovery


def _python_exec(executable: Path, arguments: list[str], *, log_path: Path) -> dict[str, Any]:
    if not executable.is_file():
        raise FileNotFoundError(f"pinned V21 runtime is missing: {executable}")
    completed = subprocess.run(
        [str(executable), str(Path(__file__).resolve()), *arguments],
        check=False, capture_output=True, text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"V21B pinned runtime exited {completed.returncode}; see external-custody log {log_path.name}")
    decoder = json.JSONDecoder()
    for index, character in enumerate(completed.stdout):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(completed.stdout[index:])
        except json.JSONDecodeError:
            continue
        if not completed.stdout[index + end :].strip() and isinstance(value, dict):
            return value
    raise ValueError("V21B pinned runtime did not return one JSON receipt")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args == ["freeze-recovery"]:
        print(json.dumps(freeze_recovery(), sort_keys=True, indent=2))
        return 0
    if args == ["verify-recovery"]:
        print(json.dumps(verify_recovery(), sort_keys=True, indent=2))
        return 0
    verify_recovery()
    _configure_v21()
    return v21.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
