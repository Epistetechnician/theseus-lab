"""Digest-bound continuation adapter for the frozen V20 experiment.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v20b-low-rank-causal-residual-continuation-v1, and
global-experiment-assurance-lifecycle-v1.

This file binds the immutable V20 runner and installs only the zero-output
direct-call adapter and fresh V20B packet identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import replacement_loss_v20_fixtures as base_fixtures
import replacement_loss_v20_runner as v20
import replacement_loss_v20b_fixtures as fixtures


STATE_SLICE = "theseus-lab-sandbox-v1"
PROTOCOL_SLICE = fixtures.PROTOCOL_SLICE
ASSURANCE_OVERLAY = "global-experiment-assurance-lifecycle-v1"
DEV_PROTOCOL_ID = fixtures.DEV_PROTOCOL_ID
CONFIRM_PROTOCOL_ID = fixtures.CONFIRM_PROTOCOL_ID
DEV_FREEZE_FILE = "docs/replacement-loss-v20b-development-freeze.json"
CONFIRM_FREEZE_FILE = "docs/replacement-loss-v20b-confirmation-freeze.json"
CONFIRM_CONFIG_FILE = "docs/replacement-loss-v20b-confirmation-config.json"
DEV_RESULT_JSON = "docs/replacement-loss-v20b-development-results.json"
DEV_RESULT_MD = "docs/replacement-loss-v20b-development-results.md"
DEV_REPLICATION_JSON = "docs/replacement-loss-v20b-development-replication.json"
CONFIRM_RESULT_JSON = "docs/replacement-loss-v20b-confirmation-results.json"
CONFIRM_RESULT_MD = "docs/replacement-loss-v20b-confirmation-results.md"
CONFIRM_REPLICATION_JSON = "docs/replacement-loss-v20b-confirmation-replication.json"
CLAIM_CEILING = "LOCAL_LOW_RANK_CAUSAL_RESIDUAL_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT"
REPO_ROOT = Path(__file__).resolve().parent
RESEARCH_ROOT = v20.RESEARCH_ROOT
V20_FAILED_ROOT = RESEARCH_ROOT / "theseus-lab-sandbox-v1-replacement-loss-v20-development-primary"
V20_FREEZE_FILE = "docs/replacement-loss-v20-development-freeze.json"
V20_PARENT_FILES = (
    V20_FREEZE_FILE,
    "docs/replacement-loss-v20-low-rank-causal-residual.md",
    "replacement_loss_v20_fixtures.py",
    "replacement_loss_v20_runner.py",
    "requirements-v20.lock.txt",
)
PROTOCOL_FILE = "docs/replacement-loss-v20b-low-rank-causal-residual.md"
FIXTURE_FILE = "replacement_loss_v20b_fixtures.py"
RUNNER_FILE = "replacement_loss_v20b_runner.py"
LOCK_FILE = "requirements-v20b.lock.txt"
DEV_PACKET_FILES = ("AGENTS.md", PROTOCOL_FILE, FIXTURE_FILE, RUNNER_FILE, LOCK_FILE, *V20_PARENT_FILES)
CONFIRM_PACKET_FILES = (
    "AGENTS.md", PROTOCOL_FILE, FIXTURE_FILE, RUNNER_FILE, LOCK_FILE,
    DEV_FREEZE_FILE, CONFIRM_CONFIG_FILE, DEV_RESULT_JSON, DEV_RESULT_MD, DEV_REPLICATION_JSON,
    *V20_PARENT_FILES,
)
_ORIGINAL_REFERENCE: Any = None
_PATCHED_REFERENCE = False
_ORIGINAL_LOAD: Any = v20._load_dependencies


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _write_json(path: Path, value: Any, *, immutable: bool = False) -> None:
    v20._write_json(path, value, immutable=immutable)


def _rows_for(files: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for relative in files:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"V20B frozen source is missing: {relative}")
        rows.append({"path": relative, "sha256": _sha_file(path), "bytes": path.stat().st_size})
    return rows


def _failed_v20_binding() -> dict[str, Any]:
    parent = v20._verify_frozen_file(V20_FREEZE_FILE)
    if parent.get("protocol_id") != "theseus-replacement-loss-v20-development":
        raise ValueError("V20B parent packet identity differs")
    if V20_FAILED_ROOT.is_symlink() or not V20_FAILED_ROOT.is_dir():
        raise FileNotFoundError("the failed V20 external custody root is missing or unsafe")
    stop_path = V20_FAILED_ROOT / "run-stop.json"
    stop = _read_json(stop_path)
    if (
        stop.get("protocol_id") != parent.get("protocol_id")
        or stop.get("packet_sha256") != parent.get("packet_sha256")
        or stop.get("status") != "INCONCLUSIVE"
        or stop.get("error_type") != "NotImplementedError"
        or stop.get("error") != 'Module [NoOpCandidate] is missing the required "forward" function'
    ):
        raise ValueError("the V20 stop receipt differs from the observed pre-score control-adapter failure")
    artifacts = []
    for path in sorted(V20_FAILED_ROOT.rglob("*")):
        if not path.is_file() or "frozen-packet" in path.relative_to(V20_FAILED_ROOT).parts:
            continue
        relative = path.relative_to(V20_FAILED_ROOT).as_posix()
        artifacts.append({"path": relative, "bytes": path.stat().st_size, "sha256": _sha_file(path)})
    if not any(row["path"] == "fixtures/adaptation-seed-2.json" for row in artifacts):
        raise ValueError("aborted V20 adaptation fixtures are incomplete; refuse to continue")
    return {
        "parent_protocol_id": parent["protocol_id"],
        "parent_packet_sha256": parent["packet_sha256"],
        "parent_freeze_sha256": parent["freeze_sha256"],
        "failed_run_stop_sha256": _sha_file(stop_path),
        "failed_run_status": "INCONCLUSIVE",
        "failed_run_phase": stop.get("phase"),
        "failed_run_error_type": stop.get("error_type"),
        "failed_run_error": stop.get("error"),
        "aborted_artifacts": artifacts,
        "aborted_artifacts_sha256": _digest(artifacts),
        "aborted_v20_data_reused": False,
        "correction": "add a direct-call zero-output adapter for the inherited no-op control",
    }


def freeze_development() -> dict[str, Any]:
    path = REPO_ROOT / DEV_FREEZE_FILE
    if path.exists():
        return verify_development_packet()
    failure = _failed_v20_binding()
    rows = _rows_for(DEV_PACKET_FILES)
    packet = {
        "protocol_id": DEV_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "parent_continuation": failure,
        "method_and_gates_unchanged_from_v20": True,
        "fresh_development_seeds": list(fixtures.DEVELOPMENT_SEEDS),
        "fresh_confirmation_seeds": list(fixtures.CONFIRMATION_SEEDS),
        "target_identity": {
            "model_id": "grushaprasad/opt125m-rootp-surprisal-curriculum",
            "revision": "8af2a2926bc772599ccb3b4e7ad7b40b35259a63",
            "tree_sha256": "6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466",
            "weight_sha256": "aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132",
            "fresh_checkpoint_claim": False,
            "selected_layer": 8,
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_development_packet() -> dict[str, Any]:
    frozen = _read_json(REPO_ROOT / DEV_FREEZE_FILE)
    failure = _failed_v20_binding()
    current = _rows_for(DEV_PACKET_FILES)
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != DEV_PROTOCOL_ID or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current) or frozen.get("parent_continuation") != failure
        or frozen.get("method_and_gates_unchanged_from_v20") is not True
        or frozen.get("fresh_development_seeds") != list(fixtures.DEVELOPMENT_SEEDS)
        or frozen.get("fresh_confirmation_seeds") != list(fixtures.CONFIRMATION_SEEDS)
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V20B development packet source bytes or failure-continuation binding differ")
    return frozen


def freeze_confirmation() -> dict[str, Any]:
    development = verify_development_packet()
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    if (
        result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
        or replay.get("scientific_results_sha256") != result.get("scientific_results_sha256")
    ):
        raise ValueError("V20B confirmation requires qualified development and exact local replay")
    config = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "development_packet_sha256": development["packet_sha256"],
        "development_freeze_sha256": development["freeze_sha256"],
        "development_result_sha256": result["scientific_results_sha256"],
        "development_replication_sha256": replay["report_sha256"],
        "seeds": list(fixtures.CONFIRMATION_SEEDS),
        "challenge_unlock_condition": "primary and replica preparations are byte-identical and every predeclared cell passes",
    }
    config["config_sha256"] = _digest(config)
    _write_json(REPO_ROOT / CONFIRM_CONFIG_FILE, config, immutable=True)
    path = REPO_ROOT / CONFIRM_FREEZE_FILE
    if path.exists():
        return verify_confirmation_packet()
    rows = _rows_for(CONFIRM_PACKET_FILES)
    packet = {
        "protocol_id": CONFIRM_PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "protocol_slice": PROTOCOL_SLICE,
        "assurance_overlay": ASSURANCE_OVERLAY,
        "claim_ceiling": CLAIM_CEILING,
        "packet_files": rows,
        "packet_sha256": _digest(rows),
        "development_binding": {
            "packet_sha256": development["packet_sha256"],
            "result_sha256": result["scientific_results_sha256"],
            "replication_sha256": replay["report_sha256"],
            "qualification_status": "QUALIFIED",
        },
    }
    packet["freeze_sha256"] = _digest(packet)
    _write_json(path, packet, immutable=True)
    return packet


def verify_confirmation_packet() -> dict[str, Any]:
    development = verify_development_packet()
    config = _read_json(REPO_ROOT / CONFIRM_CONFIG_FILE)
    result = _read_json(REPO_ROOT / DEV_RESULT_JSON)
    replay = _read_json(REPO_ROOT / DEV_REPLICATION_JSON)
    config_material = {key: value for key, value in config.items() if key != "config_sha256"}
    if (
        config.get("config_sha256") != _digest(config_material)
        or result.get("qualification_status") != "QUALIFIED"
        or result.get("reproducibility_status") != "REPRODUCED_LOCAL"
        or replay.get("assurance_level") != "REPRODUCED_LOCAL"
    ):
        raise ValueError("V20B confirmation entry gate is invalid")
    frozen = _read_json(REPO_ROOT / CONFIRM_FREEZE_FILE)
    current = _rows_for(CONFIRM_PACKET_FILES)
    expected_binding = {
        "packet_sha256": development["packet_sha256"],
        "result_sha256": result["scientific_results_sha256"],
        "replication_sha256": replay["report_sha256"],
        "qualification_status": "QUALIFIED",
    }
    material = {key: value for key, value in frozen.items() if key != "freeze_sha256"}
    if (
        frozen.get("protocol_id") != CONFIRM_PROTOCOL_ID or frozen.get("state_slice") != STATE_SLICE
        or frozen.get("protocol_slice") != PROTOCOL_SLICE or frozen.get("assurance_overlay") != ASSURANCE_OVERLAY
        or frozen.get("claim_ceiling") != CLAIM_CEILING or frozen.get("packet_files") != current
        or frozen.get("packet_sha256") != _digest(current) or frozen.get("development_binding") != expected_binding
        or config.get("development_packet_sha256") != development["packet_sha256"]
        or frozen.get("freeze_sha256") != _digest(material)
    ):
        raise ValueError("V20B confirmation packet source bytes or development binding differ")
    return frozen


def _packet(stage: str):
    if stage == "development":
        return DEV_PROTOCOL_ID, DEV_FREEZE_FILE, DEV_PACKET_FILES, verify_development_packet()
    if stage == "confirmation":
        return CONFIRM_PROTOCOL_ID, CONFIRM_FREEZE_FILE, CONFIRM_PACKET_FILES, verify_confirmation_packet()
    raise ValueError("stage must be development or confirmation")


def _python_exec(executable: Path, arguments: list[str], *, log_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(executable), str(Path(__file__).resolve()), *arguments],
        check=False, capture_output=True, text=True,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"V20B pinned runtime exited {completed.returncode}; see external-custody log {log_path.name}")
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
    raise ValueError("V20B pinned runtime did not return one JSON receipt")


def _load_dependencies() -> None:
    global _ORIGINAL_REFERENCE, _PATCHED_REFERENCE
    _ORIGINAL_LOAD()
    import torch

    def no_op_forward(self: Any, inputs: Any) -> Any:
        output_dim = int(self.fc2.output_dim)
        return torch.zeros((*inputs.shape[:-1], output_dim), dtype=inputs.dtype, device=inputs.device)

    v20.v16.NoOpCandidate.forward = no_op_forward
    if not _PATCHED_REFERENCE:
        _ORIGINAL_REFERENCE = v20.v19._reference_payloads

        def reference_payloads():
            visible, prior = _ORIGINAL_REFERENCE()
            aborted = [
                _read_json(V20_FAILED_ROOT / "fixtures" / f"adaptation-seed-{index}.json") for index in range(3)
            ]
            return visible, [*prior, *aborted]

        v20.v19._reference_payloads = reference_payloads
        _PATCHED_REFERENCE = True


def _rewrite_results_markdown(function, *args, **kwargs):
    payload = function(*args, **kwargs)
    for relative in (DEV_RESULT_MD, CONFIRM_RESULT_MD):
        path = REPO_ROOT / relative
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            text = text.replace("Replacement-loss V20 ", "Replacement-loss V20B ")
            text = text.replace("theseus-replacement-loss-v20-low-rank-causal-residual-v1", PROTOCOL_SLICE)
            path.write_text(text, encoding="utf-8")
    return payload


def _install() -> None:
    v20.STATE_SLICE = STATE_SLICE
    v20.PROTOCOL_SLICE = PROTOCOL_SLICE
    v20.ASSURANCE_OVERLAY = ASSURANCE_OVERLAY
    v20.DEV_PROTOCOL_ID = DEV_PROTOCOL_ID
    v20.CONFIRM_PROTOCOL_ID = CONFIRM_PROTOCOL_ID
    v20.DEV_FREEZE_FILE = DEV_FREEZE_FILE
    v20.CONFIRM_FREEZE_FILE = CONFIRM_FREEZE_FILE
    v20.CONFIRM_CONFIG_FILE = CONFIRM_CONFIG_FILE
    v20.DEV_RESULT_JSON = DEV_RESULT_JSON
    v20.DEV_RESULT_MD = DEV_RESULT_MD
    v20.DEV_REPLICATION_JSON = DEV_REPLICATION_JSON
    v20.CONFIRM_RESULT_JSON = CONFIRM_RESULT_JSON
    v20.CONFIRM_RESULT_MD = CONFIRM_RESULT_MD
    v20.CONFIRM_REPLICATION_JSON = CONFIRM_REPLICATION_JSON
    v20.CLAIM_CEILING = CLAIM_CEILING
    v20.DEV_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20b-development-primary"
    v20.DEV_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20b-development-replica"
    v20.CONFIRM_PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20b-confirmation-primary"
    v20.CONFIRM_REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v20b-confirmation-replica"
    v20.PROTOCOL_FILE = PROTOCOL_FILE
    v20.FIXTURE_FILE = FIXTURE_FILE
    v20.RUNNER_FILE = RUNNER_FILE
    v20.LOCK_FILE = LOCK_FILE
    v20.fixtures = fixtures
    v20._load_dependencies = _load_dependencies
    v20._python_exec = _python_exec
    v20._packet = _packet
    v20.freeze_development = freeze_development
    v20.verify_development_packet = verify_development_packet
    v20.freeze_confirmation = freeze_confirmation
    v20.verify_confirmation_packet = verify_confirmation_packet
    v20._write_development_results = lambda replay, primary: _rewrite_results_markdown(v20._write_development_results_original, replay, primary)
    v20._write_confirmation_results = lambda replay, **kwargs: _rewrite_results_markdown(v20._write_confirmation_results_original, replay, **kwargs)


def main(argv: list[str] | None = None) -> int:
    if not hasattr(v20, "_write_development_results_original"):
        v20._write_development_results_original = v20._write_development_results
        v20._write_confirmation_results_original = v20._write_confirmation_results
    _install()
    return v20.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
