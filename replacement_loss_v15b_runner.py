"""Fresh V15b packet adapter for the V15 rank/optimization diagnostic.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v15b-capacity-optimization-diagnostic-v1, and
global-experiment-assurance-lifecycle-v1. The V15 failed attempt is bound as
lineage; all new synthetic and candidate artifacts remain in external custody.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import replacement_loss_v15_runner as base
import replacement_loss_v3_runner as v3_runner


PROTOCOL_ID = "theseus-replacement-loss-v15b-capacity-optimization-diagnostic"
PROTOCOL_SLICE = "theseus-replacement-loss-v15b-capacity-optimization-diagnostic-v1"
FREEZE_FILE = "docs/replacement-loss-v15b-freeze.json"
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15b-capacity-optimization-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15b-capacity-optimization-replica"
SOURCE_V15_FAILED_ATTEMPT = {
    "protocol_id": "theseus-replacement-loss-v15-capacity-optimization-diagnostic",
    "state_slice": "theseus-lab-sandbox-v1",
    "protocol_slice": "theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1",
    "packet_sha256": "d72a6fd520a185fbfeacf63e46c55ec43ef273017fd2bc8ab103b8165efbc042",
    "freeze_sha256": "faec9cfc11e50608f6e5d43cf15860d5580f0427688db3443e0151a35f703d1d",
    "archive_sha256": "7d8850dbd0f9261717fa4352be1d7069c1d01c2b1734e3a956dc6e6241065176",
    "run_stop_sha256": "9594780fc937d6ccd92e87dc00f1e85f65078ec1c5699d67f973d69b401ef9e4",
    "phase_ledger_sha256": "c94e3ad2c2952b669f830b4caaf473ef122342e79f1143eb8e57f13a358a99ae",
    "target_selection_sha256": "020aa9b9f74d897d38d960807c4db91af5fe4fb17a1da1fd61afb4a02c7f970f",
    "status": "INCONCLUSIVE",
    "error_type": "AttributeError",
    "elapsed_seconds": 0.018893542000114394,
    "fixture_generation_started": False,
    "candidate_training_started": False,
}


def _configure() -> None:
    base.PROTOCOL_ID = PROTOCOL_ID
    base.PROTOCOL_SLICE = PROTOCOL_SLICE
    base.FREEZE_FILE = FREEZE_FILE
    base.PRIMARY_NAME = PRIMARY_NAME
    base.REPLICA_NAME = REPLICA_NAME
    base.fixtures.PROTOCOL_ID = PROTOCOL_ID
    base.fixtures.PROTOCOL_SLICE = PROTOCOL_SLICE
    base.fixtures.PROMPT_RULES_VERSION = "v15b-adaptation-capacity-optimization-synthetic-v1"

    # These public runner functions own the Checkpoint type and local model loader.
    base.v3_core.Checkpoint = v3_runner.Checkpoint
    base.v3_core._load_local_model = v3_runner._load_local_model

    base.PACKET_FILES = tuple(dict.fromkeys((
        *base.PACKET_FILES,
        "docs/replacement-loss-v15-freeze.json",
        "docs/replacement-loss-v15-results.md",
        "docs/replacement-loss-v15b-capacity-optimization-diagnostic.md",
        "replacement_loss_v15b_runner.py",
    )))

    original_freeze = base.freeze_packet
    original_verify = base.verify_packet
    original_diagnose = base.diagnose

    def freeze_packet() -> dict[str, Any]:
        packet = original_freeze()
        material = {key: value for key, value in packet.items() if key != "freeze_sha256"}
        material["source_v15_failed_attempt"] = SOURCE_V15_FAILED_ATTEMPT
        material["freeze_sha256"] = base._digest(material)
        base._write_json(base._REPO_ROOT / FREEZE_FILE, material)
        return material

    def verify_packet() -> dict[str, Any]:
        packet = original_verify()
        if packet.get("source_v15_failed_attempt") != SOURCE_V15_FAILED_ATTEMPT:
            raise ValueError("V15b freeze does not bind the exact V15 failed attempt")
        return packet

    def diagnose(custody_path: str | Path) -> dict[str, Any]:
        report = original_diagnose(custody_path)
        report["source_revision"] = "202f459d6d89df9a038d78d58e0af825d287fcd8+v15b-frozen-working-tree"
        report["source_worktree_state"] = "uncommitted; executable bytes are bound by the V15b packet digest"
        report.pop("report_sha256", None)
        report["report_sha256"] = base._digest(report)
        base._write_json(Path(custody_path) / "diagnostic-report.json", report)
        return report

    base.freeze_packet = freeze_packet
    base.verify_packet = verify_packet
    base.diagnose = diagnose


_configure()


if __name__ == "__main__":
    raise SystemExit(base.main())
