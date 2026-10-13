"""Fresh V15c packet adapter with complete V3 utility/runner bindings.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v15c-capacity-optimization-diagnostic-v1, and
global-experiment-assurance-lifecycle-v1. Prior failed receipts are bound as
lineage; new synthetic and candidate artifacts remain in external custody.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from typing import Any

import replacement_loss_v15_runner as base
import replacement_loss_v3_runner as v3_runner


PROTOCOL_ID = "theseus-replacement-loss-v15c-capacity-optimization-diagnostic"
PROTOCOL_SLICE = "theseus-replacement-loss-v15c-capacity-optimization-diagnostic-v1"
FREEZE_FILE = "docs/replacement-loss-v15c-freeze.json"
PRIMARY_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15c-capacity-optimization-primary"
REPLICA_NAME = "theseus-lab-sandbox-v1-replacement-loss-v15c-capacity-optimization-replica"
SOURCE_V15B_FAILED_ATTEMPT = {
    "protocol_id": "theseus-replacement-loss-v15b-capacity-optimization-diagnostic",
    "state_slice": "theseus-lab-sandbox-v1",
    "protocol_slice": "theseus-replacement-loss-v15b-capacity-optimization-diagnostic-v1",
    "packet_sha256": "1fbf6d62ff00c984e3181991a0b9d61863d3dcbec4f62ddcec031dc59fafeabc",
    "freeze_sha256": "30a612094ca36a99addd08e2b31f0808ef863b79b9d72864d09965044b32df1a",
    "archive_sha256": "6757e3c82ec949bd503d004fd0165edc3ebc6553c7795384740df39cdf1d4cb5",
    "run_stop_sha256": "d5241768d2049b4f30fa10073d0e81fa21f39313c9008b2200bd1ef11cf3e6c7",
    "phase_ledger_sha256": "9413e16436333cbbf2e5c62167848e1b38417eb1ccc2af3a969d995eff3143bd",
    "visible_fixture_file_sha256": "bf32c724f0cd3c3ea65a1ad485194bb0eb6e0d78b137c3276e8fc6cb0e3db907",
    "visible_fixture_example_set_sha256": "b0ab8677b2418a32f49ef878382bfd38dd83f200da7eb44207c9e2b6cad2f014",
    "target_selection_sha256": "020aa9b9f74d897d38d960807c4db91af5fe4fb17a1da1fd61afb4a02c7f970f",
    "status": "INCONCLUSIVE",
    "error_type": "AttributeError",
    "elapsed_seconds": 1.8217951249998805,
    "visible_fixture_generated": True,
    "adaptation_fixture_generation_started": False,
    "adaptation_capture_started": False,
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
    base.fixtures.PROMPT_RULES_VERSION = "v15c-adaptation-capacity-optimization-synthetic-v1"

    source_path = Path(base.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    referenced = sorted({
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "v3_core"
    })
    bindings = []
    for name in referenced:
        if hasattr(base.v3_core, name):
            continue
        if not hasattr(v3_runner, name):
            raise RuntimeError(f"V15c cannot resolve V3 dependency ownership for {name}")
        setattr(base.v3_core, name, getattr(v3_runner, name))
        bindings.append(name)
    if not bindings:
        raise RuntimeError("V15c expected to bind runner-owned V3 helpers")
    base._V15C_V3_CORE_RUNNER_BINDINGS = tuple(bindings)
    binding_sha256 = hashlib.sha256("\n".join(bindings).encode("utf-8")).hexdigest()

    base.PACKET_FILES = tuple(dict.fromkeys((
        *base.PACKET_FILES,
        "docs/replacement-loss-v15-freeze.json",
        "docs/replacement-loss-v15-results.md",
        "docs/replacement-loss-v15b-capacity-optimization-diagnostic.md",
        "docs/replacement-loss-v15b-freeze.json",
        "docs/replacement-loss-v15b-results.md",
        "replacement_loss_v15b_runner.py",
        "docs/replacement-loss-v15c-capacity-optimization-diagnostic.md",
        "replacement_loss_v15c_runner.py",
    )))

    original_freeze = base.freeze_packet
    original_verify = base.verify_packet
    original_diagnose = base.diagnose

    def freeze_packet() -> dict[str, Any]:
        packet = original_freeze()
        material = {key: value for key, value in packet.items() if key != "freeze_sha256"}
        material["source_v15b_failed_attempt"] = SOURCE_V15B_FAILED_ATTEMPT
        material["v3_core_runner_bindings"] = bindings
        material["v3_core_runner_bindings_sha256"] = binding_sha256
        material["freeze_sha256"] = base._digest(material)
        base._write_json(base._REPO_ROOT / FREEZE_FILE, material)
        return material

    def verify_packet() -> dict[str, Any]:
        packet = original_verify()
        if (
            packet.get("source_v15b_failed_attempt") != SOURCE_V15B_FAILED_ATTEMPT
            or packet.get("v3_core_runner_bindings") != bindings
            or packet.get("v3_core_runner_bindings_sha256") != binding_sha256
        ):
            raise ValueError("V15c freeze does not bind the exact V15b receipt and V3 owner bindings")
        return packet

    def diagnose(custody_path: str | Path) -> dict[str, Any]:
        report = original_diagnose(custody_path)
        report["source_revision"] = "202f459d6d89df9a038d78d58e0af825d287fcd8+v15c-frozen-working-tree"
        report["source_worktree_state"] = "uncommitted; executable bytes are bound by the V15c packet digest"
        report["v3_core_runner_bindings"] = bindings
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
