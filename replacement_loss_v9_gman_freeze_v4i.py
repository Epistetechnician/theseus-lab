#!/usr/bin/env python3
"""Build the canonical V4i continuation freeze from frozen V4h lineage.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i, and
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i"
STATE_SLICE = PROTOCOL_ID
PARENT_FREEZE = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-freeze.json"
PARENT_RECEIPT = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-terminal.json"
PARENT_RESULTS = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-results.md"
PARENT_FREEZE_COPY = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4i-parent-v4h-freeze.json"
OUTPUT = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4i-freeze.json"
PREVIOUS_OUTPUT = OUTPUT
RATE_USD_PER_MINUTE = 0.00804
AGGREGATE_CAP_USD = 5.0
PRE_V4H_REGISTER_USD = 1.167
V4H_SETTLED_USD = 0.639
EVALUATOR_MAX_MINUTES = 300
NODE_MAX_MINUTES = 315


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_digest(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_row(packet_path: str, source_path: str) -> dict[str, Any]:
    path = ROOT / source_path
    content = path.read_bytes()
    return {"path": packet_path, "source_path": source_path, "bytes": len(content), "sha256": sha_bytes(content)}


def main() -> None:
    parent_bytes = PARENT_FREEZE.read_bytes()
    parent = json.loads(parent_bytes)
    terminal_bytes = PARENT_RECEIPT.read_bytes()
    terminal = json.loads(terminal_bytes)
    if parent.get("protocol_id") != "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h":
        raise ValueError("V4h parent protocol identity mismatch")
    parent_material = {key: value for key, value in parent.items() if key != "packet_sha256"}
    if parent.get("packet_sha256") != canonical_digest(parent_material):
        raise ValueError("V4h parent canonical packet digest mismatch")
    if (
        terminal.get("packet_sha256") != parent.get("packet_sha256")
        or terminal.get("reproducibility_status") != "INCONCLUSIVE_TIMEOUT"
        or terminal.get("evaluator", {}).get("exit_code") != 124
        or terminal.get("accounting", {}).get("settled_v4h_mission_usd") != V4H_SETTLED_USD
    ):
        raise ValueError("V4h terminal receipt is not the bound timeout result")

    expected_parent_files = (
        ("wrapper", "replacement_loss_v9_gman_fresh_node_replicator_v4h.py"),
        ("importer", "replacement_loss_v9_gman_import_v4h.py"),
        ("primary_report_repair", "replacement_loss_v9_gman_repair_primary_v4h.py"),
    )
    for key, source in expected_parent_files:
        row = parent[key]
        actual = ROOT / source
        if actual.stat().st_size != row["bytes"] or sha_file(actual) != row["sha256"]:
            raise ValueError(f"V4h frozen parent helper differs from its original freeze: {source}")

    ROOT_FILES = [
        ("protocol.md", "docs/replacement-loss-v9-gman-fresh-node-v4i.md"),
        ("runner.py", "replacement_loss_v9_gman_fresh_node_replicator_v4i.py"),
        ("bootstrap.sh", "replacement_loss_v9_gman_bootstrap_v4i.sh"),
        ("importer.py", "replacement_loss_v9_gman_import_v4i.py"),
        ("repair_primary.py", "replacement_loss_v9_gman_repair_primary_v4i.py"),
        ("write_timeout_receipt.py", "replacement_loss_v9_gman_timeout_receipt_v4i.py"),
        ("watchdog.sh", "replacement_loss_v9_gman_watchdog_v4i.sh"),
        ("parent_runner_v4h.py", "replacement_loss_v9_gman_fresh_node_replicator_v4h.py"),
        ("parent_importer_v4h.py", "replacement_loss_v9_gman_import_v4h.py"),
        ("parent_repair_v4h.py", "replacement_loss_v9_gman_repair_primary_v4h.py"),
        ("parent-v4h-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4i-parent-v4h-freeze.json"),
        ("parent-v4h-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4h-terminal.json"),
    ]
    PARENT_FREEZE_COPY.write_bytes(parent_bytes)
    rows = [file_row(packet_path, source_path) for packet_path, source_path in ROOT_FILES]
    row_by_packet_path = {row["path"]: row for row in rows}
    parent_runner = row_by_packet_path["parent_runner_v4h.py"]
    parent_importer = row_by_packet_path["parent_importer_v4h.py"]
    parent_repair = row_by_packet_path["parent_repair_v4h.py"]

    freeze = json.loads(json.dumps(parent))
    freeze["protocol_id"] = PROTOCOL_ID
    freeze["state_slice"] = STATE_SLICE
    freeze["repository_state_slice"] = "theseus-lab-sandbox-v1"
    freeze["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    freeze["freeze_state"] = "FROZEN_BEFORE_V4I_EVALUATION_AFTER_V4H_TERMINAL_TIMEOUT"
    prior_v4i_freeze = None
    prior_v4i_superseded: list[dict[str, Any]] = []
    if PREVIOUS_OUTPUT.is_file():
        candidate = json.loads(PREVIOUS_OUTPUT.read_text(encoding="utf-8"))
        if candidate.get("protocol_id") == PROTOCOL_ID and candidate.get("packet_sha256"):
            prior_v4i_freeze = candidate["packet_sha256"]
            prior_v4i_superseded = list(candidate.get("superseded_prelaunch_packets", []))
    if prior_v4i_freeze and prior_v4i_freeze != freeze.get("packet_sha256"):
        superseded = list(freeze.get("superseded_prelaunch_packets", []))
        known = {row.get("packet_sha256") for row in superseded}
        for row in prior_v4i_superseded:
            if row.get("packet_sha256") not in known:
                superseded.append(row)
                known.add(row.get("packet_sha256"))
        if not any(row.get("packet_sha256") == prior_v4i_freeze for row in superseded):
            superseded.append({
                "packet_sha256": prior_v4i_freeze,
                "reason": "GMan rejected max_duration_minutes for the active-node detached launch before command creation; replaced with a frozen in-packet watchdog before evaluator launch",
            })
        freeze["superseded_prelaunch_packets"] = superseded
    freeze["claim_ceiling"] = "same-owner local cross-runtime reproduction of frozen V9 FAIL under the digest-bound V4i Linux runtime"
    freeze["reviewer_status"] = "NOT_SUBMITTED"
    freeze["packet_document"] = {key: value for key, value in row_by_packet_path["protocol.md"].items() if key != "source_path"}
    freeze["packet_document"]["path"] = "docs/replacement-loss-v9-gman-fresh-node-v4i.md"
    freeze["wrapper"] = {key: value for key, value in row_by_packet_path["runner.py"].items() if key != "source_path"}
    freeze["bootstrap"] = {key: value for key, value in row_by_packet_path["bootstrap.sh"].items() if key != "source_path"}
    freeze["importer"] = {key: value for key, value in row_by_packet_path["importer.py"].items() if key != "source_path"}
    freeze["primary_report_repair"] = {key: value for key, value in row_by_packet_path["repair_primary.py"].items() if key != "source_path"}
    freeze["parent_helpers"] = {
        "runner": {key: value for key, value in parent_runner.items() if key != "source_path"},
        "importer": {key: value for key, value in parent_importer.items() if key != "source_path"},
        "repair": {key: value for key, value in parent_repair.items() if key != "source_path"},
    }
    timeout_row = row_by_packet_path["write_timeout_receipt.py"]
    freeze["timeout_receipt_helper"] = {key: value for key, value in timeout_row.items() if key != "source_path"}
    watchdog_row = row_by_packet_path["watchdog.sh"]
    freeze["watchdog"] = {key: value for key, value in watchdog_row.items() if key != "source_path"}
    freeze["parent_v4h"] = {
        "protocol_id": terminal["protocol_id"],
        "state_slice": terminal["state_slice"],
        "packet_sha256": terminal["packet_sha256"],
        "terminal_status": terminal["reproducibility_status"],
        "mission_name": terminal["mission"]["name"],
        "mission_url": terminal["mission"]["url"],
        "mission_settled_usd": terminal["mission"]["settled_total_usd"],
        "source_revision": terminal["source_revision"],
        "freeze_file": {
            "path": "parent-v4h-freeze.json",
            "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4i-parent-v4h-freeze.json",
            "bytes": len(parent_bytes),
            "sha256": sha_bytes(parent_bytes),
        },
        "terminal_receipt": {
            "path": "parent-v4h-terminal.json",
            "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4h-terminal.json",
            "bytes": len(terminal_bytes),
            "sha256": sha_bytes(terminal_bytes),
        },
        "results_report_sha256": sha_file(PARENT_RESULTS),
        "evaluator": terminal["evaluator"],
        "partial_predictions": terminal["partial_predictions"],
        "accounting": terminal["accounting"],
        "not_reused_as_input": True,
    }
    freeze["continuation_parent"] = {
        "protocol_id": terminal["protocol_id"],
        "packet_sha256": terminal["packet_sha256"],
        "terminal_status": terminal["reproducibility_status"],
        "continuation_attempt": "V4i",
    }
    freeze["primary_report_recovery"]["quarantine_relative_path"] = "v4i/quarantine/v3-primary-outcome-manifest.pre-repair.json"

    rate = RATE_USD_PER_MINUTE
    baseline = round(PRE_V4H_REGISTER_USD + V4H_SETTLED_USD, 4)
    node_max_usd = round(rate * NODE_MAX_MINUTES, 5)
    aggregate_headroom = round(AGGREGATE_CAP_USD - baseline - node_max_usd, 4)
    freeze["budget"] = {
        "settled_related_spend_before_v4i_usd": baseline,
        "frozen_pre_v4h_related_spend_register_usd": PRE_V4H_REGISTER_USD,
        "settled_v4h_mission_spend_usd": V4H_SETTLED_USD,
        "v4i_incremental_hard_cap_usd": round(AGGREGATE_CAP_USD - baseline, 4),
        "aggregate_worst_case_usd": round(baseline + node_max_usd, 4),
        "aggregate_headroom_after_hard_cap_usd": aggregate_headroom,
        "destination_node_max_minutes": NODE_MAX_MINUTES,
        "destination_node_max_usd": node_max_usd,
        "evaluator_command_max_minutes": EVALUATOR_MAX_MINUTES,
        "receipt_export_and_shutdown_reserve_minutes": NODE_MAX_MINUTES - EVALUATOR_MAX_MINUTES,
        "post_evaluator_receipt_and_export_reserve_minutes": 7,
        "max_rate_usd_per_minute": rate,
        "related_spend_by_node_usd": dict(parent["budget"]["related_spend_by_node_usd"], **{
            "theseus-v9-linux-runtime-v4h-cpu8-20260928": V4H_SETTLED_USD,
        }),
        "unallocated_incremental_reserve_usd": aggregate_headroom,
        "storage_cost": "V4e workspace snapshot restored; current GMan organization snapshot storage is below its first-250-GiB free allowance",
        "user_authorized_related_aggregate_usd": AGGREGATE_CAP_USD,
        "cost_accounting_note": "Conservative budget baseline uses the V4h frozen pre-attempt related-spend register plus GMan's settled V4h mission cost; no stale pre-settlement estimate is carried forward.",
    }
    freeze["execution"] = {
        "gman_command": "bash /home/dev/theseus-v9/v4i/watchdog.sh",
        "evaluator_timeout": "timeout --signal=TERM --kill-after=30s at most 300m, dynamically reduced to preserve the 315-minute node ceiling",
        "outer_watchdog": "frozen watchdog terminates bootstrap by node-start+311m; GMan max_duration_minutes is omitted because this node is already running and GMan only accepts that field for on_wake boot commands",
        "outer_watchdog_max_elapsed_minutes": 311,
        "hard_node_stop_elapsed_minutes": 315,
        "watchdog_receipt_export_and_shutdown_reserve_minutes": 4,
        "max_incremental_spend_usd": round(AGGREGATE_CAP_USD - baseline, 4),
        "max_node_minutes": NODE_MAX_MINUTES,
        "gman_max_duration_minutes_parameter": "omitted_for_active_node_detached_command",
        "aggregate_related_spend_cap_usd": AGGREGATE_CAP_USD,
        "projected_max_aggregate_usd": round(baseline + node_max_usd, 4),
        "aggregate_headroom_usd": aggregate_headroom,
        "rate_ceiling_usd_per_minute": rate,
        "one_evaluator_attempt": True,
        "timeout_receipt": "write digest-only receipt after evaluator timeout or nonzero exit; partial output payloads stay in GMan custody",
    }
    freeze["custody_transfer"]["runtime_node_restore_method"] = "fresh CPU-8 node restored from the terminal V4e workspace snapshot snap-t3r2t; V4h disk and partial predictions are not reused"
    freeze["experimental_design"]["baseline"] = "frozen V9 primary outcome FAIL with 21 of 27 ridge stages passing; V4h terminal timeout produced no held-out outcome evaluation"
    freeze["experimental_design"]["direction"] = "reproduce the frozen V9 primary outcome and every eligible replication check under the same digest-bound Linux runtime with a longer evaluator allowance and a terminal timeout receipt"
    freeze["experimental_design"]["full_cost_boundary"] = (
        f"Conservative related-spend baseline {baseline:.4f} USD; aggregate hard task cap {AGGREGATE_CAP_USD:.2f} USD. "
        f"V4i CPU-8 runtime is capped at {NODE_MAX_MINUTES} minutes at {rate:.5f} USD/minute ({node_max_usd:.5f} USD). "
        f"The evaluator has at most {EVALUATOR_MAX_MINUTES} minutes, reduced dynamically to preserve node and receipt/export reserve. "
        f"Projected maximum aggregate is {baseline + node_max_usd:.4f} USD; remaining headroom is {aggregate_headroom:.4f} USD."
    )
    freeze["experimental_design"]["stopping_rule"] = (
        "One V4i attempt. Stop before evaluator launch on any packet, parent, archive, recovery, custody, runtime, input, or budget mismatch. "
        "Require digest-valid transport, recovery, and runtime preflight receipts. Enforce a maximum 300-minute evaluator timeout, dynamically "
        "reduced to stay within 315 billed node minutes. A digest-checked in-packet watchdog ends the detached command by node-start+311m, preserving four minutes for the receipt, export, and shutdown; the bootstrap separately reserves seven minutes after its evaluator timeout for receipt work. Preserve a digest-only stop receipt on timeout or nonzero exit. No retry, threshold changes, or re-fit from V4h outputs."
    )
    freeze["source_identity"] = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "packet_file_count": len(rows),
        "packet_source_sha256": canonical_digest([{key: row[key] for key in ("path", "bytes", "sha256")} for row in rows]),
        "patched_replicator_sha256": row_by_packet_path["runner.py"]["sha256"],
        "replicator_sha256": parent_runner["sha256"],
        "runner_sha256": row_by_packet_path["runner.py"]["sha256"],
        "parent_v4h_freeze_sha256": sha_bytes(parent_bytes),
        "parent_v4h_terminal_receipt_sha256": sha_bytes(terminal_bytes),
        "results_report_sha256": sha_file(PARENT_RESULTS),
        "freeze_builder_sha256": sha_file(Path(__file__)),
    }
    freeze.pop("packet_sha256", None)
    freeze["packet_sha256"] = canonical_digest(freeze)
    OUTPUT.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "state_slice": STATE_SLICE,
        "packet_sha256": freeze["packet_sha256"],
        "frozen_file_count": len(rows),
        "related_spend_baseline_usd": baseline,
        "aggregate_cap_usd": AGGREGATE_CAP_USD,
        "projected_max_aggregate_usd": round(baseline + node_max_usd, 4),
        "aggregate_headroom_usd": aggregate_headroom,
        "evaluator_max_minutes": EVALUATOR_MAX_MINUTES,
        "node_max_minutes": NODE_MAX_MINUTES,
        "freeze_path": OUTPUT.as_posix(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
