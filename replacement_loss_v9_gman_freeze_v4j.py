#!/usr/bin/env python3
"""Freeze a V4j continuation after V4i stopped before candidate preparation.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j, and
global-experiment-assurance-lifecycle-v1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j"
STATE_SLICE = PROTOCOL_ID
V4I_FREEZE = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4i-freeze.json"
V4I_TERMINAL = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4i-terminal.json"
V4I_RESULTS = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4i-results.md"
V4I_FREEZE_COPY = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-freeze.json"
V4I_TERMINAL_COPY = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-terminal.json"
V4H_FREEZE = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-freeze.json"
V4H_TERMINAL = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-terminal.json"
V4H_RESULTS = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-results.md"
V4H_FREEZE_COPY = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-freeze.json"
V4H_TERMINAL_COPY = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-terminal.json"
OUTPUT = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-freeze.json"
RATE_USD_PER_MINUTE = 0.00804
AGGREGATE_CAP_USD = 5.0
PRE_V4H_REGISTER_USD = 1.167
V4H_SETTLED_USD = 0.639
EVALUATOR_MAX_MINUTES = 300
NODE_MAX_MINUTES = 315
WATCHDOG_STOP_RESERVE_MINUTES = 4
NODE_START_EPOCH = 1790689641


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


def transformed(source: str, replacements: tuple[tuple[str, str], ...]) -> str:
    value = (ROOT / source).read_text(encoding="utf-8")
    for old, new in replacements:
        value = value.replace(old, new)
    return value


def make_protocol_doc(capture_epoch: int, baseline: float, node_remaining_minutes: int,
                      node_remaining_cost: float, aggregate_headroom: float) -> str:
    text = transformed("docs/replacement-loss-v9-gman-fresh-node-v4i.md", (
        ("V4I", "V4J"), ("v4i", "v4j"), ("V4i", "V4j"),
    ))
    boundary = (
        "## Purpose and boundary\n\n"
        "V4j follows V4i's terminal pre-evaluation error. V4i stopped during "
        "the custody importer before candidate preparation or challenge reveal. "
        "Its source identity fields bound the adapter/helper digests where the "
        "original V9 runner and replicator digests were required. V4j preserves "
        "that receipt and corrects only the source identity binding; its output "
        "directory is fresh.\n\n"
    )
    text = text.replace("## Purpose and boundary\n\n", boundary, 1)
    old_budget = (
        "The prior frozen V9 related-spend register is `$1.1670`; GMan's settled V4h\n"
        "mission charge is `$0.6390`, so V4j starts from a conservative recorded\n"
        "aggregate of `$1.8060`. The task ceiling is `$5.00` aggregate. At the verified\n"
        "CPU-8 rate of `$0.00804/min`, V4j has a 315-minute node ceiling (`$2.5326`),\n"
        "leaving `$0.6614` of aggregate headroom after the projected maximum. The\n"
        "evaluator timeout is at most 300 minutes; the bootstrap reduces it if setup\n"
        "uses the 15-minute remainder. A seven-minute post-evaluator reserve is kept\n"
        "for the digest-only stop/run summary, export, and shutdown. GMan's detached\n"
        "command ceiling is 314 minutes; stop the node by 315 billed minutes. If the\n"
        "node quote exceeds `$0.00804/min`, do not start the evaluator."
    )
    new_budget = (
        f"The conservative related-spend baseline at the V4j freeze is `${baseline:.4f}`: "
        f"the prior `$1.1670` register, settled V4h cost `$0.6390`, and V4i GMan "
        f"accrual captured at Unix epoch {capture_epoch}. V4i's charge is accrued, "
        "not final settlement. The aggregate hard cap remains `$5.00`. The existing "
        f"CPU-8 node has at most {node_remaining_minutes} billed minutes remaining to "
        f"its frozen 315-minute lifetime ceiling (`${node_remaining_cost:.4f}` at "
        f"`$0.00804/min`); projected aggregate maximum is `${baseline + node_remaining_cost:.4f}` "
        f"with `${aggregate_headroom:.4f}` headroom. The evaluator timeout is at most "
        "300 minutes and is dynamically reduced using the original node-start epoch, "
        "setup time, and seven-minute receipt/export reserve. If the live quote exceeds "
        "`$0.00804/min` or the remaining aggregate budget is insufficient, do not start."
    )
    if old_budget not in text:
        raise ValueError("V4i budget paragraph did not match the frozen source document")
    text = text.replace(old_budget, new_budget, 1)
    text = text.replace(
        "## Frozen scientific design\n",
        "## Corrected source identity binding\n\n"
        "V4i's wrapper SHA-256 was written into the V9 `runner_sha256` field, "
        "and the V4h helper SHA-256 into `replicator_sha256`. The original "
        "V4h freeze binds the V9 source files as runner "
        "`4cd2f003df9c7a82846e71c394e13a7771c40e16bae9543496197333cb4262a1` "
        "and replicator "
        "`005ff243087001441c44eee22408c55a3202645eb2d9648e968417ecb5686923`. "
        "Read-only hashes on the GMan node matched those original source digests. "
        "V4j restores those two frozen values; no scientific method, fixture, "
        "threshold, control, or outcome rule changes.\n\n"
        "## Frozen scientific design\n",
        1,
    )
    text += (
        "\n\nV4j is one continuation attempt. If its preflight or evaluator stops, "
        "retain the terminal status; do not retry or tune this packet.\n"
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-epoch", type=int, required=True)
    parser.add_argument("--v4i-accrued-usd", type=float, required=True)
    args = parser.parse_args()

    v4i_bytes = V4I_FREEZE.read_bytes()
    v4i = json.loads(v4i_bytes)
    v4i_terminal_bytes = V4I_TERMINAL.read_bytes()
    v4i_terminal = json.loads(v4i_terminal_bytes)
    if v4i.get("protocol_id") != "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i":
        raise ValueError("V4i parent packet identity mismatch")
    if v4i.get("packet_sha256") != canonical_digest({k: v for k, v in v4i.items() if k != "packet_sha256"}):
        raise ValueError("V4i parent packet canonical digest mismatch")
    if (
        v4i_terminal.get("packet_sha256") != v4i.get("packet_sha256")
        or v4i_terminal.get("receipt_sha256") != canonical_digest({k: v for k, v in v4i_terminal.items() if k != "receipt_sha256"})
        or v4i_terminal.get("reproducibility_status") != "INCONCLUSIVE_EVALUATOR_ERROR"
        or v4i_terminal.get("observed_exit_code") != 1
        or v4i_terminal.get("evaluator_started") is not False
        or v4i_terminal.get("candidate_preparation_completed") is not False
        or v4i_terminal.get("challenge_fixtures_revealed_after_candidate_freeze") is not False
        or v4i_terminal.get("prediction_inventory", {}).get("files") != 0
    ):
        raise ValueError("V4i terminal record does not establish a pre-evaluation stop")

    v4h_bytes = V4H_FREEZE.read_bytes()
    v4h = json.loads(v4h_bytes)
    v4h_terminal_bytes = V4H_TERMINAL.read_bytes()
    v4h_terminal = json.loads(v4h_terminal_bytes)
    if v4h.get("protocol_id") != "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h":
        raise ValueError("V4h base packet identity mismatch")
    if v4h.get("packet_sha256") != canonical_digest({k: v for k, v in v4h.items() if k != "packet_sha256"}):
        raise ValueError("V4h base packet canonical digest mismatch")
    if v4h_terminal.get("packet_sha256") != v4h.get("packet_sha256") or v4h_terminal.get("reproducibility_status") != "INCONCLUSIVE_TIMEOUT":
        raise ValueError("V4h terminal receipt is not bound to its frozen packet")

    for key, source in (
        ("wrapper", "replacement_loss_v9_gman_fresh_node_replicator_v4h.py"),
        ("importer", "replacement_loss_v9_gman_import_v4h.py"),
        ("primary_report_repair", "replacement_loss_v9_gman_repair_primary_v4h.py"),
    ):
        row = v4h[key]
        path = ROOT / source
        if path.stat().st_size != row["bytes"] or sha_file(path) != row["sha256"]:
            raise ValueError(f"frozen V4h helper differs from parent packet: {source}")
    expected_runner = v4h["source_identity"]["runner_sha256"]
    expected_replicator = v4h["source_identity"]["replicator_sha256"]
    if (
        v4i["source_identity"]["runner_sha256"] == expected_runner
        or v4i["source_identity"]["replicator_sha256"] == expected_replicator
    ):
        raise ValueError("V4i source identity mismatch was not reproduced from the recorded parent fields")

    base_related = round(PRE_V4H_REGISTER_USD + V4H_SETTLED_USD + args.v4i_accrued_usd, 4)
    hard_node_deadline = NODE_START_EPOCH + NODE_MAX_MINUTES * 60
    node_remaining_seconds = hard_node_deadline - args.capture_epoch
    if node_remaining_seconds <= 0 or args.capture_epoch < NODE_START_EPOCH:
        raise ValueError("V4j freeze time is outside the active node budget")
    node_remaining_minutes = math.ceil(node_remaining_seconds / 60)
    node_remaining_cost = round(node_remaining_minutes * RATE_USD_PER_MINUTE, 5)
    aggregate_headroom = round(AGGREGATE_CAP_USD - base_related - node_remaining_cost, 4)
    if aggregate_headroom < 0:
        raise ValueError("remaining node lifetime does not fit the $5 aggregate cap")

    replacements = (("V4I", "V4J"), ("v4i", "v4j"), ("V4i", "V4j"))
    file_specs = [
        ("protocol.md", "docs/replacement-loss-v9-gman-fresh-node-v4j.md", "protocol"),
        ("runner.py", "replacement_loss_v9_gman_fresh_node_replicator_v4j.py", "text"),
        ("bootstrap.sh", "replacement_loss_v9_gman_bootstrap_v4j.sh", "text"),
        ("importer.py", "replacement_loss_v9_gman_import_v4j.py", "text"),
        ("repair_primary.py", "replacement_loss_v9_gman_repair_primary_v4j.py", "text"),
        ("write_timeout_receipt.py", "replacement_loss_v9_gman_timeout_receipt_v4j.py", "text"),
        ("watchdog.sh", "replacement_loss_v9_gman_watchdog_v4j.sh", "text"),
    ]
    doc = make_protocol_doc(args.capture_epoch, base_related, node_remaining_minutes, node_remaining_cost, aggregate_headroom)
    (ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j.md").write_text(doc, encoding="utf-8")
    for _packet, target, mode in file_specs[1:]:
        source = {
            "replacement_loss_v9_gman_fresh_node_replicator_v4j.py": "replacement_loss_v9_gman_fresh_node_replicator_v4i.py",
            "replacement_loss_v9_gman_bootstrap_v4j.sh": "replacement_loss_v9_gman_bootstrap_v4i.sh",
            "replacement_loss_v9_gman_import_v4j.py": "replacement_loss_v9_gman_import_v4i.py",
            "replacement_loss_v9_gman_repair_primary_v4j.py": "replacement_loss_v9_gman_repair_primary_v4i.py",
            "replacement_loss_v9_gman_timeout_receipt_v4j.py": "replacement_loss_v9_gman_timeout_receipt_v4i.py",
            "replacement_loss_v9_gman_watchdog_v4j.sh": "replacement_loss_v9_gman_watchdog_v4i.sh",
        }[target]
        text = transformed(source, replacements)
        if target == "replacement_loss_v9_gman_bootstrap_v4j.sh":
            old_rows = 'freeze["parent_v4h"]["terminal_receipt"]]:'
            new_rows = 'freeze["parent_v4h"]["terminal_receipt"], freeze["parent_v4i"]["freeze_file"], freeze["parent_v4i"]["terminal_receipt"]]:'
            if old_rows not in text:
                raise ValueError("V4j bootstrap parent receipt validation row is missing")
            text = text.replace(old_rows, new_rows, 1)
        (ROOT / target).write_text(text, encoding="utf-8")

    V4I_FREEZE_COPY.write_bytes(v4i_bytes)
    V4I_TERMINAL_COPY.write_bytes(v4i_terminal_bytes)
    V4H_FREEZE_COPY.write_bytes(v4h_bytes)
    V4H_TERMINAL_COPY.write_bytes(v4h_terminal_bytes)
    root_files = file_specs + [
        ("parent_runner_v4h.py", "replacement_loss_v9_gman_fresh_node_replicator_v4h.py", "copy"),
        ("parent_importer_v4h.py", "replacement_loss_v9_gman_import_v4h.py", "copy"),
        ("parent_repair_v4h.py", "replacement_loss_v9_gman_repair_primary_v4h.py", "copy"),
        ("parent-v4h-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-freeze.json", "copy"),
        ("parent-v4h-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-terminal.json", "copy"),
        ("parent-v4i-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-freeze.json", "copy"),
        ("parent-v4i-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-terminal.json", "copy"),
    ]
    rows = [file_row(packet, source) for packet, source, _ in root_files]
    row_by_path = {row["path"]: row for row in rows}
    freeze = json.loads(json.dumps(v4i))
    freeze["protocol_id"] = PROTOCOL_ID
    freeze["state_slice"] = STATE_SLICE
    freeze["repository_state_slice"] = "theseus-lab-sandbox-v1"
    freeze["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    freeze["freeze_state"] = "FROZEN_BEFORE_V4J_EVALUATION_AFTER_V4I_PRE_EVALUATION_ERROR"
    freeze["claim_ceiling"] = "same-owner local cross-runtime reproduction of frozen V9 FAIL under the digest-bound existing Linux runtime"
    freeze["reviewer_status"] = "NOT_SUBMITTED"

    for packet_path, _source_path, _source_kind in file_specs:
        row = row_by_path[packet_path]
        field = {
            "protocol.md": "packet_document",
            "runner.py": "wrapper",
            "bootstrap.sh": "bootstrap",
            "importer.py": "importer",
            "repair_primary.py": "primary_report_repair",
        }.get(packet_path)
        if field:
            freeze[field] = {k: row[k] for k in ("path", "bytes", "sha256")}
        elif packet_path == "write_timeout_receipt.py":
            freeze["timeout_receipt_helper"] = {k: row[k] for k in ("path", "bytes", "sha256")}
        elif packet_path == "watchdog.sh":
            freeze["watchdog"] = {k: row[k] for k in ("path", "bytes", "sha256")}

    freeze["packet_document"]["path"] = "docs/replacement-loss-v9-gman-fresh-node-v4j.md"
    freeze["parent_helpers"] = {
        "runner": {k: row_by_path["parent_runner_v4h.py"][k] for k in ("path", "bytes", "sha256")},
        "importer": {k: row_by_path["parent_importer_v4h.py"][k] for k in ("path", "bytes", "sha256")},
        "repair": {k: row_by_path["parent_repair_v4h.py"][k] for k in ("path", "bytes", "sha256")},
    }
    freeze["parent_v4h"]["freeze_file"] = {
        "path": "parent-v4h-freeze.json", "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-freeze.json",
        "bytes": row_by_path["parent-v4h-freeze.json"]["bytes"], "sha256": row_by_path["parent-v4h-freeze.json"]["sha256"],
    }
    freeze["parent_v4h"]["terminal_receipt"] = {
        "path": "parent-v4h-terminal.json", "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-terminal.json",
        "bytes": row_by_path["parent-v4h-terminal.json"]["bytes"], "sha256": row_by_path["parent-v4h-terminal.json"]["sha256"],
    }
    freeze["parent_v4i"] = {
        "protocol_id": v4i_terminal["protocol_id"],
        "state_slice": v4i_terminal["state_slice"],
        "packet_sha256": v4i_terminal["packet_sha256"],
        "terminal_status": v4i_terminal["reproducibility_status"],
        "command_id": "cmd-w2avs",
        "observed_exit_code": v4i_terminal["observed_exit_code"],
        "evaluator_started": v4i_terminal["evaluator_started"],
        "candidate_preparation_completed": v4i_terminal["candidate_preparation_completed"],
        "prediction_inventory": v4i_terminal["prediction_inventory"],
        "freeze_file": {
            "path": "parent-v4i-freeze.json", "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-freeze.json",
            "bytes": row_by_path["parent-v4i-freeze.json"]["bytes"], "sha256": row_by_path["parent-v4i-freeze.json"]["sha256"],
        },
        "terminal_receipt": {
            "path": "parent-v4i-terminal.json", "source_path": "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-terminal.json",
            "bytes": row_by_path["parent-v4i-terminal.json"]["bytes"], "sha256": row_by_path["parent-v4i-terminal.json"]["sha256"],
        },
        "results_report_sha256": sha_file(V4I_RESULTS),
        "mission_accrued_usd_at_freeze": args.v4i_accrued_usd,
        "source_identity_error": "V4i runner_sha256 held the V4i adapter digest and replicator_sha256 held the V4h helper digest; no candidate or held-out evaluation was reached",
    }
    freeze["continuation_parent"] = {
        "protocol_id": v4i_terminal["protocol_id"],
        "packet_sha256": v4i_terminal["packet_sha256"],
        "terminal_status": v4i_terminal["reproducibility_status"],
        "continuation_attempt": "V4j",
    }
    freeze["execution"]["gman_command"] = "bash /home/dev/theseus-v9/v4j/watchdog.sh"
    freeze["execution"]["outer_watchdog"] = "frozen watchdog stops by the original node-start+311m; max_duration_minutes is omitted for the active-node detached command"
    freeze["execution"]["outer_watchdog_max_elapsed_minutes"] = NODE_MAX_MINUTES - WATCHDOG_STOP_RESERVE_MINUTES
    freeze["execution"]["hard_node_stop_elapsed_minutes"] = NODE_MAX_MINUTES
    freeze["execution"]["watchdog_receipt_export_and_shutdown_reserve_minutes"] = WATCHDOG_STOP_RESERVE_MINUTES
    freeze["execution"]["gman_max_duration_minutes_parameter"] = "omitted_for_active_node_detached_command"
    freeze["execution"]["max_node_minutes"] = NODE_MAX_MINUTES
    freeze["execution"]["max_incremental_spend_usd"] = round(AGGREGATE_CAP_USD - base_related, 4)
    freeze["execution"]["aggregate_related_spend_cap_usd"] = AGGREGATE_CAP_USD
    freeze["execution"]["projected_max_aggregate_usd"] = round(base_related + node_remaining_cost, 4)
    freeze["execution"]["aggregate_headroom_usd"] = aggregate_headroom
    freeze["execution"]["rate_ceiling_usd_per_minute"] = RATE_USD_PER_MINUTE
    freeze["execution"]["node_remaining_minutes_at_freeze"] = node_remaining_minutes
    freeze["execution"]["node_start_epoch"] = NODE_START_EPOCH
    freeze["execution"]["freeze_capture_epoch"] = args.capture_epoch
    freeze["execution"]["one_evaluator_attempt"] = True
    freeze["execution"]["gman_command"] = "bash /home/dev/theseus-v9/v4j/watchdog.sh"

    freeze["budget"] = {
        "related_spend_baseline_at_freeze_usd": base_related,
        "frozen_pre_v4h_related_spend_register_usd": PRE_V4H_REGISTER_USD,
        "settled_v4h_mission_spend_usd": V4H_SETTLED_USD,
        "accrued_v4i_mission_spend_at_freeze_usd": args.v4i_accrued_usd,
        "v4j_incremental_hard_cap_usd": round(AGGREGATE_CAP_USD - base_related, 4),
        "aggregate_worst_case_usd": round(base_related + node_remaining_cost, 4),
        "aggregate_headroom_after_hard_cap_usd": aggregate_headroom,
        "destination_node_max_minutes": NODE_MAX_MINUTES,
        "node_remaining_minutes_at_freeze": node_remaining_minutes,
        "destination_node_remaining_max_usd": node_remaining_cost,
        "evaluator_command_max_minutes": EVALUATOR_MAX_MINUTES,
        "post_evaluator_receipt_and_export_reserve_minutes": 7,
        "watchdog_receipt_export_and_shutdown_reserve_minutes": WATCHDOG_STOP_RESERVE_MINUTES,
        "max_rate_usd_per_minute": RATE_USD_PER_MINUTE,
        "related_spend_by_node_usd": dict(v4i["budget"]["related_spend_by_node_usd"], **{
            "theseus-v9-v4i-cpu8-20260929": args.v4i_accrued_usd,
        }),
        "storage_cost": v4i["budget"]["storage_cost"],
        "user_authorized_related_aggregate_usd": AGGREGATE_CAP_USD,
        "cost_accounting_note": "Conservative baseline includes the V9 register, settled V4h cost, and V4i GMan accrual captured before V4j. Remaining cost is bounded by the original node-start epoch and 315-minute node lifetime; stop at the watchdog and hard ceiling.",
    }
    freeze["experimental_design"]["baseline"] = "frozen V9 primary outcome FAIL with 21 of 27 ridge stages passing; V4h timed out and V4i stopped on source identity before candidate preparation"
    freeze["experimental_design"]["direction"] = "reproduce the frozen V9 primary outcome and each eligible check after correcting only the V9 source identity binding"
    freeze["experimental_design"]["full_cost_boundary"] = (
        f"Conservative related-spend baseline {base_related:.4f} USD; hard aggregate cap {AGGREGATE_CAP_USD:.2f} USD. "
        f"Remaining node lifetime is at most {node_remaining_minutes} minutes at {RATE_USD_PER_MINUTE:.5f} USD/min "
        f"({node_remaining_cost:.5f} USD). Projected maximum aggregate is {base_related + node_remaining_cost:.4f} USD."
    )
    freeze["experimental_design"]["stopping_rule"] = (
        "One V4j attempt. Stop before evaluator launch on any frozen packet, source identity, archive, custody, recovery, runtime, input, or budget mismatch. "
        "Run only after packet hash and runtime preflights pass. Keep the original node-start epoch; evaluator maximum is 300 minutes and dynamically reduced by setup and reserve. "
        "The watchdog stops by node-start+311m, with the 315m node lifetime as the hard ceiling. No retries, threshold changes, fixture changes, or V4i output reuse."
    )
    freeze["custody_transfer"]["runtime_node_restore_method"] = "existing CPU-8 node restored from V4e snapshot before V4i; V4j uses a fresh v4j output directory and does not reuse V4i run outputs"
    freeze["source_identity"] = {
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "packet_file_count": len(rows),
        "packet_source_sha256": canonical_digest([{k: row[k] for k in ("path", "bytes", "sha256")} for row in rows]),
        "patched_replicator_sha256": row_by_path["runner.py"]["sha256"],
        "replicator_sha256": expected_replicator,
        "runner_sha256": expected_runner,
        "parent_v4h_freeze_sha256": sha_bytes(v4h_bytes),
        "parent_v4h_terminal_receipt_sha256": sha_bytes(v4h_terminal_bytes),
        "parent_v4i_freeze_sha256": sha_bytes(v4i_bytes),
        "parent_v4i_terminal_receipt_sha256": sha_bytes(v4i_terminal_bytes),
        "results_report_sha256": sha_file(V4H_RESULTS),
        "freeze_builder_sha256": sha_file(Path(__file__)),
    }
    freeze.pop("packet_sha256", None)
    freeze["packet_sha256"] = canonical_digest(freeze)
    OUTPUT.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "packet_sha256": freeze["packet_sha256"],
        "packet_file_count": len(rows),
        "source_runner_sha256": expected_runner,
        "source_replicator_sha256": expected_replicator,
        "related_spend_baseline_usd": base_related,
        "remaining_node_minutes": node_remaining_minutes,
        "remaining_node_max_usd": node_remaining_cost,
        "aggregate_cap_usd": AGGREGATE_CAP_USD,
        "projected_aggregate_usd": round(base_related + node_remaining_cost, 4),
        "aggregate_headroom_usd": aggregate_headroom,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
