#!/usr/bin/env python3
"""Freeze a V4k continuation after V4j's pre-candidate source-patch error.

State slices: theseus-lab-sandbox-v1,
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4k, and
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
PROTOCOL_ID = "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4k"
STATE_SLICE = PROTOCOL_ID
V4J_FREEZE = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-freeze.json"
V4J_TERMINAL = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-terminal.json"
V4J_RUN_STOP = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-run-stop.json"
V4J_RESULTS = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j-results.md"
V4H_FREEZE = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4h-freeze.json"
OUTPUT = ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4k-freeze.json"
NODE_START_EPOCH = 1790689641
NODE_MAX_MINUTES = 315
WATCHDOG_STOP_RESERVE_MINUTES = 4
EVALUATOR_MAX_MINUTES = 300
RATE_USD_PER_MINUTE = 0.00804
AGGREGATE_CAP_USD = 5.0
PRE_V4H_REGISTER_USD = 1.167
V4H_SETTLED_USD = 0.639


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical_digest(value: Any) -> str:
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def file_row(packet_path: str, source_path: str) -> dict[str, Any]:
    content = (ROOT / source_path).read_bytes()
    return {"path": packet_path, "source_path": source_path, "bytes": len(content), "sha256": sha_bytes(content)}


def make_protocol_doc(capture_epoch: int, baseline: float, remaining_minutes: int,
                      remaining_cost: float, headroom: float) -> str:
    text = (ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4j.md").read_text(encoding="utf-8")
    text = text.replace("V4J", "V4K").replace("v4j", "v4k").replace("V4j", "V4k")
    old_boundary = (
        "V4k follows V4i's terminal pre-evaluation error. V4i stopped during the custody importer before "
        "candidate preparation or challenge reveal. Its source identity fields bound the adapter/helper "
        "digests where the original V9 runner and replicator digests were required. V4k preserves that "
        "receipt and corrects only the source identity binding; its output directory is fresh."
    )
    new_boundary = (
        "V4k follows V4j's terminal pre-candidate error. V4j reached the run phase but stopped before "
        "candidate preparation, candidate freeze, challenge reveal, or prediction generation. V4i also "
        "stopped before candidate preparation. V4k preserves both terminal records and uses a fresh "
        "output directory; no V4i or V4j run state is reused."
    )
    if old_boundary not in text:
        raise ValueError("V4j purpose paragraph did not match the frozen source document")
    text = text.replace(old_boundary, new_boundary, 1)
    start = text.index("## Corrected source identity binding")
    end = text.index("## Frozen scientific design", start)
    text = text[:start] + (
        "## Corrected source identity binding\n\n"
        "V4i bound adapter/helper digests in the V9 source fields. V4j corrected the V9 runner and "
        "replicator fields but still placed the wrapper hash in `patched_replicator_sha256`. V4k "
        "binds all three values from the original V4h freeze: runner "
        "`4cd2f003df9c7a82846e71c394e13a7771c40e16bae9543496197333cb4262a1`, replicator "
        "`005ff243087001441c44eee22408c55a3202645eb2d9648e968417ecb5686923`, and the frozen "
        "V3 in-memory patch `6892b0a99bdfb7fed3950f06c9adc4b16dcaf87c9dcca890b25bec1829d2436e`. "
        "No scientific method, fixture, threshold, control, or outcome rule changes.\n\n"
    ) + text[end:]
    begin = text.index("The conservative related-spend baseline at the V4k freeze is")
    finish = text.index("do not start.", begin) + len("do not start.")
    new_budget = (
        f"The conservative related-spend baseline at the V4k freeze is {baseline:.4f} USD: "
        "the prior 1.1670 USD register, settled V4h cost 0.6390 USD, and the GMan mission accrual "
        f"captured at Unix epoch {capture_epoch}. The aggregate hard cap remains 5.00 USD. The "
        f"existing node has at most {remaining_minutes} billed minutes left to its frozen "
        f"315-minute lifetime limit ({remaining_cost:.4f} USD at 0.00804 USD/min); projected aggregate "
        f"maximum is {baseline + remaining_cost:.4f} USD, leaving {headroom:.4f} USD headroom. The "
        "evaluator timeout is at most 300 minutes and dynamically reduced for setup time and the "
        "receipt/export reserve. Do not start if the quote exceeds 0.00804 USD/min or the remaining "
        "budget cannot cover the frozen ceiling."
    )
    text = text[:begin] + new_budget + text[finish:]
    text += (
        "\n\nV4k is one continuation attempt. If its preflight or evaluator stops, retain the "
        "terminal status; do not retry or tune this packet.\n"
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-epoch", type=int, required=True)
    parser.add_argument("--mission-accrued-usd", type=float, required=True)
    args = parser.parse_args()
    v4j_bytes = V4J_FREEZE.read_bytes()
    v4j = json.loads(v4j_bytes)
    terminal_bytes = V4J_TERMINAL.read_bytes()
    terminal = json.loads(terminal_bytes)
    run_stop_bytes = V4J_RUN_STOP.read_bytes()
    run_stop = json.loads(run_stop_bytes)
    if v4j.get("packet_sha256") != canonical_digest({k: v for k, v in v4j.items() if k != "packet_sha256"}):
        raise ValueError("V4j parent packet canonical digest mismatch")
    if (
        terminal.get("packet_sha256") != v4j.get("packet_sha256")
        or terminal.get("receipt_sha256") != canonical_digest({k: v for k, v in terminal.items() if k != "receipt_sha256"})
        or terminal.get("reproducibility_status") != "INCONCLUSIVE_EVALUATOR_ERROR"
        or terminal.get("candidate_preparation_completed") is not False
        or terminal.get("candidate_freeze_completed") is not False
        or terminal.get("challenge_fixtures_revealed_after_candidate_freeze") is not False
        or terminal.get("prediction_inventory", {}).get("files") != 0
    ):
        raise ValueError("V4j terminal record does not establish the recorded pre-candidate stop")
    if (
        run_stop.get("packet_sha256") != v4j.get("packet_sha256")
        or run_stop.get("v4_stop_sha256") != canonical_digest({k: v for k, v in run_stop.items() if k != "v4_stop_sha256"})
        or run_stop.get("error") != "in-memory V9 dependency adaptation differs from the frozen V3 patch"
    ):
        raise ValueError("V4j inner run-stop receipt is invalid")

    v4h = json.loads(V4H_FREEZE.read_text(encoding="utf-8"))
    expected = v4h["source_identity"]
    if (
        v4j["source_identity"]["runner_sha256"] != expected["runner_sha256"]
        or v4j["source_identity"]["replicator_sha256"] != expected["replicator_sha256"]
        or v4j["source_identity"]["patched_replicator_sha256"] == expected["patched_replicator_sha256"]
    ):
        raise ValueError("V4j patched-replicator identity error was not reproduced")

    baseline = round(PRE_V4H_REGISTER_USD + V4H_SETTLED_USD + args.mission_accrued_usd, 4)
    deadline = NODE_START_EPOCH + NODE_MAX_MINUTES * 60
    seconds_left = deadline - args.capture_epoch
    if args.capture_epoch < NODE_START_EPOCH or seconds_left <= 0:
        raise ValueError("V4k freeze time is outside the original node lifetime")
    remaining_minutes = math.ceil(seconds_left / 60)
    remaining_cost = round(remaining_minutes * RATE_USD_PER_MINUTE, 5)
    headroom = round(AGGREGATE_CAP_USD - baseline - remaining_cost, 4)
    if headroom < 0:
        raise ValueError("remaining node lifetime exceeds the authorized $5 aggregate cap")

    replacements = (("V4J", "V4K"), ("v4j", "v4k"), ("V4j", "V4k"))
    specs = [
        ("protocol.md", "docs/replacement-loss-v9-gman-fresh-node-v4k.md"),
        ("runner.py", "replacement_loss_v9_gman_fresh_node_replicator_v4k.py"),
        ("bootstrap.sh", "replacement_loss_v9_gman_bootstrap_v4k.sh"),
        ("importer.py", "replacement_loss_v9_gman_import_v4k.py"),
        ("repair_primary.py", "replacement_loss_v9_gman_repair_primary_v4k.py"),
        ("write_timeout_receipt.py", "replacement_loss_v9_gman_timeout_receipt_v4k.py"),
        ("watchdog.sh", "replacement_loss_v9_gman_watchdog_v4k.sh"),
    ]
    (ROOT / "docs/replacement-loss-v9-gman-fresh-node-v4k.md").write_text(
        make_protocol_doc(args.capture_epoch, baseline, remaining_minutes, remaining_cost, headroom),
        encoding="utf-8",
    )
    v4j_sources = {
        "replacement_loss_v9_gman_fresh_node_replicator_v4k.py": "replacement_loss_v9_gman_fresh_node_replicator_v4j.py",
        "replacement_loss_v9_gman_bootstrap_v4k.sh": "replacement_loss_v9_gman_bootstrap_v4j.sh",
        "replacement_loss_v9_gman_import_v4k.py": "replacement_loss_v9_gman_import_v4j.py",
        "replacement_loss_v9_gman_repair_primary_v4k.py": "replacement_loss_v9_gman_repair_primary_v4j.py",
        "replacement_loss_v9_gman_timeout_receipt_v4k.py": "replacement_loss_v9_gman_timeout_receipt_v4j.py",
        "replacement_loss_v9_gman_watchdog_v4k.sh": "replacement_loss_v9_gman_watchdog_v4j.sh",
    }
    for _packet, dest in specs[1:]:
        content = (ROOT / v4j_sources[dest]).read_text(encoding="utf-8")
        for old, new in replacements:
            content = content.replace(old, new)
        if dest == "replacement_loss_v9_gman_bootstrap_v4k.sh":
            old = 'freeze["parent_v4i"]["terminal_receipt"]]:'
            new = 'freeze["parent_v4i"]["terminal_receipt"], freeze["parent_v4j"]["freeze_file"], freeze["parent_v4j"]["terminal_receipt"], freeze["parent_v4j"]["run_stop"]]:'
            if old not in content:
                raise ValueError("V4k bootstrap parent receipt validation row is missing")
            content = content.replace(old, new, 1)
        (ROOT / dest).write_text(content, encoding="utf-8")

    parent_specs = [
        ("parent_runner_v4h.py", "replacement_loss_v9_gman_fresh_node_replicator_v4h.py"),
        ("parent_importer_v4h.py", "replacement_loss_v9_gman_import_v4h.py"),
        ("parent_repair_v4h.py", "replacement_loss_v9_gman_repair_primary_v4h.py"),
        ("parent-v4h-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-freeze.json"),
        ("parent-v4h-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4h-terminal.json"),
        ("parent-v4i-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-freeze.json"),
        ("parent-v4i-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-parent-v4i-terminal.json"),
        ("parent-v4j-freeze.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-freeze.json"),
        ("parent-v4j-terminal.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-terminal.json"),
        ("parent-v4j-run-stop.json", "docs/replacement-loss-v9-gman-fresh-node-v4j-run-stop.json"),
    ]
    parent_copy_paths = {
        "parent-v4h-freeze.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4h-freeze.json",
        "parent-v4h-terminal.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4h-terminal.json",
        "parent-v4i-freeze.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4i-freeze.json",
        "parent-v4i-terminal.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4i-terminal.json",
        "parent-v4j-freeze.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4j-freeze.json",
        "parent-v4j-terminal.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4j-terminal.json",
        "parent-v4j-run-stop.json": "docs/replacement-loss-v9-gman-fresh-node-v4k-parent-v4j-run-stop.json",
    }
    for packet_path, source_path in parent_specs:
        if packet_path in parent_copy_paths:
            (ROOT / parent_copy_paths[packet_path]).write_bytes((ROOT / source_path).read_bytes())
    root_files = [(p, s) for p, s in specs] + [(p, parent_copy_paths.get(p, s)) for p, s in parent_specs]
    rows = [file_row(packet, source) for packet, source in root_files]
    by_path = {row["path"]: row for row in rows}
    v4h_bytes = V4H_FREEZE.read_bytes()
    v4h = json.loads(v4h_bytes)
    freeze = json.loads(json.dumps(v4j))
    freeze["protocol_id"] = PROTOCOL_ID
    freeze["state_slice"] = STATE_SLICE
    freeze["repository_state_slice"] = "theseus-lab-sandbox-v1"
    freeze["created_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    freeze["freeze_state"] = "FROZEN_BEFORE_V4K_EVALUATION_AFTER_V4J_PRE_CANDIDATE_ERROR"
    freeze["claim_ceiling"] = "same-owner local cross-runtime reproduction of frozen V9 FAIL under the digest-bound existing Linux runtime"
    freeze["reviewer_status"] = "NOT_SUBMITTED"
    fields = {"protocol.md":"packet_document","runner.py":"wrapper","bootstrap.sh":"bootstrap",
              "importer.py":"importer","repair_primary.py":"primary_report_repair",
              "write_timeout_receipt.py":"timeout_receipt_helper","watchdog.sh":"watchdog"}
    for packet_path, field in fields.items():
        freeze[field] = {k:by_path[packet_path][k] for k in ("path","bytes","sha256")}
    freeze["packet_document"]["path"] = "docs/replacement-loss-v9-gman-fresh-node-v4k.md"
    freeze["parent_helpers"] = {
        name:{k:by_path[packet][k] for k in ("path","bytes","sha256")}
        for name,packet in (("runner","parent_runner_v4h.py"),("importer","parent_importer_v4h.py"),("repair","parent_repair_v4h.py"))
    }
    for version in ("v4h","v4i"):
        key=f"parent_{version}"
        freeze[key]["freeze_file"]={"path":f"parent-{version}-freeze.json","source_path":parent_copy_paths[f"parent-{version}-freeze.json"],
          "bytes":by_path[f"parent-{version}-freeze.json"]["bytes"],"sha256":by_path[f"parent-{version}-freeze.json"]["sha256"]}
        t=f"parent-{version}-terminal.json"
        freeze[key]["terminal_receipt"]={"path":t,"source_path":parent_copy_paths[t],
          "bytes":by_path[t]["bytes"],"sha256":by_path[t]["sha256"]}
    freeze["parent_v4i"]["results_report_sha256"]=sha_file(ROOT/"docs/replacement-loss-v9-gman-fresh-node-v4i-results.md")
    freeze["parent_v4j"]={
        "protocol_id":terminal["protocol_id"],"state_slice":terminal["state_slice"],
        "packet_sha256":terminal["packet_sha256"],"terminal_status":terminal["reproducibility_status"],
        "command_id":"cmd-jkn3t","observed_exit_code":terminal["observed_exit_code"],
        "evaluator_started":terminal["evaluator_started"],
        "candidate_preparation_completed":terminal["candidate_preparation_completed"],
        "candidate_freeze_completed":terminal["candidate_freeze_completed"],
        "challenge_fixtures_revealed_after_candidate_freeze":terminal["challenge_fixtures_revealed_after_candidate_freeze"],
        "prediction_inventory":terminal["prediction_inventory"],
        "freeze_file":{"path":"parent-v4j-freeze.json","source_path":parent_copy_paths["parent-v4j-freeze.json"],
          "bytes":by_path["parent-v4j-freeze.json"]["bytes"],"sha256":by_path["parent-v4j-freeze.json"]["sha256"]},
        "terminal_receipt":{"path":"parent-v4j-terminal.json","source_path":parent_copy_paths["parent-v4j-terminal.json"],
          "bytes":by_path["parent-v4j-terminal.json"]["bytes"],"sha256":by_path["parent-v4j-terminal.json"]["sha256"]},
        "run_stop":{"path":"parent-v4j-run-stop.json","source_path":parent_copy_paths["parent-v4j-run-stop.json"],
          "bytes":by_path["parent-v4j-run-stop.json"]["bytes"],"sha256":by_path["parent-v4j-run-stop.json"]["sha256"]},
        "run_stop_sha256":run_stop["v4_stop_sha256"],"run_stop_file_sha256":sha_file(V4J_RUN_STOP),
        "error":run_stop["error"],"results_report_sha256":sha_file(V4J_RESULTS),
    }
    freeze["continuation_parent"]={"protocol_id":terminal["protocol_id"],"packet_sha256":terminal["packet_sha256"],
      "terminal_status":terminal["reproducibility_status"],"continuation_attempt":"V4k"}
    original=v4h["source_identity"]
    freeze["source_identity"]={
        "git_head":subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip(),
        "packet_file_count":len(rows),
        "packet_source_sha256":canonical_digest([{k:r[k] for k in ("path","bytes","sha256")} for r in rows]),
        "patched_replicator_sha256":original["patched_replicator_sha256"],
        "replicator_sha256":original["replicator_sha256"],
        "runner_sha256":original["runner_sha256"],
        "parent_v4h_freeze_sha256":sha_bytes(v4h_bytes),
        "parent_v4i_freeze_sha256":v4j["source_identity"]["parent_v4i_freeze_sha256"],
        "parent_v4j_freeze_sha256":sha_bytes(v4j_bytes),
        "parent_v4j_terminal_receipt_sha256":sha_bytes(terminal_bytes),
        "results_report_sha256":sha_file(V4J_RESULTS),
        "freeze_builder_sha256":sha_file(Path(__file__)),
    }
    freeze["budget"]={
        "related_spend_baseline_at_freeze_usd":baseline,
        "frozen_pre_v4h_related_spend_register_usd":PRE_V4H_REGISTER_USD,
        "settled_v4h_mission_spend_usd":V4H_SETTLED_USD,
        "existing_gman_mission_accrued_usd":args.mission_accrued_usd,
        "v4k_incremental_hard_cap_usd":round(AGGREGATE_CAP_USD-baseline,4),
        "aggregate_worst_case_usd":round(baseline+remaining_cost,4),
        "aggregate_headroom_after_hard_cap_usd":headroom,
        "destination_node_max_minutes":NODE_MAX_MINUTES,
        "node_remaining_minutes_at_freeze":remaining_minutes,
        "destination_node_remaining_max_usd":remaining_cost,
        "evaluator_command_max_minutes":EVALUATOR_MAX_MINUTES,
        "post_evaluator_receipt_and_export_reserve_minutes":7,
        "watchdog_receipt_export_and_shutdown_reserve_minutes":WATCHDOG_STOP_RESERVE_MINUTES,
        "max_rate_usd_per_minute":RATE_USD_PER_MINUTE,
        "user_authorized_related_aggregate_usd":AGGREGATE_CAP_USD,
        "related_spend_by_node_usd":dict(v4j["budget"]["related_spend_by_node_usd"],**{
          "theseus-v9-v4i-cpu8-20260929":args.mission_accrued_usd}),
        "storage_cost":v4j["budget"]["storage_cost"],
        "cost_accounting_note":"Baseline includes the frozen V9 register, settled V4h cost, and existing GMan mission accrual through V4j. Remaining cost is bounded by the original node-start epoch and 315-minute lifetime.",
    }
    freeze["execution"].update({
        "gman_command":"bash /home/dev/theseus-v9/v4k/watchdog.sh",
        "node_start_epoch":NODE_START_EPOCH,"freeze_capture_epoch":args.capture_epoch,
        "node_remaining_minutes_at_freeze":remaining_minutes,"max_node_minutes":NODE_MAX_MINUTES,
        "outer_watchdog_max_elapsed_minutes":NODE_MAX_MINUTES-WATCHDOG_STOP_RESERVE_MINUTES,
        "hard_node_stop_elapsed_minutes":NODE_MAX_MINUTES,
        "watchdog_receipt_export_and_shutdown_reserve_minutes":WATCHDOG_STOP_RESERVE_MINUTES,
        "max_incremental_spend_usd":round(AGGREGATE_CAP_USD-baseline,4),
        "aggregate_related_spend_cap_usd":AGGREGATE_CAP_USD,
        "projected_max_aggregate_usd":round(baseline+remaining_cost,4),
        "aggregate_headroom_usd":headroom,"rate_ceiling_usd_per_minute":RATE_USD_PER_MINUTE,
        "gman_max_duration_minutes_parameter":"omitted_for_active_node_detached_command","one_evaluator_attempt":True,
    })
    freeze["experimental_design"]["baseline"]="frozen V9 primary outcome FAIL; V4h timed out, V4i had a source identity failure, and V4j stopped on the patched-source digest before candidate preparation"
    freeze["experimental_design"]["direction"]="reproduce the frozen V9 result after binding all three V9 source identities to the original V4h freeze"
    freeze["experimental_design"]["full_cost_boundary"]=(
        f"Baseline {baseline:.4f} USD; hard aggregate cap {AGGREGATE_CAP_USD:.2f} USD; "
        f"at most {remaining_minutes} node minutes remain ({remaining_cost:.5f} USD); "
        f"projected aggregate maximum {baseline+remaining_cost:.4f} USD."
    )
    freeze["experimental_design"]["stopping_rule"]=(
        "One V4k attempt. Stop before candidate evaluation on any packet, source identity, archive, custody, recovery, runtime, input, or budget mismatch. "
        "The 300-minute evaluator limit is dynamically reduced by setup and reserve. The watchdog stops by original node-start+311m; hard node lifetime is 315m. "
        "No retry, threshold change, fixture change, or reuse of V4i/V4j run outputs."
    )
    freeze["custody_transfer"]["runtime_node_restore_method"]="existing CPU-8 node restored from V4e snapshot before V4i; V4k uses a fresh v4k output directory and does not reuse V4i or V4j outputs"
    freeze.pop("packet_sha256",None)
    freeze["packet_sha256"]=canonical_digest(freeze)
    OUTPUT.write_text(json.dumps(freeze,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"protocol_id":PROTOCOL_ID,"packet_sha256":freeze["packet_sha256"],
      "packet_file_count":len(rows),"runner_sha256":original["runner_sha256"],
      "replicator_sha256":original["replicator_sha256"],"patched_replicator_sha256":original["patched_replicator_sha256"],
      "baseline_usd":baseline,"remaining_node_minutes":remaining_minutes,
      "projected_aggregate_usd":round(baseline+remaining_cost,4),"aggregate_cap_usd":AGGREGATE_CAP_USD,
      "headroom_usd":headroom},sort_keys=True))


if __name__ == "__main__":
    main()
