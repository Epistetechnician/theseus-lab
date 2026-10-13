#!/usr/bin/env bash
set -euo pipefail
# State slices: theseus-lab-sandbox-v1,
# theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j, and
# global-experiment-assurance-lifecycle-v1.
ROOT="/home/dev/theseus-v9"
PACKET="$ROOT/v4j"
VENV="$ROOT/v4e/venv"
RUNTIME_SHA="0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63"
NODE_START_EPOCH="${V4J_NODE_START_EPOCH:-$(date +%s)}"
NODE_MAX_SECONDS=$((315 * 60))
POSTRUN_RESERVE_SECONDS=$((7 * 60))
EVALUATOR_MAX_SECONDS=$((300 * 60))

test -s "$PACKET/freeze.json"
test -s "$PACKET/protocol.md"
test -s "$PACKET/runner.py"
test -s "$PACKET/importer.py"
test -s "$PACKET/repair_primary.py"
test -s "$PACKET/write_timeout_receipt.py"
test -s "$PACKET/watchdog.sh"
test -x "$VENV/bin/python"

python3 - "$PACKET/freeze.json" "$PACKET" <<'PY'
import hashlib, json, sys
from pathlib import Path
freeze_path, packet = Path(sys.argv[1]), Path(sys.argv[2])
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
if freeze.get("packet_sha256") != canonical(material):
    raise SystemExit("V4j canonical packet digest mismatch")
if freeze.get("protocol_id") != "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j":
    raise SystemExit("V4j protocol identity mismatch")
rows = [
    ("packet_document", "protocol.md"), ("wrapper", "runner.py"),
    ("bootstrap", "bootstrap.sh"), ("importer", "importer.py"),
    ("primary_report_repair", "repair_primary.py"), ("watchdog", "watchdog.sh"),
]
for key, name in rows:
    row, path = freeze[key], packet / name
    if not path.is_file() or path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit(f"V4j packet file mismatch: {key}")
for row in list(freeze["parent_helpers"].values()) + [freeze["timeout_receipt_helper"], freeze["parent_v4h"]["freeze_file"], freeze["parent_v4h"]["terminal_receipt"], freeze["parent_v4i"]["freeze_file"], freeze["parent_v4i"]["terminal_receipt"]]:
    path = packet / row["path"]
    if not path.is_file() or path.stat().st_size != row["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise SystemExit(f"V4j auxiliary packet file mismatch: {row['path']}")
if freeze.get("budget", {}).get("user_authorized_related_aggregate_usd") != 5.0:
    raise SystemExit("V4j aggregate cap differs from its frozen ceiling")
if freeze.get("budget", {}).get("evaluator_command_max_minutes") != 300 or freeze.get("budget", {}).get("destination_node_max_minutes") != 315:
    raise SystemExit("V4j evaluator or node time ceiling differs from its frozen limit")
PY

python3 "$PACKET/importer.py"
python3 "$PACKET/repair_primary.py"
"$VENV/bin/python" "$PACKET/runner.py" preflight --workspace-root "$ROOT"

python3 - "$PACKET/freeze.json" "$PACKET/v4-preflight.json" "$PACKET/runtime-preflight-v4.json" "$PACKET/transport-preflight-v4h.json" "$PACKET/primary-report-repair-v4h.json" "$RUNTIME_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
freeze_path, preflight_path, runtime_path, transport_path, repair_path = map(Path, sys.argv[1:6])
expected_runtime_sha = sys.argv[6]
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
required = (preflight_path, runtime_path, transport_path, repair_path)
if not all(path.is_file() for path in required):
    raise SystemExit("V4j preflight did not produce all required receipts")
preflight, runtime, transport, repair = [json.loads(path.read_text(encoding="utf-8")) for path in required]
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if (preflight.get("all_checks_pass") is not True or preflight.get("packet_sha256") != freeze.get("packet_sha256")
    or preflight.get("preflight_sha256") != canonical({key:value for key,value in preflight.items() if key!="preflight_sha256"})
    or preflight.get("held_out_predictions_or_outcomes_opened") is not False
    or runtime.get("runtime_match") is not True or runtime.get("target_lock_subset_pass") is not True
    or runtime.get("derived_runtime_manifest_sha256") != expected_runtime_sha or runtime.get("packet_sha256") != freeze.get("packet_sha256")
    or transport.get("all_checks_pass") is not True or transport.get("held_out_predictions_or_outcomes_opened") is not False
    or transport.get("receipt_sha256") != canonical({key:value for key,value in transport.items() if key!="receipt_sha256"})
    or transport.get("packet_sha256") != freeze.get("packet_sha256")
    or transport.get("protocol_id") != freeze.get("protocol_id")
    or repair.get("all_checks_pass") is not True or repair.get("held_out_predictions_or_outcomes_opened") is not False
    or repair.get("receipt_sha256") != canonical({key:value for key,value in repair.items() if key!="receipt_sha256"})
    or repair.get("packet_sha256") != freeze.get("packet_sha256")
    or repair.get("protocol_id") != freeze.get("protocol_id")
    or repair.get("restored_report_sha256") != freeze.get("target",{}).get("primary_outcome_manifest_sha256")):
    raise SystemExit("V4j custody, recovery, lock, or runtime preflight failed; evaluator will not start")
PY

ELAPSED_BEFORE_EVALUATOR=$(( $(date +%s) - NODE_START_EPOCH ))
EVALUATOR_TIMEOUT_SECONDS=$(( NODE_MAX_SECONDS - ELAPSED_BEFORE_EVALUATOR - POSTRUN_RESERVE_SECONDS ))
if (( EVALUATOR_TIMEOUT_SECONDS > EVALUATOR_MAX_SECONDS )); then
    EVALUATOR_TIMEOUT_SECONDS="$EVALUATOR_MAX_SECONDS"
fi
if (( EVALUATOR_TIMEOUT_SECONDS < 60 )); then
    python3 "$PACKET/write_timeout_receipt.py" --workspace-root "$ROOT" --exit-code 125 \
        --timeout-kind evaluator --timeout-seconds "$EVALUATOR_TIMEOUT_SECONDS" --evaluator-started false
    exit 125
fi

command -v timeout >/dev/null
EVALUATOR_STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$EVALUATOR_STARTED_UTC" > "$PACKET/evaluator-started.utc"
if timeout --signal=TERM --kill-after=30s "${EVALUATOR_TIMEOUT_SECONDS}s" \
    "$VENV/bin/python" "$PACKET/runner.py" run --workspace-root "$ROOT"; then
    EVALUATOR_EXIT_CODE=0
else
    EVALUATOR_EXIT_CODE=$?
fi
EVALUATOR_ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if (( EVALUATOR_EXIT_CODE != 0 )); then
    python3 "$PACKET/write_timeout_receipt.py" --workspace-root "$ROOT" \
        --exit-code "$EVALUATOR_EXIT_CODE" --timeout-kind evaluator --timeout-seconds "$EVALUATOR_TIMEOUT_SECONDS" \
        --evaluator-started true --started-utc "$EVALUATOR_STARTED_UTC" --ended-utc "$EVALUATOR_ENDED_UTC"
    exit "$EVALUATOR_EXIT_CODE"
fi

python3 - "$PACKET/freeze.json" "$PACKET/v4-run-receipt.json" "$PACKET/v4j-run-summary.json" <<'PY'
import hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
freeze_path, receipt_path, summary_path = map(Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
if not receipt_path.is_file():
    raise SystemExit("V4j evaluator exited without a completed runner receipt")
receipt_bytes = receipt_path.read_bytes()
receipt = json.loads(receipt_bytes)
status = receipt.get("reproducibility_status")
if receipt.get("packet_sha256") != freeze.get("packet_sha256") or status not in ("REPRODUCED_LOCAL", "DIVERGED"):
    raise SystemExit("V4j evaluator receipt does not bind a completed frozen comparison")
checks = receipt.get("checks")
summary = {
    "protocol_id": freeze["protocol_id"],
    "state_slice": freeze["state_slice"],
    "packet_sha256": freeze["packet_sha256"],
    "parent_v4h_packet_sha256": freeze["parent_v4h"]["packet_sha256"],
    "reproducibility_status": status,
    "assurance_level": "REPRODUCED_LOCAL" if status == "REPRODUCED_LOCAL" else "INCONCLUSIVE",
    "reviewer_status": "NOT_SUBMITTED",
    "runner_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    "check_count": len(checks) if isinstance(checks, dict) else None,
    "passing_check_count": sum(value is True for value in checks.values()) if isinstance(checks, dict) else None,
    "model_prompts_outputs_weights_and_evaluation_data_exported": False,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
result_path = os.environ.get("GMN_RESULT_PATH")
if result_path:
    Path(result_path).write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True))
PY
