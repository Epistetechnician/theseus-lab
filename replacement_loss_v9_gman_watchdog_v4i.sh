#!/usr/bin/env bash
set -uo pipefail
# State slices: theseus-lab-sandbox-v1,
# theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i, and
# global-experiment-assurance-lifecycle-v1.
ROOT="${V4I_WORKSPACE_ROOT:-/home/dev/theseus-v9}"
PACKET="$ROOT/v4i"
NODE_MAX_SECONDS=$((315 * 60))
NODE_STOP_RESERVE_SECONDS=$((4 * 60))
NODE_START_EPOCH="${V4I_NODE_START_EPOCH:-}"

if [[ ! "$NODE_START_EPOCH" =~ ^[0-9]+$ ]]; then
    echo "V4i watchdog requires the frozen node-start epoch" >&2
    exit 125
fi

if ! python3 - "$PACKET/freeze.json" "$PACKET/watchdog.sh" "$PACKET/bootstrap.sh" <<'PY'
import hashlib, json, sys
from pathlib import Path

freeze_path, watchdog_path, bootstrap_path = map(Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
material = {key: value for key, value in freeze.items() if key != "packet_sha256"}
if freeze.get("packet_sha256") != canonical(material):
    raise SystemExit("V4i watchdog freeze digest mismatch")
if freeze.get("protocol_id") != "theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i":
    raise SystemExit("V4i watchdog protocol identity mismatch")
for key, path in (("watchdog", watchdog_path), ("bootstrap", bootstrap_path)):
    row = freeze.get(key, {})
    data = path.read_bytes()
    if len(data) != row.get("bytes") or hashlib.sha256(data).hexdigest() != row.get("sha256"):
        raise SystemExit(f"V4i {key} digest mismatch; evaluator will not start")
PY
then
    exit 125
fi

NOW_EPOCH="$(date +%s)"
WATCHDOG_DEADLINE_EPOCH=$(( NODE_START_EPOCH + NODE_MAX_SECONDS - NODE_STOP_RESERVE_SECONDS ))
WATCHDOG_TIMEOUT_SECONDS=$(( WATCHDOG_DEADLINE_EPOCH - NOW_EPOCH ))
if (( WATCHDOG_TIMEOUT_SECONDS < 60 )); then
    python3 "$PACKET/write_timeout_receipt.py" --workspace-root "$ROOT" --exit-code 125 \
        --timeout-kind watchdog --timeout-seconds "$WATCHDOG_TIMEOUT_SECONDS" --evaluator-started false
    exit 125
fi
if ! command -v timeout >/dev/null 2>&1; then
    python3 "$PACKET/write_timeout_receipt.py" --workspace-root "$ROOT" --exit-code 127 \
        --timeout-kind watchdog --timeout-seconds "$WATCHDOG_TIMEOUT_SECONDS" --evaluator-started false
    exit 127
fi

set +e
timeout --signal=TERM --kill-after=30s "${WATCHDOG_TIMEOUT_SECONDS}s" \
    bash "$PACKET/bootstrap.sh"
CHILD_EXIT_CODE=$?
set -e
if (( CHILD_EXIT_CODE == 0 )); then
    exit 0
fi
if [[ -s "$PACKET/v4i-stop-receipt.json" ]]; then
    exit "$CHILD_EXIT_CODE"
fi

EVALUATOR_STARTED=false
STARTED_UTC=""
if [[ -s "$PACKET/evaluator-started.utc" ]]; then
    EVALUATOR_STARTED=true
    STARTED_UTC="$(cat "$PACKET/evaluator-started.utc")"
fi
ENDED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RECEIPT_ARGS=(--workspace-root "$ROOT" --exit-code "$CHILD_EXIT_CODE" --timeout-kind watchdog
    --timeout-seconds "$WATCHDOG_TIMEOUT_SECONDS" --evaluator-started "$EVALUATOR_STARTED"
    --ended-utc "$ENDED_UTC")
if [[ -n "$STARTED_UTC" ]]; then
    RECEIPT_ARGS+=(--started-utc "$STARTED_UTC")
fi
python3 "$PACKET/write_timeout_receipt.py" "${RECEIPT_ARGS[@]}"
exit "$CHILD_EXIT_CODE"
