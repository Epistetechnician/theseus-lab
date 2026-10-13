#!/usr/bin/env bash
set -euo pipefail
# State slices: theseus-lab-sandbox-v1 and
# theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h.
ROOT="/home/dev/theseus-v9"
PACKET="$ROOT/v4h"
VENV="$ROOT/v4e/venv"
RUNTIME_SHA="0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63"
test -s "$PACKET/freeze.json"
test -s "$PACKET/protocol.md"
test -s "$PACKET/runner.py"
test -s "$PACKET/importer.py"
test -s "$PACKET/repair_primary.py"
test -x "$VENV/bin/python"
python3 "$PACKET/importer.py"
python3 "$PACKET/repair_primary.py"
"$VENV/bin/python" "$PACKET/runner.py" preflight --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-preflight.json" "$PACKET/runtime-preflight-v4.json" "$PACKET/transport-preflight-v4h.json" "$PACKET/primary-report-repair-v4h.json" "$RUNTIME_SHA" <<'PY'
import hashlib, json, sys
from pathlib import Path
freeze_path, preflight_path, runtime_path, transport_path, repair_path = map(Path, sys.argv[1:6])
expected_runtime_sha = sys.argv[6]
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
if not all(path.is_file() for path in (preflight_path, runtime_path, transport_path, repair_path)):
    raise SystemExit("V4h preflight did not produce all digest-bound receipts")
preflight, runtime, transport, repair = [json.loads(path.read_text(encoding="utf-8")) for path in (preflight_path, runtime_path, transport_path, repair_path)]
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if (preflight.get("all_checks_pass") is not True or preflight.get("packet_sha256") != freeze.get("packet_sha256")
    or preflight.get("preflight_sha256") != canonical({key:value for key,value in preflight.items() if key!="preflight_sha256"})
    or preflight.get("held_out_predictions_or_outcomes_opened") is not False
    or runtime.get("runtime_match") is not True or runtime.get("target_lock_subset_pass") is not True
    or runtime.get("derived_runtime_manifest_sha256") != expected_runtime_sha or runtime.get("packet_sha256") != freeze.get("packet_sha256")
    or transport.get("all_checks_pass") is not True or transport.get("held_out_predictions_or_outcomes_opened") is not False
    or transport.get("receipt_sha256") != canonical({key:value for key,value in transport.items() if key!="receipt_sha256"})
    or transport.get("packet_sha256") != freeze.get("packet_sha256")
    or repair.get("all_checks_pass") is not True or repair.get("held_out_predictions_or_outcomes_opened") is not False
    or repair.get("receipt_sha256") != canonical({key:value for key,value in repair.items() if key!="receipt_sha256"})
    or repair.get("packet_sha256") != freeze.get("packet_sha256")
    or repair.get("restored_report_sha256") != freeze.get("target",{}).get("primary_outcome_manifest_sha256")):
    raise SystemExit("V4h preflight, custody, recovery, lock, or runtime guard failed; evaluator will not start")
PY
command -v timeout >/dev/null
timeout --signal=TERM --kill-after=30s 73m "$VENV/bin/python" "$PACKET/runner.py" run --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-run-receipt.json" <<'PY'
import json, sys
from pathlib import Path
freeze=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
path=Path(sys.argv[2])
if not path.is_file(): raise SystemExit("V4h run receipt is missing")
receipt=json.loads(path.read_text(encoding="utf-8"))
if receipt.get("packet_sha256")!=freeze.get("packet_sha256") or receipt.get("reproducibility_status") not in ("REPRODUCED_LOCAL","DIVERGED"):
    raise SystemExit("V4h evaluator did not complete a bound comparison")
PY

