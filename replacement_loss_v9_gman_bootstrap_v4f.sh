#!/usr/bin/env bash
set -euo pipefail

# State slices: theseus-lab-sandbox-v1 and
# theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f.
ROOT="/home/dev/theseus-v9"
PACKET="$ROOT/v4f"
VENV="$ROOT/v4e/venv"
RUNTIME_SHA="0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63"

test -s "$PACKET/freeze.json"
test -s "$PACKET/protocol.md"
test -s "$PACKET/runner.py"
test -s "$PACKET/importer.py"
test -x "$VENV/bin/python"

python3 "$PACKET/importer.py"

"$VENV/bin/python" - "$PACKET/freeze.json" <<'PY'
import json
import sys
from pathlib import Path

freeze = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if freeze.get("runtime", {}).get("python") != "3.12.13":
    raise SystemExit("V4e snapshot Python version differs from frozen V4f")
PY
"$VENV/bin/python" - "$RUNTIME_SHA" <<'PY'
import importlib.metadata
if importlib.metadata.version("packaging") != "26.3":
    raise SystemExit("active packaging module does not match frozen V4f")
print("active_packaging=26.3")
PY

"$VENV/bin/python" "$PACKET/runner.py" preflight --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-preflight.json" "$PACKET/runtime-preflight-v4.json" "$RUNTIME_SHA" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

freeze_path, preflight_path, runtime_path = map(Path, sys.argv[1:4])
expected_runtime_sha = sys.argv[4]
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
if not preflight_path.is_file() or not runtime_path.is_file():
    raise SystemExit("V4f runtime preflight did not produce bound receipts")
preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
material = {key: value for key, value in preflight.items() if key != "preflight_sha256"}
runtime_record = preflight.get("runtime_preflight", {})
if (
    preflight.get("all_checks_pass") is not True
    or preflight.get("packet_sha256") != freeze.get("packet_sha256")
    or preflight.get("preflight_sha256") != canonical(material)
    or preflight.get("held_out_predictions_or_outcomes_opened") is not False
    or runtime_record.get("runtime_match") is not True
    or runtime_record.get("target_lock_subset_pass") is not True
    or runtime_record.get("derived_runtime_manifest_sha256") != expected_runtime_sha
    or runtime_record.get("packet_sha256") != freeze.get("packet_sha256")
):
    raise SystemExit("V4f preflight failed its digest, runtime, lock, or sealed-data gate; evaluator will not start")
PY

"$VENV/bin/python" "$PACKET/runner.py" run --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-run-receipt.json" <<'PY'
import json
import sys
from pathlib import Path

freeze = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt_path = Path(sys.argv[2])
if not receipt_path.is_file():
    raise SystemExit("V4f run receipt is missing")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if (
    receipt.get("packet_sha256") != freeze.get("packet_sha256")
    or receipt.get("reproducibility_status") not in ("REPRODUCED_LOCAL", "DIVERGED")
):
    raise SystemExit("V4f evaluator did not complete a bound comparison")
PY

