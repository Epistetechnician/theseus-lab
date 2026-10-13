#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dev/theseus-v9"
PACKET="$ROOT/v4"
RUNTIME_REFERENCE="$ROOT/receipts/runtime-preflight-v3.json"
LOCKFILE="$PACKET/runtime-exact-linux-v4.lock.txt"
VENV="$PACKET/venv"

test -s "$PACKET/freeze.json"
test -s "$PACKET/protocol.md"
test -s "$PACKET/runner.py"
test -s "$RUNTIME_REFERENCE"
command -v uv >/dev/null

if [[ ! -x "$VENV/bin/python" ]]; then
  "$(command -v uv)" python install 3.12.13 >/dev/null
  "$(command -v uv)" venv --python 3.12.13 "$VENV" >/dev/null
fi
"$VENV/bin/python" - "$RUNTIME_REFERENCE" "$LOCKFILE" <<'PY'
import json
import sys
from pathlib import Path

runtime = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["runtime"]
rows = [f"{name}=={version}" for name, version in sorted(runtime["all_installed_distributions"].items())]
Path(sys.argv[2]).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
"$(command -v uv)" pip install --python "$VENV/bin/python" --no-deps --requirement "$LOCKFILE" >/dev/null

"$VENV/bin/python" "$PACKET/runner.py" preflight --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-preflight.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

freeze = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt_path = Path(sys.argv[2])
if not receipt_path.is_file():
    raise SystemExit("preflight receipt is missing; evaluator will not start")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
canonical = lambda value: hashlib.sha256(
    json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
material = {key: value for key, value in receipt.items() if key != "preflight_sha256"}
if (
    receipt.get("all_checks_pass") is not True
    or receipt.get("packet_sha256") != freeze.get("packet_sha256")
    or receipt.get("preflight_sha256") != canonical(material)
):
    raise SystemExit("preflight receipt failed its packet, status, or digest gate; evaluator will not start")
PY
"$VENV/bin/python" "$PACKET/runner.py" run --workspace-root "$ROOT"
