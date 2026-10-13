#!/usr/bin/env bash
set -euo pipefail

# State slices: theseus-lab-sandbox-v1 and
# theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e.
ROOT="/home/dev/theseus-v9"
PACKET="$ROOT/v4e"
REFERENCE="$ROOT/receipts/runtime-preflight-v3.json"
LOCKFILE="$PACKET/runtime-exact-linux-v4e.lock.txt"
VENV="$PACKET/venv"

test -s "$PACKET/freeze.json"
test -s "$PACKET/protocol.md"
test -s "$PACKET/runner.py"
test -s "$PACKET/importer.py"
test -s "$REFERENCE"
command -v uv >/dev/null

python3 "$PACKET/importer.py"

"$(command -v uv)" python install 3.12.13 >/dev/null
"$(command -v uv)" venv --python 3.12.13 "$VENV" >/dev/null
"$VENV/bin/python" - "$REFERENCE" "$LOCKFILE" "$PACKET/freeze.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

reference_path, lock_path, freeze_path = map(Path, sys.argv[1:])
reference = json.loads(reference_path.read_text(encoding="utf-8"))["runtime"]
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
spec = freeze["runtime"]
rows = dict(reference["all_installed_distributions"])
if rows.get("packaging") != spec["packaging_base_version"]:
    raise SystemExit("base packaging version differs from frozen V4e runtime derivation")
if len(rows) != spec["expected_distribution_count"]:
    raise SystemExit("V3 reference distribution count differs from frozen V4e")
rows["packaging"] = spec["packaging_target_version"]
content = "".join(f"{name}=={version}\n" for name, version in sorted(rows.items()))
lock_path.write_text(content, encoding="utf-8")
print(json.dumps({"derived_distribution_count": len(rows), "packaging": rows["packaging"], "derived_lock_sha256": hashlib.sha256(content.encode()).hexdigest()}, sort_keys=True))
PY

"$(command -v uv)" pip install --python "$VENV/bin/python" --no-deps --requirement "$LOCKFILE" >/dev/null

"$VENV/bin/python" "$PACKET/runner.py" preflight --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-preflight.json" "$PACKET/runtime-preflight-v4.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

freeze_path, receipt_path, runtime_path = map(Path, sys.argv[1:])
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
canonical = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
material = {key: value for key, value in receipt.items() if key != "preflight_sha256"}
runtime = receipt.get("runtime_preflight", {})
if (
    receipt.get("all_checks_pass") is not True
    or receipt.get("packet_sha256") != freeze.get("packet_sha256")
    or receipt.get("preflight_sha256") != canonical(material)
    or runtime.get("runtime_match") is not True
    or runtime.get("target_lock_subset_pass") is not True
    or runtime.get("derived_runtime_manifest_sha256") is None
    or not runtime_path.is_file()
):
    raise SystemExit("V4e preflight failed its packet, runtime, lock-subset, or digest gate; evaluation will not start")
PY

"$VENV/bin/python" "$PACKET/runner.py" run --workspace-root "$ROOT"
python3 - "$PACKET/freeze.json" "$PACKET/v4-run-receipt.json" <<'PY'
import json
import sys
from pathlib import Path

freeze = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
receipt_path = Path(sys.argv[2])
if not receipt_path.is_file():
    raise SystemExit("V4e run receipt is missing")
receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
if receipt.get("packet_sha256") != freeze.get("packet_sha256") or receipt.get("reproducibility_status") not in ("REPRODUCED_LOCAL", "DIVERGED"):
    raise SystemExit("V4e evaluator did not complete a bound comparison")
PY
