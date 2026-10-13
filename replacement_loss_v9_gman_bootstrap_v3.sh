#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/dev/theseus-v9"
INPUT="$ROOT/input"
PACKET="$ROOT/v3"
ARCHIVE="$INPUT/gman-v9-input.tar.gz"
MANIFEST="$INPUT/metadata/input-manifest.json"
RUNTIME_REFERENCE="$PACKET/runtime-reference.json"
RUNTIME_OUTPUT="$ROOT/receipts/runtime-preflight-v3.json"
SOURCE_ROOT="$INPUT/source-checkout"

mkdir -p "$INPUT" "$ROOT/receipts"
curl --fail --location --silent --show-error "$THESEUS_INPUT_URL" -o "$ARCHIVE"
printf '%s  %s\n' "89af9ff9add82adbff199eda90d876653b041660f4bd1d6e2803fd47dcdb5bfe" "$ARCHIVE" | sha256sum --check --status
test "$(stat --format=%s "$ARCHIVE")" = "1727986827"
mkdir -p "$INPUT/metadata"
printf '%s  %s\n' "8682b390b1c40306cf73157e9b9ca3098e5ed6c90f856f238ef23a756a8f340a" "$MANIFEST" | sha256sum --check --status
test "$(tar --list --gzip --file "$ARCHIVE" | wc -l | tr -d ' ')" = "396"
tar --extract --gzip --file "$ARCHIVE" --directory "$INPUT"

uv python install 3.12.13 >/dev/null
uv venv --python 3.12.13 "$ROOT/venv" >/dev/null
"$ROOT/venv/bin/python" - "$RUNTIME_REFERENCE" "$ROOT/v3/runtime-exact-linux.lock.txt" <<'PY'
import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["runtime"]
distributions = reference["all_installed_distributions"]
rows = [f"{name}=={version}" for name, version in sorted(distributions.items())]
Path(sys.argv[2]).write_text("\n".join(rows) + "\n", encoding="utf-8")
PY
uv pip install --python "$ROOT/venv/bin/python" --no-deps --requirement "$ROOT/v3/runtime-exact-linux.lock.txt" >/dev/null

git init --quiet "$SOURCE_ROOT"
git -C "$SOURCE_ROOT" fetch --quiet "$INPUT/source.bundle" "202f459d6d89df9a038d78d58e0af825d287fcd8"
git -C "$SOURCE_ROOT" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" = "202f459d6d89df9a038d78d58e0af825d287fcd8"
cp --archive "$INPUT/source-overlay/." "$SOURCE_ROOT/"

"$ROOT/venv/bin/python" - "$SOURCE_ROOT" "$RUNTIME_REFERENCE" "$RUNTIME_OUTPUT" "$PACKET/freeze.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source_root, reference_path, output_path, freeze_path = map(Path, sys.argv[1:])
sys.path.insert(0, str(source_root))
import torch
import replacement_loss_v9_runner as runner

torch.set_num_threads(1)
runtime = runner._runtime_receipt()
runtime["cpu_thread_count"] = torch.get_num_threads()
expected = json.loads(reference_path.read_text(encoding="utf-8"))["runtime"]
fields = (
    "python", "platform", "machine", "processor", "dependencies",
    "all_installed_distributions", "locked_distributions",
    "dependency_manifest_sha256", "locked_environment_pass",
)
mismatches = [field for field in fields if runtime[field] != expected[field]]
mismatches.extend(field for field in ("python", "cpu_thread_count") if runtime[field] != ("3.12.13" if field == "python" else 1))
freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
receipt = {
    "protocol_id": freeze["protocol_id"],
    "packet_sha256": freeze["packet_sha256"],
    "state_slice": freeze["state_slice"],
    "runtime": runtime,
    "runtime_match": not mismatches,
    "mismatch_fields": sorted(set(mismatches)),
}
Path(output_path).parent.mkdir(parents=True, exist_ok=True)
Path(output_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
digest = hashlib.sha256(Path(output_path).read_bytes()).hexdigest()
print(json.dumps({
    "status": "PASS" if not mismatches else "INCONCLUSIVE",
    "runtime_match": not mismatches,
    "mismatch_fields": receipt["mismatch_fields"],
    "dependency_manifest_sha256": runtime["dependency_manifest_sha256"],
    "runtime_receipt_sha256": digest,
    "source_revision": "202f459d6d89df9a038d78d58e0af825d287fcd8",
}, sort_keys=True))
if mismatches:
    raise SystemExit(42)
PY
