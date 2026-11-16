#!/usr/bin/env python3
"""Guard for pre-registered experiment code.

Versioned experiment files (replacement_loss_v*, theseus_v*, and their tests) are frozen
provenance: results and docs cite them by path and by the hash of their contents. They must never
be edited or moved. This script records a sha256 manifest and fails if any frozen file changes.

  python scripts/check_frozen.py            # verify (CI, pre-push)
  python scripts/check_frozen.py --update   # (re)write the manifest; only for NEW frozen files
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "frozen_manifest.json"
PATTERN = re.compile(r"^(tests/test_)?(replacement_loss_v|theseus_v)\d[^/]*\.py$")


def frozen_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    return sorted(f for f in out if PATTERN.match(f))


def digest(rel: str) -> str:
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def main() -> int:
    current = {f: digest(f) for f in frozen_files()}
    if "--update" in sys.argv:
        old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        changed = [f for f in old if f in current and old[f] != current[f]]
        if changed:
            print("refusing to re-hash modified frozen files:", *changed, sep="\n  ")
            return 1
        MANIFEST.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        print(f"manifest: {len(current)} frozen files ({len(current) - len(old)} new)")
        return 0
    manifest = json.loads(MANIFEST.read_text())
    bad = [f for f, h in manifest.items() if f not in current or current[f] != h]
    if bad:
        print("frozen experiment files were modified or removed:", *bad, sep="\n  ")
        return 1
    print(f"ok: {len(manifest)} frozen files unchanged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
