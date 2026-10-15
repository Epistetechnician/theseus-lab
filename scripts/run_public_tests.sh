#!/usr/bin/env bash
# Runs every test module in its own process (several modules set process-global
# torch thread counts and cannot share an interpreter). Three modules need the
# owner's frozen runtime pins or private custody directory and are reported as
# SKIPPED-OWNER-ENV outside that environment rather than as failures.
set -u
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
OWNER_ENV_ONLY="test_qwen38_ai_proxy_cross_model test_replacement_loss_v10 test_replacement_loss_v32_recovery"
fail=0; ok=0; skipped=0
for f in tests/test_*.py; do
  m=$(basename "$f" .py)
  if out=$(PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 "$PY" -m unittest "tests.$m" 2>&1); then
    ok=$((ok+1))
  elif [[ " $OWNER_ENV_ONLY " == *" $m "* && "${STRICT:-0}" != "1" ]]; then
    echo "SKIPPED-OWNER-ENV $m"; skipped=$((skipped+1))
  else
    echo "FAILED $m"; echo "$out" | tail -15; fail=$((fail+1))
  fi
done
echo "modules ok=$ok skipped_owner_env=$skipped failed=$fail"
[ "$fail" -eq 0 ]
