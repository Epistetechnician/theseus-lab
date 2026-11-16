# Experiment lineage and the frozen-code rule

`replacement_loss_v*` and `theseus_v*` (plus their tests) are **frozen provenance**. Each version was
pre-registered, run once, and is cited by path and content hash in `docs/` and `rsi/ledger.md`.
They are never edited, reformatted, deduplicated or moved. `scripts/check_frozen.py` verifies a sha256
manifest (`frozen_manifest.json`) in CI and fails on any change.

The ~14k lines of near-duplicate runners are the cost of that guarantee: each version is a complete,
independently re-runnable record. Do not "clean them up".

## Rules for new work
1. New experiment versions import shared logic from a maintained module (start from the newest
   `*_core.py`, currently `replacement_loss_v36_core`) and add only what is new. Do not copy a runner.
2. When a version is run and cited, run `python scripts/check_frozen.py --update` to freeze it.
   `--update` refuses to re-hash a file that already changed.
3. Non-frozen code is linted (`ruff check .`) in CI.
