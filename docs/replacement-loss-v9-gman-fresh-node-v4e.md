# Replacement-loss v9 GMan fresh-runtime continuation v4e

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4e`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Continuation boundary

V4d remains terminal as `INCONCLUSIVE_NO_EVALUATION`: GMan rejected
cross-mission reuse of its existing node, and no V4d files or scientific
evaluation were sent to that node. V4e isolates this continuation by
restoring a verified snapshot into a separately named node.

V4e is a new one-attempt continuation after terminal V4c stopped before
candidate-integrity preflight. V4c's runtime reference matched its 51-package
Linux environment, while the frozen V9 runner required all installed
distributions to equal its 26-entry lock. V4c also found `packaging` 26.0 in
the reference runtime, while V9 pins 26.3.

V4e preserves the V9 source checkout, target packet, checkpoints, fixtures,
candidate rules, thresholds, controls, seeds, challenge bytes, result order,
and scientific success criterion. It creates a new Python environment from
the exact V3 runtime distribution set, changing only `packaging` from 26.0 to
the V9-pinned 26.3. It binds the derived full distribution manifest before
candidate fitting. Its runtime adapter checks every installed distribution
against that manifest and checks all 26 V9 lock entries for exact installed
versions. It adapts the V9 runner's full-set lock check in memory; it does not
change any V9 source or scientific evaluator bytes.

V4c remains terminal and immutable. V4e verifies its freeze, stop receipt,
preflight receipts, transport receipt, and reused input custody before
evaluation. Candidate bytes and tensor state must pass integrity checks
before challenge reveal; predictions must be sealed before held-out outcomes
are opened.

V4e loads the digest-bound V4 implementation stored at:
`v4/implementation.py` in the restored workspace. It does not modify that
implementation or the V9 scientific evaluator.

## Frozen scientific design

- Frozen V9 target packet:
  `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Baseline scientific outcome: `FAIL` (21 of 27 frozen ridge stages passed).
- Same three model families, three fixture seeds, and three sequential
  replacement stages per trajectory.
- Ridge, CKA, and logit-distillation methods and the wrong-module, random,
  and no-op controls remain unchanged.
- Primary success requires the V4 replication checks to match and the
  reproduced V9 outcome to remain `FAIL`. A complete match yields
  `REPRODUCED_LOCAL`; measured disagreement yields `DIVERGED`; incomplete
  execution yields `INCONCLUSIVE`.
- Repeat count is one. No threshold changes, seed changes, fixture
  regeneration, retuning, or retries are allowed.

## Runtime and resource boundary

The new venv is created at `v4e/venv` with Python 3.12.13. Its package set
is derived from the V3-bound 51-distribution manifest with only
`packaging==26.3`; every other distribution must match the V3 manifest
exactly. The frozen V9 lock must contain 26 entries, and each entry must
match the installed version exactly. The resulting full runtime manifest is
recorded in the V4e preflight and run receipts.

GMan mission isolation requires an explicit snapshot restore into a
separately named V4e node. The source is the stopped V4 node containing the
terminal V4c evidence and frozen V9 inputs. The V4e freeze binds the ready
snapshot identifier, source node identity, compatible CPU-8 shape, and image.
After restore, V4e verifies the V4c freeze and receipts plus the V9 input,
candidate, and challenge digests before held-out evaluation.

The GMan aggregate authorization is `$2.00`. Settled related spend before
V4e was `$1.0428`, leaving `$0.9572`. The V4e incremental hard cap is
`$0.90`. The source-node snapshot wake is budgeted at 15 minutes
(`$0.1206`); the V4e bootstrap/evaluator boot command, setup, receipt export,
and shutdown allowance
is at most 93 minutes (`$0.74772`). The combined planned maximum is 108
node-minutes (`$0.86832`), for a related aggregate maximum of `$1.91112`.
The single bootstrap/evaluator boot command has a 73-minute maximum. Stop before held-out
evaluation on any packet, custody, runtime, input, candidate, or budget
mismatch. One evaluator attempt is permitted; an incomplete V4e attempt is
terminal and is not retried.

## Claim ceiling and nonclaims

The maximum claim is same-owner local cross-runtime reproduction of the
frozen V9 scientific `FAIL` under the explicitly digest-bound V4e Linux
runtime. Reviewer status is `NOT_SUBMITTED`.

This does not establish blind external replication, independent scientific
acceptance, novelty priority, architecture-general validity, general
mechanistic understanding, a Theseus identity proof, consciousness,
alignment, human equivalence, or production authorization.
