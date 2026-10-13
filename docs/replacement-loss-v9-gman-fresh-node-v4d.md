# Replacement-loss v9 GMan fresh-runtime continuation v4d

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4d`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Continuation boundary

V4d is a new one-attempt continuation after terminal V4c stopped before
candidate-integrity preflight. V4c's runtime reference matched its 51-package
Linux environment, while the frozen V9 runner required all installed
distributions to equal its 26-entry lock. V4c also found `packaging` 26.0 in
the reference runtime, while V9 pins 26.3.

V4d preserves the V9 source checkout, target packet, checkpoints, fixtures,
candidate rules, thresholds, controls, seeds, challenge bytes, result order,
and scientific success criterion. It creates a new Python environment from
the exact V3 runtime distribution set, changing only `packaging` from 26.0 to
the V9-pinned 26.3. It binds the derived full distribution manifest before
candidate fitting. Its runtime adapter checks every installed distribution
against that manifest and checks all 26 V9 lock entries for exact installed
versions. It adapts the V9 runner's full-set lock check in memory; it does not
change any V9 source or scientific evaluator bytes.

V4c remains terminal and immutable. V4d verifies its freeze, stop receipt,
preflight receipts, transport receipt, and reused input custody before
evaluation. Candidate bytes and tensor state must pass integrity checks
before challenge reveal; predictions must be sealed before held-out outcomes
are opened.

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

The new venv is created at `v4d/venv` with Python 3.12.13. Its package set is
derived from the V3-bound 51-distribution manifest with only
`packaging==26.3`; every other distribution must match the V3 manifest
exactly. The frozen V9 lock must contain 26 entries, and each entry must
match the installed version exactly. The resulting full runtime manifest is
recorded in the V4d preflight and run receipts.

The GMan aggregate authorization is `$2.00`. The settled related spend before
V4d was `$1.0428`, leaving `$0.9572`. V4d caps incremental node and receipt
handling at `$0.90`, retaining at least `$0.0572` under the cap. The single
evaluator command has a 73-minute maximum; allow at most five additional
minutes for receipt export and shutdown. Stop before held-out evaluation on
any packet, custody, runtime, input, candidate, or budget mismatch. An
incomplete V4d attempt is terminal and is not retried.

## Claim ceiling and nonclaims

The maximum claim is same-owner local cross-runtime reproduction of the
frozen V9 scientific `FAIL` under the explicitly digest-bound V4d Linux
runtime. Reviewer status is `NOT_SUBMITTED`.

This does not establish blind external replication, independent scientific
acceptance, novelty priority, architecture-general validity, general
mechanistic understanding, a Theseus identity proof, consciousness,
alignment, human equivalence, or production authorization.
