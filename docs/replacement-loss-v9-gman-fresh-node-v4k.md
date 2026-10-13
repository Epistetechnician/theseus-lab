# Replacement-loss V9 GMan fresh-node continuation V4k

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4k`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Purpose and boundary

V4k follows V4j's terminal pre-candidate error. V4j reached the run phase but stopped before candidate preparation, candidate freeze, challenge reveal, or prediction generation. V4i also stopped before candidate preparation. V4k preserves both terminal records and uses a fresh output directory; no V4i or V4j run state is reused.

V4k is a new, single-attempt continuation after V4h's terminal
`INCONCLUSIVE_TIMEOUT`. V4h remains unchanged scientifically and terminal; its
52 partial predictions are not V4k inputs. V4k uses a fresh node restored from
the V4e workspace snapshot and a new `v4k` output directory. The frozen V9
target, checkpoints, fixture bytes, replacement methods, controls, thresholds,
and outcome order do not change.

V4k carries the exact V4h freeze and a digest-only terminal summary. It reuses
the exact V4h custody importer, primary-report recovery, and scientific runner
bytes through small identity adapters; each adapter checks the parent helper
digest before use. Only the continuation identity, timeout receipt path, and
execution ceiling differ. Held-out challenge bytes are re-used as frozen; V4k
does not generate or select new challenge fixtures.

## Corrected source identity binding

V4i bound adapter/helper digests in the V9 source fields. V4j corrected the V9 runner and replicator fields but still placed the wrapper hash in `patched_replicator_sha256`. V4k binds all three values from the original V4h freeze: runner `4cd2f003df9c7a82846e71c394e13a7771c40e16bae9543496197333cb4262a1`, replicator `005ff243087001441c44eee22408c55a3202645eb2d9648e968417ecb5686923`, and the frozen V3 in-memory patch `6892b0a99bdfb7fed3950f06c9adc4b16dcaf87c9dcca890b25bec1829d2436e`. No scientific method, fixture, threshold, control, or outcome rule changes.

## Frozen scientific design

- Target V9 packet: `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Primary outcome: `FAIL`, with 21 of 27 ridge stages passing.
- Three architecture families, three fixture seeds, and three sequential
  replacement stages per trajectory.
- Ridge, CKA, and logit-distillation methods plus wrong-module, random, and
  no-op controls remain unchanged.
- Candidate and tensor-state checks precede challenge reveal; held-out
  predictions are sealed before outcome evaluation.
- `REPRODUCED_LOCAL` requires every frozen check and the primary `FAIL` outcome
  to match. A measured mismatch is `DIVERGED`; incomplete execution remains
  `INCONCLUSIVE`.
- One V4k attempt only. No retry, threshold changes, seed changes, fixture
  regeneration, candidate retuning, or outcome reinterpretation.

## Runtime, timeout, and budget

The runtime remains the V4h Linux x86_64, Python 3.12.13, one-PyTorch-thread
profile, with the frozen active distribution manifest and all 26 V9 lock rows.
`locked_environment_pass=false` remains the truthful status because the
runtime has 51 active distributions and the lock has 26 entries.

The conservative related-spend baseline at the V4k freeze is 2.1132 USD: the prior 1.1670 USD register, settled V4h cost 0.6390 USD, and the GMan mission accrual captured at Unix epoch 1790692169. The aggregate hard cap remains 5.00 USD. The existing node has at most 273 billed minutes left to its frozen 315-minute lifetime limit (2.1949 USD at 0.00804 USD/min); projected aggregate maximum is 4.3081 USD, leaving 0.6919 USD headroom. The evaluator timeout is at most 300 minutes and dynamically reduced for setup time and the receipt/export reserve. Do not start if the quote exceeds 0.00804 USD/min or the remaining budget cannot cover the frozen ceiling.

The aggregate cap is enforced by this frozen operator runbook and the command
and node deadlines; the GMan workspace's broader account cap is not changed.
Because the node is already running, GMan's `max_duration_minutes` argument is
not available for this detached command (that argument applies to queued
`on_wake` boot commands). A digest-checked packet watchdog instead terminates
the bootstrap no later than 311 minutes after the recorded node-start epoch,
leaving four minutes for a digest-only stop receipt and operator export/shutdown.
The evaluator still has a 300-minute hard limit, dynamically reduced for setup
time; it does not inherit the watchdog's longer deadline. The operator stops
the node immediately after exporting the summary or stop receipt. Snapshot
storage is under the currently verified 250-GiB free allowance.

## Result and claim limits

Prompts, model assets, held-out evaluation data, and prediction payloads stay
in GMan custody. Only digest-bound protocol, runtime, and terminal summary
receipts may be exported. Reviewer status is `NOT_SUBMITTED`. A completed
passing run can support same-owner `REPRODUCED_LOCAL` for the frozen V9 result
under this second runtime. It cannot establish independent science, external
replication, novelty priority, all-architecture validity, mechanistic
understanding, Theseus identity, consciousness, alignment, human equivalence,
or production authorization.


V4k is one continuation attempt. If its preflight or evaluator stops, retain the terminal status; do not retry or tune this packet.


V4k is one continuation attempt. If its preflight or evaluator stops, retain the terminal status; do not retry or tune this packet.
