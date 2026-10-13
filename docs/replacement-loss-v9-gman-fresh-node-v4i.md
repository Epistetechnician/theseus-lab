# Replacement-loss V9 GMan fresh-node continuation V4i

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Purpose and boundary

V4i is a new, single-attempt continuation after V4h's terminal
`INCONCLUSIVE_TIMEOUT`. V4h remains unchanged scientifically and terminal; its
52 partial predictions are not V4i inputs. V4i uses a fresh node restored from
the V4e workspace snapshot and a new `v4i` output directory. The frozen V9
target, checkpoints, fixture bytes, replacement methods, controls, thresholds,
and outcome order do not change.

V4i carries the exact V4h freeze and a digest-only terminal summary. It reuses
the exact V4h custody importer, primary-report recovery, and scientific runner
bytes through small identity adapters; each adapter checks the parent helper
digest before use. Only the continuation identity, timeout receipt path, and
execution ceiling differ. Held-out challenge bytes are re-used as frozen; V4i
does not generate or select new challenge fixtures.

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
- One V4i attempt only. No retry, threshold changes, seed changes, fixture
  regeneration, candidate retuning, or outcome reinterpretation.

## Runtime, timeout, and budget

The runtime remains the V4h Linux x86_64, Python 3.12.13, one-PyTorch-thread
profile, with the frozen active distribution manifest and all 26 V9 lock rows.
`locked_environment_pass=false` remains the truthful status because the
runtime has 51 active distributions and the lock has 26 entries.

The prior frozen V9 related-spend register is `$1.1670`; GMan's settled V4h
mission charge is `$0.6390`, so V4i starts from a conservative recorded
aggregate of `$1.8060`. The task ceiling is `$5.00` aggregate. At the verified
CPU-8 rate of `$0.00804/min`, V4i has a 315-minute node ceiling (`$2.5326`),
leaving `$0.6614` of aggregate headroom after the projected maximum. The
evaluator timeout is at most 300 minutes; the bootstrap reduces it if setup
uses the 15-minute remainder. A seven-minute post-evaluator reserve is kept
for the digest-only stop/run summary, export, and shutdown. GMan's detached
command ceiling is 314 minutes; stop the node by 315 billed minutes. If the
node quote exceeds `$0.00804/min`, do not start the evaluator.

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
