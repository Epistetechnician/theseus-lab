# Replacement-loss V9 GMan fresh-node continuation V4h

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Purpose and boundary

V4h is a new, single-attempt continuation after V4g stopped during preflight.
V4g's frozen one-attempt rule remains terminal: V4g will not be rerun. Its
stop record and importer receipt remain in GMan custody and are digest-bound
below.

V4g found that the cloned V3 primary outcome report differed from both the
frozen archive bytes and the one previously known post-error version. V4h
adds a custody-recovery step before preflight. It hashes the frozen V9 input
archive and manifest, finds exactly one archived outcome-manifest with the
already frozen primary-report digest, preserves any different cloned bytes
under the V4h quarantine path, and atomically restores only that report from
the matched archive member. It records pre-repair, preserved, and restored
digests. If any archive, manifest, uniqueness, path, or preservation check
fails, the single attempt stops before evaluator launch.

The scientific target, checkpoints, candidate rules, challenge bytes, methods,
controls, thresholds, outcome order, and V9 source identity remain unchanged.
No held-out prediction or outcome is opened by the recovery step.

## Frozen scientific design

- Target V9 packet: `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Primary outcome: `FAIL`, with 21 of 27 ridge stages passing.
- Three architecture families, three fixture seeds, and three sequential
  replacement stages per trajectory.
- Ridge, CKA, and logit-distillation methods plus wrong-module, random, and
  no-op controls remain as frozen.
- Candidate file and tensor-state checks precede challenge reveal; predictions
  are sealed before held-out outcomes open.
- `REPRODUCED_LOCAL` requires every frozen check and the primary `FAIL` outcome
  to match. A completed mismatch is `DIVERGED`; an incomplete run is
  `INCONCLUSIVE`.
- One V4h attempt only. No retry, threshold change, seed change, fixture
  regeneration, candidate retuning, or outcome reinterpretation.

## Custody, runtime, and budget

The archive, model assets, and evaluation data remain in GMan custody. V4h
restores the V4e workspace snapshot `snap-t3r2t`, reruns the frozen custody
import, performs digest-bound primary-report recovery, and then requires
passing transport and runtime preflight receipts before the evaluator starts.

Runtime identity remains Linux x86_64, Python 3.12.13, one PyTorch thread,
the frozen 51-entry active distribution map with SHA-256
`0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63`, and
all 26 exact V9 lock rows. `locked_environment_pass=false` remains truthful
because the full environment has 51 distributions and the lock has 26.

Settled related V9 node spend before V4h is `$1.1670` of the user-authorized
`$2.00` cap. V4h's incremental hard cap is `$0.75`; its CPU-8 node maximum is
82 billed minutes at `$0.00804/min` (`$0.65928`). The evaluator timeout is
73 minutes, with nine minutes reserved for setup, receipt export, and
shutdown. Worst-case related spend is `$1.9170`, leaving `$0.0830` under the
aggregate cap. Stop at the lower frozen incremental or node-time ceiling.

## Report and claim limits

Reviewer status is `NOT_SUBMITTED`. At most, a completed passing run supports
same-owner local cross-runtime reproduction of the frozen V9 `FAIL`. It does
not establish independent science, external replication, novelty priority,
all-architecture validity, mechanistic understanding, Theseus identity,
consciousness, alignment, human equivalence, or production authorization.

