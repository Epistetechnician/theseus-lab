# Replacement-loss v9 GMan fresh-runtime reproduction v4c

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c.

Assurance overlay: global-experiment-assurance-lifecycle-v1.

## Continuation boundary

V4c is a new one-attempt continuation after V4b
f131e2b4e9a7fb194b26768ef2983d431a71ef18af66412f8ffbe8e92883ee3d ended
INCONCLUSIVE during preflight. The V3 packet artifact did not include its
protocol document, and the V4b bootstrap continued after the runner caught a
preflight error and returned exit code zero. V4b never reached candidate
fitting or held-out evaluation. Its stop receipts and transport receipt remain
bound in this packet.

V4c adds the exact frozen V3 protocol document and makes the bootstrap require
a digest-valid passing preflight receipt before it can call the evaluator.
The V4b attempt is terminal and is not retried. V4c uses the same V9 target,
checkpoints, fixtures, candidate rules, thresholds, controls, outcome order,
and scientific success criterion.

## Frozen scientific design

- Target V9 packet:
  9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959.
- Three architecture families, three fixture seeds, and three sequential
  replacement stages per trajectory.
- Primary metric and thresholds remain the frozen V9 causal-effect prediction
  and retention gates comparing ridge, CKA, and logit distillation.
- Wrong-module, random, and no-op controls remain unchanged.
- Replica candidates are refit from visible/adaptation data. Candidate file
  bytes and tensor states must pass before challenge reveal; predictions are
  sealed before held-out outcome evaluation.
- One V4c attempt is allowed. A complete match to the frozen V9 outcome FAIL
  yields REPRODUCED_LOCAL; a measured mismatch yields DIVERGED; incomplete
  work yields INCONCLUSIVE. No retry or retuning is permitted.

## Custody and preflight

V4c reuses the eight input artifacts already downloaded and verified by V4b.
It verifies their digest-only V4b transport receipt, the parent V4b stop
receipts, the V9 archive and input manifest, and the added V3 protocol
document. The V3 document must match the SHA-256 already bound by the frozen
V3 packet. The importer records a V4c transport receipt before the V9 runner
preflight.

The bootstrap then checks that the V9 preflight receipt exists, matches the
V4c packet digest, has a valid canonical digest, and declares all checks
passed. Any failure exits before the evaluator command. No command may infer
preflight success from a zero process exit code alone.

## Runtime, budget, and stop rule

The rebuilt environment must match the V3-observed Python 3.12.13 Linux x86_64
runtime, one PyTorch CPU thread, and dependency-manifest SHA-256
0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4.

GMan billing records $0.2092 for V1, $0.5424 for V3, and $0.1706 for the V4
node so far, including the terminal V4b preflight attempt: $0.9222 total.
V4c caps the remaining V4 node use at 51 minutes ($0.41004), with its
evaluator command capped at 49 minutes and the remaining two minutes reserved
for packet upload, transport verification, receipt export, and shutdown. The
aggregate maximum is $1.33224 against the authorized $1.34 cap, leaving
$0.00776.

Stop before evaluation on any packet, custody, runtime, input, candidate, or
budget mismatch. An incomplete attempt is retained as INCONCLUSIVE and is not
retried.

## Claim boundary

Prompts, model weights, token IDs, captures, predictions, row-level outcomes,
and traces remain in GMan or PrimaryED custody. Exported receipts contain
digests, counts, runtime identity, checks, and terminal status only. Reviewer
status is NOT_SUBMITTED.

The maximum claim is same-owner local cross-runtime reproduction of the
frozen V9 FAIL under the digest-bound Linux runtime. This is not blind external
replication, independent scientific acceptance, novelty priority,
architecture-general validity, mechanistic understanding, a Theseus identity
proof, consciousness evidence, alignment evidence, human equivalence, or
production authorization.
