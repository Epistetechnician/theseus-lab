# Replacement-loss v9 GMan fresh-runtime reproduction v4

State slices: \`theseus-lab-sandbox-v1\` and
\`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4\`.

Assurance overlay: \`global-experiment-assurance-lifecycle-v1\`.

## Purpose and claim boundary

V4 is a new, single-attempt continuation packet for the frozen replacement-loss
v9 target. It responds to the V3 stop, which happened before held-out prediction
because V9 compared two different digest domains: a serialized candidate
file-byte digest and a canonical tensor-state digest.

V4 uses the unchanged V9 model, fixtures, selected checkpoints, candidate
training rules, prediction code, outcomes, thresholds, and controls. It binds
the exact V3 packet and stop, rebuilds a clean Python environment on a new
GMan node, and refits the replica only from the frozen visible/adaptation data.
It checks all frozen candidate file bytes and treatment tensor states before
revealing held-out challenges. At prediction and outcome time, its frozen
loader checks both digest domains separately. No V3 replica candidate,
prediction, or outcome is reused.

This packet tests same-owner local cross-runtime reproducibility of the
existing V9 verdict, including the full replica fitting and evaluation path.
It does not select new checkpoints or generate new challenges. If every
predeclared replication check passes and the reproduced scientific outcome is
still \`FAIL\`, report \`REPRODUCED_LOCAL\`. A measured mismatch is
\`DIVERGED\`; an incomplete run is \`INCONCLUSIVE\`.

## Frozen design

- Target: the V9 packet and three checkpoint identities bound by the V3 freeze.
- Unit: three model families by three fixture seeds, with three sequential
  replacement stages per trajectory.
- Primary metric: the V9 causal-effect prediction comparison among ridge, CKA,
  and logit distillation, evaluated against the frozen retention gates.
- Controls: wrong-module, random, and no-op controls exactly as frozen in V9.
- Data split: replica fitting uses only frozen visible/adaptation data.
  Candidate file and state checks run before held-out challenge files are
  copied. Held-out predictions are sealed before outcome evaluation.
- Variance sources: CPU floating-point implementation, dependency/runtime
  drift, and stochastic fitting. The Linux runtime, dependency map, one-thread
  configuration, seeds, and inputs are digest-bound.
- Repeats: one V4 attempt over the nine frozen family/seed cases. The three
  fixture seeds are not three new checkpoints or independently sourced data.
- Thresholds and fixture-generation rules: unchanged from V9. No retuning,
  challenge regeneration, checkpoint reselection, or symmetry experiment.

## Runtime, budget, and stop rule

The replica runtime must match the V3-observed Linux runtime identity:
Python 3.12.13, x86_64, one PyTorch CPU thread, and dependency-manifest SHA-256
\`0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4\`.
V4 builds a new virtual environment from that pinned distribution map.

Prior V1/V3 Theseus node charges total $0.6558. GMan requires both nodes to be
running for node-to-node file transfer. V4 therefore allows one V3 source-node
wake/stop of at most 16 billable minutes ($0.12864) to copy the frozen inputs
to a fresh V4 node, then caps the V4 node at 69 minutes ($0.55476), each at no more than
$0.00804/min. The combined maximum is $1.3392, below the user-authorized $1.34
total cap. The V4 model command itself is capped at 60 minutes. Stop on a
packet, source, runtime, custody, candidate, or budget mismatch. A timeout or
any incomplete run is retained as INCONCLUSIVE; do not retry this packet.

## Custody and output policy

The V3 node's durable parked disk is the source for the frozen input archive
and source overlay. V4 wakes it once and uses GMan's node-to-node copy to move
the input directory, V3 packet, runtime receipt, and stop receipt to a fresh
node. It stops V3 immediately after transfer and does not edit its files. The
pristine V9 primary outcome manifest is restored in the V4 clone only from
the exact bound archive if V3's known post-error report is present.

Model weights, prompts, token IDs, captures, predictions, row-level outcomes,
and traces stay in external GMan/PrimaryED custody. The V4 exported receipt
contains digests, counts, runtime identity, replication checks, and terminal
statuses only.

## Nonclaims

This is not blind external replication, independent scientific acceptance,
novelty priority, architecture-general validity, mechanistic understanding,
consciousness evidence, alignment evidence, human equivalence, or production
authorization. Reviewer status is \`NOT_SUBMITTED\`.
