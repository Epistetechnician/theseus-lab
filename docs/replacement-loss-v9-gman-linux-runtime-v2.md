# Replacement-loss v9 GMan Linux runtime reproduction v2

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-linux-runtime-reproduction-v2`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Why this is a new packet

Cloud packet v1 stopped before model evaluation. Its exact lock check rejected
the Linux environment: the full installed distribution inventory differed
from the frozen v9 lock, including extra CUDA runtime and Triton packages and
`packaging` 26.0 rather than 26.3. V1 remains `INCONCLUSIVE` and will not be
reopened or rerun.

V2 asks the same cross-runtime question under a separately frozen Linux
runtime identity. It binds the complete observed distribution inventory by
the retained v1 runtime-preflight receipt. It does not normalize package
versions or edit the v9 source, checkpoint, fixtures, candidate rules,
thresholds, causal analysis, or outcome rules.

The versioned wrapper verifies that exact Linux inventory, Python version,
source packet, and wrapper bytes before fitting candidates. In memory, it
applies one predeclared change to the v9 replication receipt's dependency
comparison: the replica dependency digest must match the V2-frozen Linux
inventory. The primary macOS and replica Linux dependency digests are recorded
separately because equality across those runtimes is not the question V2
tests. Every other frozen v9 replication check remains conjunctive and
unchanged.

## Frozen target and procedure

The target remains v9 packet
`9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`, at
source revision `202f459d6d89df9a038d78d58e0af825d287fcd8`. The exact selected
OPT, GPT-2, and Llama-family checkpoints, prior primary candidate state,
visible/adaptation fixtures, held-out challenges, predictions, controls, and
outcomes are the bytes already bound by cloud packet v1 and the v9 primary
manifest. No checkpoint or fixture is selected again.

The replicator independently fits and freezes the replica candidates from
visible/adaptation data before revealing the held-out challenge files. It
then reruns all frozen causal, behavior, candidate-state, prediction,
outcome, and control comparisons. The expected primary v9 scientific
outcome remains `FAIL`; V2 makes no threshold or analysis changes.

The runtime gate requires Python `3.12.13`, CPU execution with one PyTorch
thread, the v1-retained complete distribution map and dependency digest, and
the exact frozen source and wrapper digests. The node is CPU-8 on Linux
x86_64; no GPU or provider inference is used.

## Budget and stopping rule

V2 reuses the stopped private node and the already transferred input bytes.
The combined task spend cap remains `$1.00`; V1 has already charged `$0.2092`,
so V2 may charge at most `$0.7908`. The V2 node-session cap is 65 minutes at
no more than `$0.00804/min`; the frozen model command has a 60-minute cap.
The run starts only after all packet and runtime checks pass. Stop on any
digest mismatch, unexpected runtime change, rate or budget breach, timeout,
or incomplete result. This is one V2 attempt; do not rerun it or alter
thresholds after evaluation.

## Outcomes and claim ceiling

Report V2 as `REPRODUCED_LOCAL` only if every frozen content check passes,
the full Linux runtime matches this packet, and the replicated scientific
outcome remains `FAIL`. Report `DIVERGED` for a measured content or outcome
difference and `INCONCLUSIVE` when runtime, custody, or resource checks stop
before a complete result. Preserve V1's separate `INCONCLUSIVE` result in
all cases.

Claim ceiling:
`LOCAL_CROSS_RUNTIME_REPRODUCTION_OF_FROZEN_V9_FAIL_UNDER_THE_DIGEST_BOUND_LINUX_RUNTIME`.
This same-owner result is not blind or independent review, novelty priority,
architecture-general validity, mechanistic understanding, consciousness,
alignment, or production authorization. Reviewer status remains
`NOT_SUBMITTED`.
