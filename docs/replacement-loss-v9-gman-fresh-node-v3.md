# Replacement-loss v9 fresh-node GMan reproduction v3

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v3`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Purpose and boundary

V1 is `INCONCLUSIVE`: its frozen dependency check stopped before model
evaluation. V2 is also `INCONCLUSIVE`: GMan's worker died during restore
before transfer or preflight. V2's postmortem source audit also found that
its frozen wrapper expected a top-level `primary` manifest field that its
freeze did not contain. Neither packet will be reopened.

V3 is a fresh, separately frozen same-owner reproduction of the existing v9
target. It uses a newly created GMan node owned by the V3 mission and the
exact v9 input archive, source overlay, primary results, and prior fixtures
already in external custody. It does not select checkpoints again or change
the scientific method, candidate rules, thresholds, outcomes, or controls.
Its wrapper binds the missing primary-manifest field and adapts the V2
runtime wrapper under a digest-bound V3 packet, then binds result receipts to
the V3 packet digest. A digest-bound bootstrap creates a fresh Python 3.12.13
environment from the exact 51-package Linux inventory retained in the V1
runtime receipt. It installs only those pinned distributions, without
resolving additional dependencies, and records an exact runtime preflight
before model evaluation.

Before `run_replica` starts, the wrapper verifies the V3 freeze, both wrapper
digests, the source and v9 packet digests, archive and input-manifest
digests, primary outcome digest, prior-fixture set, and the complete runtime
receipt. The new node must match the retained Linux runtime inventory,
including Python `3.12.13`, one PyTorch CPU thread, and dependency digest
`0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4`. Any
mismatch stops the run before candidate fitting or held-out evaluation.

## Frozen target and procedure

The target is the existing v9 packet
`9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`, at
source revision `202f459d6d89df9a038d78d58e0af825d287fcd8`. The already
selected OPT, GPT-2, and Llama-family checkpoints, primary candidates,
visible/adaptation fixtures, held-out challenges, predictions, controls,
and outcome manifests are retained unchanged. The expected primary outcome
remains `FAIL`.

The replicator fits replica candidates using only the frozen visible and
adaptation data, freezes them, and evaluates the already frozen held-out
causal, behavior, activation-rescue, compositional, adversarial, and control
checks. No prompts, raw model outputs, weights, or traces are emitted to
GMan command output or copied into this repository.

## Budget and stop conditions

The combined spend cap is `$1.00`; V1's recorded charge is `$0.2092`, so the
remaining cap is `$0.7908`. The V3 node is capped by protocol at 85 minutes
and `$0.00804/min` (at most `$0.6834`), leaving at least `$0.1074` headroom.
The model command has a 60-minute wall-clock timeout. Stop on any digest,
runtime, custody, budget, or timeout mismatch. This packet permits one model
evaluation attempt; do not rerun it or change thresholds after evaluation.

## Outcomes and claim ceiling

Report `REPRODUCED_LOCAL` only if all content and runtime checks pass, the
replicator completes every frozen comparison, and the reproduced primary
outcome remains `FAIL`. Report `DIVERGED` for a measured content or outcome
difference and `INCONCLUSIVE` if a gate or resource limit stops the run
before completion. V1 and V2 remain separate `INCONCLUSIVE` attempts.

Claim ceiling:
`LOCAL_CROSS_RUNTIME_REPRODUCTION_OF_FROZEN_V9_FAIL_UNDER_THE_DIGEST_BOUND_LINUX_RUNTIME`.
This is same-owner local evidence, not blind or independent review, novelty
priority, architecture-general validity, mechanistic understanding,
consciousness, alignment, or production authorization. Reviewer status is
`NOT_SUBMITTED`.
