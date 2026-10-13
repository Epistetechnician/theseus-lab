# Replacement/loss V29: nonlinear replacement methods qualification

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v29-nonlinear-synthetic-qualification-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Narrow question and claim ceiling

Can a newly initialized GELU MLP, trained only on disjoint synthetic
adaptation pairs, predict held-out local intervention effects for three
different known nonlinear modules? Does it retain the module's downstream
synthetic behavior and improve on wrong-module, random, no-op, and affine
distillation controls?

This is a methods qualification on synthetic functions. It is not a model
checkpoint replication, evidence about a trained transformer, independent
mechanistic discovery, or a proof of Theseus-style continuity.

## Frozen design

- Three source families: two-layer ReLU, two-layer GELU, and SiLU-gated
  two-branch modules; three fixed seeds per family (nine cases).
- Each case has 3,072 adaptation vectors, split evenly across ordinary,
  compositional, and high-norm adversarially shifted strata. A fresh student is a
  16→48 GELU→16 MLP initialized independently and fit by full-batch AdamW for
  600 steps, learning rate `0.01`, weight decay `0.0001`.
- The fixed downstream readout is generated from a separate seed. Challenge
  vectors and interventions use custodian-generated entropy only after all
  nine candidates and their hashes are frozen. The challenge generator does
  not query the source functions. There are 128 rows per stratum
  per case. Challenge inputs, intervention vectors, target weights, and
  outputs remain in external custody.
- Predictions for candidate activation, downstream logits, and intervention
  effects are written and digest-sealed before the oracle is evaluated on any
  challenge input. The evaluator then records oracle outputs and scores the
  sealed predictions.
- Controls are wrong-module replacement, untrained random student, zero/no-op,
  and closed-form affine distillation. No coordinate-permutation experiment is
  part of this packet.

## Predeclared gates

Per case and per challenge stratum, the learned replacement must achieve
causal direction accuracy ≥0.80, median cosine ≥0.90, and median normalized
effect error ≤0.20 on at least 100 resolvable effects. Activation rescue must achieve
median cosine ≥0.90 and median normalized activation error ≤0.20. Downstream
behavior must have mean cross-entropy increase ≤0.05 nats and maximum paired
normalized logit error ≤0.05. It must have lower median causal normalized
error than each of the four controls. All nine cases must pass for the packet
to pass. These thresholds are fixed before synthetic oracle generation.

The run budget is `$0`, 1,800 seconds total, 4 GiB peak resident memory, and
1 GiB of external artifact storage. The runner records wall time and peak RSS;
any budget overrun yields `INCONCLUSIVE`.

## Reproduction, budget, and claim boundary

The same-owner replicator uses the frozen adaptation and challenge fixtures in
a second isolated runtime, retrains candidates from their initial seeds, and
must match candidate, prediction, and outcome digests. This can establish
`REPRODUCED_LOCAL`, not `INDEPENDENT_ACCEPT`. Monetary budget is `$0`; no Hub
queries, model weights, paid compute, remote workers, human data, or production
traffic are used. Stop on any digest mismatch, missing stratum, incomplete
prediction seal, non-finite metric, or resource cap. The maximum claim is
`SYNTHETIC_NONLINEAR_REPLACEMENT_METHODS_QUALIFICATION_ONLY`.

## Frozen execution order

1. Freeze this protocol, runner, and exact NumPy/PyTorch versions.
2. Generate synthetic source functions and adaptation fixtures; train all
   candidates and freeze their file digests.
3. Generate challenge fixtures after candidate freeze.
4. Seal all candidate predictions before opening oracle challenge outcomes.
5. Score the predeclared gates and retain every failure.
6. Rerun in the clean second runtime from frozen adaptation and challenge
   fixtures; report literal reproduction and assurance statuses.
