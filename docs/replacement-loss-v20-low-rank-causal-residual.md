# Replacement-loss V20: low-rank causal residual

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v20-low-rank-causal-residual-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol IDs: `theseus-replacement-loss-v20-development` and
`theseus-replacement-loss-v20-confirmation`.

## Question and claim ceiling

Does a low-rank residual, trained on disjoint finite-intervention examples and
anchored to a separately trained pointwise replacement, predict new causal
effects while retaining pointwise behavior on a previously used OPT/ReLU
checkpoint? This is a method qualification on one reused checkpoint and one
selected block. It is not a test across architectures or fresh checkpoints.

Maximum claim ceiling:
`LOCAL_LOW_RANK_CAUSAL_RESIDUAL_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
The study does not establish identity, consciousness, alignment, production
readiness, independent science, or architecture-general validity. The
activation restoration control is an oracle pipeline control, not learned
activation rescue.

## Frozen method

Use the archived `grushaprasad/opt125m-rootp-surprisal-curriculum` checkpoint
at revision `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`, block 8; block 9 is the
wrong-layer control. Checkpoint tree and weight digests are inherited from the
V16 frozen identity. Do not select or download model weights.

For each of three fresh development adaptation seeds (`202610801`–`202610803`),
generate 64 synthetic prompts in each of ordinary, compositional, and
adversarial strata. Split each stratum into 48 fit prompts and 16
development-validation prompts before capture. Fit and validation intervention
directions have norm `0.01 * RMS(x)` and are generated independently inside
their prompt partitions. Only fit prompts and fit effects enter gradient
updates. Validation prompts and effects are used only to score candidates.
Fixture generation must be disjoint from the V16 visible data and all
adaptation partitions through V19; confirmation fixtures must also be disjoint
from development data.

1. Train a fresh whole-MLP pointwise base on fit inputs and teacher outputs for
   1,000 AdamW updates, batch size 128, learning rate `1e-3`, weight decay
   `1e-4`, gradient clipping `1.0`, and a deterministic seed-bound draw stream.
   No source-model parameters are copied. Freeze the resulting dense base.
2. Attach rank-16 residual factors to both affine projections:
   `fc1(x) = W1 x + b1 + B1(A1 x)` and
   `fc2(h) = W2 h + b2 + B2(A2 h)`. Initialize `A` factors from the frozen
   deterministic seed and initialize `B` factors to zero, so the residual
   begins exactly at the frozen pointwise function. Keep the base matrices and
   biases frozen.
3. Train only the residual factors for 1,000 AdamW updates with the same
   optimizer settings and fit-only batches. The single frozen objective is
   normalized finite-effect MSE plus `1.0` times normalized squared drift from
   the frozen base output on the same fit inputs. Both normalizers are the
   fit-set mean square of their corresponding teacher/base outputs, floored
   at `1e-12`. There is no loss-weight sweep, early stopping, or candidate
   selection by validation. Merge the trained factors into a dense candidate
   only after training, and verify the merged module matches the factorized
   module on a fixed deterministic input probe.
4. Use the frozen result from step 1 as the paired pointwise comparator; do not
   retrain or select another base. The method under evaluation is that same
   frozen base plus its trained low-rank residual.

## Predeclared qualification and controls

Score prompt-level mean metrics separately for each seed and stratum, giving
nine cells. A development candidate qualifies only if every cell passes all
four gates:

- finite-effect MSE divided by the same-row zero-effect prediction MSE is
  `<= 0.90`;
- candidate pointwise value MSE divided by the paired pointwise candidate's
  value MSE is `<= 1.25`;
- continuation cross-entropy increase is `<= 0.05` nats/token; and
- median normalized next-logit error is `<= 0.05`.

The fixed controls are the paired pointwise base, same-row zero-effect
predictor, wrong-layer replacement, random untrained replacement, and no-op.
The confirmation stage uses a separately trained wrong-layer residual and
the V17 held-out scorer for causal-effect prediction, module-output retention,
behavior, controls, and oracle restoration. Held-out activation restoration is
not interpreted as learned rescue.

If any development cell fails, record
`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, retain the failed receipts, and stop
without a confirmation packet or challenge entropy. If development passes,
freeze a confirmation packet before creating its fresh adaptation fixtures.
Repeat the same nine-cell qualification on confirmation adaptation data in
primary and replica. Draw 32-byte challenge entropy only after both frozen
confirmation preparations reproduce byte-for-byte and all nine confirmation
cells pass. The proposer binds predictions before held-out challenge scoring.

## Runtime, custody, and stopping rules

Use the pinned V16/V17 Python 3.12.13 and package inventory, deterministic CPU,
and one Torch thread. Cap each process at four hours and 16 GiB RSS. Monetary
cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`. No provider, cloud, GMan,
paid compute, public write, or production integration is in scope.

Prompts, continuations, intervention directions and targets, captures,
candidates, challenge entropy, predictions, and row-level metrics stay under
external owner custody at `/Volumes/PrimaryED/ResearchArtifacts`. Repository
results contain only aggregate metrics, status, and digests. Never overwrite
an existing custody root or terminal receipt. Any missing dependency, digest
mismatch, cap exceedance, partial fixture, or non-identical replay stops the
stage as `INCONCLUSIVE`; do not redraw data or entropy in place.

Expected assurance ceiling is `REPRODUCED_LOCAL` for exact separate-runtime
replay. Reviewer status is `NOT_SUBMITTED`; the replay is same-owner and is not
blind or independent. Even a positive result does not establish behavior on
another checkpoint or architecture, subjective experience, human equivalence,
alignment, or production authorization.
