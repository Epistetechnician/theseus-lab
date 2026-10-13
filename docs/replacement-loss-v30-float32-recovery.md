# Replacement/loss V30: float32 recovery for the V29 methods packet

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v30-nonlinear-synthetic-recovery-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Recovery scope

V29 stopped before candidate training because generated fixture vectors could
reach a float32 source module as float64. V30 changes one implementation
detail: every sampled adaptation and challenge vector is explicitly cast to
float32 before use. It preserves V29's synthetic source families, seeds,
student architecture, optimizer, fixture counts, challenge generation,
controls, thresholds, resource caps, and claim ceiling. V30 binds V29's frozen
packet and its exact failure receipt. V29 remains `INCONCLUSIVE` and is not
reopened or retried.

## Frozen claim and sequence

The question remains whether a separately initialized GELU MLP trained on
disjoint synthetic adaptation pairs predicts held-out local intervention
effects, retains synthetic downstream behavior, and improves on wrong-module,
random, zero/no-op, and affine controls. There are three source-function
families and three seeds per family. The replacement is trained for 600
full-batch AdamW steps using only adaptation pairs. Predictions are sealed
before the evaluator runs each source on challenge inputs.

Freeze V30's protocol, wrapper, upstream V29 source, and exact dependency
versions before generating V30 oracle weights or fixtures. Keep generated
arrays and all outputs under the new external custody root
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v30`.
Use the second isolated V28-qualified runtime for same-owner reproduction.
The cap remains `$0`, 1,800 seconds, 4 GiB peak RSS, and 1 GiB custody storage.

The maximum claim is
`SYNTHETIC_NONLINEAR_REPLACEMENT_METHODS_QUALIFICATION_ONLY`. This cannot
establish a trained-transformer mechanism, architecture-general validity,
independent science, consciousness, alignment, or production readiness.
