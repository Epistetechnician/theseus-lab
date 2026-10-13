# Replacement-loss V21 post-result fit diagnostic

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v21-paired-endpoint-supervision-v1`,
`theseus-replacement-loss-v21b-run-recovery-v1`, and
`theseus-replacement-loss-v21-fit-diagnostic-v1`.

## Scope and binding

This post-result diagnostic reads only aggregate fields in the frozen V21
primary and replica candidate manifests. It does not open prompt text,
captures, predictions, or challenge data, and does not change the V21 packet,
thresholds, or outcome.

- V21 development packet SHA-256:
  `8b95876077c15779b3c215872316e68a4fc52094ed23d0492672fbd3d4d6b7bd`.
- V21 candidate-set SHA-256:
  `1b85fec821551acf1e5bcea057bc3e2d1a98fea1b61c7e553a5e3f7b7acc4889`.
- Scientific result SHA-256:
  `a8af01248ac55b71e9cc00fdf813fa08cf333b02b15ee3019180fc78d3865f42`.
- Replay report SHA-256:
  `389095013138fd7b8ef357a2d6511fca6b1c0bba40a64ea5def5a6947716be85`.

## Findings

For each training seed, the receipt reports the endpoint-supervised
replacement's finite-effect MSE divided by the same-row zero-effect MSE:

| Seed index | Fit effect MSE ratio to zero predictor | Factual endpoint MSE / endpoint energy | Counterfactual endpoint MSE / endpoint energy | Factor merge max error |
|---:|---:|---:|---:|---:|
| 0 | 1.0320 | 0.009780 | 0.009780 | 8.34e-7 |
| 1 | 1.0312 | 0.010032 | 0.010032 | 7.15e-7 |
| 2 | 1.0224 | 0.009568 | 0.009569 | 7.15e-7 |

The equal-seed mean fit effect ratio is `1.0285`. Every seed is worse than
the zero-effect predictor even on its own adaptation fit rows. The endpoint
residual therefore did not learn a useful intervention response under this
objective; the result is not only a held-out generalization miss.

At the same time, validation pointwise-value ratios were `0.718–0.812` and
all behavior gates passed. The validation effect ratios were `0.983–1.068`,
so only one of nine cells even beat the zero-effect baseline, and none met
the frozen `<=0.90` qualification gate. The primary and replica candidate,
fixture, split, capture, and selection artifacts were byte-identical.

The small factorized-to-dense merge errors rule out a material error from
merging the trained low-rank factors. Endpoint loss is normalized to endpoint
output energy, while causal-effect error is measured against a separate
zero-effect baseline. A low normalized endpoint loss therefore does not
guarantee a low relative error on the paired difference, consistent with the
conditional bound in
[`causal-effect-identifiability-boundary-v1`](causal-effect-identifiability-boundary-v1.md).
This diagnostic does not establish the cause of the remaining effect error.

## Disposition

The frozen scientific outcome remains `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`;
assurance remains `REPRODUCED_LOCAL`; reviewer status is `NOT_SUBMITTED`;
challenge entropy and confirmation fixtures were not created. This diagnostic
does not increase the claim ceiling beyond
`LOCAL_PAIRED_ENDPOINT_SUPERVISION_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.

The next development candidate must first demonstrate, on fit-only receipts,
that it beats the zero-effect baseline for causal effects while preserving
endpoint and behavior gates. Held-out challenges must remain closed until a
new frozen packet's predeclared qualification passes in primary and replica.
Further scalar loss-weight or rank sweeps are not supported by these results.
