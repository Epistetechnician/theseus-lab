# V39 training-direction replay results

2026-10-07 (client date), revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus
frozen source. Slices `theseus-lab-sandbox-v1`,
`theseus-v39-training-probe-replay-v1`,
`global-experiment-assurance-lifecycle-v1`.

**TRAINING_DIRECTION_UNDERFIT_SUPPORTED, REPRODUCED_LOCAL.** Both external
runtime prefixes reproduce all projected vectors, scores, trace summaries and
the decision. Independent mean-square closeout agrees; reviewer NOT_SUBMITTED.
V39 remains FAIL. This is a residual-fitting diagnosis, not a new retention pass.

## Findings

At the first replacement's sampled final-token adaptation inputs, JVP projected
nRMSE is **.83288–.88652 along the original training directions**, versus
**.87174–.89743 along fresh directions**. The reconstructed directions match the
original core fitter's RNG consumption in the contract test. Saved Jacobians
are projected with float64 products; these are not exact original floating-
point batch targets or a replay of the complete optimizer loss. Endpoint-arm
probes are a paired reference; that arm did not use derivative supervision.
The underfit decision applies to JVP.

Training-direction errors are also large at first positions (.66602–.70422)
and middle prefixes (.88540–.92106). Large error exists along the directions
already supplied to the objective at these sampled points. The measurements
support improving derivative fitting before making an unseen-direction or
broader architecture claim. They do not identify capacity, optimization budget,
learning rate or loss normalization as the unique cause.

The auxiliary gradient-cap coefficient has median **1 in all three JVP seeds**.
It is below 1 during .11816–.14746 of updates. The cap is inactive during most
updates, so these traces alone do not justify treating a higher cap as a repair.
Early capped updates could still affect the trajectory; gradient angles and
counterfactual changes were not measured.

## Completion and boundaries

Three contract tests pass in each prefix, including interception of the actual
original core fitter's probe tensor. Replayed 36 projection cells per role,
with 72 teacher/student arrays, from fixed adaptation points. Eight fields
reconcile exactly; numeric closeout verifies digests, shapes and formulas.
Zero model forwards, training steps or held-out queries. Paid **$0**, cap $0;
energy and full economic cost unmeasured.

All raw projections, Jacobians, traces and captures remain on PrimaryED;
runtimes on Backup ED. Shared owner, physical host and implementations limit
inference. No novelty, universal identity, consciousness, human alignment,
independent acceptance or production claim.

Next: a new first-MLP fitting pilot that separates longer optimization from
larger student capacity, with a fixed objective, fresh adaptation/held-out
fixtures and unchanged causal, behavior and rescue gates. Each factor needs its
own declared comparison; preserve V38/V39 and both diagnostics. A broader
three-family test remains a later obligation.

Claim ceiling `ADAPTATION_TRAINING_DIRECTION_RESIDUAL_DIAGNOSIS_ONLY`.
Stronger independent science and production retain their separate authority
gates. Packet `952f46661f9eacf97ebddbef4a9827a94e38960f2bed0cbbbf14f6b67409d76e`; closeout
`8d4b08720d5209b203075ef72e8658b586086caea1beb7f6a6ca33521846ae5b`. Full scalar ranges and exact bindings:
[v39-training-probe-replay-v1-results.json](v39-training-probe-replay-v1-results.json).
