# V33R1 adaptation loss-scale audit

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v33r1-loss-scale-audit-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the frozen source packet.

This post-run read-only diagnostic freezes its code and these rules before
reading adaptation gradients. It changes no training, candidate, map,
prediction, threshold, selection, or terminal outcome. Inputs are the
reproduced V33R1 guide states and exact adaptation/donor/mask first minibatch
of stage one, for three fixed development cases and both student geometries.
Reconstruct each independently initialized block from its original seed.

Measure separate squared-error output, activation, globally normalized effect,
and per-row normalized effect losses at that initialization. Compute Euclidean
gradient norms over the new block's parameters, including the output-plus-
activation anchor gradient. Report effect/anchor and row/global gradient
ratios, squared target-effect norm quantiles, and the row-floor fraction.
No updates, additional scientific seeds, intervention selection, holdout
oracle queries, or fit procedure are allowed. These six adaptation readouts
can establish an initial scale mismatch; they cannot establish why the
completed optimization failed or justify a tuned replacement passing.

Monetary cap $0, one CPU thread, existing locked local runtime. Raw inputs
stay in V33R1 development custody; the source archive and aggregate diagnostic
receipt stay in its separate loss-scale-audit custody. Claim ceiling
`INITIAL_ADAPTATION_GRADIENT_SCALE_ONLY`, assurance `INTERNAL_PASS`, reviewer
`NOT_SUBMITTED`, separate reproduction `NOT_RUN_SEPARATELY`. Original scientific
confirmation remains unopened. A future optimization change needs a fresh
protocol and fresh data; independent science and production remain gated.
