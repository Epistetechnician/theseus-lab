# V39 original training-direction replay

2026-10-07, revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus source
freeze. Slices `theseus-lab-sandbox-v1`,
`theseus-v39-training-probe-replay-v1`,
`global-experiment-assurance-lifecycle-v1`.

The reproduced first-MLP diagnosis found large local derivative errors.
Before selecting another fitting change, compare saved Jacobians along the
original four fitting directions with four fresh directions. This is an
adaptation-only projection analysis: no model forwards, training or held-out
queries. It reuses the diagnostic's fixed eight prompts per seed, three
positions, two arms and exact saved teacher/student matrices.

Freeze before computing projections. Reconstruct V39's original candidate
seed and pooled row order. Its CPU generator consumes the count-length
coordinate-mask draw, then four Rademacher probe tensors scaled by sqrt(width).
A hermetic test intercepts the actual original core fit to verify this RNG
reconstruction. Fresh directions use a new protocol-bound seed, shared across
arms. Project output-by-input Jacobians with float64 matrix products. This is
not an exact replay of the original floating-point batch JVPs or optimizer
loss; Jacobian orientation/JVP consistency was checked by the parent packet.

Report projected nRMSE, teacher projected RMS, and original gradient coefficient
min/median/max and fraction below 1. `TRAINING_DIRECTION_UNDERFIT_SUPPORTED`
requires all three JVP final-token training-probe nRMSE values >.20; otherwise
`DIRECTION_GENERALIZATION_OR_MIXED`. Neither label identifies optimization,
weighting, capacity or probe count as the unique cause. No score-based retuning.

Verify parent source, diagnostic closeout/raw arrays, selected indices, V39
captures/trace digests and locked runtime. Primary/replica use distinct
external prefixes and identical toolchains. Preserve all 36 projection cells
per role, with 72 teacher/student arrays, in PrimaryED custody; no raw data in
the repository. Reconcile eight fields, then recompute projected nRMSE with
an independent mean-square formula. Shared owner/host and semantics are limits.

Budget: 60 minutes, 6 GiB RSS, 2 GiB artifacts, 5 GiB free reserve per role,
one CPU thread, paid $0. Cooperative checks; energy/full economic cost unknown.
Changed source/custody, nonfinite data, error, cap or mismatch is INCONCLUSIVE;
no same-root retries. Three contract tests precede freeze. End state is both
roles, numeric closeout and literal report; reviewer NOT_SUBMITTED.
Claim ceiling `ADAPTATION_TRAINING_DIRECTION_RESIDUAL_DIAGNOSIS_ONLY`.
V39 remains FAIL. No new retained behavior, fresh held-out/architecture evidence,
novelty, universal identity, consciousness, human alignment, independent
acceptance or production authority. Any next fit needs a new frozen packet
and fresh held-out fixtures; stronger science/production retain separate gates.
