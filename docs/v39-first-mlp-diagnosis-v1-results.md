# V39 first-MLP diagnosis results

2026-10-07 (client date). Revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus
frozen source. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-causal-diagnosis-v1`,
`global-experiment-assurance-lifecycle-v1`.

**LOCAL_DERIVATIVE_MISMATCH_SUPPORTED, REPRODUCED_LOCAL.** The primary,
second isolated runtime and separate NumPy closeout agree. V39 remains FAIL;
this is an adaptation-only diagnosis, not a new retained-behavior success.
Reviewer NOT_SUBMITTED; same owner and physical host.

## Measured bottleneck

Across the three JVP seeds at sampled final tokens:

- Factual module-value nRMSE: .10145–.11538.
- Full local Jacobian nRMSE: **.88344–.88909**.
- Median local finite-patch relative error: **.88595–.89366**.
- Median finite-effect cosine: .45796–.46930; resolved fraction .984375–1.
- Actual decoder-effect median error: .93352–.98780.
- Student deltas anchored at the teacher factual output, using teacher prefix
  context: median decoder-effect error .90404–.91809.

The fixed decoder error accounting assigns .88413–.93823 of the error
projection to local delta transport; prefix-context shares are .00974–.06921,
with factual-anchor shares .00049–.08634. These are ordered, path-dependent
projections. They are not causal percentages, additive norm fractions or a
proof that changing one loss would repair the model. Errors remain large with
teacher prefix and teacher factual anchors.

The endpoint arm has final Jacobian error 1.15039–1.19755 and finite-patch
median error 1.12816–1.16142. JVP performs better on these measurements,
but neither supplies qualifying causal retention. All first-position finite
responses remain unresolved because those donor inputs coincide. Their
Jacobians still differ; unresolved effects are not counted as successes.

## What completed

Reused the exact V39 guide and six frozen first-stage students, without fitting
or changing their parameters. Three seeds, eight adaptation prompts per seed,
three token positions, paired 1/2/4/8-coordinate patches at scales 1 and 1.4.
Each role produced 18 local cells, 6 transport cells, **288 full Jacobians,
1,152 local delta pairs and 384 decoder pairs**. Four independent JVP checks
verified every Jacobian. All 48 actual no-op reruns match exactly. Analytic
zero-delta controls return relative error 1 on resolved references.

Nine engineering tests pass in each prefix. Eight reconciliation fields match
exactly; raw-array digest is identical. Separate persisted-array formulas
verify scalar measurements and decisions. V39's original source and terminal
results are unchanged. No prior row-level held-out fixtures, predictions or
oracle arrays were opened; the parent terminal scalar result was a provenance
and motivation input.

Paid execution **$0**, cap $0; zero new training steps and held-out queries.
All prompts, weights, Jacobians, deltas and decoder vectors remain on PrimaryED;
runtimes remain on Backup ED. Per-role resources are recorded in the JSON
receipt. Energy and total economic cost remain unmeasured. Shared owner, host,
selection and instrumentation limit the inference; this is not independent
review or architecture-held-out validation.

## Next local obligation

Before choosing a new fitting objective or increasing capacity, reconstruct
the original four training-probe directions and compare their projected errors
with fresh directions. This distinguishes residual training-target error from
an unseen-direction problem at these sampled adaptation points. It does not
itself establish why training failed. Any new fitting test needs its own
frozen packet and fresh held-out fixtures.

Claim ceiling `ADAPTATION_FIRST_MLP_LOCAL_AND_CONTEXT_ERROR_DIAGNOSIS_ONLY`.
Novelty, full mapping, universal identity, consciousness, human alignment and
production remain unestablished. Stronger scientific acceptance requires the
configured independent authority; production has a separate operational gate.

Packet `4636596602a3c2f7cf056d53f2d8f6fe22893c245d6e89a85837558620ddeba8`; closeout
`4dc4c04006b0a9740f5d17bc29f2722b53e2cc372a832fa060a2ba49035cccf1`. Exact bindings and scalar ranges:
[v39-first-mlp-diagnosis-v1-results.json](v39-first-mlp-diagnosis-v1-results.json).

