# GPT-2 visible target audit V1

2026-10-06; revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus source
freeze. Slices: `theseus-lab-sandbox-v1`,
`theseus-gpt2-visible-target-audit-v1`,
`global-experiment-assurance-lifecycle-v1`.

Hypothesis: V37R1's energy selector may have missed a causally critical
second-half MLP. Reuse only its exact 64 already visible rows and exact pinned
GPT-2 checkpoint. Freeze this diagnostic before probing every layer. The
parent must be the pre-fit `NO_QUALIFIED_TARGET` stop with no opened challenge.
No new prompt, adaptation set, held-out result, fit, optimizer change or
replacement is authorized under this diagnostic. Parent bytes stay immutable.

Measure actual removal logit nRMSE and top-1 change for all 12 layers with the
unchanged V36 capture, model adapter and zero-removal operation. A layer is
eligible only at the existing .10 nRMSE and .20 top-1-change thresholds. A
visible critical pair exists only if each depth half has an eligible layer;
recommend greatest nRMSE per half, lowest layer index on ties. Otherwise
report no critical pair. Output energy does not override these gates.

Primary and replica run in the already isolated V37R1 runtime pair. Full
baseline/removal logits stay on the artifact drive; public report contains
scalar metrics/digests only. Bind original visible file, checkpoint manifest,
parent result, source and array digests; replica reconstructs all ablations
and reconciles nine named fields. Record distinct runtime prefixes and the
same host/owner. No independent authority follows.

Per role: 1,800 seconds, 1 GiB artifact bytes, $0 paid; existing one-thread
CPU/runtime lock applies. All arrays are inference-only; no gradient or
replacement state is generated. Recheck mounts during execution. Missing or
changed custody, numerical errors, mismatch or budget exhaustion stops a
fresh root; do not retry or repair frozen results. Energy and amortization
unmeasured. Two hermetic selection checks precede the source freeze.

Claim ceiling `REUSED_VISIBLE_TARGET_DIAGNOSIS_ONLY`. `REPRODUCED_LOCAL`
may describe an exact replicated diagnostic, not predictive causal retention.
Reviewer `NOT_SUBMITTED`. An available pair may inform a new, explicitly
reused-checkpoint development packet; that packet still needs disjoint
adaptation, sealed unseen predictions, controls and a separate replica.
Unavailable pairs cannot be filled by weakening thresholds. Three-family
confirmation, novelty, subjective experience, alignment, independent science
and production remain unestablished and separately gated.
