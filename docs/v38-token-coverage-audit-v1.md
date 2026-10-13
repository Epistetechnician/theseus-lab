# V38 adaptation token-coverage audit V1

2026-10-06, revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus frozen
bytes. State slices `theseus-lab-sandbox-v1`,
`theseus-v38-token-coverage-audit-v1`, `global-experiment-assurance-lifecycle-v1`.

Hypothesis: V38's independently fitted replacements have substantially larger
activation errors at unsupervised prefix positions than at supervised final
positions. Predeclare a twofold gap: prefix nRMSE >= 2 * final nRMSE in all
six JVP seed/stage cells supports the hypothesis; otherwise record mixed or
unsupported evidence. No inference that this explains every causal/behavior
miss or that full-token fitting will pass.

Freeze source and the exact failed V38 parent before measurement. Reconstruct
all twelve fitted stages from sealed V38 weights. Use only the 128 retained
adaptation prompts for each seed, replaying the earlier replacement context
for stage two. Full-token input captures must reproduce the stored final-token
input arrays exactly. Compare student and teacher MLP output on the same input;
aggregate squared errors/teacher energies separately over prefix and final
positions, weighting every coordinate equally. Also record teacher variance
across all tokens versus the existing fitted final-token output scale.

No fit, new corpus, held-out prompt, prediction, oracle array, threshold change,
model selection, provider call, paid worker or publication. Parent receipt and
guide tree are revalidated; no old source/result is edited. Two isolated
V37R1 runtime prefixes are reused in fresh processes. Replica recomputes all
statistics and seven reconciliation fields with full toolchain fingerprints.
Raw per-row sufficient statistics stay external; repository receives scalars
and digests. These are paired within one reused checkpoint, not independent
guide samples or authority. Reviewer `NOT_SUBMITTED`.

Per role: 1,800 seconds, existing 6 GiB RSS envelope, at most 64 MiB diagnostic
artifacts, paid cap $0. Check custody throughout; missing/mismatched input,
nonfinite statistics, replay mismatch or resource failure stops a fresh root.
No retry or repair of terminal roots. Energy/amortization unmeasured. Two
hermetic statistics checks precede freeze. `REPRODUCED_LOCAL` requires exact
second-runtime results, not merely successful unit checks.

Claim ceiling `ADAPTATION_TOKEN_COVERAGE_DIAGNOSIS_ONLY`. Next authorized local
step is a separately frozen coverage intervention on fitting if supported,
with fresh disjoint adaptation and sealed held-out predictions. Existing V38
failure and its held-outs remain immutable. Three-family confirmation, novelty,
identity, consciousness, human alignment, independent science and production
remain separate obligations.
