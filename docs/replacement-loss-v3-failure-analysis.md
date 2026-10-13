# Replacement-loss-v3 GPT-2 failure analysis

State slice: `theseus-lab-sandbox-v1`.

Assurance: `INCONCLUSIVE` — post-hoc descriptive analysis only.

Claim ceiling: `POSTHOC_EXPLORATORY_ASSOCIATION_ON_V3_GPT2_CASES_ONLY`.

## Frozen analysis and custody

The analysis plan was frozen before inspecting row-level v3 arrays. Plan SHA-256:
`99445ae5889b0b12b0ff704d97ce7b4f424b7709bc778f468993a510bf4f4a4c`.
It binds the v3 packet, primary report, and all nine GPT-2 adaptation, prediction,
and outcome files. The external aggregate report is
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v3/failure-analysis/analysis-report.json`
(SHA-256 `352b67901e5c51e358a4d3c0e0b0022e653a2f62185f580d7407a0e9229d6e3b`).
The analysis script and prompt-bearing arrays remain in external custody.

No prompt text, raw rows, model weights, or row-level diagnostics are stored in
this repository. This report only summarizes those custody-bound aggregates.

## Result

The predeclared continuous-support hypothesis required both higher compositional
input novelty and a positive within-compositional error-to-novelty association
in at least two of three seeds. It did not pass: compositional novelty was higher
in all three seeds, while Spearman correlations were `-0.054`, `-0.140`, and
`0.089`.

There is a stable **class-level association**. Median centered nearest-adaptation
cosine distance for compositional inputs was about `0.458–0.460`, compared with
`0.056–0.060` for ordinary inputs and `0.093–0.102` for adversarial inputs.
Compositional activation error medians were `0.311–0.348`, compared with
`0.066–0.068` ordinary and `0.070–0.073` adversarial. These prompt-level
differences do not establish that adaptation coverage caused the errors.

The candidate-to-original activation norm ratio stayed near one
(`0.986–1.019` medians by compositional seed), while the residual was mostly
orthogonal to the original activation (`0.305–0.342`). This points to a broad
activation-pattern mismatch rather than a simple output-scale error. The
aggregate residual was spread over hundreds of output coordinates; no single
coordinate explained the compositional result.

## Falsifiable follow-up

The follow-up protocol `theseus-replacement-loss-v4-compositional-support`
tests whether compositional examples in an equal-size adaptation set improve
reconstruction on fresh compositional prompts. It compares ordinary-only
adaptation against a paired adaptation arm with the same shared ordinary
examples plus compositional examples. The prompt generator, model, and held-out
challenge data are digest-bound in the v4 packet before new fixtures are
generated.

This follow-up is scoped to the exact GPT-2 checkpoint and synthetic prompt
generator used for v3. It cannot validate the continuous-support hypothesis,
generalize to other GPT-2 checkpoints or architectures, or establish independent
science, alignment, consciousness, or production readiness.
