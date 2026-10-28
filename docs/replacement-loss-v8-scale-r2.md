# Replacement-loss-v8-scale-r2: signal-to-noise and coverage follow-up

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r2-snr-coverage`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8s2-freeze.json` before any run. v3-v8 and v8-scale remain immutable; the v8-scale Band A `FAIL` (9/9) is the reference and is not retuned.

## Question

Band A failed every case on exactly one gate, `median_cosine >= 0.90`, while direction accuracy and normalized error passed with large slack (curated analysis, `docs/replacement-loss-v8-scale-r2-DRAFT.md`, "Evidence update"). Two untested explanations: (H-SNR) the ordinary and compositional challenge interventions are random directions at 1% of activation RMS, so their logit effect sits near fp32 numerical noise, while the adversarial one-hot spike does not; (H-COV) 128 adaptation prompts under-cover the projection input space at width 2560-10240.

Does raising the intervention magnitude, or the adaptation prompt count, change whether the independently fitted affine replacement passes the unchanged frozen gates?

## Design (everything else pinned from v3/v8/v8-scale)

Same three Band A checkpoints (facebook/opt-2.7b @ 905a4b60..., openai-community/gpt2-xl @ 15ea56de..., HuggingFaceTB/SmolLM2-1.7B @ effd688a...), declared as a deliberate follow-up on the same targets, selected after this freeze. Three fresh seeds `20261501`-`20261503` (new `V8S2` fixture namespace; visible `20261500`). Four arms, each run as its own custody root with the same frozen packet:

| arm | challenge intervention scale | adaptation prompts |
|---|---|---|
| S1 (control) | 1x (0.01 RMS, as before) | 128 |
| S4 | 4x | 128 |
| S16 | 16x | 128 |
| C4 | 1x | 512 |

Thresholds, ridge `1e-6`, controls, cluster bootstrap, resource limits (24 GiB peak RSS, 7200 s per case) are unchanged from v3/v8-scale. The intervention direction code is unchanged; the vector is multiplied by the arm's scale.

## Outcomes and interpretation (preregistered)

Per arm: `PASS` only if all nine cases pass all frozen gates; measured misses are `FAIL`; resource termination, digest mismatch or missing outputs are `INCONCLUSIVE` and are not retried under this packet. Descriptive, not gated: pooled and per-stratum median cosine as a function of scale. S1 re-establishes the baseline on fresh fixtures; an arm is only informative relative to S1.
- If S4/S16 raise median cosine toward or above 0.90 while S1 stays below, the Band A failure is measurement-limited (SNR), and the replacement's causal fidelity at that scale is not what failed. Larger interventions are a different, less local probe; this is stated, not hidden.
- If C4 improves cosine over S1, coverage matters.
- If nothing moves, the failure is not explained by either, and the next hypothesis (layer/ridge) is open.

## Limits and nonclaims

CPU fp32, same-owner, synthetic prompts, one projection per checkpoint, no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`). A pass would not show mechanistic understanding, alignment, whole-model equivalence or production readiness. Overlap checks use regenerated v3/v6/v7/v8/v8-scale visible and adaptation fixtures (earlier challenge text is not rechecked).

## Budget and stop rule

gman CPU nodes (`cpu-8`, 32 GiB), one per arm, in parallel; cap $60. Stop at the first `INCONCLUSIVE` resource failure of an arm and record it.
