# Replacement-loss-v8-scale results: Band A

State slice: `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-ridge-primary-causal`, packet `a93506f349c91bb3efd0cc6738f7bbfd3343f33313d752d61d83fd5f16fbd002`. Aggregate report: `docs/replacement-loss-v8-scale-bandA-report.json`.

## Outcome: `FAIL` (9/9 cases fail the frozen primary gates)

Checkpoints (selected after freeze): facebook/opt-2.7b, openai-community/gpt2-xl, HuggingFaceTB/SmolLM2-1.7B, three seeds each. Thresholds, ridge value and controls unchanged from v3/v8.

| model | seed | dir. acc | median cos (all) | min stratum cos | CE delta (nats/tok) | max logit err | rescue cos | failed gates |
|---|---|---|---|---|---|---|---|---|
| opt-2.7b | 0 | 1.00 | 0.720 | 0.684 | 0.0178 | 0.027 | 0.995 | primary_treatment, stratum_primary |
| opt-2.7b | 1 | 1.00 | 0.693 | 0.652 | 0.0094 | 0.025 | 0.995 | primary_treatment, stratum_primary |
| opt-2.7b | 2 | 1.00 | 0.701 | 0.662 | 0.0206 | 0.022 | 0.995 | primary_treatment, stratum_primary |
| gpt2-xl | 0 | 1.00 | 0.846 | 0.826 | 0.0102 | 0.006 | 0.997 | primary_treatment, stratum_primary |
| gpt2-xl | 1 | 1.00 | 0.845 | 0.830 | 0.0210 | 0.005 | 0.998 | primary_treatment, stratum_primary |
| gpt2-xl | 2 | 1.00 | 0.853 | 0.828 | 0.0170 | 0.005 | 0.998 | primary_treatment, stratum_primary |
| SmolLM2-1.7B | 0 | 0.97 | 0.869 | 0.835 | -0.0096 | 0.065 | 0.991 | primary_treatment, stratum_primary, heldout_behavior |
| SmolLM2-1.7B | 1 | 0.93 | 0.743 | 0.710 | 0.0167 | 0.072 | 0.991 | primary_treatment, stratum_primary, heldout_behavior |
| SmolLM2-1.7B | 2 | 0.98 | 0.833 | 0.827 | 0.0006 | 0.049 | 0.991 | primary_treatment, stratum_primary |

Reading (descriptive, not a new claim): the fitted affine projection preserves held-out behavior in most cases and predicts the direction of causal effects, but the pooled median cosine of predicted vs actual effects falls under the 0.90 gate, and one or more strata fall under the stratum gates. Controls behave as designed (no-op 0.0 direction accuracy). V8's positive result at <=500M parameters therefore did not extend to 1.7B-2.7B under the frozen gates.

## Process disclosures

- First execution stopped at the final report step because the VM had no git metadata (`git rev-parse HEAD` failed) after all predictions and outcomes were sealed. I added git metadata and re-ran only the evaluation over the sealed predictions (no new fixtures, entropy, thresholds or checkpoints). This was a tooling failure, not a measured result, but it is a re-run and is recorded as one.
- A first spot VM was preempted during setup, before any scientific stage; the disk was cloned to an on-demand VM.
- Band B (opt-6.7b, open_llama_7b_v2; no eligible GPT-2-family target) was NOT run: the frozen resource limits (24 GiB peak RSS, 2 h per case) are below what 7B fp32 needs, so a run would be `INCONCLUSIVE` by construction. It needs a new packet with revised limits.
- Same-owner, CPU, synthetic prompts, one projection per checkpoint. No independent review (`EXTERNAL_REVIEW_UNAVAILABLE`).
