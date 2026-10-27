# Replacement-loss-v8-scale-r4 results: ridge ladder at 128 prompts

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r4-ridge-doubledescent`, packet `521689af884fc7e73e6516b869fd6e92531dca4ce8f1e97699d0baa2c7e05294` (identical in all four arms). Aggregate reports: `docs/replacement-loss-v8-scale-r4-{R6,R4,R2,R0}-report.json`.

## Outcome: all four arms `FAIL` (0/9 cases each); no ridge in the ladder rescues the 128-prompt fit

Ridge = alpha x (mean Gram eigenvalue of the bias-centred adaptation inputs). Mean pooled median cosine of predicted vs observed logit effect over 3 seeds (frozen gate 0.90):

| model | arm | alpha | pooled | adversarial | compositional | ordinary | cases passing |
|---|---|---:|---:|---:|---:|---:|---:|
| opt-2.7b | R6 | 1e-06 | 0.694 | 0.997 | 0.660 | 0.671 | 0/3 |
| opt-2.7b | R4 | 0.0001 | 0.687 | 0.996 | 0.651 | 0.661 | 0/3 |
| opt-2.7b | R2 | 0.01 | 0.617 | 0.976 | 0.565 | 0.567 | 0/3 |
| opt-2.7b | R0 | 1 | 0.429 | 0.814 | 0.340 | 0.359 | 0/3 |
| gpt2-xl | R6 | 1e-06 | 0.856 | 0.908 | 0.840 | 0.840 | 0/3 |
| gpt2-xl | R4 | 0.0001 | 0.851 | 0.909 | 0.836 | 0.827 | 0/3 |
| gpt2-xl | R2 | 0.01 | 0.740 | 0.832 | 0.714 | 0.716 | 0/3 |
| gpt2-xl | R0 | 1 | 0.410 | 0.405 | 0.420 | 0.413 | 0/3 |
| SmolLM2-1.7B | R6 | 1e-06 | 0.834 | 0.883 | 0.780 | 0.837 | 0/3 |
| SmolLM2-1.7B | R4 | 0.0001 | 0.690 | 0.804 | 0.534 | 0.532 | 0/3 |
| SmolLM2-1.7B | R2 | 0.01 | 0.231 | 0.440 | 0.041 | 0.146 | 0/3 |
| SmolLM2-1.7B | R0 | 1 | 0.018 | 0.061 | 0.009 | 0.018 | 0/3 |

## Reading

- **H-DD (variance near the interpolation threshold, rescued by more ridge) is not supported on this grid.** Cosine is flat or lower at alpha 1e-4, clearly lower at 1e-2, and collapses at 1 (over-shrinkage; SmolLM2-1.7B reaches 0.02). No alpha improves any model over the smallest ridge.
- **The in-packet control R6 (alpha 1e-6) reproduces the earlier 128-prompt baselines** (opt-2.7b 0.69, gpt2-xl 0.86, SmolLM2-1.7B 0.83), so the ladder is anchored.
- Together with r2 and r3 (more adaptation prompts fixes the failure; larger interventions do not), this points to **missing information in 128 prompts rather than estimator variance that regularization can remove**. That is a leaning, not a proof: the grid is coarse (two decades apart), the ridge was parametrized relative to the *mean* eigenvalue only, and a ridge relative to the spectrum's tail, a different estimator, or fitting on tokens rather than prompts were not tested.
- **Consequence for the registered Band B prediction:** the prediction (about 400 prompts for Llama-7B, about 600 for OPT-6.7B; failure at 256, pass at 1024) was derived from a rows-per-width rule and was written before any Band B run. The rule remains a descriptive fit on three families; this result removes the double-descent mechanism that motivated it but does not test the prediction. The prediction stays registered and untested.

## Limits

CPU fp32, same owner, synthetic template prompts (a natural-text arm has not been run), one projection per checkpoint, same three checkpoints as earlier packets, no independent review. Memory peak was under 24 GiB in all arms (R0 18.6 GiB). No mechanistic, alignment or production claim.
