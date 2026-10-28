# Replacement-loss-v8-scale-r2 results

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r2-snr-coverage`, packet `ca3d2143f99f8fe02266503ae9b2d5528dd71805f6d5652ffe0a31aedbf1d673` (identical in all four arms). Aggregate reports: `docs/replacement-loss-v8-scale-r2-{S1,S4,S16,C4}-report.json`.

## Arm outcomes (preregistered rules, unchanged gates)

| arm | change vs control | outcome | notes |
|---|---|---|---|
| S1 | none (control, fresh seeds) | `FAIL` | reproduces Band A: 0/9 cases pass |
| S4 | intervention 4x | `FAIL` | 0/9 cases pass |
| S16 | intervention 16x | `FAIL` | 0/9 cases pass |
| C4 | adaptation prompts 4x (512) | `INCOMPLETE` | **9/9 cases pass every frozen gate**, but arm is INCONCLUSIVE: peak RSS 25.3 GiB exceeded the frozen 24 GiB limit (`process_peak_rss_exceeded_24_GiB`) |

Peak RSS by arm (GiB): S1 18.2, S4 18.2, S16 18.3, C4 25.3. The packet says resource terminations are `INCONCLUSIVE` and are not retried under this packet, so C4 is recorded as INCONCLUSIVE, not PASS, even though `all_treatment_case_gates_pass` is true.

## Pooled median cosine of predicted vs observed logit effect (mean of 3 seeds; frozen gate 0.90)

| model | arm | pooled | adversarial | compositional | ordinary | cases passing |
|---|---|---:|---:|---:|---:|---:|
| opt-2.7b | S1 | 0.700 | 0.997 | 0.665 | 0.670 | 0/3 |
| opt-2.7b | S4 | 0.706 | 0.997 | 0.674 | 0.681 | 0/3 |
| opt-2.7b | S16 | 0.703 | 0.997 | 0.668 | 0.667 | 0/3 |
| opt-2.7b | C4 | 0.993 | 1.000 | 0.991 | 0.991 | 3/3 |
| gpt2-xl | S1 | 0.854 | 0.906 | 0.840 | 0.841 | 0/3 |
| gpt2-xl | S4 | 0.858 | 0.930 | 0.848 | 0.839 | 0/3 |
| gpt2-xl | S16 | 0.856 | 0.930 | 0.841 | 0.840 | 0/3 |
| gpt2-xl | C4 | 0.999 | 0.995 | 0.999 | 0.999 | 3/3 |
| SmolLM2-1.7B | S1 | 0.815 | 0.822 | 0.736 | 0.827 | 0/3 |
| SmolLM2-1.7B | S4 | 0.926 | 0.978 | 0.837 | 0.884 | 0/3 |
| SmolLM2-1.7B | S16 | 0.947 | 0.985 | 0.831 | 0.927 | 0/3 |
| SmolLM2-1.7B | C4 | 0.984 | 0.975 | 0.979 | 0.993 | 3/3 |

## Reading (descriptive; a follow-up packet is needed to confirm)

- **Coverage explains the Band A failure; signal-to-noise does not.** Quadrupling the adaptation prompts moves every model from FAIL to cosine 0.98-0.999 in all strata. Scaling the intervention 4x/16x leaves opt-2.7b (0.70) and gpt2-xl (0.85) essentially unchanged; the SNR hypothesis is not supported for those two.
- SmolLM2-1.7B responds partly to larger interventions (0.82 to 0.93-0.95 pooled; compositional stuck at 0.83-0.84), so a magnitude effect is possible there, but none of S4/S16 passes, and the held-out behavior gate (`max logit error` about 0.063-0.069 vs 0.05) still fails for SmolLM2 in S1/S4/S16; it passes in C4.
- This agrees with the original ridge-at-width intuition: 128 prompts under-determine a projection of width 2560-10240.

## Limits and disclosures

- C4 is INCONCLUSIVE by the preregistered resource rule. The honest summary is "9/9 cases pass but the arm breached its memory limit"; it must not be reported as a PASS. A confirmatory packet with the RSS limit raised before the run (and C4 as primary) is the next step; it would be a new freeze and new fixtures.
- Same three checkpoints as Band A were reused deliberately (declared in the prereg); seeds and fixtures are fresh (`V8S2`). Overlap checks use regenerated visible/adaptation fixtures only.
- S1 ran on a GCP VM (n2-highmem-8) and S4/S16/C4 on gman cpu-8 nodes; compute location is not part of the claim, but no cross-machine reproducibility check was run.
- CPU fp32, synthetic prompts, one projection per checkpoint, same owner, no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`). No mechanistic, alignment or production claim follows.
