# Replacement-loss-v8-scale-r6 results: natural-text challenge arm

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r6-natural-challenge`, packet `30e1c6c492733bcbca03ec137521a57dd66b8ef2dc5c35c4fd92d21c0f08b0f8` (54 files; identical in all four arms). Reports: `docs/replacement-loss-v8-scale-r6-{NN128,NN256,NN512,T128}-report.json`.

## Arm outcomes (frozen gates)

| arm | adaptation / challenge | outcome | cases passing | peak memory |
|---|---|---|---:|---:|
| NN128 | 128 natural / natural | `FAIL` | 3/9 | 18.7 GiB |
| NN256 | 256 natural / natural | `FAIL` (equivalence diagnostic) | 9/9 | 19.1 GiB |
| NN512 | 512 natural / natural | `FAIL` (equivalence diagnostic) | 9/9 | 24.1 GiB |
| T128 (same-seed control) | 128 template / template | `FAIL` | 0/9 | 18.2 GiB |

Mean pooled median cosine of predicted vs observed logit effect over 3 seeds (frozen case gate 0.90):

| model | NN128 | NN256 | NN512 | T128 |
|---|---:|---:|---:|---:|
| opt-2.7b | 0.904 (0/3) | 1.000 (3/3) | 1.000 (3/3) | 0.995 (0/3) |
| gpt2-xl | 1.000 (3/3) | 1.000 (3/3) | 1.000 (3/3) | 0.997 (0/3) |
| SmolLM2-1.7B | 0.969 (0/3) | 1.000 (3/3) | 1.000 (3/3) | 0.991 (0/3) |

## Readings

- **Natural-text challenge: the replacement predicts effects.** At 256 and 512 natural adaptation prompts all nine cases pass every per-case gate and cluster bootstrap on natural challenges (pooled cosine 1.000 in every model/stratum). Caveat 1 from r5 is closed on behavior.
- **But the arm-level outcome is FAIL, not PASS.** The frozen aggregate also requires the mapping-aligned permutation diagnostic to agree within 1e-4 absolute error. On natural prompts two SmolLM2 cases exceed it in each arm (NN256: 1.2e-4 and 6.1e-4 module-effect error; NN512: 8.5e-4 and 1.5e-4). Per-case and bootstrap gates pass. The diagnostic is an absolute-error numerical check that was calibrated on template activations; natural prompts probably have larger activation scale, so this is plausibly float32 noise rather than a substantive miss. That is a hypothesis, not tested; the preregistered outcome stays `FAIL`, and no gate was relaxed after seeing results.
- **NN128 vs T128 (same seeds):** NN128 passes 3/9 cases (gpt2-xl), T128 passes 0/9, even though T128's pooled cosine (0.99+) is high. T128 failing on pooled cosine 0.99 means a gate other than the cosine gate is binding; the descriptive table does not show which (see report JSON). Natural adaptation is not worse than template at 128 on these seeds, but with different challenge distributions the arms are not a clean like-for-like comparison.
- r5's N128 gpt2-xl-only pass replicates here (NN128 gpt2-xl 3/3, opt/SmolLM 0/3).

## Limits and disclosures

- **Pre-run correction (refreeze):** the first r6 freeze (packet `8205e127…`) crashed all three natural arms at challenge generation because the three model families drew overlapping natural challenge prompts, which v3 rejects as non-disjoint. Fixed by excluding already-issued challenge prompts within a run (test added) and refrozen as `30e1c6c4…`; the first-freeze T128 run (also FAIL, 0/9) is discarded and not reported here. Both freezes are in Git history. No challenge result from the first freeze exists for natural arms.
- NN512 was interrupted once when its node was stopped mid-run (not by a gate) and rerun from scratch under the same packet; its custody was regenerated, so its challenge entropy differs from the interrupted run (which produced no result).
- Same-owner, CPU fp32, same three checkpoints, wikitext-2 only, 3 seeds; intervention directions are still synthetic; challenge prompts are natural but strata labels only select the direction type. No independent review. Not a mechanistic, alignment or production claim.
- Spend: gman does not report cost for these nodes in the CLI output used; not measured. Nodes were reused r5 `cpu-8` nodes; cap was $12.
