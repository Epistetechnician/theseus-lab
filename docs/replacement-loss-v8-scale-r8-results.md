# Replacement-loss-v8-scale-r8 results: float64 downstream check of open_llama_7b_v2

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r8-float64-downstream-logit-cosine-diagnostic`, freeze packet `d89d4cdb6bbc5f51e3a7e3fe62ffd3a33c90301ca6e9e1a1b3d23c531b62e85d` (75 files). Preregistration (committed first): `docs/replacement-loss-v8-scale-r8.md`. Raw report: `docs/replacement-loss-v8-scale-r8-report.json`.

## Outcome: `H1_SUPPORTED_ROUNDING_FLOOR` (diagnostic only)

**The Band B outcomes are unchanged:** open_llama_7b_v2 `FAIL` at N256 and N512; opt-6.7b `FAIL` at N256, `PASS` at N512; arm-level N256 and N512 `FAIL`. No gate or frozen constant was touched; the gate numbers remain the float32 ones. r8 is not a PASS of anything.

Validity gate held: the float32 arm reproduced the recorded Band B N512 per-stratum median cosines exactly (max gap 0.0 over 9 cells; same custody, same GPU, same code path).

## Median logit-effect cosine, N512 custody, 20 prompts per stratum, all 20/20 resolvable, direction accuracy 1.00

| seed | stratum | Band B / r8 float32 | r8 float64 |
|---|---|---:|---:|
| 0 | compositional | 0.8326 | 0.9973 |
| 1 | compositional | 0.8535 | 0.9962 |
| 2 | compositional | 0.8029 | 0.9985 |
| 0 | ordinary | 0.9116 | 0.9984 |
| 1 | ordinary | 0.9335 | 0.9989 |
| 2 | ordinary | 0.9168 | 0.9988 |
| 0-2 | adversarial | 0.978 / 0.980 / 0.987 | 1.0000 (all) |

All six primary cells are >= 0.99, so the preregistered H1 rule fired (H3 rule, all < 0.90, did not). Module-boundary cosine in float64 is 0.9999999999994 (float32: 0.99996-0.99997). float64 median normalized logit error is 3.4e-8 to 7.0e-8.

## Effect size versus float32 resolution (medians, float64 effects)

| stratum | effect norm / logit norm | in float32 eps (1.19e-7) | fp32-vs-fp64 noise ratio, observed effect | same, predicted effect | cos(fp32, fp64) observed |
|---|---:|---:|---:|---:|---:|
| compositional | 2.2e-7 - 2.7e-7 | 2 | 1.8 - 2.0 | 1.7 - 2.2 | 0.45 - 0.57 |
| ordinary | 3.3e-7 - 4.4e-7 | 3 - 4 | 1.5 | 1.5 - 1.7 | 0.49 - 0.63 |
| adversarial | 1.3e-6 - 1.7e-6 | 11 - 14 | 0.45 - 0.51 | 0.42 - 0.50 | 0.89 - 0.92 |

The compositional and ordinary logit effects are only 2-4 float32 epsilons of the logit norm; the float32 effect vectors differ from the float64 ones by more than their own size (noise ratio > 1, cosine to the truth about 0.5). The adversarial stratum has ~10x larger effects, the lower noise ratio, and was already passing. Preregistered secondary expectations (H1: large noise ratio, 10^2-10^4 eps) held on noise ratio; relative size was *smaller* than the predicted 10^2-10^4 eps range (2-14 eps), a stronger version of the same pattern.

## Reading and caveats

- Note the float32 cosines (0.80-0.93) are well above what independent noise of ratio ~1.8 would give (about 0.5): predicted and observed effects share most of their rounding path (same weights, same delta, same activations through layers 0-6), so errors are correlated. The float32 cosine is therefore a noisy but not pure-noise measurement; it is the float64 result, not this argument, that carries the conclusion.
- Supports "the Band B per-stratum cosine miss for open_llama_7b_v2 at layer 7 is float32 measurement resolution, not replacement quality" on these weights and prompts. It rejects H3 as the explanation of the miss. It does not test H2 (depth): float64 removes the miss without any layer change, so depth is not needed to explain it.
- It does not change the gate. Whether a float32 instrument is the right gate for effects of a few epsilons is a separate question that would need its own preregistration (e.g. a float64 downstream gate); nothing here waives or reclassifies Band B.
- Scope: one checkpoint, layer 7, N512 fits, 3 seeds x 60 prompts, one H100, same owner, no independent review. The treatment modules are the unchanged float32 fits cast exactly to float64; only the forward precision changed. The same reasoning was not applied to opt-6.7b N256 (a coverage failure, boundary cosine 0.88), which this result does not address.

## Deviations and disclosures

- **Pre-run refreeze:** the first freeze (`a9ebe11`) shipped a test expecting 78 chain files while the chain is 75 (a failed `sed` left the test unfixed). Found before any run; refrozen at `d81ec64`. Both in Git history; the preregistration commit `c06e125` precedes all code.
- **Execution:** the repository at HEAD was shipped as a tarball to `~/repo_r8` on the existing `theseus-bandb-v8sb` node (verify + tests passed there), then `diagnose` run against `~/custody/N512` read-only. The node's own pinned venv was used (torch 2.6.0+cu124, transformers 4.57.3). Local tests used a scratch venv with newer torch/transformers (2.15.0 / 5.22.1) because torch 2.6.0 has no wheel for the local Python; the toy-Llama test passed in both.
- **Memory:** full float64 model fit on the 80 GB H100; the layer-sliced fallback was not used.
- **Cost:** node time about 5 minutes running (04:27-04:31 UTC); `gman billing` available balance fell from 506.765 to 506.412 (about $0.35). `theseus-bandb-v8sb` is `stopped (disk intact)` per `gman node ls`; final balance 506.4054 (about $0.36 total). Cap was $15. Other nodes shown `running`/`queued`/`restoring` in `gman node ls` (r5-n256, r5-n512, caduceus-proposer-h100) are not mine (started by other sessions/users; I did not touch them).
