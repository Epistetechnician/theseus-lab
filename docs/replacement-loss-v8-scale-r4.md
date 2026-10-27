# Replacement-loss-v8-scale-r4: is the Band A failure double descent?

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r4-ridge-doubledescent`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8s4-freeze.json` before any run. All earlier packets remain immutable.

## Motivation (computed from saved reports, `docs/replacement-loss-v8-scale-*-report.json`)

Across Band A, r2-S1, r3-C2 and r3-C4, mean pooled median cosine tracks adaptation rows divided by the projection input width:

| rows / width | cosine | outcome |
|---|---|---|
| 0.7-1.2 (128 prompts) | 0.70-0.85 | FAIL (Band A, r2-S1) |
| 1.5 | 0.946 | PASS (opt-2.7b, C2) |
| 2.2-2.4 | 0.98-0.999 | PASS |
| 3.0-4.7 | 0.986-0.999 | PASS |

Widths are the FFN inner sizes from the public model configs (opt-2.7b 10240, gpt2-xl 6400, SmolLM2-1.7B 8192; verify). A fit near one row per dimension with ridge `1e-6` sits at the classical interpolation threshold, where minimum-norm least squares has maximal variance (double descent).

## Questions

- **H-DD (variance near the interpolation threshold):** at 128 prompts, a sufficiently larger ridge rescues the fit. Predicted: at some `alpha` in the ladder, pooled median cosine reaches >= 0.90 for every model.
- **H-INFO (not enough information):** no ridge rescues 128-prompt fits; more data is the only fix. Predicted: median cosine stays below 0.90 for all `alpha`.
- Both are decided from the frozen gates, nothing else. Mixed outcomes (rescue for some models only) are reported as such.

## Design (everything else pinned from v3/v8/r2/r3)

Same three checkpoints (opt-2.7b, gpt2-xl, SmolLM2-1.7B at the earlier pinned revisions), re-bound after this freeze. Fresh seeds `20261701`-`20261703`, new `V8S4` namespace. 128 adaptation prompts in every arm. Intervention scale 1x.

Ridge is expressed relative to scale so it transfers across models: `ridge = alpha * ||X_c||_F^2 / width` (mean Gram eigenvalue of the bias-centred adaptation inputs), applied to the treatment fit and to the wrong-module and random controls alike. The fitter is the pinned v3 function; only its ridge argument changes.

| arm | alpha |
|---|---|
| R6 (in-packet control) | 1e-6 |
| R4 | 1e-4 |
| R2 | 1e-2 |
| R0 | 1.0 |

Gates, controls, cluster bootstrap, wall caps and the 30 GiB memory limit follow r3. R6 is not guaranteed to equal the earlier absolute `1e-6` (its effective ridge depends on activation scale); the earlier S1 and Band A runs remain the reference at absolute `1e-6`.

## Outcomes

Per arm `PASS`/`FAIL` on the frozen gates; resource or identity failures are `INCONCLUSIVE` and not retried. Descriptive: pooled and per-stratum median cosine against `alpha`, by model.

## Forward prediction for Band B (registered before any Band B run)

If the rows-per-width rule holds, then for larger models of similar family: with about 60 rows per prompt, Llama-style 7B (FFN width about 11008) needs about 24k rows (about 400 prompts) for rows/width >= 2.2, and OPT-6.7B (FFN width 16384) about 36k rows (about 600 prompts). Specifically: at 256 prompts rows/width is below 1.5 for both and cosine is predicted below 0.90; at 1024 prompts it is predicted at or above 0.98. A Band B run that contradicts this falsifies the rule as a transferable guide.

## Limits and nonclaims

CPU fp32, same owner, synthetic template prompts (a natural-text arm is a separate, later packet), one projection per checkpoint, same checkpoints as the packets that motivated this, no independent review. Neither outcome supports mechanistic, alignment or production claims.

## Budget

Four gman `cpu-8` nodes in parallel, about 1.5-2 hours each, cap $30.
