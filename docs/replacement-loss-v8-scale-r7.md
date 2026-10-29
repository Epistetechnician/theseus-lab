# Replacement-loss-v8-scale-r7: float64 check of the r6 equivalence-diagnostic misses

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r7-equivalence-float64-diagnostic`.

Status: preregistration, frozen by `docs/replacement-loss-v8s7-freeze.json` before any run. (Pre-run amendment: the first r7 freeze, commit `8a5c4c8`, lacked the read-only `diagnose` mode and described regenerated inputs; found r6 custody intact before any run and refroze. Both freezes are in Git history.) This is a **separate, diagnostic-only** packet. It does not re-score, reclassify or replace any r6 arm. **The r6 NN256 and NN512 arm outcomes remain `FAIL` whatever r7 finds.** No frozen constant (including `EQUIVALENCE_MAX_ABS_ERROR = 1e-4`) and no gate is changed; r6 files are pinned unmodified in the chain.

## Question

r6 NN256/NN512 passed every per-case and bootstrap gate (9/9) but each arm has two SmolLM2 cases whose mapping-aligned permutation diagnostic exceeds 1e-4 absolute (module-effect errors 1.2e-4..8.5e-4). Hypothesis (untested in r6): that is float32 rounding noise, larger because natural-prompt activations are larger than the template activations the 1e-4 bound was calibrated on.

The diagnostic compares the same fitted treatment module run through the model twice, normally and with the input permutation re-aligned; any difference is summation-order rounding.

## Design

- Arms NN256 and NN512, SmolLM2-1.7B only in the diagnostic, all three seeds (6 cases). Adaptation fits, seeds, natural-prompt selection and challenge construction reuse the pinned r6 code unchanged (r6 runner stages up to challenge generation); the r6 outcome pass is not run.
- **Inputs (primary mode `diagnose`):** the original r6 custody on the r5 nodes (`custody/r6-NN256`, `custody/r6-NN512`), read-only: the same fitted treatment/mapping-aligned modules and the same challenge prompts and interventions as r6. If an arm's custody turns out to be missing, the fallback is `run`, which regenerates fits and challenges with the pinned r6 code (challenge entropy then differs from r6; stated in the results doc). Each case is measured in float32 *and* float64 in r7, so float32 reproduction of the r6 miss is itself checked rather than assumed.
- Per case, on the 180 challenge prompts with their interventions: compute the treatment and mapping-aligned module-effect and logit-effect predictions with model and replacement module in float32 (as r6), then cast the same weights exactly to float64 (`model.double()`, module `.double()`, deltas double) and repeat. Report max-abs errors (module, logit) for both precisions.
- Activation scale per case: max |treatment boundary output|, median L2 norm of module input and of boundary output. Relative fp32 error = fp32 module error / max |boundary output|.
- Nothing in r7 touches original-model outcomes, thresholds, or gates.

## Predictions and classification (fixed now)

- **P1:** float64 module and logit errors are <= 1e-4 in every one of the 6 cases.
- **P2:** in every case with float32 error > 1e-4, the float64 error is <= (float32 error)/100.
- **P3 (scale claim):** relative fp32 error <= 1e-5 in all cases. Also reported descriptively: Spearman rank correlation across the 6 cases between fp32 error and max |boundary output| (n=6; not a significance claim).
- `INCONCLUSIVE_NOT_REPRODUCED`: no case has float32 error > 1e-4 (the r6 misses did not recur on these inputs; hypothesis untestable here, not support).
- `SUPPORTED_FLOAT32_NOISE`: P1 and P2. `REJECTED`: P1 fails in any case (the misses are not explained by float32 precision). `PARTIAL`: otherwise.
- P3 is reported separately as `SCALE_CONSISTENT` / `SCALE_NOT_CONSISTENT`.

## Limits and nonclaims

CPU, same owner, SmolLM2 only (opt-2.7b and gpt2-xl had no misses), one projection, 6 cases, no independent review. A SUPPORTED result says the numerical error shrinks under higher precision on these weights; it does not alter, waive or reinterpret the frozen r6 gate and is not a PASS of any arm. The claim that the 1e-4 bound was calibrated on template activations is not directly tested (no template-scale comparison here).

## Budget

Two gman `cpu-8` nodes (r5-n256, r5-n512 disks hold the venv, checkpoints, wikitext parquet), about 2 hours each; cap $10 total.
