# Replacement-loss-v8-scale-r8: float64 downstream check of the open_llama_7b_v2 logit-effect cosines

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r8-float64-downstream-logit-cosine-diagnostic`.

Status: preregistration. This document is committed before the r8 runner, tests, freeze packet, or any run exists; the runner and packet freeze follow in later commits and cannot change the predictions or decision rules below (any change would be a disclosed amendment in Git history). This is a **separate, diagnostic-only** packet. It does not re-score, reclassify or replace any Band B arm. **The Band B outcomes (open_llama_7b_v2 `FAIL` at N256 and N512; opt-6.7b `FAIL` at N256, `PASS` at N512; arm-level N256 `FAIL`, N512 `FAIL`) are unchanged whatever r8 finds.** No frozen constant and no gate is changed; Band B and r7 files are pinned unmodified in the chain.

## Question

`docs/bandb-open-llama-diagnosis.md` observed that open_llama_7b_v2 fails the per-stratum logit-effect cosine gate (0.90) at N512 (compositional 0.80-0.85, ordinary 0.91-0.93) although its module-boundary cosine is 0.99996 and direction accuracy and top-1 agreement are 1.00, and that logit-effect `median_normalized_error` (about 3e-7) sits at a few float32 epsilons (1.19e-7). Three explanations were left untested:

- **H1 (float32 resolution floor):** the compositional/ordinary logit effects are only a few orders above the float32 rounding of the logits, so the cosine measures rounding noise, not replacement quality (the r7 finding for r6's equivalence misses).
- **H2 (depth amplification):** layer 7 of 32 leaves 24 layers that amplify a small module residual. Not tested here (needs a layer sweep).
- **H3 (architecture):** an affine output projection is a worse local model for Llama's gated MLP on compositional edits; the gap is real.

r8 distinguishes H1 from H3 with one measurement: recompute the same logit-effect cosines with the downstream forward in float64.

## Design

- **Inputs (read-only):** the Band B N512 custody on node `theseus-bandb-v8sb`: the same fitted treatment modules (float32 ridge fits, unchanged), the same checkpoint (open_llama_7b_v2, revision `e5961def23172a2384543940e773ab676033c963`), and the same 180 challenge prompts per seed with the same intervention deltas (per-seed file digests verified against the custody challenge manifest). Seeds 20261901-20261903 (seed_index 0-2), selected layer 7, 60 resolvable prompts per case (20 per stratum). Nothing is re-fit or regenerated. Only open_llama_7b_v2 is measured.
- **Per case, two arms of the same pipeline:**
  - `float32`: model and replacement module in float32 on the H100, TF32 off, exactly as Band B. Treatment predicted logit effect = (module-replaced model with delta added at the module input) minus (module-replaced model without delta); observed logit effect = (original model with delta) minus (original model without delta), next-token logits at the last prompt position. Scoring uses the unchanged `causal_effect_metrics` with Band B's resolution floors and scale.
  - `float64`: the identical weights cast exactly to float64 (`model.double()`, module `.double()`, deltas cast from their stored float32), and the same two effects recomputed in float64 end to end (embedding, all layers, norm, lm_head). Scoring unchanged, floors computed by the same formula from the float64 baseline and repeat passes. The fp32 fit is not touched; this changes only the precision of the forward that produces both predicted and observed effects.
- **Memory:** the fp64 model is about 54 GB (6.74e9 parameters x 8 B) and fits one 80 GB H100 with batch-8 activations. Fallback if a CUDA OOM occurs at the cast: layers 0-6 and embeddings stay float32 and their layer-7-input activation is cast to float64 (identical for the original and replaced models); layers 7-31, final norm and lm_head run in float64. Any use of the fallback is disclosed in the results doc and changes nothing in the rules.
- **Reported per case and stratum** (compositional, ordinary, adversarial): median logit-effect cosine (float32 and float64), resolvable count, direction accuracy, median normalized error, module-boundary cosine; median L2 norm of the observed effect (float64) divided by the L2 norm of the baseline next-token logits ("relative effect size"), and that ratio divided by float32 epsilon (1.19209e-7); median ratio ||effect_fp32 - effect_fp64|| / ||effect_fp64|| for the observed and the predicted effect (direct measure of float32 rounding noise in the effect vectors); median cosine between the fp32 and fp64 observed effects, and between the fp32 and fp64 predicted effects.

## Validity gate (fixed now)

The float32 arm of r8 must reproduce Band B on the same inputs: for every one of 3 seeds x 3 strata, |r8 float32 median logit cosine - recorded Band B N512 value| <= 0.02. If any of the nine misses, the r8 outcome is `INCONCLUSIVE_NOT_REPRODUCED` (the float64 result cannot be attributed to precision rather than a changed pipeline) and the H1/H3 rules below are not applied. The 0.02 tolerance covers GPU run-to-run nondeterminism; it is set before the run.

## Predictions and decision rules (fixed now)

Primary quantity: the float64 median logit-effect cosine C64(seed, stratum) for stratum in {compositional, ordinary}: six values.

- **H1 prediction:** all six C64 >= 0.99.
- **H3 prediction:** all six C64 < 0.90 (i.e. stay in the regime of the float32 numbers).
- Outcome classes (applied only if the validity gate holds):
  - `H1_SUPPORTED_ROUNDING_FLOOR`: all six C64 >= 0.99.
  - `H3_SUPPORTED_ARCHITECTURE_OR_FIT`: all six C64 < 0.90. (This rejects H1 and leaves H2 and H3 open between them; r8 alone does not separate H2 from H3.)
  - `PARTIAL_RISE`: all six C64 >= 0.90 but not all >= 0.99: float32 noise contributes but does not explain the whole gap or the cosines are not yet at the module-boundary level.
  - `MIXED`: otherwise (some C64 below 0.90 and some at or above): reported per cell, no hypothesis claimed.
- **Secondary predictions (descriptive, not decision rules):** under H1 the float32 rounding noise in the compositional observed effect is large (median ||eff32 - eff64|| / ||eff64|| of order 0.1-1) and the relative effect size is within about 10^2-10^4 of float32 epsilon; under H3 the noise ratio is small (<= 0.05) and fp64 cosines match fp32 cosines to within about 0.02. The compositional stratum's relative effect size is expected to be smaller than the adversarial stratum's, since compositional/ordinary cosines are the low ones and adversarial is high (0.98).
- Boundary-cosine check (descriptive): float64 module-boundary median cosine stays >= 0.9999.

## Limits and nonclaims

One checkpoint, one projection (layer 7), N512 fits only, 3 seeds x 60 prompts, one host (GPU), same owner, no independent review. float64 is a diagnostic: it changes the precision of the forward that produces the effects, and says the logit cosines are (or are not) sensitive to float32 rounding on these weights. A `H1_SUPPORTED` result does not alter, waive or reinterpret the frozen per-stratum gate or the recorded Band B FAIL and is not a PASS of any arm; it would motivate a separate preregistered question about whether the gate's float32 measurement is the right instrument. `H3_SUPPORTED` does not isolate depth (H2) from architecture (H3). The pooled and adversarial results are not re-gated.

## Budget and stop rules

One gman `h100-1` node (about $5.04/h), the existing `theseus-bandb-v8sb` node and disk (custody and checkpoints already present; nothing is downloaded or re-fit unless the custody is missing, in which case r8 is recorded `INCONCLUSIVE_CUSTODY_MISSING` and not regenerated). Expected under 1 hour (about $5). Hard cap $15 (authorized); stop the node by about $12 of elapsed node time regardless of progress. Every node started is stopped and checked with `gman node ls`. No retry beyond fixing a pure infrastructure fault that occurs before any measurement, disclosed if it happens.
