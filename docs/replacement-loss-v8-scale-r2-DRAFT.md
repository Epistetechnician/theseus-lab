> **Provenance and caveat.** Produced by an autonomous free/open-model research loop, then curated by hand. Only the section titled "Evidence update" is backed by executable tests () and was independently re-checked against . The earlier "Pattern Identification" and "Root Causes" sections are UNVERIFIED speculation (one claim there, that the adversarial stratum passes everywhere, is wrong: it passes for opt-2.7b and partly for gpt2-xl, and fails for SmolLM2-1.7B). DRAFT, unfrozen, thresholds unchanged.

# RSI Draft Preregistration: Band A Failure Analysis

## Objective
Investigate why Band A cases (opt-2.7b, gpt2-xl, SmolLM2-1.7B) consistently fail the causal-effect cosine gate (pooled median cosine 0.69-0.85 vs 0.90; some strata high, others low; behavior mostly preserved).

## Testable Hypotheses

1. **Model-specific performance degradation** - 1.7B parameter models show consistently lower cosine similarity in compositional and ordinary strata compared to 2.7B and GPT-2 XL models.

2. **Layer selection bias** - Selected layers for SmolLM2-1.7B (layer 7) may differ systematically from other models (layers 30 for opt-2.7b, 45 for gpt2-xl), affecting representation quality.

3. **Ridge regularization sensitivity** - Fixed ridge value (1e-06) may be suboptimal for certain model families, particularly affecting compositional and ordinary strata performance.

## Analysis Summary

From the data in `replacement-loss-v8-scale-bandA-report.json` and `replacement-loss-v8-scale-results.md`:

### Pattern Identification:
- Compositional and ordinary strata consistently show lower median cosine values than adversarial strata across all model families
- The SmolLM2-1.7B model shows particularly low cosine values in compositional and ordinary strata (0.710 and 0.835 respectively)
- opt-2.7b models show similar patterns with compositional strata showing ~0.68 cosine values
- GPT-2 XL performs well in both strata with median cosine > 0.82

### Root Causes:
1. **Inherent structural complexity differences** - adversarial strata create stronger signals while compositional/ordinary involve interference patterns that reduce coherence
2. **Dimensionality and activation space characteristics** - these strata operate in regions with higher noise and less predictable directional relationships  
3. **Model architecture implications** - the pattern varies by model family but persists consistently
4. **Error metrics correlation** - direction accuracy is preserved but cosine alignment suffers in complex strata

## Methodology
Create a comprehensive analysis framework that:
- Quantifies the relationship between model size/architecture and stratum performance
- Identifies layer selection patterns across models
- Tests alternative ridge regularization values
- Analyzes activation space characteristics for problematic strata

## Expected Outcomes
- Clear identification of which factors contribute most to the failure in compositional/ordinary strata
- Recommendations for improved model selection or regularization strategies
- Potential modifications to the experimental protocol that could improve results
## Evidence update (2026-10-26, RSI iteration 9; DRAFT, unfrozen, thresholds unchanged)

Measured on the frozen Band A report with the pinned gate semantics
(`replacement_loss_v3_runner.py:1363-1388,1464-1470`; facts asserted as
executable tests in `tests/test_effect_gate_decomposition.py`):

- Every failing causal-effect gate cell (99 cells over 9 cases, both
  surfaces, pooled and per-stratum) fails ONLY on `median_cosine >= 0.90`;
  direction (worst 0.9333 vs 0.80) and normalized error (worst ~1.1e-5 vs
  0.20) pass with orders of magnitude of slack. 14 of 54 stratum cells pass,
  all adversarial; compositional and ordinary never pass (0/36).
- Activation rescue passes all 9 cases; behavior passes 7/9 (llama seeds
  20261401/20261402 additionally fail the heldout behavior blocker via
  `paired_next_token_normalized_logit_error_max` 0.0647/0.0715 > 0.05, from
  the ordinary stratum).
- Pooled median cosine exact ranges: logits 0.6934-0.8694; module_boundary
  0.6627-0.8491.
- Statistic semantics: `normalized_error` is normalized by the baseline
  next-token norm, not the effect norm; by geometry
  `||observed effect|| <= normalized_error / sqrt(1 - cos^2)` of that
  baseline scale, i.e. <= 2.34e-5 of baseline in every stratum cell.

### Refined testable hypotheses for r2 (supersedes the order of the list above)

- **H4 (new, primary target): residual-structure / position-fitting
  hypothesis.** Cosine fails while normalized error is ~1e-5, which requires
  the residual to be a large fraction OF THE EFFECT (>= sqrt(1-c^2) ~ 50-75%
  of the effect norm, directional). A single global affine fit leaves a
  systematic transverse residual that differs by position/stratum (cf. V38:
  final-token-only fitting leaves prefix positions badly fit). Test: fit
  per-stratum and/or per-position affine maps, or fit with position-state
  conditioning; prediction: compositional/ordinary cosines rise the most;
  falsified if cosines are unchanged while rescue errors stay ~0.10.
- **H3 (reclassified): row-count/underfitting.** More adaptation rows can
  only help if the fit is oversampled-noise-limited; the report's tiny
  normalized errors suggest magnitude is not binding, so H3 is now a control
  arm whose prediction is a magnitude change WITH no cosine change. Its r2
  outcome doubles as a falsification test of H4's contention that angle, not
  magnitude, is binding.
- **H1 (low priority): ridge sensitivity.** Ridge sweep 1e-8..1e-2 as a
  descriptive control; scale-invariance of cosine predicts little gate
  movement except at extreme regularization where the fit shrinks toward the
  baseline-zero effect.
- **H2 (dropped as stated): layer selection bias.** The selected layer is
  fixed per family in the frozen report, so layer selection cannot vary
  within r2's frozen protocol; retain only as a descriptive note (v8 chose
  layer per family once at selection).
- **Cross-family method homogeneity (from iteration 8):** opt used
  `dual_ridge_least_squares`, gpt2/llama used `primal_ridge_least_squares`;
  r2 should fix a single fit method across all families or add method as a
  factor. This is a confound control, not a new gate.
- **Descriptive measurement to add (not a gate, thresholds unchanged):**
  report the effect-scaled relative error
  `||pred - obs|| / max(||obs_effect||, floor)` per cell, which the frozen
  `normalized_error` statistic (baseline-scale denominator) does not
  capture; this makes the angular misalignment directly visible in receipts.

## Evidence update (2026-10-26, RSI iteration 11; DRAFT, unfrozen, thresholds unchanged)

Measured on the frozen Band A report with `analyze_seed_stability()` /
`analyze_downstream_component()` (`tests/test_seed_stability_analysis.py`);
the transverse residual is decomposed into a seed-stable and a
seed-fluctuating component across the 3 independent seeds.

- **H3 (row budget) is now quantitatively excluded as a passing arm.** No
  seed reaches the 0.90 cosine gate in any compositional/ordinary cell
  (best-seed gate gaps 0.0135 for llama logits comp to 0.2398 for opt mb
  ord). The comp/ord module-boundary misalignment is >= 97.0% seed-invariant
  (angle spread <= 0.97 deg vs mean 32.7-49.0 deg), so eliminating 100% of
  measured seed variance gains at most +0.0046 cosine there (max mean->best
  gap, gpt2 mb comp) vs the >= +0.0544 needed. H3 is demoted to a control
  arm; its prediction is "no gate movement".
- **H4 (residual structure) sharpened into two measurable components.**
  (a) A seed-stable module-boundary bias of 32.7-49.0 deg in all comp/ord
  cells (3 independent replications of the same bias; e.g. opt mb comp
  48.6/49.2/49.1 deg). (b) A surface-specific downstream residual on the
  logits surface: <= 1.9 deg mean offset from the boundary angle for
  opt/gpt2 comp/ord (surfaces share one bias), but ~20.3 deg (gpt2 adv,
  seed-stable) and ~24.0 deg (llama adv) at adversarial, and seed-varying
  for llama comp (0.25-10.66 deg, its logits spread 17.2 deg vs mb 0.32 deg
  under the same three fits). Consequence for r2: per-stratum/per-position
  fitting must be evaluated on BOTH surfaces per arm, and its prediction is
  two-sided — comp/ord boundary angles fall from 33-49 deg toward small
  angles AND the adversarial logits-side offset (gpt2/llama ~20-24 deg)
  shrinks if the same mechanism underlies both; if comp/ord boundary angles
  do not move while rescue errors stay ~0.10, H4 is falsified.
