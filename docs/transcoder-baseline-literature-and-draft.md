# Transcoder baseline: literature note and DRAFT preregistration (unfrozen)

Status: **DRAFT. Not frozen. No runner, packet, node, or spend.** Nothing here changes any earlier outcome, gate, or constant. Literature notes come from a short web search (2026-11-09), not a systematic review; novelty is **not** established.

## 1. Literature context (what the search found)

- **Replacement-model faithfulness is contested.** A LessWrong post, "Cross-Layer Transcoders are incentivized to learn Unfaithful Circuits", argues from toy models (plus preliminary language-model evidence) that CLTs can collapse a multi-step circuit into a single step, so the circuit shown differs from the true computation. Evidence is toy-model and preliminary, not a settled benchmark.
- **Gemma Scope (arXiv 2408.05147).** Includes a suite of transcoders for Gemma 2 2B PT (`google/gemma-scope-2b-pt-transcoders`). The paper reports transcoders raising loss more than MLP-output SAEs at fixed L0 on Gemma 2, reversing the GPT-2 small trend of Dunefsky et al. (arXiv 2406.11944). Gemma Scope 2 (Gemma 3) also ships transcoders.
- **Precision.** Gemma Scope lists bf16 vs float32 as a research question for SAEs; a secondary summary says bf16 inference barely changes SAE fidelity. I found no source measuring precision effects on transcoders, or on small-effect (intervention) fidelity specifically.
- **Benchmark reliability.** Chanin, "Are Sparse Autoencoder Benchmarks Reliable?" (arXiv 2605.18229): several SAEBench metrics show high reseed noise and low discriminability; two should not be used. This supports, but does not prove, that small metric differences in this area are hard to read.
- **Not found / unverified:** trained GPT-2 transcoder weights (code is at github.com/jacobdunefsky/transcoder_circuits; weights not confirmed), and any prior work that reports fidelity of small intervention effects in units of floating-point epsilon.

**What this does and does not say about our work.** Our replacement is a single affine MLP projection, a linear baseline, not a competitor to transcoders. The possibly-novel part is methodological (instrument floor, precision ladder, negative control, custody), pending a real literature review.

## 2. Question

Does the float32 instrument floor that limited our affine replacements (r6 to r10) also limit fidelity measurement of a **published transcoder** replacement, and how does the transcoder compare with our affine fit on the same yardstick?

## 3. Proposed design (to be edited before freezing)

- **Model/artifact:** Gemma 2 2B PT with the public Gemma Scope transcoders, a small set of layers (proposal: 3, spread early/mid/late). The model fits float64 on one H100 (about 21 GB weights).
- **Arms per layer:** (a) public transcoder; (b) our closed-form ridge affine fit on the same activations (r-series method), same data split.
- **Quantities, float32 and float64 (cast exactly, no refit):** per-stratum median logit-effect cosine for synthetic MLP-input interventions through original MLP vs replacement; effect size in units of float32 epsilon times logit norm; module-boundary cosine; delta LM loss on held-out natural text (the field's standard number, for external comparability).
- **Not applicable:** the mapping-aligned permutation diagnostic (it assumes an affine fitted map); transcoders have no such equivalence. Stated so no gate is silently dropped.
- **Gate, written before data (to confirm):** an arm-layer "passes" if float64 per-stratum median cosine >= 0.90 (the earlier case-gate value, unchanged). Reported, not gated: float32 vs float64 gap, effect size in epsilons, delta loss.
- **Predictions (numeric, before data):** P1 the transcoder float64 cosine exceeds our affine arm's at every layer (it has far more capacity); P2 the float32-to-float64 cosine gap is under 0.02 for the transcoder at layers where effects are over 1e4 epsilons, and over 0.05 where they are under 1e2 epsilons (floor depends on effect size, not method); P3 delta loss for the transcoders is within 20% of Gemma Scope's published figure at matching L0. A P1 or P3 miss is reported as such.
- **Negative control:** an under-trained or noise-perturbed transcoder (random-weight blend at a fixed interpolation) must fail the float64 gate. One control only: stated weakness.
- **Cost:** one H100 session; estimate $5 to $10 (unmeasured). Proposed hard cap $20.

## 4. Threats and ceiling

One model family, one owner, a handful of layers, synthetic interventions, no independent review. Transcoder results do not transfer to other releases (Gemma Scope itself shows GPT-2 findings did not transfer). Even a clean result only concerns measurement; it supports no mechanistic or alignment claim. Hypotheses until independently validated.

## 4b. Refinements from the model card (checked 2026-11-09)

- **License:** CC-BY-4.0 on `google/gemma-scope-2b-pt-transcoders`.
- **Module boundary (matters for the design):** the transcoder reads activations *just after the pre-MLP RMSNorm* and reconstructs the MLP output *after the post-MLP RMSNorm*; pre-MLP RMSNorm gain is folded into the MLP input matrices. The card carries an explicit clarification because this was misread before. Consequence: the replaced module is `post_norm(MLP(pre_norm(x)))`, and the affine ridge arm must be fitted on exactly that input/output pair, or the head-to-head is invalid. The paper's Figure 12 (TransformerLens) shows the weight folding; the runner must reproduce it and a test must show the unmodified model plus folded weights reproduces original logits to float32 tolerance *before* any replacement is applied.
- **Loading:** the card shows only placeholder SAELens code; a verified `sae_id` for this repo was not found. Layer/width/L0 file layout is not on the card page. Both must be read from the repo file tree before the layer set is fixed, so "3 layers spread early/mid/late" stays a proposal, not a commitment.
- **dtype:** not stated. The float64 cast is of the stored parameters (exact upcast); no refit.
- **Fallback if loading proves unreliable:** report `NOT_RUN: transcoder loading unverified` rather than substituting another checkpoint.

## 4c. Repo layout and layer rule (read from the repo file tree, 2026-11-09)

Layout: `layer_N/width_16k/average_l0_X/params.npz`; 26 layers (0 to 25); width 16k only; 198 files of about 302 MB each (Git LFS); 3 to 10 L0 variants per layer (3 to 5 for layers 0 to 10, 10 for layers 11 to 22, 9 for layers 23 to 25; L0 range 3 to 1797). No `sae_id` is needed if the `params.npz` files are loaded directly; the SAELens path stays unverified.

**Layer rule, fixed here before any data:** layers 5, 12, 20 (early, middle, late of 26). For each layer use the available `average_l0_X` closest to 100 (ties go to the smaller L0). The L0 target of 100 is an arbitrary round choice and is stated as such; it is not tuned. Three files, about 0.9 GB.

## 4d. Added literature (searched 2026-11-09, still not a systematic review)

- A LessWrong post by Gabriel Goh argues exact input/output determinism is nearly impossible under floating point and proposes "effectively identical IO" as the working standard for interpretable replacements.
- The compact-proofs paper (arXiv 2406.11779) explicitly leaves the floating-point-to-real gap to future work, and notes its models are 32-bit.
- arXiv 2605.25991 uses stochastic arithmetic in PyTorch to estimate numerical variability without changing the model: a candidate way to measure a noise floor independent of the float32-vs-float64 comparison. It is not an interpretability paper.
- **Still not found:** work that ties a machine-epsilon-based tolerance to patching or logit-effect fidelity. Absence in a few searches is weak evidence; do not claim novelty without a proper review.

## 4e. Added arm: stochastic-arithmetic noise floor (approved by owner)

For each transcoder and affine arm-layer, also estimate the logit-effect rounding floor without relying on the float32-vs-float64 comparison: re-run the float32 forward K times (K=8) with independent random perturbations of the last bit of MLP-input activations (relative size 2^-23) and report the spread of the logit-effect vector and of its cosine with the float64 effect. Reported, not gated. It tests whether the float32 floor seen via float64 matches an independently estimated floor. A mismatch is reported, not explained away. (A full stochastic-arithmetic library, as in arXiv 2605.25991, is an option if the simple perturbation proves too crude; this is stated as a weaker substitute.)

## 4f. Implementation status

`transcoder_baseline_core.py` (model-free: JumpReLU forward, ridge affine fit, epsilon-unit effect size, perturbation noise floor, gate, P1/P2 checks) and `tests/test_transcoder_baseline_core.py` (9 tests, toy tensors) exist. **Not written:** the Gemma 2 2B model glue (weight folding, hook at the pre-/post-MLP-norm boundary, downstream logit-effect forward, delta LM loss) and the runner. It cannot be tested without the model, so **the packet is deliberately not frozen**: freezing code that has not run against the model would freeze untested glue. P3 (delta-loss agreement) is evaluated in the glue, not the core.

## 4g. Model glue status and a convention caveat

`transcoder_baseline_model.py` (capture MLP input/target at the Gemma 2 norm boundaries, block replacement hook, delta LM loss, variance explained) and `tests/test_transcoder_baseline_model.py` exist; 15 tests (glue + core) pass on a **tiny random Gemma2 on CPU**, including identity replacement reproducing logits (1e-10 in float64, 1e-4 in float32). This validates the plumbing only, not the real checkpoint.

**Convention caveat:** the paper's appendix says input and target are "divided by the RMS of the input activations", and does not say whether the target is before or after the post-MLP norm; the model card says after. The glue applies no extra scaling. The gating check on the real model is therefore: transcoder **variance explained of the post-norm MLP output at the chosen layers must be at least 0.8** under the model-card convention; otherwise that layer is `NOT_RUN: convention unverified` (no silent convention search). Other conventions may be tried as diagnostics only.

**Access blocker:** `google/gemma-2-2b` is gated on Hugging Face, so a run needs the owner's accepted licence and a token on the node. A token must never be pasted into chat or committed. No ungated mirror is used without owner approval (provenance).

## 4h. Real-model smoke test (2026-11-10, `theseus-bandb-v8sb`, H100; smoke only, not a result)

Gemma 2 2B (fetched through the stored `huggingface` gman connection, 9.8 GiB) and the three transcoder files (public, fetched directly on the node, no token) were run on 8 fixed sentences truncated to 22 tokens each (176 tokens, BOS included). Code: `transcoder_baseline_smoke.py` plus two throwaway diagnostics (not committed). Numbers are smoke values from a tiny text; they say nothing about the later protocol.

| layer (L0) | identity logit diff (f32) | VE all | VE excl. BOS | non-BOS cos(pred, target) | delta CE: transcoder / zero-abl / mean-abl |
|---|---:|---:|---:|---:|---|
| 5 (87) | 1.2e-4 | 0.54 | 0.63 | 0.89 | -0.028 / 0.018 / 0.004 |
| 12 (111) | 1.1e-4 | 0.26 | 0.56 | 0.76 | 0.024 / 0.056 / 0.031 |
| 20 (88) | 5.2e-5 | 0.61 | 0.67 | 0.89 | 0.032 / 0.149 / 0.062 |

- **Plumbing works.** Identity replacement reproduces the logits to about 1e-4 (float32) on the real model; input RMS is 1.000 after un-gaining (matches the card).
- **The preregistered VE >= 0.8 gate fails at all three layers**, so under the draft as written every layer is `NOT_RUN: convention unverified`. I did not run anything past the smoke test.
- **Why that gate looks wrong, and what I am not claiming.** The two alternative input scalings are far worse (VE -0.15 to -290000), BOS positions have 2 to 4 times larger targets and are poorly reconstructed (cos 0.12 and -0.08 at layers 12 and 20), and non-BOS cosine is 0.76 to 0.89. That is consistent with the model-card convention being right and the 0.8 threshold being a guess that these L0 ~ 100 transcoders do not reach. It does not prove it.
- **Delta loss is nearly uninformative on this text.** Zero-ablating the whole block moves CE by only 0.018 to 0.149 nats (the Gemma residual stream is large), so loss-recovered ratios here are noise (2.6 at layer 5). Delta LM loss needs a much larger sample before it is a usable number; it should stay reported, not gated.
- **Amendment (approved by owner 2026-11-10, see 4i; informed by this smoke data, so not blind):** replace the VE >= 0.8 gate by a convention check that the model-card convention beats both alternative input scalings in VE and has non-BOS cosine >= 0.5, with VE reported, not gated. Exclude BOS positions from all effect statistics, stated up front. Until approved, the layers stay `NOT_RUN` and the packet unfrozen.

## 4i. Amendment adopted (owner approved 2026-11-10; informed by smoke data, so not blind)

Replaces the "VE >= 0.8" gate of 4g. Everything else in 4g stands.

- **Convention check per layer (gates whether the layer is run):** with BOS positions excluded, the model-card input scaling must beat both alternative scalings (x*sqrt(d), x/sqrt(d)) in variance explained, and non-BOS cosine(pred, target) must be >= 0.5. Fail means `NOT_RUN: convention unverified` for that layer. Implemented as `convention_check` in the core.
- **VE is reported, not gated.** Observed smoke values 0.56 to 0.67 are not evidence of a good replacement, only of a plausible convention.
- **BOS excluded from all effect statistics and the affine fit**, stated before the run.
- **Disclosure:** the 0.5 cosine threshold and the exclusion rule were chosen after seeing 176-token smoke data (non-BOS cosine 0.76 to 0.89). They are loose by design, but the choice is not blind and is reported as such in the results.
- **Negative control fixed:** `blend_with_random` at t = 0.5 (W_enc and W_dec blended with Gaussian noise of equal std; biases and thresholds unchanged). It must fail the float64 case gate (all strata median cosine >= 0.90); otherwise the instrument is reported as unable to distinguish good from bad replacements.
- **Interventions (runner `transcoder_baseline_run.py`):** add `scale * u` to the un-gained MLP input at one non-BOS position, `u` a fixed-seed random unit-RMS direction; scales 1e-3 / 1e-1 / 1 form three strata. Effect = logits(with) minus logits(without) within each arm (original block; replacement block); the compared quantity is the cosine of the two flattened effects, median over cases per stratum.
- **Data:** the corpus path is a runner argument and must be fixed (with a hash) in the frozen packet; not chosen yet. Fit/eval split is by document.
- **P3 caveat:** no published Gemma Scope transcoder delta-loss figure has been located, so P3 is reported "not evaluable" unless one is found before freezing.

## 5. Open items before freezing

1. Verify direct `params.npz` loading with the logit-reproduction test (layout and license done; layers and L0 rule fixed in 4c).
2. A real literature review (SAEBench loss-recovered handling, any epsilon-aware fidelity work); consider adding a stochastic-arithmetic noise-floor arm.
3. Owner approval of layers, gate, cap. (Gate amendment approved 2026-11-10; layers/cap still to confirm at freeze.)
4. Choose and hash the evaluation corpus (runner argument; not chosen). Needs enough documents of at least 64 tokens; the smoke text was far too small for delta loss.
5. `transcoder_baseline_run.py` is written and structure-tested on a tiny random Gemma2 (20 tests pass); it has **not** run on the real model, so it is not frozen. Run it once as an unfrozen dry run on a small corpus before freezing.
