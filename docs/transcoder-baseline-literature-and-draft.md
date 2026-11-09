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

## 5. Open items before freezing

1. Read the repo file tree for layer/width/L0 layout and the sae_id; verify the loading path with the logit-reproduction test (license is CC-BY-4.0, done).
2. A real literature review (SAEBench loss-recovered handling, any epsilon-aware fidelity work).
3. Owner approval of layers, gate, cap.
