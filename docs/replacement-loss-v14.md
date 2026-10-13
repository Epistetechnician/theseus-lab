# Replacement-loss V14: paired affine-versus-GELU causal replacement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v14-paired-activation-ablation-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v14-paired-activation-ablation`.

## Question and claim ceiling

V13's GELU causal-fit candidate failed its held-out comparison with activation
ridge on all six seed clusters. V14 asks a narrower question: under a matched
training procedure, does inserting GELU between two low-rank linear maps
improve held-out causal-effect prediction over the same maps composed with an
identity activation?

The frozen target reuses the exact OPT/ReLU checkpoint bound by V13 and keeps
`fc2` layers 0 and 7 fixed. This is an outcome-aware follow-up on one known
checkpoint, not fresh-checkpoint or architecture-held-out validation. Model
weights remain in their existing external custody.

The primary metric is the seed-clustered mean difference in held-out
next-logit intervention-effect cosine, `gelu_fit - affine_fit`. Within each
seed, average the two selected layers and the three prompt strata. Report a
10,000-draw bootstrap lower 95% bound over six seed clusters. The positive
gate is lower bound `> 0.05`. Also require both candidates to pass the V10
per-cell retention gates, activation rescue for both candidates, and the
zero-intervention control.

The maximum positive claim is
`LOCAL_GELU_ACTIVATION_EFFECT_ON_MATCHED_LOW_RANK_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS`.
Even a pass would not establish architecture generality, mechanistic
understanding beyond this intervention family, novelty priority, independent
science, consciousness, alignment, human equivalence, or production
readiness.

## Frozen design

- Reuse the V13-bound OPT/ReLU checkpoint. Fix layers 0 and 7 before new
  fixture generation; these were selected from prior causal-effect failures.
- Use six new seeds, `202610101` through `202610106`, with a separate V14
  fixture namespace. Each seed has 96 adaptation prompts and 60 held-out
  challenges, balanced across ordinary, compositional, and adversarial strata.
- Use the same visible-only input scale and the same fixed intervention
  generator for both candidates. Generate one adaptation capture and one
  held-out challenge set per seed/layer; both candidate methods share those
  exact bytes.
- Both replacements are `Linear(3072,512) -> activation -> Linear(512,768)`.
  They have identical parameter shapes and 1,967,360 parameters. The only
  forward-function difference is GELU versus identity.
- Initialize both candidates from identical parameter tensors. For each
  seed/layer pair, use the same minibatch order and exactly 200 AdamW updates,
  batch size 8, learning rate `3e-4`, weight decay `1e-4`, and gradient clip
  1.0. Use the identical sum of normalized activation MSE and normalized
  next-logit intervention-effect MSE with unit weights.
- Candidate predictions are sealed before teacher outcomes are evaluated.
  The same challenge entropy is used by the second-runtime replay.
- Random replacement, wrong-layer replacement, and zero-intervention checks
  remain controls. A complete valid run that misses any scientific gate is
  `FAIL`; invalid custody, a resource breach, or an incomplete phase is
  `INCONCLUSIVE`.

The inherited V10 execution adapter uses the internal method key
`activation_ridge` for the affine candidate slot. In V14 this key is only a
compatibility alias: the candidate is trained with the paired objective and
is a factorized `Linear-Identity-Linear` module, not a closed-form ridge fit.
The candidate receipt binds that fact, the packet records the exact alias
mapping, and outcome rows are relabeled `affine_fit` before their result
digest is written. The `causal_fit` key maps to `gelu_fit` in the result.

## Variance, stopping, and resources

The seed is the sole bootstrap cluster. Layers and prompts remain nested
within seed. Variance sources include fixture seed, shared paired
initialization, prompt-template coverage, and numeric runtime behavior.
Initialization and minibatch order are matched within each seed/layer pair.
There is no tuning, early stopping, threshold change, or prompt edit after
freeze. Stop after six seed clusters and two fixed layers.

Execution is local CPU-only with one torch thread, deterministic algorithms,
a four-hour wall-clock cap, 16-GiB RSS cap, 200-million base-model parameter
cap, and 800-MB checkpoint cap. No provider calls, paid or remote workers,
GMan nodes, real-user inputs, or public writes are allowed. Cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`.

## Procedure and custody

1. Freeze protocol, fixture generator, runner, tests, dependency lock, exact
   code dependencies, target-selection lineage, thresholds, and controls
   before creating V14 prompts or candidate states.
2. Archive the frozen packet to new external custody roots. Bind the already
   recorded V11/V13 OPT checkpoint selection without copying weights.
3. Capture each visible/adaptation fixture once, initialize the paired
   candidates identically, train each with the same batches/objective/budget,
   and seal the candidate set.
4. Generate fresh challenge entropy and prompts after candidate freeze. Seal
   predictions before scoring teacher outcomes.
5. Re-run the exact packet in the separate pinned Python runtime using the
   primary's sealed challenge entropy. Compare candidate, challenge,
   prediction, outcome, and runtime digests.
6. Preserve all stop receipts. Never retry a partially written immutable
   phase in the same custody root.

Synthetic prompts, captures, candidate tensors, entropy, predictions, and
row-level outcomes stay in external owner custody. Only protocol, code,
aggregate results, and digests are stored in the repository. Reviewer status
is `NOT_SUBMITTED`; a same-owner second-runtime replay can report
`REPRODUCED_LOCAL` but cannot satisfy independent science.
