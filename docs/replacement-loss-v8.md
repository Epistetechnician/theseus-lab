# Replacement-loss-v8: independently fitted affine replacement

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v8-ridge-primary-causal`.

Status: preselection preregistration. Freeze this document, fixture rules,
runner, tests, dependency manifest, and pinned v7/v6/v3 implementation bytes
before selecting any v8 checkpoint. v3 through v7 remain immutable. The v7
primary AdamW treatment remains a reproduced `FAIL`. Its predeclared
output-only ridge comparator motivates this separate confirmatory packet;
the v7 comparator diagnostic does not count as a v8 result.

## Question and claim ceiling

Can a fresh affine MLP output projection, fitted only from disjoint adaptation
input/output pairs with the pinned closed-form ridge estimator, predict held-
out module activations, downstream behavior, and causal effects of unseen
interventions across three fresh checkpoint families?

A passing result supports only this local replacement procedure for one
projection in each selected checkpoint. It does not establish equivalence of
whole models, general mechanistic understanding across systems, validity for
all architectures, consciousness, alignment, independent science, or
production readiness.

## Targets, fixtures, and cases

After the packet is frozen, select exactly one public base checkpoint for
each family: OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. Each checkpoint must
have a full 40-character immutable revision, at most 500 million parameters,
match the frozen model/projection contract, load with
`trust_remote_code=False`, and be absent from all checkpoint identities used
in v3-v7. Stop before candidate preparation and report
`NO_ELIGIBLE_FRESH_TARGET` if any family has no eligible checkpoint.

Run three predeclared seeds, `20261301`, `20261302`, and `20261303`, yielding
nine checkpoint-by-seed cases. Use visible-only prompts to select the MLP
output projection with the greatest mean per-feature input variance; ties go
to the lower layer index. The projections are OPT `fc2`, GPT-2 `mlp.c_proj`,
and Llama `mlp.down_proj`.

Each case uses 128 adaptation prompts (64 ordinary, 32 compositional, 32
adversarial) and 60 held-out challenge prompts (20 per stratum). Visible,
adaptation, and challenge partitions must be disjoint by example ID, exact
prompt, and exact prompt-plus-continuation, and disjoint from all prior
custodied v3-v7 fixtures. Use the v8 fixture namespace and retain all prompt
text only in external owner custody.

## Independently fitted treatment and controls

Freeze the base model. Capture selected projection inputs and outputs using
only adaptation prompts. Fit a new affine `torch.nn.Linear` with the pinned
v3 `fit_independent_linear_replacement` ridge value `1e-6`; copy no target
parameters. The fitted treatment must be created before held-out challenge
entropy is drawn. Preserve the v3 adjacent-projection fit, random-target,
raw-index permutation, mapping-aligned permutation, and no-op controls.
Record the treatment's adaptation fixture, capture, candidate-state, and
candidate-file digests.

After all nine candidates and receipts are frozen, draw fresh challenge
entropy. Seal candidate predictions for the held-out challenge before
opening original-model outcomes. Use the pinned v3 causal, behavior, and
activation-rescue metrics and thresholds unchanged:

- causal direction accuracy at least `0.80`, median cosine at least `0.90`,
  and median normalized error at most `0.20`, with all 20 prompts per stratum
  resolvable;
- held-out next-logit normalized error at most `0.05` and continuation cross-
  entropy increase at most `0.05` nats per token, overall and in each stratum;
- activation cosine at least `0.90` and normalized activation error at most
  `0.20`, overall and in each stratum;
- positive treatment-versus-control cluster-bootstrap lower bounds, with the
  existing v3 family-by-seed cluster definition and thresholds; and
- mapping-aligned permutation module-effect and logit-effect maximum absolute
  errors at most `1e-4` in every case.

Overall outcome is `PASS` only if every frozen case and aggregate gate passes.
Measured misses are `FAIL`; missing fixtures, digest mismatch, incomplete
prediction sealing, or resource termination are `INCONCLUSIVE`. Do not alter
the thresholds after checkpoint selection.

## Reproduction, custody, and nonclaims

Use local CPU only. Keep model weights, tokenizers, prompts, captures,
candidates, predictions, outcomes, and row-level diagnostics in the external
owner-custody root. The repository stores only the frozen protocol and
aggregate-safe results. Run the same packet in the configured separate clean
runtime. Report `REPRODUCED_LOCAL` only if candidate reconstruction,
prediction and outcome arrays, source identity, dependencies, and distinct
runtime checks all match. Same-owner reproduction is not independent science.

External reviewer status remains `EXTERNAL_REVIEW_UNAVAILABLE` until the
configured reviewer returns a signed decision binding the exact packet.
Consciousness, alignment, and production remain separately gated.
