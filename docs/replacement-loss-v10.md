# Replacement-loss v10: OPT causal-fit follow-up

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v10-opt-causal-fit-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Question and claim ceiling

V9 passed activation, behavior, and module-boundary retention, but its OPT/ReLU
ridge replacements missed the held-out next-logit causal-effect gate at visible
layer selections 0 and 7. V10 asks whether fitting a fresh affine replacement
with an adaptation-only next-logit intervention-effect loss improves held-out
causal-effect prediction over an activation-only ridge replacement on one
freshly selected OPT/ReLU checkpoint.

The maximum positive claim is:

`LOCAL_CAUSAL_LOSS_IMPROVEMENT_FOR_TWO_VISIBLE_SELECTED_OPT_PROJECTIONS_ON_ONE_CHECKPOINT_AND_THREE_SYNTHETIC_SEEDS`.

This packet cannot establish architecture-general validity, mechanism in
general, novelty priority, whole-model identity, consciousness, alignment,
human equivalence, independent science, or production readiness. A completed
scientific outcome is reported separately from its assurance level.

## Frozen design

The packet and runner are frozen before checkpoint identity selection. The
checkpoint must be a base OPT causal language model with ReLU MLP activation,
at least three layers, at most 200 million parameters, no remote custom code,
and a fixed 40-character Hub commit revision. The selected Hub entry must be
public and ungated, its model-card license metadata must match the recorded
license identifier, and its pinned card must not declare an upstream
`base_model`. The operator also checks that the revision-pinned card describes
a base checkpoint; that self-description does not independently verify model
training history. The runner records the card URL, card digest, and parsed
public metadata. Its exact model identity and weights must not appear in the
prior checkpoint inventory or V9 target list. Selection occurs after freeze by
applying these rules to public model metadata; if no eligible fresh checkpoint
can be verified, the run is `INCONCLUSIVE`. Checkpoint files stay in owner
custody outside this repository.

Freshness screening binds the raw SHA-256 of the V9 prior-identity inventory
`64b83b0f8458b234db87028b3495a62370e550dd0268e4d072b099d9d600633f` and the
V9 checkpoint manifest digest
`b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277`. It
rejects matching checkpoint IDs and matching weight-file byte digests before
the checkpoint-selection manifest is sealed.

The experiment has one model family, three fixture seeds, two independently
selected layers, and three strata. A fixed visible fixture of 32 synthetic
prompts selects the maximum mean input variance in each of the first two depth
thirds; ties select the lower layer index. The final depth third is not tested.
Layer choice uses visible prompts only. The two selected layers are evaluated
separately against the original model, without a cumulative replacement chain.

Each seed has 96 disjoint adaptation prompts (32 ordinary, 32 compositional,
32 adversarial) and 60 held-out challenge prompts (20 per stratum). Adaptation
interventions perturb one deterministically selected input coordinate by one
percent of that seed/layer's adaptation input RMS. Challenge prompts and
interventions are generated from 256-bit OS entropy only after both candidate
methods have been trained and their state digests frozen. The entropy, prompts,
and all run artifacts remain in external owner custody; the repository stores
only generation rules and aggregate results.

Ordinary and compositional challenge interventions use a secret-seeded sparse
coordinate. Adversarial-stratum interventions use the input coordinate with
the greatest variance in that seed/layer's adaptation activations; its sign is
secret-seeded. This adaptation-selected stress direction is frozen before
challenge generation and does not inspect held-out activations or outcomes.
Challenge intervention magnitude is one percent of the selected layer's input
RMS measured on the visible fixture.

Both candidates are new affine modules initialized without reading source
weights. They use the same selected subsystem and adaptation set:

1. `activation_ridge` is a closed-form affine ridge fit to adaptation input and
   output activations (`ridge=1e-6`).
2. `causal_fit` starts from a seeded, independently initialized affine module
   and runs exactly 100 AdamW updates (`lr=1e-3`, `weight_decay=0`, batch size
   8). Each update minimizes normalized activation MSE plus normalized
   next-logit intervention-effect MSE. The activation term is divided by the
   adaptation target mean square (floor `1e-6`). Each causal row is divided by
   the teacher effect RMS (floor `1e-5`). Both terms have weight 1.0. The batch
   order is generated from the candidate seed and cycles deterministically.

Candidate weights and predictions are sealed before teacher challenge
outcomes are computed. Predictions include next-logit causal effects,
replacement-boundary activations, baseline next-token logits, and continuation
cross-entropy. After prediction sealing, an evaluator-side activation-rescue
check replaces every held-out candidate boundary output with the original
projection output and checks that next-token logits and continuation loss
return to the original model within `1e-6`. Controls are a randomly initialized
replacement, a replacement trained for the other selected layer, and a
zero-intervention no-op check. Controls are diagnostic and do not enter the
primary comparison.

## Outcomes and decision rule

The primary comparison is per-seed mean held-out causal-effect cosine for
`causal_fit` minus `activation_ridge`, averaged across the two layers and three
strata. Bootstrap resampling is clustered by fixture seed, uses 10,000 draws,
and a fixed seed. Comparative success requires the lower 95% bootstrap bound
to exceed `0.05`.

Per-example causal-effect error is normalized by the norm of the teacher's
baseline next-token logits; effects below `1e-12` are marked unresolvable.

The causal-fit retention gates, checked in every seed/layer/stratum, are:

- at least 20 resolvable effects, direction accuracy at least `0.80`, median
  cosine at least `0.90`, and median normalized error at most `0.20`;
- activation median cosine at least `0.90` and normalized error at most
  `0.20`;
- maximum normalized next-token logit error at most `0.05` and continuation
  cross-entropy increase at most `0.05` nats per token.

Activation rescue must also recover original next-token logits and
continuation cross-entropy within `1e-6` for every seed/layer/stratum. This is
an evaluator-side mediation check run after candidate predictions are sealed.

The scientific outcome is `PASS` only if the comparative gate and all causal-
fit retention gates pass. A complete valid run that misses any gate is `FAIL`.
Missing or corrupted frozen inputs, a checkpoint identity mismatch, or a run
that cannot complete within its fixed budget is `INCONCLUSIVE`. No threshold,
seed, fixture, or target may be changed after freeze to obtain a preferred
outcome.

## Execution limits and custody

Execution is local CPU only: one thread, maximum 4 hours wall time, maximum
16 GiB resident memory, maximum 200 million model parameters, and maximum 800
MB checkpoint weights. No paid compute, cloud node, remote worker, provider
call, real-user prompt, or public write is in scope. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`. Generated prompts, tokenizer IDs,
weights, candidates, interventions, predictions, and row-level outcomes stay
outside the repository under owner custody.

The packet runs in three digest-checked phases: `prepare` selects visible
layers, captures adaptation data, fits both candidates, and freezes their
digests; `predict` creates post-freeze challenges and seals candidate/control
predictions; `score` then computes original-model held-out outcomes and the
predeclared decision. A clean local rerun may use the same owner-custodied
entropy and checkpoint, but remains `REPRODUCED_LOCAL`, not independent
acceptance.

## Assurance and reporting

The final report records the exact source revision and packet digest, state
slices, assurance level, scientific outcome, reviewer status, reproduction
status, checkpoint identity and custody, artifact digests, resource use,
limitations, claim ceiling, and next blocked authority gate. The default
reviewer state is `NOT_SUBMITTED`. `INTERNAL_PASS` describes local software and
protocol checks only; it does not override a scientific `FAIL`. This work
cannot produce `INDEPENDENT_ACCEPT` or `PRODUCTION_AUTHORIZED`.
