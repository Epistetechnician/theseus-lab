# Replacement-loss V15: capacity versus optimization diagnostic

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v15-capacity-optimization-diagnostic`.

## Question and claim ceiling

V14 showed a relative held-out causal-effect advantage for GELU over the
matched affine candidate, but neither candidate passed absolute retention.
Its activation-rescue control inserted teacher activations and therefore did
not measure whether a learned replacement could rescue behavior. V15 asks a
narrow diagnostic question: on a fresh, stratified adaptation/validation split,
does increasing replacement rank, optimizer updates, or both reduce the
adaptation objective on held-back synthetic validation examples?

This is a fit diagnostic only. It does not evaluate held-out challenge
interventions, continuation behavior, activation rescue, or model retention.
It cannot establish replacement success or Theseus continuity. The maximum
claim is `LOCAL_ADAPTATION_DIAGNOSTIC_OF_RANK_AND_OPTIMIZATION_FOR_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen design

- Bind the same exact OPT/ReLU checkpoint and layers 0 and 7 as V14. The
  target and layer choice are outcome-aware from earlier work; this is not
  checkpoint or architecture generalization.
- Generate a fresh visible fixture and three fresh adaptation fixtures after
  the V15 packet freeze. Each adaptation fixture has 96 prompts, 32 in each
  of the ordinary, compositional, and adversarial strata.
- Before model capture or fitting, deterministically assign 8 examples per
  stratum to validation and 24 per stratum to fitting. The six seed/layer
  cells are descriptive clusters, not IID observations.
- Fit GELU replacements `Linear(3072, rank) -> GELU -> Linear(rank, 768)` at
  ranks 512 and 768. Keep initialization seed mapping, adaptation rows, batch
  order, AdamW settings, objective, and intervention generator fixed within
  each seed/layer cell.
- Train each rank as one trajectory through 800 updates. Seal checkpoints and
  evaluate fit and validation objectives at initialization, update 200, and
  update 800. This makes the 200-versus-800 comparison a nested continuation
  from the same initialization, not two independent fits.
- The fixed objective is the sum of normalized activation MSE and normalized
  next-logit intervention-effect MSE with unit weights. Use batch size 8,
  learning rate `3e-4`, weight decay `1e-4`, and gradient clipping at 1.0.
- No held-out challenge set is generated in V15. The validation examples are
  adaptation-only synthetic prompts that are excluded from all gradient
  updates. No V14 held-out prompts or outcomes are read for candidate
  selection.

## Diagnostic thresholds and interpretation

The primary diagnostic quantity is the relative reduction in mean validation
objective, averaged descriptively over the six seed/layer cells.

- An optimization signal requires at least a 10% mean validation-objective
  reduction from update 200 to update 800 and a positive reduction in at least
  5 of 6 cells, at either rank.
- A rank-capacity signal requires at least a 10% mean validation-objective
  reduction for rank 768 versus rank 512 at the same update checkpoint and a
  positive reduction in at least 5 of 6 cells, at update 200 or update 800.
- An overfit signal requires at least a 10% mean fitting-objective reduction
  while validation objective rises at least 10%, in at least 5 of 6 cells for
  any paired comparison.
- Report all per-cell metrics and all three signals. If multiple signals
  occur, report them together. A signal that misses these cutoffs is
  `NO_CLEAR_SIGNAL`, not evidence that the factor has no effect.

These cutoffs classify this small diagnostic; they are not significance
thresholds or a replacement-retention gate. The oracle teacher-target values
are recorded only as an evaluator integrity reference. The former V14 teacher
activation insertion is not called learned activation rescue.

## Variance, stopping, resources, and custody

Variance sources include fixture seed, prompt template, activation scale,
paired initialization, and runtime numeric behavior. Fit/validation splits are
stratified by fixture type and fixed by example ID before training. The sole
stopping point is update 800; no adaptive tuning or rerun is permitted inside
an immutable custody root.

Execution is local CPU-only, one Torch thread, deterministic algorithms, a
four-hour wall-clock cap, 16-GiB RSS cap, 200-million base-model parameter
cap, and 800-MB checkpoint cap. No provider, paid, cloud, remote-worker, GMan,
real-user, or public-write activity is allowed. Cost is `NOT_APPLICABLE`;
energy is `NOT_MEASURED`.

The frozen packet is archived before fixture generation. Synthetic prompts,
activations, candidate tensors, splits, and metric rows remain in external
owner custody. The repository stores protocol, code, lockfile, freeze record,
and aggregate diagnostic results only. The primary and replay use separate
pinned Python runtimes on the same owner/host; exact replay is at most
`REPRODUCED_LOCAL`, not independent science.

## Next gate

Only a diagnostic with a clear, reproducible factor signal justifies selecting
a configuration for a new packet. That packet must freeze a learned, non-oracle
replacement-rescue measure and fresh held-out causal and behavior evaluations
before creating challenge fixtures. Independent science remains gated on a
blind replicator and an authorized independent reviewer. Alignment,
consciousness, and production are separate gates.
