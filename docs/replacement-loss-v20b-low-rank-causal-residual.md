# Replacement-loss V20B continuation: low-rank causal residual

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v20b-low-rank-causal-residual-continuation-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol IDs: `theseus-replacement-loss-v20b-development` and
`theseus-replacement-loss-v20b-confirmation`.

## Why this continuation exists

The frozen V20 runner stopped before completing its first development
preparation. It passed V16's `NoOpCandidate` to a direct-call metric helper,
although that inherited control implements only the projection-patching path
used by the model scorer. The exact terminal receipt remains at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v20-development-primary/run-stop.json`.
This continuation binds that receipt and V20's frozen packet, adds a zero-output
adapter for the direct-call control metric, and uses fresh fixture seeds. It
does not reuse V20 prompts, captures, candidates, or partial outputs. The
low-rank method, optimizer, thresholds, strata, and claim ceiling are unchanged.

## Frozen method and gates

The V20 protocol's low-rank method and qualification rules apply unchanged:
train an independent pointwise whole-MLP base; freeze it; train rank-16
residual factors on both affine projections with the predeclared finite-effect
plus base-drift objective; merge and verify the dense replacement; then score
the nine seed-by-stratum cells. Each cell must meet effect ratio `<=0.90`,
pointwise value ratio `<=1.25`, continuation cross-entropy increase `<=0.05`,
and normalized next-logit error `<=0.05`. See the V20 protocol bound by this
packet for the complete training and custody specification.

Development uses seeds `202610811`–`202610813`; confirmation uses
`202610911`–`202610913`. V20B fixture partitions are independently namespaced
and validated against prior visible/adaptation data, including the aborted
V20 development artifacts. Confirmation follows the same two-runtime
prechallenge qualification and entropy-unlock gate as V20.

The zero-output adapter only gives the no-op control the direct-call interface
required by the V20 cell-metric helper. In model evaluation, the inherited
zero-expansion/zero-output projections remain the no-op control. The adapter
does not alter treatment candidates or thresholds.

## Claim ceiling and assurance

Maximum claim ceiling:
`LOCAL_LOW_RANK_CAUSAL_RESIDUAL_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
The checkpoint remains the previously selected OPT/ReLU model at block 8.
V20B does not test a fresh checkpoint or another architecture. It does not
establish identity, consciousness, alignment, production readiness, or
independent scientific acceptance. A local second-runtime replay is
`REPRODUCED_LOCAL`, not blind external replication.

All prompt-bearing and row-level data remain on external owner custody. If any
gate fails, the run stops before the confirmation challenge; no challenge
entropy is drawn.
