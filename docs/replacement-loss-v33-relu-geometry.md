# Replacement-loss V33: ReLU geometry and effect weighting

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v33-relu-geometry-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the digest-bound packet.
This new local study preserves V32's frozen FAIL. Monetary cap: $0.

## Hypotheses, units, and selection

H1: matching a ReLU guide's piecewise-linear replacement geometry improves
small intervention effects and adversarial restoration. H2: per-example
effect normalization improves relative-effect tails compared with global
normalization. Development is a paired 2×2, three fresh ReLU guides with seeds
33101/33102/33103, and four variants: `gelu_global`, `relu_global`, `gelu_row`,
`relu_row`. Replacement blocks have 96 hidden units, identical parameter
counts, and identical initial weights, minibatches, and update budgets.
The guide seed is the sampling unit; 256 rows are not independent model
replicates. Factor differences and a 10% risk reduction are descriptive
diagnostics, not a statistical superiority gate.

Global effect loss is squared error divided by global target mean square,
floor 1e-8, as in V32. Row effect loss averages squared vector errors divided
by each target's squared vector norm, floor 1e-6. Output and activation losses
remain globally normalized. Every variant replaces block one, block two,
then trains the final task. Guides, task, batch sizes, optimizers, updates,
gradient clipping, and intervention masks otherwise match V32 exactly.

Select only a variant passing all native retention, positive-control, and
negative-control gates for all three development guides, snapshots, and
strata. Rank eligible variants by their worst normalized gate risk:
output RMSE/.10, activation RMSE/.15, actual rescue and rescue-prediction
RMSE/.10, effect median/.20, effect p90/.50, and .90/effect median cosine,
over input and active internal effects. Ties use the declared variant order.
Missing/unresolved effects are ineligible. If none qualifies, terminate as
`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, reproduce that outcome, and generate
no confirmation guides or challenges. There is no adaptive retry or tuning.

## Fixed data and training procedure

The immutable V32 runner supplies the task and numeric scorer. These are small
trained feedforward guides, not OPT, GPT-2, or Llama checkpoints. Each guide
has an 8→16 stem, two 16→64→16 blocks, and a 16→4 head. ReLU/GELU/SiLU-gated
are activation families. Full nonlinear blocks are removed and replaced.

Guide training: 4096 rows, 2400 AdamW updates, learning rate .002,
weight decay .0001, batch 256, gradient norm clip 1. Visible qualification:
1024 disjoint rows, guide sign accuracy ≥.95, normalized RMSE ≤.15, and
removing either block lowers sign accuracy ≥.20. Unqualified guides stop the
phase before fitting replacements or drawing challenges.

Adaptation and donor sets: 2048 rows each. Replacement stages: 1400 updates
each, final task stage: 600 updates; AdamW .001, weight decay .0001, batch
128, clip 1. Train the replaced prefix at each replacement stage; final
task training unfreezes all parameters. Training uses single-coordinate
activation interchanges, seed+14 masks. Teacher outputs and effects are
adaptation-only. Preparation seeds are case seed+10/11/12/13 for guide
train/visible/adapt/donor; alignment fit/validation use seed+20/21, 1024 each.
All inputs are independently uniform [-1,1] float32 procedural vectors.

Exact float32 row bytes must be disjoint between all preparation partitions,
all cases, and challenges. A prefreeze exclusion manifest binds V32 primary
preparation/challenge input artifacts; new rows must not overlap those inputs.
Confirmation also excludes development preparation/challenge rows. Replica
reconstructs identical rows, never counts them as additional observations.

## Confirmation, conditional on development eligibility

Only after development reproduces in a second isolated runtime may a
confirmation receipt freeze its selected variant and exact development
result/reconciliation digests, before confirmation data generation.
Nine fresh cases: three families × seeds 33201/33202/33203, with family index
times 10000 added to case seed. No checkpoint or guide screening beyond the
fixed visible qualification; no replacement or retry for a failed seed.

The selected geometry is ReLU for ReLU guides if the chosen variant uses
ReLU, and GELU otherwise. GELU/SiLU-gated guides always use GELU replacements,
retaining V32's successful conversion rule for those families. Selected
effect weighting applies to all families. Five equal-budget paired arms:
selected causal, original GELU/global causal, pointwise activation/output,
output distillation, and staged CKA followed by task training. The latter
three use selected geometry. CKA uses row-normalized centered linear CKA,
as in [Network of Theseus](https://arxiv.org/html/2512.04198v1); this is a
small procedural comparison, not reproduction of its benchmark. Interchange
supervision overlaps [Causal Distillation](https://aclanthology.org/2022.naacl-main.318/).
Novelty/priority is not an outcome of this study.

## Adaptation-only coordinate alignment

Fit a full affine candidate→guide coordinate map independently per active
site, arm, and snapshot on alignment-fit inputs, using centered float64 ridge
regression, ridge 1e-6 after division by row count. Freeze map, inverse, and
qualification before challenge entropy is drawn. Separate alignment-validation
inputs must yield normalized activation RMSE ≤.15, condition number ≤1e4,
inverse roundtrip normalized RMSE ≤1e-4, and all finite arrays. No pseudoinverse
fallback, map tuning, guide evaluation on challenge rows, or heldout fitting.

For qualified maps, map base/donor candidate activations into guide coordinates,
apply the same challenge coordinate mask, invert the result, and run the
candidate suffix. Prebind these effects and mapped activation restoration
predictions before guide evaluation. Evaluate actual restoration in the guide
suffix afterward. Apply mapping uniformly to every arm. If any active map is
unqualified, label that arm/snapshot `ALIGNMENT_UNAVAILABLE` and do not create
an aligned score for it. Unavailable comparisons are inconclusive, not zero
effects or proof of comparator failure. Native gates remain the primary
retention contract; aligned gates are a separate auxiliary comparison.

## Sealing, outcomes, controls, stopping rules

After every guide, candidate snapshot, map, and qualification is frozen,
primary draws fresh 32-byte OS entropy. Four challenge strata each have
256 base/donor pairs: ordinary, input composition, interchange composition,
adversarial. Activation masks affect 1/2/4/8 coordinates, respectively.
Inputs use [-1,1], adversarial [-1.4,1.4]; counterfactuals replace two input
coordinates. All native and qualified aligned predictions and random,
zero-module, and wrong-module control predictions are saved and digest-bound
before guide outcomes or actual guide-suffix restoration are evaluated.

Unchanged V32 gates, every snapshot/stratum: output normalized RMSE ≤.10,
guide sign agreement ≥.95, truth accuracy drop ≤.03; input/active internal
effects with ≥.90 resolved fraction (reference norm ≥1e-4), median relative
error ≤.20, p90 ≤.50, median cosine ≥.90; activation RMSE ≤.15, actual learned
rescue RMSE ≤.10, prebound rescue-prediction RMSE ≤.10. Guide self-prediction
is a positive/no-op control. Each of random/zero/swapped-block controls must
have input-effect median error at least .10 worse than treatment after block
two, in every stratum. Controls are recorded independently of qualification.
Confirmation PASS requires all nine selected-causal complete cases. Comparator
passing and mapping coverage are reported separately; raw-coordinate CKA
failure is not a superiority claim.

Budgets per phase/runtime: 1800 seconds, 4 GiB peak RSS, 1 GiB final artifacts,
one deterministic CPU thread, float32 training and float64 map fitting,
Python 3.12.13 and exact locked packages. Check training budgets every 50
updates and phase boundaries. Monetary cost $0; energy not measured. Resource,
freeze, disjointness, or reconciliation faults are `INCONCLUSIVE`; retain
partial receipts. Terminal custody roots must never be reused.

Fresh primary and replica virtual environments have separate prefixes and
site-packages. Replica rebuilds preparation and candidates before reading
primary entropy, then recomputes and seals predictions before outcomes.
Compare all fixture/model/map/prediction/outcome hashes and canonical metrics;
require equal toolchain/platform plus distinct runtimes. This is same-owner
`REPRODUCED_LOCAL`, never independent acceptance or external blind review.

## Custody, claim ceiling, and completion

Raw fixtures, entropy, weights, maps, predictions, oracle outcomes, runtime
and resource receipts stay in owner-controlled external custody under the
V33 artifact root. Repository receives only code, protocol, exclusion digests,
freezes, aggregate reports, and engineering tests. Archive the exact frozen
packet before data generation. No provider/remote calls, publication, commit,
merge, production traffic, credential storage, or model download is involved.

Claim ceiling: `LOCAL_SYNTHETIC_RELU_GEOMETRY_WEIGHTING_AND_CAUSAL_LADDER_ONLY`.
Nonclaims: pretrained Transformer replacement, all architectures, universal
identity, consciousness, alignment, novelty priority, independent science,
production readiness. Reviewer `NOT_SUBMITTED`; external authority not needed
for this bounded local phase. Frozen input byte verification and separate
runtime reproducibility complete this authorized scope, including a reproduced
development failure if no candidate qualifies. Pretrained transfer needs a
new task/corpus protocol; independent scientific acceptance, human calibration,
and production authorization remain separate gates.
