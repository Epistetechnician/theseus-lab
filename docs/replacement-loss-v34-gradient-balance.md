# Replacement-loss V34: bounded causal gradients and map qualification

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v34-gradient-balance-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the frozen packet.
Monetary cap: $0. Preserve V32 FAIL, V33 INCONCLUSIVE, and V33R1
`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`. No frozen predecessor is retuned.

## Hypotheses and end state

H1: bounding the row-normalized causal gradient relative to the behavior and
activation anchor improves weak-effect prediction while preserving all native
retention gates. V33R1's initial-batch audit found a large gradient imbalance;
it did not establish that imbalance caused the completed training failures.
H2: representation fit and invertibility alone do not qualify a coordinate
map for causal transport. Qualify transport separately on adaptation-only
interventions before emitting aligned scores.

Three fresh ReLU development guides, declared seeds 34101/34102/34103, are
the sampling units. Paired variants: `relu_global`, `relu_row`,
`relu_balanced_row`. They share guide, corpus, replacement initialization,
minibatches, updates, parameter counts, and extra gradient readout work.
Twenty percent paired reduction in ordinary first-block p90 relative-effect
error is the smallest descriptive improvement of interest. It is not a
selection gate or powered statistical superiority claim. Initialization,
guide training, procedural data, and sealed challenge entropy are variance
sources. Rows and stages are correlated measurements, not extra model seeds.

End state: tested and frozen source, primary and separate-runtime development,
conditional confirmation only after reproduced eligibility, digest-verified
custody, aggregate reports, and a literal assurance/result distinction.
No adaptation of thresholds, ratio, data, selectors, or training after results.
No paid run, provider, external review submission, publication, or deployment.

## Models, data, and fixed training

The immutable V32 runner defines the synthetic task, architectures, and numeric
scorer. Guides are 8→16 stems, two 16→64→16 nonlinear blocks, and 16→4 heads.
ReLU, GELU, and SiLU-gated are activation families in these small feedforward
models, not OPT, GPT-2, or Llama checkpoints. Replace complete nonlinear blocks
with independently initialized 16→96→16 modules. Development uses ReLU
replacements. Confirmation uses ReLU replacements for ReLU guides and GELU
replacements for GELU/SiLU-gated guides.

Guide training: 4096 procedural float32 uniform [-1,1] rows, 2400 AdamW
updates, batch 256, learning rate .002, weight decay .0001, norm clip 1.
Visible qualification: 1024 disjoint rows, truth sign accuracy ≥.95,
normalized output RMSE ≤.15, and removing either block reduces sign accuracy
≥.20. Any unqualified guide terminates the phase before replacement fitting
or challenge generation; no guide screening, replacement, or seed retry.

Adaptation and donor sets have 2048 rows each. Two replacement stages each
have 1400 AdamW updates, then 600 final task updates; batch 128, learning rate
.001, weight decay .0001, norm clip 1. Train the replaced prefix at each
stage; final training unfreezes the complete model. Teacher output, activation,
and single-coordinate interchange-effect targets use adaptation data only.
All teacher parameters are frozen. Training masks have a separate stream.

Preparation stream seed is the first 12 hexadecimal characters of SHA-256 of
`protocol_id/phase/family/declared_seed/partition`. Base labels: `train`,
`visible`, `adapt`, `donor`, `align_fit`, `align_validate`, `guide_optimizer`,
`training_masks`. Optimizer offsets retain V33R1's fixed policy. Map validation
has separate labels `map/stratum/x`, `/donor`, `/coordinates`. Exact float32
row bytes are globally disjoint among all cases, preparation partitions,
map-validation endpoints, and challenges. The exclusion manifest binds V32,
V33 partial, and V33R1 preparation/challenge inputs. Confirmation also excludes
V34 development preparation, map-validation, and challenge rows. Replica
reconstructs identical data; it is not a new scientific observation.

## Losses and gradient rule

Global normalization is V32's mean squared error divided by target mean square,
floor 1e-8. Row effect normalization averages squared vector error divided by
each target's squared norm, floor 1e-6. Output and activation losses remain
global. For causal arms the anchor A is global output loss plus mean active
activation loss, with final truth MSE added only in the final task stage.
The causal E is mean active-site effect loss. This preserves V33R1's objective
for the two unscaled arms. Loss addition retains the parent's numeric order:
output plus activation, then scaled effect, then final task MSE. The anchor
gradient includes final task MSE even though that term is added last.

On every causal update, measure detached Euclidean gradient norms across all
currently trainable parameters, using float64 sums of squared gradients:
gA = ||∇A|| and gE = ||∇E||. Additional `autograd.grad` calls retain the graph
and do not accumulate parameter `.grad`. Unscaled arms use c=1. Balanced row
uses the fixed rule c=min(1, .5*gA/max(gE,1e-12)). Backpropagate A+cE, with
the coefficient detached; clip the combined gradient to 1, then update.
Require finite losses/norms and c*gE ≤ .5*gA + 1e-8 on every balanced update.
Store all per-update norms, coefficient, raw ratio, and weighted ratio in
external custody, plus aggregate quantiles and content hashes in the receipt.
This controls gradient magnitude, not gradient direction, cancellation, or
generalization. It is a fixed cap, not a full implementation of
[GradNorm](https://proceedings.mlr.press/v80/chen18a.html).

## Separate map qualification

Fit candidate→guide full affine maps at each active site, arm, and snapshot
on 1024 alignment-fit rows, centered float64 ridge 1e-6 after division by row
count. On 1024 separate alignment-validation rows require finite arrays,
activation normalized RMSE ≤.15, condition number ≤1e4, and inverse roundtrip
normalized RMSE ≤1e-4. No pseudoinverse fallback or map/hyperparameter search.

Generate four adaptation-only map-validation strata before model fitting, with
128 base/donor pairs each. Ordinary/input-composition/interchange-composition/
adversarial mask sizes are 1/2/4/8; inputs [-1,1], adversarial [-1.4,1.4].
Two-coordinate input counterfactuals are generated and excluded from future
data, but map qualification specifically measures internal transport, mapped
activation, and restoration; it does not claim counterfactual input coverage.

Map candidate base/donor activations into guide coordinates, apply the guide
coordinate mask, invert into candidate coordinates, and evaluate the candidate
suffix's effect. Compare to the adaptation guide intervention. At each site
and every stratum require resolved fraction ≥.90 (reference norm ≥1e-4),
median relative effect error ≤.20, p90 ≤.50, median cosine ≥.90, mapped
activation normalized RMSE ≤.15, actual guide-suffix restoration RMSE ≤.10,
and candidate restoration-prediction versus actual restoration RMSE ≤.10.

Qualification is representation AND causal validation. Freeze maps, inverses,
all qualification metrics, and candidates before fresh heldout entropy.
These map queries do not train or select a replacement. Native retention alone
governs development selection. If any active map is unqualified, emit
`ALIGNMENT_UNAVAILABLE` for that snapshot, with no aligned prediction or score.
Unavailable comparisons are inconclusive, not comparator failure. Qualified
maps uniformly transform all arms; no postchallenge map fitting or querying
guide challenge outcomes before prediction freezing.

## Sealed heldout contract, controls, and selection

After candidate/map freeze, primary draws fresh 32-byte OS entropy. Four
heldout strata each have 256 base/donor pairs, 1/2/4/8 coordinate masks, and
two-coordinate input counterfactuals; same bounds as above. The challenge is
shared across paired arms and guide cases. Before any guide challenge outcome
or actual guide-suffix restoration, save and digest-bind all native and
qualified aligned behavior, effects, activations, rescue predictions, and
random/zero-module/wrong-module control predictions. Then evaluate the guide
and actual activation restoration. Guide self-prediction is the positive/no-op
control. No heldout target appears in the replacement training or map fitting.

Unchanged gates at every stage and stratum: output normalized RMSE ≤.10,
guide sign agreement ≥.95, truth accuracy drop ≤.03; input and active internal
effects resolved ≥.90, median relative error ≤.20, p90 ≤.50, median cosine
≥.90; activation RMSE ≤.15, learned rescue RMSE ≤.10, rescue-prediction RMSE
≤.10. After block two, all three negative controls must have input-effect
median error at least .10 worse than treatment in every stratum. Positive
controls must pass. No metric substitution or relaxing weak-effect tails.

Only variants passing all native/control gates for all three guides qualify.
Rank eligible variants by worst gate risk: output/.10, activation/.15,
actual rescue and rescue prediction/.10, effect median/.20, p90/.50, and
.90/cosine, across all stages/strata/input/active internal effects. Missing
or unresolved effects cannot qualify. Ties use the declared variant order.
If none qualifies, terminate `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, reproduce
that outcome, and generate no confirmation data. No retry of terminal roots.

## Conditional confirmation and reproducibility

Only after eligible development reproduces may a second receipt freeze its
selection and exact primary/replica/reconciliation digests before confirmation
data. Nine fresh guide cases: three activation families × seeds
34201/34202/34203. Six paired arms: selected causal, unscaled global causal,
unscaled row causal, pointwise activation/output, output distillation, CKA.
All use the fixed family geometry policy above. Same update counts and
initialization/corpora; causal arms additionally measure gradients, so this is
not a FLOP-matched comparison to noncausal methods. CKA uses V32's fixed
centered row-normalized linear CKA in replacement stages and truth MSE during
final training. This procedural comparison is not a benchmark reproduction of
[Network of Theseus](https://arxiv.org/abs/2512.04198); causal interchange
supervision also has prior art in
[Causal Distillation](https://aclanthology.org/2022.naacl-main.318/).
Confirmation PASS requires all nine complete selected-causal cases. All
comparator native/aligned coverage and outcomes are separately reported.
Novelty, priority, and scientific independence are not outcomes.

Fresh primary and replica Python 3.12.13 virtual environments have distinct
prefixes/site-packages and identical exact locked packages. Deterministic CPU,
one thread, float32 training, float64 maps/norm sums. Replica rebuilds source,
fixtures, guide/candidates/maps/qualifications before reading primary entropy;
then seals identical predictions before outcomes. Reconcile canonical hashes
and metrics plus toolchain identity and distinct runtime paths. Same-owner,
same-host replication can support `REPRODUCED_LOCAL`, not blind external
replication or independent acceptance.

Budgets per phase/runtime: 1800 seconds, 4 GiB peak RSS, 1 GiB final artifacts.
Check training every 50 updates and phase boundaries. Resource, freeze,
disjointness, or reconciliation faults are `INCONCLUSIVE`; retain partial
receipts. Monetary cost $0; energy and machine amortization not measured.

## Custody and report boundary

Raw fixtures, weights, entropy, gradient traces, maps, predictions, outcomes,
runtime/resources, and frozen source archives stay under owner-controlled
external V34 custody. Repo holds code, tests, protocol, exclusions, freezes,
aggregate reports, and scalar/digest evidence only. A frozen closeout utility
independently checks file/archive digests, row disjointness, reconciliation,
every gradient cap, heldout native numeric scores, and actual guide-suffix
restoration from saved predictions. It writes outside terminal run roots and
does not train, retune, or claim independent reviewing authority.
Exact target membership is
the freeze manifest, not every unrelated dirty file in the checkout.

Claim ceiling: `LOCAL_SYNTHETIC_GRADIENT_BALANCE_AND_CAUSAL_MAP_QUALIFICATION_ONLY`.
Reviewer `NOT_SUBMITTED`. Nonclaims: pretrained Transformer transfer, all
architectures, metaphysical identity, consciousness, alignment, novelty,
independent science, production. A reproduced failure completes the authorized
development protocol. Advancement requires a new protocol after failure;
pretrained transfer, independent scientific authority, human calibration, and
production remain separate future gates.
