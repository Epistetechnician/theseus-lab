# Replacement-loss V35: donor endpoints, secants, and prefix Jacobians

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v35-secant-jacobian-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus frozen source packet.
Monetary cap $0. Preserve V34's failure and its separate reproduced
adaptation-only diagnosis; never repair a predecessor identity.

## Motivation, hypotheses, and outcome contract

The V34 diagnostic met its predeclared continuation rule on all three guides:
weak coordinate-secant errors tracked effect errors, prefix-Jacobian mismatch
was substantial despite small activation error, and the shared first-stage
suffix contributed zero error. This is an adaptation-only association and
decomposition, not a heldout result or a proven cause of training failure.
V35 freeze verifies both diagnostic receipts, source, input manifest, archive
manifests, reconciliation, and closeout validation before any fresh data.

H1: supervising both activation endpoints improves learned interchange
behavior compared with the x-endpoint/global-loss baseline. H2: a separate
coordinate-scaled secant objective improves weak-effect tails beyond endpoint
matching. H3: matching prefix Jacobians adds local function information that
improves unseen finite intervention behavior beyond endpoint matching.
Neither finite secant fitting nor local derivative matching guarantees global
equivalence or crossing-region intervention prediction.

Development: three fresh ReLU guides, declared seeds 35101/35102/35103;
four paired arms in declared order `global`, `paired_endpoints`,
`paired_secant`, `paired_jacobian`. Guides are sampling units. Twenty percent
paired reduction in ordinary first-site p90 errors relative to the endpoint
control is a descriptive improvement of interest, reported by stage and
guide; it is not the selection gate or powered superiority inference.
Variance sources: guide initialization/training, input corpus, candidate
initialization, and sealed challenge entropy. Rows, stages, and reproduction
are not extra independent guide seeds.

Select only an arm passing all unchanged native and control gates at all
stages/strata for all three guides. Use V34's worst normalized gate risk and
declared-order tie break. If none qualifies, stop
`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, reproduce it, and open no confirmation.
No tuning of losses, cap, masks, updates, seeds, selector, or thresholds after
readout. Any guide failing visible qualification terminates the phase before
candidate training or challenge drawing, without seed replacement or retry.

## Models, data, pairing, and training budget

V32 task/model/scorer and V34 map/holdout/control procedures remain frozen
dependencies. Guides: 8→16 stem, two 16→64→16 nonlinear blocks, 16→4 head.
ReLU/GELU/SiLU-gated are activation families of small synthetic feedforward
models, not OPT/GPT-2/Llama checkpoints. Replace complete blocks by newly
initialized 16→96→16 modules. Guide training/qualification and all training
batch, optimizer, clipping, and update counts match V34:

- 4096 guide-training rows, 2400 AdamW updates at .002, batch 256,
  decay .0001, norm clip 1; 1024 visible rows. Sign accuracy ≥.95,
  output normalized RMSE ≤.15, each block removal accuracy drop ≥.20.
- 2048 adaptation and 2048 donor rows; stages 1400/1400/600 AdamW updates
  at .001, batch 128, decay .0001, norm clip 1. Train replaced prefixes;
  final truth-task stage unfreezes all parameters. Guide remains frozen.
- Separate alignment fit/validation sets, 1024 rows each; separate 128-row
  base/donor map-validation pairs in each of four strata.

All paired arms share guide, initialization, corpora, minibatches, masks, and
update counts. Derivative arms incur extra arithmetic, so no FLOP-matched
comparison is claimed. Record runtime/RSS/artifacts rather than claim equal
compute. Namespace every stream using first 12 SHA-256 hex characters of
`protocol/phase/family/declared_seed/partition`, with V34's labels plus fixed
map-validation labels. Original optimizer offsets are unchanged.

All preparation, map-validation, and heldout input rows are exactly globally
disjoint across cases/partitions. Inherit V34's 15 predecessor-file exclusion
and add all seven V34 preparation/map/challenge input files. Diagnostic inputs
are a subset of those excluded V34 preparation files. Confirmation additionally
excludes V35 development input files. No data selected or generated before
V35's code, loss rules, gates, controls, and fixture rules are frozen.

## Objectives and fixed auxiliary cap

Global baseline delegates to the frozen V34 global-loss fit exactly. Its
engineering ladder must reproduce identical parent parameter bytes. All
global-normalized losses use MSE divided by target mean square, floor 1e-8.
Baseline output loss uses guide(x), activation loss averages active guide(x)
states, causal loss averages active global interchange effects. Final stage
adds truth MSE. Teacher adaptation outputs/effects only; no challenge targets.

The three paired arms have a shared base objective: baseline output/global
effect losses, but activation loss is the mean of x and donor endpoint losses,
averaged across active sites. This preserves the activation term's overall
weight while adding donor supervision. Final stage adds truth MSE.

`paired_endpoints` adds no auxiliary information. `paired_secant` adds the
mean over rows, coordinates, and active sites of
`(candidate_delta - guide_delta)^2 / coordinate_scale`, where delta is full
donor−x activation displacement, and each coordinate scale is guide delta's
mean square over the entire fixed adaptation pairing, floor 1e-6. Scale is
frozen adaptation-only, not fitted from challenge effects. This controls
coordinate scale; it does not target-select difficult heldout rows.

`paired_jacobian` adds the mean globally normalized prefix-Jacobian MSE at x
and donor, across active sites. A prefix Jacobian includes stem and all blocks
up to that site, differentiating activation with respect to the 8-D input.
Exact analytic adapters from the frozen diagnostic remain differentiable
with respect to candidate weights, using ReLU's zero subgradient at zero,
exact GELU derivative, and SiLU-gated product rule. Teacher Jacobians use only
adaptation/donor inputs. No finite-difference tuning or perseed loss changes.

Every paired update measures detached float64 Euclidean gradient norms gB
for the complete base objective and gA for the auxiliary objective across
all trainable parameters. Secant/Jacobian coefficient is fixed by
`min(1, .5*gB/max(gA,1e-12))`, detached before backpropagating base+coefficient*
auxiliary, then combined norm clipping to 1. Require weighted auxiliary norm
≤.5*gB+1e-8 at every update. Endpoint control uses zero auxiliary and c=1,
with identical norm readout calls. Global baseline retains its parent
anchor/causal readouts. Trace role is recorded explicitly; traces always
contain both norms, coefficient, raw ratio, and weighted ratio.
This is magnitude control, not a guarantee of gradient alignment or success.

Derivative supervision has established prior art:
[Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html),
[Sobolev training](https://arxiv.org/abs/1706.04859), and causal interchange
supervision in [Causal Distillation](https://aclanthology.org/2022.naacl-main.318/).
This is a bounded protocol comparison, with no novelty/priority outcome.

## Sealed evaluation, maps, and unchanged gates

V34's affine map fit and adaptation-only causal qualification apply uniformly
to every arm/site/snapshot, before fresh challenge entropy. Representation,
invertibility, adaptation causal effect, mapped activation, actual restoration,
and restoration-prediction gates must all pass. Any unqualified active map
means `ALIGNMENT_UNAVAILABLE`, with no aligned prediction or score. Native
retention alone determines selection; comparator/mapping absence is inconclusive.

After candidates/maps freeze, draw fresh 32-byte OS entropy. Four challenge
strata each contain 256 base/donor pairs: ordinary, input composition,
interchange composition, adversarial. Masks use 1/2/4/8 coordinates;
two-coordinate input counterfactuals; ordinary bounds [-1,1], adversarial
[-1.4,1.4]. Prebind every native/qualified-aligned prediction and random,
zero-module, wrong-module control before guide challenge outcomes or actual
guide-suffix rescue evaluation. Shared paired challenges are not extra seeds.

Every snapshot and stratum requires output normalized RMSE ≤.10, guide sign
agreement ≥.95, truth accuracy drop ≤.03; input and active internal effects
resolved fraction ≥.90 (reference norm ≥1e-4), median relative error ≤.20,
p90 ≤.50, median cosine ≥.90; activation RMSE ≤.15, actual rescue RMSE ≤.10,
prebound rescue-prediction RMSE ≤.10. Positive/no-op guide control must pass.
At block two, each negative control's input-effect median error must exceed
treatment by ≥.10 in every stratum. Early-stage failures remain disqualifying.

## Conditional confirmation, reproduction, and custody

Only eligible and reproduced development may freeze selected arm plus exact
primary/replica/reconciliation digests before confirmation data. Nine fresh
cases: three activation families × seeds 35201/35202/35203. ReLU guide uses
ReLU replacement; GELU/SiLU-gated guide uses GELU replacement. Six arms:
selected method, global causal baseline, paired-endpoint control, pointwise,
output distillation, and CKA. All retain equal update counts. The CKA arm is
V34's procedural comparator, not a full Network of Theseus reproduction.
Confirmation PASS requires every complete selected-method case; report all
comparators and mapping availability separately. No aggregate superiority gate.

Fresh primary/replica virtual environments: Python 3.12.13, exact lock,
deterministic one-thread CPU, float32 training, float64 maps/norm sums.
Per phase/runtime: 1800 seconds, 4 GiB RSS, 1 GiB final artifacts. Replica
reconstructs candidates/maps before reading primary entropy, then seals its
predictions before oracle evaluation. Require exact canonical fixtures,
weights, maps, qualifications, traces, predictions, outcomes, and metrics,
equal toolchain/platform and distinct prefixes/site-packages. This is same
host/owner `REPRODUCED_LOCAL`, never external blind replication.

Frozen closeout checks manifest/source archive hashes, global row disjointness,
all reconciliation checks, gradient caps, native numeric scores, and actual
guide-suffix restoration from saved predictions. It writes outside terminal
roots. Retain any source/resource/reproduction fault as `INCONCLUSIVE`; no
root reuse or scientific retry. Engineering failures fixed before freeze
are recorded separately. Monetary cost $0; energy/amortization unmeasured.

Raw fixtures, models, targets, maps, traces, entropy, predictions, outcomes,
and resource receipts stay in external owner custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v35`.
Runtimes stay under `/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v35`.
Repo holds code, tests, protocol, exclusions/freezes, scalar/digest reports.
No model download, provider/remote worker, paid run, public publication,
review submission, commit/merge, production integration, or credential storage.

Claim ceiling: `LOCAL_SYNTHETIC_ENDPOINT_SECANT_AND_JACOBIAN_SUPERVISION_ONLY`.
Reviewer `NOT_SUBMITTED`. Nonclaims: all architectures, pretrained transfer,
metaphysical identity, consciousness, alignment, novelty, independent science,
production. Completing and reproducing a failure fulfills development's
stopping rule, not the practical Theseus proof objective. A qualifying method
would open only the predeclared confirmation procedure. Pretrained transfer,
blind external replication, human calibration, and production authorization
still require their separately scoped protocols and authorities.
