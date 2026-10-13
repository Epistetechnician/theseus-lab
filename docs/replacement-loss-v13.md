# Replacement-loss V13: nonlinear causal replacement on known OPT

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v13-nonlinear-causal-replacement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v13-nonlinear-causal-replacement`.

## Question and claim ceiling

V11 already initialized a new affine projection without source weights, trained
it on disjoint adaptation examples with an intervention-effect loss, and
sealed candidate predictions before held-out teacher outcomes were opened. Its
causal-fit comparison failed. V13 does not repeat that experiment or describe
the affine method as untried. It tests a distinct replacement class: a
freshly initialized two-layer GELU MLP trained on adaptation data.

The fixed target is the V11 OPT/ReLU checkpoint
`grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
Layers 0 and 7 are fixed from the previously reported V9 causal-effect misses.
This is an intentional, outcome-aware follow-up on one known checkpoint; it
does not test fresh-target generalization or an unbiased sample of layers.
Checkpoint bytes remain in the existing external V9 custody.

The primary hypothesis is that the nonlinear `causal_fit` replacement improves
held-out next-logit intervention-effect cosine over `activation_ridge`. For
each of six paired fixture seeds, first average across layers 0 and 7 and the
three frozen prompt strata. The primary effect is the mean seed-level
difference. Success requires the deterministic 10,000-draw seed-cluster
bootstrap 95% lower bound to exceed `0.05`.

The maximum positive claim is
`LOCAL_NONLINEAR_CAUSAL_REPLACEMENT_IMPROVEMENT_OVER_ACTIVATION_RIDGE_ON_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS`.
The comparison does not isolate nonlinearity from the combined training
objective, optimizer, or parameterization. A positive result would not
establish independent science, general mechanistic understanding, architecture
generality, novelty priority, consciousness, alignment, human equivalence, or
production readiness.

## Frozen design

- One reused OPT/ReLU checkpoint; fixed projections `fc2` at layers 0 and 7.
- Six fixture seeds: `202609401` through `202609406`.
- A 32-prompt visible fixture per run for input-RMS and stress-scale
  calculation; it does not select the target layers.
- Per seed, 96 adaptation prompts (32 each ordinary, compositional, and
  adversarial) and 60 held-out challenge prompts (20 per stratum).
- V13-specific namespaced prompt IDs and text prefixes. Exact IDs and prompt
  contents must be disjoint across visible, adaptation, and challenge
  partitions.
- Target interventions perturb one fixed, seeded input coordinate by one
  percent of visible input RMS for ordinary and compositional cases. The
  adversarial stress coordinate is selected from adaptation input variance;
  challenge outcomes and activations are not used to select it.
- `activation_ridge` is the same-seed closed-form affine activation fit with
  ridge `1e-6`.
- `causal_fit` is a newly initialized `in -> 512 -> out` module with GELU and
  no source weights. At the predeclared OPT `fc2` width of 3072 by 768 it has
  1,967,360 parameters, fewer than the original projection's 2,360,064.
  It uses AdamW for exactly 200 updates, batch size 8, learning rate `3e-4`,
  weight decay `1e-4`, and gradient-norm clipping at 1.0. The loss is the sum
  of normalized activation MSE and normalized next-logit intervention-effect
  MSE, each with weight 1.0. The candidate seed and batch order are
  deterministic and bound to protocol, fixture seed, and layer.
- No tuning, early stopping, threshold changes, or prompt edits are allowed
  after freeze.

Candidate states are frozen before fresh challenge entropy is generated.
Candidate predictions, including intervention-effect predictions and
behavioral outputs, are sealed before the evaluator computes held-out teacher
outcomes. The score phase verifies the prediction and challenge digests before
opening teacher outputs.

## Outcomes and controls

The primary metric and threshold are specified above. Seed is the bootstrap
cluster; prompts, strata, and layers are not treated as independent samples.
The fixed bootstrap seed is `202609399`. Main variance sources are fixture
seed, candidate initialization and batch order, prompt-template coverage, and
numeric runtime behavior.

The frozen per-cell retention requirements for `causal_fit` inherit the V10
gates: at least 20 resolvable effects, direction accuracy at least `0.80`,
median causal-effect cosine at least `0.90`, median normalized causal error at
most `0.20`, activation cosine at least `0.90`, activation normalized error
at most `0.20`, maximum normalized next-token logit error at most `0.05`, and
continuation cross-entropy increase at most `0.05` nats/token. Activation
rescue must restore teacher next-token logits and continuation loss within
`1e-6` in every seed/layer/stratum. A zero-intervention control must have
maximum normalized logit error at most `1e-6`.

Controls are the activation-ridge comparator, seeded random replacement,
wrong-layer replacement, and zero-intervention check. A complete valid run is
`PASS` only if the primary threshold and every causal-fit retention and rescue
gate pass; otherwise it is `FAIL`. Missing/corrupt frozen inputs, invalid
custody, resource-limit breach, or incomplete phases are `INCONCLUSIVE`.

The V11 affine causal-fit result remains historical context only. It is not a
paired V13 baseline because it used different fixtures. Therefore a V13 pass
does not prove that the nonlinear architecture itself caused the improvement;
a later same-fixture, same-objective affine-versus-nonlinear ablation would be
needed for that narrower causal claim.

## Procedure and custody

1. Freeze this protocol, V13 fixture generator, runner, focused tests,
   dependency lock, and exact source-code dependencies before generating any
   V13 prompt fixtures or candidates.
2. Archive the packet to two new external custody roots. Bind the already
   frozen V11 target-selection receipt; do not download or copy weights.
3. Run preparation in the primary pinned local runtime. Generate visible and
   adaptation fixtures, train and freeze both candidates for six seeds and two
   layers, and record all phase/resource receipts.
4. Only after candidate freeze, generate 256-bit challenge entropy and
   held-out prompts. Seal predictions before scoring teacher outcomes.
5. Run the exact packet in the separate pinned local runtime, using the
   primary's frozen challenge entropy. Reconcile content identities while
   recording runtime paths separately. Same-owner reproduction is
   `REPRODUCED_LOCAL`, not blind replication or independent acceptance.
6. Record the literal scientific outcome. Preserve all stop/failure receipts;
   do not retry a partially written immutable phase in the same custody root.

Execution is local CPU-only, one thread, deterministic algorithms enabled,
maximum four hours wall time, 16 GiB RSS, 200 million base-model parameters,
and 800 MB checkpoint weights. No provider call, paid or remote worker, GMan
node, real-user input, or public write is authorized by this packet. Prompts,
checkpoint files, captures, candidates, entropy, predictions, and row-level
outcomes stay in external owner custody. The repository stores code, protocol,
freeze record, aggregate results, and digests only. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`.

Reviewer status is `NOT_SUBMITTED`. The separate blind-replication and
independent-review gates remain open. This study cannot produce
`INDEPENDENT_ACCEPT` or `PRODUCTION_AUTHORIZED`.
