# Replacement-loss v9: causal replacement ladder comparison

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v9-causal-ladder`.

Status: preselection preregistration. Freeze this document, the runner,
fixture-generation rules, focused tests, dependency pins, pinned v3 helper
bytes, and exact v8-v3 exclusion packet before selecting any v9 checkpoint.
v3 through v8 remain immutable.
Checkpoint identities and challenge fixtures must not be selected or generated
until the packet freeze is recorded in owner custody.

## Prior-art boundary and question

Progressive module replacement is established prior art. BERT-of-Theseus
progressively replaces BERT modules while training compact substitutes
([Xu et al., EMNLP 2020](https://aclanthology.org/2020.emnlp-main.633/)).
Network of Theseus progressively converts architectures by aligning
representations with CKA or differentiable mutual nearest neighbors
([Subramaniam et al., 2025](https://arxiv.org/abs/2512.04198)). This packet
does not claim novelty for splitting additive terms, progressive replacement,
or preserving task scores.

Question: when three MLP output projections are replaced in sequence with
fresh affine modules fit from disjoint adaptation data, does closed-form
activation fitting retain and predict held-out causal effects better than
representation-only CKA fitting and final-logit distillation, while retaining
the guide model's held-out behavior?

The possible contribution is the prospective, held-out causal comparison. A
positive local result is not a priority determination, independent scientific
acceptance, or proof of numerical identity of a whole model.

## Frozen targets and design

After packet freeze, select exactly one eligible public checkpoint in each
family: OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. Each must have an immutable
40-character revision, at most 500 million parameters, an Apache-2.0 or MIT
license permitting local execution and authorized artifact transfer, and a
custodied artifact tree outside this repository. It must not have been used in
any prior Theseus-lab model experiment or protocol design. The frozen v9
runner excludes all v3-v8 targets and earlier recorded Theseus-lab targets; the
custodian also supplies the prior identity inventory used for the exact
freshness check. Record the selection time, license evidence, revision,
artifact-tree digest, weight-file digest, and freshness attestation in an
owner-custodied manifest. Reject a candidate whose weight-file digest matches
an earlier custodied checkpoint identity.
If any family has no eligible checkpoint, stop with
`NO_ELIGIBLE_FRESH_TARGET`.

Use the pinned CPU toolchain: Python 3.12.13, PyTorch 2.6.0, Transformers
4.57.3, NumPy 2.4.6, Safetensors 0.8.0, Tokenizers 0.22.2, and
Hugging Face Hub 0.36.2. No provider calls, paid execution, accelerator
execution, or assessment-corpus data are permitted.
Freeze the complete transitive package set in `requirements-v9.lock.txt`.
Tokenized prompts may not exceed 128 tokens; continuations are truncated only
to fit that same 128-token sequence ceiling and must retain at least one token.

For each checkpoint, use 32 visible prompts to select one MLP output projection
from each of three contiguous depth thirds. Select the projection with the
highest mean per-feature input variance in that third; ties go to the lower
layer index. Replace the selected projections in ascending layer order.
Require at least three transformer layers. The 3,600-second case budget
applies to the complete family-by-seed trajectory across all three methods.

Run paired seeds `202609271`, `202609272`, and `202609273` per checkpoint, for
nine family-by-seed trajectories.
Each seed has 128 adaptation prompts (64 ordinary, 32 compositional, 32
adversarial) and 60 held-out prompts (20 per stratum). The fixture generator
uses a v9 namespace and must verify visible/adaptation/challenge disjointness
against each other and all prior custodied Theseus-lab fixtures. Prompts,
continuations, token IDs, captures, model assets, predictions, and row-level
outcomes stay in owner custody.

Each trajectory independently runs three replacement methods from the same
frozen guide checkpoint and paired fixtures. Each method replaces the same
three selected projections sequentially. At stage `k`, earlier replacements
from that method remain installed; all later projections remain original. The
current projection is fitted using adaptation-only data at that stage's actual
hybrid input distribution. Freeze each candidate before fitting the next stage.
All non-replaced model parameters remain frozen.

| Method | Frozen fitting rule |
| --- | --- |
| `ridge` treatment | Fresh affine projection; closed-form ridge regression on current-stage adaptation input/output activations; ridge `1e-6`; no target parameter copy. |
| `cka` comparator | Fresh affine projection; maximize centered linear CKA between current-stage adaptation outputs and guide projection outputs; AdamW, learning rate `1e-3`, 100 updates, batch 128 activation rows, zero weight decay. This is a NoT-style CKA comparator, not a reproduction of the full NoT method. |
| `logit_distill` comparator | Fresh affine projection; minimize temperature-1 KL divergence between guide and hybrid next-token logits on adaptation prompts; AdamW, learning rate `1e-3`, 32 updates, batch 8 prompts, zero weight decay. |

Initialize learned candidates deterministically from method, family, seed, and
stage. Use single-thread CPU execution. Use the same adaptation examples and
visible-selected layer sequence for all three methods. Do not use held-out
challenge prompts or outcomes for fitting, early stopping, or method selection.

## Held-out predictions and outcomes

Freeze digests for all 81 learned candidate placements (9 trajectories × 3
methods × 3 stages) and the exact candidate forward-prediction rule for each
placement before drawing or revealing v9 challenge entropy. Each rule binds
the candidate state, checkpoint, hybrid-prefix semantics, predictor
implementation, and outputs. After challenge reveal, the frozen rule computes
numeric predictions; source-model intervention outcomes remain unavailable
until every prediction receipt is sealed. Generate challenge intervention
directions after candidate freeze: ordinary Gaussian, compositional sum of two
normalized random directions, and adversarial signed largest-magnitude input
coordinate. Scale each to 1% of the guide activation RMS and share the same
intervention per family, seed, stage, and prompt across methods.

Before opening source outcomes, each method must seal candidate predictions for
all stages and held-out prompts: module-boundary intervention effects,
next-logit intervention effects, candidate next logits, continuation loss, and
candidate projection outputs. Then evaluate the original projection's causal
effects in the same hybrid-prefix context and evaluate cumulative hybrid
behavior against the untouched guide checkpoint. Preserve all prediction and
outcome digests.

For the treatment, both comparators, and random/wrong-module/no-op controls,
report per stage, per family-by-seed, and per prompt stratum. Controls are
evaluated in the ridge trajectory's current hybrid-prefix context. For
module-boundary causal-effect normalized error, divide by the source
projection-output norm for that prompt. For next-logit causal-effect error,
divide by the source next-logit norm. Behavior logit error is the maximum
per-prompt relative next-logit error. Report:

- causal-effect direction accuracy ≥ `0.80`, median cosine ≥ `0.90`, median
  normalized error ≤ `0.20`, and at least 20 resolvable interventions per
  stratum for both module-boundary and next-logit effects;
- cumulative held-out next-logit normalized error ≤ `0.05` and continuation
  cross-entropy increase ≤ `0.05` nats/token, overall and in every stratum;
- held-out projection-output cosine ≥ `0.90` and normalized error ≤ `0.20`,
  overall and in every stratum. This is activation reconstruction, reported
  separately from causal-effect prediction.

Primary hypothesis: `ridge` satisfies every retention gate through all three
replacement stages and exceeds both comparators in mean per-trajectory
next-logit causal-effect median cosine by at least `0.05`. Calculate paired
differences by averaging the three stage scores within each family-by-seed
trajectory. Use a 10,000-draw paired bootstrap over the nine trajectories; the
95% lower bound must exceed `0.05` against each comparator. Use NumPy's linear
quantile interpolation and bootstrap seeds `202609279` for CKA and `202609280`
for logit distillation. Stages and prompts are nested measurements, not
independent bootstrap units. Top-1 agreement is diagnostic only.

Evaluate `random`, `wrong_module`, and `no_op` controls at each stage in the
ridge trajectory's current hybrid-prefix context. Do not run parameter
permutation or symmetry sweeps in this protocol.

Overall result is `PASS` only if all ridge retention gates and both comparative
effect-size gates pass. CKA, logit-distillation, and all control results are
retained even when they miss thresholds. Measured misses are `FAIL`; missing or
overlapping fixtures, digest mismatches, unresolved challenges, unsupported
adapters, or resource termination are `INCONCLUSIVE`. Do not change thresholds,
replace checkpoints, regenerate failed challenge fixtures, or rerun after
inspecting outcomes. Any implementation or threshold change requires a new
frozen packet.

## Reproducibility and cost boundary

Run locally on CPU with 24 GiB maximum process RSS, 3,600 seconds maximum per
family-by-seed trajectory, and 18 hours maximum total primary wall time. Apply
the same limits to the clean second-runtime reproduction. Record wall time,
CPU time, peak RSS, storage, and dependency/source/checkpoint/fixture/output
digests. Monetary cost and energy must be reported `NOT_MEASURED` unless
directly measured. Retain `FAIL`, `INCONCLUSIVE`, and control receipts.

Run the exact frozen packet in a separate clean Python environment on the same
host. Refit candidates independently from the frozen protocol, checkpoint
bytes, visible/adaptation partitions, and toolchain. Only after the replica has
frozen its own candidate states may the primary custodian reveal the held-out
prompt/intervention fixture bytes. Report `REPRODUCED_LOCAL` only if
candidate-state, prediction, outcome, dependency, source, checkpoint, and
fixture digests reproduce exactly. This is same-owner local reproduction, not
independent scientific acceptance.

## Claim ceiling and nonclaims

Even after a complete pass, the claim ceiling is
`LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS`.

Nonclaims: novelty priority, all-architecture validity, whole-model identity,
general mechanistic understanding, independent science, subjective experience,
consciousness, alignment or human equivalence, and production readiness.
External submission is outside this packet; the configured reviewer and
artifact-transfer gates remain separately authorized.
