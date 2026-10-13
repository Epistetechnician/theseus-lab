# Replacement/loss v3: independent replacement and blind causal prediction

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v3`.

Protocol state: preregistration specification. The separate digest manifest
is authoritative for whether these exact bytes have been frozen before fresh
checkpoint selection. The completed v2 algebraic split remains a positive
control, not a v3 treatment or mechanistic-understanding result.

## Claim and hypothesis

**Target claim:** a selected subsystem can be removed and replaced by a
separately constructed, independently trained module that restores held-out
behavior and predicts causal effects of interventions that were hidden from
the proposer until after its predictions were frozen.

**Hypothesis:** across at least three predeclared architecture families and
three seeds per family, the learned replacement will meet the held-out
behavior and intervention-prediction thresholds below, while outperforming
wrong-module, random, and no-op controls on the primary causal-prediction
metric.

The three predeclared families are OPT/ReLU decoder, GPT-2/GELU decoder, and
Llama/SiLU-gated decoder. Use one fresh, pinned checkpoint per family that was
not used in protocol design or threshold selection. If a qualifying fresh
checkpoint is unavailable for any family, the v3 run does not start; do not
substitute a previously studied checkpoint after freezing the packet.

Checkpoint selection is post-freeze and must use public base-model weights,
an exact 40-character source revision, `trust_remote_code=False`, and at most
500 million parameters. The architecture config must match the family and
activation contract. Exclude the previously exercised `gpt2`,
`distilgpt2`, `gpt2-large`, `facebook/opt-125m`, `facebook/opt-1.3b`,
`HuggingFaceTB/SmolLM-135M`, `HuggingFaceTB/SmolLM2-135M`, and
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` checkpoint identities. The checkpoint
manifest records the post-freeze selection time and attests that each exact
identity was not used to design this protocol.

The target subsystem is one layer's MLP output projection: OPT `fc2`, GPT-2
`mlp.c_proj`, or Llama `mlp.down_proj`. Select the layer using only the visible
selection set: choose the projection whose input activations have the greatest
mean per-feature variance, breaking ties by the lower layer index. The
replacement is a newly initialized affine module with the same input/output
dimensions and bias layout. Fit it by closed-form ridge least squares using
only captured input/output pairs from that seed's adaptation prompts, solving
in the smaller primal or dual Gram space. This is supervised projection
distillation from the original projection's adaptation outputs: the target
projection's parameter values are never copied or read by the fitter. This
defines a bounded, independently parameterized projection replacement, not
independent discovery of the mechanism and not a claim about replacing an
entire transformer block.

Passing supports a bounded local mechanistic-prediction result for the exact
frozen target and protocol. It does not establish consciousness, alignment,
general mechanistic understanding, independent science, or production
readiness. The maximum claim ceiling is
`LOCAL_INDEPENDENT_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ONLY`.

## Roles and data separation

- **Proposer:** selects the subsystem using visible selection data, removes
  it, constructs and trains the replacement using only the disjoint
  adaptation set, and commits the candidate, predictor, and deterministic
  prediction code before challenge access.
- **Evaluator:** holds the challenge prompts and interventions, queries the
  committed predictor without exposing outcomes, seals its predictions, then
  runs the original model and controls and scores the committed predictions.
- **Red-team:** specifies compositional and activation-targeted challenge
  classes before freeze. The frozen generator instantiates challenge examples
  from a custodian-held seed after the candidate is frozen; the red-team does
  not tune the replacement or alter generator code post-freeze.
- **Replicator:** a separate process or operator receives the frozen packet,
  including the custodian-supplied adaptation fixture needed to rerun training,
  but without proposer-only tuning notes or held-out outcome access.
- **Governor/operator:** keeps external review, assessment-data use, spend,
  deployment, and publication gates closed unless separately authorized.

The visible selection set, each seed's adaptation set, and the held-out
challenge set must be pairwise disjoint by source identity, exact prompt text,
and exact example digest. Each generated prompt begins with a deterministic
24-hex-character reference derived from its partition, seed, and record index;
the reference prevents accidental prompt reuse across partitions. The fixture
generator is deterministic given a seed. Visible and
adaptation seeds are fixed before training; the challenge seed is drawn from
the custodian's CSPRNG after the candidate packet is frozen and never printed
by the generator. The challenge seed, prompt text, and intervention vectors
stay outside the repository in evaluator custody. Use only explicitly
authorized local synthetic or otherwise approved test data. Keep prompts,
traces, weights, prediction arrays, and row-level outputs in owner-approved
custody.

## Frozen procedure

1. Run hermetic runner and fixture tests on tiny constructed models only.
   Freeze the protocol, capture/evaluator code, tests, requirements, and
   fixture-generation rules with `replacement_loss_v3_runner.py freeze-packet`.
   Archive the exact files and freeze manifest in external custody. Predeclare
   the three architecture families, checkpoint selection rules, exactly three
   seeds per family, prompt and intervention construction, metrics, thresholds,
   controls, budgets, retention, and stop conditions. The visible fixture has
   32 synthetic prompts. Each seed's disjoint adaptation fixture has 128
   prompts and must yield at least 512 paired projection-input/output token
   rows. Only after this packet is digest-frozen may fresh checkpoint
   identities be selected. A qualifying checkpoint must not have informed
   protocol design or threshold selection.
2. Select one fresh, pinned checkpoint per family under the frozen rules.
   Generate the fixed visible and seed-specific adaptation fixtures. Use
   visible prompt activations only to select the projection with greatest mean
   per-feature variance, breaking ties by lower layer index. Record target
   identity, checkpoint tree digest, and visible-fixture digest.
3. Capture selected- and adjacent-layer projection inputs and outputs on
   adaptation prompts. Fit a fresh treatment, a wrong-module control trained
   on the adjacent projection, and a random control trained on deterministically
   shuffled target outputs. A raw-index permutation reorders the fitted input
   mapping; a mapping-aligned permutation applies the compensating coordinate
   transform and is an equivalence diagnostic. The no-op control is a zero
   projection. Every fitted control uses the same adaptation row count. No
   treatment or control copies or reads target parameters.
4. Digest and freeze all nine candidates and controls before generating any
   challenge fixture. Then generate 60 challenge prompts per checkpoint/seed
   case: 20 ordinary, 20 compositional, and 20 adversarial. The custodian draws
   challenge entropy from the operating-system CSPRNG after candidate freeze.
   Prompts, continuation text, interventions, and challenge entropy remain
   outside the repository. Entropy is not retained; exact sealed challenge
   fixtures are retained for replication.
5. For each sealed challenge sequence, the evaluator stops the original-model
   forward at the selected projection's pre-hook and gives the predictor only
   the selected projection input, prompt tokens, and sealed intervention
   vector. The predictor records replacement boundary effects,
   first-continuation-token logit effects, next-token logits, and continuation
   loss for treatment and controls. The prediction array and receipt are sealed
   before the original-model forward continues. Then the evaluator runs the
   original baseline twice, runs the intervention once, and scores the
   committed predictions. The proposer does not inspect challenge prompts or
   row-level outcomes before receipts are sealed.
6. Run the held-out challenge against the original model, learned replacement,
   and every control. The no-op control is a zero projection whose frozen
   causal prediction is zero effect. Do not disclose row-level predictions or
   outcomes to the proposer before the outcome receipt is sealed.
7. The separate replicator runs the same frozen packet and records its own
   source, runtime, fixture, and output digests. Differences remain explicit;
   they are not averaged away.

## Outcomes and locked thresholds

The primary metric is causal-effect prediction on held-out interventions.
For each intervention, compare the frozen predicted change with the measured
change at the replaced module boundary and in model logits. Normalize effect
error by the paired original-model baseline logit norm. The resolution floor
is `max(1e-12, 1e-8 × ||baseline_logits_run_1 − baseline_logits_run_2||₂)`;
the baseline is evaluated twice with identical inputs before interventions.
Exclude only interventions whose observed logit-effect norm is at or below
that floor, and report the excluded count.

The intervention recipe is fixed before checkpoint selection. Ordinary
interventions use a Gaussian direction. Compositional interventions sum two
independently generated unit directions and renormalize the sum. Adversarial
interventions perturb only the largest-magnitude feature of the challenge
row's projection input, with sign drawn from challenge-only entropy. Every
direction has L2 norm `0.01 × RMS(projection input)`. The evaluator draws
intervention entropy after candidate freeze and retains only the resulting
sealed direction vectors. The predictor may use the challenge row and its
direction, but has no access to original-projection outputs or original
intervention effects before prediction commitment.

Treatment passes the primary gate only if all of these hold:

- at least 80% of resolvable held-out intervention effects have the predicted
  direction;
- median cosine similarity between predicted and observed effect vectors is
  at least `0.90`;
- median normalized effect error is at most `0.20`;
- the cluster-bootstrap 95% lower confidence bound for direction accuracy is
  greater than `0.70`; and
- the treatment's primary score exceeds each raw-index permutation,
  wrong-module, random, and no-op control with a paired cluster-bootstrap 95%
  lower confidence bound greater than zero for each treatment-minus-control
  score difference.

Resampling clusters by checkpoint and seed is mandatory; prompt-level
observations are not treated as independent replicates. Each checkpoint/seed
case must include at least 20 resolvable held-out interventions in each of the
ordinary, compositional, and adversarial strata. A case below this coverage is
`INCONCLUSIVE`, not a pass.

Secondary gates are:

- **Held-out behavior:** cross-entropy over all sealed continuation tokens is
  no more than `0.05` nats/token above the original model. Next-token top-1
  agreement and paired normalized error for the next-token logit vector
  (maximum `0.05`) are reported.
- **Activation rescue:** at the replaced subsystem boundary, median activation
  cosine similarity to the original is at least `0.90` and median normalized
  activation error is at most `0.20`.
- **Compositional and adversarial behavior:** report each stratum separately;
  both must meet the same behavior and causal-prediction thresholds, with no
  post-hoc exclusion of difficult cases.

The permutation control reports both raw-index and mapping-aligned scores. The
raw-index permutation permutes the fitted projection's input-weight columns
while retaining the original input coordinates; it is a negative control
included in the primary superiority comparisons. The mapping-aligned
permutation uses those same permuted columns and applies the matching input
coordinate permutation to both the activation and intervention. It is an
equivalence diagnostic: it should preserve the fitted function to absolute
error at most `1e-4` under float32 inference. The tolerance allows for changed
floating-point accumulation order after coordinate permutation. The aligned
control is not required to be beaten by treatment. Both forms are retained in
the report.
The wrong-module control is trained on the same prompt set from the adjacent
projection layer and transferred to the selected layer. The random control
uses the target adaptation inputs with deterministically shuffled targets and
the same fitting procedure. These controls receive the same adaptation row
count as treatment. The no-op control removes the subsystem and inserts no
replacement; its frozen causal prediction is zero effect. Control failures
and unexpected control passes are both retained.

## Repeats, cost, and stopping

Run exactly three seeds for each of at least three architecture families
(minimum nine checkpoint/seed cases). Use paired seeds and identical data
partitions across treatment and controls within a checkpoint. Do not add seeds
after inspecting held-out outcomes. A failed or incomplete case is retained;
any protocol change requires a new version and a new held-out packet.

Use only locally custodied checkpoints and local CPU compute. This frozen
packet does not authorize Gman or any other remote worker, provider call, paid
execution, production traffic, real-user payload, or irreversible external
write. Cap wall time at two hours per checkpoint/seed case and 18 hours total,
and process peak RSS at 24 GiB. Stop at any cap and report incomplete cases,
process CPU time, wall time, peak memory, storage, and available energy
measurements. Monetary cost and energy are not measured by this local runner;
report them as unavailable, not zero.

## Assurance, custody, and claim gates

The local run may report `INTERNAL_PASS` only for the frozen synthetic/local
scope and only if its gates pass. The clean separate rerun may report
`REPRODUCED_LOCAL`. A separate blind internal reviewer may report
`SEGREGATED_INTERNAL_REVIEW`. Neither status is `INDEPENDENT_ACCEPT`.
The replication receipt also compares the full installed-package inventory
digest against the primary runtime.

Every report binds exact source revision and dirty-tree identity, protocol and
configuration digests, checkpoint and fixture digests, state slice, assurance
level, reviewer and reproducer status, data/artifact custody, toolchain,
resource use, thresholds, all control outcomes, known limitations, claim
ceiling, nonclaims, and the next blocked authority gate. Store row-level
prompts, traces, predictions, and model artifacts only in approved external
custody; repository receipts remain aggregate-safe.

The next external-science gate requires a frozen digest-bound packet and an
external reviewer with independent authority. No review task, signature, or
attestation is created by this protocol. Consciousness requires a separate
theory-driven research program. The human alignment packet remains
`BLINDED_CALIBRATION_PACKET_READY`; subagents cannot establish human
equivalence or independent acceptance. Production requires separate provider,
security, reliability, rollback, monitoring, and operational authorization.

The program sequence remains:

1. Prove the additive-module theorem.
2. Reproduce its implementation across the fresh, predeclared architecture
   class, then obtain `SEGREGATED_INTERNAL_REVIEW`.
3. Run this independently trained replacement and blind causal-prediction
   protocol.
4. Freeze a digest-bound packet and seek blind external replication through
   an independently authorized reviewer.
5. Conduct the separate human-calibrated alignment evaluation; the existing
   human packet remains `BLINDED_CALIBRATION_PACKET_READY` until its own gates
   pass.
6. Seek separate production authorization after provider, security,
   reliability, rollback, monitoring, and operational gates pass.

## Current status

The v2 local structural split is accepted here as `INTERNAL_PASS` for local
structural replacement only. It duplicates a channel and divides its
outgoing projection algebraically; it does not train a replacement or predict
unseen causal effects. This v3 specification and its utilities remain
unfrozen until the exact source manifest is created before checkpoint
selection. Runner validation uses only tiny constructed models and synthetic
fixtures; those tests do not count as any of the nine checkpoint/seed cases.
