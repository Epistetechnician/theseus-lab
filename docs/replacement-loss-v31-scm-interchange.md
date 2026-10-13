# Replacement-loss V31: explicit SCM input-intervention qualification

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v31-scm-interchange-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Question and claim ceiling

Does adding paired structural-causal input-intervention supervision to an
independently initialized replacement MLP improve prediction of held-out
intervention effects inside a small trained decoder Transformer, while
preserving its task behavior and target-module outputs, compared with a
parameter-matched pointwise-only replacement?

This is a local method qualification on a procedurally generated task. It is
not a reproduction of IIT, Type-level IIT, Causal Distillation, CMR, or
Network of Theseus. Character reversal is an established task in prior work;
this packet makes no task-novelty claim ([Huang et al., Findings of ACL
2023](https://aclanthology.org/2023.findings-acl.770/)). The replacement
objective is an explicitly named paired input-intervention loss, not an
IIT-style hidden-variable interchange objective.

Maximum claim ceiling:
`SYNTHETIC_TRAINED_TRANSFORMER_INPUT_INTERVENTION_REPLACEMENT_QUALIFICATION`.
No claim is made about pretrained foundation models, architecture-general
validity, novelty priority, consciousness, alignment, independent science, or
production readiness.

## Task and structural causal model

After the packet is frozen, the runner generates a finite task support with
four character variables `X_0..X_3`, each taking values in `{A,B,C,D}`. The
exogenous variables are independent and uniform. The deterministic task
equations are

```text
Y_0 = X_3
Y_1 = X_2
Y_2 = X_1
Y_3 = X_0
```

The model receives `BOS X_0 X_1 X_2 X_3 SEP` and generates `Y_0..Y_3 EOS`.
The owner-generated support enumerates the 256 possible input strings. It has
no third-party dataset or model-asset dependency. The guide is trained on all
256 strings; only replacement adaptation is split by `X_0`. Thus the held-out
claim is replacement adaptation across the frozen prefix-composition split,
not guide generalization to unseen task examples.

For a base context `x` and a source context `s` that differs from it at one
position `k`, the intervention is

```text
do(X_k := s_k)
```

and the counterfactual context is `x_cf = x` with only `x_k` replaced by
`s_k`. The SCM predicts that only `Y_(3-k)` changes, to `s_k`. This protocol
uses `k ∈ {1,2,3}` so both endpoints remain within the same predeclared split
on `X_0`.

## Frozen guide, split, and candidate methods

- Guide: a decoder-only causal Transformer with two blocks, hidden width 32,
  four attention heads, and a `32 -> 64 -> 32` GELU MLP in each block.
  Train three independent guide seeds (`3101`, `3102`, `3103`) for exactly
  1,500 full-batch AdamW updates at learning rate `0.003`, weight decay
  `0.0001`, on the full 256-example task support. CPU only, one Torch thread,
  deterministic algorithms, no dropout.
- Replace block 0's entire MLP. Block 1 is the wrong-module control. No guide
  checkpoint is selected from a model hub; each guide is reconstructed from
  its frozen seed and the generated full task support.
- Replacement adaptation contexts have `X_0 ∈ {A,B}`. Held-out challenge
  contexts have `X_0 ∈ {C,D}`. The two sets are disjoint. Adaptation pairs
  apply the defined `do(X_k := s_k)` for each `k ∈ {1,2,3}` and remain inside
  the adaptation split.
- For each guide seed, initialize pointwise and paired-causal replacement
  MLPs from identical fresh weights. Train both for exactly 600 full-batch
  AdamW updates at learning rate `0.001`, weight decay `0.0001`.
- The pointwise arm minimizes normalized squared error to the guide MLP on
  adaptation endpoint activations. The paired-causal arm uses the same
  pointwise term plus equally weighted normalized squared error on paired
  MLP-output effects for the declared SCM interventions. Both normalizers
  use adaptation endpoints only. No validation selection or early stopping.
- Controls: untrained random replacement, zero-output/no-op replacement, and
  the trained paired-causal replacement installed at block 1 instead of
  block 0. The pointwise arm is the primary matched comparator.

After all guide and replacement state digests are sealed, draw 256 bits of OS
CSPRNG entropy per guide seed. Use it to select 32 distinct held-out base
contexts per `X_0` value (`C` and `D`), then construct one source/base
intervention per selected base and each `k ∈ {1,2,3}`. Save the entropy and
fixtures only in owner-controlled external custody. Seal replacement and
control predictions before evaluating held-out guide outputs or scoring.

## Outcomes and gates

The guide must reach exact reversal accuracy `>= 0.99` on the complete finite
task support before replacement training. Otherwise the study is
`INCONCLUSIVE` because the guide did not qualify for the replacement question.
Guide accuracy on sampled held-out endpoints is descriptive only and is not a
second guide-qualification gate.

For each of the three guide seeds, the paired-causal replacement must meet all
of these gates on held-out contexts and interventions:

- exact task accuracy `>= 0.95`;
- mean task cross-entropy increase over the guide `<= 0.05` nats per output
  character and median normalized output-logit error `<= 0.05`;
- target-module activation median cosine `>= 0.90` and median normalized
  activation error `<= 0.20`;
- at least 140 of 192 intervention effects are resolvable, where the observed
  target-position logit-effect norm exceeds
  `max(1e-8 * baseline-logit norm, 1e-12)`;
- median held-out target-position effect cosine `>= 0.90` and median
  normalized effect error `<= 0.20`;
- at least 10% relative reduction in median causal-effect error versus the
  pointwise arm, and at least `0.05` absolute reduction versus both the
  random and no-op controls. Report the wrong-module result as a diagnostic:
  the untouched target block may preserve task effects, so it is not a
  required superiority gate; and
- oracle teacher-activation restoration maximum per-endpoint normalized
  logit error `<= 1e-5` as an evaluator-pipeline control. Each endpoint's
  normalized error is its logit-vector L2 error divided by the target
  logit-vector L2 norm, floored at `1e-12`.

All three seeds must pass for scientific outcome `PASS`. A measured miss is
`FAIL`; missing/corrupt inputs, digest mismatch, failed guide qualification,
or resource termination is `INCONCLUSIVE`. Thresholds and seed counts are
not changed after the freeze.

The exact finite support makes task correctness a bounded sanity condition,
not a generalization claim. Held-out contexts test replacement adaptation to
the predeclared `X_0` composition split; they are not independent model or
architecture holdouts.

Intervention-effect metrics are specifically the change in the guide or
replacement's target-position next-token logits between the base and
counterfactual inputs, evaluated with teacher-forced correct prior output
tokens. Those prior outputs are unchanged by the declared SCM intervention.
Greedy exact-match accuracy is scored separately with free-running generation;
cross-entropy and logit effects are teacher-forced. This is not a claim about
free-running sequence-level causal effects, interventions on generated
tokens, adversarial interventions, or arbitrary prompts.

## Runtime, custody, and reporting

- Budget: `$0`, 1,800 seconds and 4 GiB peak RSS per run, 1 GiB external
  artifact cap. Local CPU only; no provider, cloud, GMan, remote worker, or
  paid execution.
- Runtime: Python 3.12.13, NumPy 2.4.6, PyTorch 2.6.0; the complete installed
  dependency inventory and runtime fingerprint are recorded in each receipt.
- The frozen packet comprises this protocol, the runner, the focused unit
  checks, and the dependency lock. The freeze manifest binds their exact
  bytes and is verified before data generation.
- Prompts, task examples, model weights, activations, intervention entropy,
  predictions, and row-level metrics stay under the external custody root
  `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v31`.
  The repository stores only frozen protocol/code and aggregate results.
- A clean second virtual environment reruns the same packet from frozen seeds
  and the primary's sealed challenge entropy. Matching packet, task, guide,
  replacement, challenge, prediction, and outcome digests can support
  `REPRODUCED_LOCAL`, not independent acceptance.

The report must state source revision and dirty-tree digest, all state slices,
assurance, reviewer status, reproducibility, custody, resource use, results by
seed, limitations, claim ceiling, nonclaims, and the next authority gate.
Blind replication and external review remain separate gates for independent
science. This experiment cannot establish consciousness, alignment, or
production readiness.
