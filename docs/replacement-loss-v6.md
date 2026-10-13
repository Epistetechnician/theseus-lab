# Replacement-loss-v6: downstream-aware independent projection replacement

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v6-downstream-aware`.

Status: preselection preregistration. The runner freezes this document,
implementation, tests, dependency manifest, and fixture rules before selecting
checkpoint identities or generating v6 adaptation and challenge records.
After freeze, packet members are immutable; lifecycle outcomes go in the
separate results document. The v3, v4, and v5 packets remain immutable and
retain their reported outcomes.

## Question and bounded claim

The v3 treatment metrics passed in six case-level evaluations, but its
aggregate outcome was `FAIL`: all three GPT-2 cases failed activation rescue,
one also failed held-out behavior, and four cases exceeded the aligned
permutation equivalence tolerance. The v4/v5 GPT-2 adaptation follow-ups did
not pass their full behavior gates. v6 tests a different fitting objective
on a fresh checkpoint per architecture family:

> Can a newly initialized affine projection trained jointly on target-boundary
> activation reconstruction and downstream next-token behavior meet the
> frozen causal-prediction, behavior, activation, and numerical-equivalence
> gates across OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated models, and improve
> held-out maximum next-logit error over an output-only ridge fit to the same
> adaptation rows?

The result can support only a local observation about this fitting procedure
and these frozen checkpoints. It cannot establish discovery of the original
mechanism, all-architecture validity, independent science, consciousness,
alignment, or production readiness.

## Target selection and cases

Use exactly one public base checkpoint from each frozen family:
OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. Checkpoint identities and exact
40-character source revisions must be selected only after packet freeze.
Require `trust_remote_code=False`, at most 500 million parameters, the
family's frozen activation and MLP projection contract, and no checkpoint
identity previously used in Theseus Lab protocol design or evaluation. The
post-freeze manifest records selection time, base-model status, custody-relative
path, revision, and the v6 packet digest. If any family has no eligible fresh
checkpoint, stop before candidate preparation and report
`NO_ELIGIBLE_FRESH_TARGET`; do not substitute a previously used checkpoint.

Run three predeclared seeds: `20261101`, `20261102`, and `20261103`, for nine
checkpoint-by-seed cases. Use visible-only prompts to select the MLP output
projection whose input activations have the greatest mean per-feature
variance, breaking ties by lower layer index. The selected projection is
OPT `fc2`, GPT-2 `mlp.c_proj`, or Llama `mlp.down_proj`.

## Adaptation and treatment

Each case uses a deterministic 128-prompt adaptation set: 64 ordinary, 32
compositional, and 32 adversarial synthetic prompts. The visible, adaptation,
and held-out challenge partitions must be pairwise disjoint by example ID,
exact prompt, and exact prompt-plus-continuation content. New v6 IDs and
prompt prefixes must also make them disjoint from the custodied v3-v5
partitions. Adaptation and held-out challenge text remains in external owner
custody.

The treatment is a newly initialized `torch.nn.Linear` with the selected
projection's dimensions and bias layout. It receives no copied or read target
parameters. Freeze the base model. Train only the replacement for exactly
eight epochs using adaptation prompts. Initialize PyTorch with the case seed.
For each epoch, generate a new row permutation from a persistent NumPy
`default_rng(seed XOR 0x6A09E667)` and batch that order consecutively. Use CPU
float32, AdamW with learning rate `1e-3`, zero weight decay, and batch size 4.
No early stopping or challenge-driven tuning is allowed.

For each adaptation batch, first capture the original projection outputs and
teacher logits. Then replace the projection output with the candidate output
and compute:

```text
L = L_activation + L_behavior
L_activation = sum(mask * (candidate_output - teacher_output)^2)
               / max(sum(mask * teacher_output^2), 1e-12)
L_behavior = mean KL(teacher_next_token_distribution || candidate_distribution)
```

`L_behavior` uses only the final prompt position, temperature 1, and no
continuation or held-out token. Apply the activation mask to all non-padding
prompt positions. The objective weights are fixed at 1:1. Candidate
initialization uses the case seed; the optimizer schedule, epoch count, and
batch size are frozen.

Fit an output-only ridge baseline from the same adaptation rows using the
unchanged v3 closed-form fitter. Preserve v3 random-target, raw-index
permutation, mapping-aligned permutation, and no-op controls. In v6 the
evaluator's existing `wrong_module` slot is explicitly aliased to this
output-only ridge baseline so treatment and comparator predictions are sealed
together. The historical v3 protocol separately tested the wrong-module
control; v6 does not claim to repeat that control.

## Freeze, blind challenge, and metrics

After packet freeze, select and pin the three fresh checkpoints. Generate
visible and adaptation fixtures, train all nine treatments and baselines, and
freeze every candidate and its source receipt before drawing any held-out
challenge entropy. Then draw 60 fresh prompts per case from OS entropy: 20
ordinary, 20 compositional, and 20 adversarial. Seal predictions for treatment
and every control before the evaluator opens original-model challenge
outcomes. The v3 capture/evaluator implementation is reused byte-for-byte and
its exact source packet is pinned as a dependency of v6.

Retain the v3 locked thresholds for the treatment:

- at least 80% direction accuracy, median cosine at least `0.90`, and median
  normalized causal-effect error at most `0.20`, with at least 20 resolvable
  effects per stratum;
- the 95% cluster-bootstrap lower bound for direction accuracy exceeds
  `0.70`, mean direction accuracy is at least `0.80`, and treatment-minus-
  control lower bounds exceed zero for the output-only ridge, random-target,
  raw-index permutation, and no-op controls;
- in every case and stratum, maximum normalized next-logit error is at most
  `0.05` and continuation cross-entropy increase is at most `0.05` nats per
  token;
- activation cosine is at least `0.90` and normalized activation error is at
  most `0.20` in every case and stratum; and
- mapping-aligned module-effect and logit-effect maximum absolute errors are
  each at most `1e-4` in every case. These errors must appear in case-level
  blockers as well as the aggregate gate.

The v6 primary incremental metric is held-out maximum normalized next-logit
error per checkpoint-by-seed case. Compute paired improvement as
`ridge_error - treatment_error`, then bootstrap the nine family-by-seed
clusters with 10,000 draws, seed `20261125`, and a 95% lower bound. Require a
mean improvement of at least `0.005`, a positive bootstrap lower bound, and a
non-negative mean improvement within each of the three architecture families.
Prompt rows are not independent replicates. Overall v6 outcome is `PASS` only
if every treatment gate and the incremental metric gate pass; a measured miss
is `FAIL`, while missing fixtures, digest mismatches, incomplete prediction
sealing, or resource termination are `INCONCLUSIVE`.

## Controls, budgets, and stopping

All candidates share the checkpoint, tokenizer, visible and adaptation
fixtures, selected projection, and challenge prompts. The output-only ridge
baseline isolates the change in fitting objective. Random-target, raw-index
permutation, mapping-aligned permutation, and no-op controls retain the v3
negative-control coverage. The v3 historical wrong-module result remains
separately reported.

Use local CPU only. Stop a case above 24 GiB peak RSS or two hours wall time;
stop the experiment above 18 hours total. Monetary cost and energy remain
`NOT_MEASURED`. No provider, remote worker, paid execution, real-user data,
production traffic, or irreversible external write is authorized.

## Custody and assurance

Keep model weights, prompts, captures, candidates, predictions, outcomes, and
row-level diagnostics in an owner-approved external custody root. Repository
files contain protocol code, frozen identities, and aggregate results only.
The report binds the exact source revision and worktree digest, v6 and pinned
v3 packet digests, checkpoint and fixture digests, dependency inventory,
resource use, all gates, custody, nonclaims, reviewer status, and next gate.

A clean same-owner second runtime may establish `REPRODUCED_LOCAL`; it is not
independent science or `INDEPENDENT_ACCEPT`. External reviewer status remains
`EXTERNAL_REVIEW_UNAVAILABLE` until the configured reviewer returns a signed
decision binding the exact frozen packet. Alignment, consciousness, and
production remain separate programs and gates.
