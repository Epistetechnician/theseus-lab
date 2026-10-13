# Replacement-loss-v5: balanced adversarial adaptation

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v5-balanced-adversarial-support`.

Status: preregistration draft. Freeze this protocol, source, tests, dependency
manifest, target identity, thresholds, and fixture rules before generating any
v5 adaptation records or selecting from held-out data.

## Motivation and hypothesis

The frozen v4 failure diagnosis found adversarial next-logit guard violations
across the row distribution, with weak and inconsistent within-seed
activation-to-logit rank association. The v4 disposition remains `FAIL`.
The diagnosis packet digest is
`49928db7c35e5890fd948e85c832633f1598e6f86b7b5be7aa32dd030b906311`; its
aggregate report digest is
`b704c0033c10ce447006da283ff661ce76a6ecc4bf9b8d010567deec12dbff0f` and its
exact file SHA-256 is
`4b7517666b56fdd3e544406d022f2f4284ebd1d5533714c7a1e3e6574e54fd14`.
The clean-runtime reconciliation digest is
`f6a07ca9b50987b455624bb1009fa8b7fefbe7ab237ef3cc8b9d534cc285a677`; its
exact file SHA-256 is
`1606c49edbf55f603f75add8f4b43d406f83f494a9f347414db3ab7f33aaad31`.
The analysis was reproduced as `REPRODUCED_LOCAL` in a distinct local runtime.
This new experiment tests a distinct intervention:

> For the exact GPT-2 checkpoint and synthetic prompt generator, replacing
> half of the composition-only arm's unique adaptation examples with
> adversarial examples improves held-out adversarial behavior while retaining
> the composition-only arm's held-out compositional activation benefit.

This is a single-checkpoint follow-up, not an architecture-general test. The
target is `openai-community/gpt2-medium`, revision
`6dcaa7a952f72f9298047fd5137cd6e4f05f41da`, GPT-2 layer 3 `mlp.c_proj`, the
same target selected in all three v3 GPT-2 cases. The v3 pinned independent
linear fitter and local checkpoint assets are used unchanged.

## Paired adaptation arms

Run three predeclared seeds: `20261001`, `20261002`, and `20261003`. Each arm
uses 128 prompts and exactly 2,048 selected projection rows (16 distinct,
evenly spaced valid token positions per prompt):

- `ordinary_only`: 64 shared ordinary prompts plus 64 ordinary prompts;
- `compositional_augmented`: the same 64 shared ordinary prompts plus 64
  compositional prompts;
- `balanced_augmented`: the same 64 shared ordinary prompts plus 32 prompts
  from the composition-only arm and 32 adversarial prompts.

The balanced arm therefore replaces half of the composition-only arm's
unique examples with adversarial examples at a fixed data and row budget. All
three candidates are independently fitted by the frozen v3 ridge fitter with
`ridge=1e-6`; no target projection parameters are copied into a candidate.
Adaptation records and candidate states are frozen before challenge entropy
is generated.

Development smoke fixtures generated before this freeze for candidate seeds
`20260930` through `20260932`, test-only adaptation seeds `20991231` and
`20991232`, and the
fixed challenge-test entropy are excluded from the v5 run. The runner
reconstructs those exclusion fixtures in memory after packet freeze and
requires disjointness before admitting each v5 challenge set. No v5 outcomes
were read for those development fixtures.

After candidate freeze, generate 60 fresh challenge prompts per seed: 20
ordinary, 20 compositional, and 20 adversarial. Use OS entropy. Require exact
ID, prompt, and prompt-plus-continuation disjointness from the v3 visible,
adaptation, and challenge partitions, all v4 adaptation and challenge
partitions, and all v5 adaptation arms. Evaluate all three candidates on the
same challenge prompts. Seal candidate predictions for all seeds before
opening original-model challenge outcomes.

## Metrics and locked gates

Use the v4 per-row activation error
`||candidate_activation - original_activation|| /
max(||original_activation||, 1e-12)`, normalized next-logit error with the
same denominator rule, activation cosine, and continuation cross-entropy
delta in nats per token. Seeds are replication units; prompt rows are not
independent trials. Do not report row-IID confidence intervals or p-values.

Every seed must satisfy all gates for `balanced_augmented`:

- for every challenge stratum, maximum normalized next-logit error is at most
  `0.05` and mean continuation cross-entropy increase is at most `0.05`;
- adversarial maximum normalized next-logit error improves by at least `0.02`
  versus `compositional_augmented`;
- adversarial median activation error is at most `0.20` and improves by at
  least `0.03` versus `compositional_augmented`;
- compositional median activation error is at most `0.20`, median paired
  improvement over `ordinary_only` is at least `0.10`, and median regression
  versus `compositional_augmented` is at most `0.05`;
- median activation regression over combined ordinary and adversarial rows
  versus `ordinary_only` is at most `0.05`.

The overall result is `PASS` only if all gates pass in all three seeds. A
measured gate miss is `FAIL`; missing rows, digest mismatch, incomplete
prediction sealing, or resource-cap termination is `INCONCLUSIVE`. Keep the
v4 outcome unchanged regardless of v5.

## Variance, controls, resource limits, and stopping

The three arms share the exact checkpoint, seed, 64 ordinary prompts, model
runtime, tokenizer, row sampler, fitter, ridge value, and challenge prompts.
The ordinary-only arm is the negative control; the composition-only arm is
the active comparator. The treatment changes only 32 of the 64 unique
composition examples into adversarial examples. Three seeds measure
fixture-level repeatability, not independent checkpoints.

Use local CPU only. Stop any case if peak RSS exceeds 24 GiB or case wall time
exceeds two hours. Do not tune prompt rules, targets, metrics, thresholds,
fitter settings, or adaptation sizes after freeze. Preserve any failure. A
changed protocol requires a new version and packet.

## Custody and claim ceiling

The frozen packet, model selection, model assets, prompts, captures, candidate
states, predictions, outcomes, and row-level diagnostics remain in external
owner custody. Repository outputs contain protocol and aggregate results
only. Claim ceiling:
`LOCAL_GPT2_BALANCED_ADVERSARIAL_ADAPTATION_EFFECT` for this checkpoint,
projection, synthetic generator, and three seeds.

This protocol does not establish fresh-checkpoint generalization, all-GPT-2
or all-architecture validity, mechanistic understanding, independent science,
alignment, consciousness, or production readiness. A clean second local
runtime may establish `REPRODUCED_LOCAL`; it is not independent acceptance.
External reviewer status remains separately reported.
