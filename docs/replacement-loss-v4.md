# Replacement-loss-v4: compositional adaptation support

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v4-compositional-support`.

Status: preregistration draft. The external digest-bound freeze must be created
before any v4 adaptation fixture, candidate, or challenge is generated.

## Motivation and hypothesis

The v3 post-hoc analysis found a repeatable class-level association: held-out
compositional inputs were much farther from ordinary adaptation inputs and had
larger activation reconstruction residuals. The predeclared within-class
novelty correlation did not replicate, so v4 tests a new, narrower causal
hypothesis:

> For this exact GPT-2 checkpoint and synthetic prompt generator, adding
> compositional examples to a fixed-size adaptation set reduces activation
> reconstruction error on fresh compositional prompts.

The exact target is the same external base checkpoint used by v3:
`openai-community/gpt2-medium`, revision
`6dcaa7a952f72f9298047fd5137cd6e4f05f41da`. The target projection is GPT-2
layer 3 `mlp.c_proj`, selected in all three v3 seeds. Reusing that checkpoint
isolates the adaptation-distribution intervention; v4 makes no new-checkpoint
or cross-checkpoint claim.

## Paired arms

Run three predeclared seeds. For each seed, create two 128-prompt adaptation
arms with exactly 2,048 selected projection rows each:

- `ordinary_only`: 64 shared ordinary prompts plus 64 additional ordinary
  prompts.
- `compositional_augmented`: the same 64 shared ordinary prompts plus 64
  compositional prompts.

Capture projection inputs and outputs only on adaptation prompts. Select 16
evenly spaced valid prompt-token positions per prompt, giving exactly 2,048
paired rows per arm. Construct a fresh affine replacement for each arm using
the frozen v3 ridge least-squares fitter and `ridge=1e-6`. No target projection
parameters are read by the fitter or copied into either candidate.

Freeze both candidate states and all adaptation fixture/capture digests before
generating challenge entropy. Then generate 60 new challenge prompts per seed:
20 ordinary, 20 compositional, and 20 adversarial. Challenge prompts, outcomes,
model weights, and row-level diagnostics remain in external custody. The exact
same challenge prompts evaluate both candidates.

Candidate activation and logit predictions are written and digest-sealed for
all seeds before the evaluator opens original-model challenge activations or
logits. Then evaluate the original model and both replacements on the same
paired challenge rows.

## Metrics and locked gates

The primary per-row metric is
`||candidate_activation - original_activation|| / max(||original_activation||, 1e-12)`.
The primary effect is the ordinary-only median error minus the
compositional-augmented median error on compositional challenge rows.

Each of the three seeds must satisfy all of these:

- compositional-augmented median normalized activation error is at most `0.20`;
- its paired median improvement over ordinary-only adaptation is at least
  `0.10`;
- its median error regression across ordinary and adversarial rows is at most
  `0.05`;
- for every stratum, maximum normalized next-logit error is at most `0.05` and
  mean continuation cross-entropy increase is at most `0.05` nats/token.

The result is `PASS` only if every seed meets every gate. A measured gate miss is
`FAIL`; missing rows, digest mismatch, insufficient coverage, or resource-cap
termination is `INCONCLUSIVE`. Seeds are the replication units; prompt rows are
not treated as independent trials, and no row-IID confidence interval is
reported.

## Variance, controls, budget, and stopping

The paired arms use the same model, seed, shared adaptation prompts, challenge
prompts, tokenizer, runtime, capture positions, fitter, and ridge value. The
only intended difference is the stratum of 64 arm-specific adaptation prompts.
The ordinary-only arm is the negative control. The three seeds measure
repeatability across generated fixtures, not independent checkpoints.

Use local CPU only, with no provider execution. Stop a case if peak RSS exceeds
24 GiB or the case exceeds two hours. Do not tune prompt rules, row selection,
fitter settings, metrics, or thresholds after freeze. A failed result remains
failed; any changed protocol needs a new version and packet.

## Custody and claim ceiling

The exact source packet, adaptation fixtures/captures, candidate states,
challenge fixtures, predictions, outcomes, and aggregate reports are retained
under owner-approved external custody. This protocol authorizes local synthetic
research only. Its claim ceiling is `LOCAL_GPT2_COMPOSITIONAL_ADAPTATION_EFFECT`
for the exact checkpoint, target projection, prompt generator, and three seeds.

It does not establish all-architecture validity, general mechanistic
understanding, independent science, alignment, consciousness, or production
readiness. `REPRODUCED_LOCAL` requires exact output reproduction in the clean
second runtime. External reviewer status remains separately reported.
