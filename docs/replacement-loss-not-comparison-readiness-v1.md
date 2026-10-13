# Network of Theseus comparison readiness v1

As of 2026-10-02.

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-not-comparison-readiness-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Repository revision reviewed: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
This is a literature and local-packet audit. It changes no frozen experiment
protocol, result, checkpoint, or external-custody artifact. The literature
search is targeted, not systematic, and does not establish novelty priority.

## What the published work already covers

The replacement framing itself is established prior art. *BERT-of-Theseus*
(EMNLP 2020) progressively substitutes compact modules into BERT, trains them
with a stochastic replacement schedule, and fine-tunes the resulting model on
the task objective. The later *Network of Theseus* (Sci4DL 2026, 2nd edition)
generalizes progressive replacement across architectures. It aligns target
and guide activations with CKA or D-MNN, progressively trains the replaced
prefix, and then fine-tunes the converted target network end-to-end on the
downstream task. Its examples include GPT-2 attention replaced with recurrent
modules. [BERT-of-Theseus paper](https://aclanthology.org/2020.emnlp-main.633/),
[Network of Theseus paper](https://openreview.net/pdf?id=gbhF4xW25c).

Those papers evaluate representation alignment and downstream task performance.
They do not make precommitted prediction of held-out causal intervention
effects the central replacement criterion. This leaves a possible narrower
evaluation contribution, not a novelty claim: test whether a separately fitted
replacement predicts intervention effects that were not used for training,
and report that result separately from representation similarity and task
performance.

## What the local comparison actually covers

- V9's CKA candidate is one affine map fit to MLP output-projection captures
  with 100 AdamW updates and batches of 128. It runs through three sequential
  projection replacements and does not progressively co-train a converted
  target network or fine-tune that network on a downstream task.
- V16 trains a full-width, independent OPT MLP at one block and evaluates held-
  out interventions and behavior. It has no CKA treatment arm and no post-
  conversion downstream fine-tuning.
- V17 compares full-MLP pointwise and causal-supervised replacements at one
  block. It also has no CKA arm or converted-network fine-tuning.
- V9's result remains `FAIL` under its frozen aggregate gate, even though its
  paired comparison favored the ridge method over its named CKA and logit
  distillation comparators. That is evidence about the V9 procedures only.

Therefore the local record is not a head-to-head against the published NoT
procedure. Calling V9 a full NoT reproduction, or using it to claim causal
replacement beats NoT, would exceed the evidence.

## Why no new CKA run was started

A fresh one-block, whole-MLP CKA fit would close only one gap. It would still
omit NoT's progressive conversion of a target architecture and end-to-end
downstream fine-tuning. The actual local OPT packets also reuse a checkpoint
and adaptation design whose outcomes are already known. Such a run could be
described accurately as a *NoT-style, single-module CKA comparison*, but would
not settle the paper-level comparison or novelty question. Repeating that
partial comparison now has lower value than first specifying the complete
target conversion and its task-training data.

No experiment, checkpoint selection, challenge generation, model training, or
external transfer was performed for this audit. The source inspection used the
existing repository and already-custodied V16/V17 result records. A bounded
depth-five path-name search of the owner artifact volume found no apparent
downstream corpus directory; its WikiText-named match was an OPT model cache,
not a frozen corpus identity. This search does not establish that no corpus
exists elsewhere. The V16/V17 failed outcomes remain unchanged.

## Requirements for a meaningful next comparison

Create a new, executable packet before selecting any fresh checkpoint or
drawing held-out challenge entropy. The packet must:

1. Define a concrete target architecture and replacement unit, with a staged
   schedule that replaces the complete predeclared subsystem set. Keep the
   target architecture and replacement schedule identical across methods.
2. Implement the NoT-style arm as representation alignment at each stage,
   progressive optimization of the already-replaced prefix, followed by
   downstream task fine-tuning of the completed target network. Compare it
   with an independently trained causal-prediction arm under a matched
   adaptation split and a predeclared update or compute budget.
3. Freeze the task-training corpus, exact checkpoint identity, source code,
   dependency inventory, candidate initialization rules, optimizer schedules,
   thresholds, controls, and stop rules before model selection and challenge
   generation. Existing synthetic prompts alone do not constitute an
   established downstream language-modeling benchmark.
4. Freeze all final candidate weights and intervention predictions before
   opening held-out interventions. Report representation alignment, task
   behavior, activation retention, and module-boundary and downstream causal
   effects as separate outcomes. Include naive replacement, wrong-module,
   random, and no-op controls.
5. Reproduce the exact frozen packet in a clean second runtime, then report
   only `REPRODUCED_LOCAL`. A later architecture-held-out study and blind
   external replication remain separate gates.

The first comparison can make only a one-checkpoint, one-target-architecture
claim. It cannot establish novelty priority, architecture-general validity,
consciousness, alignment, production readiness, or independent science.

## Current disposition

- Literature status: `TARGETED_PRIOR_ART_REVIEW_COMPLETE`;
  `NOVELTY_NOT_ESTABLISHED`.
- Existing local NoT comparator: `MODULE_LEVEL_ONLY`.
- New head-to-head run: `NOT_STARTED`; the full conversion target and
  downstream task corpus are not frozen.
- Assurance for this source audit: `INTERNAL_PASS`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `NOT_APPLICABLE` (no run performed).
- Custody: paper sources are publicly linked; model artifacts and previous
  prompt-bearing records remain in external owner custody; no new artifact
  bytes were created outside this repository.
- Next blocked gate: freeze an executable full-target conversion protocol and
  identify its authorized, locally custodied downstream task data.
