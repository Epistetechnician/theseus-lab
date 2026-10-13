# Replacement-loss prior-art boundary v1

As of 2026-10-02.

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-prior-art-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Repository revision reviewed: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
This is a targeted literature scan, not a systematic review, novelty search,
or independent scientific acceptance. It changes no frozen experiment packet.

## Claim being assessed

The broad claim “replace model components while retaining task behavior” is
not new. The narrower candidate claim is:

> Fit a replacement module from disjoint adaptation data, bind its predictions
> before scoring, and test whether it predicts held-out causal intervention
> effects while retaining behavior and activation-level properties.

This note asks whether prior work already covers that exact combination. A
search that does not find an exact match cannot establish novelty priority.

## Closest work

| Work | Direct overlap | Difference from the candidate claim |
| --- | --- | --- |
| [Network of Theseus (2025)](https://arxiv.org/abs/2512.04198) | Progressively replaces guide-network components with target modules, aligns guide and target representations using CKA or D-MNN, and measures task performance across major architecture changes, including GPT-2 to an RNN. | Its headline evaluation is representation alignment and retained task performance; the paper does not make frozen, held-out intervention-effect prediction the central replacement criterion. |
| [Interchange Intervention Training (2021)](https://arxiv.org/abs/2112.00826) | Trains neural models to match counterfactual behavior under interchange interventions and evaluates whether a target causal model is realized in the neural model. | It trains a neural model toward a causal specification; it is not the same setup as replacing a component inside a frozen trained model using disjoint module-adaptation captures. |
| [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2301.04709) | Gives a formal framework for testing whether interventions in a high-level causal model correspond to interventions and outcomes in a neural network. | It supplies causal correspondence concepts and tests, rather than the particular independently fitted component-replacement procedure here. |
| [RNN modular decomposition, reuse, and replacement (2022)](https://arxiv.org/abs/2212.05970) | Decomposes trained recurrent models into modules and studies reuse and replacement, reporting task-level metrics. | Its replacement pathway is based on modular decomposition/pruning and task performance; the paper does not report the proposed held-out causal-effect prediction criterion. |
| [Model stitching (2021)](https://arxiv.org/abs/2106.07682) | Fits trainable connections between frozen networks and tests whether representations can be composed while retaining accuracy. | Stitching tests composability and task accuracy; it is not a held-out causal intervention forecast by a replacement module. |
| [pyvene (2024)](https://arxiv.org/abs/2403.07809) | Provides trainable interventions and examples that fit intervention parameters to counterfactual outputs. | It is an intervention research toolkit and training setup, not evidence that the specific replacement-and-held-out-effect protocol is novel. |

## What the local packets establish

- V8 reports nine passing cases for a ridge-fitted affine MLP output
  projection, trained on disjoint captures, with held-out effects and behavior
  checks. Its assurance is `REPRODUCED_LOCAL`.
- V3 tests independently trained replacements across three families and
  reports `FAIL` (6/9 cases pass); all GPT-2/GELU cases miss activation-rescue
  gates.
- V9 compares a replacement ladder with CKA and logit-distillation baselines.
  The overall outcome is `FAIL` (21/27 ridge stages pass retention). Its CKA
  arm is a module-level NoT-style baseline, not a full reproduction of NoT.
- V28 reports all nine per-case treatment checks passing but an overall
  `FAIL` because the separately frozen equivalence gate misses its threshold.
- V30 is synthetic-only and reports `FAIL` (1/9 complete passes); it is not
  trained-checkpoint evidence.

These results show that the local project has tested the candidate outcome
criterion, but they do not establish that the criterion is novel relative to
all relevant work or that the strongest result generalizes beyond its frozen
checkpoints and procedure.

## Assessment and next discriminating test

Novelty status: `NOVELTY_NOT_ESTABLISHED`.

My inference from the sources above is that the most defensible possible
differentiator is the *evaluation target*: precommitted prediction of unseen
causal intervention effects by an independently fitted replacement, scored
separately from representation similarity and ordinary task behavior. It is
not component replacement itself, causal intervention itself, or functional
retention. The current V9 comparison does not settle the head-to-head claim
because its CKA comparator is not a faithful full NoT implementation.

The next discriminating study, if pursued, is a frozen head-to-head comparison
between the V8-style causal predictor and a faithful NoT replacement baseline
on the same eligible guide checkpoint, replacement schedule, adaptation
examples, and held-out interventions. It must compare task behavior,
representation alignment, and causal-effect prediction as separate outcomes;
freeze each method's predictions before revealing the intervention targets;
use matched compute and controls; and preserve V28's and V30's failures as
unchanged prior evidence. A positive local result would still be `REPRODUCED_LOCAL`
until an independently operated replication and authorized external review.

Do not state that the approach is novel based on this scan. A publishable
priority claim requires a broader, documented search of papers, workshop
proceedings, theses, and code, followed by independent review of the exact
claim and frozen empirical packet.
