# Replacement-loss V31 results: SCM input-intervention qualification

As of 2026-10-02.

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v31-scm-interchange-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Verdict

**Scientific outcome: `FAIL`. Assurance: `INTERNAL_PASS`. Reviewer:
`NOT_SUBMITTED`. Reproducibility: `REPRODUCED_LOCAL`.**

The frozen three-seed procedure completed within its local CPU and storage
budgets. Seeds 3101 and 3103 passed every gate. Seed 3102 missed only the
predeclared requirement that paired-causal supervision reduce median held-out
effect error by at least 10% relative to the matched pointwise replacement.
It was 5.03% worse than pointwise on that metric. The frozen aggregate outcome
is therefore `FAIL`; the seed result and threshold remain unchanged.

| Guide seed | Paired-causal median effect error | Pointwise median effect error | Relative improvement | Seed result |
| --- | ---: | ---: | ---: | --- |
| 3101 | 0.00739 | 0.00821 | 10.03% | `PASS` |
| 3102 | 0.01250 | 0.01190 | -5.03% | `FAIL` |
| 3103 | 0.00926 | 0.01140 | 18.76% | `PASS` |

All three guides reached 100% exact reversal accuracy on the full 256-context
support. Every seed had 192/192 resolvable held-out effects. Across seeds, the
paired-causal replacements retained 100% free-running task accuracy; their
median normalized output-logit error ranged from 0.0251 to 0.0303, activation
cosine from 0.9829 to 0.9907, activation error from 0.1379 to 0.1893, and
effect cosine from 0.99994 to 0.99998. The oracle activation-rescue error was
zero. Random and no-op effect-error margins passed in every seed.

This is evidence that the independently initialized replacements fit this
small synthetic task and predict its defined target-position effects. It does
not show a reliable benefit from adding the paired-causal term over pointwise
supervision: that comparative gate passed in two seeds and failed in one.
The extremely low pointwise errors also leave little room for a consistent
improvement in this task.

There was no exact-match behavior divergence across arms: every arm scored
100% on free-running task accuracy. Mechanism-level activation and effect
metrics did diverge slightly; the paired-causal effect error was lower than
pointwise in two seeds and higher in seed 3102.

## Frozen identity and reproduction

- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Worktree-status digest at both runs:
  `c0979023a0f188d6c1f979bc3a53d3cbdb8fc3d5a77e01e859a1764737d9696e`.
- V31 packet SHA-256:
  `0922be564b628c5ac4dafd0271ecf96969001fccb0abbbcc833e0d9382818513`.
- The packet was frozen as
  `FROZEN_BEFORE_TASK_DATA_GENERATION`; no task examples or challenge entropy
  were generated before that freeze.
- Focused unit checks: 4 passed. Runner self-check: `PASS`, including all 768
  finite SCM substitutions; it drew no challenge entropy.
- Primary: Python 3.12.13, NumPy 2.4.6, PyTorch 2.6.0, CPU, one Torch thread;
  43.96 seconds, 312,868,864 bytes peak RSS, 3,456,245 artifact bytes.
- Replica: a separate Python environment with the same dependency inventory;
  43.73 seconds, 318,963,712 bytes peak RSS, 3,456,246 artifact bytes.
- Cost: `$0`.
- Primary result receipt SHA-256: `d3b0ff71d9de49aae59a4fb0a9d79faf96f9aba3a7bb0a9aede08328eb8a45b6`.
- Replica result receipt SHA-256: `7db0541878036c7ec9f873c8661dfc0e30a1980e46fe1610abc5198808bb6ae4`.
- Canonical reconciliation audit SHA-256:
  `d1738274f10c24f88440634e5b393cf10638558f853b24967b1e37b53f27435d`.

The frozen runner's `reconciliation.json` initially labeled reproduction
`INCONCLUSIVE`: it compared in-memory seed maps with integer keys against
JSON-loaded maps with string keys. The primary and replica result files have
identical values for every declared reconciliation field, including task,
adaptation, challenge, guide, candidate, prediction, oracle, and outcome
digests. A post-run canonical reconciliation record documents the type-only
comparison defect and the exact matches. The original runner receipts remain
unchanged. `REPRODUCED_LOCAL` here means same-owner reproduction in a clean
second runtime; it is not independent acceptance.

## Custody and claim boundary

Generated contexts, guide weights, activations, challenge entropy, candidate
weights, predictions, and row-level scores remain outside the repository at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v31`.
The repository contains V31's frozen packet and aggregate report; no V31 task
data or model assets were copied here. No pretrained checkpoint, third-party
corpus, user payload, cloud worker, or paid provider was used.

The maximum claim ceiling is
`SYNTHETIC_TRAINED_TRANSFORMER_INPUT_INTERVENTION_REPLACEMENT_QUALIFICATION`.
The guide is trained on all 256 task strings; the held-out split tests
replacement adaptation across `X_0` composition groups, not guide
generalization to unseen task examples. Intervention metrics are
teacher-forced target-position next-token logit differences conditioned on
the correct prior outputs. Free-running greedy accuracy is reported
separately. The study uses one small decoder design and a character-reversal
task; it is not an architecture-held-out checkpoint study, adversarial prompt
study, full-sequence intervention study, Network of Theseus reproduction,
IIT reproduction, or novelty result.

This result does not establish architecture-general validity, consciousness,
alignment, independent scientific acceptance, or production readiness.
External review was not submitted. The authority gate for claims beyond this
local ceiling remains a blind external replication and authorized external
review. Production and human-calibrated alignment each require separate
gates.

## Next step

Do not tune V31's loss weight, seed set, split, or thresholds and do not turn
its two passing seeds into an overall pass. A follow-on study should test a
distinct hypothesis on a task with nontrivial multi-variable interactions,
where a causal objective can plausibly add information beyond pointwise
module matching. Freeze that task, effect definition, model class, outcomes,
and controls before generating examples or selecting any checkpoint. The
architecture-held-out and external-review gates remain separate.
