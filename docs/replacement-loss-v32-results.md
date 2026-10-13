# Replacement-loss V32 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v32-causal-ladder-v1`,
`global-experiment-assurance-lifecycle-v1`. Historical audit:
`theseus-all-runs-audit-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8`, plus the digest-bound uncommitted
packet; unrelated dirty work is preserved.

## Literal result

**Scientific outcome: FAIL. Cases passing every gate: 6/9. Assurance:
REPRODUCED_LOCAL. Reviewer: NOT_SUBMITTED.** All three GELU and all three
SiLU-gated cases passed; all three ReLU cases failed. The exact failed aggregate,
candidate states, fixtures, prediction arrays, oracle arrays, and metrics
reproduced in a second isolated local runtime. All 12 reconciliation checks
are true. No threshold, fixture, model, or failed outcome was changed.

This is stronger local evidence than an affine projection fit for the six
passing cases: both complete nonlinear blocks were removed and replaced by
freshly initialized, trained blocks, with each intermediate snapshot scored.
Removal of either original block harms the task substantially. It remains a
finite **synthetic trained feedforward** experiment; the three families are
activation choices, not pretrained OPT/GPT-2/Llama models.

## Audit and frozen design

The [all-recorded-runs audit](theseus-all-runs-audit-v1.md) and its machine
inventory cover 186 research documents and 134 aggregate receipts in 70
accessible custody roots through V31. Counts include copied receipts and
partial attempts; they are not counts of independent experiments. Unknown or
inaccessible runs remain unknown. V8 is still the strongest positive result
on real pretrained checkpoints, within its affine projection scope.

The [V32 protocol](replacement-loss-v32-causal-ladder.md) was frozen before
guides or task data were generated. It uses an interaction task with four
continuous outputs and their sign decisions; nine fixed guide seeds across
ReLU/GELU/SiLU-gated; disjoint guide-training, visible, adaptation, donor, and
heldout data; and independently initialized 16→96→16 GELU replacement blocks.

Each method replaces block one, then block two while training the converted
prefix, then performs final task training with all parameters unfrozen.
There are 1,400 updates per replacement and 600 final updates per method.
Methods share initialization, minibatches, data, and optimizer budgets:
causal interchange supervision, pointwise activation/output matching,
output distillation, and progressive CKA alignment plus final task training.

The CKA schedule follows the core staged procedure in
[Network of Theseus](https://arxiv.org/html/2512.04198v1), using this study's
small models and procedural corpus. It does not reproduce that paper's
benchmark. Interchange supervision overlaps with
[Causal Distillation](https://aclanthology.org/2022.naacl-main.318/), so this
result is not a novelty or priority claim.

## Task sensitivity and heldout outcomes

All nine guides passed visible-only qualification. Sign accuracy ranged
98.51–99.85%; normalized RMSE ranged .00766–.04611. Removing either block
lowered accuracy by 48.05–51.56 percentage points. This closes V31's particular
task-insensitivity problem in this task; it does not establish universal
component necessity.

After freezing all 108 method-by-stage candidate snapshots, the runner drew
fresh challenge entropy and generated 256 base/donor pairs in each of four
strata. Training used one-coordinate neural output interchanges; challenge
masks used 1/2/4/8 coordinates and fresh contexts, with adversarial input
range expanded from [-1,1] to [-1.4,1.4]. Two-coordinate input substitutions
were also tested. All predictions were on disk and digest-bound before
guide outcomes and actual learned-restoration outputs were evaluated.

| Guide family | Complete causal-method cases | All-gate cells after block 1 | After block 2 | After final task training |
|---|---:|---:|---:|---:|
| ReLU | 0/3 | 8/12 | 6/12 | 6/12 |
| GELU | 3/3 | 12/12 | 12/12 | 12/12 |
| SiLU-gated | 3/3 | 12/12 | 12/12 | 12/12 |

Across the six passing cases, all snapshots and strata had output normalized
RMSE ≤.09697, sign agreement with the guide ≥.98633, internal effect median
relative error ≤.10883, internal effect 90th-percentile error ≤.32792, and
median effect cosine ≥.99662. Learned restoration RMSE was ≤.09697; the
candidate's prebound prediction of actual restoration had RMSE ≤.06943.
These are descriptive extrema for the passing, predeclared families, not an
upgrade of the failed nine-case protocol.

All nine cases passed random, zero-module, and wrong-module input-effect
sensitivity gates in all four strata (36/36); the unchanged-guide positive
controls also passed. Learned restoration uses candidate activations inserted
into the original suffix after prediction sealing. Oracle self-restoration is
reported separately and does not count as learned rescue.

## ReLU failures and method comparison

ReLU's ordinary one-coordinate effects at the first block failed the
90th-percentile relative-error bound. After both blocks were replaced,
the three seeds had 90th-percentile errors .65216/.67768/.74575 against ≤.50.
After final task training these were .64014/.57796/.83886. Adversarial
restoration also failed in every ReLU seed at one or more snapshots; block-two
first-block restoration errors were .15457/.10597/.16088 against ≤.10.
Seeds 3201 and 3203 additionally missed adversarial output-RMSE gates.
High sign agreement and effect cosine therefore do not erase the failures.

The cause is not established. Plausible next hypotheses are mismatched smooth
replacement geometry for ReLU boundaries, underweighting small effects by a
global-mean-square loss normalizer, and finite coverage outside the training
domain. These require new development data and a new frozen test, not changes
to V32's gates.

The final snapshot comparison gives task retention its own column:

| Method | Final behavior cells passing /36 | Final internal-effect cells passing /36 | Final complete cells passing /36 |
|---|---:|---:|---:|
| Causal | 34 | 33 | 30 |
| Pointwise | 36 | 6 | 5 |
| Output distillation | 36 | 0 | 0 |
| Staged CKA + task training | 31 | 0 | 0 |

The named complete-case comparator retention gates passed 0/9 for each
comparator. This is evidence of a task/causal-contract distinction under this
protocol, not a statistical superiority claim or proof that CKA destroys
causal identity. CKA permits changes of coordinate basis, while this packet
uses partial interventions at **fixed 16-dimensional coordinates** and fits
no alignment map for that comparator. A future comparison must freeze an
adaptation-only alignment and its invertibility/coverage checks before
intervening in corresponding coordinates. Earlier symmetry experiments already
showed why raw coordinate labels alone are unreliable.

Causal supervision also uses extra guide-effect targets and extra forward
passes. Methods have equal updates and source data, not equal information or
FLOPs. Paired input-effect differences are mixed against the pointwise and
output-only methods, especially for ReLU; average task error cannot substitute
for the internal effect gate. The aggregate JSON retains these comparisons.

## Exact identities, custody, and resources

- Frozen packet SHA-256:
  `a50766c1611c72a444fa8213378761adaef84811f5f3afdc0181aafce7f11c67`.
- Freeze manifest SHA-256:
  `ad051b45cc8ccc9ee6f1e2d329c8684290542a819cd70d1411b6cde45409066e`.
- Primary result SHA-256:
  `4a28af448478e15f0f4e50cb3bcb5c8d68c98da9a0a4865fb9565c2ad9af5000`.
- Replica result SHA-256:
  `a0efe594ee036d30761c3c85b6ced116db9d5e6ea30393d6a05c3af952ca6158`.
- Reconciliation receipt SHA-256:
  `28337ed9f46caf76df78877a0bb700e7ec7da31b1ad0525db7470a1b909f7992`.
- Primary candidate/challenge/prediction manifest file hashes are recorded in
  [the aggregate result](replacement-loss-v32-results.json); no raw rows or
  challenge entropy appear in the repository.

Owner custody:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v32/primary`
and sibling `replica`. Both archived source packets match all frozen members.
Every new custody directory/file has no group/world permissions; root modes
are 0700. Guide weights, raw fixtures, entropy, candidates, predictions,
outcomes, and original receipts remain there. Report/inventory files are
aggregate only. The historical audit is a pre-V32 snapshot, not a current
worktree-integrity manifest.

Runtime: Python 3.12.13, Torch 2.6.0, NumPy 2.4.6, exact 11-package lock,
macOS 26.6.1 arm64 CPU, one intra/inter-op thread, float32 deterministic
algorithms. Fresh primary and replica environments have distinct prefixes
and site-packages, with no system packages. Same base interpreter/owner is
permitted and does not imply independent authority.

| Resource | Primary | Replica |
|---|---:|---:|
| Protocol execution wall time | 68.805 s | 68.224 s |
| Complete process elapsed time | 69.89 s | 69.35 s |
| Peak RSS | 308,477,952 B | 308,281,344 B |
| Final custody artifact bytes | 46,919,879 | 46,920,502 |
| Monetary spend | $0 | $0 |

Limits were 1,800 seconds, 4 GiB RSS, and 1 GiB artifacts per runtime. Energy
was not measured. No provider, GMan, cloud deployment, external transfer, or
review submission was performed. Nine frozen engineering checks passed in
both runtimes before/around execution. Separate post-run fault-injection
checks exercise failure receipt retention, existing-root protection, and
invalid-packet rejection in automatically cleaned external dummy fixtures;
they do not modify the scientific packet or rerun its custody roots.
The final combined suite passed 11/11 tests in each runtime (6.986/7.521 seconds).
Constructed fault-injection fixtures were removed after testing. The
aggregate engineering receipt is stored at the V32 custody root; its assurance
is INTERNAL_PASS for those contracts only.

## What is left

Keep V32 FAIL and its six narrower positive cases separate. The next method
study should isolate ReLU replacement geometry and small-effect weighting on
fresh development guides, compare activation-matched and smooth students at
equal parameter count, then freeze the selected method and aligned comparator
before fresh confirmation data. Do not weaken the .50 relative-error or .10
restoration gates. Transfer to pretrained whole-block targets requires a
separate frozen checkpoint/task protocol; V8 remains an affine result.

Maximum claim ceiling:
`LOCAL_SYNTHETIC_TRAINED_NONLINEAR_LADDER_CAUSAL_RETENTION_ONLY`.
Novelty remains NOT_ESTABLISHED. Reproduction strengthens local credibility;
it does not prove the Ship of Theseus theory, all-model identity, subjective
experience, human-calibrated alignment, independent acceptance, or production
readiness. Independent science, alignment, consciousness research, and
production authorization remain their separate authority gates.
