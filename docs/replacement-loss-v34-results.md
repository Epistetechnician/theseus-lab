# Replacement-loss V34 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v34-gradient-balance-v1`,
`global-experiment-assurance-lifecycle-v1`. As of 2026-10-03.
Revision: `202f459d6d89df9a038d78d58e0af825d287fcd8` plus the digest-bound uncommitted packet.
[Protocol](replacement-loss-v34-gradient-balance.md),
[freeze](replacement-loss-v34-freeze.json),
[aggregate receipt](replacement-loss-v34-results.json).

## Verdict

Scientific outcome: **`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`**.
Assurance and reproducibility: **`REPRODUCED_LOCAL`**.
Reviewer: `NOT_SUBMITTED`. Confirmation: `NOT_OPENED_NO_ELIGIBLE_VARIANT`.
The authorized development, reproduction, artifact verification, and closeout
completed. No confirmation freeze, confirmation data, family expansion, or
CKA/distillation comparator run was created. The frozen stopping rule was
enforced; neither thresholds nor the failed packet were retuned.

All three fresh ReLU guides passed visible behavior and task-critical-block
qualification. Replacement guides, initializations, adaptation rows, optimizer
steps, minibatches, and gradient measurement overhead were paired across arms.
Predictions were sealed before guide challenge outcomes and actual restoration.
These are small synthetic trained feedforward models, not pretrained
Transformer checkpoints.

## Paired development

A cell requires every behavior, input/internal effect, activation, and rescue
gate at that stage and stratum. Complete cases additionally require all
stages/strata and positive/negative controls.

| Arm | Behavior /36 | Activation /36 | Rescue and prediction /36 | Input effects /36 | Complete cells /36 | Complete cases /3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ReLU/global | 36 | 36 | 36 | 36 | 29 | 0 |
| ReLU/unscaled row | 0 | 0 | 0 | 0 | 0 | 0 |
| ReLU/gradient-balanced row | 36 | 36 | 36 | 36 | 15 | 0 |

| Arm | Block-one cells /12 | Block-two cells /12 | Final-task cells /12 |
| --- | ---: | ---: | ---: |
| ReLU/global | 9 | 9 | 11 |
| ReLU/unscaled row | 0 | 0 | 0 |
| ReLU/gradient-balanced row | 9 | 2 | 4 |

Every global-arm miss was the ordinary first-site internal-effect p90 gate.
Other complete-cell failures in the balanced arm were also internal-effect
checks; preserving output, activation, and rescue was insufficient.
All 12 guide positive-control strata passed, and all 12 negative-control
sensitivity strata passed for each arm. Controls passing do not qualify a
failing treatment.

Ordinary first-site relative-effect p90, threshold ≤.50:

| Guide seed | Global block one | Global block two | Global final | Unscaled row final | Balanced row final |
| --- | ---: | ---: | ---: | ---: | ---: |
| 34101 | .614 | .502 | .410 | 1.089 | .755 |
| 34102 | 1.001 | .664 | .670 | 1.080 | .773 |
| 34103 | .599 | .509 | .426 | 1.244 | .652 |

The balanced arm's final ordinary p90 decreased 30.7%, 28.4%, and 47.6%
relative to the collapsing unscaled row arm, but increased 84.2%, 15.4%,
and 53.1% relative to global normalization. These are descriptive paired
measurements, not a statistical superiority result. The global arm passed
all final-task strata for two guides, yet its earlier stage misses remain
disqualifying under the unchanged cumulative retention contract.

The rule capped the weighted causal-gradient norm at half the anchor-gradient
norm on all 10,200 balanced updates per runtime. The maximum stored weighted
ratio was .5000000000000001, within the frozen floating-point tolerance.
Median row coefficients by guide/stage ranged from .000452 to .001757.
Unscaled row ratios reached 5734.67; balanced raw ratios reached 30165.65
before weighting. The cap prevents gradient magnitude dominance; it does not
control gradient direction or guarantee weak-effect generalization.

H1 did not qualify: this fixed rule restored the measured behavior/activation/
rescue failures relative to direct row weighting, but it did not produce a
causally retaining ladder or improve on the global arm's complete-cell count.
The result does not rule out all balancing methods.

## Causal map qualification

| Arm | Representation-qualified sites /15 | Causally qualified sites /15 | Available complete aligned snapshots /9 |
| --- | ---: | ---: | ---: |
| ReLU/global | 15 | 4 | 0 |
| ReLU/unscaled row | 10 | 0 | 0 |
| ReLU/gradient-balanced row | 15 | 0 | 0 |

Across arms, 40/45 affine maps passed representation/inverse checks, but only
4/45 passed every adaptation-only causal/restoration check. No snapshot had
all active maps qualify, so every aligned snapshot was
`ALIGNMENT_UNAVAILABLE`; no aligned score was emitted. The global arm's
representation-qualified causal misses were intervention-effect failures,
rather than mapped activation or restoration failures. This supports H2 for
these measured maps: representation fit and invertibility did not establish
causal transport. It is not evidence that CKA or another unrun comparator
failed.

## Frozen identity and validation

- Packet SHA-256: `91497500548479499d0f7de61d524adb9f215cd93118414e052644f67289ab91` (29 source members).
- Freeze SHA-256: `7b5b869ac99595e5d3e9bfd037908c9c3c95d16b861be99c27703a2522e49779`.
- Primary result: `119c68276383f4104fbe190bafb4ff77143fe5e03276ac06201ae8b5c1b2aa3b`.
- Replica result: `934b4e332a6211f8b4235ffd17901bb199de8baf80bc53aa283e84fa3a3d3671`.
- Reconciliation: `de1cedbfb54e9c0daa6e8c42d3f433c266b4369bac57b54656be8fc6dc05cd2d`.
- Closeout validation: `d0237561fed89f94eb8dc6f7eeecdf0e2253a2fd5098c7d512281df4ca8d107d`.
- Engineering receipt: `56c9d62d6da933f6b73f98ee9241bf48f51927e1004bc515cc4250a2d29747fa`.

Both fresh Python 3.12.13 environments passed all 28 contract tests.
The extra gradient measurements preserve the parent global-loss parameters
exactly on the engineering ladder. Tests also cover every-update caps,
nonfinite/zero norms, guide lineage and freezing, unavailable/singular maps,
causal qualification failure despite representation qualification, independent
numeric scoring, stream disjointness, artifact tampering, terminal-root reuse,
crash receipts, and the confirmation gate.

Before freezing, an exact parameter-equality test failed at final-task
training. Task-loss graph construction order caused floating-point differences;
the fix restored the parent's order and avoided a multiplication by one in
unscaled arms. Both final suites passed before scientific data generation.
That engineering failure is retained in the qualification receipt; no
scientific identity was repaired or retried.

The separate frozen closeout utility recomputed all 16 reconciliation checks,
verified all 167 manifested files and 30 archived source/freezes per run,
recomputed all 108 native score cells and actual guide-suffix restorations,
and checked all 30,600 gradient readouts per run. Exact preparation, guide,
candidate, map, trace, prediction, oracle, qualification, and canonical metric
identities matched across runtimes. All 41,472 new input rows per runtime
passed exact global disjointness, including exclusions from 15 predecessor
input artifacts. Replica rows are identical and are not additional observations.

This is a same-owner, same-host reproduction with distinct runtime prefixes
and site-packages. It is not external blind replication or independent authority.
Engineering/audit checks are `INTERNAL_PASS`; numeric procedure reproduction
is `REPRODUCED_LOCAL`; the scientific development outcome remains a failure.

## Resources and custody

| Runtime | Phase seconds | Peak RSS bytes | Final custody bytes |
| --- | ---: | ---: | ---: |
| Primary | 28.93 | 317292544 | 20404269 |
| Replica | 29.69 | 317505536 | 20405017 |

All phases stayed below 1800 seconds, 4 GiB RSS, and 1 GiB artifacts.
Recorded phase byte counts precede result/reconciliation writes; the table
uses final verified custody size. Runtime installation used the local cache
and exact lock, with no model/provider download. Monetary execution cost:
**$0**. Energy and machine amortization: `NOT_MEASURED`.

Owner-controlled external artifact root:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v34`.
Primary and replica terminal roots: `development/primary`,
`development/replica`. Qualification and closeout receipts remain outside
those immutable terminal roots. Runtime root:
`/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v34/toolchains`.
The repo stores only code, tests, protocol, digest manifests/freezes, and
aggregate evidence; raw inputs, weights, entropy, traces, maps, predictions,
and oracle outcomes stay in external custody.

Read-only verification from the repository:

```sh
PYTHONDONTWRITEBYTECODE=1 '/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v34/toolchains/primary/bin/python' replacement_loss_v34_runner.py verify
PYTHONDONTWRITEBYTECODE=1 '/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v34/toolchains/primary/bin/python' replacement_loss_v34_closeout.py development
```

Never reuse these roots for a scientific rerun or open confirmation against
this ineligible development result. Earlier frozen outcomes remain unchanged.

## What remains

The local next obligation is to explain the weak first-site intervention
errors using adaptation-only secants, Jacobians, and activation-boundary
diagnostics. If those support a concrete mechanism, freeze a new derivative/
secant-supervision hypothesis with fresh globally disjoint data and unchanged
all-stage gates. V34 does not authorize continuing tuning inside this packet.
A different scalar cap or a larger run is not justified as a guaranteed fix.

Claim ceiling:
`LOCAL_SYNTHETIC_GRADIENT_BALANCE_AND_CAUSAL_MAP_QUALIFICATION_ONLY`.
No qualifying method was selected. Nine-case family/seed confirmation,
noncausal comparator evaluation, and pretrained whole-block transfer remain
unperformed. Stronger science still requires blind external replication and
authorized independent review; alignment requires human calibration;
production requires separate operational authorization. Nothing here proves
metaphysical identity or consciousness, universal architectural replacement,
or novelty priority.

