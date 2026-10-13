# Replacement-loss V33R1 development results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v33r1-relu-geometry-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the frozen uncommitted packet.

## Literal outcome and current completion boundary

Closeout completed on 2026-10-03 after both external drives reconnected.

**Outcome: NO_QUALIFIED_DEVELOPMENT_CANDIDATE.** None of the four paired
configurations passed all native retention gates on all three fresh ReLU
guides. No variant was selected. No confirmation receipt, confirmation
guides, or confirmation challenge was generated. The frozen stopping rule
was followed; no parameter, threshold, seed, or failed result was retuned.

Assurance: **REPRODUCED_LOCAL**. Reviewer: `NOT_SUBMITTED`. All 16 saved
reconciliation checks were independently recomputed and matched the receipt;
all 205 manifested files per runtime and 20 source-archive files per runtime
verified. The terminal result digests match those captured before the drives
disconnected. Both drives are available again, and final custody verification
is complete. This is same-owner local reproduction, not independent scientific
acceptance. The terminal experiments were not rerun.

The [machine aggregate](replacement-loss-v33r1-results.json) contains the
captured metrics, diagnostic results, and closeout bindings. Raw inputs,
challenge entropy, guide
and candidate weights, maps, predictions, oracle and restoration outcomes,
source archives, and detailed execution receipts remain in owner custody;
their present availability is `AVAILABLE_AND_VERIFIED`.

## Preparation fault and correction

The initial [V33 packet](replacement-loss-v33-results.md) stopped
`INCONCLUSIVE` during preparation, before replacement fitting. Adjacent
case seeds and small partition offsets reused streams across cases.
Its packet and partial artifacts were preserved. V33R1 freezes a correction:
namespaced SHA-256 stream identities, global exact-row disjointness checks,
an added regression test, and exclusion of V33 partial preparation inputs.
All 17 constructed engineering tests passed in both fresh locked runtimes.

The [V32 addendum](replacement-loss-v32-row-overlap-audit-v1.md) found 24
cross-case preparation overlap pairs in the predecessor, zero within-case
overlaps, and zero challenge/preparation overlap rows. V32's numeric
reproduction and frozen FAIL remain; globally independent preparation draws
across its case seeds were not established. V33R1's checks cover every
case and partition, prior inputs, and challenge endpoints.

## Frozen paired design and measured outcomes

The [protocol](replacement-loss-v33r1-relu-geometry.md) was frozen before
new scientific data generation. Three fresh ReLU guide seeds were paired
across GELU/ReLU student geometry and global/per-example effect normalization,
at equal parameter count and identical initialized weights, data, minibatches,
optimizer, and budgets. Each configuration replaced block one, replaced
block two, and then trained the task with all parameters unfrozen. All three
guides passed visible-only qualification: sign accuracy 98.75–99.17%; removing
either block reduced accuracy by 47.85–50.93 percentage points.

All 36 candidate snapshots and their adaptation-fitted maps were frozen
before fresh challenge entropy. Native and qualified aligned predictions,
including negative controls, were bound before guide challenge outcomes and
actual restoration were evaluated. Four strata used 1/2/4/8-coordinate
activation interchanges, new two-coordinate input counterfactuals, and the
expanded adversarial input domain. These observations are development data,
not an independent confirmation set or additional replicas.

| Configuration | Complete cases /3 | All-gate cells /36 | Block-one cells /12 | Block-two cells /12 | Final cells /12 |
|---|---:|---:|---:|---:|---:|
| GELU + global | 0 | 19 | 7 | 5 | 7 |
| ReLU + global | 0 | 29 | 9 | 10 | 10 |
| GELU + per-example | 0 | 0 | 0 | 0 | 0 |
| ReLU + per-example | 0 | 0 | 0 | 0 | 0 |

Matching ReLU geometry improved the measured global-weighted retention
cells, and final adversarial first-block restoration RMSE fell from
.09219/.13305/.13761 to .05365/.06223/.04481. The complete ReLU/global
cases still fail: first-block ordinary effect p90 errors are
.53877/.61873/.53087, versus the unchanged ≤.50 gate. At the final snapshot
they are .31525/.62546/.65663. The first seed's successful final snapshot
does not repair its failed intermediate snapshot. Every ReLU/global failed
cell is an ordinary first-block effect-tail miss; its other measured
retention gates passed.

Direct per-example normalization regressed behavior and causal magnitudes:
neither geometry passed a single complete cell. Final adversarial output
RMSE ranged .34656–.41605, versus ≤.10. This refutes the simple frozen
weighting proposal under this configuration and budget. It does not prove
all relative-effect weighting is ineffective. The predeclared adaptation-only
[initial loss-gradient diagnostic](replacement-loss-v33r1-loss-audit-results.md)
completed with `INTERNAL_PASS`; separate reproduction is `NOT_RUN_SEPARATELY`.
At the original first minibatch and stage-one initialization, per-example
effect-gradient norms were 31.95–1009.88 times the combined output-and-
activation gradient; global effect gradients were .391–.608 times that
gradient. This establishes an initial scale mismatch for these inputs.
It supports a fresh gradient-balancing hypothesis but does not establish
the cause of the complete training regression or qualify a replacement.

The positive guide/no-op controls passed 12/12 case-stratum checks. Random,
zero-module, and wrong-module controls passed the treatment-sensitivity
comparison in 48/48 arm-stratum checks. Repeated shared controls and the
replica are not additional independent model observations.

## Alignment comparison and limitations

Adaptation-only affine maps qualified at 48/60 active sites. All nine
arm-by-snapshot maps were available for each global-weighted geometry;
only 2/9 and 4/9 were available for GELU/row and ReLU/row respectively.
Missing aligned scores remain `ALIGNMENT_UNAVAILABLE`. Qualified aligned
scoring passed 1/36 cells for GELU/global and 2/36 for ReLU/global, so
ordinary activation fit and numerical invertibility did not establish
aligned intervention retention in this study. This is an additional
limitation of the fitted transport, not evidence against a CKA competitor:
the conditional CKA/pointwise/output comparison never opened.

The causal objective overlaps [Causal Distillation](https://aclanthology.org/2022.naacl-main.318/).
The planned staged CKA schedule follows [Network of Theseus](https://arxiv.org/html/2512.04198v1)
on small procedural models; no paper benchmark was reproduced. Three guide
seeds give descriptive paired evidence, not a powered statistical superiority
claim. These are trained synthetic feedforward guides, not pretrained
OPT/GPT-2/Llama checkpoints or a proof of universal identity.

## Identity, resources, custody, and remaining work

| Binding | SHA-256 |
|---|---|
| V33R1 packet | `09422f3caeecea46614c5bf779ee5ae80740d685e0285dde89f399ec8e01f3ad` |
| V33R1 freeze file | `0d049329ac3fec939182f2d7dc0a5fec32ebfa9d6e8f60053cd5afa530101088` |
| Primary terminal result, captured before disconnection | `0fec829d30ca4ad1e7a68917f18e370ea78b4b6ca1047fcf52dc59f53058197e` |
| Replica terminal result, captured before disconnection | `b09a358159e60b4ddfc180d3828b54890aa3baf615fe643e791388a97024ef8c` |
| Saved reconciliation receipt | `423c1300b18cbc66243254d58e7a9daf742c004dcec8fe721f40e42c202a8085` |
| Completed closeout validation | `eb8cf13ed0cfe274da977cf66c58882fab9c954072dec1b8733f971ebaa91fef` |

Primary frozen-phase resources: 35.996 seconds, 326,189,056 peak RSS bytes,
29,780,847 artifact bytes before the terminal JSON receipt. Replica resources:
37.081 seconds and 323,092,480 peak RSS bytes. Final retained artifact bytes:
30,504,048 primary and 30,504,796 replica, including terminal receipts.
The separate loss diagnostic took 3.230 seconds, peaked at 213,729,280 RSS
bytes, and retained 17,804 artifact bytes. Runtime: Python 3.12.13, Torch
2.6.0, NumPy 2.4.6, deterministic single-thread CPU, macOS 26.6.1 arm64,
fresh separate virtual-environment prefixes and site-packages. Monetary
spend $0; energy not measured. Installation and whole-session wall time
are not included in the frozen-phase timing.

Custody root:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v33r1`.
Runtimes:
`/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v33r1/toolchains/{primary,replica}`.
Local source-file checks confirm unchanged V32, V33, and V33R1 freezes.
The loss diagnostic has its own frozen source packet and ceiling. The
separate `closeout-validation.json` receipt binds the recovered manifests,
reconciliation, runtime/resource receipts, and diagnostic identity. It lives
outside both frozen terminal run directories. All pending closeout actions
are complete; confirmation remains unopened because no variant qualified.

Claim ceiling:
`LOCAL_SYNTHETIC_RELU_GEOMETRY_WEIGHTING_AND_CAUSAL_LADDER_ONLY`.
No pretrained-model, all-architecture, novelty-priority, metaphysical-identity,
consciousness, alignment, independent-science, or production claim is opened.
The next scientific obligation is a fresh development protocol testing a
bounded or balanced causal-gradient objective for small-effect tails,
while retaining behavior and validating
coordinate transport on adaptation-only causal interventions. Any future
optimization change needs new data and its own frozen selection/confirmation
rules. External review, human calibration, and production authorization remain
separate authority gates.
