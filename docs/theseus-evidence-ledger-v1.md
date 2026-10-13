# Theseus Lab evidence and claim ledger v1

As of 2026-10-07.

State slices: `theseus-lab-sandbox-v1`,
`theseus-additive-module-validity-v1`,
`theseus-sequential-causal-replacement-bound-v1`,
`theseus-replacement-loss-v31-scm-interchange-v1`,
`theseus-replacement-loss-v32-causal-ladder-v1`,
`theseus-v32-row-disjointness-audit-v1`,
`theseus-replacement-loss-v33-relu-geometry-v1`,
`theseus-replacement-loss-v33r1-relu-geometry-v1`,
`theseus-replacement-loss-v33r1-loss-scale-audit-v1`,
`theseus-replacement-loss-v34-gradient-balance-v1`,
`theseus-v34-adaptation-jacobian-audit-v1`,
`theseus-replacement-loss-v35-secant-jacobian-v1`,
`theseus-checkpoint-exposure-feasibility-v1`,
`theseus-pretrained-development-transfer-v37`,
`theseus-pretrained-development-transfer-v37r1`,
`theseus-gpt2-visible-target-audit-v1`,
`theseus-reused-gpt2-early-ladder-v38`,
`theseus-v38-token-coverage-audit-v1`,
`theseus-reused-gpt2-all-token-ladder-v39`,
`theseus-v39-first-mlp-causal-diagnosis-v1`,
`theseus-v39-training-probe-replay-v1`,
`theseus-v39-first-mlp-fit-factorial-v3`,
`theseus-all-runs-audit-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`. The worktree was
already dirty when this synthesis was prepared. This document changes no
frozen protocol, threshold, result, or artifact. It contains aggregate claims
and links only; prompts, model assets, captures, candidates, predictions, and
row-level outcomes remain in owner-controlled external custody.

## Current evidence

| Question | Evidence, outcome, assurance, reviewer | What it supports |
| --- | --- | --- |
| Exact additive split | [Theorem](additive-module-theorem.md) and [five-family adapter result](additive-module-validity-v1-results.md): `REPRODUCED_LOCAL`, followed by `SEGREGATED_INTERNAL_REVIEW`; external reviewer `NOT_SUBMITTED` | The distributive identity is proved for the stated additive-module class. Five finite checkpoint adapters reproduced it; this does not prove all architectures. |
| Sequential replacement error | [Conditional ladder bound](sequential-causal-replacement-bound-v1.md): formal derivation, not independently reviewed | Local endpoint errors plus downstream sensitivity bounds imply an output-error bound; bounding both endpoints implies a causal-effect-error bound. The premises are not established by finite training averages. |
| Independently fitted affine projection | [V8](replacement-loss-v8-results.md): all 9 cases passed; `REPRODUCED_LOCAL`; reviewer `EXTERNAL_REVIEW_UNAVAILABLE` | A ridge-fitted MLP output projection predicted the frozen held-out effects and retained measured behavior for the exact checkpoints and protocol. It is a narrow local observation. |
| Independent replacements across families | [V3](replacement-loss-v3-results.md): 6/9 cases passed; overall `FAIL`, `REPRODUCED_LOCAL`; reviewer `EXTERNAL_REVIEW_UNAVAILABLE` | OPT and SiLU-gated cases passed; all GPT-2/GELU cases missed activation-rescue gates, with one behavior miss. |
| Progressive causal replacement ladder | [V9](replacement-loss-v9-results.md): 21/27 ridge stages passed retention; overall `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED`; external gate `EXTERNAL_REVIEW_UNAVAILABLE` | The paired causal-effect comparison beat the named CKA and logit-distillation baselines under this packet, but the full ladder did not meet its retention gates. The registered reviewer Agent Card and health route returned HTTP 404 on 2026-10-07. No task was submitted. Its CKA arm was not a full reproduction of Network of Theseus. |
| Fresh-checkpoint V8-style run | [V28](replacement-loss-v28-results.md): all nine treatment cases passed their listed behavior, causal, and activation-rescue checks; the frozen aggregate outcome is `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | The separate mapping-aligned equivalence gate exceeded its frozen numeric limit. The packet remains a failure; the threshold must not be retuned or its case passes presented as an overall pass. |
| Nonlinear replacement method qualification | [V30](replacement-loss-v30-results.md): 1/9 synthetic cases passed every gate; overall `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | The fixed GELU student beat controls and usually predicted effect direction, but failed predeclared causal-magnitude or worst-case-logit gates in eight cases. This used no trained checkpoint. |
| Paired causal loss on a synthetic trained decoder | [V31](replacement-loss-v31-results.md): 2/3 seeds passed all gates; overall `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | All five arms achieved exact task accuracy across the three seeds, and the treatment passed its absolute held-out effect gates. The paired-causal objective failed its required improvement over pointwise supervision in seed 3102. A canonical post-run audit resolved the runner's key-type reconciliation mismatch; original receipts were preserved. |
| Two-block nonlinear ladder on task-critical guides | [V32](replacement-loss-v32-results.md): 6/9 cases passed listed gates; overall `FAIL`, numeric procedure `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED`. [Row audit addendum](replacement-loss-v32-row-overlap-audit-v1.md): `INTERNAL_PASS` | Both complete nonlinear blocks were newly initialized and fitted. GELU and SiLU-gated guides pass three seeds each; ReLU fails small-effect tails and adversarial restoration. The later audit found 24 cross-case preparation overlap pairs, zero within-case overlap, and zero challenge/preparation overlap; globally independent preparation draws across seeds are not established. This is a synthetic feedforward task with fixed-coordinate interventions, not a pretrained Transformer result or a CKA superiority proof. |
| ReLU geometry/weighting preparation | [V33](replacement-loss-v33-results.md): `INCONCLUSIVE`; engineering `INTERNAL_PASS`; scientific replica `NOT_RUN`; reviewer `NOT_SUBMITTED` | Global disjointness caught adjacent-seed stream reuse before fitting any replacement. Frozen source and partial custody were retained. It supplies no scientific replacement finding. |
| Fresh globally disjoint ReLU geometry/weighting development | [V33R1](replacement-loss-v33r1-results.md): `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | ReLU/global passed 29/36 stage-stratum cells versus GELU/global 19/36, but complete cases remain 0/3 for every arm. Direct per-example weighting passed 0/36 cells under either geometry. After reconnecting both drives, all 16 reconciliation checks, 205 manifested files per run, and both frozen source archives verified. No confirmation opened. |
| Initial causal-loss gradient scales | [V33R1 loss audit](replacement-loss-v33r1-loss-audit-results.md): `INTERNAL_PASS`, separate reproduction `NOT_RUN_SEPARATELY`; reviewer `NOT_SUBMITTED` | Six readouts across three existing guide cases show per-example effect gradients 31.95–1009.88 times the behavior-plus-activation anchor gradient at initialization; global effect gradients are .391–.608 times it. No optimizer update or heldout query was made. This measures an initial scale mismatch, not the cause of completed training failure or a qualifying method. |
| Bounded causal gradients and map transport qualification | [V34](replacement-loss-v34-results.md): `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | The fixed half-anchor gradient cap restored behavior, activation, rescue, and input-effect checks in 36/36 cells relative to unscaled row weighting, yet passed only 15/36 complete cells versus global normalization 29/36. All arms remain 0/3 complete cases. Representation-qualified maps were 40/45; only 4/45 qualified causally, with no complete aligned snapshot. All 16 reconciliation checks, 167 manifested files per run, 108 native scores, and 30,600 gradient readouts per run verified. No confirmation opened. |
| Adaptation-only endpoint/derivative diagnosis | [V34 derivative audit](replacement-loss-v34-derivative-results.md): `DIAGNOSTIC_SUPPORTS_ENDPOINT_OR_DERIVATIVE_HYPOTHESIS`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | All three primary guide cells met the frozen continuation rule. Prefix Jacobian nRMSE was .270–.277 despite activation nRMSE near .046; weak-secant/error associations were .916–.938 with zero shared-suffix error. All 45 diagnostic cells, 11 reconciliation checks, and 80-file manifests per run verified. No training or heldout evaluation occurred; this is a development hypothesis, not a proven cause of V34 failure. |
| Learned two-block Jacobian-supervised ladder | [V35](replacement-loss-v35-results.md): development `DEVELOPMENT_QUALIFIED`; fresh confirmation `PASS`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | The selected derivative arm passes all nine synthetic cases (three activation families × three seeds), all 108 native stage-stratum cells, and controls. Global causal/endpoints pass 3/9 cases each; pointwise/output/CKA pass none. Confirmation closeout verifies all 16 reconciliation checks, 816 manifested files, 648 native cells, and 91,800 gradient readouts per run. Only 9/27 selected aligned snapshots are available and pass; no case has complete mapping availability. This supports finite native causal retention, not pretrained transfer, full mapping equivalence, universal identity, novelty, or independent science. |
| Causal/value loss variants on reused OPT | [V17](replacement-loss-v17-results.md), [V19](replacement-loss-v19-development-results.md), [V20B](replacement-loss-v20b-development-results.md), and [V21](replacement-loss-v21-development-results.md): no qualifying candidate; completed procedures reproduced locally; reviewers `NOT_SUBMITTED` | V17 missed strict causal gates; V20B retained behavior in all nine cells while only 3/9 passed the causal gate; V21 did not qualify any cell. These results do not supply a qualifying alternative replacement method. |
| Pretrained whole-MLP transfer intake | [V36 result](replacement-loss-v36-results.md), slice `theseus-pretrained-whole-block-transfer-v36`: `INCONCLUSIVE`; previous engineering `INTERNAL_PASS`; scientific reproduction `NOT_RUN`; reviewer `NOT_SUBMITTED` | Both drives reconnected and V35 custody verified. The first intake encountered non-JSON legacy Cloud Run exports before checkpoint selection or learning. Frozen source and the failure were preserved. No pretrained retention finding. |
| Digest-bound intake recovery and public acquisition | [V36R1](replacement-loss-v36r1-results.md): `NO_ELIGIBLE_CHECKPOINT_COHORT`; engineering `INTERNAL_PASS` (29 tests); scientific reproduction `NOT_RUN`; reviewer `NOT_SUBMITTED` | The fresh intake bound 1,613 historical files with three exact opaque quarantines. Filtered catalogs supplied no complete cohort. The separate catalog-contract diagnosis did not change or retry R1. |
| Catalog compatibility and checkpoint feasibility | [V36R2](replacement-loss-v36r2-results.md): `NO_ELIGIBLE_CHECKPOINT_COHORT`; engineering `INTERNAL_PASS` (31 tests); scientific reproduction `NOT_RUN`; reviewer `NOT_SUBMITTED` | A fresh adapter removed the server parameter hint and retained local 250M and all scientific gates. It bound 1,626 historical files and screened 200-entry catalogs per family. One GPT-2 nomination qualified; OPT/Llama did not. No weights, replacements or held-out outcomes were opened. Recorded-identity exclusions include metadata-only screening; this is no global scarcity or method-retention finding. |
| Retained checkpoint-exposure classification | [Audit V1](checkpoint-exposure-v1-results.md): `INTERNAL_PASS`; separate reproduction `NOT_RUN_SEPARATELY`; reviewer `NOT_SUBMITTED` | Bound 1,634 inputs, 48,262 identities and 1,469 weight paths. Classified 48,233 metadata-only identities, 17 with weight evidence and 12 with execution evidence; three opaque exports remain unresolved. All inputs rechecked. This allows metadata-known development within retained custody, not global never-used claims. |
| Metadata-known development intake | [V37](replacement-loss-v37-results.md): `INCONCLUSIVE`; engineering `INTERNAL_PASS`; scientific reproduction `NOT_RUN`; reviewer `NOT_SUBMITTED` | HTTP 429 stopped the fresh screen before nomination freeze, weights or fitting. Its retained cards/configs informed a separately frozen continuation; the terminal root was not retried. |
| Actual trained GPT-2 qualification | [V37R1](replacement-loss-v37r1-results.md): `NO_QUALIFIED_TARGET`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | Acquired a pinned 124M Spanish GPT-2 and ran actual removal in two isolated runtimes. Energy-selected layer 11 failed both visible gates. No fit/challenge opened; all nine reconciliation fields and numeric closeout matched. No full three-family cohort. |
| All-layer visible diagnosis | [Audit V1](gpt2-visible-target-audit-v1-results.md): `NO_VISIBLE_CRITICAL_PAIR`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | All twelve layers were ablated on retained visible rows. Only early layers 0 and 2 qualify; no second-half target. This supports narrower reused development, not unseen retention. |
| Independently learned early GPT-2 ladder | [V38](replacement-loss-v38-results.md): `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | Two-stage replacements were fitted for three seeds under an explicitly reused/early-block scope. Both arms pass 0/3 cases and 0/24 complete cells. Rescue passes 24/24, while behavior and joint-effect forecasts fail every cell. Closeout verifies 196 files, 96 native cells and 24,576 gradient readouts across exact primary/replica. No pretrained retention upgrade or practical novel proof. |
| Adaptation token-coverage diagnosis | [Audit V1](v38-token-coverage-audit-v1-results.md): `COVERAGE_GAP_SUPPORTED`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | All twelve fitted stages show prefix nRMSE 20.80–46.44 times final-token nRMSE. All-token output variation is 8.30–32.82 times the fitted final-token variation. Actual earlier-replacement contexts replay exactly; seven reconciliation checks pass. Zero training and new held-out queries. This motivates all-token fitting, without attributing every failure to coverage or proving unseen retention. |
| All-token learned early GPT-2 ladder | [V39](replacement-loss-v39-results.md): `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | All twelve fits per role supervise every adaptation token. Both arms pass 0/3 cases and 0/24 complete cells. JVP logit nRMSE and rescue pass 24/24, but full behavior and native causal effects pass 0/24; first-stage causal retention is also 0/12. Exact candidates, predictions, oracles and verdicts reproduce. Closeout verifies 196 files, 96 cells, 24,576 gradient readouts and 24 token pools. No practical novel proof or broader pretrained retention upgrade. |
| First-MLP local response and decoder diagnosis | [Diagnosis V1](v39-first-mlp-diagnosis-v1-results.md): `LOCAL_DERIVATIVE_MISMATCH_SUPPORTED`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | JVP final-token value nRMSE is .101–.115, versus full-Jacobian nRMSE .883–.889 and median finite-patch error .886–.894 on fixed adaptation samples. Ordered local-delta transport projections are .884–.938 of decoder error; these are path-dependent accounting, not unique causal percentages. Each role retains 288 Jacobians, 1,152 local pairs and 384 decoder pairs; 48 actual no-ops match. No new fitting, held-out queries or retention pass. |
| Original training-probe replay | [Replay V1](v39-training-probe-replay-v1-results.md): `TRAINING_DIRECTION_UNDERFIT_SUPPORTED`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | Projected final-token error remains .833–.887 along reconstructed training directions, versus .872–.897 on fresh directions. The sampler matches the original core fitter in a contract test. Each role reproduces 36 projection cells; no model forwards, training or held-out queries. The coefficient is uncapped in most updates, without isolating optimizer, weighting, update budget or capacity as the cause. |
| First-MLP fit-factorial diagnosis | [V3](v39-first-mlp-fit-factorial-v3-results.md): `FAIL`, `REPRODUCED_LOCAL`; reviewer `NOT_SUBMITTED` | On one reused GPT-2 guide, 4,096 updates reduced adaptation training-direction error by 22.4% under the frozen factorial rule; 4× capacity reduced it by 12.0% and did not meet that rule. No factorial cell passed held-out retention across all seeds; joint causal effects pass 0/48 strata. Same-owner, same-host reproduction only. |
| Finite-observation identifiability | [Conditional endpoint bound](causal-effect-identifiability-boundary-v1.md) | If a shared downstream map is Lipschitz and factual and intervened endpoint errors are uniformly bounded, causal-effect error is bounded. Finite average fit scores alone do not establish those premises. |

Key frozen packet SHA-256 identities: V3 `d9cf01d6caa2ac6145c0b4a7e5ddb9c2b105978993d77c1d58b5d0ad4dfc9b83`; V8 `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`; V9 `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`; V28 `f341a7f8e673dd2248f38b73659da25dcf62088b29176e01e1b914f1e5925b08`; V30 `a2e8860f30642be8a2c53386c64b87db7e17e6e1a7b3212e4a4274b72cd0cda3`. Each linked result document contains its source revision, execution and replication status, resource record, and custody details.

V31 packet SHA-256: `0922be564b628c5ac4dafd0271ecf96969001fccb0abbbcc833e0d9382818513`.

V32 packet SHA-256: `a50766c1611c72a444fa8213378761adaef84811f5f3afdc0181aafce7f11c67`.
The [historical audit](theseus-all-runs-audit-v1.md) binds 186 source documents,
134 aggregate receipts, and 70 accessible custody roots through V31. It
preserves copied, failed, partial, and unknown records separately.

V33 packet SHA-256: `3de9ce8a82cf7695a38e8b22fd768a0e1533d6e4f667505e454c68c481108349`.
V33R1 packet SHA-256: `09422f3caeecea46614c5bf779ee5ae80740d685e0285dde89f399ec8e01f3ad`.
V34 packet SHA-256: `91497500548479499d0f7de61d524adb9f215cd93118414e052644f67289ab91`.
V34 derivative packet SHA-256: `932443312451ba4e5cc67d7eac4f743e61ae8f1d97edb076f4633766a6ea9e84`.
V35 packet SHA-256: `2b5a23c1a7c1fd5979b384f559ae152cd6f4febd8c5f2d455db05a4abd569656`.
V35 confirmation freeze SHA-256: `e58f96fa57c39b523087602227c475ecbdbcd05b6f38b88de259b51d2f73f72c`.
V36 source packet SHA-256: `370a074309eeea6b57cf5b529892932d5d93474a9381b7ffd47bc1a6bb367458`.
V36 source freeze state: `FROZEN_SOURCE_CUSTODY_PENDING`; it binds the protocol,
fixture/selection rules, evaluator, tests, exact dependency lock, parent source
reports, and historical checkpoint registry before fresh checkpoint selection.
That freeze records its original preparation state; the later execution outcome
is `INCONCLUSIVE`, retained without changing the freeze.
V36R1 packet SHA-256: `81d58c3867fcae4a6b48549153d6b3b6769df49b8bdc04b382de5e53be270de7`.
V36R2 packet SHA-256: `020f59238f0781538d68e617826b15deb4e68b7b8feed0b1f2f665e32e8b7755`.
V38 packet SHA-256: `c9739b055049219d1c9f21ed45a71265df1af6886ea0933249fff41a226aa013`.
V38 token-coverage audit packet SHA-256: `ae4fd5ae25807c194aa3b1dabfc2bcd6041b52def932cee72968d897ded92a40`.
V39 all-token source packet SHA-256: `6fd6849dd663380b5bb2103a57f0989108c86c123ac9ef46759d48330c5a9f98`.
V39 first-MLP diagnosis packet SHA-256: `4636596602a3c2f7cf056d53f2d8f6fe22893c245d6e89a85837558620ddeba8`.
V39 training-probe replay packet SHA-256: `952f46661f9eacf97ebddbef4a9827a94e38960f2bed0cbbbf14f6b67409d76e`.
V39 first-MLP fit-factorial V3 packet SHA-256: `bc233412da532b1eee72611febe7b510086dd0b8c5de90ec3e93769865b37f3d`.

## Claim ceiling

The exact additive-module split is a mathematical theorem, not a theorem that
arbitrary AI systems preserve identity through replacement. The empirical
record supports bounded local observations about particular modules,
checkpoints, prompts, interventions, and evaluators. It does not establish
whole-model mechanistic identity, architecture-universal replacement,
subjective experience, human-equivalent alignment, independent scientific
acceptance, or production readiness.

The strongest positive learned-replacement result on pretrained checkpoints is V8's
`REPRODUCED_LOCAL` affine-projection observation. V28 is fresh-checkpoint
evidence with a frozen overall `FAIL`; V30 is a reproduced synthetic-method
failure. These verdicts must remain separate. `REPRODUCED_LOCAL` means a
same-owner second environment reproduced the frozen procedure and outputs; it
does not mean `INDEPENDENT_ACCEPT`.

V32 adds a stronger whole-nonlinear-block observation on six synthetic trained
guide cases, where the targets are behaviorally necessary under visible data.
Its entire nine-case protocol is still FAIL; the later row audit also limits
claims of independent preparation draws across seeds. Neither positive subset replaces
the scope or status of the other experiment.

V35 adds a qualifying native two-block ladder on nine fresh synthetic guides,
after selecting the method on three separate ReLU guides. Globally disjoint
inputs, prebound predictions, same-owner reproduction, and custody closeout
passed. Its nonlinear blocks are independently initialized; stem/head are
initially retained and the final candidate is task tuned. All nine native
cases pass, but complete mapping-aligned retention remains unavailable.
This strengthens the finite synthetic observation without changing earlier
failures or proving transfer to pretrained Transformer blocks.

## Novelty boundary

The additive split follows directly from distributivity, so the identity alone
is not a novelty claim. *Network of Theseus* already studies progressive
component replacement, representation alignment with CKA and D-MNN, and
cross-architecture conversion including GPT-2 to recurrent networks
([paper](https://arxiv.org/abs/2512.04198)). Causal abstraction work already
formalizes mechanism replacement and broader mechanism transformations for
mechanistic interpretability ([paper](https://arxiv.org/abs/2301.04709)).

A narrower possible contribution is to train a replacement on disjoint data,
freeze predictions, and score its causal effects on unseen interventions,
alongside behavior and activation-rescue checks. V3, V8, V9, and V28 investigate
parts of this claim. The focused [prior-art assessment v2](replacement-loss-prior-art-v2.md)
adds IIT, causal distillation, and CMR as comparators; the [comparison-readiness
audit](replacement-loss-not-comparison-readiness-v1.md) finds the local V9 CKA
arm is not a full Network of Theseus reproduction. Novelty remains
`NOVELTY_NOT_ESTABLISHED`; the search was targeted, not systematic, and local
results do not establish priority.

V35's derivative objective also has prior art in
[Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html) and
[Sobolev training](https://arxiv.org/abs/1706.04859). Its frozen causal ladder
and comparator results do not establish a novel theorem or priority.

## Next gate

Do not launch another local symmetry/permutation sweep or treat V30 as the
first real-checkpoint test. Real-checkpoint replacement studies already exist
in V3, V8, V9, and V28. Another same-owner method variant would add little
confidence unless it tests a distinct, frozen hypothesis and preserves
separate development and held-out evaluation.

V33R1 and V34 both completed globally disjoint paired development,
same-owner reproduction, and custody closeout without an eligible candidate.
V34 tested the distinct gradient-balance hypothesis motivated by the initial
loss-scale audit. It prevented the observed behavior/activation/rescue collapse
in the unscaled row arm, but did not solve the weak first-site causal effects
or exceed global normalization's complete-cell count. The fixed cap's failure
must not be generalized to all balancing methods or repaired inside V34.
Its separate adaptation-only map tests also show that a representation fit
and inverse do not establish measured intervention transport.

The adaptation-only diagnosis and conditional V35 sequence are now complete.
The diagnosis reproduced its continuation rule; V35's Jacobian arm alone
qualified across all three development guides. Its frozen selection then
passed all nine fresh family/seed confirmation cases and reproduced exactly,
with the six declared comparators and all-stage causal/rescue gates retained.
The procedural CKA arm is still not a full Network of Theseus reproduction.

The next scientific obligation remains pretrained whole-block transfer.
V36's directional module JVPs are a new transfer hypothesis, not V35's
full-prefix Jacobian objective. After drive recovery and V35 custody
verification, V36 stopped on historical JSON; fresh V36R1/R2 packets repaired
intake/catalog handling without editing terminal source. The frozen exposure
audit is now complete; metadata-known development was named prospectively.
V37's rate-limited screen, V37R1's real target stop and the all-layer visible
diagnosis were retained. V38 then completed actual independent two-block
fitting, sealed unseen predictions, exact second-runtime reconstruction and
numeric closeout on a narrower reused GPT-2 scope. Its scientific result is
FAIL: all three cases miss, despite every activation-rescue cell passing.
The separately frozen token-position audit reproduces a large prefix/final
coverage gap on adaptation data. V39 then completed all-token fitting, fresh
sealed predictions, exact second-runtime reproduction and numeric/token
closeout. It remains FAIL: every JVP logit-norm cell passes, but no complete
behavior or causal cell passes. The first replacement alone fails all twelve
causal cells. The first-MLP diagnosis now reproduces a large local derivative
and finite-response error; fixed decoder accounting puts most error projection
in local-delta transport. The separately frozen training-probe replay also
finds large residuals along the original fitting directions. These are
adaptation-only observations, not a new retention pass or unique causal
attribution. The V3 first-MLP factorial now shows that 4,096 updates improve
the adaptation training-direction error by 22.4% under its frozen support
rule, while 4× student capacity improves it by 12.0% and misses that rule.
Still, no factorial cell passes held-out retention across all three seeds;
joint causal-effect prediction passes 0/48 strata. This isolates an
adaptation-fit effect, not a cause of the held-out failures. Do not extend
optimizer or capacity sweeps as a presumed remedy. A follow-on local packet
needs a distinct causal-transfer hypothesis, fresh disjoint fixtures, sealed
predictions, unchanged held-out gates, and an explicit stop rule. The full
three-family/nine-case obligation remains open, with eligible
OPT/Llama guides and a separately frozen broader protocol still required.
Do not label metadata-known/reused models wholly unseen. A larger cap or
broader search does not guarantee method or target qualification.
Prior checkpoint studies already exist; their affine projection findings do not
prove transfer of V35's independently learned nonlinear blocks. The persistent
mapping-unavailability finding also merits a separate adaptation-only
diagnosis; do not change V35's coordinates, thresholds, or map gates.
Do not rerun terminal roots, relax the p90 threshold, discard early-stage
misses, or claim a larger run guarantees success. Keep native behavior,
representation alignment, and causal transport separate.

The authority gate for stronger independent science is still a blind
replication and authorized external review. V8 reviewer status is
`EXTERNAL_REVIEW_UNAVAILABLE`; no review task was submitted. Per the current
local-only scope, no external transfer or submission was performed here.

The experiment state is complete only to this local evidence ceiling. V31
does not meet its paired-causal-over-pointwise gate consistently; V32 shows
six whole-block ladder successes but misses the all-family retention gate.
V35 meets its finite native synthetic gate, while complete mapping-aligned
equivalence and pretrained transfer remain unestablished. Preserve every
earlier failure. Any follow-on packet
must state a distinct causal prediction on a task where causal supervision
can add information beyond pointwise matching, freeze code, thresholds,
model selection, and fixture rules before selection, and stop on a measured
miss. Blind external replication and authorized review remain the next gate
for stronger scientific claims; consciousness, alignment, and production
remain separate gates.
