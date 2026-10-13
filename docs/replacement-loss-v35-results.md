# Replacement-loss V35 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v35-secant-jacobian-v1`,
`global-experiment-assurance-lifecycle-v1`. As of 2026-10-03.
Revision: `202f459d6d89df9a038d78d58e0af825d287fcd8` plus digest-bound uncommitted packet.
[Protocol](replacement-loss-v35-secant-jacobian.md),
[development freeze](replacement-loss-v35-freeze.json),
[confirmation freeze](replacement-loss-v35-confirmation-freeze.json),
[aggregate receipt](replacement-loss-v35-results.json).

## Verdict and tested claim

Development: **`DEVELOPMENT_QUALIFIED`**, selecting `paired_jacobian`.
Fresh nine-case confirmation: **`PASS`**.
Assurance/reproduction: **`REPRODUCED_LOCAL`**. Reviewer: `NOT_SUBMITTED`.
Both phases completed in two distinct fresh runtime environments; all
reconciliation and custody checks passed. This closes the authorized local
diagnosis, development, conditional confirmation, reproduction, and reporting
sequence. It does not complete the broader practical Theseus proof program.

The tested claim is finite: independently initialized replacements for both
nonlinear blocks retained measured behavior, input/internal intervention
effects, activation rescue, and rescue predictions at all three saved ladder
stages and all four frozen challenge strata in nine fresh synthetic guide
cases. The selected method passed 108/108 native stage-stratum cells and all
positive/no-op and negative-control sensitivity gates. It passed three seeds
in each ReLU, GELU, and SiLU-gated family.

These are small synthetic feedforward networks, not OPT/GPT-2/Llama
checkpoints. Guides contain an 8→16 stem, two 16→64→16 blocks, and a 16→4
head. Replacements are newly initialized 16→96→16 modules. The guide stem
and head are initially retained; later stages train the replaced prefix and
the final truth-task stage updates all candidate parameters. The guide
remains frozen. This is a two-block learned ladder, not complete whole-model
replacement or universal identity.

## Diagnostic and paired development

The separate [adaptation-only V34 diagnosis](replacement-loss-v34-derivative-results.md)
met its frozen continuation rule on all three existing guides and reproduced.
It found sizeable prefix-Jacobian mismatch despite small activation error,
weak-secant associations, and zero first-stage shared-suffix error. It used
no optimizer updates or heldout evaluation. It motivated a fresh packet;
it did not repair V34 or establish the cause of that failure.

V35 froze source, thresholds, losses, fixture-generation rules, and exclusions
before generating fresh development data. Three ReLU guides, seeds
35101/35102/35103, were paired across four arms. All guides passed visible
qualification. Endpoint, secant, and Jacobian arms share endpoint/output/
global-effect supervision, initializations, minibatches, masks, and update
counts. Auxiliary secant/Jacobian gradients are capped at half the base norm.
The baseline delegates to the frozen parent fit and reproduces its parameter
bytes in the engineering fixture.

A complete cell requires every behavior, input/internal effect, activation,
actual rescue, and prebound rescue-prediction gate. A complete case additionally
requires all stages/strata and positive/negative controls.

| Development arm | Behavior /36 | Activation /36 | Rescue/prediction /36 | Input effects /36 | Complete cells /36 | Complete cases /3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Global causal | 35 | 36 | 35 | 35 | 29 | 0 |
| Paired endpoints | 35 | 36 | 35 | 35 | 29 | 0 |
| Paired secant | 36 | 36 | 36 | 36 | 32 | 1 |
| Paired Jacobian | 36 | 36 | 36 | 36 | 36 | 3 |

Only the Jacobian arm qualified for selection: all 36 cells and three cases.
The secant arm passed one complete case; endpoints and global passed none.
All 12 positive-control strata and all 12 negative sensitivity strata per arm
passed. Ordinary first-site p90 errors decreased 26.6–50.6% versus the paired
endpoint control across the nine guide/stage comparisons, exceeding the
predeclared descriptive 20% effect of interest. This is not a powered
superiority inference, and guides—not rows or stages—are sampling units.

The exact development primary, replica, and reconciliation digests were
bound into a separate confirmation receipt before any confirmation data.
No seed, arm, threshold, loss, cap, or update budget was tuned after readout.

## Fresh confirmation

Nine fresh cases use seeds 35201/35202/35203 in each declared activation
family. All nine guides passed qualification before replacement fitting.
ReLU guides use ReLU replacements; GELU/SiLU-gated guides use GELU
replacements. Six arms share guide/data/initialization/update pairing.
The derivative arm incurs extra arithmetic; no equal-FLOP comparison is
claimed. CKA is a procedural comparator, not a full Network of Theseus
paper reproduction.

| Confirmation arm | Behavior /108 | Activation /108 | Rescue/prediction /108 | Input effects /108 | Complete cells /108 | Complete cases /9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected Jacobian | 108 | 108 | 108 | 108 | 108 | 9 |
| Global causal | 102 | 108 | 102 | 108 | 94 | 3 |
| Paired endpoints | 102 | 108 | 102 | 108 | 95 | 3 |
| Pointwise | 103 | 108 | 102 | 108 | 45 | 0 |
| Output distillation | 101 | 0 | 33 | 108 | 0 | 0 |
| CKA | 31 | 0 | 0 | 36 | 0 | 0 |

The selected method passes 3/3 ReLU, 3/3 GELU, and 3/3 SiLU-gated cases.
Global causal and endpoint arms pass only the three GELU cases. Pointwise,
output, and CKA pass no complete case. These are descriptive paired results
for this frozen packet, not aggregate superiority or novelty claims.
All 36 positive-control strata pass; all 36 negative sensitivity strata pass
for selected, global, endpoints, pointwise, and output. CKA passes zero
negative sensitivity strata; its failure remains explicit.

Four strata use 256 base/donor pairs each: ordinary, input composition,
interchange composition, and bounded adversarial interventions. Coordinate
masks contain 1/2/4/8 sites; input counterfactuals alter two coordinates.
These are vector interventions, not language compositional prompts.
Candidates and map qualifications were frozen before fresh OS challenge
entropy; native and available aligned predictions plus controls were sealed
before guide challenge outcomes and actual guide-suffix restoration.
Replica rebuilt all candidates/maps before opening primary entropy, and
matched sealed predictions before oracle evaluation.

Ordinary first-site relative-effect p90, threshold ≤.50:

| Case | Block one | Block two | Final task |
| --- | ---: | ---: | ---: |
| gelu-35201 | 0.074 | 0.108 | 0.078 |
| gelu-35202 | 0.075 | 0.153 | 0.090 |
| gelu-35203 | 0.095 | 0.107 | 0.066 |
| relu-35201 | 0.324 | 0.282 | 0.239 |
| relu-35202 | 0.359 | 0.430 | 0.303 |
| relu-35203 | 0.424 | 0.280 | 0.267 |
| silu_gated-35201 | 0.197 | 0.267 | 0.254 |
| silu_gated-35202 | 0.113 | 0.330 | 0.271 |
| silu_gated-35203 | 0.229 | 0.312 | 0.310 |

Worst selected-method metrics over all 108 cells:

| Metric | Observed | Frozen gate |
| --- | ---: | ---: |
| Output normalized RMSE | .094039 | ≤.10 |
| Guide sign agreement | .976563 minimum | ≥.95 |
| Truth accuracy drop | .012695 | ≤.03 |
| Activation normalized RMSE | .097658 | ≤.15 |
| Actual rescue normalized RMSE | .094039 | ≤.10 |
| Rescue-prediction normalized RMSE | .078559 | ≤.10 |
| Internal-effect median relative error | .136879 | ≤.20 |
| Internal-effect p90 relative error | .430195 | ≤.50 |
| Internal-effect median cosine | .995975 minimum | ≥.90 |
| Input-effect p90 relative error | .289666 | ≤.50 |

Resolved-fraction, input-effect median/cosine, and all other required gates
also passed. The output/rescue margin is finite and relatively small; no
guarantee outside the sampled interventions follows.

## Mapping-aligned outcome remains limited

| Confirmation arm | Representation-qualified sites /45 | Causally qualified sites /45 | Aligned snapshots available /27 | Available snapshots passing |
| --- | ---: | ---: | ---: | ---: |
| Selected Jacobian | 45 | 24 | 9 | 9 |
| Global causal | 45 | 13 | 4 | 4 |
| Paired endpoints | 45 | 14 | 5 | 4 |
| Pointwise | 45 | 7 | 3 | 3 |
| Output distillation | 45 | 0 | 0 | 0 |
| CKA | 45 | 0 | 0 | 0 |

All 270 maps pass representation/inverse checks; 58/270 qualify causally.
The selected arm qualifies 24/45 sites. Its nine available aligned snapshots
are the block-one snapshots and all pass; all block-two/final snapshots are
`ALIGNMENT_UNAVAILABLE`. No selected case has mapping available at every
stage. Native retention drives the predeclared selection/PASS contract;
complete mapping-aligned equivalence remains unestablished. Unavailable
aligned checks are not passed scores.

In development, representation-qualified sites are 15/15 for every arm.
Causally qualified sites are global 1/15, endpoints 2/15, secant 4/15,
Jacobian 7/15. Secant/Jacobian each have one available snapshot; the secant
snapshot fails, while the Jacobian snapshot passes. No arm has a complete
all-stage aligned case. Representation fit and invertibility therefore
remain insufficient to establish this packet's causal transport.

## Frozen identity, reproduction, and engineering

- Packet: `2b5a23c1a7c1fd5979b384f559ae152cd6f4febd8c5f2d455db05a4abd569656` (41 source members).
- Development freeze: `af4ebfd06f3b3e2c5481307fa5c0d344f3d61d8c29b9af5a0a50fe2d48c88b86`.
- Confirmation freeze: `e58f96fa57c39b523087602227c475ecbdbcd05b6f38b88de259b51d2f73f72c`.
- Input exclusion manifest: `17a6c0aff07452a1704d93f8b27d8b066f57866a00c2937e7614a6bfe66fedc2`.
- Engineering receipt: `3ddfb635744713b0950c7144c9dbf29cd93e8235b28af86a63a1c3fe4eef9661`.

### Development

- Primary result: `a108fd2e635e00f7b5b84bc2db785682b78194eb0c6846a03161cdb5c6c644fd`.
- Replica result: `f1819011e3eb12a7170fa959a9d46b24c8a308527e889ec4385e4fe3f40f9d0f`.
- Reconciliation: `d20ed3a6298d3fe332dcc18380cc04875d799ec47fed9fce85f8acc8a9915698`.
- Closeout validation: `8953c06d732db68f2e197d8a9b054ef989242adfc3f02850da085e5d1f2f63c3`.
- Challenge fixture: `d7306e7a9264e3253c6c14e177baa6004b839ba405346c9aaba763774b22bef9`.

### Confirmation

- Primary result: `6f196a9e9d5ab71dc9fa70849dc41b2d9633c65abcbc8ff88183edd01c1a09c6`.
- Replica result: `12e302341ae362377c67c70fe19abf940a0d95891704ad0f078a9795b3569b6f`.
- Reconciliation: `418c4722e1058f884a7eedc3b1ef55a098460241fb83f34153d1d76d4d76bba3`.
- Closeout validation: `90aa0845bc7403b518dc5157322a973de15d48b3882c654cd151e046e057798a`.
- Challenge fixture: `f0e0b38994b40e2667d177c2bd64b04d5633f452c27f11a057bf5341596d557a`.

Each fresh Python 3.12.13 environment passed all 32 engineering tests before
the source freeze and scientific data. Coverage includes trainable prefix
derivatives, baseline parameter
equality, every-update caps, fixture namespaces/disjointness, terminal-root
and crash contracts, mapping qualification, numeric scoring, and conditional
confirmation. Initial prefreeze test-contract and lock-newline errors were
fixed and retained in the engineering receipt; no frozen scientific packet
was repaired or retried. The separate diagnosis's eight tests per runtime
also verified analytic derivative adapters against autograd.

All 16 reconciliation checks passed in each phase. Development closeout
verified 220 manifested files, 42 archived source/freezes, 144 native score
cells, and 40,800 gradient readouts per run. Confirmation closeout verified
816 manifested files, 43 archives, 648 native cells, and 91,800 readouts per
run. Scores were recomputed separately from saved arrays, and stored rescues
matched actual guide-suffix outputs exactly. Candidate/map/trace/fixture/
qualification/prediction/oracle/metric identities matched across runtimes.
Selected auxiliary weighted norm ratios stayed ≤.50 within tolerance.

Exact global disjointness passed for 41,472 development input rows and
118,272 confirmation input rows per runtime, including predecessor
exclusions; confirmation additionally excludes development inputs.
The corpus exclusion binds 22 predecessor artifacts. Replica rows are
identical observations, not extra samples. Shared paired challenge inputs
do not add independent seeds.

Runtimes use distinct prefixes and site-packages on the same owner/host,
matching Python, exact package lock, macOS arm64 toolchain, deterministic
one-thread CPU, float32 training, and float64 map/norm arithmetic.
This is `REPRODUCED_LOCAL`; engineering and audit checks are
`INTERNAL_PASS`. No separate agent or independent reviewer acceptance is
claimed.

## Resources, custody, and read-only verification

| Phase/runtime | Phase seconds | Peak RSS bytes | Final custody bytes |
| --- | ---: | ---: | ---: |
| development primary | 71.70 | 359153664 | 26951125 |
| development replica | 72.07 | 357203968 | 26951875 |
| confirmation primary | 274.44 | 477872128 | 114813622 |
| confirmation replica | 282.09 | 481198080 | 114814370 |

Every phase stayed within 1800 seconds, 4 GiB RSS, and 1 GiB artifacts.
Final sizes include result/reconciliation receipts; runner resource byte
counts precede them. Baseline inner-loop wall checks are inherited from the
parent; wall budget is checked between arms. No hard-preemption guarantee is
claimed. Runtime installation used the local cache and exact lock. Monetary
cost **$0**; energy and machine amortization `NOT_MEASURED`.

Raw inputs, guide/candidate weights, entropy, traces, maps,
predictions, oracle outcomes, and source archives remain in owner-controlled
external custody:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v35`.
Terminal roots are `development/{primary,replica}` and
`confirmation/{primary,replica}`; qualification/closeout receipts are outside
those immutable roots. Runtimes:
`/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v35/toolchains`.
Only code, protocols, tests, scalar results, and digest manifests enter the
repo. No provider, cloud worker, paid execution, external review submission,
publication, commit, or merge occurred.

Read-only verification from this repo:

```sh
PYTHONDONTWRITEBYTECODE=1 '/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v35/toolchains/primary/bin/python' replacement_loss_v35_runner.py verify
PYTHONDONTWRITEBYTECODE=1 '/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v35/toolchains/primary/bin/python' replacement_loss_v35_closeout.py development
PYTHONDONTWRITEBYTECODE=1 '/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v35/toolchains/primary/bin/python' replacement_loss_v35_closeout.py confirmation
```

Never reuse terminal roots for scientific reruns. Earlier outcomes remain
unchanged, including V34's failed development and V32's row-overlap limitation.

## What this supports and the next obligation

V35 supports a bounded practical continuity observation: learned two-block
replacements can retain the measured native causal behavior through this
synthetic ladder, with predictions committed before evaluation. It supports
taking the method beyond synthetic task guides. It does not prove identity,
all interventions, all architectures, or complete mapping-aligned equivalence.

The next distinct local study is a frozen whole-block pretrained Transformer
transfer packet: define task-critical sites, an external-custody corpus,
a scalable derivative adapter, retained causal/rescue gates, controls, budget,
and clean replication before selecting fresh checkpoints or evaluating them.
Prior V3/V8/V9/V28 checkpoint work remains relevant; affine projection success
is not proof of this nonlinear two-block method's pretrained transfer.
The current mapping failure also needs a separate adaptation-only diagnosis,
without changing V35's intervention semantics or availability gates.

Jacobian supervision itself has prior art in
[Jacobian matching](https://proceedings.mlr.press/v80/srinivas18a.html) and
[Sobolev training](https://arxiv.org/abs/1706.04859); causal supervision also
has prior [causal-distillation work](https://aclanthology.org/2022.naacl-main.318/).
The finite ladder/controls do not establish novelty priority. A faithful
Network of Theseus comparison still requires its own agreed benchmark.

Claim ceiling:
`LOCAL_SYNTHETIC_ENDPOINT_SECANT_AND_JACOBIAN_SUPERVISION_ONLY`.
Stronger independent science still requires blind external replication and
authorized independent review. Human-calibrated alignment, consciousness
research, and production operational authorization remain separate gates.
