# V39 first-MLP fit-factorial V3 results

2026-10-07. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-fit-factorial-v3`, and
`global-experiment-assurance-lifecycle-v1`.

## Outcome

The frozen procedure reproduced in a second isolated runtime on the same
owner's physical host: `REPRODUCED_LOCAL`. The scientific outcome is `FAIL`.
The result is limited to one reused Spanish GPT-2 guide and its first MLP site.
Reviewer status is `NOT_SUBMITTED`.

The paired 2×2 factorial compared 2× and 4× student width with 1,024 and 4,096
optimizer updates for three candidate seeds. The optimizer-budget effect on
adaptation training-direction JVP error met its frozen rule: geometric-mean
error ratio 0.7764 (22.36% lower error), with all three seed contrasts
favoring 4,096 updates. The capacity effect was 0.8801 (12.00% lower error),
below the required ratio of 0.80 despite favorable direction in all three
seeds. These are paired, adaptation-side effects on one guide, not population
estimates.

None of the four factorial cells passed the held-out retention rule across
all three seeds. Across 48 held-out seed-by-stratum cells, output nRMSE and
both rescue measures passed 48/48, while top-1 agreement passed 17/48, NLL
increase 12/48, activation nRMSE 37/48, joint causal-effect prediction 0/48,
and the required control separation 5/48. The rescue and output results do
not compensate for the causal, behavior, and control misses.

The aggregate stratum breakdown shows different failure profiles. Across the
12 candidate-by-seed rows per stratum, interchange-composition has median
causal-effect relative error 0.989 and cosine 0.407, with no row meeting the
full control margin. Adversarial prompts have median top-1 agreement 0.672
and median NLL increase 0.153. Ordinary prompts are behaviorally strongest,
but their median causal-effect relative error remains 0.849 and cosine 0.582.
The best factorial cell (4× width, 4,096 updates) has median effect error
0.832 and cosine 0.649, still far from the frozen causal gates.

This supports one narrow finding: longer optimization improved fit along the
adaptation training directions under the preregistered metric. It does not
show that optimization budget caused the held-out causal failures, and added
capacity did not meet its own support rule. Do not treat either finding as a
replacement-retention result.

## Protocol and closeout

The source freeze preceded fixture generation and fitting. Twelve candidates
used the unchanged V39 first-block JVP objective and V36 fitter. Three seeds
each used 128 adaptation prompts and 128 donor prompts; the sealed challenge
contained 32 held-out pairs per seed in each ordinary, input-composition,
interchange-composition, and adversarial stratum. The 384 new held-out pairs
were disjoint from inherited exclusions and prior V39 fixtures. Predictions
were digest-sealed before guide-oracle evaluation. Prior V2 prompt and token
identities were exclusion-only.

The second runtime rebuilt candidate bytes before accessing the held-out
challenge and matched prediction seals before oracle evaluation. Closeout
recomputed all 15 primary/replica reconciliation fields, 48 probe-formula
cells, four factorial effect formulas, 12 held-out seed-strata per role, and
72 held-out/control arrays per role. It byte-matched 182 deterministic
artifacts per role. Runtime dependency fingerprints matched across the two
isolated environments. Timing and peak-memory values are recorded per role;
the complete result JSONs therefore need not be byte-identical.

## Evidence and custody

The frozen packet SHA-256 is
`bc233412da532b1eee72611febe7b510086dd0b8c5de90ec3e93769865b37f3d`.
Primary and replica result digests are recorded in the
[structured result](v39-first-mlp-fit-factorial-v3-results.json), alongside
the candidate, challenge, prediction-freeze, and closeout digests. Experimental
prompts, token IDs, guide assets, captures, candidate weights, predictions,
and oracle arrays remain under `/Volumes/PrimaryED`; the isolated runtimes
remain under `/Volumes/Backup ED`. They were not copied into this repository.

Paid execution was $0. Energy and full economic cost were not measured. The
primary run used 3,594 seconds, 5.24 GB peak RSS, and 3.69 GB of artifacts;
the replica used 3,576 seconds, 5.55 GB peak RSS, and the same artifact-byte
count. Both used Python 3.12.13, macOS 26.6.1 arm64, CPU, one thread, Torch
2.6.0, NumPy 2.4.6, and Transformers 5.17.0.

## Claim boundary and next step

`REPRODUCED_LOCAL` is same-owner reproduction, not independent acceptance.
This is not fresh-checkpoint or architecture-held-out evidence, a blind
external replication, a proof of causal mechanism identity, or evidence for
consciousness, human-equivalent alignment, or production readiness. The claim
ceiling is
`LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V3_ONLY`.

Do not continue increasing width or optimizer budget as a presumed remedy.
The next local study should test a distinct causal-transfer hypothesis with
new disjoint fixtures, frozen predictions, the existing held-out gates, and
an explicit stop rule. Stronger scientific acceptance remains gated on the
configured independent authority; production authorization remains separate.
