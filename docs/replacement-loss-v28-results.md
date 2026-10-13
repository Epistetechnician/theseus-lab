# Replacement-loss V28 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v28-fresh-checkpoint-recovery-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Verdict

**Outcome: `FAIL`. Assurance: `REPRODUCED_LOCAL`. Reviewer: `NOT_SUBMITTED`.**

V28 completed the frozen nine-case evaluation on fresh OPT/ReLU, GPT-2/GELU,
and Llama/SiLU-gated checkpoints. All nine case-level treatment, held-out
behavior, causal-effect, and activation-rescue checks passed. The global
packet outcome is still `FAIL`: its separately frozen mapping-aligned
permutation equivalence gate failed in the aggregate.

The global gate requires both module-effect and logit-effect maximum absolute
differences to be at most `1e-4` in every case. Across the nine cases, the
largest module-effect difference was `4.119873e-4`; the largest logit-effect
difference was `2.212524e-4`. The gate was frozen before checkpoint selection
and was not retuned. The aggregate's individual case statuses do not display
this global-control miss as a case blocker, so the outcomes document it here
explicitly.

The control permutes the same projection coordinates in the input and weight
columns, then compares its effect arrays with the treatment arrays. This is
the same mathematical linear map, evaluated with a different feature order.
The observed differences are consistent with float32 accumulation-order
sensitivity, but V28 did not independently adjudicate that explanation. The
frozen numerical gate therefore remains failed; this result does not establish
that the discrepancy is harmless.

## Reproduction and resources

The exact same-owner run in the second isolated Python 3.12.13 environment
matched candidate reconstruction, prediction and outcome array digests,
dependencies, source identity, and runtime separation. Both environments had
the same package inventory digest and distinct `sys.prefix` and site-packages
paths. The reproduction confirms the `FAIL`; it is not independent scientific
acceptance.

The primary evaluator reported 658.27 seconds across protocol stages, peak
RSS of 4.36 GB, and no resource blockers. It downloaded 1,487,286,551 bytes
for three checkpoints. Monetary cost was recorded by the evaluator as
`NOT_MEASURED`; energy was not measured. There was no paid compute or remote
worker.

The fixture-disjointness check passed before freeze and target selection:
416 V28 visible/adaptation records were checked against 107 prior custodied
fixture sets. The receipt stores counts and digests only. Model weights,
prompts, captures, candidates, predictions, outcomes, and detailed checkpoint
receipts remain under external custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v28`
and its `-replicator` sibling.

## Execution and verification notes

The V28 evaluator core completed and wrote its digest-bound replication
report. Its wrapper then exited on a post-run alias collision while trying to
replace the pre-replication summary. The earlier summary was preserved as
`replacement-loss-v28-report-before-reconciliation.json`; the final aliases
were copied from the completed core reports and bound in the external
`replacement-loss-v28-reconciliation.json`. No evaluation was repeated and
the frozen packet source was not changed.

The normal post-run dynamic verifier also treats the populated V28 replica
root as a new prior experiment root and therefore reports a changed
preselection ledger. A post-run audit rechecked the frozen packet and source
digests while excluding both current V28 custody roots from the prior ledger;
the original count of 11,930 identities and the frozen ledger digest matched.
The audit is external at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v28/replacement-loss-v28-postrun-audit.json`.
This verifier-scoping defect does not change the core replication checks, but
the frozen V28 `verify` command is not reusable unchanged after the replica
root is populated.

## Frozen identities

- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`
- Packet SHA-256: `f341a7f8e673dd2248f38b73659da25dcf62088b29176e01e1b914f1e5925b08`
- Fixture preflight receipt: `d3a62dfb0a164690f38c19abab785f8b6fb71d815e0086ef9ea256eccdd56aa9`
- Hub screen: `c4637c02e26c6dbe69c6102f4970799c0cb03953d8f5ba6ba348d9ef9539231d`
- Selection binding: `6708730f8abbfd1102625e0ec9c5f755b8a038b6d8007ae8901430edb08a9207`
- Dependency inventory: `c28e0c06fde626df2486715e034065fb37df98d9192ca675581d1e6ffac01a17`
- Final primary report: `1fd263e347d4427b25cbe3c7824dfe254f35bf2967d66cf0a0699a0a3e5cdde9`
- Replication report: `d88f2d1b66c11c5bc6279d52af83dd07cf89cc7342452f75ee47c1e95b7a4618`
- Post-run audit: `0ff29b5c4134ae0caaeec097724aafbb02c7f4cebde371f7e4e5d4e142fbe654`

## Claim boundary and next step

This is a reproduced local result for the frozen V8/V22 output-projection
procedure on three fresh checkpoints in three named model families. It is not
evidence for all architectures, a whole-model mechanism, independent science,
consciousness, alignment, or production readiness. The method does not define
a separate behavior-versus-mechanism divergence statistic.

Do not rerun the permutation sweep or relax V28's threshold. The next
scientific upgrade should use a distinct proof obligation: an independently
constructed replacement module, frozen before held-out evaluation, with
precommitted predictions for unseen causal interventions and a separate
replicator. Any future protocol should also make its global-control failures
visible in the case and aggregate receipts and exclude both current custody
roots when recomputing its preselection identity ledger.

The next blocked authority gate for `INDEPENDENT_ACCEPT` is an external
reviewer with independent authority. The V28 result has not been submitted
for that review.
