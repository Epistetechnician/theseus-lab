# Replacement-loss v9 cloud cross-runtime reproduction v1

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-cloud-reproduction-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v9-cloud-cross-runtime-reproduction-v1`.

Status: frozen before GMan node provisioning and cloud data transfer. The
companion freeze manifest binds this protocol to the completed v9 packet and
its owner-custodied primary run.

## Question and scope

Does a clean GMan CPU-8 runtime reproduce the exact candidate states,
predictions, outcomes, controls, and `FAIL` verdict from the already-frozen
replacement-loss-v9 run?

This is a cross-runtime replication of the existing v9 result. It reuses the
three selected checkpoints, all nine family-by-seed trajectories, the same
visible/adaptation fixtures, and the same held-out challenge fixtures. It does
not select new checkpoints or change the v9 methods, thresholds, controls,
seeds, or analysis. The v9 packet remains immutable and its original
same-host `REPRODUCED_LOCAL` result remains authoritative for that packet.

The cloud runner is the frozen v9 clean-runtime replicator. It refits its
candidates from visible/adaptation inputs and freezes them before its code
copies held-out challenge files into the replica root. The operator already
knows the primary result, so this run is not blind. GMan is compute in the
same user's account and does not provide independent reviewer authority.

## Frozen target and execution

The target is the v9 packet with canonical digest
`9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`, its
existing source identity, checkpoint manifest, fixtures, candidate freeze,
prediction manifest, and outcome report. The exact digest-bound identities
are listed in the companion freeze manifest.

Use one private GMan `cpu-8` node, Linux CPU execution, and one PyTorch thread.
Use Python 3.12.13 and `requirements-v9.lock.txt`; retain actual platform,
package, and wheel-build identities. Do not use a GPU, provider inference,
assessment-corpus data, or public artifact sharing. The existing synthetic
fixtures and model assets are transferred only into the user's private GMan
workspace for this run.

Run the frozen v9 replicator once. Its command limit is 75 minutes. The total
node session must remain within 90 minutes and $1.00, including setup and
teardown. Before execution, require GMan's locked node rate to be no more than
$0.0083 per minute. Stop immediately on a packet, source, checkpoint, fixture,
dependency, or custody verification failure; a resource limit; timeout; or
cost-bound breach. Preserve the stop receipt and do not retry this packet.

## Outcomes and stopping rule

Preserve the v9 thresholds and conjunctive decision rule unchanged. The
primary v9 outcome is `FAIL`: ridge passed 21/27 complete stages, and both
predeclared comparative gates passed. No threshold or evaluation rule is
retuned for the cloud runtime.

Report cloud reproducibility as `REPRODUCED_LOCAL` only when every frozen v9
replication check passes, including exact candidate-state, prediction,
outcome, source, checkpoint, dependency, and fixture identity. Report
`DIVERGED` when a measured check or scientific outcome differs. Report
`INCONCLUSIVE` for missing inputs, an unsupported runtime, a custody or digest
failure, or resource termination before a complete result. Retain every
completed or failed receipt. A successful reproduction preserves the v9
scientific verdict `FAIL`; it does not change it to `PASS`.

One cloud attempt is the stopping rule. No outcome-dependent rerun, threshold
change, fixture regeneration, or checkpoint substitution is allowed. If the
actual Linux CPU package build cannot satisfy the frozen dependency check,
stop with `INCONCLUSIVE` and record the exact build; do not normalize or
rewrite its identity.

## Cost, custody, and retention

The maximum cloud spend is $1.00. Record the GMan mission, node rate, billed
duration, and actual charge from GMan's mission and usage receipts. If the
rate exceeds the frozen limit, do not run the experiment.

The source bundle, checkpoint and candidate bytes, synthetic fixtures, and
replica outputs stay in owner-controlled external custody. Do not copy raw
prompts, model weights, predictions, or row-level outcomes into this
repository. Keep only protocol, input/output digests, aggregate outcomes,
runtime and cost receipts, and the GMan mission URL here. Do not publish or
share GMan artifacts. Stop the node immediately after receipts are safely
retained; preserve its stopped private disk as the run record.

## Claim ceiling and nonclaims

Claim ceiling:
`LOCAL_CROSS_RUNTIME_REPRODUCTION_OF_THE_FROZEN_V9_FAIL_RESULT_ON_THE_SELECTED_CHECKPOINTS_AND_FIXTURES`.

Even a successful cloud reproduction is same-owner evidence. It is not
`INDEPENDENT_ACCEPT`, blind external replication, novelty priority,
architecture-general validity, whole-model identity, general mechanistic
understanding, subjective experience, consciousness, alignment or human
equivalence, or production authorization. The existing next authority gate
for independent science remains a frozen digest-bound packet and an external
reviewer with independent authority; this cloud run does not submit one.
