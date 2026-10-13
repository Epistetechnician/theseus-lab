# V4k failure audit result

## Scope and frozen identity

- State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v9-gman-v4k-failure-audit-v1`, `theseus-replacement-loss-v9-gman-v4k-custody-recovery-v1`, and `global-experiment-assurance-lifecycle-v1`.
- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Audit assurance: `INCONCLUSIVE`.
- Frozen V4k packet: `e030b23cf67a04793d464b669fbd285273d08a5e5c968f5b915e5d6781cfed2c`.
- Frozen V4k result: `DIVERGED`, assurance `INCONCLUSIVE`; seven of twelve checks passed.
- Reviewer status: `NOT_SUBMITTED`.

The audit attempted a read-only inspection limited to the V4k receipt digest,
terminal status, and failed check identifiers. It did not inspect or export
prompts, predictions, outcome arrays, model outputs, weights, or held-out data.

## Audit disposition

The V4k receipt was not recovered. Its GMan node began restoring a 22,375,802,880-byte snapshot, then reported no recent host progress while extracting. A one-minute digest-only command registered for wake as `cmd-xparh` became `lost` before returning a diagnostic. The node was stopped afterward and GMan reported the audit mission at `$0` with accrual stopped. No V4k files or outcomes were changed.

The failed check identifiers remain unknown. The aggregate count does not establish whether the divergence came from source/runtime identity, candidate state, challenge fixtures, prediction arrays, outcome checks, or controls. No cause or fix is inferred from the seven-of-twelve count.

## Follow-up custody paths

The retained server-side log for original command `cmd-7bbdu` was read without
waking the node. Its result binds the expected V4k packet and `DIVERGED` status,
but the log does not contain a validated twelve-key boolean check map. The
isolated check-name fragments in the log do not identify which checks failed.
The local wrapper source confirms that its final stdout summary carries counts,
while the authoritative check map is stored in `v4-run-receipt.json`.

A `snapshot_node` request was refused because the node disk is already parked
as a snapshot. GMan's supported route to make a new durable snapshot requires
waking the node, stopping it again, and retrying the snapshot. The earlier wake
stalled during extraction with no recent host progress, so that restore was not
repeated. The node remains stopped; no new evaluator or node was started. The
durable snapshot listing contains five unrelated snapshots and shows 149,034,137,600
stored bytes against the 250 GiB free allowance, but no snapshot ID for this node.
The log and metadata checks added no GMan mission cost.

## Evidence and custody

- V4k runner receipt SHA-256: `d070dc3bfe599ec6588cba3bea86259151951ecfad82719aa40638736cb759eb`.
- V4k receipt SHA-256: `4381beb2fd8df40cff847ab86fc2e6967b2bb0159040e30a51d2c666227e3bfa`.
- Existing digest-only exported summary SHA-256: `543cbdb4465d08d0844ce7f3f2d8b43cbe3b09fcd40a2e8536dd29f1b54f7a1a`.
- The existing V4k terminal record accounts for `$3.271` against its `$5.00` aggregate cap. This audit added `$0` according to its GMan mission record.
- Prompts, outputs, weights, predictions, held-out data, and detailed check values remain in GMan custody.

Relevant local source identities at the recorded revision/worktree:

- `replacement_loss_v9_replicator.py`: `005ff243087001441c44eee22408c55a3202645eb2d9648e968417ecb5686923`.
- `replacement_loss_v9_gman_fresh_node_replicator_v4.py`: `2ae1cd0f8b813c28d8815c1b1bfab01a46fcd65afa7d0da24aad81a78b22fc97`.
- `replacement_loss_v9_gman_fresh_node_replicator_v4k.py`: `f403b54ea548b98ce00c751afb34b6c1f1269bdcae44f2b48c2f2b5663e70e04`.
- `docs/replacement-loss-v9-gman-fresh-node-v4k-freeze.json`: `4cba55f782ec45c845aee3ca11348c0e6ce12e9ac8de4e9fe7377d5418dcb4f0`.

## Fix path and claim ceiling

There is not enough diagnostic evidence to change the evaluator or the model.
Keep V4k terminal and do not tune its frozen checks or reuse its held-out
outcomes. Recover only the digest-bound failed check identifiers through a
working owner-side custody path. If the failed checks are runtime or integrity
checks, repair that harness defect and freeze a new packet. If they are
candidate, prediction, outcome, or control checks, treat them as divergence and
test any revised method only on fresh held-out challenges under a new packet.

Claim ceiling: one frozen local reproduction attempt diverged. This audit does
not establish `REPRODUCED_LOCAL`, independent acceptance, novelty, broad
mechanistic understanding, architecture-general validity, Theseus identity,
consciousness, alignment, human equivalence, or production authorization.

Next blocked gate: digest-bound access to the failed check identifiers. Any
new evaluation also requires a new frozen protocol/packet, fresh challenge
data, and budget/time that fit the predeclared design.

## Custody-recovery follow-up

Provider state checked through 2026-09-29 19:06:19 UTC; the support ticket was
updated and re-read at 19:07:53 UTC under the additional state slice
`theseus-replacement-loss-v9-gman-v4k-custody-recovery-v1`:

- Recovery disposition remains `INCONCLUSIVE`; the failed check IDs were not
  recovered. The frozen V4k disposition remains `DIVERGED` and its packet is
  unchanged.
- The digest-only on-wake command `cmd-hit2n` is recorded as `lost`, ended at
  19:03:43 UTC, and has no exit status or captured output. At the same time,
  `get_node` reports `stopped (disk intact)` with `boot_command.state` still
  `pending_wake`. The node record's `durable_as_of` is 19:03:33 UTC. GMan's
  audit stream records a system `node.stop` for `worker_died` at 18:02:02 UTC
  and no later `node.wake` event for `cmd-hit2n`; whether that command actually
  started is unresolved.
- A second `snapshot_node` request returned the same refusal: the disk is
  already snapshot-parked and must be woken, stopped, then snapshotted. The
  current five-item snapshot inventory has no snapshot for
  `theseus-v9-v4i-cpu8-20260929`. No fresh snapshot or export was created.
- Support ticket `tkt-ruq88` was still `open` at 19:07:53 UTC after our
  follow-up recorded the lost command, stopped-node state, worker-death event,
  wake estimate, snapshot refusal, and missing snapshot. No operator response
  has arrived.
- Recovery mission
  [`theseus-v9-v4k-custody-recovery-20260929-v1`](https://autoresearch.sfcompute.com/missions/theseus-v9-v4k-custody-recovery-20260929-v1)
  has `$0` cost and is not accruing. The source V4i reproduction mission has
  `$1.5048` total cost. No evaluator was started and no V4k payload was
  exported during recovery.

## Recovery route screen

The current GMan tool contracts leave these paths:

1. An on-wake command can read the exact receipt allowlist, but both registered
   diagnostics (`cmd-xparh`, `cmd-hit2n`) ended `lost`; the second has no output
   and conflicts with the node's still-pending boot-command field. It is not
   being retried without operator guidance.
2. `snapshot_node` requires a stopped node, but GMan twice refused because the
   disk is already snapshot-parked. A matching customer-listed snapshot ID is
   absent, so there is no V4i snapshot to restore independently.
3. `export_file` and `export_data` require a running node or an explicit wake
   before exporting. They depend on the same stalled restore path and transfer
   source bytes outside the node, while this recovery is limited to allowlisted
   receipt metadata; neither was used.
4. `create_node(from_snapshot=...)` requires a snapshot ID. The listed snapshots
   are for other nodes and were not substituted for V4i's exact disk state.
5. The supported remaining route is the open GMan operator ticket
   `tkt-ruq88`, which asks for a provider-side restore/read path while keeping
   all experiment payload in GMan custody.

Assurance level: `INCONCLUSIVE`. Reviewer status: `NOT_SUBMITTED`. V4k
reproducibility status: `DIVERGED`; receipt-recovery status: `INCONCLUSIVE`.
Prompts, predictions, outcomes, model outputs, weights, held-out data, and the
authoritative check map remain in GMan custody and were not exported. The claim
ceiling remains one frozen same-owner local reproduction attempt that diverged;
the failed-check causes are unknown. The next blocked gate is an operator-
supported recovery path or confirmation that the parked volume is no longer
recoverable. No further wake, snapshot, export, or evaluation retry should be
made until that guidance arrives.
