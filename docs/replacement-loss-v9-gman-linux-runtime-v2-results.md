# Replacement-loss v9 GMan Linux runtime reproduction v2 results

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-linux-runtime-reproduction-v2`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Outcome

Execution status: `INCONCLUSIVE`.

The frozen v2 packet was not evaluated. GMan reported a system
`worker_died` stop for node `theseus-v9-cloud-20260928` at
`2026-09-28T01:24:05.603634Z`. Its registered no-op boot command was returned
as `lost` with no captured output or exit status. No v2 protocol files were
transferred, no v2 runtime preflight ran, no replica directory was created,
and no model evaluation started.

V2 packet digest:
`6e2d83abcc1b54c2cc21f4c229924bc461d1351d809d807a55881c11e58ddf4f`.

GMan mission: `theseus-v9-linux-runtime-reproduction-20260928-v2`.
The v2 mission reported `$0.00` charged. The existing node belongs to the
completed v1 mission; it is not reused for another mission. V1 remains
`INCONCLUSIVE` with its recorded charge of `$0.2092` and is not reopened.

## Evidence and custody

- The frozen v2 packet and wrapper passed their local digest and syntax
  checks before the cloud attempt.
- A postmortem source audit found a second V2 blocker: the frozen wrapper reads
  `freeze["primary"]["outcome_manifest_sha256"]`, while the frozen V2 manifest
  stores that digest at `target.primary_outcome_manifest_sha256`. Thus V2 would
  have stopped before `run_replica` even if the node restore had succeeded.
  V2 remains immutable; a new continuation must bind a corrected wrapper.
- GMan audit event `event_c96e559b024442599ae096035aa6e353` records the system
  worker-death stop.
- GMan command `cmd-nwp3v` has status `lost`; no command output was captured.
- The v1 input archive and v1 receipts remain in external local custody under
  `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v9-gman-cloud-reproduction-v1/`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: not reached; no second runtime executed the
  replicator.

## Claim ceiling and nonclaims

No cross-runtime reproduction claim was reached. The original v9 result
remains `FAIL`; v1 and v2 are separate `INCONCLUSIVE` execution attempts.
This outcome gives no evidence for novelty priority, architecture-general
validity, mechanistic understanding, consciousness, alignment, independent
science, or production readiness.

## Next gate

Any continuation must use a new frozen packet and a node created under that
packet's own GMan mission. It must validate the exact runtime and retained
input digests before starting the replicator. Do not mutate this packet or
reopen v1.
