# Replacement-loss v9 replication-result binding v4

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Continuation ID: `theseus-replacement-loss-v9-replication-binding-v4`.

## Why this binding exists

The clean-runtime v9 replicator completed and wrote matching primary and
replica manifests. All frozen checks passed with result status `FAIL`, giving
`REPRODUCED_LOCAL`. The v3 wrapper then stopped while copying review context
because the destination folder was missing. This stop occurred after the
replicator completed and did not alter the reports or outcome arrays.

V4 records that postprocessing stop, verifies the existing report and
replication digests, freezes their identities, then copies the already frozen
context and writes a separate binding receipt. It does not rerun any model,
refit candidates, alter predictions or outcomes, or change the `FAIL` verdict.

## Assurance and limits

The result is `REPRODUCED_LOCAL` by the exact v9 replication checks in a fresh
Python environment owned by the same operator. Reviewer status is
`NOT_SUBMITTED`; the replication is not an independent scientific acceptance.
The remaining authority gate is blind external replication and independent
signed review. Alignment, consciousness, novelty priority, and production
authorization remain outside this result.
