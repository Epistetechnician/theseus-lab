# Retained-custody checkpoint exposure audit V1

Date: 2026-10-06. Revision `202f459d6d89df9a038d78d58e0af825d287fcd8`
plus frozen uncommitted bytes. State slices: `theseus-lab-sandbox-v1`,
`theseus-checkpoint-exposure-feasibility-v1`,
`global-experiment-assurance-lifecycle-v1`.

## End state and boundary

Bind the V36R2 historical input set, its acquisition records, the original
checkpoint registry, and all retained Theseus weight paths. Classify identities
conservatively as execution recorded, weights recorded, metadata only within
retained custody, or unknown. This is a deterministic provenance audit, not a
model experiment. No network, loading, replacement training, scientific
evaluation, paid execution, remote worker, external review task or publication
is opened. V36, R1 and R2 source and terminal results remain immutable.

Source is frozen before classification. Historical file bytes and JSON pointer
contexts bind receipt evidence; source archive copies are not new events.
Download popularity, public weight-file metadata and model-card screening do
not imply that model weights were fetched or executed. Actual download
receipts, retained Hub cache weight files, parameter/tokenizer preflight,
candidate preparation, and outcome digests are positive exclusion evidence.
Unrecognized context and inherited prior-use assertions remain unknown.

All identities in the prior exclusion list begin unclassified. Metadata-only
requires direct metadata context and absence of weight, execution or unknown
context. No positive evidence does not establish never-used status outside
this retained corpus. Opaque legacy exports remain quarantined and unknown;
removed, remote, alias and other-project history is not fully covered.
Retained candidates are hashed but are not attributed to a guide merely by
being weight files. All retained weight hashes remain exclusions for a future
cohort, including unattributed files. Full evidence stays in external custody;
repository reports contain scalar counts and digests only.

## Qualification and stop rules

Hermetic checks cover metadata/download/execution distinctions, unknowns,
source copies, catalog identities and monotone exposure. Run:
`.venv/bin/python -m unittest discover -s tests -p test_checkpoint_exposure_v1.py`.
Execution cap: 900 seconds, 64 MiB per historical JSON, 36 GiB unique physical
weight bytes hashed, paid cap $0. Missing or changed input, broken/escaping
links, custody loss or cap exhaustion fails closed. No repair of prior inputs.
No claim of independent review or whole-checkpoint novelty follows.

Public utility: `theseus_checkpoint_exposure_v1.py audit`, `verify`, and
`verify-result`. `audit` creates one fresh root; `verify-result` rechecks the
bound retained inputs without adding exposure events. Source freeze and
result digests identify the exact classification rules and coverage.

Assurance ceiling: `INTERNAL_PASS` for this engineering/provenance scope.
Reviewer `NOT_SUBMITTED`; separate reproduction not asserted. Next scientific
obligation is a prospectively frozen cohort definition with an honest
freshness ceiling, followed by actual primary/replica and numeric closeout.
