# Checkpoint exposure audit V1 results

2026-10-06; revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus
digest-bound source. Slices: `theseus-lab-sandbox-v1`,
`theseus-checkpoint-exposure-feasibility-v1`,
`global-experiment-assurance-lifecycle-v1`.

**`INTERNAL_PASS` for retained-custody classification. Reviewer
`NOT_SUBMITTED`; separate reproduction `NOT_RUN_SEPARATELY`.**

The source-frozen audit bound 1,634 inputs and 618,137 evidence pointers.
Of 48,262 identities, 48,233 have only metadata evidence in this retained
set, 17 have weight evidence, and 12 have execution evidence. It hashed
1,469 weight paths (22,139,740,420 unique physical bytes), retaining 510
distinct file digests as conservative exclusions. Eight hermetic
classification checks passed. The execution took 244.75 seconds, paid $0;
local energy and amortization are unmeasured.

Three opaque predecessor exports remain quarantined and unresolved. Unknown
contexts were not converted to metadata-only. No identity remained classified
unknown after stronger retained evidence was considered; this does not resolve
the opaque exports, missing records, remote history, aliases or other projects.
No model was downloaded, loaded, trained or evaluated by this audit.

Metadata-only has a narrow meaning: no weight, execution or unknown context
was found for that identity in the bound retained input set. It does not
establish never-used status globally. Receipt evidence is conservative
provenance, not independent authentication of old scientific outcomes.

Packet `86a8d7ca23764f99a55a3db48d83c016594271ac0973f14b46a733772d12e633`;
full result `67865bd5a0f4d4c4a36f79c1499a57261c58fa5d5634c5516f806c6291584554`.
[Protocol](checkpoint-exposure-v1.md) and [scalar bindings](checkpoint-exposure-v1-results.json).
Full input and pointer evidence stays under the external
`theseus-lab-sandbox-v1-checkpoint-exposure-feasibility-v1` custody root.

This supports a new metadata-known, weight-held-out development packet.
It supplies no new causal retention finding or wholly unseen confirmation.
Claim ceiling `RETAINED_CUSTODY_EXPOSURE_CLASSIFICATION_ONLY`. Independent
science, alignment, consciousness and production gates remain closed.
