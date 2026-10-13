# V36R2: prospective catalog compatibility repair

Date 2026-10-06. Revision `202f459d6d89df9a038d78d58e0af825d287fcd8`
plus digest-bound uncommitted source. State slices:
`theseus-lab-sandbox-v1`, `theseus-pretrained-whole-block-transfer-v36r2`,
`theseus-public-checkpoint-acquisition-v36r2`,
`global-experiment-assurance-lifecycle-v1`.

## Exact scope

Finish the newly frozen [V36R1](replacement-loss-v36r1-continuation.md) procedure
with one catalog compatibility correction. R1 remains
`NO_ELIGIBLE_CHECKPOINT_COHORT`; no weights, replacement or held-out outcome
was produced. It returned zero OPT/GPT-2 catalog entries and eight rejected
Llama entries. The [read-only catalog contract audit](replacement-loss-v36r1-catalog-contract-audit.json)
found different unfiltered returns for known baseline searches; this does not
establish global model availability or an internal server cause.

This is a fresh packet and process namespace. R1 source/receipts remain
unchanged. A small client subclass removes only the server-side
`num_parameters='max:250M'` catalog hint. It preserves query strings, library/
pipeline filters, creation sort, 200-entry limit, lexicographic nomination,
public token-free access and all publisher provenance requirements.

**The parameter ceiling remains 250M.** The existing trusted local
meta-device model construction still counts parameters before nomination or
weight acquisition; the same 1.1 GiB file, hidden/depth/vocabulary, architecture,
activation, safe-format, freshness and license checks remain mandatory.
There is no change to loss, training updates, gradient cap, corpus rules,
three-family/three-seed repetitions, scientific gates, control checks,
prediction seal, entropy release, clean reproduction or numerical closeout.
The new protocol ID changes deterministic corpus/seed namespaces.

H1 and claim ceiling remain
`LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY`. A metadata retrieval
repair supplies no scientific retention result. Nomination is frozen before
any candidate weight acquisition; no second cohort follows a downstream fault.

## Custody, costs, and stop conditions

All original R1 custody rules and the exact three-export opaque quarantine
apply. New artifact root:
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-pretrained-whole-block-transfer-v36r2`.
New runtime root:
`/Volumes/Backup ED/ResearchRuntimes/theseus-lab-v36r2/toolchains`.
Original V36 and R1 artifacts are retained and included in historical intake.
Source archives and all historical hashes are checked before the new screen.

Acquire remains explicit opt-in with `--execute-public-acquisition`. No
credentialed requests, inference API, paid worker, cloud execution, reviewer
submission, production, public release, commit or merge is enabled. Monetary
cap $0. Energy, amortization and retry-inclusive network economics remain
unmeasured. R1's 1800-second/8 GiB acquisition cap and V36's 24-hour/6 GiB
RSS/24 GiB artifact caps remain cooperative and unchanged.

No complete metadata cohort means `NO_ELIGIBLE_CHECKPOINT_COHORT` and no
weights. A visible target miss means `NO_QUALIFIED_TARGET` and no learning.
Scientific retention misses are `FAIL`; source/custody/runtime/transport/
resource/numeric faults are `INCONCLUSIVE`. Preserve every terminal root;
do not repair or retry it. Matching actual primary/replica procedure and
closeout can reach only `REPRODUCED_LOCAL`, never independent acceptance.

## Commands

Use the same [R1 command sequence](replacement-loss-v36r1-continuation.md),
replacing the driver with `replacement_loss_v36r2_runner.py` and the runtime
paths with V36R2's paths. Source must freeze before the first new public
catalog or checkpoint nomination. Run metadata/asset acquisition first,
then bootstrap/verify the actual runtime pair, primary, replica and closeout.

Review status `NOT_SUBMITTED`. Novelty priority, mapping equivalence,
universal/metaphysical identity, consciousness, human-equivalent alignment,
independent science and production remain unestablished. Stronger science
still requires the separately authorized configured external reviewer.
