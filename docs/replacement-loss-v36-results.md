# V36: historical intake terminated before model selection

Date: 2026-10-06. Revision `202f459d6d89df9a038d78d58e0af825d287fcd8`
plus [frozen source](replacement-loss-v36-freeze.json). State slices:
`theseus-lab-sandbox-v1`, `theseus-pretrained-whole-block-transfer-v36`,
`global-experiment-assurance-lifecycle-v1`.

**Scientific outcome: `INCONCLUSIVE`. Reproduction: `NOT_RUN`. Reviewer:
`NOT_SUBMITTED`. Previous engineering qualification remains `INTERNAL_PASS`.**

Both approved drives were mounted and every V35 manifested artifact, result,
reconciliation and closeout digest verified. V36 source packet
`370a074309eeea6b57cf5b529892932d5d93474a9381b7ffd47bc1a6bb367458`
also verified. The one intake attempt then stopped on `JSONDecodeError: Extra data`
before creating an intake, selecting a checkpoint, fitting a replacement or
querying a held-out outcome.

The read-only follow-up audit found 1,609 valid JSON files, three invalid
legacy Cloud Run exports and no oversized JSON files. The final export is
550 bytes of plain log text; the progress and task-start exports are empty.
Their exact paths, sizes and digests are retained in the
[scalar receipt](replacement-loss-v36-results.json). Original bytes were not
changed. The initial attempt was not retried and frozen source was not edited.

A terminal receipt is retained under the V36 external artifact root. Raw
historical logs remain external. No candidate, prompt corpus, prediction,
oracle outcome or scientific replica was produced. Paid execution cost $0;
local energy and machine amortization remain unmeasured.

Claim ceiling: `LOCAL_PRETRAINED_TWO_MLP_INTERFACE_REPLACEMENT_ONLY`.
This intake failure provides no finding about whether the replacement method
works. Independent science, novelty, identity, consciousness, alignment and
production remain unestablished.

The fresh [V36R1 continuation](replacement-loss-v36r1-continuation.md) adds a
digest-bound quarantine for those exact legacy exports and an explicit public
asset adapter. It preserves V36 as terminal and keeps the loss, intervention
gates, controls and cost ceiling unchanged. Eligibility remains unknown until
the newly frozen prospective model screen runs.
