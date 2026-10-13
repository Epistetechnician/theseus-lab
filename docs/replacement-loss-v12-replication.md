# Replacement-loss V12: content-addressed V11 replication audit

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v12-replication-normalization-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v12-replication-normalization-audit`.

## Purpose and scope

V11's primary and separate-runtime replica produced identical candidate,
fixture, challenge, prediction, and scientific-result files. Its formal
comparator nevertheless returned `INCONCLUSIVE`: the V11 candidate-set digest
includes the absolute Python executable path, and the separate-runtime check
requires those paths to differ. The downstream V11 challenge and prediction
manifests inherit that path-bound digest.

V12 defines a content-addressed reconciliation for the already completed V11
run. It is a **post-run audit**: the source runs and their artifact contents
were observed before this packet was frozen. V12 does not rerun model
inference, generate prompts, change outcomes, or modify V11. It does not claim
blind or independent replication. The V11 scientific outcome remains `FAIL`;
V12 can only determine whether its outputs reproduce under a stable content
identity while preserving runtime provenance separately.

The V11 target is fixed before freeze:
`grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
No new checkpoint is selected. The V11 packet digest is
`2bd4e9903b2611ac454a0ff26ffabb051231bd2469f6895aee67504d8c99c582`.

## Identity and comparison rules

Candidate content identity binds the source packet, logical checkpoint ID and
revision, model-tree/config/weight digests, visible and adaptation fixture
digests, selected layers, captured adaptation artifacts, fitted candidate
rows and state digests, source revision, and stable runtime versions and
dependency pins. It excludes absolute executable paths, custody paths, host
identity, and wall-clock fields. Runtime provenance records those excluded
environment fields separately.

Challenge content identity binds the candidate content identity, post-freeze
entropy digests, challenge fixture digests, prompt-file digests, and
intervention-file digests. Prediction content identity binds the candidate
and challenge content identities plus sealed prediction rows and file
digests. The scientific-result digest uses V11's predeclared outcome fields.
V11's own candidate, challenge, prediction, and outcome digests are also
checked for internal validity; their original values are retained unchanged.

V12 compares every file by relative path, byte count, and SHA-256 across six
predeclared groups:

- visible and adaptation fixture files: 4;
- adaptation captures: 6;
- candidate states: 12;
- challenge entropy files: 3;
- challenge prompt and intervention files: 9;
- prediction arrays: 30.

The decision is `REPRODUCED_LOCAL` only when all group bytes and normalized
content identities match, both scientific and prior-failure summaries match,
both V11 run ledgers completed within budget, the target and frozen V11 packet
verify, runtime versions and dependency pins match, the executable paths are
distinct, and both runs have the same owner and host. Otherwise V12 reports
`INCONCLUSIVE`. This remains same-owner evidence, not independent acceptance.

## Frozen procedure

1. Freeze this document, the V12 auditor, focused unit tests, dependency
   statement, and the exact V11 packet/results context before running the V12
   audit.
2. Archive the frozen V12 packet to two new external audit custody roots.
3. Run the same read-only V12 auditor once in the V11 primary Python 3.12.13
   environment and once in the separate V11 replica Python 3.12.13
   environment. Each run verifies the V11 packet archive, target selection,
   source checkpoint tree, completed phase ledgers, internal manifests, and
   artifact files. No model weights are copied.
4. Write one deterministic reconciliation report into each V12 audit
   custody. Compare the report bytes and SHA-256. Preserve the V11 comparator
   status as `INCONCLUSIVE` even if V12's content-addressed audit passes.
5. Record the V12 result separately. Do not reinterpret it as a successful
   V11 scientific outcome or as external review.

The protocol was frozen after the V11 run and after its output bytes had been
inspected. This chronology is recorded explicitly; the V12 audit is not a
blind replication.

## Custody, budget, and claim ceiling

The auditor is read-only with respect to V11 custodies and writes only V12
packet archives and reconciliation reports under explicitly supplied
external paths. Hashes, digests, aggregate metrics, and relative filenames may
appear in those reports; fixture contents, prompts, model weights, and
row-level outputs stay outside the repository. Execution is local, streaming,
CPU-only, capped at 30 minutes and 2 GiB RSS. No paid or remote compute,
provider call, cloud/GMan node, or public write is in scope.

Claim ceiling:
`LOCAL_REPRODUCIBILITY_OF_V11_FAIL_OUTCOME_AND_ARTIFACTS_UNDER_CONTENT_IDENTITY`.
V12 cannot establish a positive causal-fit result, fresh-checkpoint
generalization, architecture-general validity, novelty priority, general
mechanistic understanding, consciousness, alignment, human equivalence,
independent science, or production readiness. Reviewer status remains
`NOT_SUBMITTED`.
