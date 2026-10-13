# Replacement-loss v9 fresh-node GMan reproduction v3 results

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v3`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Outcome

Execution status: `INCONCLUSIVE`.

The frozen v3 packet did not complete the held-out comparison. The expected
primary scientific outcome remains `FAIL`; this run did not reproduce or
change that outcome. `REPRODUCED_LOCAL` was not reached. Reviewer status is
`NOT_SUBMITTED`, as authorized for this local run.

After candidate fitting and candidate freeze, the replicator revealed the
held-out challenges and entered `seal_predictions`. A candidate load then
raised `ValueError: frozen candidate bytes changed`. The error occurred in
`replacement_loss_v9_runner._load_candidate` while sealing held-out
predictions. The run stopped before completing the frozen comparisons and
writing a v3 replication manifest. A read-only source audit localized the
deterministic cause: `_save_module` serializes a tensor state and returns its
canonical tensor-state digest, which is stored in each stage's
`candidate_sha256`. `_load_candidate` compares that field to `_sha_file(path)`,
the serialized file-byte digest. The candidate-freeze inventory separately
binds the actual file-byte digest, and that inventory check had passed. The
exception therefore does not show that candidate bytes changed; it compares
two different digest domains. The run stopped before held-out predictions or
outcome comparisons completed. No retry was made; v3 authorizes one evaluation
attempt.

The underlying v9 replicator's exception handler updated the retained primary
report's reproducibility field to `INCONCLUSIVE` and recorded its stop digest.
Its scientific `outcome_status` remains `FAIL`. This post-error report is not
the pre-run primary file bound by the v3 freeze.

## Frozen identity and execution

- V3 packet SHA-256:
  `f824c78811f3a1daa60c14eb496cd7364481a02fbc529f02248620228f1d0f93`.
- Target v9 packet SHA-256:
  `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Repository base revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- GMan mission: `theseus-v9-linux-runtime-reproduction-20260928-v3`.
- GMan node: `theseus-v9-linux-runtime-v3-cpu8-20260928`, ID
  `306cb242-38b3-424a-961b-a02128bcca66`; node is stopped and its disk is
  `snapshotted` and restorable.
- Model command: `cmd-nmc86`; it exited with the candidate digest error rather
  than reaching the frozen 60-minute timeout.
- Runtime preflight passed with `runtime_match: true` and no mismatched fields.
  The observed runtime was Python `3.12.13`, Linux x86_64, PyTorch `2.6.0`, one
  CPU thread, and dependency-manifest SHA-256
  `0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4`.

## Custody and spend

Aggregate receipts are retained outside the repository at:

`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v9-gman-cloud-reproduction-v3/`

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `runtime-preflight-v3.json` | 3,443 | `0b980887e9d2299031ab0f92db175d6e71d8d7d5bb29f667621c8f03375781b5` |
| `runtime-preflight-v3-observed.json` | 3,393 | `9cf21451d191b8fbd66e0bf030cb44a52565b0460f6c83154abaccb711ac9f3c` |
| `primary-outcome-manifest.json` (post-error report) | 874,932 | `0753bfbef92562940e3aca07ec5cdbe497d1b1659b82f8d2e06d97c65c645b08` |
| `replication-v3-stop.json` | 499 | `5c823ee3578d845b782fb413739ac00bdfce916c6bc82b9ce2a0110d6145c90e` |

The frozen pre-run primary outcome-manifest digest was
`12e0befa4c329d3a6a18b14d19144ab6296e5ac5b82f0cb2326eaeff7c776431`.
The post-error report has a different file digest because the existing
replicator recorded the `INCONCLUSIVE` stop; its report digest is
`ebc2971c8b6ade97f1ab820d7471518747d4a77229c214d93fb9d4abf520a882`.
All downloaded custody files were checked against GMan's artifact SHA-256 and
byte count.

The candidate preparation, frozen candidates, and generated challenge files
remain on the stopped GMan node; they were not copied into the repository or
the local custody folder. No prediction manifest or replication manifest was
completed. Current GMan status reports the node disk as `snapshotted` and
restorable; no additional named snapshot was created. The transferred input
context was deleted; deletion was reported as complete and byte reclamation
was confirmed at `2026-09-28T03:30:36.786474Z`.

The V3 mission's stored result reports `$0.4232`, but the current September
GMan billing ledger and node usage report `$0.4466` for this node. Use the
ledger amount for spend accounting. Including V1's `$0.2092` and V2's `$0.00`,
the billed combined total is `$0.6558` of the frozen `$1.00` cap, leaving
`$0.3442`. The node was stopped after receipts were exported.

## Claim ceiling and limitations

The v3 ceiling remains
`LOCAL_CROSS_RUNTIME_REPRODUCTION_OF_FROZEN_V9_FAIL_UNDER_THE_DIGEST_BOUND_LINUX_RUNTIME`;
it was not reached. This is not a reproduction, independent review, novelty
finding, architecture-general result, mechanistic-understanding result,
consciousness claim, alignment result, or production authorization.

The frozen v9 result remains `FAIL` (21/27 ridge stages met every retention
gate). A separate, already completed same-owner local continuation of the
same v9 packet used the two-domain digest check and passed its candidate
integrity preflight (81 treatment stages and 162 candidate/control files).
Its bound replication result is `REPRODUCED_LOCAL` with the same `FAIL`
outcome. That evidence is recorded under continuation packet
`a4d3c491f36bca054ec65b7a061e4d36deaabe2e2fb0a64669d745201d1a2be1`; the
final local binding receipt file SHA-256 is
`41f2107e6e0a133679c70a4a7b916886da39304fee34f5275273028503897ca2` in
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v9-continuation-v3-primary/`.
This does not upgrade the separate GMan V3 run or establish independent
acceptance. V1 and V2 remain separate `INCONCLUSIVE` attempts, and GMan V3 is
also `INCONCLUSIVE`.

## Next gate

Keep GMan V3 immutable. The digest-domain correction is already implemented in
the separate local continuation described above. A GMan cross-runtime rerun
would need a new packet and mission that bind the corrected loader and run the
two-domain candidate audit after candidate freeze but before generating held-out
challenge entropy. It must keep the same frozen v9 candidates, fixtures,
thresholds, outcomes, and claim ceiling. V3's one-attempt slot is spent. The
remaining `$0.3442` under the prior cap is below the previous full-node bound of
`$0.6834` (85 minutes at the frozen rate); increasing the aggregate cap to
`$1.34` would cover one bounded full attempt. No V4 model evaluation has been
started or budgeted.

The higher-value scientific step after this reproduction gate is a new,
separately frozen replacement-loss protocol with fresh held-out interventions
and a replacement module trained independently of the algebraic split. It
should predict causal effects before challenge reveal and should not retune
the frozen v9 thresholds or reuse its held-out outcomes for model selection.

The external reviewer was not contacted. External review, alignment,
consciousness, novelty priority, and production authorization remain outside
this result's scope.
