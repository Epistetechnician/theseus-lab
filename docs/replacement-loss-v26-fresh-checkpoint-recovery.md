# Replacement-loss V26: fresh-checkpoint execution recovery

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v26-fresh-checkpoint-recovery-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v26-fresh-checkpoint-recovery`.

## Question and claim ceiling

Does the frozen V8/V22 independently fitted affine MLP output-projection
procedure complete and reproduce on newly selected public checkpoints after
V25 stopped before candidate preparation because its runtime omitted
`protobuf`? V26 uses exactly one fresh checkpoint from each predeclared
OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated family, with three fixed seeds per
family (nine cases total).

V25's three selected checkpoints and every identity in its six metadata
queries are treated as previously used. V26 excludes them, along with the
prior experiment ledger, before selection. V26 changes the dependency lock and
the bounded Hub result limit only. The V8 treatment, V22 fixtures, evaluator,
thresholds, controls, case count, and claim ceiling remain fixed. This is a
fresh-checkpoint replication within three named architecture families, not an
architecture-held-out test or evidence about all architectures.

## Freeze, runtime, and target selection

Freeze this protocol, source, adapter, lock, thresholds, and fixture rules
before any V26 Hub query or target selection. Verify and bind the V25 packet,
screen, selection, download manifest, checkpoint manifest, and terminal
receipt. Preserve V25 unchanged. Public metadata access and model downloads
use `token=False`; model files and all experiment data stay under external
owner custody. No hosted inference, remote worker, paid execution, or prompt
transfer is permitted.

Use Python 3.12.13 on local macOS arm64 CPU with the exact 27-package
`requirements-v26.lock.txt` inventory, including `protobuf==5.29.6`. Build
separate primary and clean-replica environments. Verify both inventories and
distinct environment paths before issuing Hub queries. Query the same frozen
family terms as V25, ordered by downloads, with a limit of 5,000 per query.
Require explicit full 40-character revisions. Reject private or gated models,
ineligible configs/licenses, identities present in the frozen prior ledger,
and any checkpoint that exceeds the frozen parameter or download caps. If
any family has no eligible fresh checkpoint, retain the complete screen and
stop before downloading weights.

## Evaluation and reproduction

After a complete three-family screen, download only the three selected
checkpoints into the pinned external V26 custody root and validate every
allowlisted file against the metadata-derived byte count and SHA-256 manifest.
Run the exact nine V22 cases: three fixed seeds per architecture family; 128
adaptation prompts and 60 held-out challenge prompts per case, split across
ordinary, compositional, and adversarial strata. Candidate predictions must
be sealed before held-out outcomes are opened. Preserve all V8 causal-effect,
behavior, activation-rescue, control, and aggregate thresholds without
retuning.

Use local CPU only, with caps of 7,200 seconds per case, 64,800 seconds total,
and 24 GiB peak RSS. Run the primary packet, then a second clean Python
3.12.13 environment with the identical 27-package inventory. Report
`REPRODUCED_LOCAL` only if the frozen V8 exact candidate reconstruction,
prediction/outcome digests, dependency inventory, source identity, and
distinct-environment checks all pass. Same-owner local reproduction is not
independent review.

## Outcomes, custody, and gates

Use the frozen V8 thresholds. Report measured misses as `FAIL`; missing inputs,
digest mismatches, incomplete prediction sealing, dependency errors, or
resource termination as `INCONCLUSIVE`. A complete local screen with no
eligible fresh checkpoint may be `INTERNAL_PASS` for the metadata-screen
scope only. Preserve failed and partial receipts. The terminal report must
include exact packet and artifact digests, revision and state slice,
assurance, reviewer status, reproduction status, data and artifact custody,
limitations, claim ceiling, and next blocked authority gate.

Reviewer status is `NOT_SUBMITTED`. V26 does not establish unseen-architecture
validity, all-architecture validity, whole-model mechanistic identity,
independent science, consciousness, alignment, or production readiness.
External scientific review, human-calibrated alignment evaluation,
consciousness research, and production authorization remain separate gates.
