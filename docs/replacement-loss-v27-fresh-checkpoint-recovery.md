# Replacement-loss V27: fresh-checkpoint execution recovery

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v27-fresh-checkpoint-recovery-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v27-fresh-checkpoint-recovery`.

## Question and claim ceiling

Does the frozen V8/V22 independently fitted affine MLP output-projection
procedure complete and reproduce on new public checkpoints after V25 stopped
before candidate preparation for lack of protobuf and V26 stopped before
checkpoint selection because its copied-interpreter preflight was
incompatible with the managed Python runtime? V27 selects one fresh checkpoint
from each frozen OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated family and runs
three fixed seeds per family, nine cases total.

V27 excludes every identity from the prior experiment ledger, including all
11,907 model identities screened by V25. V26 issued no Hub query and selected
no target. The V8 treatment, V22 fixtures, evaluator, thresholds, controls,
case count, and claim ceiling remain fixed. This is checkpoint replication
inside three named families, not an architecture-held-out test or a claim
about all architectures.

## Freeze, runtime, and target selection

Freeze this protocol, source, adapter, exact dependency lock, thresholds, and
fixture rules before any V27 Hub query or target selection. Bind the immutable
V25 and V26 packets, prior stop receipts, and upstream V8/V22 method bytes.
Public metadata access and downloads use `token=False`; model files and all
experiment data stay under external owner custody. No hosted inference,
remote worker, paid execution, or prompt transfer is permitted.

Use Python 3.12.13 on local macOS arm64 CPU with the exact 27-package
`requirements-v27.lock.txt` inventory, including `protobuf==5.29.6`. Create
two ordinary isolated virtual environments with system site-packages disabled.
Qualify separation by checking that `sys.prefix` and the installed
site-packages path are distinct for primary and replica, both differ from
`sys.base_prefix`, and both package inventories match the lock exactly. The
two environments may share the same base Python executable; V26 established
that this managed Python distribution is not relocatable by copying its
launcher.

Only after both runtime receipts pass, query the same frozen family terms as
V25, ordered by downloads, with a limit of 5,000 per query. Require full
40-character immutable revisions. Reject gated/private models, ineligible
configs or licenses, identities present in the frozen prior ledger, and any
checkpoint exceeding the frozen parameter or download caps. If any family
has no eligible target, retain the complete screen and stop before downloading
weights.

## Evaluation and reproduction

After a complete three-family screen, download only the three selected
checkpoints into the V27 external custody root and validate every allowlisted
file against the metadata-derived byte count and SHA-256 manifest. Run the
exact nine V22 cases: three fixed seeds per family; 128 adaptation prompts
and 60 held-out challenge prompts per case, split across ordinary,
compositional, and adversarial strata. Seal candidate predictions before
opening held-out outcomes. Preserve all V8 causal-effect, behavior,
activation-rescue, control, and aggregate thresholds without retuning.

Use local CPU only, with caps of 7,200 seconds per case, 64,800 seconds total,
and 24 GiB peak RSS. Run the primary packet and then a second clean Python
3.12.13 environment with the identical 27-package inventory. Report
`REPRODUCED_LOCAL` only if the frozen V8 exact candidate reconstruction,
prediction/outcome digests, dependency inventory, source identity, and
distinct-environment checks all pass. This remains same-owner reproduction.

## Outcomes, custody, and gates

Use the frozen V8 thresholds. Report measured misses as `FAIL`; missing inputs,
digest mismatches, incomplete prediction sealing, dependency errors, or
resource termination as `INCONCLUSIVE`. A complete local screen with no
eligible fresh checkpoint may be `INTERNAL_PASS` for the metadata-screen
scope only. Preserve failed and partial receipts. The terminal report must
include exact packet and artifact digests, revision and state slice,
assurance, reviewer status, reproduction status, data and artifact custody,
limitations, claim ceiling, and next blocked authority gate.

Reviewer status is `NOT_SUBMITTED`. V27 does not establish unseen-architecture
validity, all-architecture validity, whole-model mechanistic identity,
independent science, consciousness, alignment, or production readiness.
External scientific review, human-calibrated alignment evaluation,
consciousness research, and production authorization remain separate gates.
