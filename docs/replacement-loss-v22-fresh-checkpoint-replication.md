# Replacement-loss V22: fresh-checkpoint replication

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v22-fresh-checkpoint-replication-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v22-fresh-checkpoint-replication`.

## Question and claim ceiling

Does the frozen V8 independently fitted affine MLP output-projection method
replicate on fresh checkpoint identities selected only after this packet is
frozen, within the already declared OPT/ReLU, GPT-2/GELU, and
Llama/SiLU-gated families?

This is a checkpoint-generalization replication within three predeclared
families. It is not an architecture-held-out test and does not establish
validity across all architectures, whole-model identity, consciousness,
alignment, independent science, or production readiness. The additive-module
theorem remains limited to additive components.

## Frozen procedure

V22 reuses the exact frozen V8 runner, treatment, evaluator, controls,
thresholds, and dependency versions. The upstream V8 packet digest is
`b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`.
V22 changes only the external-custody fixture namespace/seeds and the
post-freeze checkpoint manifest. The V8 source and freeze digests are bound in
the V22 freeze receipt. If those V8 bytes or the V22 packet fail verification,
stop before target selection.

After freeze, select exactly one local, already custodied public base
checkpoint per family. Each checkpoint must have an immutable 40-character
revision, no more than 500 million parameters, load with
`trust_remote_code=False`, satisfy the frozen V8 projection contract, and be
absent from checkpoint identities recorded by V3–V21. Do not download weights,
call model providers, use cloud/GMan nodes, or spend money. If any family has no
eligible local checkpoint, stop with `NO_ELIGIBLE_FRESH_TARGET` before candidate
preparation. Record exact identities only in external custody.

Run the same nine cases and V8 thresholds: three fixed seeds per architecture
family; visible-only projection selection; 128 adaptation prompts and 60
held-out challenge prompts per case, split across ordinary, compositional, and
adversarial strata. V22 fixture IDs and prompt prefixes are distinct from V8.
Check example IDs, prompts, and prompt-plus-continuation against custodied
fixtures from V3–V21 before candidate fitting. Candidate predictions are sealed
before held-out outcomes are opened. Keep prompts, captures, weights,
candidates, predictions, outcomes, and row-level diagnostics outside the
repository.

Use local CPU only and the V8 caps: 7,200 seconds per case, 64,800 seconds
total, and 24 GiB peak RSS. Run the primary and a second clean Python 3.12.13
environment with identical pinned dependencies. `REPRODUCED_LOCAL` requires the
V8 exact candidate reconstruction, prediction/outcome digest, dependency,
source-identity, and distinct-environment checks to pass. This remains
same-owner local reproduction.

## Outcomes and gates

Use the V8 gates without retuning: causal effect direction accuracy ≥0.80,
median cosine ≥0.90, median normalized error ≤0.20; held-out next-logit
normalized error ≤0.05 and continuation cross-entropy increase ≤0.05
nats/token; activation cosine ≥0.90 and normalized error ≤0.20; all frozen
cluster-bootstrap and mapping-aligned permutation gates; all challenge strata
resolvable. Overall experiment outcome is `PASS` only if every case and
aggregate gate passes. Measured misses are `FAIL`; missing inputs, digest
mismatches, incomplete prediction sealing, or resource termination are
`INCONCLUSIVE`. Reproduction assurance is reported independently of experiment
outcome.

## Authority, custody, and nonclaims

Reviewer status is `NOT_SUBMITTED`; this local task does not create an external
review task. No human-calibrated alignment, consciousness, deployment, or
production authorization is requested or implied. Keep those gates closed.
Report source revision and state slices, assurance, reviewer status,
reproducibility, custody, limitations, claim ceiling, and the next authority
gate. Preserve all failed or incomplete receipts.
