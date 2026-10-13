# Replacement-loss V25: metadata-complete fresh checkpoint acquisition

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v25-metadata-complete-acquisition-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v25-metadata-complete-acquisition`.

## Why V25 exists

The V24 packet was frozen and its bounded screen completed, but the query did
not request immutable revision metadata. Its reported
`NO_ELIGIBLE_FRESH_TARGET` is therefore audited as inconclusive. V25 corrects
that metadata request using the pinned `huggingface_hub==0.36.2` client's
`expand` projection, including `sha`, `config`, `gated`, `private`, `downloads`,
and `tags`. A query with results but no valid commit SHA fails closed as
`INCONCLUSIVE_SETUP_OR_EXECUTION`; it cannot be reported as no target.

V25 excludes every model identity returned by V24, all V3-V23 previously used
or screened identities, and every identity in its own post-freeze candidate
ledger. This is fresh-checkpoint acquisition inside OPT/ReLU, GPT-2/GELU, and
Llama/SiLU-gated families, not an unseen-architecture test.

## Frozen screen and eligibility

After the V25 packet is frozen, make public unauthenticated Hub requests with
`token=False`, `gated=False`, downloads-descending order, deterministic
`(downloads, model_id, full revision)` ordering, and at most 2,000 results per
query. Search `opt`, `gpt2`, `llama`, `smollm`, `open-llama`, and `tinyllama`.
Request only the frozen metadata projection; do not call hosted inference.
Deduplicate by `(model_id, full revision)` and exclude all pre-V25 identity
ledger entries before eligibility checks.

Select the first eligible candidate per family. Require public and ungated
status, a full 40-character revision, model type `opt`, `gpt2`, or `llama` as
appropriate, declared license metadata, no frozen instruction/chat/fine-tune/
quantization/distillation/adapter name or tag markers, supported config,
tokenizer and safetensors files, and an exact safetensors-header parameter
count no greater than 500,000,000. No selected model may exceed 2.2 GB of
allowlisted files; all three together may not exceed 6.6 GB. Do not widen
queries or relax filters after freeze. Persist query projections,
rejections, selected revisions, license metadata, and file digests only in
external custody.

If any family has no eligible candidate within the frozen result bound, stop
with `NO_ELIGIBLE_FRESH_TARGET`. Metadata incompleteness, API errors, or
custody failures produce `INCONCLUSIVE_SETUP_OR_EXECUTION` instead.

## Execution and claim ceiling

Require the explicit `--allow-public-hub-downloads` flag before any model file
download. Cache roots, checkpoints, prompts, captures, candidates, predictions,
outcomes, and diagnostics stay under the pinned external V25 custody root.
Use local CPU only, `trust_remote_code=False`, direct monetary budget `$0`, and
the unchanged V22/V8 nine-case treatment, fixtures, controls, thresholds, and
resource limits. A clean Python 3.12.13 runtime must reproduce the primary
candidate, prediction, outcome, dependency, and source digests for
`REPRODUCED_LOCAL`.

The packet allows one metadata screen, primary evaluation, and clean
reproduction. Retain phase-specific stop receipts and partial outputs; do not
retry or retune a frozen packet. The frozen V8 method reports behavior, causal
effects, and activation rescue separately but has no formal behavior-versus-
mechanism divergence statistic, so record that limitation explicitly.

Assurance may reach `INTERNAL_PASS` for a complete local protocol run and
`REPRODUCED_LOCAL` after exact same-owner reproduction. Reviewer status remains
`NOT_SUBMITTED`. The claim ceiling is fresh-checkpoint replication within the
three named families. This does not establish unseen-architecture validity,
mechanistic understanding beyond the frozen intervention metrics, independent
science, consciousness, alignment, or production readiness.
