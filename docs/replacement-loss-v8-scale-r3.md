# Replacement-loss-v8-scale-r3: confirmatory coverage packet

State slice: `theseus-lab-sandbox-v1`. Protocol ID: `theseus-replacement-loss-v8-scale-r3-coverage-confirm`.

Status: preselection preregistration, frozen by `docs/replacement-loss-v8s3-freeze.json` before any run. v3-v8, v8-scale and v8-scale-r2 remain immutable.

## Question

In r2 the C4 arm (4x adaptation prompts, 512) reached pooled median cosine 0.98-0.999 and passed all nine case gates, but was recorded INCONCLUSIVE because its peak memory (27.1 GiB) exceeded the frozen 24 GiB limit; the S1/S4/S16 arms failed (`docs/replacement-loss-v8-scale-r2-results.md`). Does the coverage effect replicate on fresh seeds and fixtures, and how does it scale with prompt count (2x vs 4x), when the memory limit is set high enough to measure it?

## Design (everything else pinned from v3/v8/r2)

- Checkpoints: the same three Band A/r2 models and revisions (declared deliberate reuse), re-bound after this freeze.
- Fresh seeds `20261601`-`20261603`, visible `20261600`, new `V8S3` fixture namespace. Overlap checks regenerate v3/v6/v7/v8/v8-scale/r2 visible and adaptation fixtures.
- Arms: **C2** (256 adaptation prompts) and **C4** (512). Intervention scale 1x in both. No S1 control is re-run: S1 on fresh r2 seeds and Band A already established the 128-prompt baseline (0/9 cases pass twice); this is stated, and the comparison is against those two prior baselines, not a concurrent control.
- Thresholds, ridge `1e-6`, controls, cluster bootstrap, per-case wall cap (7200 s) and total wall cap are unchanged.
- **Only change to limits:** peak memory limit 24 -> 30 GiB. The frozen v3 evaluator hard-codes 24 GiB and stays immutable, so `replacement_loss_v8s3_runner._reclassify_outcome` re-derives the outcome from the evaluator's own recorded case results, bootstrap values and equivalence diagnostics using the identical gate conditions and replaces only the memory limit. Tests cover pass at 27 GiB, incomplete above 30 GiB, and that failed cases, bootstrap and equivalence misses still FAIL.
- Nodes: gman `cpu-8` (32 GiB RAM). Exceeding 30 GiB is INCONCLUSIVE and not retried.

## Outcomes (preregistered)

Per arm: `PASS` only if all nine cases and the aggregate gates pass with no resource blocker; measured misses `FAIL`; resource/identity/missing-output problems `INCONCLUSIVE`, not retried. Descriptive, not gated: pooled and per-stratum median cosine by arm and model.
- Coverage is confirmed if C4 is `PASS`. C2's outcome says whether 256 prompts suffice (descriptive; no gate depends on it).
- If C4 fails or is inconclusive, the r2 C4 result stands as a single INCONCLUSIVE observation and is not upgraded.

## Limits and nonclaims

CPU fp32, same owner, synthetic prompts, one projection per checkpoint, same three checkpoints as the packets that motivated this, no independent review (`EXTERNAL_REVIEW_UNAVAILABLE`). A pass supports only: the independently fitted affine replacement of one projection predicts held-out causal effects when fitted from enough adaptation prompts, at 1.7-2.7B parameters. Not whole-model equivalence, mechanistic understanding, alignment, or production readiness.

## Budget and stop rule

Two gman cpu-8 nodes in parallel; cap $30. No retries.
