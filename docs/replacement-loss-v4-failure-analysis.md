# Replacement-loss-v4 adversarial failure diagnosis

State slice: `theseus-lab-sandbox-v1`.

Protocol ID: `theseus-replacement-loss-v4-adversarial-failure-analysis-v1`.

Status: preregistered post-hoc diagnostic. The v4 primary and replication
outcomes are frozen and remain `FAIL`; this analysis does not change its gates
or disposition. The analysis packet must be digest-bound before opening any
row-level v4 prediction or outcome values.

## Question and scope

The v4 compositional-augmented arm exceeded the locked adversarial next-logit
error ceiling in all three GPT-2 seeds. This one-pass diagnosis distinguishes a
small upper-tail concentration from a broader adversarial mismatch, checks
whether per-row activation and next-logit errors move together, and measures
the paired adversarial difference between the ordinary-only and
compositional-augmented candidates.

The exact target remains GPT-2 medium revision
`6dcaa7a952f72f9298047fd5137cd6e4f05f41da`, layer 3 `mlp.c_proj`, and the
already frozen v4 synthetic challenge set. This is descriptive analysis of
three seeds with 20 adversarial prompts per seed. Prompt rows are not treated
as independent trials. No p-values, new thresholds, or confirmatory claim are
permitted.

## Frozen inputs and metrics

Before loading row-level arrays, freeze the v4 packet digest, primary and
replica report digests, replication receipt digest, and digests for all three
prediction arrays and all three outcome arrays. Require the replica to report
matching prediction and outcome digests.

For each seed, candidate arm, and adversarial challenge row, compute:

- normalized activation error: `||candidate - original|| / max(||original||,
  1e-12)`;
- normalized next-logit error using the same normalization;
- each metric's median, 90th percentile, maximum, and count above the v4
  behavior ceiling `0.05` for next-logit error;
- Spearman rank correlation between the two row-wise errors, computed over the
  20 adversarial rows;
- paired compositional-augmented minus ordinary-only median error differences
  and the paired fraction of rows improved, separately for activation and
  next-logit error.

Classify an adversarial logit failure as `UPPER_TAIL_CONCENTRATED` only when at
most 2 of 20 rows exceed `0.05`; otherwise classify it as `BROAD_GUARD_MISS`.
This descriptive label does not alter the v4 `FAIL`. Report the per-seed
metrics and the across-seed range only. Do not add metrics after opening the
arrays.

## Custody and limits

The diagnostic reads the digest-bound v4 NPZ files from owner custody and
emits aggregate statistics only. It must not print or persist prompt text,
example IDs, tokens, per-row errors, activations, or logits. The analysis is
post-hoc and cannot establish causality, mechanistic understanding,
architecture-general validity, independent science, alignment, consciousness,
or production authorization. Reviewer status remains
`EXTERNAL_REVIEW_UNAVAILABLE` unless a separately authorized external review
returns a signed decision.

