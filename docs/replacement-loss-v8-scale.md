# Replacement-loss-v8-scale: size scaling of the independently fitted affine replacement

State slice: `theseus-lab-sandbox-v1` (continuation; scale-up authorized by the owner in chat 2026-10-22).

Protocol ID: `theseus-replacement-loss-v8-scale-ridge-primary-causal`.

Status: preselection preregistration, DRAFT. Nothing here is frozen or run. v3-v8 remain immutable; the v8 `PASS` is the reference result and is not retuned.

## Question and claim ceiling

Does the v8 result (a fresh affine MLP output projection, fitted by closed-form ridge from disjoint adaptation pairs, predicts held-out activations, behavior and causal effects) hold at larger scale, where v8 was capped at 500M parameters?

A pass supports only this local replacement procedure for one projection per checkpoint at the tested sizes. It says nothing about whole models, other architectures, mechanism, alignment, consciousness, or production readiness.

## Changes from v8 (everything else is pinned unchanged)

- Parameter ceiling raised from 500M to 9B, with two bands: Band A 500M-3B, Band B 3B-9B. One checkpoint per family (OPT/ReLU, GPT-2/GELU, Llama/SiLU-gated) per band where an eligible public checkpoint exists; a family with no eligible checkpoint in a band is reported `NO_ELIGIBLE_FRESH_TARGET` for that band, not substituted.
- Full 40-character immutable revisions, `trust_remote_code=False`, absent from all v3-v8 checkpoint identities.
- Compute: one H100 per case via `gman job`, bf16 weights, fp32 ridge solve. Compute location is not part of the scientific claim; outputs must still match the v8 digests procedure.
- Fixture namespace `v8-scale`, new seeds `20261401`-`20261403`, visible seed `20261400`. Counts, strata, prompts-per-stratum unchanged.
- Thresholds, ridge value `1e-6`, controls, cluster bootstrap and `1e-4` permutation tolerances: unchanged from v3/v8. No post-selection retuning.

## Outcomes

`PASS` per band only if every case and aggregate gate passes. Measured misses are `FAIL`. Missing fixtures, digest mismatch, unsealed predictions, or resource termination are `INCONCLUSIVE` and are not retried under this packet. Also report, as descriptive only, how pass margins move with parameter count.

## Budget and stop rule

Cap $300 on gman. Stop at the first `INCONCLUSIVE` resource failure and record it.

## Work remaining before freeze

1. Thin runner: override `MAX_PARAMETER_COUNT`, device and dtype in the v3 chain, add scale fixtures namespace, tests.
2. Freeze packet with pinned v7/v8 hashes (new freeze file, no edits to v8's).
3. Only then select checkpoints.
