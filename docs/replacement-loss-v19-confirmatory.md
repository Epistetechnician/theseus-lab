# Replacement-loss V19 confirmatory packet

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v19-causal-fit-qualification-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v19-causal-fit-confirmation`.

## Entry condition and claim ceiling

Create this packet only when V19 development has selected a single globally
qualifying loss weight and its primary and separate-runtime development
receipts match exactly. Bind the selected weight and development result digest
into this freeze. If no development weight qualifies, this packet must not be
created.

This post-result confirmation reuses the previously selected OPT/ReLU
checkpoint and block 8. Its maximum claim is
`LOCAL_CAUSAL_VALUE_PARETO_CONFIRMATION_ON_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen confirmation design

- Use fresh confirmation adaptation seeds `202610701`–`202610703`, with 64
  synthetic prompts per stratum and the same 48-fit/16-validation split.
  Prompt IDs and exact prompt-plus-continuation content must be disjoint from
  V16, V17, V18, V19 development, and other confirmation partitions.
- Train one causal/value candidate per seed using the selected development
  weight, plus paired pointwise, wrong-layer, and random controls. Reuse the
  exact frozen update budget and runtime from development.
- Before drawing challenge entropy, require the fixed candidate to meet the
  same `<=0.90` validation effect-MSE ratio to the zero-effect predictor on
  the exact same validation prompts and directions, and `<=1.25x` paired
  pointwise validation value-MSE criteria in all nine seed-by-stratum cells.
  Also require validation continuation cross-entropy increase `<=0.05`
  nats/token and normalized next-logit error `<=0.05` in every cell. These are one-shot
  eligibility gates; they cannot be used to retune lambda.
- If any pre-challenge gate fails, record
  `CONFIRMATION_STOPPED_BEFORE_CHALLENGE`, preserve candidates and receipts,
  and do not generate challenge entropy. This is an eligibility stop, not a
  held-out scientific pass.
- Only after eligibility passes, generate 60 fresh challenge prompts per
  seed (20 per stratum) from 256-bit OS entropy after candidate freeze. Seal
  all candidate predictions before scoring.
- Apply the V17 held-out absolute gates unchanged: at least 16/20 resolvable
  effects per cell; module and next-logit effect direction accuracy `>=0.80`,
  cosine `>=0.90`, and normalized error `<=0.20`; activation cosine `>=0.90`
  and relative error `<=0.20`; behavior cross-entropy increase `<=0.05` and
  next-logit error `<=0.05`; causal-vs-pointwise cosine gain `>=0.05` and
  error reduction `>=0.10`; wrong-layer, random, and no-op next-logit cosine
  margins `>=0.05`; oracle restoration max error `<=1e-5`.
- The activation cosine/error fields test module-output preservation. Oracle
  restoration tests the intervention instrumentation only; neither is a
  learned activation-rescue test, which remains outside this packet's claim.
- A measured challenge-gate miss is `FAIL`; missing resolvability, digest or
  runtime mismatch, or resource termination is `INCONCLUSIVE`. Never tune on
  confirmation validation or challenge outcomes. A completed primary must be
  replayed from this exact packet in the separate pinned Python executable.

## Execution and nonclaims

Use local CPU, deterministic algorithms, one Torch thread, the pinned package
inventory, four-hour and 16 GiB caps. Keep all prompt-bearing and row-level
artifacts in external owner custody. No paid, remote, provider, public, or
production activity is authorized.

Even `REPRODUCED_LOCAL` would not establish independent acceptance,
architecture-general validity, consciousness, alignment, production
readiness, or a novel proof of Theseus continuity. Blind external replication
and authorized independent review remain separate gates.
