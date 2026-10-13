# Replacement-loss V19 development: causal/value objective qualification

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v19-causal-fit-qualification-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v19-causal-fit-development`.

## Question and claim ceiling

V17's equal-weight causal objective improved local effect fit while degrading
value retention. V18's projected objective retained value fit but remained
worse than a zero-effect predictor on its own fit data. V19 development asks
whether a predeclared causal/value loss-weight grid contains a candidate that
learns finite effects and retains the pointwise function on new adaptation
data.

This is a post-result development screen on the already reused OPT/ReLU
checkpoint and block 8. It is not a fresh-checkpoint confirmation and has no
architecture-general claim.

The later V19 held-out packet retains V17's module-output preservation metric
and oracle activation-restoration pipeline control. The oracle control is not
a learned activation-rescue result, and V19 does not claim one.

Maximum development claim ceiling:
`LOCAL_CAUSAL_VALUE_PARETO_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen development design

- Target the archived OPT checkpoint from V16 at block 8; block 9 is the
  wrong-layer comparator. Do not download or select new weights.
- Generate three adaptation seeds (`202610601`–`202610603`), each with 64
  synthetic prompts per stratum: ordinary, compositional, and adversarial.
  Split by prompt within stratum into 48 fit and 16 development-validation
  prompts before model capture.
- Generate deterministic fit and validation intervention directions at norm
  `0.01 * RMS(x)` and teacher effects `f(x + delta) - f(x)` within their
  already disjoint prompt partitions. Only fit targets enter gradient updates;
  validation directions and effects are used exclusively for scoring.
- Train independently initialized whole-MLP candidates for loss weights
  `lambda = 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00`, where each update
  minimizes `(1-lambda) * normalized_value_MSE + lambda * normalized_effect_MSE`.
  Train a paired `lambda=0` pointwise-only baseline with the same seed,
  initialization, minibatch draw stream, architecture, and update budget
  (1,000 AdamW updates, batch size 128, learning rate `1e-3`, weight decay
  `1e-4`, gradient clipping `1.0`). No optimizer, weight, seed, or threshold
  may be added after this packet is frozen.
- Score every candidate by prompt and stratum on development validation. For
  each cell, compare candidate effect MSE with the zero-effect predictor's
  effect MSE on the exact same validation prompts and intervention directions.
  Also report effect MSE divided by the fit-set mean squared teacher effect;
  that normalized value is descriptive and is not the qualification baseline.
- A single global loss weight qualifies only if, in all nine seed-by-stratum
  cells, its mean prompt-level validation effect-MSE ratio to the paired
  zero-effect predictor is `<=0.90`, its mean prompt-level normalized
  validation value MSE is `<=1.25` times the paired pointwise candidate's
  value MSE, its continuation cross-entropy increase is `<=0.05` nats/token,
  and its median normalized next-logit error is `<=0.05`. This sets a minimum
  meaningful 10% effect-loss reduction while limiting value-loss drift to
  25% and preserving the frozen behavior gates.
- If more than one weight qualifies, select the one with the lowest
  equal-cell mean validation effect-MSE ratio to the paired zero-effect
  predictor; break an exact tie toward the lower lambda. If none qualifies, record
  `NO_QUALIFIED_DEVELOPMENT_CANDIDATE` and do not make a confirmatory packet or
  draw challenge entropy.
- Development is a method-selection stage, not confirmatory evidence. Repeat
  it in the pinned separate Python runtime; require byte-identical generated
  fixtures, captures, candidate tensors, and selection report before freezing
  confirmation.

## Cost, custody, and nonclaims

Use the pinned V16/V17 Python 3.12.13 inventory, local CPU, one Torch thread,
and deterministic algorithms. Cap each process at four hours and 16 GiB RSS.
Monetary cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`.

Store prompts, captures, fit arrays, candidates, and row-level development
metrics only in external owner custody. The repository may contain aggregate
metrics and digests only. No provider, cloud, GMan, paid compute, public write,
or production integration is in scope.

Development cannot establish held-out causal prediction, architecture-general
validity, independent science, consciousness, alignment, production
readiness, or Theseus continuity. A selected weight advances only to the
separately frozen V19 confirmatory packet.
