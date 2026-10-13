# Replacement-loss V18: projected causal refinement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v18-projected-causal-refinement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v18-projected-causal-refinement`.

## Question and claim ceiling

V17 showed that finite-intervention supervision improved some local validation
effects but did not retain pointwise function or reliably predict held-out
causal effects. V18 asks whether a predeclared gradient-conflict projection,
applied after pointwise training, can improve held-out causal-effect
prediction while retaining the adaptation-set pointwise fit.

This is an outcome-aware, post-V17 exploratory follow-up using the same reused
OPT/ReLU checkpoint and visible-selected block 8. It does not confirm a fresh
checkpoint or generalize across architecture families.

Maximum claim ceiling:
`LOCAL_PROJECTED_CAUSAL_REFINEMENT_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen target, fixtures, and training

- Reuse `grushaprasad/opt125m-rootp-surprisal-curriculum` at revision
  `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`, tree SHA-256
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`, and
  weight SHA-256
  `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
- Fix block 8 from V16's archived visible-only selection. Block 9 remains the
  wrong-layer control. Neither layer is reselected after V16/V17 outcomes.
- Generate three new synthetic adaptation partitions with seeds 202610501,
  202610502, and 202610503. Each has 96 prompts, balanced across ordinary,
  compositional, and adversarial strata. Split by prompt within each stratum
  into 24 fit and 8 validation prompts before model capture. V16 and V17
  challenge fixtures, predictions, outcomes, and row-level records are never
  training or validation examples.
- At each captured fit-token `fc1` input, create one fixed local direction
  with norm `0.01 * RMS(x)`: Gaussian for ordinary, normalized sum of two
  Gaussian draws for compositional, and a randomly signed largest-magnitude
  coordinate for adversarial. The teacher target is the module's finite
  response `f(x + delta) - f(x)`. Directions and targets are generated from
  fit prompts only; validation is report-only.
- For block 8 and separately for block 9, train an independently initialized
  pointwise candidate for exactly 1,000 AdamW updates, batch size 128,
  learning rate `1e-3`, weight decay `1e-4`, and gradient clipping `1.0`.
  Both candidates use the normalized pointwise output MSE. The block 8
  pointwise candidate is the treatment comparator; the block 9 candidate is
  the wrong-layer control.
- Copy each frozen pointwise candidate as the starting state for its matching
  projected refinement. For exactly 1,000 further updates, compute normalized
  pointwise loss `L_v` and normalized finite-effect loss `L_c` on the same
  fit minibatch. Let `g_v = grad(L_v)` and `g_c = grad(L_c)`. If
  `dot(g_c, g_v) < 0` and `||g_v||^2 > 1e-24`, replace `g_c` with
  `g_c - dot(g_c,g_v)/||g_v||^2 * g_v`; otherwise leave it unchanged. Apply
  one plain SGD update of `g_v + projected(g_c)`, global norm clipped to 1.0,
  at learning rate `1e-4`, with no momentum or weight decay. Use a fixed
  deterministic draw stream per seed and layer. Record conflict frequency,
  pre/post-projection gradient dot products, draw digest, and candidate digests.
- Before challenge generation, require each projected candidate's normalized
  full-fit pointwise MSE to be no more than 1.25 times its matched pointwise
  candidate's value MSE (with a `1e-12` denominator floor). This is a frozen
  fit-set gate, not a held-out selection rule. Record validation values and
  effects without tuning or early stopping.
- Compare projected block 8, its matched pointwise-only candidate, the
  projected block 9 wrong-layer candidate substituted at block 8, a fresh
  random candidate, and a zero-output no-op. Freeze all candidate bytes and
  prediction code before drawing challenge entropy.

## Held-out evaluation and gates

After candidate freeze, generate 60 fresh challenge prompts per seed, balanced
20 per stratum. Draw 256 bits of operating-system entropy per seed after the
candidate freeze. Require disjoint example IDs and exact prompt-plus-
continuation content from the archived V16 visible fixture and all V16/V17
adaptation fixtures. Derive held-out direction streams from each secret and
consume them continuously in prompt order across inference batches.

Seal every method's local module-effect and next-logit-effect predictions
before computing teacher outcomes. In each seed and stratum, require at least
16/20 resolvable teacher effects. For the projected block 8 treatment,
require module and next-logit effect direction accuracy `>=0.80`, median
cosine `>=0.90`, and median relative error `<=0.20`; base activation median
cosine `>=0.90` and relative error `<=0.20`; continuation cross-entropy
increase `<=0.05` nats/token; and normalized next-logit behavior error
`<=0.05`.

For every seed and stratum, require projected block 8 to improve next-logit
effect cosine over matched pointwise-only by `>=0.05` and reduce median
relative effect error by `>=0.10`. Require its next-logit effect cosine to
exceed the wrong-layer, random, and no-op controls by `>=0.05`. The
teacher-output restoration check must have maximum absolute logit error
`<=1e-5`; this is an oracle instrumentation control, not learned activation
rescue. Overall outcome is `PASS` only when every frozen gate passes; a
measured miss is `FAIL`; missing resolvability, digest mismatch, or resource
termination is `INCONCLUSIVE`.

## Execution, custody, and nonclaims

Use the exact V16/V17 Python 3.12.13 package inventory, local CPU, one Torch
thread, and deterministic algorithms. Cap each run at four hours, 16 GiB peak
RSS, 200 million checkpoint parameters, and 800 MiB checkpoint weights.
Archive the packet before fixture generation. Retain prompt-bearing records,
captures, fit arrays, candidate tensors, challenge entropy, predictions, and
row-level outcomes only in external owner custody. Repository results contain
aggregate metrics, protocol identity, and digests only. Preserve failed and
stopped receipts; never reuse a stopped custody root.

No provider, cloud, GMan, remote worker, paid run, external review submission,
public write, or production integration is authorized. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`.

This protocol cannot establish architecture-general validity, consciousness,
alignment, production readiness, independent science, subjective identity, or
a novel proof of Theseus continuity. Same-owner second-runtime replay can
establish `REPRODUCED_LOCAL` only. Blind external replication and authorized
independent review remain required for independent scientific acceptance.
