# Replacement-loss V17: causal-supervised whole-MLP replacement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v17-causal-supervised-whole-mlp-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v17-causal-supervised-whole-mlp`.

## Question and claim ceiling

V16 found that pointwise-trained whole-MLP replacements preserved unperturbed
OPT behavior while failing held-out causal-effect prediction. V17 asks a
narrower mechanistic question: does adding adaptation-only local-intervention
supervision improve causal prediction on newly generated held-out challenges,
compared with an equal-architecture, equal-update pointwise-only candidate?

This is a post-V16, outcome-aware follow-up on the same reused OPT/ReLU
checkpoint and preselected block 8. It is exploratory, not a fresh-checkpoint
confirmation. V16 challenge entropy, prompts, outcomes, candidate tensors, and
prediction arrays are not V17 training inputs.

Maximum claim ceiling:
`LOCAL_CAUSAL_SUPERVISION_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen target and training data

- Reuse checkpoint
  `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`
  with tree SHA-256
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466` and
  weight SHA-256
  `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
- Fix the intervention target to OPT decoder block 8, based on V16's
  visible-only selection. V16 selected the adjacent block 9 as its wrong-layer
  control. V17 does not reselect after seeing V16 outcomes.
- Use three new fixed synthetic adaptation seeds, 96 prompts per seed (32 per
  ordinary, compositional, and adversarial stratum). Split each stratum by
  prompt into 24 fit and 8 validation prompts before model capture.
- For every fit-token activation `x`, generate one deterministic adaptation
  direction `δ` of L2 norm `0.01 * RMS(x)`: Gaussian for ordinary,
  two-Gaussian normalized sum for compositional, and signed largest-magnitude
  coordinate for adversarial. The teacher target is the original module's
  finite effect `f(x+δ)-f(x)`.
- Train a fresh `Linear(768,3072) -> ReLU -> Linear(3072,768)` candidate for
  exactly 1,000 AdamW updates, batch size 128, learning rate `1e-3`, weight
  decay `1e-4`, and gradient clipping `1.0`. The causal-supervised objective
  is the equal-weight mean of normalized pointwise output MSE and normalized
  finite-effect MSE. Both normalizers are computed on fit data only. No
  validation-based selection or early stopping is allowed.
- Train an equal-budget pointwise-only treatment comparator on the same fit
  prompts from the same initialization, a causal-supervised wrong-layer
  candidate on block 9, an untrained random candidate, and a zero-output
  no-op. Candidate bytes and all
  predictions code are frozen before challenge entropy is drawn.

## Held-out evaluation and gates

After candidate freeze, generate 60 fresh challenge prompts per seed, balanced
over the same three strata. Draw 256 bits of OS CSPRNG entropy per seed after
candidate freeze. Require disjoint example IDs and exact prompt-plus-
continuation content from all V17 adaptation fixtures and the archived V16
visible fixture.

At block 8's `fc1` input on the final prompt token, generate held-out
intervention directions using the same three frozen stratum rules and
`0.01 * RMS(x)` magnitude. Seed one per-seed direction generator from the
first eight bytes of `SHA256(entropy || "v17-heldout-directions")` and consume
it continuously in prompt order across inference batches. Seal
causal-supervised, pointwise, wrong-layer,
random, and no-op predictions before scoring against original-model outcomes.

The causal-supervised treatment must satisfy every gate for each seed and
stratum:

- At least 16 of 20 observed module and next-logit effects are resolvable;
  resolvable means observed L2 norm exceeds
  `max(1e-8 * baseline-output L2 norm, 1e-12)`.
- For both module and next-logit effects: direction accuracy `>=0.80`, median
  cosine `>=0.90`, and median relative effect error `<=0.20`, where relative
  error is `||prediction-observation||₂ / ||observation||₂`.
- Replaced-boundary activation median cosine `>=0.90` and relative error
  `<=0.20`.
- Unperturbed challenge continuation cross-entropy increase `<=0.05`
  nats/token and normalized next-logit error `<=0.05`.
- For every seed/stratum, causal supervision improves median next-logit effect
  cosine over pointwise-only by `>=0.05`, and lowers next-logit median relative
  effect error by `>=0.10`.
- For every seed/stratum, causal treatment's median next-logit effect cosine
  exceeds wrong-layer, random, and no-op controls by `>=0.05`.
- Teacher-output restoration control has maximum absolute next-logit error
  `<=1e-5`; it is an oracle pipeline control, not a learned rescue.

Overall outcome is `PASS` only if every gate passes. A measured miss is `FAIL`;
missing resolvability, digest mismatch, or resource termination is
`INCONCLUSIVE`. Three seeds on one reused checkpoint are descriptive only.

## Runtime, custody, and nonclaims

Use the exact V16 Python 3.12.13 package inventory, local CPU, one Torch thread,
and deterministic algorithms. Each run is capped at four hours, 16 GiB peak
RSS, 200 million checkpoint parameters, and 800 MiB checkpoint weights.
Preserve partial receipts and never reuse a stopped custody root.

Archive this packet before fixture generation. Keep prompt-bearing records,
captures, causal training arrays, candidate tensors, entropy, predictions, and
row-level outcomes only in external owner custody. Repository results may
contain aggregate metrics and digests only. No provider, cloud, GMan, remote
worker, paid run, external review submission, public write, or production
integration is authorized by this packet. Monetary cost is `NOT_APPLICABLE`;
energy is `NOT_MEASURED`.

This protocol cannot establish architecture-general validity, consciousness,
alignment, production readiness, independent science, or subjective identity.
A separate-runtime replay can establish `REPRODUCED_LOCAL` only. Blind
external replication and authorized independent review remain required for
independent scientific acceptance.
