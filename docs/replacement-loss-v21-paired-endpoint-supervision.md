# Replacement-loss V21: paired endpoint supervision

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v21-paired-endpoint-supervision-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol IDs: `theseus-replacement-loss-v21-development` and
`theseus-replacement-loss-v21-confirmation`.

## Question and claim ceiling

Does training an independently initialized replacement to match the teacher's
outputs at both the factual module input `x` and a paired intervention input
`x + delta` improve held-out causal-effect prediction while retaining
ordinary behavior? This tests endpoint-value supervision, derived from the
conditional bound in
[`causal-effect-identifiability-boundary-v1`](causal-effect-identifiability-boundary-v1.md).

This is a development qualification on the previously used OPT/ReLU
checkpoint and visible-selected block 8. It is not fresh-checkpoint,
architecture-general, or blind external evidence.

Maximum claim ceiling:
`LOCAL_PAIRED_ENDPOINT_SUPERVISION_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen method

Bind the V20B negative development packet, results, and same-owner replay.
Use the same local OPT checkpoint, block 8 target, block 9 wrong-layer
control, synthetic prompt strata, capture/evaluation path, optimizer budget,
runtime caps, and held-out gates as V20B. V20B's fixtures and candidate
artifacts are excluded from V21 data.

For each of three new development seeds and each of the ordinary,
compositional, and adversarial strata:

1. Generate 64 synthetic prompts; split each stratum into 48 fit and 16
   validation prompts before capture. Prompts are the split unit.
2. Generate fit and validation directions independently within their prompt
   partitions. The teacher endpoint targets are `m(x)` and `m(x + delta)`;
   verify the latter against the captured base output plus the teacher's
   finite effect before training.
3. Train an independent full-width pointwise base for 1,000 AdamW updates
   using fit-only `m(x)` targets. Copy no source-model parameters.
4. Freeze the pointwise base. Attach rank-16 residual factors to both affine
   projections, initialized so the residual initially equals the pointwise
   base. Train only those factors for 1,000 AdamW updates on the single
   objective

   ```text
   [MSE(g(x), m(x)) + MSE(g(x + delta), m(x + delta))]
   / [2 * mean-square of the pooled teacher endpoint targets].
   ```

   The two endpoint terms have equal weight. Do not include a finite-effect
   loss, a base-drift anchor, a validation-based model choice, or early
   stopping. The optimizer is AdamW with learning rate `1e-3`, weight decay
   `1e-4`, batch size 128, and gradient clipping at `1.0`.
5. Freeze candidate bytes before validation scoring. Compare with the paired
   pointwise base, random untrained replacement, and zero-output no-op. The
   confirmation stage also trains the same endpoint-supervised method on the
   wrong-layer control.

The endpoint targets are generated only from adaptation fit prompts during
training. Validation endpoint targets are used for qualification only. The
candidate predicts held-out effects as `g(x + delta) - g(x)`; held-out teacher
effects remain sealed until prediction is frozen.

## Development qualification and confirmation

Report nine cells (three seeds by three strata). Each development cell must
pass all V20B gates:

- validation finite-effect MSE divided by the same-row zero-effect MSE `<=
  0.90`;
- validation pointwise value MSE divided by the paired pointwise base MSE
  `<= 1.25`;
- continuation cross-entropy increase `<= 0.05` nats/token; and
- normalized next-logit error `<= 0.05`.

Require every cell to pass in both the primary and separate pinned runtime,
with byte-identical fixtures, splits, captures, candidates, and selection
reports. A measured gate miss stops before confirmation fixtures or challenge
entropy. A mismatch or resource stop is `INCONCLUSIVE`.

Report behavior/mechanism divergence as true when any cell passes both
behavior gates but fails the causal-effect gate, or fails behavior while
passing the causal-effect gate. A cell's behavior gates are continuation
cross-entropy and next-logit error; its development causal gate is the
finite-effect ratio. This flag does not replace the individual gate results.

Only after qualified, reproduced development may the frozen confirmation
packet be created. Confirmation uses three new seeds, repeats the nine-cell
prechallenge qualification, and runs the frozen V17 challenge predictor and
scorer only after primary and replica preparation match and every cell
qualifies. Each confirmation seed has 20 held-out prompts per stratum;
predictions are frozen before challenge scoring. Preserve V17's absolute
module-effect and next-logit-effect gates, activation and behavior gates,
wrong-layer/random/no-op margins, and oracle-restoration check unchanged.
Oracle restoration is a pipeline control, not learned rescue.

No thresholds may be retuned from V21 validation or confirmation results. A
failure does not justify a follow-on packet with relaxed gates.

## Runtime, custody, and nonclaims

Use the already pinned Python 3.12.13 primary and replica environments,
PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6, deterministic CPU, and one
Torch thread. Cap each phase at four hours and 16 GiB RSS. Monetary cost is
`NOT_APPLICABLE`; energy is `NOT_MEASURED`. No provider, cloud, GMan, paid
compute, remote worker, public write, external-review submission, or
production integration is in scope.

Prompts, captures, endpoint targets, intervention directions, candidates,
challenge entropy, predictions, and row-level outcomes remain in external
owner custody at `/Volumes/PrimaryED/ResearchArtifacts`. The repository keeps
only protocol code, aggregate-safe receipts, and digests. Preserve prior
V20/V20B roots and every failed receipt.

Even a passing `REPRODUCED_LOCAL` result would establish only this method on
one reused OPT checkpoint. It would not establish architecture-general
learned replacement, independent science, novelty priority, consciousness,
alignment, or production readiness. Blind external replication and an
authorized independent reviewer remain separate gates.
