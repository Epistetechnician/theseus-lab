# Replacement-loss V16: independently trained whole-MLP replacement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v16-whole-mlp-independent-replacement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v16-whole-mlp-independent-replacement`.

## Question and claim ceiling

V8 reported a reproduced local pass for replacing one affine MLP output
projection. V9 tested sequential projection replacements and failed its
conjunctive retention gate; V14 later found no absolute retention for either
of its low-rank projection candidates, and V15c found no clear rank or update
signal. V16 tests a different unit: the complete OPT feed-forward function
`fc2(ReLU(fc1(x)))`, with both replacement layers independently initialized
and trained without reading source-block weights.

The hypothesis is that a full-width, independently trained replacement can
predict held-out causal effects and preserve downstream behavior at one
visible-selected feed-forward block on this one previously studied OPT
checkpoint. The target is explicitly outcome-aware. No fresh-checkpoint,
architecture-general, cumulative-ladder, novelty-priority, consciousness,
alignment, or production claim is made.

Maximum claim ceiling:
`LOCAL_WHOLE_MLP_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ON_ONE_REUSED_OPT_CHECKPOINT`.

## Frozen target and selection

Use the locally custodied `grushaprasad/opt125m-rootp-surprisal-curriculum`
checkpoint at revision `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`, with target
tree SHA-256
`6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466` and
weight SHA-256
`aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
This checkpoint and its prior results are known before V16 design. Reuse is
deliberate and the result is limited to this target.

After the V16 packet is frozen and archived, generate the visible synthetic
fixture. Select one of the 12 OPT feed-forward blocks by the highest mean
per-feature variance of `fc1` input activations on visible prompts; ties go to
the lower layer index. Select the adjacent block for the wrong-module control
(`layer + 1`, or `layer - 1` for the final layer). No adaptation or challenge
data participates in selection.

## Data partitions and training

- Use one visible fixture of 32 prompts.
- Use exactly three fixed adaptation seeds. Each has 96 prompts, balanced over
  ordinary, compositional, and adversarial strata (32 each).
- Before model capture, split each stratum into 24 fit prompts and 8
  adaptation-validation prompts using the frozen deterministic split rule.
  Prompts, not token rows, are the split unit.
- Train the replacement on valid prompt-token activations from fit prompts
  only. The candidate is `Linear(768, 3072) -> ReLU -> Linear(3072, 768)`,
  independently initialized with no copied source parameters, for exactly
  1,000 AdamW updates, batch size 128, learning rate `1e-3`, weight decay
  `1e-4`, and gradient clipping at `1.0`. Minimize mean squared error of the
  complete feed-forward output, normalized by the fit-target mean square. No
  early stopping, validation-based model selection, or outcome tuning is
  allowed.
- Train the wrong-module control with the same procedure and adaptation
  prompts, targeting the adjacent block. Retain an untrained random candidate
  and a zero-output no-op control.
- Freeze candidate bytes, training receipts, and predictions code before
  drawing challenge entropy or writing challenge fixtures.

## Held-out evaluation

After candidate freeze, generate 60 challenge prompts per seed: 20 ordinary,
20 compositional, and 20 adversarial. Challenge entropy is drawn from the
operating-system CSPRNG after candidate freeze and remains in external owner
custody. The challenge fixture must be disjoint from visible and adaptation
fixtures by example ID and exact prompt-plus-continuation content.

For each prompt, intervene at the selected block's `fc1` input at the last
prompt token. Ordinary directions are Gaussian; compositional directions sum
two independent Gaussian directions and renormalize; adversarial directions
target the largest-magnitude input feature with an entropy-selected sign. Each
intervention has L2 norm `0.01 * RMS(fc1 input)`. The frozen predictor is the
independently trained replacement run inside the otherwise unchanged model.
Commit its module-output and next-logit effect predictions before scoring
against original-model outcomes.

Report each seed and each prompt stratum separately. Treatment gates are
conjunctive:

- At least 16 of 20 challenge effects in every stratum are resolvable; below
  that coverage, classify the case `INCONCLUSIVE`. An effect is resolvable when
  its observed L2 norm exceeds `max(1e-8 * baseline-output L2 norm, 1e-12)`.
- For both feed-forward-boundary and next-logit causal effects: direction
  accuracy at least `0.80`, median cosine at least `0.90`, and median
  relative effect error at most `0.20`, where relative error is
  `||predicted effect - observed effect||₂ / ||observed effect||₂` for each
  resolvable effect.
- At the replaced block boundary: median activation cosine at least `0.90`
  and median normalized activation error at most `0.20`.
- On unperturbed challenge continuations: cross-entropy increase at most
  `0.05` nats/token and normalized next-logit error at most `0.05`.
- The treatment's median next-logit effect cosine exceeds the random,
  wrong-module, and no-op controls by at least `0.05` in each of the three
  seeds. This is a descriptive paired gate, not an inferential test.
- The teacher-activation oracle control restores downstream logits to maximum
  absolute error at most `1e-5`. This checks the evaluation path; it is not a
  learned-rescue result.

Report `behavior_and_mechanism_diverged` as true if any seed/stratum crosses a
behavior threshold (mean continuation cross-entropy increase `>0.05` or median
normalized next-logit error `>0.05`) or a mechanism threshold (median
activation error or either causal relative effect error `>0.20`). Otherwise
report false. Preparation and prediction phases report
`NOT_YET_EVALUATED`.

The overall scientific result is `PASS` only if every treatment gate passes in
all three seeds and all three strata, all control gates pass, and the oracle
check passes. Measured threshold misses are `FAIL`; missing coverage, digest
mismatch, or resource termination is `INCONCLUSIVE`. Three seeds on one
checkpoint are descriptive and are not treated as independent model or
architecture replications.

## Execution, custody, and stop rules

Use local CPU execution, one Torch thread, deterministic algorithms, Python
3.12.13, and the exact package pins in `requirements-v16.lock.txt`. Limit each
primary or replica run to four hours, 16 GiB peak RSS, 200 million checkpoint
parameters, and 800 MiB checkpoint weights. Stop on target-digest mismatch,
fixture overlap, non-finite candidate values, incomplete training, or a
resource cap. Preserve every partial receipt; do not reuse a stopped custody
root. Record the repository HEAD, dirty-worktree count, complete installed
package inventory and digest, checkpoint identity, and frozen packet digest in
the runtime and result receipts.

Archive the protocol before fixture generation. Keep prompts, token IDs,
captures, candidate tensors, intervention vectors, predictions, and
row-level metrics in external owner custody. The repository receives only
protocol code, aggregate-safe results, and digests. No provider, cloud, GMan,
remote worker, paid execution, real-user data, external review submission,
public write, or production integration is authorized by this packet.
Monetary cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`.

The separate clean-runtime replay may establish `REPRODUCED_LOCAL` for the
exact packet on the same owner and host. It cannot establish independent
science. A blind external replication and authorized reviewer remain the next
authority gate for independent scientific acceptance.
