# Replacement-loss V15b: capacity versus optimization diagnostic

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15b-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v15b-capacity-optimization-diagnostic`.

## Continuation boundary

V15 was frozen, but its primary attempt stopped before fixture generation or
model loading because the runner resolved `Checkpoint` and the local model
loader from the utility module instead of `replacement_loss_v3_runner`. The
failed V15 packet and stop receipt remain immutable. V15b is a new packet and
state slice that binds the V15 source, V15 freeze record, and V15 execution
receipt. Its small adapter binds the two model-loading functions from their
owning runner module before any V15b command executes.

V15b freezes this adapter, protocol, thresholds, fixture rules, and exact
source files before generating any V15b prompts or training any candidates.
The protocol ID creates a distinct deterministic fixture namespace. The
previous V15 primary root is never reused. Any V15b execution failure also
leaves its root immutable and requires another named continuation.

## Question and claim ceiling

V14 showed a relative held-out causal-effect advantage for GELU over the
matched affine candidate, but neither candidate passed absolute retention.
Its activation-restoration control inserted teacher activations; it did not
test a learned rescue. V15b asks only whether increasing replacement rank,
optimizer updates, or both reduce the adaptation objective on held-back
synthetic validation examples.

The maximum claim is
`LOCAL_ADAPTATION_DIAGNOSTIC_OF_RANK_AND_OPTIMIZATION_FOR_ONE_REUSED_OPT_CHECKPOINT`.
This is not a replacement-retention result, architecture-general result,
causal challenge result, or proof of Theseus continuity.

## Frozen design

- Reuse the exact OPT/ReLU checkpoint and layers 0 and 7 already bound by
  V14. This target is outcome-aware from earlier work; V15b does not claim
  checkpoint or architecture generalization.
- After freezing, generate one visible synthetic fixture and three
  adaptation fixtures under the V15b protocol namespace. Each adaptation
  fixture has 96 examples: 32 each in ordinary, compositional, and adversarial
  strata.
- Before model capture, deterministically split each stratum into 24 fit and
  8 validation examples. The six seed/layer cells are descriptive clusters,
  not IID observations.
- Fit `Linear(3072, rank) -> GELU -> Linear(rank, 768)` replacements at ranks
  512 and 768. Train each rank as one trajectory through 800 AdamW updates;
  record initialization, update 200, and update 800.
- Use a fixed batch size of 8, learning rate `3e-4`, weight decay `1e-4`,
  gradient clipping at 1.0, and the sum of normalized activation MSE and
  normalized next-logit intervention-effect MSE with unit weights.
- The validation examples are never used in gradient updates. No challenge
  fixture, continuation behavior evaluation, activation rescue, or behavior
  retention check is run in this packet.
- Classify optimization, rank-capacity, and overfit signals using a 10%
  relative reduction threshold and at least 5 positive cells out of 6.
  Report every signal. These cutoffs are descriptive and not inferential.

## Execution and custody

Execution is local CPU-only, one Torch thread, deterministic algorithms, a
four-hour wall-clock cap, a 16-GiB RSS cap, at most 200 million base-model
parameters, and at most 800 MiB of checkpoint weights. No provider, paid,
cloud, remote-worker, GMan, real-user, or public-write activity is in scope.
Cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`.

The packet is archived before fixture generation. Synthetic prompts, model
captures, split identifiers, candidate weights, and per-cell metrics remain in
external owner custody. Primary and replay use distinct pinned Python
environments on the same owner and host. Exact replay can reach
`REPRODUCED_LOCAL`, not independent scientific acceptance.

## Next gate

Only a clear, reproducible diagnostic signal warrants a new packet for
learned, non-oracle activation rescue and fresh held-out causal and behavior
evaluation. Independent science still requires a blind replicator and an
authorized reviewer. Alignment, consciousness, and production remain
separate gates.
