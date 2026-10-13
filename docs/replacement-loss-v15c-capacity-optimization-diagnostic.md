# Replacement-loss V15c: capacity versus optimization diagnostic

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v15c-capacity-optimization-diagnostic-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol ID: `theseus-replacement-loss-v15c-capacity-optimization-diagnostic`.

## Continuation boundary

V15 stopped before fixture generation because the runner referenced the
`Checkpoint` type from the utility module. V15b corrected that first binding,
loaded the target model, and created a visible-only synthetic fixture before
stopping on the next missing runner helper, `_batches`. Both attempts and
their receipts remain immutable. V15c binds every V15 runner reference to the
owning utility or runner module, verifies those bindings before protocol
commands, and freezes both prior attempts and receipts in a new packet.

V15c gets a new protocol namespace and fresh deterministic fixtures. It does
not reuse V15b's visible fixture. The two prior roots and their artifacts
remain in external owner custody. Any V15c execution failure leaves that root
immutable and requires a new named continuation.

## Question and claim ceiling

V14 showed a relative held-out causal-effect advantage for GELU over the
matched affine candidate, but neither candidate passed absolute retention.
Its activation-restoration control inserted teacher activations; it did not
test a learned rescue. V15c asks only whether increasing replacement rank,
optimizer updates, or both reduce an adaptation objective on held-back
synthetic validation examples.

The maximum claim is
`LOCAL_ADAPTATION_DIAGNOSTIC_OF_RANK_AND_OPTIMIZATION_FOR_ONE_REUSED_OPT_CHECKPOINT`.
This is not replacement-retention evidence, architecture generalization,
held-out causal prediction, or proof of Theseus continuity.

## Frozen design

- Reuse the exact OPT/ReLU checkpoint and layers 0 and 7 already bound by
  V14. This target is outcome-aware from earlier work; no fresh-checkpoint or
  architecture-general claim is made.
- After packet freeze, generate one visible synthetic fixture and three
  adaptation fixtures under the V15c namespace. Each adaptation fixture has
  96 examples, with 32 in each ordinary, compositional, and adversarial
  stratum.
- Before model capture, deterministically split each stratum into 24 fit and
  8 validation examples. The six seed/layer cells are descriptive clusters,
  not IID observations.
- Fit `Linear(3072, rank) -> GELU -> Linear(rank, 768)` replacements at ranks
  512 and 768. Each rank follows one trajectory through 800 AdamW updates;
  measure initialization, update 200, and update 800.
- Use batch size 8, learning rate `3e-4`, weight decay `1e-4`, gradient
  clipping at 1.0, and the sum of normalized activation MSE and normalized
  next-logit intervention-effect MSE with unit weights.
- Validation rows never enter gradient updates. V15c creates no held-out
  challenge fixture and evaluates no continuation behavior, learned
  activation rescue, or replacement retention.
- Report optimization, rank-capacity, and overfit signals using a 10%
  relative change threshold and at least 5 positive cells out of 6. These
  thresholds are descriptive, not inferential.

## Execution and custody

Execution is local CPU-only, one Torch thread, deterministic algorithms, a
four-hour wall-clock cap, a 16-GiB RSS cap, no more than 200 million base
model parameters, and no more than 800 MiB of checkpoint weights. No
provider, paid, cloud, remote-worker, GMan, real-user, or public-write action
is in scope. Cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`.

Archive the frozen packet before generating fixtures. Keep synthetic
prompts, captures, split IDs, candidate tensors, and per-cell metric rows in
external owner custody. Primary and replay use distinct pinned Python
environments on the same owner and host. Exact replay can establish
`REPRODUCED_LOCAL`, not independent scientific acceptance.

## Next gate

A clear, reproducible diagnostic signal can justify a new packet for
independently learned activation rescue and fresh held-out causal and behavior
checks. Independent science still needs a blind replicator and an authorized
reviewer. Alignment, consciousness, and production remain separate gates.
