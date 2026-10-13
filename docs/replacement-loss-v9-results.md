# Replacement-loss v9 results: causal replacement ladder

State slice: `theseus-lab-sandbox-v1`

Assurance overlay: `global-experiment-assurance-lifecycle-v1`

Protocol: `replacement-loss-v9`

## Verdict

**Protocol outcome: `FAIL`. Final assurance: `REPRODUCED_LOCAL`. Reviewer status: `NOT_SUBMITTED`.**

The frozen protocol required every ridge replacement stage to pass its held-out
retention gates, as well as a comparative causal-effect prediction advantage
over both baselines. The comparative gate passed, but ridge retention passed
only 21 of 27 stages. Six stages missed the predeclared next-logit
causal-effect cosine threshold, so the conjunctive protocol outcome is `FAIL`.
The clean local replication reproduced that same `FAIL` result. `REPRODUCED_LOCAL`
describes reproducibility; it does not turn the scientific verdict into a pass.

## What the result supports

This is evidence for a narrow, local comparative result: on these three fresh
decoder-only checkpoints, the ridge replacement's predicted held-out
next-logit causal effects scored better than the CKA and logit-distillation
baselines under the frozen paired comparison. In both comparisons, the lower
95% bootstrap bound across the nine family-by-seed trajectories exceeded the
predeclared `0.05` minimum.

It is not a successful demonstration of three-stage causal retention. The
retention failures were concentrated in OPT/ReLU: layers 0 and 7 failed across
all three seeds. The failed per-stratum cells were ordinary prompts at both
layers for every seed, plus adversarial prompts at layer 0 for every seed.
Thus the next-logit effect gate passed 72/81 ridge strata; the activation,
behavior, and module-boundary effect gates each passed 81/81. Only 21/27 full
ridge stages passed all their gates.

The v9 CKA comparator is a module-level NoT-style alignment baseline, not a
full reproduction of *Network of Theseus*. This result therefore does not
establish novelty priority against the complete published method. It also does
not test encoder-decoder, mixture-of-experts, recurrent, or state-space
architectures, and does not extend the additive-component identity into a
general theorem about AI systems.

Prior-art boundary: [*Network of Theseus*](https://arxiv.org/abs/2512.04198)
already progressively replaces components and aligns target representations
with guide activations using CKA or D-MNN; it also studies GPT-2-to-RNN
conversion and progressive distillation. V9's candidate distinction is its
pre-bound prediction of held-out causal intervention effects, but this packet
compares against a module-level CKA baseline rather than reproducing the full
NoT procedure. That distinction remains a research hypothesis, not a novelty
finding.

## Frozen design and targets

The packet was frozen before selecting fresh checkpoints. It evaluated three
replacement stages for each of three model families and three synthetic
fixture seeds: nine trajectories, 27 ridge stages, 81 method-stage records
across ridge/CKA/logit distillation, and 81 retained control-stage records.
Held-out checks covered causal intervention effects, behavior, activation
reconstruction/rescue, compositional prompts, adversarial prompts, and
wrong-module, random, and no-op controls. No parameter-permutation or symmetry
sweep was run.

| Family | Pinned checkpoint | Architecture tested |
| --- | --- | --- |
| OPT/ReLU | `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63` | OPT MLP |
| GPT-2/GELU | `distilbert/distilgpt2@2290a62682d06624634c1f46a6ad5be0f47f38aa` | GPT-2 MLP |
| Llama/SiLU-gated | `StentorLabs/Stentor-30M@6a69098f55961cc0df3a0751e44082616e4afdfa` | Gated MLP |

There was one checkpoint per family. The three fixture seeds are not three
independently sourced datasets or checkpoints.

## Aggregate outcomes

| Gate | Result |
| --- | ---: |
| Ridge stages passing every retention gate | 21/27 |
| Per-stratum next-logit causal-effect gates | 72/81 |
| Per-stratum module-boundary causal-effect gates | 81/81 |
| Per-stratum behavior gates | 81/81 |
| Per-stratum activation reconstruction gates | 81/81 |

The paired effect is ridge minus comparator in mean per-trajectory median
cosine for next-logit causal effects. Bootstrap units were the nine
family-by-seed trajectories, as frozen.

| Comparator | Mean paired difference | 95% bootstrap interval | Predeclared lower-bound threshold | Gate |
| --- | ---: | ---: | ---: | --- |
| CKA | 0.910427 | [0.735126, 1.073952] | > 0.05 | Pass |
| Logit distillation | 0.951197 | [0.820481, 1.080619] | > 0.05 | Pass |

## Execution, custody, and reproduction

- Primary run: pinned CPU runtime on macOS ARM64, Python 3.12.13, PyTorch
  2.6.0, Transformers 4.57.3, one CPU thread. Primary protocol time was
  3,076.27 seconds; peak resident memory was 4,686,561,280 bytes.
- Clean local replication: a separate clean Python environment on the same
  host and under the same owner. All 12 frozen replication checks passed,
  including candidate states, checkpoint identities, fixtures, prediction
  arrays, outcome arrays, summaries, controls, and dependency manifest.
  Dependency manifest SHA-256:
  `bea06dc458f522768a207ea383d8220ab3b9e0eda8c9de72c5f6468cafd3d94d`.
- No GMan node or paid compute was used. Monetary cost and energy were not
  measured.
- Model assets, prompts, token IDs, captures, candidates, predictions, and
  row-level outcomes remain in external owner custody (512 files,
  6,497,373,770 bytes). They were not copied into this repository.
- The primary runner's execution subrecord is `INTERNAL_PASS`; the experiment's
  final reproducibility assurance is `REPRODUCED_LOCAL`. This was a same-owner
  local reproduction, not independent scientific review.
- No v9 packet was submitted to an external reviewer. `NOT_SUBMITTED` is the
  reviewer status; external acceptance has not occurred.

## Frozen identities

Repository base revision at execution: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
The experiment sources also included uncommitted files; their frozen worktree
source packet SHA-256 is
`30ccf63209aa70ed8821b2614214114b010bd0c95f61bda772fa7e5ffbe548f4`.

| Artifact | SHA-256 |
| --- | --- |
| v9 protocol packet | `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959` |
| Checkpoint manifest | `b52bd7b94eb1e14c0453ba9318fcec2b39f0a210ca4c35f88b070bd57cd8e277` |
| Candidate freeze | `309ed7dfde5a4277daa57ed21e77bc8d2e05307268789a79819e31b19f3f8443` |
| Challenge manifest | `5de1335c01d3fd8bf0729977842067a002c83338cf58c223faf59fa0df7f5600` |
| Prediction manifest | `5c7602bde1beb52d477809accb627eef9b396db32b7726395b12bf9e8bef911c` |
| Primary outcome report, canonical digest | `65a3c17bcd544bf6d03a9cd76f9c2ccd1d60c6b1fbdf09fd827b961c56f4551b` |
| Primary outcome report, file-byte digest after replication binding | `12e0befa4c329d3a6a18b14d19144ab6296e5ac5b82f0cb2326eaeff7c776431` |
| Replication manifest, file-byte digest | `4f759a1396147bf723b9cec5c8f97716128cf02673808a7212da8de41634a61c` |
| v3 continuation packet | `a4d3c491f36bca054ec65b7a061e4d36deaabe2e2fb0a64669d745201d1a2be1` |
| v4 result-binding packet | `edf615792c8e193f74e8b840faeb0e9c2c021eba938c4b20a0d31db1fce55753` |
| v4 binding receipt, canonical digest | `9c89ac6db660dc7948002279278db7059f3b8fc3d2abf7b15110cc32e439542e` |

The v4 binding receipt is byte-identical in the primary and replica custody
roots. Its internal digest and all nine copied context-file digests verified;
the finalizer rechecked the bound reports and replication manifest after
writing the receipt. The two distinct outcome-report digests above are
intentional: one is the report's canonical digest, the other hashes its final
serialized file bytes.

## Claim ceiling and nonclaims

Frozen claim ceiling:
`LOCAL_CAUSAL_EFFECT_PREDICTION_ADVANTAGE_OF_AFFINE_REPLACEMENT_OVER_CKA_AND_LOGIT_DISTILLATION_THROUGH_THREE_REPLACEMENTS`.

Because the overall frozen outcome is `FAIL`, this ceiling is not a claim that
the full three-stage retention hypothesis succeeded. The observed comparative
gate is limited to the exact checkpoints, fixtures, implementations, and
baselines above. This result does not establish novelty priority, all-architecture
validity, whole-model identity, general mechanistic understanding, independent
science, consciousness, alignment, human equivalence, or production readiness.
The human-calibration packet remains
`BLINDED_CALIBRATION_PACKET_READY`.

## What next

The next highest-value scientific step is a blind external replication of this
exact frozen packet, preserving its `FAIL` verdict if reproduced. That would
test whether the observed comparative advantage and OPT/ReLU retention failures
hold under an independently operated run; it would not convert this packet into
a pass. The v9 reviewer status is still `NOT_SUBMITTED`; current live setup
checks and the remaining external authority and transfer blockers are recorded
in [replacement-loss-v9-external-review-readiness.md](replacement-loss-v9-external-review-readiness.md).

After that, a new protocol can test a preregistered explanation for the OPT
early-layer misses using fresh held-out fixtures and an independently selected
checkpoint. Keep v9's thresholds and outcomes immutable; any changed method or
threshold requires a new packet. Separate gates remain for human-calibrated
alignment, theory-driven consciousness research, and production authorization.
