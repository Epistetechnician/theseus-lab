# V33R1 initial adaptation loss-gradient results

Closeout date: 2026-10-03. State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v33r1-loss-scale-audit-v1`,
`global-experiment-assurance-lifecycle-v1`. Revision:
`202f459d6d89df9a038d78d58e0af825d287fcd8` plus the frozen uncommitted
diagnostic packet.

## Literal result and scope

**Diagnostic completed; assurance INTERNAL_PASS; separate reproduction
NOT_RUN_SEPARATELY; reviewer NOT_SUBMITTED.** This is a read-only audit of
six initial gradient readouts across three existing guide cases and two
replacement geometries. It made no optimizer updates or heldout queries
and changed no model, scientific score, threshold, selection, or outcome.
Its claim ceiling is `INITIAL_ADAPTATION_GRADIENT_SCALE_ONLY`.

The [protocol](replacement-loss-v33r1-loss-audit.md) and code were frozen
before gradient readout. Each readout reconstructed the original stage-one
initialization and exact first adaptation/donor/mask minibatch. Paired
geometries have matching initialization and input digests. The anchor is
the original output loss plus activation loss; gradient norms use only
the freshly initialized replacement block parameters.

| Guide case | Replacement | Global effect / anchor gradient | Row effect / anchor gradient | Row / global effect gradient |
|---|---|---:|---:|---:|
| relu-33101 | gelu | 0.411 | 1009.88 | 2455.91 |
| relu-33101 | relu | 0.396 | 790.32 | 1994.07 |
| relu-33102 | gelu | 0.608 | 132.52 | 217.94 |
| relu-33102 | relu | 0.523 | 167.18 | 319.67 |
| relu-33103 | gelu | 0.434 | 31.95 | 73.65 |
| relu-33103 | relu | 0.391 | 46.23 | 118.13 |

Per-example effect normalization yields gradients **31.95–1009.88 times**
the behavior-plus-activation anchor gradient. Global normalization yields
**0.391–0.608 times** that gradient. Row/global effect-gradient ratios range
73.65–2455.91. All three first minibatches have zero reference-effect floor
fraction at squared norm ≤1e-6; other minibatches were not audited.
The [aggregate receipt](replacement-loss-v33r1-loss-audit-results.json)
retains exact losses, norms, ratios, effect-norm quantiles, and identity hashes.

## Interpretation and next proof obligation

The fixed unit coefficient did not produce comparable initial gradient
influence for the two normalizers. This establishes an initial scale
mismatch for these inputs and initializations. It supports investigating
a causal objective with bounded or balanced gradient influence in a new,
fresh-data development packet. It does not establish that scale mismatch
caused the complete training regression, prove that balancing will pass,
or qualify a replacement. No new training study was run.

[V33R1 development](replacement-loss-v33r1-results.md) remains
`NO_QUALIFIED_DEVELOPMENT_CANDIDATE`, now `REPRODUCED_LOCAL` after both
drives were reconnected and the saved artifacts were verified. A future
candidate must still meet behavior, small-effect-tail, activation-rescue,
intervention-prediction, and control gates at every ladder stage.
Adaptation-only coordinate transport must also be checked on causal
interventions; activation fit and numerical inversion alone were insufficient.
No confirmation or comparator stage opened.

## Identity, resources, and custody

| Binding | SHA-256 |
|---|---|
| Diagnostic packet | `abafc92e50de12e7ce12684e948b30f3f16bbd132f25be90a8cc6763256ece5d` |
| Freeze file | `707c13f37650872c925ebb8bb9160bdbe53242280091ce30647fc8e1f1723fd3` |
| External diagnostic result | `d8b42f7b45a465c15869fa63774dab0bfdaa9680978cba35f409ae1789af91da` |
| Parent development result | `0fec829d30ca4ad1e7a68917f18e370ea78b4b6ca1047fcf52dc59f53058197e` |
| Closeout validation receipt | `eb8cf13ed0cfe274da977cf66c58882fab9c954072dec1b8733f971ebaa91fef` |

Locked runtime: Python 3.12.13, Torch 2.6.0, NumPy 2.4.6, one deterministic
CPU thread, float32 computation, macOS 26.6.1 arm64. Frozen-phase timing:
3.230 seconds; peak RSS: 213,729,280 bytes; final retained diagnostic artifacts:
17,804 bytes. Monetary spend $0; energy not measured. Whole-session and
environment-installation time are not included in the diagnostic timing.

The raw adaptation inputs remain in their original development custody.
The diagnostic source archive and aggregate runtime/result receipt are under
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v33r1/loss-scale-audit`.
Its freeze and source archive were verified; all ratios and paired identities
passed direct consistency checks. Both parent artifact manifests and all
16 parent reconciliation checks passed without rerunning terminal experiments.

This small local diagnostic supplies no pretrained-model, universal-identity,
novelty-priority, consciousness, alignment, independent-science, or production
evidence. External scientific acceptance, human calibration, and deployment
authorization remain separate gates.

