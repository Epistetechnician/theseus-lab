# Replacement/loss V29 result

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v29-nonlinear-synthetic-qualification-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Terminal status

**Outcome: `INCONCLUSIVE`. Assurance: `INCONCLUSIVE`. Reproduction: not
attempted. Reviewer: `NOT_SUBMITTED`.**

The frozen packet stopped in candidate preparation on the first case. A
compositional/adversarial fixture batch reached the float32 source module as
float64, and PyTorch raised:

`RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float`

No candidate was trained; no challenge was generated; no prediction or
held-out outcome was produced. The partial source arrays and exact failure
receipt remain in external custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v29`.
The failure receipt SHA-256 is
`40c64c471ab051795049c72b7b346a0dd6d5098cd351cc2d60f15d2d0cb4d007`.

The frozen source was not changed or retried. The packet SHA-256 is
`33016b8c3f97066222077c06e4f8fe6a97349380df500935cb1fcb7d07d99b8d`. A new
recovery packet is required to fix dtype normalization before any further
synthetic generation. No scientific claim is supported by this incomplete
run.

The claim ceiling remains
`SYNTHETIC_NONLINEAR_REPLACEMENT_METHODS_QUALIFICATION_ONLY`. Model
checkpoints, paid compute, remote workers, prompts, and human data were not
used. The next scientific gate remains a separately frozen real-checkpoint
packet for independently trained replacement and blind causal prediction.
