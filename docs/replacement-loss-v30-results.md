# Replacement/loss V30 result

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v30-nonlinear-synthetic-recovery-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Verdict

**Outcome: `FAIL`. Assurance: `REPRODUCED_LOCAL`. Reviewer:
`NOT_SUBMITTED`.**

V30 completed all nine frozen synthetic cases. One passed every predeclared
gate; eight failed. Candidate direction was correct in all nine, and the
treatment beat wrong-module, random, zero/no-op, and affine controls in each
stratum. However, ReLU cases missed the causal normalized-error limit (about
0.25–0.28) and the maximum logit-error limit (about 0.15–0.19). Two of three
GELU cases missed only the maximum logit-error limit (about 0.067–0.072). All
three SiLU-gated cases missed the causal normalized-error limit (about
0.31–0.35) and maximum logit-error limit (about 0.48–0.53). Mean cross-entropy
delta and activation-rescue gates passed in all nine cases. Thresholds were
not changed.

The result suggests that a single GELU student predicts the direction of
local effects across these synthetic functions, but does not retain effect
magnitude and worst-case downstream logits across the ReLU and gated
families. This is a negative methods result, not evidence about trained
transformers.

## Frozen identity and reproduction

- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`
- V30 packet SHA-256: `a2e8860f30642be8a2c53386c64b87db7e17e6e1a7b3212e4a4274b72cd0cda3`
- Upstream V29 packet SHA-256: `33016b8c3f97066222077c06e4f8fe6a97349380df500935cb1fcb7d07d99b8d`
- V29 failure receipt SHA-256: `40c64c471ab051795049c72b7b346a0dd6d5098cd351cc2d60f15d2d0cb4d007`
- V30 primary report file SHA-256: `b5cf567cf63477dbd0555d32db1b7f3c32cf3808178a831a797aa98772b208f3`
- Reconciled replication receipt SHA-256: `b8c4dc5acea02bb87b29813ce85d03d7fba257f81275de302099b156bb87562e`

Same-owner reproduction matched candidate state arrays (126 arrays), all nine
prediction and outcome digests, challenge fixtures, dependency versions, and
the frozen packet in a second isolated Python 3.12.13 runtime with NumPy
2.4.6, PyTorch 2.6.0, and one CPU thread. Peak RSS was about 247 MiB in the
primary and 235 MiB in the reconciled replica (259,457,024 and 246,775,808
bytes). No paid compute or remote workers were used.

The frozen replication wrapper omitted `challenge-manifest.json`. Candidate
reconstruction completed before that error. A first direct evaluation then
ran with the replica runtime's 10-thread default and did not match prediction
digests; that attempt remains in external custody as inconclusive. The
reconciled run copied the exact primary manifest byte-for-byte, used the
frozen one-thread runtime contract, retrained candidates from adaptation
arrays, and matched all nine cases. The wrapper defect and both attempts are
preserved in external custody; no scientific threshold or source byte was
changed.

## Custody and claim boundary

All synthetic source weights, adaptation and challenge arrays, candidates,
predictions, and outcomes remain outside the repository under
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v30`
and its replication roots. No model checkpoints, user prompts, assessment
data, or human data were used.

The claim ceiling is
`SYNTHETIC_NONLINEAR_REPLACEMENT_METHODS_QUALIFICATION_ONLY`. This result does
not establish a trained-model mechanism, architecture-general validity,
independent science, alignment, consciousness, or production readiness. Real-
checkpoint replacement studies already exist in V3, V8, V9, and V28, so the
next gate is not a first real-checkpoint protocol. V30's failed cases remain
synthetic development evidence and must not be retuned into a pass. The
project-wide evidence state and next authority gate are summarized in the
[Theseus evidence ledger](theseus-evidence-ledger-v1.md).
