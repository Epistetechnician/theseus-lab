# V39 first-MLP fit-factorial V3

2026-10-07. State slices `theseus-lab-sandbox-v1`,
`theseus-v39-first-mlp-fit-factorial-v3`, and
`global-experiment-assurance-lifecycle-v1`.

## Prior packet stops

V1 stopped during source verification before runtime checks, artifact creation,
fixture generation, or model execution. Its frozen source is retained.

V2 passed source and runtime preflight, then generated one adaptation and donor
partition. V39's fitting adapter represented each capture as 96 padded rows
with a final metadata mask feature. V2 pooled the 769-wide representation
without removing that metadata column and stopped at the first teacher target
forward. It performed zero optimizer updates and generated no challenge data.
Its terminal receipt and prompt/token identities are bound as V3 inputs.

## V3 capture correction and design

V3 strips the adapter mask only after checking its binary contiguous-prefix
contract and zero-padding rule. The resulting feature array must be exactly
768 wide. A new protocol namespace, adaptation entropy label, and artifact
root create fresh training and challenge streams. Every V2 prompt and token
identity is loaded as exclusion-only input; none can enter V3 training or
assessment.

The fixed target and 2×2 design are unchanged: one reused
`mrm8488/spanish-gpt2` checkpoint at revision
`fc396ce08c65ea334dcf541de7dc05cac89a29b2`, first MLP site 0, three candidate
seeds, hidden expansion 2×/4×, and 1,024/4,096 updates. V3 uses the saved V36
core fitter directly. The training objective, optimizer, held-out thresholds,
controls, resource caps, and closeout rules remain those in the V1 protocol.

The source freeze precedes all V3 fixture generation and fitting. All twelve
candidates and their traces must be frozen before challenge entropy is drawn.
The second runtime must rebuild matching candidates before reading that
entropy, then match candidate and control prediction digests before oracle
evaluation.

## Claim ceiling

`LOCAL_REUSED_GPT2_FIRST_MLP_FIT_FACTOR_DIAGNOSIS_V3_ONLY`. Evidence is limited
to one reused GPT-2 guide and one local MLP. It establishes no architecture-
general theorem, whole-model mechanistic identity, consciousness,
human-alignment equivalence, independent science, or production readiness.
Reviewer status is `NOT_SUBMITTED`; production remains a separate
authorization gate.
