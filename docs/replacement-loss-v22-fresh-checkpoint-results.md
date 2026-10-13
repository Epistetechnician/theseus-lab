# Replacement-loss V22 results: stopped before execution

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v22-fresh-checkpoint-replication-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Terminal status

`NO_ELIGIBLE_FRESH_TARGET`. V22's protocol and source packet were frozen
before the local model-asset screen. The screen found no eligible fresh local
checkpoint set across OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. Candidate
preparation, challenge generation, evaluation, and replication did not run.
This is a target-availability stop, not a scientific `FAIL` or an experiment
result.

## Frozen identity and assurance

- Protocol: `theseus-replacement-loss-v22-fresh-checkpoint-replication`
- Packet SHA-256: `924be878356660c79ffc3ec4b0aa32f34b3a32926f381cc10a184db459e6179d`
- Freeze manifest SHA-256: `8bfdcc32bf36ea6923f5fa769649dcda2abe0f3428170d0d54ed9f0f6091da9f`
- Upstream V8 method packet SHA-256:
  `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`
- Assurance: `INTERNAL_PASS` for the frozen packet, pinned V7/V8 integrity,
  and local target-availability preflight only. No experimental outcome exists.
- Reviewer: `NOT_SUBMITTED`
- Reproducibility: `NOT_RUN`

## Availability evidence

The standard local Hugging Face cache had 24 model repositories. The frozen
family screen found no OPT or GPT-2 config. Its only sub-500M Llama-family
checkpoint was already present in the V3–V21 prior-use ledger. The other
compatible Llama-family entry was an instruct, 4-bit checkpoint whose local
config dimensions exceed the 500M-parameter cap. Model assets under the V7–V9
experiment custody roots were prior-protocol assets and excluded.

The V3–V21 exclusion ledger contained 48 model identities from 79 source
manifests; its digest is
`f7f2d91402d96a068b5903aec40d8e70a2b47cc99c6af096de224045843b9a62`. The
local snapshot-config inventory digest is
`f261e03c2abfb0ff418acafe121abc636ba611395e18c5e424ce9e4146c6727b7`.
The detailed model metadata and screening receipt remain in external owner
custody at `/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v22`.

## Custody, claim ceiling, and next gate

No V22 weights, prompts, captures, candidates, predictions, outcomes, or
row-level diagnostics were created or copied. Cost was `$0`; energy was not
measured. The claim ceiling remains
`FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES`. This does not
extend the additive-module theorem or establish mechanistic understanding,
all-architecture validity, independent science, consciousness, alignment, or
production readiness.

The next blocked step is availability of eligible fresh checkpoints. V22
prohibits downloads and remote execution; changing acquisition policy requires
a new frozen packet. No external-review or deployment gate was reached.

External terminal-report SHA-256:
`e7a0a669f03a7dd64103cadd22653a401be6a234b93cf6f328663cccafdfa2d2`
