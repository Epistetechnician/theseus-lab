# Replacement-loss V23 results: setup stopped before target discovery

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v23-public-checkpoint-acquisition-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Terminal status

`INCONCLUSIVE_SETUP_OR_EXECUTION`. The V23 packet and upstream V22/V8 sources
verified, but the pinned `huggingface_hub==0.36.2` client rejected the frozen
`num_parameters` argument to `HfApi.list_models`. The call failed in local
argument validation before any model record was returned. No checkpoint
identity was selected and no weights were downloaded. No experiment ran.

This frozen packet remains immutable. V24 carries forward the same
post-freeze acquisition goal with the 500M parameter cutoff applied from exact
safetensors metadata after bounded Hub search results are fetched.

## Identity, assurance, and custody

- V23 packet SHA-256: `7600d653f906db5411a6ab3121e078f8134e311e8fe0ecb37ef05552bdbe890e`
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`
- Assurance: `INCONCLUSIVE` for acquisition setup; V23/V22/V8 packet integrity
  checks passed.
- Reviewer: `NOT_SUBMITTED`
- Reproducibility: `NOT_RUN`
- Target IDs selected: none
- Weights, prompts, captures, candidate modules, predictions, and outcomes:
  none created for V23
- Direct monetary cost: `$0`

The failure receipt is in external owner custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v23`.
Its SHA-256 is
`0bff6a0367ca5b39b1decc3bdcb229a38b7bd551891d80901f731c673b4123d9`.

## Claim ceiling

No scientific result follows. The claim ceiling remains
`FRESH_CHECKPOINT_REPLICATION_WITHIN_OPT_GPT2_LLAMA_FAMILIES`; all-architecture
validity, mechanistic understanding, independent science, consciousness,
alignment, and production readiness remain unestablished.
