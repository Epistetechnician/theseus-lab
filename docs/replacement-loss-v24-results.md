# Replacement-loss V24 audit: inconclusive checkpoint screen

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v24-public-checkpoint-acquisition-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Audit conclusion

V24's terminal receipt says `NO_ELIGIBLE_FRESH_TARGET`, but an audit of the
frozen query and screen records found that every returned model entry lacked
an immutable commit SHA. The pinned `HfApi.list_models` call did not request
full metadata, so the V23 candidate checker rejected every entry as
`missing_immutable_revision` before it could establish candidate eligibility.
Therefore the V24 availability conclusion is invalid and is reclassified here
as `INCONCLUSIVE_SETUP_OR_EXECUTION`. The original terminal receipt is
preserved unchanged.

The completed bounded query set contained 1,000 OPT results, 1,000 GPT-2
results, and 3,964 unique Llama-family results across four searches, for 6,000
query rows and 5,964 family-scoped unique identities (5,946 unique IDs across
all three families). Every entry in the
screen projections was rejected for the missing revision field. No checkpoint
identity was selected; no model weights, prompts, candidate modules,
predictions, or outcomes were produced; primary
evaluation and reproduction did not run. No safetensors files or snapshot
directories were present in the V24 Hub cache. Direct monetary cost was `$0`.

## Frozen identity and evidence

- Protocol: `theseus-replacement-loss-v24-public-checkpoint-acquisition`
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`
- V24 packet SHA-256: `1d1b99baa7dbe41f62a989a3653dc18b353f9d7fee5f8cbd6ed810796b162197`
- Freeze manifest SHA-256: `caa82b873077c5789133d14c2ddf668cb2eaacf1e566a792822dda9f8444350c`
- Upstream V23 packet SHA-256: `7600d653f906db5411a6ab3121e078f8134e311e8fe0ecb37ef05552bdbe890e`
- Upstream V22 packet SHA-256: `924be878356660c79ffc3ec4b0aa32f34b3a32926f381cc10a184db459e6179d`
- Upstream V8 method packet SHA-256: `b6bc9563f242bbf20b9acf0decf40798481538930b17be1342cfca47f8158fc2`
- Hub screen SHA-256: `8764c4487c229140ebbbb3b77b4d8a4f0d8bee40bac8355ed607b48b6c31d68c`
- Query-progress receipt SHA-256: `12eae1140782a0fde258520e89759ab27493b5a8c9560214460c5b2dde3f6041`
- Original terminal-report file SHA-256: `b0c69983e023b7452c27e4107dcfc48951bfc5821eb882ffb08442ad249701e8`
- Original terminal self-digest: `50bea0dba151f35e9d0f991d44297f22354c11ea1cc16953a1572c6d850be82a`

The Hub screen and full public model-ID projections remain in external owner
custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v24`.
The V24 source packet and terminal receipt are unchanged. Assurance is
`INTERNAL_PASS` for packet integrity and query completion only; the target
availability determination is `INCONCLUSIVE`. Reviewer status is
`NOT_SUBMITTED`, reproducibility is `NOT_RUN`, and behavior/mechanism
divergence is `NOT_ASSESSED_NO_MODEL_RUN`.

## Claim ceiling and next gate

No scientific result follows. The V24 claim ceiling remains fresh-checkpoint
replication within OPT/ReLU, GPT-2/GELU, and Llama/SiLU-gated. The next step is
a new frozen acquisition packet that requests full model metadata and excludes
every identity already returned by V24 before selecting any checkpoint.
Unseen-architecture validity, mechanistic understanding, independent
science, consciousness, alignment, and production readiness remain
unestablished.
