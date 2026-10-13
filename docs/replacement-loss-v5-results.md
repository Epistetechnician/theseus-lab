# Replacement-loss-v5 balanced adversarial adaptation results

State slice: `theseus-lab-sandbox-v1`.

Assurance: `REPRODUCED_LOCAL` for the exact frozen packet and aggregate
outputs in a distinct clean local runtime. The outcome remains `FAIL` under
the frozen all-seed gates.

## Frozen identities and custody

The digest-bound protocol packet was frozen and executed as reported below.
The frozen source file `docs/replacement-loss-v5.md` retains its earlier
“preregistration draft” status wording; that wording is stale. It is recorded
here as a post-freeze documentation erratum rather than changing a packet
member after execution.

- Protocol packet SHA-256: `f8b34296d492249521dec0f803df5fa61f4f0187694384b1e93b7db288bfe04a`.
- Candidate freeze digest: `cb3d85e35d41967cd9148cc2a48d3efcd9f6ef847790a658dfc042d009e451e7`.
- Challenge manifest digest: `7b9e99892fecec35b1a59d4e3c4ab837b45fe9c0710c6949fc89805aedc7da7d`.
- Prediction freeze digest: `47c971c9cb2bbb2d7b38e3252331e3a199ae8d8b8b097dc15b899a8575c8436a`.
- Primary report file SHA-256: `026a4baacd7031d90fc1033ca747d048f932b4d9f258a3a7c0190ea84586d4cc`.
- Primary aggregate metrics digest: `4617d5618a64b390ce1101559856e716b2eebb1946dec255d3204f753ad1d929`.
- Replica report file SHA-256: `8616df67bbf65f5773876cb4be51d0ff6f2dfd09bb7ed9b1326b54db450a191e`.
- Replication receipt SHA-256: `4d09570fc8dd0057871ba1156398dbeae2946f9c9fdc9caac8c299f252214891`.
- Reconciliation digest: `8ba57a502d501dc48abe7cea913cd36c697e937b311609e2dec729c939c99519`.
- Reconciliation file SHA-256: `ea8bb34e17b90bab33293e01021b05eeebcf2b57f7085073f6127ac0e3a60185`.
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Source worktree-status SHA-256: `d6969f76125026b60f76c228972aec75598064aba10247a9931c643001d81507`.
- Primary runtime inventory SHA-256: `8298a7f89688ba2a19c5034ae4863903cc248fac6341197b45592540e4215cbb`.

The model, prompts, captures, candidates, predictions, outcomes, and row-level
diagnostics remain under external owner custody at
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v5`.
Replica outputs are under
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v5-replicator`.
The held-out challenge data was generated from OS entropy only after the
candidate freeze. Its partitions were checked against v3 and v4 custodied
fixtures and excluded development fixtures.

## Findings

The balanced arm's maximum normalized next-logit error stayed at or below
`0.05` in every stratum for all three seeds. Its compositional activation
gates, adversarial activation gates, cross-entropy guards, and
ordinary/adversarial non-inferiority gates also passed in all three seeds.
The compositional median activation improvement over ordinary-only ranged
from `0.7059` to `0.7335`; every compositional challenge row improved.

The complete protocol still failed. The minimum required improvement in
adversarial maximum next-logit error over the composition-only arm was `0.02`
per seed. The observed improvements were `0.0163`, `0.0650`, and `0.0535`.
Seed `20261001` missed that locked effect-size gate, so the two passing seeds
do not change the overall `FAIL`. No thresholds were changed.

This result supports the narrow statement that the balanced arm met the
absolute held-out behavior ceiling on these three fresh synthetic fixtures.
It does not establish a repeatable minimum adversarial behavior effect across
seeds or a broader mechanistic claim.

## Reproduction and claim ceiling

The separate clean runtime reproduced candidate, adaptation-fixture,
prediction, outcome, and aggregate-metric digests. Both primary and replica
reports retain `FAIL`; the aggregate metrics digest matches exactly. Python,
NumPy, PyTorch, Transformers, Tokenizers, and Safetensors versions matched;
runtime prefixes were distinct. Monetary cost and energy were not measured.

The first replication attempt stopped after prediction because the archived
runner's path lacked Git metadata for its report source receipt. That partial
attempt is preserved under
`/Users/shaanp/.codex/custody/theseus-lab-sandbox-v1-replacement-loss-v5-replicator-failed-git-root`.
The successful retry used the byte-identical checked-out runner, verified
against the archived packet, in the clean second runtime. No frozen source or
primary report bytes changed.

Reviewer status is `EXTERNAL_REVIEW_UNAVAILABLE`; no external review task was
submitted. This same-owner second runtime is reproducibility evidence, not
independent acceptance. The next authority gate for an independent-science
claim is the configured external reviewer; a broader architecture claim needs
a separately frozen, architecture-held-out protocol.

Claim ceiling: `LOCAL_GPT2_BALANCED_ADVERSARIAL_ADAPTATION_EFFECT` for this
checkpoint, layer 3 projection, synthetic generator, and three seeds. This
does not establish fresh-checkpoint generalization, all-architecture
validity, mechanistic understanding, independent science, alignment,
consciousness, or production readiness. Monetary cost and energy were not
measured.
