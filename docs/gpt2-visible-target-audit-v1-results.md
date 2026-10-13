# Visible all-layer GPT-2 target audit: no critical pair across depth halves

2026-10-06; revision `202f459d6d89df9a038d78d58e0af825d287fcd8` plus frozen
source. Slices `theseus-lab-sandbox-v1`, `theseus-gpt2-visible-target-audit-v1`,
`global-experiment-assurance-lifecycle-v1`.

**`NO_VISIBLE_CRITICAL_PAIR`, `REPRODUCED_LOCAL` for this diagnostic. Reviewer
`NOT_SUBMITTED`.** Primary and replica independently ablated all 12 MLPs on
the exact already visible 64 V37R1 rows. All nine reconciliation fields and
canonical baseline/removal arrays matched. No new data, optimizer step,
replacement or held-out result was opened. Two hermetic selection checks
passed before the source freeze.

Only layers 0 and 2 met the existing .10 nRMSE and .20 top-1-change gates.
Both lie in the first half. Layer 2 removal had nRMSE .113122 and top-1 change
96.875%. Every second-half MLP had nRMSE below .10; the largest was .076124
at layer 11. The high-energy last layer was not a qualifying target, but
changing the selector alone cannot supply a pair spanning depth halves on
this visible corpus. The frozen wider-scope study must remain stopped.

Packet `266cabc17fec751169eceacbfc5b0814cbae9022833726e84204f8d67a7fa5b8`.
Primary result `3fe55129f0b3c96f3e1283996bca54b2dacd51c74ed9d69e2ce5abdf919d5054`;
replica `b78424253ab2045920ade8571bf54b0efe488309b0415ea05236813d567f0e03`.
Canonical arrays `c96e6fbeb2b69db89eb35540ef39a3a1aed3e01528686088b7e37a9375e2aca3`.
Study wall times 19.58/19.72 seconds. Raw full-vocabulary arrays and their
file hashes/reconciliation remain in the external
`theseus-lab-sandbox-v1-gpt2-visible-target-audit-v1` custody root.
Paid $0; energy/amortization unmeasured. No provider call, remote worker,
review submission or publication.

Claim ceiling `REUSED_VISIBLE_TARGET_DIAGNOSIS_ONLY`. This is reused visible
development evidence, not unseen causal prediction or replacement retention.
Same host/owner and model-adapter limitations remain. No independent authority
or broad Theseus claim follows. A separately frozen early-block-only pilot
may use layers 0 and 2, with fresh adaptation/holdouts and explicit reuse;
wider depth/family coverage and independent science remain separate gates.
[Protocol](gpt2-visible-target-audit-v1.md).
