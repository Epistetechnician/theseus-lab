# Replacement-loss V9 GMan fresh-node continuation V4h results

**Verdict: `INCONCLUSIVE_TIMEOUT`.** The frozen V4h evaluator started after custody recovery and runtime preflight passed, but its one allowed run reached the 73-minute timeout (GMan command `cmd-4h5pn`, exit code `124`). It did not produce a completed reproduction receipt. This is not `REPRODUCED_LOCAL` and does not reproduce the target V9 `FAIL` result.

## Frozen identity and assurance

- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4h`, and `global-experiment-assurance-lifecycle-v1`.
- V4h packet SHA-256: `1ddbc1571c61c588bfa3d9115d53983123c42f8678c9ace3d50c8aba7cc1456c`.
- Target V9 packet SHA-256: `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Overall assurance: `INCONCLUSIVE`; reviewer status: `NOT_SUBMITTED`; reproducibility status: `INCONCLUSIVE_TIMEOUT`.
- GMan mission: [V4h frozen-runtime reproduction](https://autoresearch.sfcompute.com/missions/theseus-v9-linux-runtime-reproduction-20260928-v4h).

## What completed

The importer, digest-bound primary-report recovery, transport checks, and runtime preflight passed. Recovery preserved the discrepant cloned report (`dec06c5d336e7f0f161cb100de672248fbf6295eaceeff85b9e2358e8b1ddbcd`) and restored the archive-bound target report (`12e0befa4c329d3a6a18b14d19144ab6296e5ac5b82f0cb2326eaeff7c776431`). The frozen target report's outcome is `FAIL`.

The candidate preparation and candidate freeze were written in the replica custody. Challenge fixtures were then revealed to the replica after candidate freeze. The evaluator created 52 partial prediction files, totaling 1,201,283,613 bytes. Their aggregate file-inventory digest is `d5c0627b2948831a2ffe4ea05918bddd116072601c71aabf34604d0455ecae50`.

There is no replica `prediction-manifest.json`, replica `outcome-manifest.json`, replica `replication-manifest.json`, V4h `v4-run-receipt.json`, or runner-authored `v4-run-stop.json`. From the frozen call order and these file-presence checks, prediction sealing did not complete and held-out outcome evaluation was not reached. The partial predictions remain unvalidated; no outcome comparison or reproduction verdict can be inferred from them. The timeout bypassed the runner's exception handler, so no runner-authored stop receipt exists; the GMan command exit status and this report record the timeout.

## Custody and spend

The model assets, prompts, challenge data, and partial prediction outputs remain in GMan custody. No prompts, model weights, prediction files, or evaluation data were copied into this repository. The CPU-8 node in `us-west-2` is stopped, and GMan reports its disk snapshotted. Snapshot storage currently reports zero billable bytes under the organization's 250-GiB free allowance.

- Settled V4h mission spend: `$0.6390` (GMan mission settlement; node ran for 4,769 seconds).
- Frozen pre-V4h related V9 spend: `$1.1670`.
- Corrected total against the authorized `$2.00` V9 cap: `$1.8060`; remaining: `$0.1940`.
- External-review status: `NOT_SUBMITTED`.

Accounting correction: the earlier `$0.6042` V4h charge and `$1.7712`
aggregate were pre-settlement estimates. The live GMan mission receipt reports
`$0.6390`; all experimental and timeout conclusions above are unchanged.

## Claim ceiling and next gate

This attempt supports only the statement that V4h's local custody/runtime preflight passed and the frozen evaluator timed out during prediction generation. It does not establish a local reproduction, independent science, novelty priority, architecture-general validity, mechanistic understanding, Theseus identity, consciousness, alignment, human equivalence, or production authorization.

V4h is terminal under its one-attempt rule. Continuing requires a new frozen packet and a separately bounded execution plan; the remaining `$0.2288` does not authorize changing or rerunning this packet. Any completed same-owner result would still be `REPRODUCED_LOCAL`, not independent acceptance. External review and all other claim gates remain separate.
