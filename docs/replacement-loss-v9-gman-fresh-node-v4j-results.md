# Replacement-loss V9 GMan fresh-node V4j result

## Terminal disposition

- Protocol/state slice: `theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4j`
- Frozen packet SHA-256: `d35c9357c147f97e0857537232685a547418393d44e98c477759675fc5ff6fd1`
- Command: `cmd-jkn3t`; exit code 1.
- Result: `INCONCLUSIVE_EVALUATOR_ERROR`; assurance `INCONCLUSIVE`; reviewer `NOT_SUBMITTED`.
- The evaluator process started, but stopped during the run phase before candidate preparation. Candidate freeze is false, challenge reveal is false, predictions and outcomes are absent, and the prediction inventory is empty.
- No model prompts, outputs, weights, or evaluation data were exported.

The digest-bound run stop reports `ValueError: in-memory V9 dependency adaptation differs from the frozen V3 patch`. V4j restored the original V9 runner and replicator hashes, but its `patched_replicator_sha256` field still held the V4j wrapper hash. The frozen V4h packet binds that field to the exact V3 in-memory patch hash `6892b0a99bdfb7fed3950f06c9adc4b16dcaf87c9dcca890b25bec1829d2436e`. The error happened before candidate preparation or held-out evaluation. The internal run-stop digest is recorded in [the run-stop receipt](replacement-loss-v9-gman-fresh-node-v4j-run-stop.json); the outer watchdog receipt is [here](replacement-loss-v9-gman-fresh-node-v4j-terminal.json).

## Cost and custody

The latest GMan mission observation was $0.243 accrued, not settled. It includes node idle/setup and the V4i/V4j attempts. The same node remains in GMan custody under the original 315-minute lifetime limit. No prediction payload or model output was exported.

## Claim ceiling and next gate

V4j is not a reproduction result and does not support `REPRODUCED_LOCAL`. V4k must preserve this terminal attempt and bind all three source identity digests from the original V4h freeze: the V9 runner, V9 replicator, and exact patched-replicator digest. The scientific target, adaptation, replacement methods, fixtures, thresholds, controls, and outcomes remain frozen. The aggregate run cap stays $5.00; external review, alignment, consciousness, and production claims remain outside scope.

