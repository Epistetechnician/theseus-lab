# Replacement-loss V9 GMan fresh-node V4i result

## Terminal disposition

- Protocol/state slice: `theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4i`
- Frozen packet SHA-256: `09e6434cd0c773354d1e78fbdfa2c9f179c171db4d927dd450f6859cee52c089`
- Command: `cmd-w2avs`
- Result: `INCONCLUSIVE_EVALUATOR_ERROR`; command exit code 1.
- Assurance: `INCONCLUSIVE`; reviewer status `NOT_SUBMITTED`.
- The evaluator did not start. Candidate preparation and candidate freeze are false; challenge fixtures were not revealed; prediction inventory is empty; no outcome or replication manifest exists.
- Prompts, model outputs, weights, and evaluation data were not exported.

The importer stopped on `frozen V9 source identity mismatch: runner_sha256`. The V4i freeze had populated the V9 source identity fields with adapter/helper hashes. The frozen V4h source identity records the actual V9 runner and replicator hashes, and read-only SHA-256 values from the GMan node match those V4h values. This is a packet identity-binding error, not a measured scientific mismatch.

## Cost and custody

At 2026-09-29 14:06:42 UTC, the V4i GMan mission reported $0.1224 accrued. This is not a settled final charge and includes the node's idle/setup time. The V4i terminal receipt is digest-bound in [the terminal JSON](replacement-loss-v9-gman-fresh-node-v4i-terminal.json). The node remains in GMan custody; no model or held-out payload was exported.

## Claim ceiling and next gate

V4i establishes no reproduction result. It does not change V4h's terminal `INCONCLUSIVE_TIMEOUT`, and it does not support `REPRODUCED_LOCAL`. The next local step is a distinct V4j freeze that retains V4i as a terminal pre-evaluation error, binds the original V9 runner/replicator digests correctly, and uses a fresh output directory. Any V4j run remains bounded by the same $5 aggregate cap and the existing node deadline. Independent science, alignment, consciousness, and production claims remain outside scope.

