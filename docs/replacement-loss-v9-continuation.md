# Replacement-loss v9 execution continuation

State slice: theseus-lab-sandbox-v1.

Assurance overlay: global-experiment-assurance-lifecycle-v1.

Continuation ID: theseus-replacement-loss-v9-execution-continuation-v1.

## Boundary

The frozen v9 primary stopped during challenge generation, before creating
challenge entropy, held-out prompts, predictions, or outcomes. Its candidate
preparation and candidate-freeze receipts are retained unchanged.

The frozen candidate receipt contains 81 prediction rules and sets
predictions_functionally_bound_before_challenge_entropy to true. The
challenge-generation guard checks the same predicate under the different key
prediction_rules_functionally_bound_before_challenge_entropy. The continuation
exposes a read-time alias for that guard only. It does not edit the v9 runner,
candidate receipt, candidate weights, selected checkpoints, thresholds,
predictor, or result calculations.

The continuation packet binds the exact v9 packet, checkpoint manifest,
candidate preparation, candidate freeze, original stop receipt, continuation
source, test, and this document. It must be frozen before challenge entropy is
generated. The continuation resumes at challenge generation with the existing
frozen candidates; it does not refit them or reuse any held-out data.

## Execution and claims

After challenge generation, the original frozen v9 prediction-sealing and
outcome-evaluation functions run unchanged. The existing v9 clean-runtime
replicator refits candidates from the visible and adaptation partitions, then
receives the held-out fixture bytes only after it freezes its own candidates.
The continuation binds its receipts to the v9 replication receipt.

The registered v9 thresholds and claim ceiling remain unchanged. A primary
PASS or FAIL is a measured local outcome. REPRODUCED_LOCAL requires all frozen
replication checks to match in a separate clean Python environment on the
same host. Reviewer status remains NOT_SUBMITTED; this does not establish
INDEPENDENT_ACCEPT, novelty priority, all-architecture validity, subjective
experience, alignment, or production authorization.

If the continuation stops, retain its stop receipt and do not regenerate
challenges or retry this packet.
