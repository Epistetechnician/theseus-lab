# Replacement-loss v9 execution continuation v2

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Continuation ID: `theseus-replacement-loss-v9-execution-continuation-v2`.

## Why this continuation exists

The original v9 candidate freeze is preserved. Continuation v1 fixed the
candidate-freeze predicate name at read time, then stopped during challenge
generation after creating one secret and one challenge fixture. It did not
create intervention arrays, predictions, or outcomes. Those partial held-out
files remain in the v1 custody root and are bound by digest in this packet.

The v9 challenge caller indexes captured projection inputs by batch row and
sequence position. The frozen v3 capture helper returns valid tokens in a
flattened `[tokens, hidden]` layout, so that indexing returned a scalar. V2
qualifies an adapter on visible fixtures across all three selected families,
restores the helper output to `[batch, sequence, hidden]`, and checks that all
valid activation values round-trip exactly. This adapter is active only while
the frozen v9 challenge generator runs. The v3 helper is restored afterward.

## Frozen execution boundary

V2 stages a fresh external custody root containing the exact v9 packet,
checkpoints, visible and adaptation fixtures, candidate preparation, and
candidate freeze. It excludes all partial challenge entropy and challenge
fixtures from v1 custody. The v2 packet binds the staged-file inventory, the
candidate and checkpoint digests, the v1 stop and partial-material digests,
the shape preflight, source code, tests, documentation, and locked runtime.
Fresh held-out entropy is generated only after the v2 freeze.

Candidate fitting, checkpoints, prediction rules, thresholds, prediction
sealing, outcome evaluation, and claim ceiling remain unchanged. The adapter
does not refit candidates, reuse partial held-out material, or modify the
frozen v9 source packet. If v2 stops, retain its stop receipt and do not retry
that custody root.

## Assurance and limits

A successful primary result is local evidence. `REPRODUCED_LOCAL` requires the
frozen clean-runtime checks to match in a separate Python environment on the
same host. Reviewer status remains `NOT_SUBMITTED`. This work does not
establish independent science, novelty priority, all-architecture validity,
subjective experience, alignment, or production authorization.
