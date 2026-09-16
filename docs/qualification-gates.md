# Qualification gates

This repository uses three separate evidence gates in the
`theseus-lab-sandbox-v1` state slice. The implementation is in
[`qualification_gate.py`](../qualification_gate.py) and is intentionally
pure: it validates runner attestations but cannot create weights, provider
receipts, or assessment evidence.

## Gate meanings

`mechanistic_equivalence_observation` requires a verified checkpoint, locked
probe and threshold digests, held-out observations, raw digest divergence,
blind canonical equality, mapping recovery, behavior preservation, aligned
intervention agreement, and distinct control canonical digests. Its ceiling
is `LOCAL_VALIDATED_EQUIVALENCE_CLASS_OBSERVATION_ONLY`.

`alignment_validation` additionally requires an explicit target digest, a
real assessment-corpus digest, disjoint selection and assessment probes,
thresholds locked before assessment, an assessment receipt, aggregate-only
held-out metrics, and a passing target threshold. Local synthetic probes do
not satisfy this gate.

`production_readiness` additionally requires a named provider/model, a real
provider call, checkpoint/input/output/receipt digests, a passing alignment
gate, verified latency SLOs, repeatability, observability, failure recovery,
rollback, credential hygiene, and explicit production authorization.

The aggregate summary never widens an individual claim ceiling. Missing
evidence is reported as a blocker rather than inferred from a successful
local test.
