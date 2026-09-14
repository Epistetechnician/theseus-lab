# Prototype notes

Question: Can a system preserve its observed behavior while its explicitly
tracked computational mechanism is replaced or lost?

Current result: the prototype separates two effects. A structural split keeps
visible and held-out behavior at `1.0` while mechanism continuity falls to
`0.4167`, producing a structure-only Theseus counterexample. A replacement
matches visible probes but fails held-out probes, demonstrating that visible
behavioral equivalence can be a probe-coverage artifact. This is a deterministic
state-machine observation, not evidence about a trained model.

The causal intervention score and component-set overlap are diagnostic signals;
they do not establish mechanistic understanding. The next experiment is to
replace the synthetic mechanism profiles with a caller-owned checkpoint summary
while retaining held-out probes, interventions, and the explicit lineage ledger.
