# Prototype notes

Question: Can a system preserve its observed behavior while its explicitly
tracked computational mechanism is replaced or lost?

Current result: the prototype produces a counterexample when behavior remains
unchanged while lineage coverage falls below the configured threshold. This is
a state-machine observation, not evidence about a trained model.

Next experiment: replace the synthetic `components` mapping with a caller-owned
checkpoint summary while retaining the same explicit lineage ledger.
