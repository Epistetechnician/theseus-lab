# Theseus lab sandbox rules

All changes belong to the state slice
`theseus-lab-sandbox-v1`.

This repository is an isolated exploratory prototype. It has no configured
remote, no production authority, and no credential or model-asset storage.
Experiments may be locally permissive and may evolve their state model, but
provider calls, model weights, assessment data, and any production-facing
integration require an explicit adapter and must remain opt-in. Credentials
must come from the runtime environment or an approved secret manager and must
never be committed.

The prototype has no persistence by default. Every action must print the full
relevant state, preserve explicit lineage, and report whether behavior and
mechanism diverged. Treat all results as hypotheses until independently
validated elsewhere.
