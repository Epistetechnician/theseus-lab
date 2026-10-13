# Replacement-loss v9 execution continuation v3

State slice: `theseus-lab-sandbox-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Continuation ID: `theseus-replacement-loss-v9-execution-continuation-v3`.

## Why this continuation exists

Continuation v2 generated and sealed the complete nine-case challenge packet,
then stopped before prediction sealing. All candidate file bytes still match
the frozen candidate inventory. The frozen runner compares each serialized
file digest with a stage field that contains the digest of the loaded tensor
state. Those are distinct digest domains, and the byte digest is separately
bound by the candidate freeze.

V3 copies the exact v2 challenge packet and candidate artifacts into a new
external custody root. Its preflight checks every candidate file against the
frozen byte inventory, loads all 81 treatment stages, and verifies each
tensor-state digest against its stage receipt. At prediction and outcome
evaluation time, a read-time loader performs those same two checks separately.
It does not change any candidate, prediction, or outcome calculation.

## Frozen execution boundary

The v3 packet binds the v2 packet and stop, candidate freeze, full challenge
manifest, secret and fixture byte inventory, source files, tests, documentation,
and locked runtime. It resumes at prediction sealing. It does not regenerate
challenge entropy or refit candidates. The candidate prediction rules were
frozen before the v2 challenge packet was generated.

If v3 stops, retain its receipt and do not retry that custody root.

## Assurance and limits

A successful primary result is local evidence. `REPRODUCED_LOCAL` requires the
same frozen checks to match in a separate clean Python environment on the same
host. Reviewer status remains `NOT_SUBMITTED`. This does not establish
independent science, novelty priority, all-architecture validity, subjective
experience, alignment, or production authorization.
