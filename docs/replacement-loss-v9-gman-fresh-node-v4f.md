# Replacement-loss v9 GMan fresh-runtime continuation V4f

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4f.

Assurance overlay: global-experiment-assurance-lifecycle-v1.

## Continuation and scope

V4f is a new, one-attempt continuation after V4e ended as
INCONCLUSIVE_NO_EVALUATION. V4e transport checks passed. Its evaluator did not
start and held-out predictions and outcomes remained unopened. The terminal
stop receipt and runtime diagnostic are bound below.

The measured V4e issue was distribution metadata enumeration after importing
PyTorch: setuptools added its vendored packaging metadata (26.0) to the
enumerated distributions, shadowing the active site-packages packaging
distribution (26.3). The direct import and distribution checks showed all
other 50 active distributions matched the expected 51-package map.

V4f reuses the exact stopped V4e snapshot and its Python environment. The
runtime adapter excludes only distribution metadata whose metadata path
contains a _vendor segment, rejects duplicate active package names, and
requires the complete normalized map to match the frozen 51-entry map with
packaging 26.3. It then checks all 26 requirements-v9 lock entries exactly.
It does not alter installed packages, V9 source, model weights, candidate
fitting, data, evaluator logic, controls, thresholds, or outcome order. The
runner truthfully retains locked_environment_pass=false because the environment
has 51 distributions while the V9 lock lists 26; the separate frozen lock
subset check requires all 26 exact versions.

## Frozen scientific design

- Target V9 packet:
  9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959.
- Frozen primary outcome: FAIL, with 21 of 27 ridge stages passing.
- Three architecture families, three fixture seeds, and three sequential
  replacement stages per trajectory.
- Ridge, CKA, and logit-distillation methods plus wrong-module, random, and
  no-op controls are unchanged.
- Candidate file bytes and tensor-state digests must pass before challenge
  reveal. Replica predictions must be sealed before held-out outcomes open.
- The digest-bound V4 implementation runs the existing frozen comparisons.
  REPRODUCED_LOCAL requires every frozen check and the primary FAIL outcome to
  match. A completed mismatch is DIVERGED. A failed preflight or incomplete
  run is INCONCLUSIVE.
- One attempt only. No retries, threshold changes, seed changes, fixture
  regeneration, candidate retuning, or outcome reinterpretation.

## Custody, runtime, and budget

The input archive, primary results, checkpoints, source overlay, and prior
fixtures remain in GMan custody. The V4f importer verifies the V4e freeze,
terminal preflight stop receipt, passing transport receipt, V4c lineage, V9
input manifest, target lock, and V4 implementation digest before preflight.

V4f restores snapshot snap-t3r2t, containing the stopped V4e workspace and
Python 3.12.13 environment. Runtime identity is Linux x86_64, one PyTorch
thread, 51 normalized active distributions, and full active distribution-map
SHA-256 0214786b54e5b8629740fc194eaca188d7627c6ae208997ac38068b0f1fe2a63.
The 26-row V9 lock SHA-256 remains
034f815d48112752efb5b7278447b6f4c6653ce5c9316da0eb5a1a15ab188a5e.

At freeze time, settled related spend is $1.1422 of the user-authorized $2.00
aggregate cap, leaving $0.8578. V4f has a $0.75 incremental hard cap. Its
single CPU-8 node is limited to 82 billed minutes at $0.00804 per minute
($0.65928). The bootstrap and evaluator command is limited to 73 minutes;
the remaining nine node-minutes cover wake setup, receipt export, and
shutdown. Worst-case related spend is $1.8922, leaving $0.1078 under the
aggregate cap. Stop if the ledger or remaining budget cannot honor those
limits.

The bootstrap runs the importer and runtime preflight first. It starts the
single evaluator attempt only after digest-bound transport and runtime
preflight receipts pass and held-out data remain unopened. On any mismatch,
stop before held-out evaluation. Stop the node immediately after exporting
digest-only receipts.

## Claim ceiling

At most: same-owner local cross-runtime reproduction of the frozen V9 FAIL
under the digest-bound V4f Linux runtime. Reviewer status is NOT_SUBMITTED.

This does not establish independent science, external replication, novelty
priority, all-architecture validity, general mechanistic understanding,
Ship of Theseus identity, consciousness, alignment, human equivalence, or
production authorization.

