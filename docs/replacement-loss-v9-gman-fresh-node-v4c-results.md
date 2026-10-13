# Replacement-loss v9 GMan fresh-runtime reproduction v4c results

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4c`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

## Outcome

- Execution status: `INCONCLUSIVE`.
- Assurance level reached: none. The evaluator did not complete.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `INCONCLUSIVE`.
- The frozen V9 primary outcome remains `FAIL`; V4c did not reproduce or
  change it. Do not report this attempt as `REPRODUCED_LOCAL`.
- V4c packet SHA-256:
  `94a7b89d8154a59c0d7d878e16897d6e14425a1720bd6d3ec532f7e1c83702fc`.
- GMan mission `theseus-v9-linux-runtime-reproduction-20260928-v4` was
  finished with this `INCONCLUSIVE` result.
- Frozen V9 target packet SHA-256:
  `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.

The V4c runtime-reference and input preflights passed. The evaluator then
stopped before the fresh candidate-integrity preflight and before held-out
predictions or outcomes were opened. Its stop receipt records
`ValueError: installed distributions differ from requirements-v9.lock.txt`.
The run-stop receipt's recorded digest is
`02834bdc7ae9a31a432e9d96ae1648954c7388fc712e6e6522036758320c90fc`.
The command process exited with status 0 because the runner records caught
errors in a stop receipt; that exit code is not an evaluation pass. No V4c
replication manifest or outcome comparison was produced.

## Runtime mismatch

The reference runtime check matched the V3 Linux environment, including the
dependency-manifest digest
`0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4`.
The V9 evaluator applies a second, stricter check against the repository's
26-entry `requirements-v9.lock.txt`. The installed environment contains 51
distributions: the six primary dependency pins match, 25 additional
distributions are installed, and `packaging` is version `26.0` rather than the
lock's `26.3`. This makes the two frozen checks inconsistent for this runtime.
The stop occurred at `idle`; no scientific comparison ran.

The next continuation must freeze a consistent runtime contract while keeping
the V9 scientific packet, fixtures, thresholds, controls, and outcome order
unchanged. The runtime must still be bound to the verified Linux reference;
silently weakening the runtime check or editing V4c is not an acceptable
repair. V4c is terminal and must not be retried.

## Custody and spend

The node `theseus-v9-linux-runtime-v4-cpu8-20260928` is stopped with disk
intact. V4c receipts are retained as private GMan artifacts:

| Receipt | GMan artifact | Bytes | Artifact SHA-256 |
| --- | --- | ---: | --- |
| `v4-run-stop.json` | `art-cbars` | 607 | `25d898d562f8c4dce1d668b52942abc4520b81af3904f527a4f99a07a60ba7cd` |
| `runtime-preflight-v4.json` | `art-pb8jc` | 3,607 | `6523789e57f3a7b3c14de7493fca300ce82acd78eadc872369769d44ce20153a` |
| `v4-preflight.json` | `art-f9mq3` | 4,875 | `175b6a9f36d3d8b2722bdf5d1b2d9c49a36992ce91fc18f734570746ee0360dc` |
| `transport-preflight-v4c.json` | `art-nvcbj` | 6,260 | `9aa1987fcb08d1a31a0a4340aed32e4e06c275a946f1fbde66dfe2f806b8a9a4` |

The V4c transport receipt binds the reused V4b inputs and the supplemental
V3 protocol document. Model and fixture payloads remain outside this
repository in private GMan artifact custody.

At the final ledger read, the relevant node charges were V1 `$0.2092`, V3
`$0.5424`, and V4 `$0.2670` (33.23 node-minutes), totaling `$1.0186` of the
authorized `$1.34` cap. Remaining cap: `$0.3214`, or about 39.98 minutes at
the current `$0.00804/min` rate. A prior complete V3 path used 67.47 minutes;
the remaining cap cannot fund another complete attempt of comparable length.
The V4 node was stopped after receipt export. No more GMan compute is
authorized by the remaining cap for a full reproduction.

Post-stop ledger settlement before V4d raised the V4 node total to 36.23
minutes / `$0.2912`. The settled V1+V3+V4 total is `$1.0428`; this supersedes
the first post-stop snapshot above. At V4c's `$1.34` cap, settled headroom was
`$0.2972`. The user later authorized a `$2.00` aggregate cap for the separate
V4d continuation, leaving `$0.9572` at its start.

## Claim ceiling and next gate

This is not a reproduced result, independent review, novelty finding,
architecture-general result, mechanistic-understanding result, Theseus
identity proof, consciousness claim, alignment result, or production
authorization. No external reviewer was contacted.

The next blocked gate is a newly frozen continuation with a consistent,
digest-bound runtime contract and enough authorized budget for a complete
run, receipt export, and stop. A full-length attempt comparable to V3 would
require roughly `$0.54` for node time alone; the current `$0.3214` remainder is
insufficient. Independent science, alignment, consciousness, and production
claims remain separately gated.
