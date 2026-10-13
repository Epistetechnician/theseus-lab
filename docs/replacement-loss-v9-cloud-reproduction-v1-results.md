# Replacement-loss v9 GMan cloud reproduction v1 results

State slices: `theseus-lab-sandbox-v1` and
`theseus-replacement-loss-v9-gman-cloud-reproduction-v1`.

Assurance overlay: `global-experiment-assurance-lifecycle-v1`.

Protocol packet SHA-256:
`f39c2f22e085df8f563b2f0f2a3b32a5aafcc4a3ef03bd0b09f1886dd4d5409c`.

## Outcome

Cloud reproducibility status: `INCONCLUSIVE`. The model replicator did not
run. Its frozen runtime preflight rejected the Linux environment because the
installed distribution inventory did not exactly match
`requirements-v9.lock.txt`. No candidate was refit, no held-out intervention
was evaluated, and no replica predictions or outcomes were produced. This is
not a reproduction or a scientific result.

The original v9 outcome remains `FAIL` (21/27 ridge stages passed and both
predeclared comparative gates passed), with its prior same-owner
`REPRODUCED_LOCAL` status unchanged. The cloud result does not revise that
result.

Assurance: `INTERNAL_PASS` for the bounded archive, source, checkpoint, and
runtime-preflight integrity checks only. Scientific execution is
`INCONCLUSIVE`. Reviewer status: `NOT_SUBMITTED`, as authorized by the user.
No independent or external acceptance is claimed.

## Frozen target and execution receipt

- Frozen v9 packet SHA-256:
  `9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959`.
- Source revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- Frozen source packet SHA-256:
  `30ccf63209aa70ed8821b2614214114b010bd0c95f61bda772fa7e5ffbe548f4`.
- GMan mission: [theseus-v9-cloud-reproduction-20260928-v1](https://autoresearch.sfcompute.com/missions/theseus-v9-cloud-reproduction-20260928-v1).
- Node: `theseus-v9-cloud-20260928` (`a48bae49-4fac-43ee-a6ee-e378e99fa51f`),
  CPU-8, x86_64, Ubuntu 24.04; rate `$0.00804/min`; stopped after 26.03
  billed minutes for `$0.2092`.
- Input archive: 1,727,986,827 bytes, 396 members,
  SHA-256 `89af9ff9add82adbff199eda90d876653b041660f4bd1d6e2803fd47dcdb5bfe`.
- The archive input manifest verified 280 source/primary files against
  inventory SHA-256
  `e31b70ed20278a974bf96f60a2217d94f3c695668ca288e1afc5580302c6e962`,
  plus 112 prior-fixture files against inventory SHA-256
  `3b46ce7f74bd8b8a24f5324cd5aae5ef34933b71a3f8ce731dcb8488480c9421`.
- The copied primary outcome manifest retained its frozen SHA-256
  `12e0befa4c329d3a6a18b14d19144ab6296e5ac5b82f0cb2326eaeff7c776431`.
- The runtime check used Python `3.12.13`. The frozen lock lists 26
  distributions; the installed environment contained 51, including 25 extra
  distributions such as CUDA runtime libraries and Triton. It also resolved
  pinned `packaging` to `26.0` instead of `26.3`. The dependency
  inventory SHA-256 was `0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4`,
  rather than the frozen expected
  `bea06dc458f522768a207ea383d8220ab3b9e0eda8c9de72c5f6468cafd3d94d`.
  The core model package versions matched, but the exact lock check failed.
- The runner reported `ValueError: installed distributions differ from
  requirements-v9.lock.txt`. The replica directory did not exist, confirming
  the evaluator did not start.
- The temporary GMan upload context was deleted after the node verified the
  archive digest. Its 1,727,986,827 bytes were confirmed deleted at
  `2026-09-28T00:43:53Z`. The node is stopped with its private disk parked;
  scratch was cleared by GMan.

## Custody

The input archive and exported receipts remain in owner-controlled external
custody at
`/Volumes/PrimaryED/ResearchArtifacts/theseus-lab-sandbox-v1-replacement-loss-v9-gman-cloud-reproduction-v1/`.
The runtime preflight receipt SHA-256 is
`55454740858795bcd7027d43a070284684632f07abc58cfe56252f71801003ea` and the
setup stop receipt SHA-256 is
`f458a36e4879e06b09eb6945e705c39673432b013176fb115b5c46daf1b079c5`.
Their exported receipt archive SHA-256 is
`5c76cde5d971f1a48dc7ebe3c0ba2bc95fded64a6df2a71b6f9509aa18232ba4`.

Model assets and fixture rows remain outside this repository. Only protocol
metadata and aggregate execution details are recorded here.

## Claim ceiling and next gate

The frozen cloud claim ceiling was
`LOCAL_CROSS_RUNTIME_REPRODUCTION_OF_THE_FROZEN_V9_FAIL_RESULT_ON_THE_SELECTED_CHECKPOINTS_AND_FIXTURES`.
That claim was not reached. The evidence supports only successful transfer
integrity and a recorded runtime incompatibility. No claims follow about
mechanistic understanding, novel proof, all architectures, consciousness,
alignment, or production readiness.

The next technical step is a separately frozen Linux CPU runtime packet that
binds a compatible exact distribution inventory and its verifier before any
new model evaluation. Preserve this v1 `INCONCLUSIVE` result; do not rerun it
or normalize its runtime identity. Independent science still requires a
digest-bound packet and a reviewer with independent authority if that claim
is later pursued.
