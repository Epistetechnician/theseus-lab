# Replacement-loss v9 GMan fresh-runtime reproduction v4b

State slices: theseus-lab-sandbox-v1 and
theseus-replacement-loss-v9-gman-fresh-node-reproduction-v4b.

Assurance overlay: global-experiment-assurance-lifecycle-v1.

## Amendment and scope

This packet supersedes two unexecuted packets: V4
53705b8feadbe173f37879295db476156049bb4a32c9ff7a1f308dc1a85c2035 and the
first V4b freeze 1cae87f4dc4b96f62f82f9d260660325ce1386caf8ac49850b7d0e151ac30fdd.
The V4 CPU-node queue made overlapping node-to-node transfer exceed the
authorized cost boundary. V4b changes only the custody transport: it exports the exact
frozen V3 input components and receipts into GMan's private artifact store,
then verifies and imports them onto the fresh V4 node. The model, checkpoints,
fixtures, candidate rules, thresholds, controls, held-out order, and success
criterion are unchanged. No held-out outcome was opened during this amendment.

## Frozen scientific design

- Target V9 packet: 9366fa49da0513be2f405fdff7f8ecd685b4a29991a31ec4f1eac0b947aae959.
- The target contains three architecture families, three fixture seeds, and
  three sequential replacement stages per trajectory.
- The primary metric and all thresholds remain the V9 causal-effect prediction
  and retention gates comparing ridge, CKA, and logit distillation.
- Wrong-module, random, and no-op controls remain unchanged.
- Replica candidates are refit from visible/adaptation data. Candidate file
  bytes and tensor states are checked before challenge reveal; predictions are
  sealed before held-out outcome evaluation.
- One V4b attempt is allowed. A complete match to the frozen V9 outcome FAIL
  yields REPRODUCED_LOCAL; a measured mismatch yields DIVERGED; incomplete
  work yields INCONCLUSIVE. No retry or retuning is permitted.

## Artifact custody transfer

The exports were made from the V3 node after its stopped disk was restored.
GMan's private artifact store returned these content receipts. The V4b
importer downloads each artifact through a short-lived private URL supplied
only in the command environment, checks exact byte count and SHA-256, rejects
unsafe archive members, and records a digest-only transport receipt before
the V9 preflight can run.

| Key | Artifact | Bytes | SHA-256 | V4 destination |
| --- | --- | ---: | --- | --- |
| input_archive | art-beb8f | 1,727,986,827 | 89af9ff9add82adbff199eda90d876653b041660f4bd1d6e2803fd47dcdb5bfe | input/gman-v9-input.tar.gz |
| primary | art-v7dz8 | 2,072,340,480 | 26efd0c8730d041cd2ab322aa75b82afbb13ef4e7cd82bda60b7894317cc5978 | input/primary |
| metadata | art-fzzez | 112,640 | b2c607b224fddacfd5e38d472e4c964f4d23a56047000d7c10c033dee1408b18 | input/metadata |
| prior_fixtures | art-jjqsg | 3,983,360 | 23b22a317af40df063d629f8772eda9eb7fc9926beb67d36022f1e445d4cc6af | input/prior-fixtures |
| source_checkout | art-edubf | 1,792,000 | 275dceb5b444fab9a531ac414b73cc5a4a1846c74bd59cc7ab2d7f3f3b9c2a48 | input/source-checkout |
| v3_packet | art-bz3f7 | 81,920 | 7cc03b650285c715b1239ab56c6d2ba932dbf6d2380f16b3be2b29b65ace1793 | v3 |
| receipts | art-8te66 | 10,240 | cd22dfd75b538ab20e28b69449251df153a7bcb0e4db3b4a3d3b98e8c75ed66c | receipts |
| v3_stop | art-f7kfg | 499 | 5c823ee3578d845b782fb413739ac00bdfce916c6bc82b9ce2a0110d6145c90e | v3-stop/replication-v3-stop.json |

The first all-input-directory export returned an error and no artifact ID was
returned or used. Its exact diagnostic was not retained in the status summary.
The bounded component exports listed above succeeded. The V3 stop receipt
matches the frozen V4 binding. The imported input manifest, source identity,
V3 packet, V3 runtime receipt, stop receipt, primary outcome, candidate
freeze, and challenge manifest are checked again by the V4 runner before
evaluation.

## Runtime, budget, and stopping rule

The rebuilt environment must match the V3-observed Python 3.12.13 Linux x86_64
runtime, one PyTorch CPU thread, and dependency-manifest SHA-256
0496a9c718bce898530689821630c5a4874823bfb152c672081b4c6779328bc4.

GMan billing now records $0.2092 for V1 and $0.5424 for V3, including the
completed V3 recovery and artifact exports. The fresh V4 CPU-8 node accrued
8.23 minutes ($0.066) before evaluation while the custody amendment was
frozen. Its total node ceiling is 72.5 minutes ($0.5829), with the evaluator
command capped at 60 minutes. The aggregate maximum is $1.3345 against the
authorized $1.34 cap, leaving $0.0055. Stop immediately after exporting
digest-only receipts.

Stop before evaluation on any freeze, artifact, runtime, input, candidate, or
budget mismatch. An incomplete attempt is retained as INCONCLUSIVE and is not
retried.

## Custody and claim boundary

Prompts, model weights, token IDs, captures, predictions, row-level outcomes,
and traces remain in GMan or PrimaryED custody. The exported V4b receipt is
limited to digests, counts, runtime identity, checks, and terminal status.
Reviewer status is NOT_SUBMITTED.

The maximum claim is same-owner local cross-runtime reproduction of the
frozen V9 FAIL under the digest-bound Linux runtime. This is not blind external
replication, independent scientific acceptance, novelty priority,
architecture-general validity, mechanistic understanding, a Theseus identity
proof, consciousness evidence, alignment evidence, human equivalence, or
production authorization.
