# Replacement-loss-v8-scale-r10 results: float64 measurement gate

State slice `theseus-lab-sandbox-v1`. Protocol `theseus-replacement-loss-v8-scale-r10-float64-measurement-gate`, freeze packet `fa2baf3e97b4c04a709858d5b354cca6ec9e64e5976c0079ac26d63d47c8938c` (118 files; count asserted by test). Preregistration (committed first, `0f1f8bf`): `docs/replacement-loss-v8-scale-r10.md`. Runner/tests/freeze: `a4755d0`. Raw report: `docs/replacement-loss-v8-scale-r10-report.json`.

## Outcome: `FLOAT32_WAS_LIMITING` (measurement of the instrument only)

**All earlier outcomes are unchanged:** r6 NN256/NN512 `FAIL`, r9 NN256/NN512 `FAIL`, Band B N256/N512 `FAIL`, opt-6.7b N256 `FAIL` and N512 `PASS`, as recorded. r10 turns nothing into a PASS, waives no frozen gate, and changes no constant. It says only that, on these inputs, a float64 forward removes the float32 misses on the positive cases and does not rescue the negative control.

48 of 48 planned cases ran. No arm is `NOT_RUN` (custody existed for every arm; nothing was regenerated).

| prediction | result |
|---|---|
| P1 Band A positives (36 cases) pass the float64 gate | **held**, 36/36 |
| P2 Band B N512 positives (9 cases; open_llama per-stratum >= 0.99 and within 0.005 of r8) | **held**, 9/9; open_llama reproduces r8 float64 exactly (compositional 0.9973/0.9962/0.9985, ordinary 0.9984/0.9989/0.9988) |
| P3 negative control (opt-6.7b N256) fails the cosine condition in all 3 seeds | **held, narrowly** (see below), 0/3 pass |
| P4 no positive module error above 1e-10 | **held**, max 9.3e-12 |

## Per-arm table (float64; gate: module <= 1e-10, logit <= 1e-4, every stratum median cosine >= 0.90)

| arm (device) | model | cases passing | max module err | max logit err | min stratum median cosine |
|---|---|---:|---:|---:|---:|
| r6 NN256, r6 NN512, r9 NN256, r9 NN512 (H100) | opt-2.7b | 12/12 | 2.3e-13 | 8.1e-14 | 0.9996 |
| same four arms (cpu-8) | gpt2-xl | 12/12 | 1.3e-12 | 1.0e-13 | 1.0000 |
| same four arms (cpu-8) | SmolLM2-1.7B | 12/12 | 9.3e-12 | 7.4e-06 | 0.9981 |
| Band B N512 (H100) | opt-6.7b, gpt2-xl, open_llama_7b_v2 | 9/9 | 6.4e-13 | 1.3e-05 | 0.9962 |
| **Band B N256, negative control (H100)** | opt-6.7b | **0/3** | 2.7e-13 | 3.2e-14 | **0.8869** |

The nine SmolLM2 cases that missed the frozen float32 1e-4 diagnostic (r6: 2 per arm; r9: 3 and 2) reproduce in the float32 informational arm (module errors 1.2e-4 to 8.5e-4) and fall to <= 9.3e-12 in float64.

## Negative control: read this carefully

float64 raised opt-6.7b N256 per-stratum median cosines from 0.79-0.81 (float32, as recorded) to 0.887-0.895 (compositional 0.892/0.890/0.887, ordinary 0.895/0.890/0.892; adversarial 0.99). All six primary cells are below 0.90, so P3 holds as written, but by about 0.005-0.013, not by a wide margin. The reading consistent with the data is that float32 resolution costs roughly 0.1 of cosine here and the remaining gap (module-boundary cosine about 0.88) is the real fit gap. Float64 therefore does not rescue the coverage failure, but it is not true that float32 explains the whole N256 miss. One negative-control arm cannot show the gate is discriminating in general (stated in the preregistration).

## Deviations and disclosures

- **Pre-run interpretation choices, fixed in the frozen runner before any run but not written in the preregistration text:** (a) "negative control passes the float64 gate" is implemented as *any* negative-control case passing conditions 1-3 (stricter than "all three"); (b) P2 conditions 1-3 are applied to all nine Band B N512 cases, including gpt2-xl, not only the two models named; (c) per-stratum cosine uses all three strata present (compositional, ordinary, adversarial), not only the two primary ones used in r8; (d) a fallback label `UNCLASSIFIED_NEGATIVE_CONTROL_NOT_EVALUABLE` exists for P3 failing without a negative case passing (not triggered); (e) planned cases = 48 (36 + 9 + 3). None changed a threshold.
- **Informational float32 arm** (not gating, not in the preregistration): each case was also measured in float32. On the same hardware as the original runs it reproduced the recorded per-stratum cosines exactly (max gap 0.0: all CPU arms and Band B on H100). The four opt-2.7b arms ran on the H100 while the originals ran on cpu, so their float32 cosines differ from the recorded ones by up to 1.1e-3 (device, not data).
- **Custody check:** for every arm, `candidate-preparation.json` and `checkpoints.json` match the sha256 recorded in the earlier report, and every per-case prompt and intervention file matches the digests in the custody challenge manifest. The `challenge-manifest.json` file hash does **not** match the one recorded in any earlier report (all six arms). I did not investigate why (the manifest is rewritten after the report hash may have been taken); the per-case digests, and exact float32 reproduction on matching hardware, are the evidence that inputs are the same. Treat the manifest-hash mismatch as an open lineage gap.
- **Where things ran:** r6 NN256, r9 NN256, r9 NN512 custody on `r5-n256`; r6 NN512 on `r5-n512`; Band B on `theseus-bandb-v8sb`. `~/r6-NN256` on r5-n256 is a stale partial directory (no challenge manifest); the intact copy is `~/custody/r6-NN256`, which was used. opt-2.7b float64 does not fit the 32 GiB cpu-8 cgroup, so (as planned) opt-2.7b went to the H100: the opt-2.7b weights (same files in every arm's custody) and each arm's metadata/candidates/challenges were copied to `~/opt-custody/<arm>` on the H100 node (read-only copies; nothing regenerated). gpt2-xl and SmolLM2 ran on cpu-8 with no layer slicing.
- **Order:** cpu-8 arms ran before the H100 arms (the H100 sat in the capacity queue for over an hour); results do not depend on order.
- **Environment:** pinned node venvs (torch 2.6.0+cu124, transformers 4.57.3); TF32 disabled on CUDA. Packet verified on every node.
- **Operational:** r5-n256 restore took over an hour and the node once showed `stopping` mid-session; its disk was intact and it was restarted. A broad `pkill` I issued was denied by the permission classifier and killed nothing.
- **Spend:** `gman usage` totals for my nodes read r5-n256 $5.72, r5-n512 $1.14, theseus-bandb-v8sb $3.92 = $10.78, an upper bound because those totals include earlier sessions' usage of the same nodes (r9: $3.14 + $0.22; r8: $0.36); netting those out gives about $7. Under the $20 cap either way. Other nodes in the usage list are not mine.
- **Nodes:** I started r5-n256, r5-n512 and theseus-bandb-v8sb; all stopped (`gman node ls` verified). Nodes I did not start were not touched.

## Limits (claim ceiling)

Same owner, five checkpoints, 3 seeds, two corpora, one projection, synthetic intervention directions, no independent review; fitted modules are the unchanged float32 fits cast exactly to float64, so r10 cannot detect problems in the fits or corpora. The 1e-10 module threshold was informed by r7 for SmolLM2 (preregistration section 4); opt-2.7b, gpt2-xl and Band B module errors are the out-of-sample part (all <= 1.3e-12). `FLOAT32_WAS_LIMITING` is a statement about the instrument; it supports neither a mechanistic nor an alignment nor a production claim. Whether to adopt a float64 gate for future protocols is a separate decision. Hypotheses until independently validated.

## Addendum: challenge-manifest hash gap (code reading, not yet confirmed against custody)

Earlier reports record `challenge_manifest_sha256`, which `_add_v8s5_binding_to_challenge_manifest` (`replacement_loss_v8s5_runner.py`, same pattern in `v8sb`) computes as the canonical digest of the manifest's JSON content with that field removed, after adding the `v8s5_*` binding fields. r10 recorded `custody_challenge_manifest_sha256`, the sha256 of the file's raw bytes. These are different quantities, so they would differ even for an unmodified file. This explains the mismatch without any change to the inputs. It is a reading of the code: I have not recomputed the canonical digest on the custody manifests, which would settle it (read-only, one node restore). The per-case prompt/intervention digests and exact float32 reproduction remain the evidence that inputs match. No outcome or classification changes.

### Confirmation (r6 NN512 custody, `r5-n512`)

Recomputed the canonical digest (manifest JSON without `challenge_manifest_sha256`, sorted keys, compact separators) on the custody manifest for r6 NN512 (a read-only check on the stored files). It equals both the value stored in the manifest and the `challenge_manifest_sha256` in `docs/replacement-loss-v8-scale-r6-NN512-report.json` (`7ac8255d30b0a987705991f62853992d03a45dfb83b0f4dd73b161f2274045cb`). The r5 N512 manifest also self-verifies. So for r6 NN512 the "mismatch" was a difference in digest definition (file bytes vs canonical), and the manifest is unchanged since the report. Not rechecked for the other five arms (r6 NN256, r9 NN256/NN512, Band B N256/N512), which sit on other nodes; the same code path wrote them, so the same explanation is expected but is not verified.
