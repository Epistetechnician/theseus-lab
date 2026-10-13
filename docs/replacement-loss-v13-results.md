# Replacement-loss V13 results: nonlinear causal replacement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v13-nonlinear-causal-replacement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

Protocol: `theseus-replacement-loss-v13-nonlinear-causal-replacement`.

## Verdict

- Scientific outcome: `FAIL`.
- Primary execution assurance: `INTERNAL_PASS`.
- Same-owner second-runtime assurance: `REPRODUCED_LOCAL`.
- Reviewer status: `NOT_SUBMITTED`.
- Blind status: `NOT_BLIND_SAME_OWNER_REPRODUCTION`.

The preregistered comparison was the held-out next-logit causal-effect cosine
difference between the nonlinear causal-fit replacement and activation ridge,
clustered by fixture seed. The mean paired difference was `-0.321133`; the
10,000-draw clustered 95% lower bound was `-0.470225`, below the required
`> 0.05`. All six per-seed paired differences were negative. Causal-fit
retention failed. Activation rescue and the zero-intervention control passed.
The run therefore provides no positive evidence that this trained nonlinear
replacement improves held-out causal-effect prediction over activation ridge.

The second runtime produced the same scientific-results digest,
`3312c3052caaa6f00b0fe999d84222048cd3ad809464490a7b2095430df81aea`. The
frozen comparator reports `REPRODUCED_LOCAL`: all seven artifact groups were
byte-identical, and candidate, challenge, prediction, and scientific-result
content identities matched. This is local reproducibility on the same host
and under the same owner, not blind replication or independent acceptance.
The primary `outcome-report.json` retains its pre-comparison
`NOT_REPRODUCED` field; the later digest-bound comparator is the source of the
final `REPRODUCED_LOCAL` status.

## Frozen identity and target

- Repository base revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`.
- V13 packet digest: `93c1ecfce1f39b76433ebcf2452d2723824ce41de5e00ab6df6cd7e314af839c`.
- Freeze-file digest: `3cece104ef48d410b8e412f77fb3a694f3c9ed94494ccbfca3d5d7bcfadfaae0`.
- Frozen packet archive digest: `8fa33f30a9229c80c71e916e01c385e49faa0c7201e026aaab233e96ff59d2b7`.
- Target: `grushaprasad/opt125m-rootp-surprisal-curriculum@8af2a2926bc772599ccb3b4e7ad7b40b35259a63`.
- Target tree digest: `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`.
- Target-selection receipt digest: `77a69c8407aacf6e121777d4e051d3ddefa759cdcb0c68268afe09f09eb3b034`.
- Selected layers: OPT `fc2` layers 0 and 7; six synthetic fixture seeds.
- Stable candidate identity: `40ede5297df75cefda4d48ec787af1f9cde0ba64ca1211bc0e61c48a450288fa`.
- Stable challenge identity: `b8dc55ea656d720001b692df8b8dbef280706a48b2d52a1ab7f390da3870802c`.
- Stable prediction identity: `1b469ca57994f0483bb6b74e9918abb8134ddf286ea38d32d19c4fb450325685`.
- Comparator report digest: `e319a31cd558dc9b37ad43ce8e07433fe27705027e670d3640420aa629c35981`.

The tested treatment was a newly initialized `3072 -> 512 -> 768`
Linear-GELU-Linear module with no source weights, trained for the fixed 200
updates on adaptation captures. The comparator was an activation-ridge fit.
This reused one previously studied OPT/ReLU checkpoint; it was not a fresh
checkpoint or architecture-held-out evaluation.

## Execution and reproduction

Both runs used Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy
2.4.6, Tokenizers 0.22.2, Safetensors 0.8.0, and Hugging Face Hub 0.36.2,
with dependency-lock digest
`8f06ddb4781ec812c5c17eaf39e19c750995a8b6f7f20ebe3f07f3429752c84c`. They
used distinct Python runtimes on the same macOS ARM64 host. The primary took
2,043.00 seconds across preparation, prediction, and scoring, with peak RSS
3,619,422,208 bytes. The completed replica took 2,064.04 seconds, with peak
RSS 3,642,884,096 bytes. Both were within the frozen four-hour and 16-GiB
limits. Protocol qualification passed 6/6 focused tests in each pinned
runtime before freeze; all 27 frozen packet files still match their recorded
digests.

An initial replica preparation was interrupted after two of 24 candidate
files had been written. That partial custody root was preserved and not
reused, as required by the frozen procedure. The completed replica used a
fresh external custody root, the same frozen packet and target selection, and
the primary run's sealed challenge entropy. The partial attempt did not score
held-out outcomes and is not part of the reproduction comparison.

## Custody, claim ceiling, and limitations

Synthetic prompts, checkpoint files, captures, candidates, entropy,
predictions, and row-level outcomes remain in external PrimaryED custody. The
repository contains aggregate results and digests only. No paid compute,
cloud worker, GMan node, provider call, or public write was used. Monetary
cost is `NOT_APPLICABLE`; energy is `NOT_MEASURED`.

Frozen claim ceiling:
`LOCAL_NONLINEAR_CAUSAL_REPLACEMENT_IMPROVEMENT_OVER_ACTIVATION_RIDGE_ON_ONE_REUSED_OPT_CHECKPOINT_AND_SIX_SYNTHETIC_SEEDS`.
Because the outcome is `FAIL`, V13 supports no positive improvement claim.
The result does not establish architecture-general validity, mechanistic
understanding beyond this protocol, a novel proof of the Ship of Theseus,
consciousness, alignment, human equivalence, independent science, or
production readiness. V11 remains scientific `FAIL` with its original
comparator `INCONCLUSIVE`.

## Next gate

Keep V13 frozen. Diagnose why the nonlinear causal-fit candidate underperformed
activation ridge before expanding model coverage. A fresh, predeclared packet
could compare affine and nonlinear causal-fit candidates on the same fixtures
with the same objective and training budget, which would isolate the
architecture change more cleanly. Independent science still requires a
separate blind replicator and an authorized independent reviewer; alignment,
consciousness, and production gates remain separate.
