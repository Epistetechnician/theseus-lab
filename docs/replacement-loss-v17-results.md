# Replacement-loss V17 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v17-causal-supervised-whole-mlp-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Disposition

- Scientific outcome: `FAIL` under the frozen absolute and comparative gates.
- Assurance: `REPRODUCED_LOCAL` in a separate Python runtime on the same host
  and owner. This is not blind replication, independent review, or
  `INDEPENDENT_ACCEPT`.
- Behavior and mechanism diverged: `true`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `REPRODUCED_LOCAL`.

V17 trained full OPT MLP replacements with finite-intervention supervision on
adaptation data and compared them to equal-budget pointwise-only replacements.
The causal loss lowered its adaptation-validation effect error, and it
improved held-out next-logit effects in some strata. It did not generalize
across seeds and prompt strata: absolute causal prediction failed throughout,
activation-error retention failed in eight of nine cases, and the frozen
causal-versus-pointwise improvement gate passed in only three of nine cases.
Those misses remain the result; thresholds and code were not changed.

## Frozen identity and execution

- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`; the worktree
  was already dirty with 254 entries before the run. No unrelated state was
  changed, and no commit was created.
- V17 packet SHA-256:
  `e08b5664e8ebbc536f5ed8ff13ab0aa8b536caddb61a3540a5cb56a94f403869`.
- V17 external archive SHA-256:
  `6d786968f6f3941c0c944fc9c20d624b6450e82ae3a54cd0f8cd3bcd7120053e`.
- Parent V16 packet SHA-256:
  `72469f59ff36c465542caab84970953b0eac69532f0d2fde5d292ee424b2a763`;
  parent scientific outcome: `FAIL`.
- Target: `grushaprasad/opt125m-rootp-surprisal-curriculum`, revision
  `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`; tree SHA-256
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`;
  weights SHA-256
  `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
  V17 is an outcome-aware follow-up on this reused checkpoint and preselected
  block 8, not a fresh-target confirmation.
- Three new adaptation seeds trained causal, pointwise, wrong-layer, and
  random candidates (12 saved candidate artifacts). Three fresh challenge
  partitions contributed 180 held-out prompts, with predictions frozen before
  scoring.
- Runtime: Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6,
  one Torch thread, deterministic algorithms, local CPU. Both runtimes matched
  the complete locked package inventory.
- Primary phase times were 130.7 seconds for preparation, 41.2 seconds for
  prediction, and 15.2 seconds for scoring. Peak RSS was 2.82 GB. Replay peak
  RSS was 2.94 GB. Both remained within the four-hour and 16 GiB limits.
- Monetary cost: `NOT_APPLICABLE`; energy: `NOT_MEASURED`.

## Results

All nine seed-by-stratum cases had 20/20 resolvable module and next-logit
effects. The separate-runtime replay reproduced all six artifact groups
byte-for-byte: fixtures, splits, captures, candidates, challenge entropy, and
predictions. Scientific outcome and digest also matched.

| Measure | Observed range | Frozen gate | Result |
|---|---:|---:|---|
| Module-effect direction accuracy | 0.95–1.00 | ≥0.80 | Passed |
| Module-effect median cosine | 0.177–0.951 | ≥0.90 | Failed overall |
| Module-effect median relative error | 0.310–1.049 | ≤0.20 | Failed in all 9 cases |
| Next-logit-effect direction accuracy | 0.35–0.95 | ≥0.80 | Failed in most cases |
| Next-logit-effect median cosine | −0.392–0.683 | ≥0.90 | Failed in all 9 cases |
| Next-logit-effect median relative error | 0.801–1.531 | ≤0.20 | Failed in all 9 cases |
| Causal vs pointwise cosine gain | −0.692–+1.105 | ≥+0.05 | Both comparative gates passed in 3/9 cases |
| Causal vs pointwise error reduction | −0.395–+0.798 | ≥+0.10 | Both comparative gates passed in 3/9 cases |
| Replaced-boundary activation cosine | 0.925–0.982 | ≥0.90 | Passed in all 9 cases |
| Replaced-boundary activation relative error | 0.190–0.382 | ≤0.20 | Passed in 1/9 cases |
| Continuation cross-entropy change | −0.0749–+0.05002 nat/token | ≤+0.05 | One case narrowly failed |
| Normalized next-logit behavior error | 0.0073–0.0295 | ≤0.05 | Passed in all cases |
| Oracle-restoration max logit error | 0.0 | ≤1e−5 | Passed |

The requirement that causal supervision exceed wrong-layer, random, and no-op
controls by at least `+0.05` in every seed/stratum passed in 4/9 cases. Causal
supervision improved its normalized adaptation-validation effect MSE to
0.607–0.615 versus 1.049–1.054 for pointwise-only, but its normalized
validation value MSE was worse (0.063–0.086 versus 0.022–0.023). The observed
tradeoff is consistent with the held-out retention failures; it does not show
that another loss weighting would pass.

The oracle error of zero validates the intervention and restoration path. It
does not count as a learned activation rescue.

## Interpretation and claim ceiling

On the reused OPT checkpoint and preselected block, adaptation-only finite
intervention supervision produced better local-effect fit on validation data
and occasional held-out gains over pointwise-only training. Those gains were
not reliable across prompt strata or seeds and did not meet the absolute
causal-effect or activation-retention gates. Strong baseline behavior alone
still did not establish causal-mechanism preservation.

Maximum claim ceiling:
`LOCAL_CAUSAL_SUPERVISION_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT`.
This post-result follow-up does not establish architecture-general validity,
consciousness, alignment, production readiness, independent science, or a
novel proof of Theseus continuity.

## Evidence digests and custody

- Candidate set SHA-256:
  `f518394b15287582b5ce39172c6974b4b9eb2afba957975bd710998484f379cc`.
- Prediction set SHA-256:
  `b4e2b9740066786add55a1e0c54d6b2c3e4a79a6ad4c6d3f43194ed98730fe9b`.
- Scientific result SHA-256:
  `bb6132cb9a7e41ddf076955ac98f9d75adf01004e40be819aabaf9d2defaa731`.
- Replication report SHA-256:
  `9297cea0c24e5f8d237f78b97ac3155dcd95db5acb31b6f2de7caabed950e5cf`.
- Final primary outcome report SHA-256:
  `429c27b697ecb0e9dac525f45d321c85604f07afb65aee92ef5d02c3974071c0`.

Prompts, captures, causal targets, candidates, entropy, predictions, and
row-level outcomes remain in external owner custody. This repository contains
only protocol code and aggregate-safe results.

## Next gate

Keep V17 frozen as `FAIL`. A future local study should separate representation
retention from causal response and avoid changing weights against V17's
challenges. One useful question is whether the causal/value tradeoff can be
improved by a predeclared multi-objective or Jacobian-targeted training rule on
new adaptation prompts, followed by a newly generated challenge set and the
same absolute gates. That needs a new packet. Blind external replication and
an authorized independent reviewer remain necessary for independent scientific
acceptance.
