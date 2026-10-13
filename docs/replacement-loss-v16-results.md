# Replacement-loss V16 results

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v16-whole-mlp-independent-replacement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Disposition

- Scientific outcome: `FAIL` under the frozen conjunctive thresholds.
- Assurance: `REPRODUCED_LOCAL` in a second Python runtime on the same host
  and owner. This does not constitute blind replication, independent review,
  or `INDEPENDENT_ACCEPT`.
- Behavior and mechanism diverged: `true`.
- Reviewer status: `NOT_SUBMITTED`.
- Reproducibility status: `REPRODUCED_LOCAL`.

V16 tested an independently initialized and trained, full-width replacement
for the entire `fc2(ReLU(fc1(x)))` function at one visible-selected block in a
previously studied OPT/ReLU checkpoint. The replacement preserved ordinary
held-out behavior but did not predict the original model's held-out causal
effects. The failure is the result; no threshold or interpretation was changed
after scoring.

## Frozen identity and execution

- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`; the worktree
  was already dirty with 248 entries before the run. No unrelated state was
  changed, and no commit was created.
- Protocol packet SHA-256:
  `72469f59ff36c465542caab84970953b0eac69532f0d2fde5d292ee424b2a763`.
- External packet archive SHA-256:
  `ab9da55b9eeb58ad0dcc28050f0fe502a3c9651daa18115096a69a1a4f132f80`.
- Checkpoint: `grushaprasad/opt125m-rootp-surprisal-curriculum`, revision
  `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`; tree digest
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`;
  weight digest
  `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
  This is a reused, outcome-aware checkpoint, not a fresh held-out target.
- Visible-only selection chose block 8; block 9 was the wrong-module control.
  Three adaptation seeds produced nine frozen candidate artifacts. Challenge
  predictions were sealed before scoring.
- Runtime: Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6,
  one Torch thread, deterministic algorithms, local CPU. Both runtimes matched
  the complete locked package inventory.
- Primary phase times were 80.9 seconds for preparation, 31.0 seconds for
  prediction, and 24.4 seconds for scoring. Peak RSS was 2.09 GB. The replay
  also stayed below the four-hour and 16 GiB limits.
- Monetary cost: `NOT_APPLICABLE`; energy: `NOT_MEASURED`.

## Results

All nine seed-by-stratum cases had 20/20 resolvable module and next-logit
effects. The separate-runtime replay matched all six artifact groups byte-for-byte:
fixtures, splits, captures, candidates, challenge entropy, and predictions.

| Measure | Observed range | Frozen gate | Result |
|---|---:|---:|---|
| Module-effect direction accuracy | 0.95–1.00 | ≥0.80 | Passed |
| Module-effect median cosine | 0.057–0.201 | ≥0.90 | Failed in all 9 cases |
| Module-effect median relative error | 1.02–1.28 | ≤0.20 | Failed in all 9 cases |
| Next-logit-effect direction accuracy | 0.15–0.80 | ≥0.80 | Failed in most cases |
| Next-logit-effect median cosine | −0.768–0.826 | ≥0.90 | Failed in all 9 cases |
| Next-logit-effect median relative error | 0.817–1.425 | ≤0.20 | Failed in all 9 cases |
| Replaced-boundary activation cosine | 0.972–0.994 | ≥0.90 | Passed in all 9 cases |
| Replaced-boundary activation relative error | 0.118–0.235 | ≤0.20 | Failed in all 3 compositional cases |
| Continuation cross-entropy change | −0.0721–+0.0206 nat/token | ≤+0.05 | Passed in all cases |
| Normalized next-logit behavior error | 0.0057–0.0085 | ≤0.05 | Passed in all cases |
| Top-1 agreement | 0.95–1.00 | Diagnostic | High |
| Oracle-restoration max logit error | 0.0 | ≤1e−5 | Passed |

The per-seed treatment-minus-control median-cosine margins were:

| Seed | Random | Wrong layer | No-op |
|---:|---:|---:|---:|
| 0 | +0.394 | −0.276 | +0.005 |
| 1 | +0.019 | +0.358 | −0.037 |
| 2 | +1.089 | −0.036 | +0.514 |

The frozen control requirement was a margin of at least `+0.05` over every
control in every seed. No seed met all three margins.

Candidate normalized validation MSE for the treatment ranged from 0.0197 to
0.0236. This confirms the fitted candidate reproduced much of the adaptation
output function, but it did not establish local causal-response equivalence.
The oracle error of zero validates the restoration path only; it is not a
learned rescue.

## Interpretation and claim ceiling

On this checkpoint and selected block, high activation similarity and strong
unperturbed behavior preservation did not imply preservation of the model's
local causal response. The replacement failed the held-out causal prediction
gates by large margins. This is evidence against the tested pointwise
output-matching recipe as a sufficient mechanism-preservation method. It does
not establish that independently learned replacement is impossible.

Maximum claim ceiling:
`LOCAL_WHOLE_MLP_REPLACEMENT_CAUSAL_PREDICTION_OBSERVATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
The result does not support architecture-general validity, consciousness,
alignment, production readiness, independent scientific acceptance, or a
novel proof of Theseus continuity.

## Evidence digests and custody

- Candidate set SHA-256:
  `877c295462e229b6568d1ba373d2f88bcbdc7dcccade7a661b5ceb3627e0d9e3`.
- Prediction set SHA-256:
  `809033f67be70d289b17b13a98f5ea98d167bc80d70f0253662599214f92bb11`.
- Scientific result SHA-256:
  `57e434f41bf1f8c5ebbdf395423fc3d526955d8f1882a92c6a378e484c6a253d`.
- Replication report SHA-256:
  `187345e72ec2b7aeba11907bb650f28441d1971e68c02798550f685df7aa2a6a`.
- Final primary outcome report SHA-256:
  `d7208193fec30ebf2d3ed9de02ebca7a92dcef4ccac5c13a002ecdf4d19b9e30`.

Prompts, token IDs, captures, candidate tensors, challenge entropy,
predictions, and row-level outcomes remain in external owner custody. This
repository contains only protocol code and aggregate-safe results.

## Next gate

Keep V16 frozen as `FAIL`. A scientifically useful follow-up would train a new
replacement with adaptation-only causal supervision (for example, local
Jacobian-vector or finite-intervention targets), then freeze it before new
held-out directions are generated. The primary endpoint should remain
relative causal-effect prediction; ordinary behavior and activation similarity
stay separate outcomes. That requires a new packet and must not reuse V16's
held-out entropy or outcomes for fitting. Blind external replication and an
authorized independent reviewer remain necessary for independent scientific
acceptance.
