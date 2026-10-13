# Replacement-loss V18 results: projected causal refinement

State slices: `theseus-lab-sandbox-v1`,
`theseus-replacement-loss-v18-projected-causal-refinement-v1`, and
`global-experiment-assurance-lifecycle-v1`.

## Verdict

- Scientific outcome: **`FAIL`** under the frozen all-gates-must-pass rule.
- Assurance level after replay: **`REPRODUCED_LOCAL`**.
- Reviewer status: **`NOT_SUBMITTED`**; this was not a blind run or an
  independent review.
- Behavior and mechanism diverged: **`true`**.

The projected-refinement replacement retained held-out behavior, but did not
meet the predeclared held-out causal-effect prediction gates. The second local
runtime reproduced that negative result exactly.

## Frozen identity and execution

- Repository revision: `202f459d6d89df9a038d78d58e0af825d287fcd8`; the working
  tree was already dirty at execution. The packet digest binds the executed
  protocol bytes.
- Protocol: `theseus-replacement-loss-v18-projected-causal-refinement`.
- Protocol slice: `theseus-replacement-loss-v18-projected-causal-refinement-v1`.
- Freeze SHA-256: `68fb56d0b33ee47072f0eaefb161c54f6c7c2d4d8e561363f409e3abca5eec39`.
- Packet SHA-256: `855b2c12d09edb0964845054991743ccb00c28a5d9c5583a6a0f7b2b65207739`.
- External packet archive SHA-256: `fc171bb0ce7e7c5cc2a04f8e198c1f8d7b1f6045adf2f43e6defc33ecbeb6dc8`.
- Target: reused OPT/ReLU checkpoint `grushaprasad/opt125m-rootp-surprisal-curriculum`,
  revision `8af2a2926bc772599ccb3b4e7ad7b40b35259a63`; tree SHA-256
  `6e4c3445311ab3d03830414693cc3c19c6ebfe85176a501629476cf7e2a81466`; weights
  SHA-256 `aadbd43968ed14753343d18042654e812970320b7194982b59f6d4aaa40c3132`.
- Three seeds, three held-out strata, and 180 challenge prompts total. All
  prompt-bearing fixtures, captures, candidate tensors, challenge entropy,
  predictions, and row-level outcomes remain in external owner custody.
- Runtime: Python 3.12.13, PyTorch 2.6.0, Transformers 4.57.3, NumPy 2.4.6,
  deterministic algorithms enabled, one Torch thread, local CPU. Primary and
  replica package-inventory SHA-256:
  `1a03631f1ec9139dfee2ee5a82280c27401387c01718fff2de4916b99199fda9`.
- Recorded prepare/predict/score phase time was 245.17 seconds for primary and
  247.41 seconds for replica. Both completed without a configured wall-time or
  memory stop. Peak RSS was not retained in the aggregate receipt. Monetary
  cost: `NOT_APPLICABLE`; energy: `NOT_MEASURED`.

## Held-out results

There were nine seed-by-stratum groups. Every group had 20/20 resolvable
teacher effects, so the causal failures were measured gate misses rather than
missing or unresolvable examples.

| Frozen check | Result |
| --- | ---: |
| Fit-retention precondition | 3/3 seeds passed |
| Held-out behavior gates | 9/9 groups passed |
| Absolute module-effect prediction gates | 0/9 groups passed |
| Absolute next-logit-effect prediction gates | 0/9 groups passed |
| Projected-vs-pointwise improvement gates | 0/9 groups passed |
| Activation similarity gates | 6/9 groups passed |
| Wrong-layer control margin | 4/9 groups passed |
| Random control margin | 5/9 groups passed |
| No-op control margin | 6/9 groups passed |
| Oracle restoration check | 3/3 seeds passed; maximum logit error 0.0 |

Across the nine groups, median projected module-effect cosine was `0.1473`
with median normalized error `1.0536`. Median projected next-logit-effect
cosine was `0.1764` with median normalized error `1.1178`. Projected refinement
gained a median `0.02017` next-logit cosine over pointwise fitting (required
`0.05`) and reduced median normalized error by `0.00777` (required `0.10`).
These aggregate metrics explain the frozen `FAIL`; they do not change its
thresholds or disposition.

The oracle restoration control checks instrumentation by restoring the
teacher's own activations. It is not learned activation rescue. Behavioral
retention and exact oracle restoration therefore do not repair the failed
causal-prediction result.

## Post-result training diagnostic

The frozen fit receipts show that the miss began before held-out evaluation.
The normalized fit-effect loss is `MSE / mean(target_effect^2)`, making a
zero-effect predictor's fit score `1.0`. Across seeds, the pointwise
candidates' normalized fit-effect MSE had median `1.0564` (range
`1.0515–1.0581`); projected candidates had median `1.0413` (range
`1.0370–1.0440`). Thus projected refinement improved this descriptive metric
slightly but remained worse than the zero-effect baseline on its own fit data.
By contrast, projected value-fit MSE remained within the frozen retention
ratio. The gradient-conflict projection was active on 34.4%–63.5% of updates
(median 54.1%).

This is a post-result diagnostic from aggregate receipts, not an added frozen
gate or a rescoring of the outcome. It points to causal-effect learning during
adaptation as the first issue to investigate; the held-out failure is not
explained only by a shortage of resolvable challenge effects.

A source-level inspection of the frozen target-construction path found that
the fit directions are generated from fit-token inputs, the teacher's finite
effect is recomputed from those same inputs and directions, and the direct
teacher output is checked against the captured output before training. The
reviewed path shows no evident input/effect row-alignment or normalization
defect. This bounded inspection does not prove the absence of other bugs; it
narrowly shifts the leading explanation toward the replacement's training
objective or parameterization failing to learn the local effect target.

## Separate-runtime reproduction

The primary and replica used separate Python executables with matching pinned
runtime inventories on the same host and under the same owner. The replay
matched all compared artifact groups byte-for-byte: fixtures, splits,
adaptation captures, candidates, challenge entropy, and predictions. The
scientific result digest and outcome also matched.

- Candidate-set SHA-256: `8d58f4ceb2861ecafa5607ac82cb1a9d27c72b47ca9571159cf0def01dac656a`.
- Prediction-set SHA-256: `901c7adf7a6907c1f5a42a847a9badc39ef8a06144d675e11319713f2d10aa89`.
- Scientific-results SHA-256: `2fd19fcbcb577600be5eec37eb081399ea4a48c30d33a6b310444ad6b48c9a95`.
- Replication-report SHA-256: `e4377974e4e6f7ff332af815ccd7796f35538f5514a41727de3538b807654faa`.
- Final primary outcome-report SHA-256: `94b4ff2725d8008145094e877e305919388dfea77eed6e0a7de40f5ff54cda4a`.

This is same-owner, same-host `REPRODUCED_LOCAL` evidence. It is not blind
replication, independent acceptance, or external validation.

## Claim ceiling and next gate

The maximum claim remains
`LOCAL_PROJECTED_CAUSAL_REFINEMENT_EFFECT_ON_WHOLE_MLP_REPLACEMENT_FOR_ONE_REUSED_OPT_CHECKPOINT`.
This result does not establish architecture-general validity, a learned
replacement preserving causal mechanism, a novel proof of Theseus continuity,
subjective identity or consciousness, alignment, or production readiness.

The bounded aggregate and source-path audits are complete. Before architecture
or checkpoint generalization, a new method must first beat the zero-effect
baseline on a fresh adaptation set while retaining pointwise behavior, under a
new frozen protocol with a predeclared objective and holdout. V18's challenge
fixtures remain sealed and must not become tuning data. Another parameter
sweep of V18 would not change this result. Blind external replication and an
authorized independent reviewer remain the next authority gates for
independent scientific acceptance.
