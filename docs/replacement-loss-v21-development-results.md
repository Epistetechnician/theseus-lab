# Replacement-loss V21 development results

State slices: `theseus-lab-sandbox-v1`, `theseus-replacement-loss-v21-paired-endpoint-supervision-v1`, and `global-experiment-assurance-lifecycle-v1`.

- Outcome: `NO_QUALIFIED_DEVELOPMENT_CANDIDATE`.
- Assurance: `REPRODUCED_LOCAL`; reviewer: `NOT_SUBMITTED`; blind status: `NOT_BLIND_SAME_OWNER_REPLAY`.
- Packet SHA-256: `8b95876077c15779b3c215872316e68a4fc52094ed23d0492672fbd3d4d6b7bd`; result SHA-256: `a8af01248ac55b71e9cc00fdf813fa08cf333b02b15ee3019180fc78d3865f42`.
- Claim ceiling: `LOCAL_PAIRED_ENDPOINT_SUPERVISION_QUALIFICATION_ON_ONE_REUSED_OPT_CHECKPOINT`.
- Custody: all prompt-bearing and row-level artifacts remain in external owner custody.

## Nine predeclared development cells

Each cell must pass the effect ratio <=0.90, pointwise value ratio <=1.25, continuation CE increase <=0.05, and normalized next-logit error <=0.05 gates.

| Seed index | Stratum | Effect ratio | Value ratio | CE increase | Next-logit error | Cell passes |
|---:|---|---:|---:|---:|---:|:---:|
| 0 | ordinary | 1.0538 | 0.7668 | -0.0372 | 0.0070 | False |
| 0 | compositional | 1.0661 | 0.7856 | 0.0179 | 0.0159 | False |
| 0 | adversarial | 1.0029 | 0.7414 | -0.0310 | 0.0042 | False |
| 1 | ordinary | 1.0503 | 0.7223 | -0.0689 | 0.0139 | False |
| 1 | compositional | 1.0659 | 0.7598 | 0.0265 | 0.0107 | False |
| 1 | adversarial | 1.0044 | 0.7184 | -0.0188 | 0.0064 | False |
| 2 | ordinary | 1.0567 | 0.7849 | -0.0551 | 0.0094 | False |
| 2 | compositional | 1.0679 | 0.8125 | 0.0187 | 0.0092 | False |
| 2 | adversarial | 0.9834 | 0.7935 | -0.0197 | 0.0048 | False |

Equal-cell mean effect ratio: `1.039045`; worst cell: `1.067865`.

Next gate: new method packet required; do not create confirmation fixtures.

Behavior and mechanism diverged: `true`.
